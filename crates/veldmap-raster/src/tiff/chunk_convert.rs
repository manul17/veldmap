//! Pure pixel-format conversion routines for decoded raw TIFF chunks across
//! Display (`Rgba8UnormSrgb`), Scientific (`R32Float`), and Categorical
//! (`R8Uint`/`R16Uint`) streams, plus shared single-band sample iteration.

use crate::RasterError;
use crate::pipeline::{
    DisplayMapping, PixelLayout, SampleType, is_data_f64, normalize_alpha_u8, raw_bytes_to_rgba,
    read_native_sample_f64,
};

use super::ifd::{CachedIfd, PlanarConfig};

/// Narrow 2-variant categorical output format eliminating unreachable wildcard
/// arms in categorical chunk conversion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CategoricalFormat {
    U8,
    U16,
}

pub(super) enum PaletteAlpha<'a> {
    None,
    Chunky(usize),
    Planar(&'a [u8]),
}

/// Iterate single-band native numeric samples (or YCbCr-converted RGB band
/// samples) across `pixels` with stride `step`, invoking `visit` with `Some(v)`
/// for valid data samples or `None` for nodata/out-of-range samples.
#[inline]
pub(super) fn for_each_band_sample_f64(
    data: &[u8],
    ifd: &CachedIfd,
    band: usize,
    pixels: usize,
    step: usize,
    nodata_f64: Option<f64>,
    mut visit: impl FnMut(Option<f64>),
) {
    let step = step.max(1);
    if let Some(params) = ifd.ycbcr_band_params(band) {
        for at in (0..pixels).step_by(step) {
            let off = at * 3;
            let sample = if off + 3 <= data.len() {
                let rgb = params.convert_u8_pixel_to_rgb(data[off], data[off + 1], data[off + 2]);
                let raw = f64::from(rgb[band]);
                is_data_f64(raw, nodata_f64).then_some(raw)
            } else {
                None
            };
            visit(sample);
        }
        return;
    }

    let (stride, ch_offset) = ifd.band_stride_offset(band);
    for at in (0..pixels).step_by(step) {
        let sample = read_native_sample_f64(data, ifd.sample_type, at * stride + ch_offset)
            .filter(|&v| is_data_f64(v, nodata_f64));
        visit(sample);
    }
}

pub(super) fn expand_palette_to_rgba(
    data: &[u8],
    alpha_mode: PaletteAlpha<'_>,
    ifd: &CachedIfd,
    color_map: &[[u8; 4]],
    nodata_f64: Option<f64>,
    pixels: usize,
) -> Vec<u8> {
    let mut rgba = vec![0u8; pixels * 4];
    let chunky_spp = ifd.samples_in_chunk();
    for (px, dst) in rgba.chunks_exact_mut(4).enumerate() {
        let Some(raw_val) = read_native_sample_f64(data, ifd.sample_type, px * chunky_spp) else {
            continue;
        };
        if !is_data_f64(raw_val, nodata_f64) {
            continue;
        }
        let idx = raw_val.round().clamp(0.0, (color_map.len() - 1) as f64) as usize;
        let mut color = color_map[idx];
        let alpha_val = match alpha_mode {
            PaletteAlpha::None => None,
            PaletteAlpha::Planar(ap) => read_native_sample_f64(ap, ifd.sample_type, px),
            PaletteAlpha::Chunky(a_ch) => {
                read_native_sample_f64(data, ifd.sample_type, px * chunky_spp + a_ch)
            }
        };
        if let Some(a) = alpha_val {
            color[3] = if a.is_finite() {
                normalize_alpha_u8(a, ifd.sample_type)
            } else {
                0
            };
        }
        dst.copy_from_slice(&color);
    }
    rgba
}

/// Scatter one decoded planar band buffer into a packed interleaved buffer so
/// callers can decode and drop one plane at a time without holding all `K`
/// planes in RAM simultaneously.
pub(super) fn scatter_plane_into_interleaved(
    interleaved: &mut [u8],
    plane_data: &[u8],
    slot: usize,
    interleaved_spp: usize,
    sample_bytes: usize,
    pixels: usize,
) {
    for px in 0..pixels {
        let s = px * sample_bytes;
        let d = (px * interleaved_spp + slot) * sample_bytes;
        if s + sample_bytes <= plane_data.len() && d + sample_bytes <= interleaved.len() {
            interleaved[d..d + sample_bytes].copy_from_slice(&plane_data[s..s + sample_bytes]);
        }
    }
}

/// Convert an interleaved subset of planar display channels into RGBA8.
pub(super) fn convert_interleaved_planar_display(
    interleaved: &[u8],
    ifd: &CachedIfd,
    pixels: usize,
    mapping: DisplayMapping,
) -> Vec<u8> {
    let packed_layout = match (ifd.pixel_layout.colors(), ifd.alpha_channel().is_some()) {
        (1, false) => PixelLayout::named(1),
        (1, true) => PixelLayout::named(2),
        (_, false) => PixelLayout::named(3),
        (_, true) => PixelLayout::named(4),
    };
    raw_bytes_to_rgba(
        &mapping,
        interleaved,
        ifd.sample_type,
        packed_layout,
        pixels,
    )
}

pub(super) fn convert_chunky_display(
    data: &[u8],
    ifd: &CachedIfd,
    pixels: usize,
    mapping: DisplayMapping,
) -> Vec<u8> {
    if let Some(params) = ifd.ycbcr
        && ifd.sample_type == SampleType::U8
    {
        return params.convert_u8_to_rgba(data, pixels, mapping.nodata());
    }
    raw_bytes_to_rgba(&mapping, data, ifd.sample_type, ifd.pixel_layout, pixels)
}

pub(super) fn convert_scientific_chunk(
    data: &[u8],
    ifd: &CachedIfd,
    band: usize,
    pixels: usize,
    nodata_f64: Option<f64>,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(pixels * 4);
    for_each_band_sample_f64(data, ifd, band, pixels, 1, nodata_f64, |sample| {
        let out_val = match sample {
            Some(v) => {
                let f = v as f32;
                if f.is_finite() { f } else { f32::NAN }
            }
            None => f32::NAN,
        };
        out.extend_from_slice(&out_val.to_ne_bytes());
    });
    out
}

pub(super) fn sparse_categorical_chunk(
    format: CategoricalFormat,
    pixels: usize,
    nodata_f64: Option<f64>,
) -> Vec<u8> {
    let fill_u16 = nodata_f64
        .filter(|v| v.is_finite() && *v >= 0.0 && *v <= f64::from(u16::MAX))
        .map_or(0u16, |v| v.round() as u16);
    match format {
        CategoricalFormat::U8 => vec![fill_u16.min(255) as u8; pixels],
        CategoricalFormat::U16 => {
            let b = fill_u16.to_ne_bytes();
            let mut out = Vec::with_capacity(pixels * 2);
            for _ in 0..pixels {
                out.extend_from_slice(&b);
            }
            out
        }
    }
}

pub(super) fn convert_categorical_chunk(
    mut data: Vec<u8>,
    ifd: &CachedIfd,
    band: usize,
    pixels: usize,
    format: CategoricalFormat,
) -> Result<Vec<u8>, RasterError> {
    if let Some(params) = ifd.ycbcr_band_params(band) {
        return Ok(match format {
            CategoricalFormat::U8 => data
                .chunks_exact(3)
                .take(pixels)
                .map(|px| params.convert_u8_pixel_to_rgb(px[0], px[1], px[2])[band])
                .collect(),
            CategoricalFormat::U16 => {
                let mut out = Vec::with_capacity(pixels * 2);
                for px in data.chunks_exact(3).take(pixels) {
                    let v = u16::from(params.convert_u8_pixel_to_rgb(px[0], px[1], px[2])[band]);
                    out.extend_from_slice(&v.to_ne_bytes());
                }
                out
            }
        });
    }

    // Preserve direct zero-copy/byte-slice fast paths for native U8/U16 bands.
    let (stride, ch_offset) = ifd.band_stride_offset(band);
    match (format, ifd.sample_type) {
        (CategoricalFormat::U8, SampleType::U8) if stride == 1 => {
            data.truncate(pixels);
            Ok(data)
        }
        (CategoricalFormat::U8, SampleType::U8) => {
            let mut out = Vec::with_capacity(pixels);
            for px in 0..pixels {
                out.push(data[px * stride + ch_offset]);
            }
            Ok(out)
        }
        (CategoricalFormat::U16, SampleType::U16) if stride == 1 => {
            data.truncate(pixels * 2);
            Ok(data)
        }
        (CategoricalFormat::U16, SampleType::U16) => {
            let mut out = Vec::with_capacity(pixels * 2);
            for px in 0..pixels {
                let off = (px * stride + ch_offset) * 2;
                out.extend_from_slice(&data[off..off + 2]);
            }
            Ok(out)
        }
        (CategoricalFormat::U16, SampleType::U8) => {
            let mut out = Vec::with_capacity(pixels * 2);
            for px in 0..pixels {
                let val = u16::from(data[px * stride + ch_offset]);
                out.extend_from_slice(&val.to_ne_bytes());
            }
            Ok(out)
        }
        (fmt, st) => Err(RasterError::Decode(format!(
            "categorical format {fmt:?} is incompatible with source sample type {st:?}"
        ))),
    }
}

pub(super) fn palette_alpha_mode<'a>(
    ifd: &CachedIfd,
    alpha_plane: Option<&'a [u8]>,
) -> PaletteAlpha<'a> {
    match (ifd.planar, ifd.alpha_channel(), alpha_plane) {
        (PlanarConfig::Planar, Some(_), Some(ap)) => PaletteAlpha::Planar(ap),
        (PlanarConfig::Chunky, Some(a_ch), _) => PaletteAlpha::Chunky(a_ch),
        _ => PaletteAlpha::None,
    }
}
