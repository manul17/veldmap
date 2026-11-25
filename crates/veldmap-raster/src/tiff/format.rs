//! TIFF sample-format validation, chunk grid admission, and decoded buffer
//! conversion (`TiffFormat`, `ensure_readable`, `chunk_grid`, `typed_samples`,
//! `chunk_rgba`).

use tiff::decoder::{Decoder, DecodingResult};
use tiff::tags::Tag;
use veldmap_core::budget::MemoryBudget;
use veldmap_core::io::BufferedByteReader;
use veldmap_core::sync::CancelToken;

use crate::RasterError;
use crate::pipeline::grid;
use crate::pipeline::metadata::SampleType;
use crate::pipeline::peak::DecodePeak;
use crate::pipeline::radiometry::{DisplayMapping, PixelLayout, Samples};

use super::tags::{map_tiff_error, opt_tag_strict, read_tag_u16_vec_opt};
use super::ycbcr::{YCbCrParams, read_ycbcr_params};

const DECODED_COMPRESSIONS: [u16; 8] = [
    1,      // None
    5,      // LZW
    7,      // JPEG
    8,      // Deflate
    0x8005, // PackBits
    0x80B2, // Deflate (old code)
    0xC350, // ZSTD
    0xC351, // WebP (50001)
];

/// Compact sample-layout descriptor shared across TIFF overview scanning,
/// chunk reading, and lazy display-mapping stretch computation.
#[derive(Clone, Copy, Debug)]
pub(super) struct TiffFormat {
    pub(super) pixel_layout: PixelLayout,
    pub(super) sample_bits: u8,
    pub(super) sample_type: SampleType,
    pub(super) depth: u32,
    pub(super) nodata: Option<f32>,
}

pub(super) const fn depth_of_layout(layout: PixelLayout, bits: u8) -> u32 {
    layout.channels as u32 * bits.div_ceil(8) as u32
}

fn ensure_chunky(
    decoder: &mut Decoder<BufferedByteReader>,
    cancel: &CancelToken,
) -> Result<(), RasterError> {
    let planar = opt_tag_strict(
        decoder.get_tag_unsigned::<u16>(Tag::PlanarConfiguration),
        cancel,
    )?
    .unwrap_or(1);
    if planar != 1 {
        return Err(RasterError::Decode(
            "planar sample layout (PlanarConfiguration != 1) is not supported yet".to_string(),
        ));
    }
    Ok(())
}

fn model(
    decoder: &mut Decoder<BufferedByteReader>,
    color: tiff::ColorType,
    cancel: &CancelToken,
) -> Result<(PixelLayout, u8, Option<YCbCrParams>), RasterError> {
    match color {
        tiff::ColorType::Gray(bits) => Ok((PixelLayout::named(1), bits, None)),
        tiff::ColorType::RGB(bits) => Ok((PixelLayout::named(3), bits, None)),
        tiff::ColorType::RGBA(bits) => Ok((PixelLayout::named(4), bits, None)),
        tiff::ColorType::YCbCr(8) => {
            let params = read_ycbcr_params(decoder, cancel)?;
            Ok((PixelLayout::named(3), 8, Some(params)))
        }
        tiff::ColorType::Multiband {
            bit_depth,
            num_samples,
        } => {
            let photometric = opt_tag_strict(
                decoder.get_tag_unsigned::<u16>(Tag::PhotometricInterpretation),
                cancel,
            )?;
            if photometric == Some(0) && num_samples > 1 {
                return Err(RasterError::Decode(
                    "WhiteIsZero with multiple samples is not supported".to_string(),
                ));
            }
            let layout = if num_samples == 2 && photometric == Some(1) {
                let extra = read_tag_u16_vec_opt(decoder, Tag::ExtraSamples, cancel)?;
                if extra.iter().any(|&v| v == 1 || v == 2) {
                    PixelLayout::named(2)
                } else {
                    PixelLayout::stack(usize::from(num_samples))
                }
            } else {
                PixelLayout::stack(usize::from(num_samples))
            };
            Ok((layout, bit_depth, None))
        }
        other => Err(RasterError::Decode(format!(
            "color model {other:?} is not supported"
        ))),
    }
}

fn resolve_sample_type(sample_format: u16, bits: u8) -> Result<SampleType, RasterError> {
    match (sample_format, bits) {
        (1, 8) => Ok(SampleType::U8),
        (1, 16) => Ok(SampleType::U16),
        (2, 16) => Ok(SampleType::I16),
        (3, 32) => Ok(SampleType::F32),
        _ => Err(RasterError::Decode(format!(
            "unsupported TIFF sample format ({sample_format}, {bits}-bit)"
        ))),
    }
}

fn ensure_decodable(
    decoder: &mut Decoder<BufferedByteReader>,
    cancel: &CancelToken,
) -> Result<u16, RasterError> {
    let code =
        opt_tag_strict(decoder.get_tag_unsigned::<u16>(Tag::Compression), cancel)?.unwrap_or(1);
    if DECODED_COMPRESSIONS.contains(&code) {
        return Ok(code);
    }
    Err(RasterError::Decode(format!(
        "unsupported TIFF compression code {code}"
    )))
}

fn read_sample_type(
    decoder: &mut Decoder<BufferedByteReader>,
    bits: u8,
    cancel: &CancelToken,
) -> Result<SampleType, RasterError> {
    let format = match opt_tag_strict(decoder.get_tag_u16_vec(Tag::SampleFormat), cancel) {
        Ok(Some(formats)) => formats.first().copied().unwrap_or(1),
        Ok(None) => 1,
        Err(err) if err.is_cancelled() => return Err(err),
        Err(err) => {
            match opt_tag_strict(decoder.get_tag_unsigned::<u16>(Tag::SampleFormat), cancel) {
                Ok(Some(v)) => v,
                Ok(None) => 1,
                Err(inner) if inner.is_cancelled() => return Err(inner),
                Err(_) => return Err(err),
            }
        }
    };
    resolve_sample_type(format, bits)
}

pub(super) fn chunk_grid(
    decoder: &mut Decoder<BufferedByteReader>,
    budget: &MemoryBudget,
    depth: u32,
    cancel: &CancelToken,
) -> Result<(u32, u32), RasterError> {
    ensure_chunky(decoder, cancel)?;
    let (cw, ch) = decoder.chunk_dimensions();
    if cw == 0 || ch == 0 {
        return Err(RasterError::InvalidDimensions {
            width: cw,
            height: ch,
        });
    }
    DecodePeak::new()
        .with(
            "fresh chunk",
            grid::chunk_working_bytes(cw, ch, depth, true),
        )
        .admit(budget.decode_working_bytes)?;
    Ok((cw, ch))
}

pub(super) fn ensure_readable(
    decoder: &mut Decoder<BufferedByteReader>,
    cancel: &CancelToken,
) -> Result<(PixelLayout, u8, SampleType, Option<YCbCrParams>), RasterError> {
    let color = decoder.colortype().map_err(|e| map_tiff_error(e, cancel))?;
    let (layout, bits, ycbcr) = model(decoder, color, cancel)?;
    let compression = ensure_decodable(decoder, cancel)?;
    // Non-JPEG TIFFs with YCbCr chroma subsampling != (1, 1) cannot be decoded
    // by `tiff 0.11.3`; reject upfront at open() instead of failing mid-tile.
    if ycbcr.is_some() && compression != 7 {
        let sub = read_tag_u16_vec_opt(decoder, Tag::ChromaSubsampling, cancel)?;
        if !sub.is_empty() && sub.as_slice() != [1, 1] {
            return Err(RasterError::Decode(format!(
                "non-JPEG subsampled YCbCr TIFF (YCbCrSubSampling={sub:?}) is not supported"
            )));
        }
    }
    let stype = read_sample_type(decoder, bits, cancel)?;
    Ok((layout, bits, stype, ycbcr))
}

pub(super) fn typed_samples(data: &DecodingResult) -> Result<Samples<'_>, RasterError> {
    match data {
        DecodingResult::U8(v) => Ok(Samples::U8(v)),
        DecodingResult::U16(v) => Ok(Samples::U16(v)),
        DecodingResult::I16(v) => Ok(Samples::I16(v)),
        DecodingResult::F32(v) => Ok(Samples::F32(v)),
        other => Err(RasterError::Decode(format!(
            "unsupported TIFF decoded buffer variant ({other:?})"
        ))),
    }
}

pub(super) fn chunk_rgba(
    mapping: &DisplayMapping,
    data: &DecodingResult,
    pixel: PixelLayout,
    ycbcr: Option<YCbCrParams>,
    dw: u32,
    dh: u32,
) -> Result<Vec<u8>, RasterError> {
    let pixels = (dw as usize) * (dh as usize);
    let rgba = if let (Some(params), DecodingResult::U8(raw)) = (ycbcr, data) {
        params.convert_u8_to_rgba(raw, pixels, mapping.nodata())
    } else {
        mapping.rgba(&typed_samples(data)?, pixel, pixels)
    };
    if rgba.len() == pixels * 4 {
        Ok(rgba)
    } else {
        Err(RasterError::Decode(format!(
            "chunk {dw}x{dh} decoded {} pixels instead of {pixels}",
            rgba.len() / 4
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_type_resolution_accepts_supported_and_rejects_invalid_formats() {
        assert_eq!(resolve_sample_type(1, 8).unwrap(), SampleType::U8);
        assert_eq!(resolve_sample_type(1, 16).unwrap(), SampleType::U16);
        assert_eq!(resolve_sample_type(2, 16).unwrap(), SampleType::I16);
        assert_eq!(resolve_sample_type(3, 32).unwrap(), SampleType::F32);
        assert!(resolve_sample_type(1, 12).is_err());
        assert!(resolve_sample_type(1, 32).is_err());
    }
}
