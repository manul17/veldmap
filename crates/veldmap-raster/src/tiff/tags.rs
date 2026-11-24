//! TIFF tag validation, sample decoding, and error mapping helpers.

use tiff::decoder::{Decoder, DecodingResult};
use tiff::tags::Tag;
use veldmap_core::budget::MemoryBudget;
use veldmap_core::io::BufferedByteReader;
use veldmap_core::sync::CancelToken;

use crate::RasterError;
use crate::adapters::map_std_io_error;
use crate::pipeline::grid;
use crate::pipeline::metadata::SampleType;
use crate::pipeline::peak::DecodePeak;
use crate::pipeline::radiometry::{DisplayMapping, PixelLayout, Samples, is_data};

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

const TAG_YCBCR_COEFFICIENTS: u16 = 529;
const TAG_REFERENCE_BLACK_WHITE: u16 = 532;

/// TIFF 6.0 Section 21 YCbCr -> RGB conversion parameters (`YCbCrCoefficients`
/// tag 529 defaulting to BT.601 `[0.299, 0.587, 0.114]`, and
/// `ReferenceBlackWhite` tag 532 defaulting to full-range `[0, 255, 128, 255, 128, 255]`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct YCbCrParams {
    pub(super) luma: [f64; 3],
    pub(super) ref_black_white: [f64; 6],
}

impl Default for YCbCrParams {
    fn default() -> Self {
        Self {
            luma: [0.299, 0.587, 0.114],
            ref_black_white: [0.0, 255.0, 128.0, 255.0, 128.0, 255.0],
        }
    }
}

impl YCbCrParams {
    /// Convert upsampled 8-bit `YCbCr` triplets directly to tightly packed
    /// `RGBA8` without allocating an intermediate `RGB` buffer.
    pub(super) fn convert_u8_to_rgba(
        &self,
        ycbcr: &[u8],
        pixels: usize,
        nodata: Option<f32>,
    ) -> Vec<u8> {
        let [lr, lg, lb] = self.luma;
        let [rb_y, rw_y, rb_cb, rw_cb, rb_cr, rw_cr] = self.ref_black_white;
        let scale_y = if (rw_y - rb_y).abs() > 1e-6 {
            255.0 / (rw_y - rb_y)
        } else {
            1.0
        };
        let scale_cb = if (rw_cb - rb_cb).abs() > 1e-6 {
            127.0 / (rw_cb - rb_cb)
        } else {
            1.0
        };
        let scale_cr = if (rw_cr - rb_cr).abs() > 1e-6 {
            127.0 / (rw_cr - rb_cr)
        } else {
            1.0
        };
        let kr = 2.0 - 2.0 * lr;
        let kb = 2.0 - 2.0 * lb;
        let inv_lg = if lg.abs() > 1e-6 {
            1.0 / lg
        } else {
            1.0 / 0.587
        };

        let count = pixels.min(ycbcr.len() / 3);
        let mut rgba = Vec::with_capacity(count * 4);
        for px in ycbcr.chunks_exact(3).take(count) {
            let y = (f64::from(px[0]) - rb_y) * scale_y;
            let cb = (f64::from(px[1]) - rb_cb) * scale_cb;
            let cr = (f64::from(px[2]) - rb_cr) * scale_cr;
            let r = (cr * kr + y).clamp(0.0, 255.0).round() as u8;
            let b = (cb * kb + y).clamp(0.0, 255.0).round() as u8;
            let g = ((y - lb * f64::from(b) - lr * f64::from(r)) * inv_lg)
                .clamp(0.0, 255.0)
                .round() as u8;
            let void = nodata.is_some()
                && !is_data(f32::from(r), nodata)
                && !is_data(f32::from(g), nodata)
                && !is_data(f32::from(b), nodata);
            if void {
                rgba.extend_from_slice(&[0, 0, 0, 0]);
            } else {
                rgba.extend_from_slice(&[r, g, b, 255]);
            }
        }
        rgba
    }
}

pub(super) fn map_tiff_error(err: tiff::TiffError, cancel: &CancelToken) -> RasterError {
    if cancel.is_cancelled() {
        return RasterError::Cancelled;
    }
    match err {
        tiff::TiffError::IoError(io_err) => map_std_io_error(io_err, cancel),
        tiff::TiffError::LimitsExceeded => RasterError::DecoderLimitsExceeded(err.to_string()),
        other => RasterError::Decode(other.to_string()),
    }
}

fn is_missing_tag(err: &tiff::TiffError) -> bool {
    matches!(
        err,
        tiff::TiffError::FormatError(tiff::TiffFormatError::RequiredTagNotFound(_))
    )
}

/// Convert an optional TIFF tag read into `Option<T>`, treating only a missing
/// tag (`RequiredTagNotFound`) as `Ok(None)` and propagating all other format,
/// I/O, or cancellation errors.
fn opt_tag_strict<T>(
    res: Result<T, tiff::TiffError>,
    cancel: &CancelToken,
) -> Result<Option<T>, RasterError> {
    match res {
        Ok(v) => Ok(Some(v)),
        Err(err) if is_missing_tag(&err) => Ok(None),
        Err(err) => Err(map_tiff_error(err, cancel)),
    }
}

/// Convert an optional TIFF tag read into `Option<T>`, propagating cancellation
/// and I/O errors while treating missing or non-matching tags as `Ok(None)`.
pub(super) fn opt_tag<T>(
    res: Result<T, tiff::TiffError>,
    cancel: &CancelToken,
) -> Result<Option<T>, RasterError> {
    match res {
        Ok(v) => Ok(Some(v)),
        Err(err) if is_missing_tag(&err) => Ok(None),
        Err(_) if cancel.is_cancelled() => Err(RasterError::Cancelled),
        Err(err @ tiff::TiffError::IoError(_)) => Err(map_tiff_error(err, cancel)),
        Err(_) => Ok(None),
    }
}

pub(super) fn is_overview(subfile_type: u32) -> bool {
    const REDUCED: u32 = 1;
    const MASK: u32 = 4;
    subfile_type & REDUCED != 0 && subfile_type & MASK == 0
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

fn read_ycbcr_params(
    decoder: &mut Decoder<BufferedByteReader>,
    cancel: &CancelToken,
) -> Result<YCbCrParams, RasterError> {
    let mut params = YCbCrParams::default();
    if let Some(coeffs) =
        read_tag_numeric_f64_vec_opt(decoder, Tag::Unknown(TAG_YCBCR_COEFFICIENTS), cancel)?
        && coeffs.len() == 3
        && coeffs.iter().all(|v| v.is_finite() && *v > 0.0 && *v < 1.0)
        && coeffs[1] > 1e-4
    {
        params.luma = [coeffs[0], coeffs[1], coeffs[2]];
    }
    if let Some(rbw) =
        read_tag_numeric_f64_vec_opt(decoder, Tag::Unknown(TAG_REFERENCE_BLACK_WHITE), cancel)?
        && rbw.len() == 6
        && rbw.iter().all(|v| v.is_finite())
        && (rbw[1] - rbw[0]).abs() > 1e-6
        && (rbw[3] - rbw[2]).abs() > 1e-6
        && (rbw[5] - rbw[4]).abs() > 1e-6
    {
        params.ref_black_white = [rbw[0], rbw[1], rbw[2], rbw[3], rbw[4], rbw[5]];
    }
    Ok(params)
}

fn value_to_f64(val: &tiff::decoder::ifd::Value) -> Option<f64> {
    use tiff::decoder::ifd::Value;
    match *val {
        Value::Byte(v) => Some(f64::from(v)),
        Value::Short(v) => Some(f64::from(v)),
        Value::Unsigned(v) => Some(f64::from(v)),
        Value::UnsignedBig(v) => Some(v as f64),
        Value::SignedByte(v) => Some(f64::from(v)),
        Value::SignedShort(v) => Some(f64::from(v)),
        Value::Signed(v) => Some(f64::from(v)),
        Value::SignedBig(v) => Some(v as f64),
        Value::Float(v) => Some(f64::from(v)),
        Value::Double(v) => Some(v),
        Value::Rational(n, d) if d != 0 => Some(f64::from(n) / f64::from(d)),
        Value::SRational(n, d) if d != 0 => Some(f64::from(n) / f64::from(d)),
        _ => None,
    }
}

fn read_tag_numeric_f64_vec_opt(
    decoder: &mut Decoder<BufferedByteReader>,
    tag: Tag,
    cancel: &CancelToken,
) -> Result<Option<Vec<f64>>, RasterError> {
    let Some(val) = opt_tag(decoder.get_tag(tag), cancel)? else {
        return Ok(None);
    };
    let out = match val {
        tiff::decoder::ifd::Value::List(items) => items.iter().filter_map(value_to_f64).collect(),
        single => value_to_f64(&single).into_iter().collect(),
    };
    Ok(Some(out))
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

pub(super) const fn depth_of_layout(layout: PixelLayout, bits: u8) -> u32 {
    layout.channels as u32 * bits.div_ceil(8) as u32
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

pub(super) fn read_tag_u16_vec_opt(
    decoder: &mut Decoder<BufferedByteReader>,
    tag: Tag,
    cancel: &CancelToken,
) -> Result<Vec<u16>, RasterError> {
    Ok(opt_tag(decoder.get_tag_u16_vec(tag), cancel)?.unwrap_or_default())
}

pub(super) fn read_tag_f64_vec_opt(
    decoder: &mut Decoder<BufferedByteReader>,
    tag: Tag,
    cancel: &CancelToken,
) -> Result<Vec<f64>, RasterError> {
    Ok(opt_tag(decoder.get_tag_f64_vec(tag), cancel)?.unwrap_or_default())
}

pub(super) fn read_tag_ascii_opt(
    decoder: &mut Decoder<BufferedByteReader>,
    tag: Tag,
    cancel: &CancelToken,
) -> Result<String, RasterError> {
    Ok(opt_tag(decoder.get_tag_ascii_string(tag), cancel)?.unwrap_or_default())
}

pub(super) fn parse_gdal_nodata(
    decoder: &mut Decoder<BufferedByteReader>,
    cancel: &CancelToken,
) -> Result<Option<f64>, RasterError> {
    let Some(raw) = opt_tag(decoder.get_tag_ascii_string(Tag::GdalNodata), cancel)? else {
        return Ok(None);
    };
    let trimmed = raw.trim_matches(|c: char| c == '\0' || c.is_whitespace());
    if trimmed.eq_ignore_ascii_case("nan") {
        return Ok(Some(f64::NAN));
    }
    Ok(trimmed.parse::<f64>().ok().filter(|v| v.is_finite()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overview_bit_and_sample_type_resolution_reject_invalid_formats() {
        assert!(!is_overview(0));
        assert!(is_overview(1));
        assert!(!is_overview(5)); // mask bit set
        assert_eq!(resolve_sample_type(1, 8).unwrap(), SampleType::U8);
        assert_eq!(resolve_sample_type(1, 16).unwrap(), SampleType::U16);
        assert_eq!(resolve_sample_type(2, 16).unwrap(), SampleType::I16);
        assert_eq!(resolve_sample_type(3, 32).unwrap(), SampleType::F32);
        assert!(resolve_sample_type(1, 12).is_err());
        assert!(resolve_sample_type(1, 32).is_err());
    }
}
