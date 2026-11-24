//! TIFF tag validation, sample decoding, and error mapping helpers.

use tiff::decoder::{Decoder, DecodingResult};
use tiff::tags::Tag;
use veldmap_core::budget::MemoryBudget;
use veldmap_core::io::{BufferedByteReader, IoError};
use veldmap_core::sync::CancelToken;

use crate::RasterError;
use crate::pipeline::grid;
use crate::pipeline::metadata::SampleType;
use crate::pipeline::peak::DecodePeak;
use crate::pipeline::radiometry::{DisplayMapping, PixelLayout, Samples};

const DECODED_COMPRESSIONS: [u16; 7] = [
    1,      // None
    5,      // LZW
    7,      // JPEG
    8,      // Deflate
    0x8005, // PackBits
    0x80B2, // Deflate (old code)
    0xC350, // ZSTD
];

pub(super) fn map_tiff_error(err: tiff::TiffError, cancel: &CancelToken) -> RasterError {
    if cancel.is_cancelled() {
        return RasterError::Cancelled;
    }
    match err {
        tiff::TiffError::IoError(io_err) => match io_err.downcast::<IoError>() {
            Ok(core_err) => RasterError::from(core_err),
            Err(io_err) => RasterError::from(IoError::Io(io_err)),
        },
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
) -> Result<(PixelLayout, u8), RasterError> {
    match color {
        tiff::ColorType::Gray(bits) => Ok((PixelLayout::named(1), bits)),
        tiff::ColorType::RGB(bits) => Ok((PixelLayout::named(3), bits)),
        tiff::ColorType::RGBA(bits) => Ok((PixelLayout::named(4), bits)),
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
            Ok((layout, bit_depth))
        }
        other => Err(RasterError::Decode(format!(
            "color model {other:?} is not supported"
        ))),
    }
}

pub(super) fn resolve_sample_type(sample_format: u16, bits: u8) -> Result<SampleType, RasterError> {
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
) -> Result<(PixelLayout, u8, SampleType), RasterError> {
    let color = decoder.colortype().map_err(|e| map_tiff_error(e, cancel))?;
    let (layout, bits) = model(decoder, color, cancel)?;
    ensure_decodable(decoder, cancel)?;
    let stype = read_sample_type(decoder, bits, cancel)?;
    Ok((layout, bits, stype))
}

fn ensure_decodable(
    decoder: &mut Decoder<BufferedByteReader>,
    cancel: &CancelToken,
) -> Result<(), RasterError> {
    let code =
        opt_tag_strict(decoder.get_tag_unsigned::<u16>(Tag::Compression), cancel)?.unwrap_or(1);
    if DECODED_COMPRESSIONS.contains(&code) {
        return Ok(());
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
    dw: u32,
    dh: u32,
) -> Result<Vec<u8>, RasterError> {
    let pixels = (dw as usize) * (dh as usize);
    let rgba = mapping.rgba(&typed_samples(data)?, pixel, pixels);
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
