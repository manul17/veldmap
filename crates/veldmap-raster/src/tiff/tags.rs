//! TIFF error mapping, typed optional tag readers, and GDAL nodata parsing.

use tiff::decoder::Decoder;
use tiff::tags::Tag;
use veldmap_core::io::BufferedByteReader;
use veldmap_core::sync::CancelToken;

use crate::{RasterError, map_std_io_error};

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
pub(super) fn opt_tag_strict<T>(
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

pub(super) fn read_tag_numeric_f64_vec_opt(
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

pub(super) fn read_tag_u16_vec_opt(
    decoder: &mut Decoder<BufferedByteReader>,
    tag: Tag,
    cancel: &CancelToken,
) -> Result<Vec<u16>, RasterError> {
    Ok(opt_tag(decoder.get_tag_u16_vec(tag), cancel)?.unwrap_or_default())
}

fn value_to_u64(val: &tiff::decoder::ifd::Value) -> Option<u64> {
    use tiff::decoder::ifd::Value;
    match *val {
        Value::Byte(v) => Some(u64::from(v)),
        Value::Short(v) => Some(u64::from(v)),
        Value::Unsigned(v) => Some(u64::from(v)),
        Value::UnsignedBig(v) => Some(v),
        _ => None,
    }
}

pub(super) fn read_tag_u64_vec_opt(
    decoder: &mut Decoder<BufferedByteReader>,
    tag: Tag,
    cancel: &CancelToken,
) -> Result<Option<Vec<u64>>, RasterError> {
    let Some(val) = opt_tag(decoder.get_tag(tag), cancel)? else {
        return Ok(None);
    };
    let out = match val {
        tiff::decoder::ifd::Value::List(items) => items.iter().filter_map(value_to_u64).collect(),
        single => value_to_u64(&single).into_iter().collect(),
    };
    Ok(Some(out))
}

pub(super) fn read_tag_u8_vec_opt(
    decoder: &mut Decoder<BufferedByteReader>,
    tag: Tag,
    cancel: &CancelToken,
) -> Result<Option<Vec<u8>>, RasterError> {
    let Some(val) = opt_tag(decoder.get_tag(tag), cancel)? else {
        return Ok(None);
    };
    let out = match val {
        tiff::decoder::ifd::Value::List(items) => items
            .iter()
            .filter_map(|v| match *v {
                tiff::decoder::ifd::Value::Byte(b) => Some(b),
                _ => None,
            })
            .collect(),
        tiff::decoder::ifd::Value::Byte(b) => vec![b],
        _ => return Ok(None),
    };
    Ok(Some(out))
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
    fn overview_bit_filters_reduced_resolution_and_excludes_masks() {
        assert!(!is_overview(0));
        assert!(is_overview(1));
        assert!(!is_overview(5)); // mask bit set
    }
}
