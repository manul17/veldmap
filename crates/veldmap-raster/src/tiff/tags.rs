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

fn value_to_u16(val: &tiff::decoder::ifd::Value, tag: Tag) -> Result<u16, RasterError> {
    use tiff::decoder::ifd::Value;
    match *val {
        Value::Byte(v) => Ok(u16::from(v)),
        Value::Short(v) => Ok(v),
        Value::Unsigned(v) => u16::try_from(v)
            .map_err(|_| RasterError::Decode(format!("TIFF tag {tag:?} value {v} overflows u16"))),
        Value::UnsignedBig(v) => u16::try_from(v)
            .map_err(|_| RasterError::Decode(format!("TIFF tag {tag:?} value {v} overflows u16"))),
        _ => Err(RasterError::Decode(format!(
            "TIFF tag {tag:?} has unexpected non-integer value type"
        ))),
    }
}

/// Strictly read an optional unsigned 16-bit integer tag (scalar or list,
/// admitting `BYTE`/`SHORT`/`LONG`/`LONG8` if values fit in `u16`), propagating
/// type errors and `LimitsExceeded` instead of silently falling back.
pub(super) fn read_tag_u16_vec_strict(
    decoder: &mut Decoder<BufferedByteReader>,
    tag: Tag,
    cancel: &CancelToken,
) -> Result<Option<Vec<u16>>, RasterError> {
    let Some(val) = opt_tag_strict(decoder.get_tag(tag), cancel)? else {
        return Ok(None);
    };
    let out = match val {
        tiff::decoder::ifd::Value::List(items) => items
            .iter()
            .map(|v| value_to_u16(v, tag))
            .collect::<Result<Vec<_>, _>>()?,
        single => vec![value_to_u16(&single, tag)?],
    };
    Ok(Some(out))
}

fn value_to_u64(val: &tiff::decoder::ifd::Value, tag: Tag) -> Result<u64, RasterError> {
    use tiff::decoder::ifd::Value;
    match *val {
        Value::Byte(v) => Ok(u64::from(v)),
        Value::Short(v) => Ok(u64::from(v)),
        Value::Unsigned(v) => Ok(u64::from(v)),
        Value::UnsignedBig(v) => Ok(v),
        _ => Err(RasterError::Decode(format!(
            "TIFF tag {tag:?} has unexpected non-integer value type"
        ))),
    }
}

/// Strictly read an optional unsigned 64-bit integer tag (scalar or list,
/// admitting only `BYTE`/`SHORT`/`LONG`/`LONG8`), rejecting malformed entries
/// instead of silently dropping them.
pub(super) fn read_tag_u64_vec_strict(
    decoder: &mut Decoder<BufferedByteReader>,
    tag: Tag,
    cancel: &CancelToken,
) -> Result<Option<Vec<u64>>, RasterError> {
    let Some(val) = opt_tag_strict(decoder.get_tag(tag), cancel)? else {
        return Ok(None);
    };
    let out = match val {
        tiff::decoder::ifd::Value::List(items) => items
            .iter()
            .map(|v| value_to_u64(v, tag))
            .collect::<Result<Vec<_>, _>>()?,
        single => vec![value_to_u64(&single, tag)?],
    };
    Ok(Some(out))
}

/// Strictly read an optional raw byte-array tag (`BYTE` or `UNDEFINED`, such as
/// `JPEGTables`), propagating type and limit errors.
pub(super) fn read_tag_u8_vec_strict(
    decoder: &mut Decoder<BufferedByteReader>,
    tag: Tag,
    cancel: &CancelToken,
) -> Result<Option<Vec<u8>>, RasterError> {
    opt_tag_strict(decoder.get_tag_u8_vec(tag), cancel)
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
