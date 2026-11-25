//! Pure-Rust lossless QOI codec helpers for 8-bit `Rgba8UnormSrgb` display
//! tiles, kept inside `veldmap-raster` so `veldmap-storage` remains strictly
//! codec-free storage mechanics.

use crate::RasterError;

/// Default maximum decoded RGBA8 byte size admitted for QOI tile caching
/// (`16 MiB`, matching the disk-cache tile size cap in `veldmap-storage`).
pub const MAX_QOI_DECODED_BYTES: usize = 16 * 1024 * 1024;

/// Encode an 8-bit `Rgba8UnormSrgb` tile into lossless QOI bytes.
///
/// Symmetrically enforces `width > 0`, `height > 0`, exact `width * height * 4`
/// length, and `rgba.len() <= max_decoded_bytes` so no encoded entry is ever
/// written that [`decode_qoi_rgba8`] would immediately reject.
pub fn encode_qoi_rgba8(
    width: u32,
    height: u32,
    rgba: &[u8],
    max_decoded_bytes: usize,
) -> Result<Vec<u8>, RasterError> {
    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|px| px.checked_mul(4));
    if width == 0 || height == 0 || expected != Some(rgba.len()) {
        return Err(RasterError::InvalidDimensions { width, height });
    }
    if rgba.len() > max_decoded_bytes {
        return Err(RasterError::LimitExceeded {
            width,
            height,
            estimated_bytes: rgba.len() as u64,
            limit_bytes: max_decoded_bytes as u64,
        });
    }
    qoi::encode_to_vec(rgba, width, height).map_err(|e| RasterError::Decode(e.to_string()))
}

/// Decode a lossless QOI-encoded 8-bit `Rgba8UnormSrgb` tile after validating
/// the header (`width`, `height`, `Channels::Rgba`, `ColorSpace::Srgb`, and
/// `width * height * 4 <= max_decoded_bytes`) *before* allocating pixel memory.
pub fn decode_qoi_rgba8(
    payload: &[u8],
    max_decoded_bytes: usize,
) -> Result<(u32, u32, Vec<u8>), RasterError> {
    let header = qoi::decode_header(payload).map_err(|e| RasterError::Decode(e.to_string()))?;
    let expected = (header.width as usize)
        .checked_mul(header.height as usize)
        .and_then(|px| px.checked_mul(4));
    if header.width == 0 || header.height == 0 {
        return Err(RasterError::InvalidDimensions {
            width: header.width,
            height: header.height,
        });
    }
    if header.channels != qoi::Channels::Rgba || header.colorspace != qoi::ColorSpace::Srgb {
        return Err(RasterError::Decode(format!(
            "unexpected QOI format: channels={:?}, colorspace={:?}",
            header.channels, header.colorspace
        )));
    }
    let Some(expected_bytes) = expected else {
        return Err(RasterError::LimitExceeded {
            width: header.width,
            height: header.height,
            estimated_bytes: u64::MAX,
            limit_bytes: max_decoded_bytes as u64,
        });
    };
    if expected_bytes > max_decoded_bytes {
        return Err(RasterError::LimitExceeded {
            width: header.width,
            height: header.height,
            estimated_bytes: expected_bytes as u64,
            limit_bytes: max_decoded_bytes as u64,
        });
    }
    let (decoded_header, rgba) =
        qoi::decode_to_vec(payload).map_err(|e| RasterError::Decode(e.to_string()))?;
    if decoded_header.width != header.width
        || decoded_header.height != header.height
        || decoded_header.channels != qoi::Channels::Rgba
        || decoded_header.colorspace != qoi::ColorSpace::Srgb
        || rgba.len() != expected_bytes
    {
        return Err(RasterError::Decode(
            "QOI decoded header or byte length mismatch".to_owned(),
        ));
    }
    Ok((decoded_header.width, decoded_header.height, rgba))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qoi_round_trips_and_enforces_symmetric_bounds_and_format_checks() {
        let (w, h) = (32u32, 24u32);
        let mut rgba = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                rgba.extend_from_slice(&[
                    (x * 7) as u8,
                    (y * 11) as u8,
                    ((x ^ y) * 5) as u8,
                    if (x + y) % 5 == 0 { 0 } else { 255 },
                ]);
            }
        }
        let encoded = encode_qoi_rgba8(w, h, &rgba, MAX_QOI_DECODED_BYTES).expect("encode");
        let (dw, dh, decoded) = decode_qoi_rgba8(&encoded, MAX_QOI_DECODED_BYTES).expect("decode");
        assert_eq!((dw, dh), (w, h));
        assert_eq!(decoded, rgba);

        // Symmetric size limit rejects encode before writing oversized tile.
        assert!(matches!(
            encode_qoi_rgba8(w, h, &rgba, rgba.len() - 1),
            Err(RasterError::LimitExceeded { .. })
        ));
        // Bad buffer length / zero dimensions rejected.
        assert!(matches!(
            encode_qoi_rgba8(0, 4, &[], MAX_QOI_DECODED_BYTES),
            Err(RasterError::InvalidDimensions { .. })
        ));
        assert!(matches!(
            encode_qoi_rgba8(4, 4, &[0u8; 8], MAX_QOI_DECODED_BYTES),
            Err(RasterError::InvalidDimensions { .. })
        ));

        // Oversized header (10000x10000 = 400 MiB) rejected before allocation.
        let mut bad_header = Vec::with_capacity(22);
        bad_header.extend_from_slice(b"qoif");
        bad_header.extend_from_slice(&10_000u32.to_be_bytes());
        bad_header.extend_from_slice(&10_000u32.to_be_bytes());
        bad_header.push(4);
        bad_header.push(0);
        bad_header.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 1]);
        assert!(matches!(
            decode_qoi_rgba8(&bad_header, MAX_QOI_DECODED_BYTES),
            Err(RasterError::LimitExceeded { .. })
        ));

        // Truncated payload after valid header is rejected cleanly.
        assert!(decode_qoi_rgba8(&encoded[..encoded.len() / 2], MAX_QOI_DECODED_BYTES).is_err());

        // 3-channel RGB QOI blob is rejected.
        let rgb_only = qoi::encode_to_vec([10u8; 4 * 4 * 3], 4, 4).expect("rgb qoi");
        assert!(decode_qoi_rgba8(&rgb_only, MAX_QOI_DECODED_BYTES).is_err());

        // Linear colorspace (colorspace = 1) header is rejected.
        let mut linear = encoded.clone();
        linear[13] = 1;
        assert!(decode_qoi_rgba8(&linear, MAX_QOI_DECODED_BYTES).is_err());
    }
}
