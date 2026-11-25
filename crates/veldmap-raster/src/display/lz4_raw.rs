//! Pure-Rust byte-shuffled LZ4 block codec (`TileBlobEncoding::Lz4Raw`) for
//! mip-0 scientific (`R32Float`) and categorical (`R8Uint`/`R16Uint`) tiles.
//!
//! Kept inside `veldmap-raster` so `veldmap-storage` remains strictly
//! codec-free. Payloads carry a compact 24-byte self-describing header
//! validated **before** allocating decompression buffers.

use veldmap_core::tile::{DecodedTile, PixelFormat};

use crate::RasterError;

/// Default maximum decoded tile byte size admitted for `Lz4Raw` caching
/// (`16 MiB`, matching `MAX_QOI_DECODED_BYTES` and `veldmap-storage`).
pub const MAX_LZ4_RAW_DECODED_BYTES: usize = 16 * 1024 * 1024;

const MAGIC: [u8; 4] = *b"VML4";
const VERSION: u8 = 1;
const HEADER_LEN: usize = 24;
const FLAG_BYTE_SHUFFLED: u8 = 1;

fn format_to_tag(format: PixelFormat) -> Result<u8, RasterError> {
    match format {
        PixelFormat::R32Float => Ok(1),
        PixelFormat::R8Uint => Ok(2),
        PixelFormat::R16Uint => Ok(3),
        PixelFormat::Rgba8UnormSrgb => Err(RasterError::Decode(
            "Lz4Raw codec is for scientific/categorical formats; use QOI for Rgba8UnormSrgb"
                .to_owned(),
        )),
    }
}

fn tag_to_format(tag: u8) -> Result<PixelFormat, RasterError> {
    match tag {
        1 => Ok(PixelFormat::R32Float),
        2 => Ok(PixelFormat::R8Uint),
        3 => Ok(PixelFormat::R16Uint),
        other => Err(RasterError::Decode(format!(
            "unknown Lz4Raw pixel format tag {other}"
        ))),
    }
}

/// Byte-shuffle tightly packed `elem_size`-byte elements so corresponding byte
/// lanes (e.g., IEEE-754 sign/exponent bytes) cluster contiguously for LZ4.
fn byte_shuffle(src: &[u8], elem_size: usize) -> Vec<u8> {
    if elem_size <= 1 {
        return src.to_vec();
    }
    let count = src.len() / elem_size;
    let mut dst = vec![0u8; src.len()];
    for (i, chunk) in src.chunks_exact(elem_size).enumerate() {
        for (lane, &byte) in chunk.iter().enumerate() {
            dst[lane * count + i] = byte;
        }
    }
    dst
}

/// Inverse of [`byte_shuffle`].
fn byte_unshuffle(src: &[u8], elem_size: usize) -> Vec<u8> {
    if elem_size <= 1 {
        return src.to_vec();
    }
    let count = src.len() / elem_size;
    let mut dst = vec![0u8; src.len()];
    for (i, chunk) in dst.chunks_exact_mut(elem_size).enumerate() {
        for (lane, byte) in chunk.iter_mut().enumerate() {
            *byte = src[lane * count + i];
        }
    }
    dst
}

/// Encode mip-0 of a scientific or categorical [`DecodedTile`] into a
/// self-describing byte-shuffled `Lz4Raw` blob.
///
/// Enforces `tile.data().len() <= max_decoded_bytes` symmetrically with
/// [`decode_lz4_raw_tile`].
pub fn encode_lz4_raw_tile(
    tile: &DecodedTile,
    max_decoded_bytes: usize,
) -> Result<Vec<u8>, RasterError> {
    let format = tile.format();
    let format_tag = format_to_tag(format)?;
    let raw = tile.data();
    let width = tile.width();
    let height = tile.height();
    let valid_w = tile.valid_w();
    let valid_h = tile.valid_h();
    let gutter = tile.gutter();

    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|px| px.checked_mul(format.bytes_per_pixel()));
    if width == 0 || height == 0 || expected != Some(raw.len()) {
        return Err(RasterError::InvalidDimensions { width, height });
    }
    if raw.len() > max_decoded_bytes {
        return Err(RasterError::LimitExceeded {
            width,
            height,
            estimated_bytes: raw.len() as u64,
            limit_bytes: max_decoded_bytes as u64,
        });
    }

    let bpp = format.bytes_per_pixel();
    let (flags, compressed) = if bpp > 1 {
        let shuffled = byte_shuffle(raw, bpp);
        (FLAG_BYTE_SHUFFLED, lz4_flex::block::compress(&shuffled))
    } else {
        (0u8, lz4_flex::block::compress(raw))
    };

    let mut out = Vec::with_capacity(HEADER_LEN + compressed.len());
    out.extend_from_slice(&MAGIC);
    out.push(VERSION);
    out.push(format_tag);
    out.push(gutter);
    out.push(flags);
    out.extend_from_slice(&width.to_le_bytes());
    out.extend_from_slice(&height.to_le_bytes());
    out.extend_from_slice(&valid_w.to_le_bytes());
    out.extend_from_slice(&valid_h.to_le_bytes());
    out.extend_from_slice(&compressed);
    Ok(out)
}

/// Decode a self-describing byte-shuffled `Lz4Raw` blob into a mip-0
/// [`DecodedTile`], validating magic, version, format, geometry, and
/// `max_decoded_bytes` **before** allocating decompression buffers.
pub fn decode_lz4_raw_tile(
    payload: &[u8],
    expected_format: PixelFormat,
    max_decoded_bytes: usize,
) -> Result<DecodedTile, RasterError> {
    if payload.len() < HEADER_LEN {
        return Err(RasterError::Decode(
            "truncated Lz4Raw tile header".to_owned(),
        ));
    }
    if payload[0..4] != MAGIC {
        return Err(RasterError::Decode("invalid Lz4Raw magic".to_owned()));
    }
    let version = payload[4];
    if version != VERSION {
        return Err(RasterError::Decode(format!(
            "unsupported Lz4Raw version {version}"
        )));
    }
    let format = tag_to_format(payload[5])?;
    if format != expected_format {
        return Err(RasterError::Decode(format!(
            "Lz4Raw format mismatch: expected {expected_format:?}, got {format:?}"
        )));
    }
    let gutter = payload[6];
    let flags = payload[7];
    if flags & !FLAG_BYTE_SHUFFLED != 0 {
        return Err(RasterError::Decode(format!(
            "unknown Lz4Raw flags 0x{flags:02x}"
        )));
    }
    let read_u32 = |off: usize| -> u32 {
        u32::from_le_bytes([
            payload[off],
            payload[off + 1],
            payload[off + 2],
            payload[off + 3],
        ])
    };
    let width = read_u32(8);
    let height = read_u32(12);
    let valid_w = read_u32(16);
    let valid_h = read_u32(20);

    if width == 0 || height == 0 || valid_w == 0 || valid_h == 0 {
        return Err(RasterError::InvalidDimensions { width, height });
    }
    let gutter_twice = u32::from(gutter).saturating_mul(2);
    if valid_w.saturating_add(gutter_twice) > width || valid_h.saturating_add(gutter_twice) > height
    {
        return Err(RasterError::InvalidDimensions { width, height });
    }

    let bpp = format.bytes_per_pixel();
    let Some(expected_bytes) = (width as usize)
        .checked_mul(height as usize)
        .and_then(|px| px.checked_mul(bpp))
    else {
        return Err(RasterError::LimitExceeded {
            width,
            height,
            estimated_bytes: u64::MAX,
            limit_bytes: max_decoded_bytes as u64,
        });
    };
    if expected_bytes > max_decoded_bytes {
        return Err(RasterError::LimitExceeded {
            width,
            height,
            estimated_bytes: expected_bytes as u64,
            limit_bytes: max_decoded_bytes as u64,
        });
    }

    let decompressed = lz4_flex::block::decompress(&payload[HEADER_LEN..], expected_bytes)
        .map_err(|e| RasterError::Decode(format!("Lz4Raw decompression failed: {e}")))?;
    if decompressed.len() != expected_bytes {
        return Err(RasterError::Decode(format!(
            "Lz4Raw decompressed {} bytes, expected {expected_bytes}",
            decompressed.len()
        )));
    }

    let data = if flags & FLAG_BYTE_SHUFFLED != 0 && bpp > 1 {
        byte_unshuffle(&decompressed, bpp)
    } else {
        decompressed
    };

    DecodedTile::try_with_valid_region(width, height, valid_w, valid_h, gutter, format, data)
        .map_err(|e| RasterError::Decode(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lz4_raw_round_trips_scientific_and_categorical_formats() {
        // R32Float including NaN bit patterns and smooth gradients (high shuffle compression).
        let (w, h) = (32u32, 24u32);
        let mut f32_bytes = Vec::with_capacity((w * h * 4) as usize);
        for i in 0..(w * h) {
            let v = if i % 7 == 0 {
                f32::NAN
            } else {
                100.0 + (i as f32) * 0.25
            };
            f32_bytes.extend_from_slice(&v.to_ne_bytes());
        }
        let tile =
            DecodedTile::try_new(w, h, PixelFormat::R32Float, f32_bytes.clone()).expect("tile");
        let blob = encode_lz4_raw_tile(&tile, MAX_LZ4_RAW_DECODED_BYTES).expect("encode f32");
        assert!(
            blob.len() < f32_bytes.len(),
            "byte-shuffled LZ4 should compress smooth f32 data"
        );
        let decoded = decode_lz4_raw_tile(&blob, PixelFormat::R32Float, MAX_LZ4_RAW_DECODED_BYTES)
            .expect("decode f32");
        assert_eq!(decoded.width(), w);
        assert_eq!(decoded.height(), h);
        assert_eq!(decoded.valid_w(), w);
        assert_eq!(decoded.valid_h(), h);
        assert_eq!(decoded.data(), f32_bytes.as_slice());

        // R8Uint and R16Uint with gutter / partial valid region.
        let u8_tile = DecodedTile::try_with_valid_region(
            18,
            18,
            16,
            16,
            1,
            PixelFormat::R8Uint,
            vec![7u8; 18 * 18],
        )
        .expect("u8 tile");
        let u8_blob = encode_lz4_raw_tile(&u8_tile, MAX_LZ4_RAW_DECODED_BYTES).expect("encode u8");
        let u8_dec = decode_lz4_raw_tile(&u8_blob, PixelFormat::R8Uint, MAX_LZ4_RAW_DECODED_BYTES)
            .expect("decode u8");
        assert_eq!(
            (u8_dec.valid_w(), u8_dec.valid_h(), u8_dec.gutter()),
            (16, 16, 1)
        );
        assert_eq!(u8_dec.data(), u8_tile.data());

        let mut u16_bytes = Vec::with_capacity(16 * 16 * 2);
        for i in 0u16..(16 * 16) {
            u16_bytes.extend_from_slice(&(i * 3).to_ne_bytes());
        }
        let u16_tile =
            DecodedTile::try_new(16, 16, PixelFormat::R16Uint, u16_bytes.clone()).expect("u16");
        let u16_blob =
            encode_lz4_raw_tile(&u16_tile, MAX_LZ4_RAW_DECODED_BYTES).expect("encode u16");
        let u16_dec =
            decode_lz4_raw_tile(&u16_blob, PixelFormat::R16Uint, MAX_LZ4_RAW_DECODED_BYTES)
                .expect("decode u16");
        assert_eq!(u16_dec.data(), u16_bytes.as_slice());
    }

    #[test]
    fn lz4_raw_enforces_header_bounds_format_and_corruption_guards() {
        let rgba_tile =
            DecodedTile::try_new(4, 4, PixelFormat::Rgba8UnormSrgb, vec![0u8; 64]).expect("rgba");
        assert!(encode_lz4_raw_tile(&rgba_tile, MAX_LZ4_RAW_DECODED_BYTES).is_err());

        let small = DecodedTile::try_new(8, 8, PixelFormat::R32Float, vec![0u8; 256]).expect("ok");
        assert!(matches!(
            encode_lz4_raw_tile(&small, 128),
            Err(RasterError::LimitExceeded { .. })
        ));
        let blob = encode_lz4_raw_tile(&small, MAX_LZ4_RAW_DECODED_BYTES).expect("encode");

        // Format mismatch rejected.
        assert!(
            decode_lz4_raw_tile(&blob, PixelFormat::R8Uint, MAX_LZ4_RAW_DECODED_BYTES).is_err()
        );

        // Oversized crafted header rejected BEFORE allocation.
        let mut bomb = Vec::with_capacity(HEADER_LEN + 4);
        bomb.extend_from_slice(&MAGIC);
        bomb.push(VERSION);
        bomb.push(1); // R32Float
        bomb.push(0);
        bomb.push(FLAG_BYTE_SHUFFLED);
        bomb.extend_from_slice(&10_000u32.to_le_bytes());
        bomb.extend_from_slice(&10_000u32.to_le_bytes());
        bomb.extend_from_slice(&10_000u32.to_le_bytes());
        bomb.extend_from_slice(&10_000u32.to_le_bytes());
        bomb.extend_from_slice(&[0u8; 4]);
        assert!(matches!(
            decode_lz4_raw_tile(&bomb, PixelFormat::R32Float, MAX_LZ4_RAW_DECODED_BYTES),
            Err(RasterError::LimitExceeded { .. })
        ));

        // Truncated payload rejected.
        assert!(
            decode_lz4_raw_tile(
                &blob[..HEADER_LEN],
                PixelFormat::R32Float,
                MAX_LZ4_RAW_DECODED_BYTES
            )
            .is_err()
        );
    }
}
