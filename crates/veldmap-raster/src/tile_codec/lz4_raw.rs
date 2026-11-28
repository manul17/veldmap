//! Pure-Rust byte-shuffled LZ4 block codec (`TileBlobEncoding::Lz4Raw`) for
//! mip-0 scientific (`R32Float`) and categorical (`R8Uint`/`R16Uint`) tiles.
//!
//! Kept inside `veldmap-raster` so `veldmap-storage` remains strictly
//! codec-free. Payloads carry a compact 24-byte little-endian self-describing
//! header validated **before** allocating decompression buffers. Multi-byte
//! sample payloads (`R32Float`/`R16Uint`) are stored in host native endianness
//! (matching `DecodedTile` and desktop x86_64/aarch64 little-endian targets)
//! inside the local ephemeral SQLite tile cache.

use veldmap_core::tile::{DecodedTile, PixelFormat};

use crate::RasterError;

use super::checked_tile_bytes;

const MAGIC: [u8; 4] = *b"VML4";
const VERSION: u8 = 1;
const HEADER_LEN: usize = 24;
const FLAG_BYTE_SHUFFLED: u8 = 1;

fn format_to_tag(format: PixelFormat) -> Result<u8, RasterError> {
    match format {
        PixelFormat::R32Float => Ok(1),
        PixelFormat::R8Uint => Ok(2),
        PixelFormat::R16Uint => Ok(3),
        PixelFormat::Rgba8UnormSrgb => Err(RasterError::Encode(
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

    let bpp = format.bytes_per_pixel();
    let expected = checked_tile_bytes(width, height, bpp, max_decoded_bytes)?;
    if raw.len() != expected {
        return Err(RasterError::InvalidDimensions { width, height });
    }
    let shuffled;
    let (flags, src): (u8, &[u8]) = if bpp > 1 {
        shuffled = byte_shuffle(raw, bpp);
        (FLAG_BYTE_SHUFFLED, &shuffled)
    } else {
        (0u8, raw)
    };

    let max_comp = lz4_flex::block::get_maximum_output_size(src.len());
    let mut out = vec![0u8; HEADER_LEN + max_comp];
    out[0..4].copy_from_slice(&MAGIC);
    out[4] = VERSION;
    out[5] = format_tag;
    out[6] = gutter;
    out[7] = flags;
    out[8..12].copy_from_slice(&width.to_le_bytes());
    out[12..16].copy_from_slice(&height.to_le_bytes());
    out[16..20].copy_from_slice(&valid_w.to_le_bytes());
    out[20..24].copy_from_slice(&valid_h.to_le_bytes());
    let written = lz4_flex::block::compress_into(src, &mut out[HEADER_LEN..])
        .map_err(|e| RasterError::Encode(format!("Lz4Raw compression failed: {e}")))?;
    out.truncate(HEADER_LEN + written);
    Ok(out)
}

struct Lz4Header {
    format: PixelFormat,
    gutter: u8,
    flags: u8,
    width: u32,
    height: u32,
    valid_w: u32,
    valid_h: u32,
}

fn parse_lz4_header(
    payload: &[u8],
    expected_format: PixelFormat,
) -> Result<Lz4Header, RasterError> {
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
    if valid_w.checked_add(gutter_twice).is_none_or(|w| w > width)
        || valid_h.checked_add(gutter_twice).is_none_or(|h| h > height)
    {
        return Err(RasterError::InvalidDimensions { width, height });
    }

    Ok(Lz4Header {
        format,
        gutter,
        flags,
        width,
        height,
        valid_w,
        valid_h,
    })
}

/// Decode a self-describing byte-shuffled `Lz4Raw` blob into a mip-0
/// [`DecodedTile`], validating magic, version, format, geometry, and
/// `max_decoded_bytes` **before** allocating decompression buffers.
pub fn decode_lz4_raw_tile(
    payload: &[u8],
    expected_format: PixelFormat,
    max_decoded_bytes: usize,
) -> Result<DecodedTile, RasterError> {
    let header = parse_lz4_header(payload, expected_format)?;
    let bpp = header.format.bytes_per_pixel();
    let expected_bytes = checked_tile_bytes(header.width, header.height, bpp, max_decoded_bytes)?;

    let decompressed = lz4_flex::block::decompress(&payload[HEADER_LEN..], expected_bytes)
        .map_err(|e| RasterError::Decode(format!("Lz4Raw decompression failed: {e}")))?;
    if decompressed.len() != expected_bytes {
        return Err(RasterError::Decode(format!(
            "Lz4Raw decompressed {} bytes, expected {expected_bytes}",
            decompressed.len()
        )));
    }

    let data = if header.flags & FLAG_BYTE_SHUFFLED != 0 && bpp > 1 {
        byte_unshuffle(&decompressed, bpp)
    } else {
        decompressed
    };

    DecodedTile::try_with_valid_region(
        header.width,
        header.height,
        header.valid_w,
        header.valid_h,
        header.gutter,
        header.format,
        data,
    )
    .map_err(|e| RasterError::Decode(e.to_string()))
}

#[cfg(test)]
#[path = "lz4_raw_tests.rs"]
mod tests;
