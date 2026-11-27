//! Overlay disk-cache tile codecs: lossless QOI for 8-bit `Rgba8UnormSrgb`
//! display tiles and byte-shuffled LZ4 (`Lz4Raw`) for scientific (`R32Float`)
//! and categorical (`R8Uint` / `R16Uint`) tiles.
//!
//! Kept inside `veldmap-raster` so `veldmap-storage` remains strictly
//! codec-free storage mechanics.

mod lz4_raw;
mod qoi;

use crate::RasterError;

/// Default maximum decoded tile byte size admitted for overlay disk-cache tile
/// codecs (`16 MiB`, matching the tile size cap in `veldmap-storage`).
pub const MAX_CACHED_TILE_DECODED_BYTES: usize = 16 * 1024 * 1024;

/// Validate non-zero tile dimensions `(width, height)`, compute exact tightly
/// packed byte size `width * height * bpp`, and enforce `<= max_decoded_bytes`.
pub(super) fn checked_tile_bytes(
    width: u32,
    height: u32,
    bpp: usize,
    max_decoded_bytes: usize,
) -> Result<usize, RasterError> {
    if width == 0 || height == 0 {
        return Err(RasterError::InvalidDimensions { width, height });
    }
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
    Ok(expected_bytes)
}

pub use lz4_raw::{decode_lz4_raw_tile, encode_lz4_raw_tile};
pub use qoi::{decode_qoi_rgba8, encode_qoi_rgba8};
