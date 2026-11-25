//! Overlay disk-cache tile codecs: lossless QOI for 8-bit `Rgba8UnormSrgb`
//! display tiles and byte-shuffled LZ4 (`Lz4Raw`) for scientific (`R32Float`)
//! and categorical (`R8Uint` / `R16Uint`) tiles.
//!
//! Kept inside `veldmap-raster` so `veldmap-storage` remains strictly
//! codec-free storage mechanics.

mod lz4_raw;
mod qoi;

/// Default maximum decoded tile byte size admitted for overlay disk-cache tile
/// codecs (`16 MiB`, matching the tile size cap in `veldmap-storage`).
pub const MAX_CACHED_TILE_DECODED_BYTES: usize = 16 * 1024 * 1024;

pub use lz4_raw::{MAX_LZ4_RAW_DECODED_BYTES, decode_lz4_raw_tile, encode_lz4_raw_tile};
pub use qoi::{MAX_QOI_DECODED_BYTES, decode_qoi_rgba8, encode_qoi_rgba8};
