//! Planetary and overlay tile utilities: offline equirectangular basemap
//! tiling, memory-bounded PNG/JPEG map-tile decoding, lossless QOI RGBA8 and
//! byte-shuffled LZ4 scientific/categorical tile encoding, and CPU mipmaps.

mod basemap;
mod lz4_raw;
mod map_tile;
mod mipmap;
mod qoi;

pub use basemap::{BundledBasemap, BundledBasemapConfig};
pub use lz4_raw::{MAX_LZ4_RAW_DECODED_BYTES, decode_lz4_raw_tile, encode_lz4_raw_tile};
pub use map_tile::decode_map_tile_with_mipmaps;
pub use mipmap::{generate_linear_mipmaps, generate_scientific_mipmaps, generate_tile_mipmaps};
pub use qoi::{MAX_QOI_DECODED_BYTES, decode_qoi_rgba8, encode_qoi_rgba8};
