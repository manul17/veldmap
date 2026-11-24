//! Planetary and overlay display-tile utilities: offline equirectangular
//! basemap tiling, memory-bounded PNG/JPEG map-tile decoding, lossless QOI
//! RGBA8 tile encoding, and CPU linear-light mipmaps.

mod basemap;
mod map_tile;
mod mipmap;
mod qoi;

pub use basemap::{BundledBasemap, BundledBasemapConfig};
pub use map_tile::{decode_map_tile, decode_map_tile_with_mipmaps};
pub use mipmap::generate_linear_mipmaps;
pub use qoi::{MAX_QOI_DECODED_BYTES, decode_qoi_rgba8, encode_qoi_rgba8};
