//! Planetary display-tile utilities: offline equirectangular basemap tiling,
//! memory-bounded PNG/JPEG map-tile decoding, and CPU linear-light mipmaps.

mod basemap;
mod map_tile;
mod mipmap;

pub use basemap::{BundledBasemap, BundledBasemapConfig};
pub use map_tile::{decode_map_tile, decode_map_tile_with_mipmaps};
pub use mipmap::generate_linear_mipmaps;
