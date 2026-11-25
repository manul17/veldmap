//! Planetary basemap tile providers: virtual offline equirectangular basemap
//! tiling and memory-bounded PNG/JPEG web map-tile decoding.

mod bundled;
mod map_tile;

pub use bundled::{BundledBasemap, BundledBasemapConfig};
pub use map_tile::decode_map_tile_with_mipmaps;
