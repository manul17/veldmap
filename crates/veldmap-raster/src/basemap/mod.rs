//! Planetary basemap tile providers: virtual offline equirectangular basemap
//! tiling and memory-bounded PNG/JPEG web map-tile decoding.

mod bundled;
mod map_tile;
mod polar_rim;

pub use bundled::{BundledBasemap, BundledBasemapConfig};
pub use map_tile::decode_map_tile_with_mipmaps;
pub use polar_rim::{
    POLAR_RIM_BINS, PolarRimPyramid, extract_srgb_rim_row, render_harmonic_polar_cap_with_mipmaps,
};
