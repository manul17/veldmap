//! Unified tile addressing (`TileKey`, `TileScheme`, `PlanetaryScheme`,
//! `GlobeScheme`, `BandKey`, `PixelFormat`, `SourceFingerprint`,
//! `TileStreamSpec`, `StreamKey`), raster pyramid arithmetic (`pyramid`), pure
//! O(1) LRU `TileCache<K, V>`, planetary tile projection/sampling, and
//! scheme-matched WGS84 planetary quadtree with adaptive Polar Stereographic
//! caps.

mod cache;
mod decoded;
mod geometry;
mod identity;
mod key;
mod pyramid;
mod quadtree;
mod scheme;

pub use cache::TileCache;
pub use decoded::{
    DecodedTile, DecodedTileError, max_mip_levels, mip_chain_byte_size, mip_dimensions,
};
pub use geometry::{
    POLAR_R_CAP, TileSurfaceSample, polar_cap_point, sample_tile_surface, web_mercator_y_bounds,
};
pub use identity::{
    BandKey, PixelFormat, SourceFingerprint, SourceId, StreamKey, TileStreamSpec, fnv1a_64,
};
pub use key::{QuadTileId, TileKey, compose_uv_rect};
pub use pyramid::{
    RASTER_TILE_SIZE, cell_image_rect, cell_uv, grid_dim, level_count, level_size, tile_extent,
};
pub use quadtree::{DEFAULT_GRID_CELLS, QuadtreeConfig, VisibleTile, describe_tile, traverse};
pub use scheme::{GlobeScheme, PlanetaryScheme, TileScheme};
