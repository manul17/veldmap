//! Unified tile addressing (`TileKey`, `TileScheme`, `BandKey`, `PixelFormat`),
//! pure O(1) LRU `TileCache<K, V>`, and scheme-matched WGS84 planetary quadtree
//! with adaptive Polar Stereographic caps.

pub mod cache;
pub mod decoded;
pub mod key;
pub mod quadtree;

pub use cache::TileCache;
pub use decoded::{DecodedTile, max_mip_levels, mip_dimensions};
pub use key::{
    BandKey, PixelFormat, QuadTileId, SourceId, TileKey, TileScheme, compose_uv_rect, fnv1a_64,
};
pub use quadtree::{
    DEFAULT_GRID_CELLS, LatLonTileBounds, POLAR_R_CAP, QuadtreeConfig, VisibleTile,
    adaptive_grid_step_deg, describe_tile, lat_lon_bounds, polar_cap_point, sample_tile_surface,
    tile_grid_anchor, traverse,
};
