//! Unified tile addressing (`TileKey`, `TileScheme`, `PlanetaryScheme`,
//! `GlobeScheme`, `BandKey`, `PixelFormat`), pure O(1) LRU `TileCache<K, V>`,
//! planetary tile projection/sampling, and scheme-matched WGS84 planetary
//! quadtree with adaptive Polar Stereographic caps.

pub mod cache;
pub mod decoded;
pub mod geometry;
pub mod key;
pub mod quadtree;
pub mod scheduler;

pub use cache::TileCache;
pub use decoded::{
    DecodedTile, DecodedTileError, max_mip_levels, mip_chain_byte_size, mip_dimensions,
};
pub use geometry::{
    LatLonTileBounds, POLAR_R_CAP, lat_lon_bounds, polar_cap_point, sample_tile_surface,
    web_mercator_y_bounds,
};
pub use key::{
    BandKey, GlobeScheme, PixelFormat, PlanetaryScheme, QuadTileId, SourceId, TileKey, TileScheme,
    compose_uv_rect, fnv1a_64,
};
pub use quadtree::{
    DEFAULT_GRID_CELLS, DEFAULT_POLAR_MAX_TEX_LEVEL, DEFAULT_TILE_PX, QuadtreeConfig, VisibleTile,
    describe_tile, traverse,
};
pub use scheduler::{
    LevelPlan, PassBusy, PassKey, PyramidMeta, ServeMode, TileFailureOutcome, TileScheduler,
};
