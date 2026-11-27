//! Unified tile addressing (`TileKey`, `TileScheme`, `PlanetaryScheme`,
//! `GlobeScheme`, `BandKey`, `PixelFormat`, `SourceFingerprint`,
//! `TileStreamSpec`, `StreamKey`), raster pyramid arithmetic (`pyramid`), pure
//! O(1) LRU `TileCache<K, V>`, planetary tile projection/sampling, scheme-matched
//! WGS84 planetary quadtree with adaptive Polar Stereographic caps, and overlay
//! pyramid culling / CPU `f64` RTC warp-mesh generation (`overlay`).

mod cache;
mod decoded;
mod geometry;
mod identity;
mod key;
mod overlay;
mod pyramid;
mod quadtree;
mod scheme;

pub use cache::TileCache;
pub use decoded::{
    DecodedTile, DecodedTileError, max_mip_levels, mip_chain_byte_size, mip_dimensions,
};
pub use geometry::{
    POLAR_R_CAP, TileSurfaceSample, geodetic_lat_bounds, polar_cap_point, polar_cap_uv_bounds,
    sample_tile_surface, tile_lon_bounds, web_mercator_y_bounds,
};
pub use identity::{
    BandKey, PixelFormat, SourceFingerprint, SourceId, StreamKey, TileStreamSpec, fnv1a_64,
};
pub use key::{QuadTileId, TileKey, compose_uv_rect};
#[cfg(any(test, feature = "test-support"))]
pub use overlay::MIN_WARP_SEGMENTS;
pub use overlay::{
    MAX_WARP_SEGMENTS, OverlayCullTree, WarpPatch, build_warp_patch, level_segments,
};
pub use pyramid::{
    CarrierLayout, RASTER_TILE_SIZE, cell_image_rect, cell_uv_rect, grid_dim, level_count,
    level_size, tile_extent,
};
#[cfg(any(test, feature = "test-support"))]
pub use quadtree::describe_tile;
pub use quadtree::{DEFAULT_GRID_CELLS, QuadtreeConfig, VisibleTile, traverse};
pub use scheme::{GlobeScheme, PlanetaryScheme, TileScheme};

#[cfg(test)]
fn make_key(scheme: TileScheme, level: u8, x: u32, y: u32) -> TileKey {
    TileKey {
        source: SourceId::BUNDLED_BASEMAP,
        band: BandKey::Display,
        scheme,
        format: PixelFormat::Rgba8UnormSrgb,
        gutter: 0,
        decode_rev: 0,
        level,
        x,
        y,
    }
}
