//! Native raster decoding, bounded CPU decode pool, linear-light mipmap
//! generation, and virtual offline equirectangular basemap tiling for VeldMap.
//!
//! Keeps `veldmap-core` permanently free of image/codec dependencies and keeps
//! `veldmap-net` decoupled from raster codecs via an injected decode executor.

use thiserror::Error;
use veldmap_core::tile::PlanetaryScheme;

pub mod basemap;
pub mod decode_pool;
pub mod map_tile;
pub mod mipmap;

pub use basemap::{BUNDLED_MAX_LEVEL, BUNDLED_POLAR_MAX_LEVEL, BUNDLED_TILE_PX, BundledBasemap};
pub use decode_pool::{DecodeCompletion, DecodeJob, DecodePool, DecodePoolHandle};
pub use map_tile::{
    MAX_MAP_TILE_ALLOC_BYTES, MAX_MAP_TILE_DIM, decode_map_tile, decode_map_tile_with_mipmaps,
};
pub use mipmap::generate_linear_mipmaps;

/// Errors returned by raster tile decoders and virtual tile providers.
#[derive(Debug, Error)]
pub enum RasterError {
    #[error("unsupported or unrecognized image format")]
    UnsupportedFormat,
    #[error("image decode failed: {0}")]
    Decode(#[from] image::ImageError),
    #[error("invalid decoded image dimensions {width}x{height}")]
    InvalidDimensions { width: u32, height: u32 },
    #[error("image {width}x{height} exceeds decode limits ({estimated_bytes} B > {limit_bytes} B)")]
    LimitExceeded {
        width: u32,
        height: u32,
        estimated_bytes: u64,
        limit_bytes: u64,
    },
    #[error("tile request out of bounds or above max_level: {scheme:?} z={level} ({x}, {y})")]
    TileOutOfRange {
        scheme: PlanetaryScheme,
        level: u8,
        x: u32,
        y: u32,
    },
}
