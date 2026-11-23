//! Native raster decoding, bounded CPU decode pool, linear-light mipmap
//! generation, and virtual offline equirectangular basemap tiling for VeldMap.
//!
//! Keeps `veldmap-core` permanently free of image/codec dependencies and keeps
//! `veldmap-net` decoupled from raster codecs via an injected decode executor.

use thiserror::Error;
use veldmap_core::budget::BudgetError;
use veldmap_core::io::IoError;
use veldmap_core::tile::PlanetaryScheme;

pub mod band;
mod basemap;
mod cascade;
pub mod chunk_cache;
mod decode_pool;
pub mod grid;
mod map_tile;
mod mipmap;
pub mod pyramid;
mod radiometry;
pub mod resample;
pub mod tiff;

pub use band::{
    BandInfo, GeoKeyParams, GeoKeyValue, GeoPlacement, GeoRasterType, GeoTie, RasterDescription,
    SampleType,
};
pub use basemap::{BundledBasemap, BundledBasemapConfig};
pub use chunk_cache::{ChunkKey, DecodedChunk, SharedChunkCache};
pub use decode_pool::{
    DEFAULT_MAX_PENDING_DECODE_BYTES, DEFAULT_MAX_PENDING_DECODE_JOBS, DecodeJob, DecodeLane,
    DecodePool, DecodePoolConfig, DecodePoolHandle, TwoLaneDecodePool, TwoLaneDecodePoolConfig,
    TwoLaneDecodePoolHandle,
};
pub use map_tile::{decode_map_tile, decode_map_tile_with_mipmaps};
pub use mipmap::generate_linear_mipmaps;
pub use pyramid::RASTER_TILE_SIZE;
pub use tiff::{DecodeContext, TiffSource};

/// Errors returned by raster tile decoders and virtual tile providers.
#[derive(Debug, Error)]
pub enum RasterError {
    #[error("raster decode cancelled")]
    Cancelled,
    #[error("unsupported or unrecognized image format")]
    UnsupportedFormat,
    #[error("invalid raster format: {0}")]
    InvalidFormat(String),
    #[error("TIFF decode error: {0}")]
    Tiff(String),
    #[error("I/O error: {0}")]
    Io(#[from] IoError),
    #[error("memory budget exceeded: {0}")]
    Budget(#[from] BudgetError),
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
