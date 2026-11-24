//! Native raster decoding, bounded CPU decode pools, linear-light mipmap
//! generation, virtual offline equirectangular basemap tiling, and analytical
//! GeoTIFF/BigTIFF/COG pyramid decoding for VeldMap.
//!
//! Keeps `veldmap-core` permanently free of image/codec dependencies and keeps
//! `veldmap-net` decoupled from raster codecs via an injected decode executor.

use std::sync::Arc;

use thiserror::Error;
use veldmap_core::budget::BudgetError;
use veldmap_core::io::IoError;
use veldmap_core::tile::TileScheme;

mod color;
mod display;
mod pipeline;
mod pool;
mod tiff;

pub use display::{
    BundledBasemap, BundledBasemapConfig, decode_map_tile, decode_map_tile_with_mipmaps,
    generate_linear_mipmaps,
};
pub use pipeline::DecodeContext;
pub use pipeline::band::{
    BandInfo, GeoKeyParams, GeoKeyValue, GeoPlacement, GeoRasterType, GeoTie, RasterDescription,
    SampleType,
};
pub use pipeline::cascade::TileEmit;
pub use pipeline::chunk_cache::{ChunkKey, DecodedChunk, SharedChunkCache};
pub use pipeline::peak::DecodePeak;
pub use pipeline::plan::{LevelPlan, PassKey, PyramidMeta, ServeMode};
pub use pipeline::pyramid::{
    RASTER_TILE_SIZE, cell_image_rect, cell_uv, grid_dim, level_count, level_size, tile_extent,
};
pub use pipeline::scheduler::{PassBusy, TileFailureOutcome, TileScheduler};
pub use pool::{
    DEFAULT_MAX_PENDING_DECODE_BYTES, DEFAULT_MAX_PENDING_DECODE_JOBS, DecodeJob, DecodeJobKey,
    DecodeLane, DecodePool, DecodePoolConfig, DecodePoolHandle, TwoLaneDecodePool,
    TwoLaneDecodePoolConfig, TwoLaneDecodePoolHandle,
};
pub use tiff::{MAX_IFDS, MAX_SOURCE_SIDE, TIFF_DECODE_REV, TiffSource, has_tiff_magic};

/// Errors returned by raster tile decoders and virtual tile providers.
///
/// Cancellation from `IoError::Cancelled` or `BudgetError::Cancelled` is
/// normalized into [`Self::Cancelled`] via explicit `From` implementations so
/// schedulers and caches never have to match three cancellation shapes.
#[derive(Clone, Debug, Error)]
pub enum RasterError {
    #[error("raster decode cancelled")]
    Cancelled,
    #[error("unsupported or unrecognized image format")]
    UnsupportedFormat,
    #[error("invalid raster format: {0}")]
    InvalidFormat(String),
    #[error("TIFF decode error: {0}")]
    Tiff(String),
    #[error("codec decoder limits exceeded: {0}")]
    DecoderLimitsExceeded(String),
    #[error("raster decoder panicked: {0}")]
    DecoderPanicked(String),
    #[error("I/O error: {0}")]
    Io(#[source] Arc<IoError>),
    #[error("memory budget exceeded: {0}")]
    Budget(BudgetError),
    #[error("image decode failed: {0}")]
    Decode(String),
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
        scheme: TileScheme,
        level: u8,
        x: u32,
        y: u32,
    },
}

impl RasterError {
    /// Return whether this error represents task cancellation.
    pub const fn is_cancelled(&self) -> bool {
        matches!(self, Self::Cancelled)
    }
}

impl From<IoError> for RasterError {
    fn from(err: IoError) -> Self {
        if err.is_cancelled() {
            Self::Cancelled
        } else {
            Self::Io(Arc::new(err))
        }
    }
}

impl From<BudgetError> for RasterError {
    fn from(err: BudgetError) -> Self {
        match err {
            BudgetError::Cancelled => Self::Cancelled,
            other => Self::Budget(other),
        }
    }
}
