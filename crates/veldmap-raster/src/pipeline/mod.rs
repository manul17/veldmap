//! Format-agnostic analytical raster pyramid pipeline: band metadata, static
//! pyramid plans, chunk grid & execution driver, singleflight chunk cache,
//! cascade emission, resampling, radiometry, and `TileScheduler`.

use veldmap_core::budget::{DecodeBudgetTracker, MemoryBudget};
use veldmap_core::sync::CancelToken;

pub(crate) mod cascade;
pub(crate) mod chunk_cache;
pub(crate) mod driver;
pub(crate) mod grid;
pub(crate) mod metadata;
pub(crate) mod peak;
pub(crate) mod plan;
pub(crate) mod radiometry;
pub(crate) mod resample;
pub(crate) mod scheduler;
pub(crate) mod singleflight;
pub(crate) mod source;

pub use cascade::TileEmit;
pub use chunk_cache::SharedChunkCache;
pub use metadata::{
    BandInfo, GeoKeyParams, GeoKeyValue, GeoPlacement, GeoRasterType, GeoTie, RasterDescription,
    SampleType,
};
pub use plan::{LevelPlan, PassKey, PyramidMeta, ServeMode};
pub use scheduler::{PassBusy, TileFailureOutcome, TileScheduler};
pub use source::{RasterSource, frame_from_description};

/// Shared runtime context passed into analytical raster decoding calls.
pub struct DecodeContext<'a> {
    pub chunk_cache: &'a SharedChunkCache,
    pub budget_tracker: &'a DecodeBudgetTracker,
    pub budget: &'a MemoryBudget,
    pub cancel: &'a CancelToken,
}
