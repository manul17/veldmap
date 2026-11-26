//! Format-agnostic analytical raster pyramid pipeline: band metadata, static
//! pyramid plans, chunk grid & execution driver, singleflight chunk cache,
//! cascade emission, resampling, radiometry, and `TileScheduler`.

use veldmap_core::budget::{DecodeBudgetTracker, MemoryBudget};
use veldmap_core::sync::CancelToken;

mod binding;
pub(crate) mod cascade;
pub(crate) mod chunk_cache;
pub(crate) mod driver;
pub(crate) mod grid;
pub(crate) mod metadata;
pub(crate) mod peak;
pub(crate) mod plan;
pub(crate) mod radiometry;
pub(crate) mod resample;
mod scheduler;
pub(crate) mod singleflight;
mod source;

pub use binding::{frame_from_description, frame_from_description_with_override};
pub use cascade::TileEmit;
pub use chunk_cache::SharedChunkCache;
pub use metadata::{
    BandInfo, BandSample, BandStats, GeoPlacement, GeoTie, HISTOGRAM_BINS, PlacementCrs,
    RasterDescription, SampleType,
};
pub use plan::{
    DISPLAY_STREAM_SPEC, LevelPlan, MAX_SOURCE_SIDE, PyramidMeta, ServeMode, stream_spec_for_band,
};
pub use scheduler::{AdmittedPass, AdmittedTile, TileFailureOutcome, TileScheduler};
pub use source::{ByteRange, RasterSource, coalesce_byte_ranges};

/// Shared runtime context passed into analytical raster decoding calls.
pub struct DecodeContext<'a> {
    pub chunk_cache: &'a SharedChunkCache,
    pub budget_tracker: &'a DecodeBudgetTracker,
    pub budget: &'a MemoryBudget,
    pub cancel: &'a CancelToken,
}
