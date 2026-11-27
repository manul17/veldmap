//! Format-agnostic analytical raster pyramid pipeline: band metadata, static
//! pyramid plans, chunk grid & execution driver, singleflight chunk cache,
//! cascade emission, resampling, radiometry, and `TileScheduler`.

use veldmap_core::budget::{DecodeBudgetTracker, MemoryBudget};
use veldmap_core::sync::CancelToken;

mod binding;
mod cascade;
mod chunk_cache;
mod driver;
mod grid;
mod metadata;
mod peak;
mod plan;
mod radiometry;
mod resample;
mod scheduler;
mod singleflight;
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
pub use source::{
    ByteRange, DEFAULT_COALESCE_GAP_BYTES, DEFAULT_COALESCE_MAX_SPAN_BYTES, RasterSource,
    coalesce_byte_ranges,
};

pub(crate) use cascade::{Cascade, bleed_display_emit, bytes as cascade_bytes};
pub(crate) use chunk_cache::{ChunkKey, DecodedChunk};
pub(crate) use driver::{Chunked, DirectRead, alloc_empty_surface, direct, pass, plan_direct};
pub(crate) use grid::{Grid, Overview, admit_chunk_working_bytes, chunk_working_bytes};
pub(crate) use metadata::is_valid_lat_lon;
pub(crate) use plan::{build_chunked_level_table, check_max_source_side};
pub(crate) use radiometry::{
    DisplayMapping, PixelLayout, STRETCH_SAMPLES, compute_band_stats, expand_u8_into_rgba,
    is_data_f64, percentile_stretch, raw_bytes_to_rgba, read_native_sample_f64,
};
pub(crate) use resample::{
    DownsampleRounding, Window, downsample_r32f_2x, downsample_rgba8_2x, resample_rgba8_window,
};
pub(crate) use singleflight::SingleflightMap;

/// Shared runtime context passed into analytical raster decoding calls.
pub struct DecodeContext<'a> {
    pub chunk_cache: &'a SharedChunkCache,
    pub budget_tracker: &'a DecodeBudgetTracker,
    pub budget: &'a MemoryBudget,
    pub cancel: &'a CancelToken,
}
