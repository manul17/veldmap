//! Format-agnostic analytical raster pyramid pipeline: band metadata, static
//! pyramid plans, chunk grid & execution driver, singleflight chunk cache,
//! cascade emission, resampling, radiometry, and `TileScheduler`.

use veldmap_core::budget::{DecodeBudgetTracker, MemoryBudget};
use veldmap_core::sync::CancelToken;

pub(crate) mod band;
pub(crate) mod cascade;
pub(crate) mod chunk_cache;
pub(crate) mod driver;
pub(crate) mod grid;
pub(crate) mod peak;
pub(crate) mod plan;
pub(crate) mod pyramid;
pub(crate) mod radiometry;
pub(crate) mod resample;
pub(crate) mod scheduler;
pub(crate) mod singleflight;

/// Shared runtime context passed into analytical raster decoding calls.
pub struct DecodeContext<'a> {
    pub chunk_cache: &'a chunk_cache::SharedChunkCache,
    pub budget_tracker: &'a DecodeBudgetTracker,
    pub budget: &'a MemoryBudget,
    pub cancel: &'a CancelToken,
}
