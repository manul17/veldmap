//! Grouped public-API integration tests for `veldmap-net` (live basemap specs,
//! mock HTTP streaming pipeline, host concurrency limits, disk cache
//! integration, error handling, and cancellation).

#![expect(
    clippy::expect_used,
    reason = "INVARIANT: integration test harness helpers panic on setup failure"
)]

mod basemap;
mod fetcher;
