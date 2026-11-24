//! Grouped public-API integration tests for `veldmap-raster` (GeoTIFF/COG
//! read amplification, BigTIFF/ZSTD/Predictor=3/GeoKey parsing, memory budget
//! enforcement, singleflight chunk deduplication, and two-lane decode pool).

#![expect(
    clippy::expect_used,
    reason = "INVARIANT: integration test harness helpers panic on setup failure"
)]

mod codecs_and_sidecars;
mod common;
mod stress_and_budget;
mod tiff_and_cog;
