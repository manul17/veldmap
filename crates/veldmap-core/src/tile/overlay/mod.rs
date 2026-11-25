//! Raster overlay pyramid culling (`OverlayCullTree`) and CPU `f64` RTC
//! warp-mesh geometry (`WarpPatch`, `build_warp_patch`, `level_segments`).
//!
//! Lives in `veldmap-core::tile` alongside `quadtree` so `geo` stays free of
//! `tile` dependencies.

mod cull;
mod grid;
mod warp;

pub use cull::OverlayCullTree;
pub use grid::MAX_WARP_SEGMENTS;
pub use warp::{MIN_WARP_SEGMENTS, WarpPatch, build_warp_patch, level_segments};

#[cfg(test)]
use cull::compute_cell_ball;

#[cfg(test)]
#[path = "overlay_tests.rs"]
mod tests;
