//! Raster overlay pyramid culling (`OverlayCullTree`) and CPU `f64` RTC
//! warp-mesh geometry (`WarpPatch`, `build_warp_patch`, `level_segments`).
//!
//! Lives in `veldmap-core::tile` alongside `quadtree` so `geo` stays free of
//! `tile` dependencies.

mod cull;
mod warp;

use glam::DVec3;

use crate::geo::{Frame, WGS84};

pub use cull::OverlayCullTree;
pub use warp::{MAX_WARP_SEGMENTS, MIN_WARP_SEGMENTS, WarpPatch, build_warp_patch, level_segments};

#[cfg(test)]
use cull::compute_cell_ball;

/// Sampling grid specification for a raster cell rectangle inside `(width, height)`.
pub(super) struct FrameGridSpec {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) cell: [f64; 4],
    pub(super) side: usize,
}

/// Sample a `side x side` endpoint-preserving WGS84 surface grid across pixel
/// rectangle `spec.cell = [x0, y0, x1, y1]` inside a `(width, height)` raster
/// bound by `frame`, writing row-major ECEF positions and outward unit normals
/// (and optionally raw `(lat_deg, lon_deg)` pairs).
///
/// Uses the `O(side)` separable trig fast path whenever `frame.axial()` holds.
pub(super) fn sample_frame_grid(
    frame: &Frame,
    spec: FrameGridSpec,
    points: &mut [DVec3],
    normals: &mut [DVec3],
    mut geodetic_out: Option<&mut [(f64, f64)]>,
) {
    let FrameGridSpec {
        width,
        height,
        cell,
        side,
    } = spec;
    debug_assert!(side >= 2);
    debug_assert!(points.len() >= side * side);
    debug_assert!(normals.len() >= side * side);

    let w = f64::from(width.max(1));
    let h = f64::from(height.max(1));
    let denom = (side - 1) as f64;
    let along = |from: f64, to: f64, step: usize| {
        if step == 0 {
            from
        } else if step + 1 == side {
            to
        } else {
            from + (to - from) * (step as f64 / denom)
        }
    };

    if frame.axial() {
        const MAX_STACK_SIDE: usize = MAX_WARP_SEGMENTS as usize + 1;
        debug_assert!(side <= MAX_STACK_SIDE);
        let mut lats_buf = [0.0_f64; MAX_STACK_SIDE];
        let mut lons_buf = [0.0_f64; MAX_STACK_SIDE];
        let lats = &mut lats_buf[..side];
        let lons = &mut lons_buf[..side];
        for (row, lat) in lats.iter_mut().enumerate() {
            *lat = frame
                .geodetic(cell[0] / w, along(cell[1], cell[3], row) / h)
                .0;
        }
        for (col, lon) in lons.iter_mut().enumerate() {
            *lon = frame
                .geodetic(along(cell[0], cell[2], col) / w, cell[1] / h)
                .1;
        }
        if let Some(out) = geodetic_out.as_deref_mut() {
            for (row, &lat) in lats.iter().enumerate() {
                for (col, &lon) in lons.iter().enumerate() {
                    out[row * side + col] = (lat, lon);
                }
            }
        }
        WGS84.fill_separable_surface_grid(lats, lons, points, normals);
    } else {
        for row in 0..side {
            let py = along(cell[1], cell[3], row) / h;
            for col in 0..side {
                let px = along(cell[0], cell[2], col) / w;
                let (lat, lon) = frame.geodetic(px, py);
                let idx = row * side + col;
                if let Some(out) = geodetic_out.as_deref_mut() {
                    out[idx] = (lat, lon);
                }
                let (p, n) = WGS84.surface_ecef_and_normal(lat, lon);
                points[idx] = p;
                normals[idx] = n;
            }
        }
    }
}

#[cfg(test)]
#[path = "overlay_tests.rs"]
mod tests;
