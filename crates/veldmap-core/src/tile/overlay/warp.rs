//! CPU `f64` RTC warp-mesh generation (`WarpPatch`, `build_warp_patch`,
//! `level_segments`) for native-CRS raster overlays draped on WGS84.

use glam::{DVec3, Vec3};

use crate::geo::{Frame, SEMI_MAJOR_M};

use super::super::pyramid::{RASTER_TILE_SIZE, level_px};
use super::{FrameGridSpec, sample_frame_grid};

/// Minimum segments per warp-mesh cell side.
///
/// Using `16` segments per side keeps nonlinear transverse-Mercator curvature
/// and chord placement error strictly `< 2 cm` on UTM-scale pyramid cells
/// (`~110 km`) near zone edges (`±3 deg` from the central meridian).
pub const MIN_WARP_SEGMENTS: u32 = 16;

/// Upper bound on segments per warp-mesh cell side (covers global cells up to
/// `360 deg` without exploding vertex counts).
pub const MAX_WARP_SEGMENTS: u32 = 64;

/// Target maximum surface arc per warp-mesh segment in degrees before clamping
/// to `[MIN_WARP_SEGMENTS, MAX_WARP_SEGMENTS]`.
const TARGET_WARP_SEGMENT_DEG: f64 = 0.5;

/// Compute a uniform warp-mesh segment count per cell side for pyramid `level`
/// of a `(width, height)` raster bound by `frame`.
///
/// Using one uniform segment count across every cell at the same pyramid level
/// eliminates T-junction seam cracks along shared boundaries (including
/// clipped right/bottom edge cells).
pub fn level_segments(frame: &Frame, width: u32, height: u32, level: u8) -> u32 {
    let w = f64::from(width.max(1));
    let h = f64::from(height.max(1));
    let step_px = f64::from(RASTER_TILE_SIZE) * level_px(u32::from(level));
    let tile_w = step_px.min(w);
    let tile_h = step_px.min(h);
    let max_x0 = (w - tile_w).max(0.0);
    let max_y0 = (h - tile_h).max(0.0);

    // Probe a 3x3 grid of tile-sized windows across the raster so equatorial
    // or mid-swath distortion governs even if cell (0, 0) lies near a pole.
    let mut worst_span_deg = 0.0_f64;
    for ry in [0.0, 0.5, 1.0] {
        let y0 = max_y0 * ry;
        let y1 = (y0 + tile_h).min(h);
        for rx in [0.0, 0.5, 1.0] {
            let x0 = max_x0 * rx;
            let x1 = (x0 + tile_w).min(w);
            let span = cell_span_deg(frame, w, h, [x0, y0, x1, y1]);
            worst_span_deg = worst_span_deg.max(span);
        }
    }

    if let Some((widest_row_m, widest_col_m)) = frame.widest_grid_spans_m() {
        let frac_x = (tile_w / w).clamp(0.0, 1.0);
        let frac_y = (tile_h / h).clamp(0.0, 1.0);
        let grid_span_m = (widest_row_m * frac_x).max(widest_col_m * frac_y);
        let grid_span_deg = (grid_span_m / SEMI_MAJOR_M).to_degrees();
        worst_span_deg = worst_span_deg.max(grid_span_deg);
    }

    segments_for(worst_span_deg)
}

fn segments_for(span_deg: f64) -> u32 {
    if !span_deg.is_finite() || span_deg <= 0.0 {
        return MIN_WARP_SEGMENTS;
    }
    let raw = (span_deg / TARGET_WARP_SEGMENT_DEG).ceil();
    if !raw.is_finite() {
        return MAX_WARP_SEGMENTS;
    }
    (raw as u32).clamp(MIN_WARP_SEGMENTS, MAX_WARP_SEGMENTS)
}

fn cell_span_deg(frame: &Frame, width: f64, height: f64, cell: [f64; 4]) -> f64 {
    let (span_x, span_y) = frame.probe_rect_spans_deg(
        cell[0] / width,
        cell[2] / width,
        cell[1] / height,
        cell[3] / height,
    );
    span_x.max(span_y)
}

/// CPU `f64` RTC warp-mesh patch for a single raster pyramid cell.
#[derive(Clone, Debug)]
pub struct WarpPatch {
    /// Cell-center `f64` ECEF origin in metres.
    pub origin_ecef_m: DVec3,
    /// Segments per cell side (`[1, MAX_WARP_SEGMENTS]`).
    pub segments: u32,
    /// `true` when the binding has negative orientation (e.g. south-up or
    /// mirrored affine) so triangle indices must flip to keep CCW outward
    /// normals facing away from the WGS84 ellipsoid.
    pub flip_winding: bool,
    /// Row-major `(segments + 1)^2` surface positions relative to
    /// `origin_ecef_m` (`(P_ecef_m - origin_ecef_m) as f32`).
    pub rel_positions: Vec<[f32; 3]>,
    /// Row-major `(segments + 1)^2` outward unit WGS84 surface normals for
    /// sag-safe fragment horizon facing checks.
    pub normals: Vec<[f32; 3]>,
}

impl WarpPatch {
    /// Evaluate the exact triangulated `f64` ECEF position produced by this
    /// patch (including `f32` RTC vertex quantization) at cell-local
    /// `(tx, ty) in [0, 1]^2`, matching the GPU quad triangulation
    /// `(top, bottom, top + 1)` and `(top + 1, bottom, bottom + 1)`.
    ///
    /// Returns `None` if `(tx, ty)` is non-finite or the vertex buffer is
    /// malformed.
    pub fn evaluate_ecef(&self, tx: f64, ty: f64) -> Option<DVec3> {
        if !(tx.is_finite() && ty.is_finite()) {
            return None;
        }
        let segs = self.segments.max(1);
        let side = (segs + 1) as usize;
        if self.rel_positions.len() < side.checked_mul(side)? {
            return None;
        }
        let sx = tx.clamp(0.0, 1.0) * f64::from(segs);
        let sy = ty.clamp(0.0, 1.0) * f64::from(segs);
        let col = (sx.floor() as usize).min((segs - 1) as usize);
        let row = (sy.floor() as usize).min((segs - 1) as usize);
        let u = sx - col as f64;
        let v = sy - row as f64;

        let v_at = |r: usize, c: usize| {
            let p = self.rel_positions[r * side + c];
            Vec3::from_array(p).as_dvec3()
        };
        let p00 = v_at(row, col);
        let p10 = v_at(row, col + 1);
        let p01 = v_at(row + 1, col);
        let p11 = v_at(row + 1, col + 1);

        let rel = if u + v <= 1.0 {
            p00 * (1.0 - u - v) + p01 * v + p10 * u
        } else {
            p10 * (1.0 - v) + p01 * (1.0 - u) + p11 * (u + v - 1.0)
        };
        Some(self.origin_ecef_m + rel)
    }
}

/// Build the per-tile CPU `f64` RTC warp-mesh patch for a raster cell covering
/// pixel rectangle `cell_rect = [x0, y0, x1, y1]` inside a `(width, height)`
/// raster using `segments` subdivisions per side (clamped to
/// `[1, MAX_WARP_SEGMENTS]`).
pub fn build_warp_patch(
    frame: &Frame,
    width: u32,
    height: u32,
    cell_rect: [f64; 4],
    segments: u32,
) -> WarpPatch {
    let segs = segments.clamp(1, MAX_WARP_SEGMENTS);
    let side = (segs + 1) as usize;
    let total = side * side;
    let mut ecef_points = vec![DVec3::ZERO; total];
    let mut ecef_normals = vec![DVec3::ZERO; total];
    sample_frame_grid(
        frame,
        FrameGridSpec {
            width,
            height,
            cell: cell_rect,
            side,
        },
        &mut ecef_points,
        &mut ecef_normals,
        None,
    );

    let mid = (segs / 2) as usize;
    let origin_ecef_m = ecef_points[mid * side + mid];
    let flip_winding = needs_flipped_winding(&ecef_points, side);

    let rel_positions = ecef_points
        .iter()
        .map(|&p| (p - origin_ecef_m).as_vec3().to_array())
        .collect();
    let normals = ecef_normals
        .iter()
        .map(|&n| n.as_vec3().to_array())
        .collect();

    WarpPatch {
        origin_ecef_m,
        segments: segs,
        flip_winding,
        rel_positions,
        normals,
    }
}

/// Check whether standard CCW quad winding `(top, bottom, top + 1)` points
/// inward relative to the outward WGS84 ellipsoid direction and therefore
/// requires flipped triangle winding.
fn needs_flipped_winding(ecef_points: &[DVec3], side: usize) -> bool {
    if side < 2 {
        return false;
    }
    let mut signed_outward = 0.0_f64;
    for row in 0..side - 1 {
        for col in 0..side - 1 {
            let top = ecef_points[row * side + col];
            let right = ecef_points[row * side + col + 1];
            let bottom = ecef_points[(row + 1) * side + col];
            let tri_normal = (bottom - top).cross(right - top);
            signed_outward += tri_normal.dot(top);
        }
    }
    signed_outward < 0.0
}
