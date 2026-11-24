//! Raster overlay pyramid culling (`OverlayCullTree`) and CPU `f64` RTC
//! warp-mesh geometry (`WarpPatch`, `build_warp_patch`, `level_segments`).
//!
//! Lives in `veldmap-core::tile` alongside `quadtree` so `geo` stays free of
//! `tile` dependencies.

use std::collections::HashMap;

use glam::DVec3;

use crate::geo::binding::Frame;
use crate::geo::cull::{Ball, CullFrustum};
use crate::geo::geodesy::{self, WGS84};
use crate::geo::sag;

use super::pyramid::{RASTER_TILE_SIZE, cell_image_rect, grid_dim, level_size};

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

/// Oblate polar-to-equatorial radius-of-curvature ratio `a / (b^2/a)` scaling
/// spherical sagitta formulas conservatively for WGS84.
const POLAR_CURVATURE_SCALE: f64 = 1.0 / (1.0 - geodesy::FLATTENING);

/// Probe grid side used to construct conservative [`Ball`] culling spheres.
///
/// A `5 x 5` lattice (25 probes) keeps high-latitude parallel-circle sagitta
/// strictly bounded with `< 0.5%` radius bloat on regional tiles.
const CULL_PROBE_SIDE: usize = 5;

/// Upper bound on lazily cached culling spheres per [`OverlayCullTree`] to
/// keep CPU memory bounded across long panning sessions on gigapixel rasters.
const MAX_CACHED_BALLS: usize = 8_192;

/// Compute a uniform warp-mesh segment count per cell side for pyramid `level`
/// of a `(width, height)` raster bound by `frame`.
///
/// Using one uniform segment count across every cell at the same pyramid level
/// eliminates T-junction seam cracks along shared boundaries (including
/// clipped right/bottom edge cells).
pub fn level_segments(frame: &Frame, width: u32, height: u32, level: u8) -> u32 {
    let w = f64::from(width.max(1));
    let h = f64::from(height.max(1));
    let step_px = f64::from(RASTER_TILE_SIZE)
        * f64::from(1u32.checked_shl(u32::from(level)).unwrap_or(u32::MAX));
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
        let grid_span_deg = (grid_span_m / geodesy::SEMI_MAJOR_M).to_degrees();
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

/// Top-down lazy cache of `f64` ECEF [`Ball`] culling spheres per pyramid rung.
///
/// Computing conservative `5 x 5`-probe balls lazily from `max_level`
/// (coarsest) down to `target_level` avoids eagerly allocating tens of
/// thousands of level-0 spheres when opening large orthophotos while delegating
/// visibility entirely to [`CullFrustum::admits`] (exact oblate WGS84 horizon
/// plane + near plane + 4 frustum side planes).
#[derive(Clone, Debug)]
pub struct OverlayCullTree {
    width: u32,
    height: u32,
    max_level: u8,
    balls: Vec<HashMap<(u32, u32), Ball>>,
    cached_total: usize,
}

impl OverlayCullTree {
    /// Create an empty lazy culling tree for a raster of `(width, height)`
    /// pixels with coarsest level `max_level`.
    pub fn new(width: u32, height: u32, max_level: u8) -> Self {
        let levels = usize::from(max_level) + 1;
        Self {
            width: width.max(1),
            height: height.max(1),
            max_level,
            balls: vec![HashMap::new(); levels],
            cached_total: 0,
        }
    }

    /// Clear all cached balls (e.g. after re-laying the georeference binding).
    pub fn clear(&mut self) {
        for map in &mut self.balls {
            map.clear();
        }
        self.cached_total = 0;
    }

    /// Total number of balls currently materialized in the lazy cache.
    pub fn cached_ball_count(&self) -> usize {
        self.cached_total
    }

    /// Return (computing and caching on demand) the conservative `f64` ECEF
    /// bounding [`Ball`] for cell `(level, x, y)`.
    pub fn ball(&mut self, frame: &Frame, level: u8, x: u32, y: u32) -> Ball {
        let clamped = level.min(self.max_level);
        let lvl_idx = usize::from(clamped);
        if let Some(&cached) = self.balls[lvl_idx].get(&(x, y)) {
            return cached;
        }
        let computed = compute_cell_ball(frame, self.width, self.height, u32::from(clamped), x, y);
        if self.cached_total >= MAX_CACHED_BALLS {
            self.evict_finest_non_empty(lvl_idx);
        }
        if self.balls[lvl_idx].insert((x, y), computed).is_none() {
            self.cached_total += 1;
        }
        computed
    }

    fn evict_finest_non_empty(&mut self, keep_level: usize) {
        for idx in 0..self.balls.len() {
            if idx != keep_level && !self.balls[idx].is_empty() {
                self.cached_total = self.cached_total.saturating_sub(self.balls[idx].len());
                self.balls[idx].clear();
                return;
            }
        }
        // Fallback: all cached balls live on `keep_level`.
        self.cached_total = self
            .cached_total
            .saturating_sub(self.balls[keep_level].len());
        self.balls[keep_level].clear();
    }

    /// Collect all visible `(x, y)` cells at `target_level` using top-down lazy
    /// [`Ball`] pruning against `frustum`.
    pub fn visible_cells(
        &mut self,
        frame: &Frame,
        target_level: u8,
        frustum: &CullFrustum,
    ) -> Vec<(u32, u32)> {
        if !frame.measurable() {
            return Vec::new();
        }
        let target = target_level.min(self.max_level);
        let top = self.max_level;
        let top_w = grid_dim(level_size(self.width, u32::from(top)));
        let top_h = grid_dim(level_size(self.height, u32::from(top)));

        let mut stack: Vec<(u8, u32, u32)> = Vec::with_capacity((top_w * top_h) as usize);
        for y in (0..top_h).rev() {
            for x in (0..top_w).rev() {
                stack.push((top, x, y));
            }
        }

        let mut visible = Vec::new();
        while let Some((level, x, y)) = stack.pop() {
            let ball = self.ball(frame, level, x, y);
            if !frustum.admits(&ball) {
                continue;
            }
            if level == target {
                visible.push((x, y));
                continue;
            }

            let child_level = level - 1;
            let child_w = grid_dim(level_size(self.width, u32::from(child_level)));
            let child_h = grid_dim(level_size(self.height, u32::from(child_level)));
            let base_x = x * 2;
            let base_y = y * 2;
            for dy in (0..2u32).rev() {
                let cy = base_y + dy;
                if cy >= child_h {
                    continue;
                }
                for dx in (0..2u32).rev() {
                    let cx = base_x + dx;
                    if cx >= child_w {
                        continue;
                    }
                    stack.push((child_level, cx, cy));
                }
            }
        }
        visible
    }
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
            DVec3::new(f64::from(p[0]), f64::from(p[1]), f64::from(p[2]))
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
    let w = f64::from(width.max(1));
    let h = f64::from(height.max(1));
    let side = (segs + 1) as usize;
    let total = side * side;
    let mut ecef_points = Vec::with_capacity(total);
    let mut ecef_normals = Vec::with_capacity(total);

    let coord = |c0: f64, c1: f64, step: u32| {
        if step == 0 {
            c0
        } else if step == segs {
            c1
        } else {
            c0 + (c1 - c0) * (f64::from(step) / f64::from(segs))
        }
    };

    if frame.axial() {
        let lats: Vec<f64> = (0..=segs)
            .map(|row| {
                frame
                    .geodetic(cell_rect[0] / w, coord(cell_rect[1], cell_rect[3], row) / h)
                    .0
            })
            .collect();
        let lons: Vec<f64> = (0..=segs)
            .map(|col| {
                frame
                    .geodetic(coord(cell_rect[0], cell_rect[2], col) / w, cell_rect[1] / h)
                    .1
            })
            .collect();
        WGS84.fill_separable_surface_points_and_normals(
            &lats,
            &lons,
            &mut ecef_points,
            &mut ecef_normals,
        );
    } else {
        for row in 0..=segs {
            let py = coord(cell_rect[1], cell_rect[3], row) / h;
            for col in 0..=segs {
                let px = coord(cell_rect[0], cell_rect[2], col) / w;
                let (lat, lon) = frame.geodetic(px, py);
                let (p, n) = WGS84.surface_ecef_and_normal(lat, lon);
                ecef_points.push(p);
                ecef_normals.push(n);
            }
        }
    }

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

fn compute_cell_ball(frame: &Frame, width: u32, height: u32, level: u32, x: u32, y: u32) -> Ball {
    const TOTAL: usize = CULL_PROBE_SIDE * CULL_PROBE_SIDE;
    let cell = cell_image_rect(x, y, level, width, height);
    let w = f64::from(width.max(1));
    let h = f64::from(height.max(1));
    let denom = (CULL_PROBE_SIDE - 1) as f64;
    let along = |from: f64, to: f64, step: usize| {
        if step == 0 {
            from
        } else if step + 1 == CULL_PROBE_SIDE {
            to
        } else {
            from + (to - from) * (step as f64 / denom)
        }
    };

    let mut seen = [(0.0_f64, 0.0_f64); TOTAL];
    let mut points = [DVec3::ZERO; TOTAL];
    if frame.axial() {
        let mut lats = [0.0_f64; CULL_PROBE_SIDE];
        let mut lons = [0.0_f64; CULL_PROBE_SIDE];
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
        for row in 0..CULL_PROBE_SIDE {
            for col in 0..CULL_PROBE_SIDE {
                seen[row * CULL_PROBE_SIDE + col] = (lats[row], lons[col]);
            }
        }
        let mut normals_scratch = [DVec3::ZERO; TOTAL];
        WGS84.fill_separable_surface_grid(lats, lons, &mut points, &mut normals_scratch);
    } else {
        for row in 0..CULL_PROBE_SIDE {
            let py = along(cell[1], cell[3], row) / h;
            for col in 0..CULL_PROBE_SIDE {
                let px = along(cell[0], cell[2], col) / w;
                let (lat, lon) = frame.geodetic(px, py);
                let idx = row * CULL_PROBE_SIDE + col;
                seen[idx] = (lat, lon);
                points[idx] = WGS84.surface_ecef_and_normal(lat, lon).0;
            }
        }
    }

    let mut max_subcell_diag_deg = 0.0_f64;
    let mut par_sag_m = 0.0_f64;
    let check_parallel_step = |p0: (f64, f64), p1: (f64, f64), acc: &mut f64| {
        let dlon = (geodesy::unwind(p0.1, p1.1) - p0.1).abs().min(180.0);
        let max_lat = p0.0.abs().max(p1.0.abs()).min(90.0);
        let r_par = geodesy::SEMI_MAJOR_M * max_lat.to_radians().cos();
        let sag = r_par * (1.0 - (dlon * 0.5).to_radians().cos());
        *acc = (*acc).max(sag);
    };

    for row in 0..CULL_PROBE_SIDE {
        for col in 0..CULL_PROBE_SIDE {
            let idx = row * CULL_PROBE_SIDE + col;
            if col + 1 < CULL_PROBE_SIDE {
                check_parallel_step(seen[idx], seen[idx + 1], &mut par_sag_m);
            }
            if row + 1 < CULL_PROBE_SIDE {
                check_parallel_step(seen[idx], seen[idx + CULL_PROBE_SIDE], &mut par_sag_m);
            }
            if row + 1 < CULL_PROBE_SIDE && col + 1 < CULL_PROBE_SIDE {
                let d0 = geodesy::separation(seen[idx], seen[idx + CULL_PROBE_SIDE + 1]);
                let d1 = geodesy::separation(seen[idx + 1], seen[idx + CULL_PROBE_SIDE]);
                max_subcell_diag_deg = max_subcell_diag_deg.max(d0).max(d1);
            }
        }
    }

    let bulge_m = sag::sag_m((max_subcell_diag_deg * 1.15).min(180.0)) * POLAR_CURVATURE_SCALE;
    Ball::over(&points).widened(bulge_m + par_sag_m)
}

fn cell_span_deg(frame: &Frame, width: f64, height: f64, cell: [f64; 4]) -> f64 {
    let at = |x: f64, y: f64| frame.geodetic(x / width, y / height);
    let xs = [cell[0], (cell[0] + cell[2]) * 0.5, cell[2]];
    let ys = [cell[1], (cell[1] + cell[3]) * 0.5, cell[3]];
    let mut probes = [(0.0_f64, 0.0_f64); 9];
    for row in 0..3 {
        for col in 0..3 {
            probes[row * 3 + col] = at(xs[col], ys[row]);
        }
    }
    let mut span = 0.0_f64;
    for row in 0..3 {
        let a = probes[row * 3];
        let b = probes[row * 3 + 1];
        let c = probes[row * 3 + 2];
        span = span.max(geodesy::separation(a, b) + geodesy::separation(b, c));
    }
    for col in 0..3 {
        let a = probes[col];
        let b = probes[3 + col];
        let c = probes[6 + col];
        span = span.max(geodesy::separation(a, b) + geodesy::separation(b, c));
    }
    span
}

#[cfg(test)]
#[path = "overlay_tests.rs"]
mod tests;
