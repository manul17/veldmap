//! Lazy top-down pyramid culling (`OverlayCullTree`) with conservative `5 x 5`
//! WGS84 surface-probe bounding spheres.

use std::collections::HashMap;

use glam::DVec3;

use crate::geo::{Ball, CullFrustum, Frame, sag};

use super::super::pyramid::{cell_image_rect, grid_dim, level_size};
use super::grid::{FrameGridSpec, sample_frame_grid};

/// Probe grid side used to construct conservative [`Ball`] culling spheres.
///
/// A `5 x 5` lattice (25 probes) keeps high-latitude parallel-circle sagitta
/// strictly bounded with `< 0.5%` radius bloat on regional tiles.
const CULL_PROBE_SIDE: usize = 5;

/// Upper bound on lazily cached culling spheres per [`OverlayCullTree`] to
/// keep CPU memory bounded across long panning sessions on gigapixel rasters.
const MAX_CACHED_BALLS: usize = 8_192;

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

    /// Total number of balls currently materialized in the lazy cache.
    #[cfg(test)]
    pub(super) fn cached_ball_count(&self) -> usize {
        self.cached_total
    }

    /// Return (computing and caching on demand) the conservative `f64` ECEF
    /// bounding [`Ball`] for cell `(level, x, y)`.
    fn ball(&mut self, frame: &Frame, level: u8, x: u32, y: u32) -> Ball {
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

    /// Return whether any coarsest-level cell of this overlay intersects `frustum`.
    pub fn intersects_frustum(&mut self, frame: &Frame, frustum: &CullFrustum) -> bool {
        if !frame.measurable() {
            return false;
        }
        let top = self.max_level;
        let top_w = grid_dim(level_size(self.width, u32::from(top)));
        let top_h = grid_dim(level_size(self.height, u32::from(top)));
        for y in 0..top_h {
            for x in 0..top_w {
                let ball = self.ball(frame, top, x, y);
                if frustum.admits(&ball) {
                    return true;
                }
            }
        }
        false
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

pub(super) fn compute_cell_ball(
    frame: &Frame,
    width: u32,
    height: u32,
    level: u32,
    x: u32,
    y: u32,
) -> Ball {
    const TOTAL: usize = CULL_PROBE_SIDE * CULL_PROBE_SIDE;
    let cell = cell_image_rect(x, y, level, width, height);

    let mut seen = [(0.0_f64, 0.0_f64); TOTAL];
    let mut points = [DVec3::ZERO; TOTAL];
    let mut normals = [DVec3::ZERO; TOTAL];
    sample_frame_grid(
        frame,
        FrameGridSpec {
            width,
            height,
            cell,
            side: CULL_PROBE_SIDE,
        },
        &mut points,
        &mut normals,
        Some(&mut seen),
    );

    let max_subcell_diag_deg = sag::max_subcell_arc_deg(&normals, CULL_PROBE_SIDE);
    let par_sag_m = sag::max_parallel_sag_m(&seen, CULL_PROBE_SIDE);

    let bulge_m = sag::subcell_bulge_m(max_subcell_diag_deg);
    Ball::over(&points).widened(bulge_m + par_sag_m)
}
