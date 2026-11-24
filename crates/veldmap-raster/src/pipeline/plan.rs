//! Static pyramid serving plans (`ServeMode`, `LevelPlan`, `PassKey`,
//! `PyramidMeta`) and chunked level-table planning derived from [`Grid`] and
//! [`MemoryBudget`].

use std::collections::HashSet;

use veldmap_core::budget::MemoryBudget;
use veldmap_core::tile::{
    BandKey, PixelFormat, SourceId, StreamKey, TileKey, TileScheme, TileStreamSpec,
};

use super::grid::Grid;
use super::pyramid;

/// How one pyramid level is produced from its underlying source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ServeMode {
    /// Only chunks intersecting requested tiles are decoded.
    Pointwise,
    /// Sequential pass starting from level `from` producing `from..=max_level`
    /// together in one cascade pass.
    Pass { from: u8 },
}

/// Cost and execution plan for one pyramid level (`0` = finest native).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LevelPlan {
    pub serve: ServeMode,
    /// Estimated source pixels read for one tile (pointwise) or full pass.
    pub source_pixels: u64,
    /// Estimated working memory peak in bytes.
    pub estimated_peak_bytes: u64,
    /// Whether this level fits inside the configured memory budget.
    pub fits: bool,
}

/// Identifies one independent raster pyramid stream (`source`, `band`,
/// `format`, `gutter`) so multi-band sources sharing a single [`SourceId`]
/// schedule and forgive sequential passes independently per band.
pub type PassKey = StreamKey;

/// Scheduler-facing description of one raster pyramid.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PyramidMeta {
    pub source: SourceId,
    pub spec: TileStreamSpec,
    pub width: u32,
    pub height: u32,
    pub tile_size: u32,
    /// Level plans ordered from `0` (finest native) to `max_level` (coarsest
    /// single-tile apex).
    pub levels: Vec<LevelPlan>,
}

impl PyramidMeta {
    /// Return the [`PassKey`] identifying this pyramid stream.
    pub const fn pass_key(&self) -> PassKey {
        PassKey::new(
            self.source,
            self.spec.band,
            self.spec.format,
            self.spec.gutter,
        )
    }

    /// Coarsest apex level index (`levels.len().saturating_sub(1)`).
    pub fn max_level(&self) -> u8 {
        self.levels.len().saturating_sub(1).min(u8::MAX as usize) as u8
    }

    /// Corresponding [`TileScheme::RasterPyramid`].
    pub fn scheme(&self) -> TileScheme {
        TileScheme::RasterPyramid {
            max_level: self.max_level(),
        }
    }

    /// Finest level index whose `LevelPlan::fits` is `true`, or `None` when
    /// no level fits within the configured memory budget.
    pub fn finest_fitting_level(&self) -> Option<u8> {
        self.levels
            .iter()
            .position(|row| row.fits)
            .map(|idx| idx as u8)
    }

    /// Return the [`LevelPlan`] for `level`, if in range.
    pub fn level(&self, level: u8) -> Option<LevelPlan> {
        self.levels.get(level as usize).copied()
    }

    /// Return whether `level` is served pointwise.
    pub fn is_pointwise(&self, level: u8) -> bool {
        self.level(level)
            .is_some_and(|row| row.serve == ServeMode::Pointwise)
    }

    /// Return the pass start level if `level` is served by `ServeMode::Pass`.
    pub fn pass_from(&self, level: u8) -> Option<u8> {
        match self.level(level)?.serve {
            ServeMode::Pass { from } => Some(from),
            ServeMode::Pointwise => None,
        }
    }

    /// Progressive ladder of fitting levels from `max_level()` down to
    /// `target_level` (clamped to `[finest_fitting_level, max_level()]`),
    /// keeping at most one coarsest representative per distinct `ServeMode::Pass`
    /// cascade so redundant full-file passes are skipped.
    ///
    /// Returns an empty `Vec` when no level has `fits == true`.
    pub fn ladder(&self, target_level: u8) -> Vec<u8> {
        let Some(min_fit) = self.finest_fitting_level() else {
            return Vec::new();
        };
        let top = self.max_level();
        let target = target_level.clamp(min_fit, top);
        let target_plan = match self.level(target) {
            Some(plan) if plan.fits => Some(plan),
            _ => None,
        };
        let goal_pass = target_plan.and_then(|plan| match plan.serve {
            ServeMode::Pass { from } => Some(from),
            ServeMode::Pointwise => None,
        });

        let mut rungs = Vec::new();
        let mut seen_passes = HashSet::new();
        if target < top {
            for lvl_u16 in (u16::from(target) + 1..=u16::from(top)).rev() {
                let lvl = lvl_u16 as u8;
                let Some(plan) = self.level(lvl) else {
                    continue;
                };
                if !plan.fits {
                    continue;
                }
                match plan.serve {
                    ServeMode::Pointwise => rungs.push(lvl),
                    ServeMode::Pass { from } => {
                        if Some(from) != goal_pass && seen_passes.insert(from) {
                            rungs.push(lvl);
                        }
                    }
                }
            }
        }
        if target_plan.is_some() {
            rungs.push(target);
        }
        rungs
    }

    /// Construct the canonical [`TileKey`] for `(level, x, y)`.
    pub fn tile_key(&self, level: u8, x: u32, y: u32) -> TileKey {
        TileKey::from_stream(self.source, self.spec, self.scheme(), level, x, y)
    }

    /// Band selector of this pyramid stream.
    pub const fn band(&self) -> BandKey {
        self.spec.band
    }

    /// Pixel format of this pyramid stream.
    pub const fn format(&self) -> PixelFormat {
        self.spec.format
    }
}

/// Build the per-level serving plan (`0..level_count`) for a chunked raster.
pub(crate) fn build_chunked_level_table(
    grid: &Grid,
    width: u32,
    height: u32,
    budget: &MemoryBudget,
) -> Vec<LevelPlan> {
    let base = (width, height);
    let whole_pixels = u64::from(width) * u64::from(height);
    let count = pyramid::level_count(width, height).min(u32::from(u8::MAX));

    (0..count)
        .map(|lvl_u32| {
            let level = lvl_u32 as u8;
            match grid.footprint(base, level, budget) {
                Some(fp) => {
                    let peak = grid.direct_peak_for_footprint(&fp).total();
                    LevelPlan {
                        serve: ServeMode::Pointwise,
                        source_pixels: fp.held_pixels,
                        estimated_peak_bytes: peak,
                        fits: peak <= budget.decode_working_bytes,
                    }
                }
                None => {
                    let peak = grid.pass_peak(base).total();
                    LevelPlan {
                        serve: ServeMode::Pass { from: 0 },
                        source_pixels: whole_pixels,
                        estimated_peak_bytes: peak,
                        fits: peak <= budget.decode_working_bytes,
                    }
                }
            }
        })
        .collect()
}
