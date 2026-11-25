//! Static pyramid serving plans (`ServeMode`, `LevelPlan`, `PyramidMeta`)
//! and chunked level-table planning derived from [`Grid`] and [`MemoryBudget`].

use std::collections::HashSet;

use veldmap_core::budget::MemoryBudget;
use veldmap_core::tile::{
    BandKey, PixelFormat, SourceId, StreamKey, TileKey, TileScheme, TileStreamSpec, level_count,
};

use crate::RasterError;

use super::grid::Grid;

/// Maximum raster side dimension accepted for untiled/stripped rasters and
/// full-frame PNG sources.
///
/// Tiled TIFFs/COGs may exceed this limit provided their chunk grid and at
/// least one pyramid level fit within [`MemoryBudget`].
pub const MAX_SOURCE_SIDE: u32 = 65_536;

/// Current display decode revision encoded into analytical display `TileKey`s.
const DISPLAY_DECODE_REV: u32 = 2;

/// Current scientific/categorical decode revision encoded into `TileKey`s.
const TRACK_DECODE_REV: u32 = 2;

/// Canonical display stream spec shared across analytical display decoders
/// (`TiffSource`, `PngSource`, `JpegSource`).
pub const DISPLAY_STREAM_SPEC: TileStreamSpec = TileStreamSpec::new(
    BandKey::Display,
    PixelFormat::Rgba8UnormSrgb,
    0,
    DISPLAY_DECODE_REV,
);

/// Construct the canonical [`TileStreamSpec`] for `(band, format)` (`gutter = 0`).
pub const fn stream_spec_for_band(band: BandKey, format: PixelFormat) -> TileStreamSpec {
    let rev = match format {
        PixelFormat::Rgba8UnormSrgb => DISPLAY_DECODE_REV,
        PixelFormat::R32Float | PixelFormat::R8Uint | PixelFormat::R16Uint => TRACK_DECODE_REV,
    };
    TileStreamSpec::new(band, format, 0, rev)
}

/// Enforce [`MAX_SOURCE_SIDE`] on untiled/full-frame raster dimensions.
pub(crate) fn check_max_source_side(width: u32, height: u32) -> Result<(), RasterError> {
    if width > MAX_SOURCE_SIDE || height > MAX_SOURCE_SIDE {
        return Err(RasterError::LimitExceeded {
            width,
            height,
            estimated_bytes: u64::from(width)
                .saturating_mul(u64::from(height))
                .saturating_mul(4),
            limit_bytes: u64::from(MAX_SOURCE_SIDE)
                .saturating_mul(u64::from(MAX_SOURCE_SIDE))
                .saturating_mul(4),
        });
    }
    Ok(())
}

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
    /// Estimated working memory peak in bytes.
    pub estimated_peak_bytes: u64,
    /// Whether this level fits inside the configured memory budget.
    pub fits: bool,
}

/// Scheduler-facing description of one raster pyramid.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PyramidMeta {
    pub source: SourceId,
    pub spec: TileStreamSpec,
    /// Level plans ordered from `0` (finest native) to `max_level` (coarsest
    /// single-tile apex).
    pub levels: Vec<LevelPlan>,
}

impl PyramidMeta {
    /// Return the [`StreamKey`] identifying this pyramid stream.
    pub const fn stream_key(&self) -> StreamKey {
        StreamKey::new(
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
}

/// Build the per-level serving plan (`0..level_count`) for a chunked raster in
/// `format`.
pub(crate) fn build_chunked_level_table(
    grid: &Grid,
    width: u32,
    height: u32,
    format: PixelFormat,
    budget: &MemoryBudget,
) -> Vec<LevelPlan> {
    let base = (width, height);
    let count = level_count(width, height).min(u32::from(u8::MAX));

    (0..count)
        .map(|lvl_u32| {
            let level = lvl_u32 as u8;
            match grid.footprint(base, level, format, budget) {
                Some(fp) => {
                    let peak = grid.direct_peak_for_footprint(&fp, format).total();
                    LevelPlan {
                        serve: ServeMode::Pointwise,
                        estimated_peak_bytes: peak,
                        fits: peak <= budget.decode_working_bytes,
                    }
                }
                None => {
                    let peak = grid.pass_peak(base, format).total();
                    LevelPlan {
                        serve: ServeMode::Pass { from: 0 },
                        estimated_peak_bytes: peak,
                        fits: peak <= budget.decode_working_bytes,
                    }
                }
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_table_assigns_pass_when_chunk_exceeds_decoded_chunk_cache_cap() {
        let tiny_budget = MemoryBudget::for_host_ram(1024 * 1024 * 1024);
        let grid = Grid {
            chunk: (8192, 4096), // 33.5M px * 4 B = 134 MB > 64 MiB chunk cache
            overviews: Vec::new(),
            depth: 3,
        };
        let tbl =
            build_chunked_level_table(&grid, 8192, 8192, PixelFormat::Rgba8UnormSrgb, &tiny_budget);
        assert!(!tbl.is_empty());
        assert!(matches!(tbl[0].serve, ServeMode::Pass { from: 0 }));
    }
}
