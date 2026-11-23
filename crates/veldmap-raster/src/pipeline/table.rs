//! Level table planning (`LevelPlan`, `ServeMode`) derived from a chunk [`Grid`]
//! and [`MemoryBudget`].

use veldmap_core::budget::MemoryBudget;

use super::grid::Grid;
use super::pyramid;
use super::scheduler::{LevelPlan, ServeMode};

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
                    let peak = grid
                        .direct_peak(base, level, budget)
                        .map_or(0, |p| p.total());
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
