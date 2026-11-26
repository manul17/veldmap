//! Two-lane pointwise/pass decode scheduling, RAII flight guards, and worker
//! execution over [`super::disk_cache`].

pub(super) mod band_stats;
mod pass;
mod pointwise;

use std::sync::Arc;

use veldmap_core::tile::TileKey;
use veldmap_raster::ServeMode;

use crate::gpu_tiles::{GpuTileCache, PendingUploads};

use super::runtime::OverlayRuntime;
use super::stack::OverlayLayer;

use self::pass::{PassJobSpec, submit_pass_job};
use self::pointwise::submit_pointwise_job;

pub(super) fn schedule_layer_rungs(
    layer: &OverlayLayer,
    ladder_cells: &[(u8, Vec<(u32, u32)>)],
    runtime: &Arc<OverlayRuntime>,
    gpu_cache: &GpuTileCache,
    pending: &PendingUploads,
) {
    // Walk the ladder coarsest-first so coarse root coverage lands before fine
    // tiles queue up, while never letting a terminally failed tile on a coarse
    // rung block finer rungs.
    for (rung, rung_cells) in ladder_cells {
        let rung = *rung;
        let Some(plan) = layer.meta().level(rung) else {
            continue;
        };
        if !plan.fits || rung_cells.is_empty() {
            continue;
        }
        let candidates: Vec<TileKey> = rung_cells
            .iter()
            .map(|&(x, y)| layer.meta().tile_key(rung, x, y))
            .collect();

        match plan.serve {
            ServeMode::Pointwise => {
                let (admitted, rung_unfinished) = runtime.admit_pointwise_rung(
                    layer.meta(),
                    rung,
                    &candidates,
                    gpu_cache,
                    pending,
                );
                for tile in admitted {
                    if !submit_pointwise_job(
                        tile,
                        layer.source(),
                        layer.remote_range(),
                        layer.fingerprint_hex(),
                        runtime,
                    ) {
                        break;
                    }
                }
                if rung_unfinished {
                    break;
                }
            }
            ServeMode::Pass { from } => {
                let (admitted_pass, pass_active) = runtime.admit_pass_rung(
                    layer.meta(),
                    rung,
                    from,
                    &candidates,
                    gpu_cache,
                    pending,
                );
                if let Some(admitted) = admitted_pass {
                    submit_pass_job(PassJobSpec {
                        admitted,
                        wanted_shared: layer.wanted_shared(),
                        source: Arc::clone(layer.source()),
                        meta: layer.meta().clone(),
                        fingerprint_hex: layer.fingerprint_hex().to_owned(),
                        runtime: Arc::clone(runtime),
                    });
                }
                if pass_active {
                    break;
                }
            }
        }
    }
}
