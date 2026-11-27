//! Two-lane pointwise/pass decode scheduling, RAII flight guards, shared
//! remote-COG range coalescing/prefetching, and worker execution over
//! [`super::disk_cache`].

pub(super) mod band_stats;
mod pass;
mod pointwise;

use std::sync::Arc;

use veldmap_core::io::IoError;
use veldmap_core::sync::CancelToken;
use veldmap_core::tile::TileKey;
use veldmap_net::{HttpRangeSource, PrefetchPin};
use veldmap_raster::{
    ByteRange, DEFAULT_COALESCE_GAP_BYTES, DEFAULT_COALESCE_MAX_SPAN_BYTES, ServeMode,
    coalesce_byte_ranges,
};

use crate::gpu_tiles::{GpuTileCache, PendingUploads};

use super::runtime::OverlayRuntime;
use super::stack::OverlayLayer;

use self::pass::{PassJobSpec, submit_pass_job};
use self::pointwise::submit_pointwise_job;

/// Coalesce remote COG chunk byte ranges into `(offset, len)` spans ready for
/// `HttpRangeSource::prefetch_ranges_async`.
fn coalesce_remote_ranges(mut ranges: Vec<ByteRange>) -> Vec<(u64, u64)> {
    coalesce_byte_ranges(
        &mut ranges,
        DEFAULT_COALESCE_GAP_BYTES,
        DEFAULT_COALESCE_MAX_SPAN_BYTES,
    );
    ranges.into_iter().map(|r| (r.offset, r.len)).collect()
}

/// Coalesce `ranges` and asynchronously prefetch/pin all needed byte blocks on
/// `range_src`, returning `Ok(None)` when no byte ranges are needed.
async fn prefetch_coalesced_ranges(
    range_src: &HttpRangeSource,
    ranges: Vec<ByteRange>,
    cancel: &CancelToken,
) -> Result<Option<PrefetchPin>, IoError> {
    let byte_pairs = coalesce_remote_ranges(ranges);
    if byte_pairs.is_empty() {
        return Ok(None);
    }
    range_src
        .prefetch_ranges_async(&byte_pairs, cancel)
        .await
        .map(Some)
}

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
