//! Pointwise (`ServeMode::Pointwise`) fast-lane decode jobs and RAII flight guard.

use std::sync::Arc;

use veldmap_core::sync::CancelToken;
use veldmap_core::tile::{DecodedTile, TileKey};
use veldmap_raster::{
    AdmittedTile, DecodeJob, DecodeJobKey, RasterSource, TileFailureOutcome,
    generate_linear_mipmaps,
};
use veldmap_storage::DiskTileKey;

use super::super::default_display_tile_bytes;
use super::super::qoi_cache::{load_qoi_display_tile, store_qoi_display_tile};
use super::super::runtime::OverlayRuntime;

/// RAII guard guaranteeing a pointwise tile never remains stuck in-flight in
/// [`veldmap_raster::TileScheduler`] if its `DecodeJob` is evicted from
/// `DecodeQueue`, cancelled, or panics inside a worker thread.
struct PointwiseFlightGuard {
    runtime: Arc<OverlayRuntime>,
    key: TileKey,
    job_id: u64,
    completed: bool,
}

impl PointwiseFlightGuard {
    fn new(runtime: Arc<OverlayRuntime>, key: TileKey, job_id: u64) -> Self {
        Self {
            runtime,
            key,
            job_id,
            completed: false,
        }
    }

    fn mark_arrived(mut self) {
        self.completed = true;
        if self.runtime.on_pointwise_arrived(&self.key, self.job_id) {
            (self.runtime.wakeup)();
        }
    }

    fn mark_failed(mut self, terminal: bool) {
        self.completed = true;
        let outcome = self
            .runtime
            .on_pointwise_failed(&self.key, self.job_id, terminal);
        // Both RetryEligible and TerminalFailed require re-evaluating scheduling
        // (either to retry or to unblock finer ladder rungs behind a terminal failure).
        if matches!(
            outcome,
            Some(TileFailureOutcome::RetryEligible | TileFailureOutcome::TerminalFailed)
        ) {
            self.runtime.request_reschedule();
        }
    }

    fn mark_cancelled(mut self) {
        self.completed = true;
        self.cleanup_cancelled();
    }

    fn cleanup_cancelled(&self) {
        // Only wake/reschedule if `(key, job_id)` was still registered in the
        // scheduler (e.g. evicted from DecodeQueue or panicked). Intentional
        // cancellations via `reconcile_pointwise`, `reset_source`, or `try_submit`
        // rejection already removed the entry and skip spurious wakeups.
        if self
            .runtime
            .cancel_pointwise_in_flight(&self.key, self.job_id)
        {
            self.runtime.request_reschedule();
        }
    }
}

impl Drop for PointwiseFlightGuard {
    fn drop(&mut self) {
        if !self.completed {
            self.cleanup_cancelled();
        }
    }
}

fn deliver_pointwise_tile(
    rt: &OverlayRuntime,
    tile_key: TileKey,
    tile: DecodedTile,
    cancel: &CancelToken,
    guard: PointwiseFlightGuard,
) {
    if matches!(
        rt.done_tx.send_if_uncancelled((tile_key, tile), cancel),
        Ok(true)
    ) {
        guard.mark_arrived();
    } else {
        guard.mark_cancelled();
    }
}

fn run_pointwise_job(
    rt: &OverlayRuntime,
    source: &dyn RasterSource,
    fp: String,
    tile_key: TileKey,
    cancel: &CancelToken,
    guard: PointwiseFlightGuard,
) {
    if cancel.is_cancelled() {
        guard.mark_cancelled();
        return;
    }
    let disk_key = DiskTileKey::from_tile_key(fp, &tile_key);
    if let Some(ref disk) = rt.disk_cache
        && let Some(tile) = load_qoi_display_tile(disk, &disk_key)
    {
        deliver_pointwise_tile(rt, tile_key, tile, cancel, guard);
        return;
    }

    let ctx = rt.decode_context(cancel);
    match source.decode_pointwise_tile(tile_key.level, tile_key.x, tile_key.y, &ctx) {
        Ok(mut tile) => {
            if cancel.is_cancelled() {
                guard.mark_cancelled();
                return;
            }
            if let Some(ref disk) = rt.disk_cache {
                store_qoi_display_tile(
                    disk,
                    &disk_key,
                    tile.width(),
                    tile.height(),
                    tile.data(),
                    cancel,
                );
            }
            generate_linear_mipmaps(&mut tile);
            deliver_pointwise_tile(rt, tile_key, tile, cancel, guard);
        }
        Err(err) if err.is_cancelled() => {
            guard.mark_cancelled();
        }
        Err(err) => {
            tracing::warn!(
                source = ?tile_key.source,
                level = tile_key.level,
                x = tile_key.x,
                y = tile_key.y,
                error = %err,
                "Pointwise overlay tile decode failed"
            );
            guard.mark_failed(err.is_terminal());
        }
    }
}

/// Submit a single pointwise overlay decode job to `fast_pool`.
/// Returns `false` if the lane is full or closed so the caller stops submitting
/// this rung and backs off cleanly without spinning.
pub(super) fn submit_pointwise_job(
    admitted: AdmittedTile,
    source: &Arc<dyn RasterSource>,
    fingerprint_hex: &str,
    runtime: &Arc<OverlayRuntime>,
) -> bool {
    let tile_key = admitted.key;
    let job_id = admitted.job_id;
    let cancel = admitted.cancel;
    let source = Arc::clone(source);
    let fp = fingerprint_hex.to_owned();
    let rt = Arc::clone(runtime);
    let guard = PointwiseFlightGuard::new(Arc::clone(runtime), tile_key, job_id);

    let job = DecodeJob {
        key: DecodeJobKey::Tile(tile_key),
        queued_bytes: default_display_tile_bytes(),
        cancel: cancel.clone(),
        work: Box::new(move || {
            run_pointwise_job(&rt, source.as_ref(), fp, tile_key, &cancel, guard);
        }),
    };

    runtime.try_submit_guarded(&runtime.fast_pool, job, true, || {
        let _ = runtime.cancel_pointwise_in_flight(&tile_key, job_id);
    })
}
