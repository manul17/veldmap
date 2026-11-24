//! Two-lane pointwise/pass decode scheduling, RAII flight guards, and worker
//! execution over [`super::qoi_cache`].

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use veldmap_core::sync::{CancelToken, lock_unpoisoned};
use veldmap_core::tile::{DecodedTile, PixelFormat, TileKey};
use veldmap_raster::{
    AdmittedPass, AdmittedTile, DecodeJob, DecodeJobKey, PassKey, PyramidMeta, RasterError,
    RasterSource, ServeMode, SubmitError, TileFailureOutcome, generate_linear_mipmaps,
};
use veldmap_storage::DiskTileKey;

use crate::gpu_tiles::{GpuTileCache, PendingUploads};

use super::default_display_tile_bytes;
use super::qoi_cache::{QoiWriteBatch, load_qoi_display_tile, store_qoi_display_tile};
use super::runtime::OverlayRuntime;
use super::stack::OverlayLayer;

struct PassJobSpec {
    admitted: AdmittedPass,
    wanted_shared: Arc<Mutex<HashSet<TileKey>>>,
    source: Arc<dyn RasterSource>,
    meta: PyramidMeta,
    fingerprint_hex: String,
    runtime: Arc<OverlayRuntime>,
}

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
        if lock_unpoisoned(&self.runtime.scheduler).on_pointwise_arrived(&self.key, self.job_id) {
            (self.runtime.wakeup)();
        }
    }

    fn mark_failed(mut self, terminal: bool) {
        self.completed = true;
        let outcome = lock_unpoisoned(&self.runtime.scheduler).on_pointwise_failed(
            &self.key,
            self.job_id,
            terminal,
        );
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
        if lock_unpoisoned(&self.runtime.scheduler).cancel_in_flight(&self.key, self.job_id) {
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

/// RAII guard guaranteeing an active sequential pass in
/// [`veldmap_raster::TileScheduler`] is always closed even if evicted from
/// `DecodeQueue` or unwound by a worker panic.
struct PassFlightGuard {
    runtime: Arc<OverlayRuntime>,
    pass_key: PassKey,
    pass_id: u64,
    finished: bool,
}

impl PassFlightGuard {
    fn new(runtime: Arc<OverlayRuntime>, pass_key: PassKey, pass_id: u64) -> Self {
        Self {
            runtime,
            pass_key,
            pass_id,
            finished: false,
        }
    }

    fn finish(mut self, succeeded: bool) {
        self.finished = true;
        if lock_unpoisoned(&self.runtime.scheduler).finish_pass(
            self.pass_key,
            self.pass_id,
            succeeded,
        ) {
            self.runtime.request_reschedule();
        }
    }

    fn abort(mut self) {
        self.finished = true;
        self.cleanup_aborted();
    }

    fn cleanup_aborted(&self) {
        let token =
            lock_unpoisoned(&self.runtime.scheduler).abort_pass(self.pass_key, self.pass_id);
        if let Some(tok) = token {
            tok.cancel();
            self.runtime.request_reschedule();
        }
    }
}

impl Drop for PassFlightGuard {
    fn drop(&mut self) {
        if !self.finished {
            self.cleanup_aborted();
        }
    }
}

pub(super) fn schedule_layer_rungs(
    layer: &OverlayLayer,
    ladder_cells: &[(u8, Vec<(u32, u32)>)],
    runtime: &Arc<OverlayRuntime>,
    gpu_cache: &GpuTileCache,
    pending: &PendingUploads,
) {
    let is_cached = |key: &TileKey| gpu_cache.contains(key) || pending.contains(key);

    // Walk the ladder coarsest-first so coarse root coverage lands before fine
    // tiles queue up, while never letting a terminally failed tile on a coarse
    // rung block finer rungs.
    for (rung, rung_cells) in ladder_cells {
        let rung = *rung;
        let Some(plan) = layer.meta.level(rung) else {
            continue;
        };
        if !plan.fits || rung_cells.is_empty() {
            continue;
        }
        let candidates: Vec<TileKey> = rung_cells
            .iter()
            .map(|&(x, y)| layer.meta.tile_key(rung, x, y))
            .collect();

        match plan.serve {
            ServeMode::Pointwise => {
                let (admitted, rung_unfinished) = {
                    let mut sched = lock_unpoisoned(&runtime.scheduler);
                    let admitted = sched.admit_pointwise(
                        &layer.meta,
                        rung,
                        &candidates,
                        is_cached,
                        &runtime.shutdown,
                    );
                    let unfinished = candidates
                        .iter()
                        .any(|k| !gpu_cache.contains(k) && !sched.is_terminal_failed(k));
                    (admitted, unfinished)
                };
                for tile in admitted {
                    if !submit_pointwise_job(tile, &layer.source, &layer.fingerprint_hex, runtime) {
                        break;
                    }
                }
                if rung_unfinished {
                    break;
                }
            }
            ServeMode::Pass { from } => {
                let pass_key = layer.meta.pass_key();
                let (cancelled_old, admitted_pass, pass_active) = {
                    let mut sched = lock_unpoisoned(&runtime.scheduler);
                    let cancelled_old = sched.update_target(pass_key, rung, Some(from));
                    let admitted = sched.admit_pass(
                        &layer.meta,
                        rung,
                        &candidates,
                        is_cached,
                        &runtime.shutdown,
                    );
                    let active = admitted.is_some() || sched.has_active_pass(pass_key);
                    (cancelled_old, admitted, active)
                };
                if let Some(tok) = cancelled_old {
                    tok.cancel();
                }
                if let Some(admitted) = admitted_pass {
                    submit_pass_job(PassJobSpec {
                        admitted,
                        wanted_shared: Arc::clone(&layer.wanted_shared),
                        source: Arc::clone(&layer.source),
                        meta: layer.meta.clone(),
                        fingerprint_hex: layer.fingerprint_hex.clone(),
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
fn submit_pointwise_job(
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

    match runtime.fast_pool.try_submit(job) {
        Ok(()) => true,
        Err(err) => {
            // Unregister BEFORE dropping the rejected job so `PointwiseFlightGuard::drop`
            // does not trigger an immediate 100% CPU repaint loop.
            let _ = lock_unpoisoned(&runtime.scheduler).cancel_in_flight(&tile_key, job_id);
            let is_full = matches!(err, SubmitError::Full(_));
            drop(err);
            if is_full {
                runtime.note_submit_backoff();
            }
            false
        }
    }
}

fn deliver_pass_tile(
    rt: &OverlayRuntime,
    pass_key: PassKey,
    pass_id: u64,
    tk: TileKey,
    tile: DecodedTile,
    cancel: &CancelToken,
) -> bool {
    if matches!(rt.done_tx.send_if_uncancelled((tk, tile), cancel), Ok(true)) {
        lock_unpoisoned(&rt.scheduler).on_pass_tile_arrived(pass_key, pass_id, &tk);
        (rt.wakeup)();
        true
    } else {
        false
    }
}

fn run_pass_job(spec: PassJobSpec, guard: PassFlightGuard) {
    let PassJobSpec {
        admitted:
            AdmittedPass {
                key: pass_key,
                pass_id,
                cancel: pass_cancel,
                tiles: missing,
                ..
            },
        wanted_shared,
        source,
        meta,
        fingerprint_hex: fp,
        runtime: rt,
    } = spec;

    // Stream any QOI disk-cache hits immediately (never buffering an entire
    // rung in RAM or discarding partial hits on 1 miss).
    let mut already_hit: HashSet<TileKey> = HashSet::new();
    let mut any_miss = false;
    if let Some(ref disk) = rt.disk_cache {
        for &tk in &missing {
            if pass_cancel.is_cancelled() {
                guard.abort();
                return;
            }
            let dk = DiskTileKey::from_tile_key(fp.clone(), &tk);
            if let Some(tile) = load_qoi_display_tile(disk, &dk) {
                if deliver_pass_tile(&rt, pass_key, pass_id, tk, tile, &pass_cancel) {
                    already_hit.insert(tk);
                }
            } else {
                any_miss = true;
            }
        }
        if !any_miss {
            guard.finish(true);
            return;
        }
    }

    let initial_missing: HashSet<TileKey> = missing
        .into_iter()
        .filter(|tk| !already_hit.contains(tk))
        .collect();
    let max_level = meta.max_level();
    let mut qoi_batch = QoiWriteBatch::new();

    let ctx = rt.decode_context(&pass_cancel);

    let result = source.produce_pass(&ctx, &mut |level, tx, ty, tw, th, rgba| {
        if pass_cancel.is_cancelled() {
            return Err(RasterError::Cancelled);
        }
        let tk = meta.tile_key(level, tx, ty);
        if already_hit.contains(&tk) {
            return Ok(());
        }
        if let Some(ref disk) = rt.disk_cache {
            qoi_batch.push_and_maybe_flush(
                disk,
                DiskTileKey::from_tile_key(fp.clone(), &tk),
                tw,
                th,
                &rgba,
                &pass_cancel,
            );
        }

        // Read live `wanted_shared` briefly (never nesting locks) so mid-pass
        // zooms upload tiles matching the current viewport.
        let should_upload = level == max_level
            || initial_missing.contains(&tk)
            || lock_unpoisoned(&wanted_shared).contains(&tk);
        if should_upload {
            let mut tile = DecodedTile::try_new(tw, th, PixelFormat::Rgba8UnormSrgb, rgba)
                .map_err(|e| RasterError::Decode(e.to_string()))?;
            generate_linear_mipmaps(&mut tile);
            let _ = deliver_pass_tile(&rt, pass_key, pass_id, tk, tile, &pass_cancel);
        }
        Ok(())
    });

    if let Some(ref disk) = rt.disk_cache {
        if result.is_ok() {
            qoi_batch.flush(disk, &pass_cancel);
        } else {
            qoi_batch.clear();
        }
    }

    match result {
        Ok(()) => guard.finish(true),
        Err(err) if err.is_cancelled() => guard.abort(),
        Err(err) => {
            tracing::warn!(
                source = ?meta.source,
                error = %err,
                "Sequential overlay pass failed"
            );
            guard.finish(false);
        }
    }
}

fn submit_pass_job(spec: PassJobSpec) {
    let pass_key = spec.admitted.key;
    let pass_id = spec.admitted.pass_id;
    let pass_cancel = spec.admitted.cancel.clone();
    let runtime = Arc::clone(&spec.runtime);
    let queued_bytes = spec
        .source
        .pass_peak_bytes()
        .min(runtime.budget.bulk_lane_pending_bytes as u64) as usize;
    let guard = PassFlightGuard::new(Arc::clone(&runtime), pass_key, pass_id);

    let job = DecodeJob {
        key: DecodeJobKey::Pass(pass_key),
        queued_bytes: queued_bytes.max(default_display_tile_bytes()),
        cancel: pass_cancel,
        work: Box::new(move || run_pass_job(spec, guard)),
    };

    if let Err(err) = runtime.bulk_pool.try_submit(job) {
        let token = lock_unpoisoned(&runtime.scheduler).abort_pass(pass_key, pass_id);
        let is_full = matches!(err, SubmitError::Full(_));
        drop(err);
        if let Some(tok) = token {
            tok.cancel();
        }
        if is_full {
            runtime.note_submit_backoff();
        }
    }
}
