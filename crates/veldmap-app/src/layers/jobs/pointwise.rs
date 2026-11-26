//! Pointwise (`ServeMode::Pointwise`) fast-lane decode jobs and RAII flight guard.

use std::sync::Arc;

use veldmap_core::sync::CancelToken;
use veldmap_core::tile::{DecodedTile, PixelFormat, RASTER_TILE_SIZE, TileKey};
use veldmap_net::{HttpRangeSource, PrefetchPin};
use veldmap_raster::{
    AdmittedTile, DecodeJob, DecodeJobKey, RasterSource, TileFailureOutcome, coalesce_byte_ranges,
    generate_tile_mipmaps,
};
use veldmap_storage::DiskTileKey;

use crate::gpu_tiles::tile_format_bytes;

use super::super::disk_cache::load_cached_tile;
use super::super::runtime::OverlayRuntime;

/// Gap threshold (`64 KiB`) when coalescing remote COG chunk byte ranges.
const REMOTE_COALESCE_GAP_BYTES: u64 = 64 * 1024;

/// Maximum span (`4 MiB`) of a single coalesced remote COG byte range.
const REMOTE_COALESCE_MAX_SPAN_BYTES: u64 = 4 * 1024 * 1024;

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
) -> bool {
    if matches!(
        rt.done_tx.send_if_uncancelled((tile_key, tile), cancel),
        Ok(true)
    ) {
        guard.mark_arrived();
        true
    } else {
        guard.mark_cancelled();
        false
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "internal helper carries explicit runtime, source, prefetch pin, and flight guard"
)]
fn run_pointwise_job(
    rt: &OverlayRuntime,
    source: &dyn RasterSource,
    fp: String,
    tile_key: TileKey,
    cancel: &CancelToken,
    check_disk: bool,
    _pin: Option<PrefetchPin>,
    guard: PointwiseFlightGuard,
) {
    if cancel.is_cancelled() {
        guard.mark_cancelled();
        return;
    }
    let disk_key = DiskTileKey::from_tile_key(fp, &tile_key);
    if check_disk
        && let Some(ref disk) = rt.disk_cache
        && let Some(tile) = load_cached_tile(disk, &disk_key)
    {
        let _ = deliver_pointwise_tile(rt, tile_key, tile, cancel, guard);
        return;
    }

    let spec = tile_key.stream_spec();
    let ctx = rt.decode_context(cancel);
    match source.decode_pointwise_tile(spec, tile_key.level, tile_key.x, tile_key.y, &ctx) {
        Ok(mut tile) => {
            if cancel.is_cancelled() {
                guard.mark_cancelled();
                return;
            }
            let disk_enabled = rt.disk_cache.as_ref().is_some_and(|d| d.max_bytes() > 0);
            let mip0_for_disk = if disk_enabled && rt.tile_writer.is_some() {
                DecodedTile::try_new(
                    tile.width(),
                    tile.height(),
                    tile.format(),
                    tile.data().to_vec(),
                )
                .ok()
            } else {
                None
            };
            generate_tile_mipmaps(&mut tile);
            // Deliver to the GPU upload queue FIRST, then offload QOI/LZ4
            // encoding and batched SQLite commits to `veldmap-tile-write`.
            if deliver_pointwise_tile(rt, tile_key, tile, cancel, guard)
                && let (Some(writer), Some(mip0)) = (&rt.tile_writer, mip0_for_disk)
            {
                writer.enqueue_mip0_tile(disk_key, mip0, cancel);
            }
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

fn plan_remote_tile_ranges(
    rt: &OverlayRuntime,
    source: &dyn RasterSource,
    tile_key: TileKey,
) -> Vec<(u64, u64)> {
    let spec = tile_key.stream_spec();
    let mut ranges = source
        .plan_pointwise_byte_ranges(
            spec,
            tile_key.level,
            &[(tile_key.x, tile_key.y)],
            &rt.budget,
            &rt.chunk_cache,
        )
        .unwrap_or_default();
    if spec.format == PixelFormat::Rgba8UnormSrgb
        && source.needs_display_stretch()
        && let Ok(stretch_ranges) = source.plan_stretch_byte_ranges()
    {
        ranges.extend(stretch_ranges);
    }
    coalesce_byte_ranges(
        &mut ranges,
        REMOTE_COALESCE_GAP_BYTES,
        REMOTE_COALESCE_MAX_SPAN_BYTES,
    );
    ranges.into_iter().map(|r| (r.offset, r.len)).collect()
}

/// Submit a single pointwise overlay decode job.
///
/// Local rasters submit directly to `fast_pool`. Remote COGs spawn an async
/// prefetch task on `NetContext` that checks SQLite, plans + coalesces + pins
/// required HTTP byte ranges into a [`PrefetchPin`], and then submits the
/// pure-CPU decode (capturing `PrefetchPin`) onto `fast_pool`.
///
/// Returns `false` if the lane is full or closed so the caller stops submitting
/// this rung and backs off cleanly without spinning.
pub(super) fn submit_pointwise_job(
    admitted: AdmittedTile,
    source: &Arc<dyn RasterSource>,
    remote_range: Option<&HttpRangeSource>,
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

    if source.is_remote()
        && let (Some(range_src), Some(net)) = (remote_range.cloned(), runtime.net.clone())
    {
        let handle = net.handle().clone();
        let disk_handle = handle.clone();
        handle.spawn(async move {
            if cancel.is_cancelled() {
                guard.mark_cancelled();
                return;
            }

            // 1. Fast path: check SQLite disk cache off the async worker thread.
            if let Some(disk) = rt.disk_cache.clone() {
                let disk_key = DiskTileKey::from_tile_key(fp.clone(), &tile_key);
                let cached = disk_handle
                    .spawn_blocking(move || load_cached_tile(&disk, &disk_key))
                    .await
                    .ok()
                    .flatten();
                if let Some(tile) = cached {
                    let _ = deliver_pointwise_tile(&rt, tile_key, tile, &cancel, guard);
                    return;
                }
            }

            if cancel.is_cancelled() {
                guard.mark_cancelled();
                return;
            }

            // 2. Plan and asynchronously prefetch/pin all needed byte ranges.
            let byte_pairs = plan_remote_tile_ranges(&rt, source.as_ref(), tile_key);
            let pin = if byte_pairs.is_empty() {
                None
            } else {
                match range_src.prefetch_ranges_async(&byte_pairs, &cancel).await {
                    Ok(pin) => Some(pin),
                    Err(err) if err.is_cancelled() => {
                        guard.mark_cancelled();
                        return;
                    }
                    Err(err) => {
                        tracing::warn!(
                            source = ?tile_key.source,
                            level = tile_key.level,
                            x = tile_key.x,
                            y = tile_key.y,
                            error = %err,
                            "Remote COG range prefetch failed"
                        );
                        let terminal = matches!(err, veldmap_core::io::IoError::InvalidInput(_));
                        guard.mark_failed(terminal);
                        return;
                    }
                }
            };

            if cancel.is_cancelled() {
                guard.mark_cancelled();
                return;
            }

            // 3. Submit pure CPU decode (holding `PrefetchPin` so zero HTTP I/O
            //    occurs on `fast_pool`) with rejection cleanup.
            let job_rt = Arc::clone(&rt);
            let job_cancel = cancel.clone();
            let job = DecodeJob {
                key: DecodeJobKey::Tile(tile_key),
                queued_bytes: tile_format_bytes(RASTER_TILE_SIZE, tile_key.format),
                cancel,
                work: Box::new(move || {
                    run_pointwise_job(
                        &job_rt,
                        source.as_ref(),
                        fp,
                        tile_key,
                        &job_cancel,
                        false,
                        pin,
                        guard,
                    );
                }),
            };
            let _ = rt.try_submit_guarded(&rt.fast_pool, job, true, || {
                let _ = rt.cancel_pointwise_in_flight(&tile_key, job_id);
            });
        });
        return true;
    }

    let pool = if source.is_remote() {
        &runtime.remote_pool
    } else {
        &runtime.fast_pool
    };
    let job = DecodeJob {
        key: DecodeJobKey::Tile(tile_key),
        queued_bytes: tile_format_bytes(RASTER_TILE_SIZE, tile_key.format),
        cancel: cancel.clone(),
        work: Box::new(move || {
            run_pointwise_job(
                &rt,
                source.as_ref(),
                fp,
                tile_key,
                &cancel,
                true,
                None,
                guard,
            );
        }),
    };

    runtime.try_submit_guarded(pool, job, true, || {
        let _ = runtime.cancel_pointwise_in_flight(&tile_key, job_id);
    })
}
