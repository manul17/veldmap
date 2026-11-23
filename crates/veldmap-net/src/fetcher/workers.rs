//! Per-tile disk lookup, HTTP fetch, decode-submission backpressure, and
//! failure-recording workers for [`super::LiveTilePipeline`].

use std::sync::Arc;
use std::time::{Duration, Instant};

use veldmap_core::sync::{CancelToken, lock_unpoisoned};
use veldmap_core::tile::{QuadTileId, TileKey};

use crate::basemap::LiveBasemapSpec;
use crate::http::{FetchFailure, FetchOutcome, download_capped};
use crate::{DecodeCompletion, DecodeFailure, DecodeRequest, TileBlobStore};

use super::PipelineContext;
use super::dispatch::HttpFetchInput;
use super::guards::{DiskSlotGuard, InFlightGuard, PermitGuard};

/// Short silent cooldown before re-attempting a tile whose decode submission
/// was rejected or evicted under CPU decode-pool backpressure.
const DECODE_BACKPRESSURE_BACKOFF: Duration = Duration::from_millis(50);

fn schedule_decode_backpressure(
    ctx: &PipelineContext,
    key: TileKey,
    guard: &mut InFlightGuard,
    retry_disk: bool,
) {
    if lock_unpoisoned(&ctx.state).schedule_decode_backpressure(
        key,
        guard.generation,
        DECODE_BACKPRESSURE_BACKOFF,
        retry_disk,
        Instant::now(),
    ) {
        guard.disarm();
        ctx.notify.notify_one();
    }
}

/// Submit encoded tile bytes to the injected [`crate::TileDecodeExecutor`] and
/// await its [`DecodeCompletion`] inside the owning Tokio task.
///
/// Rejections, post-acceptance queue evictions, or decoder panics all route to
/// [`schedule_decode_backpressure`] unless cancelled/shutting down.
async fn submit_and_await_decode(
    ctx: &PipelineContext,
    key: TileKey,
    bytes: Arc<[u8]>,
    cancel: &CancelToken,
    mut guard: InFlightGuard,
    retry_disk: bool,
) -> Option<(InFlightGuard, Result<(), DecodeFailure>)> {
    let (on_decoded, rx) = DecodeCompletion::channel();
    let req = DecodeRequest {
        key,
        bytes,
        cancel: cancel.clone(),
        on_decoded,
    };

    if !ctx.decoder.submit(req) {
        if !cancel.is_cancelled() && !ctx.shutdown.is_cancelled() {
            schedule_decode_backpressure(ctx, key, &mut guard, retry_disk);
        }
        return None;
    }

    let outcome = tokio::select! {
        _ = cancel.cancelled() => return None,
        res = rx.recv() => res,
    };

    if matches!(outcome, Err(DecodeFailure::Aborted)) {
        if !cancel.is_cancelled() && !ctx.shutdown.is_cancelled() {
            schedule_decode_backpressure(ctx, key, &mut guard, retry_disk);
        }
        return None;
    }

    Some((guard, outcome))
}

async fn delete_corrupt_cached_tile(
    ctx: &PipelineContext,
    cache: Arc<dyn TileBlobStore>,
    spec: Arc<LiveBasemapSpec>,
    key: TileKey,
    id: QuadTileId,
    cancel: &CancelToken,
) {
    if cancel.is_cancelled() || ctx.shutdown.is_cancelled() {
        return;
    }
    let Ok(_queue_permit) = Arc::clone(&ctx.disk_put_queue_sem).try_acquire_owned() else {
        tracing::debug!(
            ?key,
            "Skipping corrupt disk tile deletion under write-queue backpressure"
        );
        return;
    };
    let permit = tokio::select! {
        _ = cancel.cancelled() => return,
        _ = ctx.shutdown.cancelled() => return,
        res = Arc::clone(&ctx.disk_put_sem).acquire_owned() => match res {
            Ok(p) => p,
            Err(_) => return,
        },
    };
    let cancel_for_delete = cancel.clone();
    let shutdown = ctx.shutdown.clone();
    let del_res = tokio::select! {
        _ = cancel.cancelled() => return,
        res = tokio::task::spawn_blocking(move || {
            let _permit = permit;
            if cancel_for_delete.is_cancelled() || shutdown.is_cancelled() {
                return Ok(());
            }
            cache.delete(spec.cache_key(), id)
        }) => res,
    };
    if let Ok(Err(del_err)) = del_res {
        tracing::debug!(
            ?key,
            error = %del_err,
            "Failed to delete corrupt cached tile entry"
        );
    }
}

pub(super) async fn run_disk_lookup(
    ctx: Arc<PipelineContext>,
    cache: Arc<dyn TileBlobStore>,
    key: TileKey,
    spec: Arc<LiveBasemapSpec>,
    cancel: CancelToken,
    mut guard: InFlightGuard,
    disk_slot: DiskSlotGuard,
) {
    if cancel.is_cancelled() {
        return;
    }
    let Some(id) = key.quad_id() else {
        return;
    };
    let spec_for_read = Arc::clone(&spec);
    let cache_for_read = Arc::clone(&cache);
    let disk_res = tokio::select! {
        _ = cancel.cancelled() => return,
        res = tokio::task::spawn_blocking(move || cache_for_read.get(spec_for_read.cache_key(), id)) => res,
    };
    // Release the disk-lookup concurrency slot immediately before decode/HTTP.
    drop(disk_slot);

    if cancel.is_cancelled() {
        return;
    }

    match disk_res {
        Ok(Ok(Some(bytes))) => {
            let Some((mut guard, outcome)) =
                submit_and_await_decode(&ctx, key, Arc::from(bytes), &cancel, guard, true).await
            else {
                return;
            };
            match outcome {
                Ok(()) => {
                    let transitioned =
                        lock_unpoisoned(&ctx.state).mark_completed(key, guard.generation, true);
                    guard.disarm();
                    if transitioned {
                        ctx.notify.notify_one();
                    }
                }
                Err(DecodeFailure::InvalidPayload(err)) => {
                    tracing::debug!(
                        ?key,
                        error = %err,
                        "Cached disk tile failed decode; deleting corrupt entry and falling back to HTTP"
                    );
                    delete_corrupt_cached_tile(&ctx, cache, spec, key, id, &cancel).await;
                    promote_to_http(&ctx, key, &cancel, &mut guard);
                }
                Err(DecodeFailure::Aborted) => {}
            }
        }
        Ok(Ok(None)) => promote_to_http(&ctx, key, &cancel, &mut guard),
        Ok(Err(err)) => {
            tracing::debug!(?key, error = %err, "Disk cache lookup failed; falling back to HTTP");
            promote_to_http(&ctx, key, &cancel, &mut guard);
        }
        Err(join_err) => {
            tracing::warn!(
                ?key,
                error = %join_err,
                "Disk cache lookup blocking task failed; falling back to HTTP"
            );
            promote_to_http(&ctx, key, &cancel, &mut guard);
        }
    }
}

fn promote_to_http(
    ctx: &PipelineContext,
    key: TileKey,
    cancel: &CancelToken,
    guard: &mut InFlightGuard,
) {
    let can_requeue = !cancel.is_cancelled() && !ctx.shutdown.is_cancelled();
    let promoted =
        lock_unpoisoned(&ctx.state).promote_disk_miss_to_http(key, guard.generation, can_requeue);
    if promoted {
        guard.disarm();
        ctx.notify.notify_one();
    }
}

fn persist_validated_tile_async(
    ctx: &PipelineContext,
    spec: &Arc<LiveBasemapSpec>,
    key: TileKey,
    id: QuadTileId,
    bytes_for_cache: Arc<[u8]>,
) {
    let Some(cache) = ctx.disk_cache.clone() else {
        return;
    };
    if ctx.shutdown.is_cancelled() {
        return;
    }
    let Ok(queue_permit) = Arc::clone(&ctx.disk_put_queue_sem).try_acquire_owned() else {
        tracing::debug!(
            ?key,
            "Skipping best-effort tile disk write under write-queue backpressure"
        );
        return;
    };
    let spec_for_put = Arc::clone(spec);
    let put_sem = Arc::clone(&ctx.disk_put_sem);
    let shutdown = ctx.shutdown.clone();
    tokio::spawn(async move {
        let _queue_permit = queue_permit;
        let permit = tokio::select! {
            _ = shutdown.cancelled() => return,
            res = put_sem.acquire_owned() => match res {
                Ok(p) => p,
                Err(_) => return,
            },
        };
        let _ = tokio::task::spawn_blocking(move || {
            let _permit = permit;
            if shutdown.is_cancelled() {
                return;
            }
            if let Err(err) = cache.put(spec_for_put.cache_key(), id, &bytes_for_cache) {
                tracing::debug!(
                    error = %err,
                    "Failed to persist validated live tile to disk cache"
                );
            }
        })
        .await;
    });
}

pub(super) async fn run_http_fetch(
    ctx: Arc<PipelineContext>,
    input: HttpFetchInput,
    mut guard: InFlightGuard,
    permit_guard: PermitGuard,
) {
    let HttpFetchInput {
        key,
        spec,
        host,
        cancel,
    } = input;

    if cancel.is_cancelled() {
        return;
    }

    let Some(id) = key.quad_id() else {
        return;
    };

    // Expand full tile URL outside `PipelineState`'s mutex.
    let url = match spec.resolve_url(id) {
        Ok(url) => url,
        Err(err) => {
            drop(permit_guard);
            let safe_msg = spec.redact_secrets(&err.to_string());
            record_terminal_failure(&ctx, &cancel, key, &mut guard, Some(safe_msg));
            return;
        }
    };

    let fetch_res = tokio::select! {
        _ = cancel.cancelled() => return,
        res = download_capped(&ctx.client, &url, ctx.max_tile_bytes) => res,
    };

    // Release the per-host HTTP permit BEFORE submitting CPU decode / disk put.
    drop(permit_guard);

    if cancel.is_cancelled() {
        return;
    }

    match fetch_res {
        Ok(FetchOutcome::Payload(bytes)) => {
            let shared_bytes: Arc<[u8]> = Arc::from(bytes);
            let bytes_for_cache = Arc::clone(&shared_bytes);
            let Some((mut guard, outcome)) =
                submit_and_await_decode(&ctx, key, shared_bytes, &cancel, guard, false).await
            else {
                return;
            };
            match outcome {
                Ok(()) => {
                    let transitioned =
                        lock_unpoisoned(&ctx.state).mark_completed(key, guard.generation, false);
                    guard.disarm();
                    if transitioned {
                        ctx.notify.notify_one();
                        persist_validated_tile_async(&ctx, &spec, key, id, bytes_for_cache);
                    }
                }
                Err(DecodeFailure::InvalidPayload(err)) => {
                    let safe_msg = spec.redact_secrets(&err);
                    record_terminal_failure(&ctx, &cancel, key, &mut guard, Some(safe_msg));
                }
                Err(DecodeFailure::Aborted) => {}
            }
        }
        Ok(FetchOutcome::Empty204) => {
            // Valid empty/nodata tile: suppress infinite refetch without
            // bumping `failed_count` or displaying an error badge.
            if !cancel.is_cancelled() {
                let transitioned =
                    lock_unpoisoned(&ctx.state).mark_empty_204(key, guard.generation);
                guard.disarm();
                if transitioned {
                    ctx.notify.notify_one();
                }
            }
        }
        Err(FetchFailure::Terminal(msg)) => {
            let safe_msg = spec.redact_secrets(&msg);
            tracing::debug!(?key, error = %safe_msg, "Terminal HTTP tile failure");
            record_terminal_failure(&ctx, &cancel, key, &mut guard, Some(safe_msg));
        }
        Err(FetchFailure::Transient {
            msg,
            offline,
            rate_limited,
            retry_after,
        }) => {
            let safe_msg = spec.redact_secrets(&msg);
            let delay = retry_after.unwrap_or(ctx.transient_retry_backoff);
            let until = Instant::now() + delay;
            if rate_limited || retry_after.is_some() {
                ctx.host_limiters.record_cooldown(&host, until);
            }
            tracing::debug!(
                ?key,
                offline,
                delay_s = delay.as_secs(),
                error = %safe_msg,
                "Transient HTTP tile failure"
            );
            if !cancel.is_cancelled() {
                let should_wake = lock_unpoisoned(&ctx.state).mark_transient_backoff(
                    key,
                    guard.generation,
                    until,
                    offline,
                    Some(safe_msg),
                );
                guard.disarm();
                if should_wake {
                    ctx.notify.notify_one();
                    (ctx.wakeup)();
                }
            }
        }
    }
}

fn record_terminal_failure(
    ctx: &PipelineContext,
    cancel: &CancelToken,
    key: TileKey,
    guard: &mut InFlightGuard,
    error_msg: Option<String>,
) {
    if cancel.is_cancelled() {
        return;
    }
    let should_wake =
        lock_unpoisoned(&ctx.state).mark_terminal_failure(key, guard.generation, error_msg);
    guard.disarm();
    if should_wake {
        ctx.notify.notify_one();
        (ctx.wakeup)();
    }
}
