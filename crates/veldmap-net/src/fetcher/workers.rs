//! Per-tile disk lookup, HTTP fetch, decode-submission backpressure, and
//! failure-recording workers for [`super::LiveTilePipeline`].

use std::sync::Arc;
use std::time::{Duration, Instant};

use veldmap_core::io::{CancelToken, lock_unpoisoned};
use veldmap_core::tile::{QuadTileId, TileKey};

use crate::basemap::LiveBasemapSpec;
use crate::http::{FetchFailure, FetchOutcome, download_capped};
use crate::{DecodeRequest, TileBlobStore};

use super::PipelineContext;
use super::dispatch::HttpFetchInput;
use super::guards::{DiskSlotGuard, InFlightGuard, PermitGuard};
use super::state::FailureState;

/// Short silent cooldown before re-attempting a tile whose decode submission
/// was rejected under CPU decode-pool backpressure.
const DECODE_BACKPRESSURE_BACKOFF: Duration = Duration::from_millis(50);

/// Submit `request` to the injected [`crate::TileDecodeExecutor`]; if rejected
/// under backpressure while still active and uncancelled, schedule a short
/// silent backoff (`DECODE_BACKPRESSURE_BACKOFF`) routing back to
/// `pending_disk` (when `retry_disk` is `true`, avoiding unnecessary HTTP
/// traffic for cached tiles) or `pending_http` so rejected tiles neither vanish
/// nor spin in a tight retry loop while the decode pool drains.
fn submit_or_requeue(ctx: &PipelineContext, request: DecodeRequest, retry_disk: bool) {
    let key = request.key;
    let cancel = request.cancel.clone();
    if ctx.decoder.submit(request) {
        return;
    }
    if cancel.is_cancelled() || ctx.shutdown.is_cancelled() {
        return;
    }
    let scheduled = lock_unpoisoned(&ctx.state).schedule_decode_backpressure(
        key,
        DECODE_BACKPRESSURE_BACKOFF,
        retry_disk,
        Instant::now(),
    );
    if scheduled {
        ctx.notify.notify_one();
    }
}

struct DecodeOutcomeContext<'a> {
    ctx: &'a Arc<PipelineContext>,
    spec: &'a Arc<LiveBasemapSpec>,
    key: TileKey,
    id: QuadTileId,
    cancel: &'a CancelToken,
}

fn handle_disk_decode_result(
    cb: &DecodeOutcomeContext<'_>,
    cache: &Arc<dyn TileBlobStore>,
    guard: InFlightGuard,
    outcome: Result<(), String>,
) {
    match outcome {
        Ok(()) => {
            let _guard = guard;
            lock_unpoisoned(&cb.ctx.state).mark_completed(cb.key, true);
        }
        Err(err) => {
            tracing::debug!(
                key = ?cb.key,
                error = %err,
                "Cached disk tile failed decode; deleting corrupt entry and falling back to HTTP"
            );
            // Keep `guard` alive inside the blocking delete task so `delete`
            // strictly happens-before re-queueing into `pending_http`.
            let del_ctx = Arc::clone(cb.ctx);
            let del_cache = Arc::clone(cache);
            let del_source_key = cb.spec.cache_key().to_owned();
            let del_cancel = cb.cancel.clone();
            let key = cb.key;
            let id = cb.id;
            cb.ctx.rt_handle.spawn_blocking(move || {
                if let Err(del_err) = del_cache.delete(&del_source_key, id) {
                    tracing::debug!(
                        ?key,
                        error = %del_err,
                        "Failed to delete corrupt cached tile entry"
                    );
                }
                let mut guard = guard;
                promote_to_http(&del_ctx, key, &del_cancel, &mut guard);
            });
        }
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
    let source_key = spec.cache_key().to_owned();
    let cache_for_read = Arc::clone(&cache);
    let disk_res = tokio::select! {
        _ = cancel.cancelled() => return,
        res = tokio::task::spawn_blocking(move || cache_for_read.get(&source_key, id)) => res,
    };
    // Release the disk-lookup concurrency slot immediately before decode/HTTP.
    drop(disk_slot);

    if cancel.is_cancelled() {
        return;
    }

    match disk_res {
        Ok(Ok(Some(bytes))) => {
            let cb_ctx = Arc::clone(&ctx);
            let cb_cache = Arc::clone(&cache);
            let cb_cancel = cancel.clone();
            let cb_spec = Arc::clone(&spec);
            let req = DecodeRequest {
                key,
                bytes: Arc::from(bytes),
                cancel,
                on_decoded: Box::new(move |outcome| {
                    let cb = DecodeOutcomeContext {
                        ctx: &cb_ctx,
                        spec: &cb_spec,
                        key,
                        id,
                        cancel: &cb_cancel,
                    };
                    handle_disk_decode_result(&cb, &cb_cache, guard, outcome);
                }),
            };
            submit_or_requeue(&ctx, req, true);
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
    spec: &LiveBasemapSpec,
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
    let source_key = spec.cache_key().to_owned();
    let put_sem = Arc::clone(&ctx.disk_put_sem);
    let shutdown = ctx.shutdown.clone();
    ctx.rt_handle.spawn(async move {
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
            if let Err(err) = cache.put(&source_key, id, &bytes_for_cache) {
                tracing::debug!(
                    error = %err,
                    "Failed to persist validated live tile to disk cache"
                );
            }
        })
        .await;
    });
}

fn handle_http_decode_result(
    cb: &DecodeOutcomeContext<'_>,
    guard: InFlightGuard,
    bytes_for_cache: Arc<[u8]>,
    outcome: Result<(), String>,
) {
    let _guard = guard;
    match outcome {
        Ok(()) => {
            lock_unpoisoned(&cb.ctx.state).mark_completed(cb.key, false);
            persist_validated_tile_async(cb.ctx, cb.spec, cb.key, cb.id, bytes_for_cache);
        }
        Err(err) => {
            let safe_msg = cb.spec.redact_secrets(&err);
            record_failure(
                cb.ctx,
                cb.cancel,
                cb.key,
                FailureState::Terminal,
                false,
                Some(safe_msg),
            );
        }
    }
}

pub(super) async fn run_http_fetch(
    ctx: Arc<PipelineContext>,
    input: HttpFetchInput,
    guard: InFlightGuard,
    permit_guard: PermitGuard,
) {
    let HttpFetchInput {
        key,
        spec,
        url,
        host,
        cancel,
    } = input;

    if cancel.is_cancelled() {
        return;
    }

    let Some(id) = key.quad_id() else {
        return;
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
            let cb_ctx = Arc::clone(&ctx);
            let cb_cancel = cancel.clone();
            let cb_spec = Arc::clone(&spec);
            let req = DecodeRequest {
                key,
                bytes: shared_bytes,
                cancel,
                on_decoded: Box::new(move |outcome| {
                    let cb = DecodeOutcomeContext {
                        ctx: &cb_ctx,
                        spec: &cb_spec,
                        key,
                        id,
                        cancel: &cb_cancel,
                    };
                    handle_http_decode_result(&cb, guard, bytes_for_cache, outcome);
                }),
            };
            submit_or_requeue(&ctx, req, false);
        }
        Ok(FetchOutcome::Empty204) => {
            // Valid empty/nodata tile: suppress infinite refetch without
            // bumping `failed_count` or displaying an error badge.
            if !cancel.is_cancelled() {
                lock_unpoisoned(&ctx.state).mark_empty_204(key, Instant::now());
            }
        }
        Err(FetchFailure::Terminal(msg)) => {
            let safe_msg = spec.redact_secrets(&msg);
            tracing::debug!(?key, error = %safe_msg, "Terminal HTTP tile failure");
            record_failure(
                &ctx,
                &cancel,
                key,
                FailureState::Terminal,
                false,
                Some(safe_msg),
            );
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
            record_failure(
                &ctx,
                &cancel,
                key,
                FailureState::BackoffUntil {
                    until,
                    retry_disk: false,
                },
                offline,
                Some(safe_msg),
            );
        }
    }
}

fn record_failure(
    ctx: &PipelineContext,
    cancel: &CancelToken,
    key: TileKey,
    failure: FailureState,
    offline: bool,
    error_msg: Option<String>,
) {
    if cancel.is_cancelled() {
        return;
    }
    let should_wake =
        lock_unpoisoned(&ctx.state).mark_failure(key, failure, offline, error_msg, Instant::now());
    if should_wake {
        (ctx.wakeup)();
    }
}
