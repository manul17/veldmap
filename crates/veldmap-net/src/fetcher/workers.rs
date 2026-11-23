//! Per-tile disk lookup, HTTP fetch, decode-submission backpressure, and
//! failure-recording workers for [`super::LiveTilePipeline`].

use std::sync::Arc;
use std::time::Instant;

use veldmap_core::io::{CancelToken, lock_unpoisoned};
use veldmap_core::tile::{PlanetaryScheme, TileKey};

use crate::DecodeRequest;
use crate::basemap::LiveBasemapSpec;
use crate::http::{FetchFailure, FetchOutcome, download_capped};

use super::dispatch::HttpFetchInput;
use super::guards::{DiskSlotGuard, InFlightGuard, PermitGuard};
use super::state::FailureState;
use super::{DECODE_BACKPRESSURE_BACKOFF, PipelineContext};

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
    let mut should_notify = false;
    {
        let now = Instant::now();
        let mut st = lock_unpoisoned(&ctx.state);
        if st
            .active_spec
            .as_ref()
            .is_some_and(|s| s.source_id() == key.source)
            && st.last_wanted.contains(&key)
            && !st.in_flight.contains_key(&key)
            && !st.failures.contains_key(&key)
        {
            st.insert_failure(
                key,
                FailureState::BackoffUntil {
                    until: now + DECODE_BACKPRESSURE_BACKOFF,
                    retry_disk,
                },
                now,
            );
            should_notify = true;
        }
    }
    if should_notify {
        ctx.notify.notify_one();
    }
}

pub(super) async fn run_disk_lookup(
    ctx: Arc<PipelineContext>,
    key: TileKey,
    spec: Arc<LiveBasemapSpec>,
    cancel: CancelToken,
    mut guard: InFlightGuard,
    disk_slot: DiskSlotGuard,
) {
    if cancel.is_cancelled() {
        return;
    }
    let Some(cache) = ctx.disk_cache.clone() else {
        promote_to_http(&ctx, key, &cancel, &mut guard);
        return;
    };
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
            let cb_cancel = cancel.clone();
            let cb_spec = Arc::clone(&spec);
            let req = DecodeRequest {
                key,
                bytes: Arc::from(bytes),
                cancel,
                on_decoded: Box::new(move |outcome| match outcome {
                    Ok(()) => {
                        let _guard = guard;
                        let mut st = lock_unpoisoned(&cb_ctx.state);
                        if st
                            .active_spec
                            .as_ref()
                            .is_some_and(|s| s.source_id() == key.source)
                        {
                            st.last_wanted.remove(&key);
                            st.completed_from_disk = st.completed_from_disk.saturating_add(1);
                        }
                    }
                    Err(err) => {
                        tracing::debug!(
                            ?key,
                            error = %err,
                            "Cached disk tile failed decode; deleting corrupt entry and falling back to HTTP"
                        );
                        // Keep `guard` alive inside the blocking delete task so
                        // `delete` strictly happens-before re-queueing into
                        // `pending_http` (preventing delete-vs-put and
                        // `reconcile_wanted` races).
                        let del_ctx = Arc::clone(&cb_ctx);
                        let del_cache = Arc::clone(&cache);
                        let del_source_key = cb_spec.cache_key().to_owned();
                        cb_ctx.rt_handle.spawn_blocking(move || {
                            if let Err(del_err) = del_cache.delete(&del_source_key, id) {
                                tracing::debug!(
                                    ?key,
                                    error = %del_err,
                                    "Failed to delete corrupt cached tile entry"
                                );
                            }
                            let mut guard = guard;
                            promote_to_http(&del_ctx, key, &cb_cancel, &mut guard);
                        });
                    }
                }),
            };
            submit_or_requeue(&ctx, req, true);
        }
        Ok(Ok(None)) => {
            promote_to_http(&ctx, key, &cancel, &mut guard);
        }
        Ok(Err(err)) => {
            tracing::debug!(
                ?key,
                error = %err,
                "Disk cache lookup failed; falling back to HTTP"
            );
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
    let mut should_notify = false;
    {
        let mut st = lock_unpoisoned(&ctx.state);
        if st
            .in_flight
            .get(&key)
            .is_some_and(|(gen_id, _)| *gen_id == guard.generation)
        {
            st.in_flight.remove(&key);
            guard.disarm();
            should_notify = true;
            if !cancel.is_cancelled()
                && !ctx.shutdown.is_cancelled()
                && st
                    .active_spec
                    .as_ref()
                    .is_some_and(|s| s.source_id() == key.source)
                && !st.failures.contains_key(&key)
            {
                st.insert_pending_http(key);
            }
        }
    }
    if should_notify {
        ctx.notify.notify_one();
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
    // Final defense-in-depth contract check: never touch PolarCap.
    if matches!(id.scheme, PlanetaryScheme::PolarCap { .. }) {
        return;
    }

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
                    let _guard = guard;
                    match outcome {
                        Ok(()) => {
                            // Update stats synchronously (nanoseconds, race-free
                            // even if UI `reconcile_wanted` already cancelled
                            // `cb_cancel` after receiving the tile), and persist
                            // validated bytes off the decode thread under both
                            // a bounded pending-write queue cap and a bounded
                            // active-writer semaphore.
                            {
                                let mut st = lock_unpoisoned(&cb_ctx.state);
                                if st
                                    .active_spec
                                    .as_ref()
                                    .is_some_and(|s| s.source_id() == key.source)
                                {
                                    st.last_wanted.remove(&key);
                                    st.completed_from_net = st.completed_from_net.saturating_add(1);
                                    st.offline = false;
                                    st.last_error = None;
                                }
                            }
                            if let Some(cache) = cb_ctx.disk_cache.clone()
                                && !cb_ctx.shutdown.is_cancelled()
                            {
                                let Ok(queue_permit) =
                                    Arc::clone(&cb_ctx.disk_put_queue_sem).try_acquire_owned()
                                else {
                                    tracing::debug!(
                                        ?key,
                                        "Skipping best-effort tile disk write under write-queue backpressure"
                                    );
                                    return;
                                };
                                let source_key = cb_spec.cache_key().to_owned();
                                let put_sem = Arc::clone(&cb_ctx.disk_put_sem);
                                let shutdown = cb_ctx.shutdown.clone();
                                let rt = cb_ctx.rt_handle.clone();
                                rt.spawn(async move {
                                    let _queue_permit = queue_permit;
                                    if shutdown.is_cancelled() {
                                        return;
                                    }
                                    let permit = tokio::select! {
                                        _ = shutdown.cancelled() => return,
                                        res = put_sem.acquire_owned() => match res {
                                            Ok(p) => p,
                                            Err(_) => return,
                                        },
                                    };
                                    if shutdown.is_cancelled() {
                                        return;
                                    }
                                    let _ = tokio::task::spawn_blocking(move || {
                                        let _permit = permit;
                                        if shutdown.is_cancelled() {
                                            return;
                                        }
                                        if let Err(err) =
                                            cache.put(&source_key, id, &bytes_for_cache)
                                        {
                                            tracing::debug!(
                                                error = %err,
                                                "Failed to persist validated live tile to disk cache"
                                            );
                                        }
                                    })
                                    .await;
                                });
                            }
                        }
                        Err(err) => {
                            let safe_msg = cb_spec.redact_secrets(&err);
                            record_failure(
                                &cb_ctx,
                                &cb_cancel,
                                key,
                                FailureState::Terminal,
                                false,
                                Some(safe_msg),
                            );
                        }
                    }
                }),
            };
            submit_or_requeue(&ctx, req, false);
        }
        Ok(FetchOutcome::Empty204) => {
            // Valid empty/nodata tile: suppress infinite refetch without
            // bumping `failed_count` or displaying an error badge.
            let now = Instant::now();
            let mut st = lock_unpoisoned(&ctx.state);
            if !cancel.is_cancelled()
                && st
                    .active_spec
                    .as_ref()
                    .is_some_and(|s| s.source_id() == key.source)
            {
                st.last_wanted.remove(&key);
                st.insert_failure(key, FailureState::Empty, now);
                st.offline = false;
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
            retry_after,
        }) => {
            let safe_msg = spec.redact_secrets(&msg);
            let delay = retry_after.unwrap_or(ctx.transient_retry_backoff);
            let until = Instant::now() + delay;
            if retry_after.is_some() {
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
    let now = Instant::now();
    let mut should_wake = false;
    {
        let mut st = lock_unpoisoned(&ctx.state);
        // Re-check cancellation and active source under lock so a late error
        // from a previous basemap never contaminates the newly active source.
        if !cancel.is_cancelled()
            && st
                .active_spec
                .as_ref()
                .is_some_and(|s| s.source_id() == key.source)
        {
            st.insert_failure(key, failure, now);
            st.failed_count = st.failed_count.saturating_add(1);
            if offline {
                st.offline = true;
            }
            if let Some(msg) = error_msg {
                st.last_error = Some(msg);
            }
            should_wake = true;
        }
    }
    if should_wake {
        (ctx.wakeup)();
    }
}
