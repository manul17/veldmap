//! Per-tile disk lookup, HTTP fetch, decode-submission backpressure, and
//! failure-recording workers for [`crate::LiveTilePipeline`].

use std::sync::Arc;
use std::time::{Duration, Instant};

use veldmap_core::sync::CancelToken;
use veldmap_core::tile::{QuadTileId, TileKey};

use crate::basemap::LiveBasemapSpec;
use crate::{DecodeCompletion, DecodeFailure, DecodeRequest};

use super::PipelineContext;
use super::disk::{DiskCacheGate, DiskReadOutcome};
use super::download::{
    FetchFailure, FetchOutcome, TransientFetchError, TransientKind, download_capped,
};
use super::guards::{DiskSlotGuard, InFlightGuard, PermitGuard};
use super::state::{PipelineState, TileTarget, Transition};

/// Guarded disk-cache lookup task handed from the dispatcher to
/// [`run_disk_lookup`].
pub(super) struct DiskTask {
    pub(super) target: TileTarget,
    pub(super) guard: InFlightGuard,
    pub(super) disk_slot: DiskSlotGuard,
}

/// Guarded HTTP fetch task handed from the dispatcher to [`run_http_fetch`].
pub(super) struct HttpTask {
    pub(super) target: TileTarget,
    pub(super) host: String,
    pub(super) guard: InFlightGuard,
    pub(super) permit_guard: PermitGuard,
}

/// Short silent cooldown before re-attempting a tile whose decode submission
/// was rejected or evicted under CPU decode-pool backpressure.
const DECODE_BACKPRESSURE_BACKOFF: Duration = Duration::from_millis(50);

/// Non-aborted decode outcome returned by [`submit_and_await_decode`].
enum DecodeOutcome {
    Ok,
    InvalidPayload(String),
}

fn finish_with_wakeup(
    ctx: &PipelineContext,
    guard: InFlightGuard,
    f: impl FnOnce(&mut PipelineState, TileKey, u64) -> Transition,
) {
    if guard.finish(f).should_wakeup() {
        (ctx.wakeup)();
    }
}

fn schedule_decode_backpressure(guard: InFlightGuard, retry_disk: bool) {
    let now = Instant::now();
    guard.finish(|st, key, generation| {
        st.schedule_decode_backpressure(
            key,
            generation,
            DECODE_BACKPRESSURE_BACKOFF,
            retry_disk,
            now,
        )
    });
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
    guard: InFlightGuard,
    retry_disk: bool,
) -> Option<(InFlightGuard, DecodeOutcome)> {
    let (on_decoded, rx) = DecodeCompletion::channel();
    let req = DecodeRequest {
        key,
        bytes,
        cancel: cancel.clone(),
        on_decoded,
    };

    if !ctx.decoder.submit(req) {
        if !cancel.is_cancelled() {
            schedule_decode_backpressure(guard, retry_disk);
        }
        return None;
    }

    let outcome = tokio::select! {
        _ = cancel.cancelled() => return None,
        res = rx.recv() => res,
    };

    match outcome {
        Ok(()) => Some((guard, DecodeOutcome::Ok)),
        Err(DecodeFailure::InvalidPayload(msg)) => {
            Some((guard, DecodeOutcome::InvalidPayload(msg)))
        }
        Err(DecodeFailure::Aborted) => {
            if !cancel.is_cancelled() {
                schedule_decode_backpressure(guard, retry_disk);
            }
            None
        }
    }
}

fn promote_to_http(cancel: &CancelToken, guard: InFlightGuard) {
    let can_requeue = !cancel.is_cancelled();
    guard.finish(|st, key, generation| st.promote_disk_miss_to_http(key, generation, can_requeue));
}

pub(super) async fn run_disk_lookup(
    ctx: Arc<PipelineContext>,
    gate: DiskCacheGate,
    task: DiskTask,
) {
    let DiskTask {
        target:
            TileTarget {
                key,
                id,
                spec,
                cancel,
            },
        guard,
        disk_slot,
    } = task;
    if cancel.is_cancelled() {
        return;
    }

    // Move `disk_slot` into `gate.get` so the concurrency slot stays held until
    // the blocking SQLite read returns even if `cancel` fires mid-read.
    let read_outcome = gate.get(&spec, id, &cancel, disk_slot).await;

    match read_outcome {
        DiskReadOutcome::Cancelled => {}
        DiskReadOutcome::Hit(bytes, lease) => {
            let Some((guard, outcome)) =
                submit_and_await_decode(&ctx, key, Arc::from(bytes), &cancel, guard, true).await
            else {
                return;
            };
            match outcome {
                DecodeOutcome::Ok => {
                    gate.touch_async(lease, &ctx.shared.shutdown);
                    guard.finish(|st, key, generation| st.mark_completed(key, generation, true));
                }
                DecodeOutcome::InvalidPayload(err) => {
                    tracing::debug!(
                        ?key,
                        error = %err,
                        "Cached disk tile failed decode; deleting corrupt entry and falling back to HTTP"
                    );
                    gate.delete_corrupt(lease, key, &cancel).await;
                    promote_to_http(&cancel, guard);
                }
            }
        }
        DiskReadOutcome::Miss => promote_to_http(&cancel, guard),
        DiskReadOutcome::Error(err) => {
            tracing::debug!(?key, error = %err, "Disk cache lookup failed; falling back to HTTP");
            promote_to_http(&cancel, guard);
        }
    }
}

async fn handle_http_payload(
    ctx: &PipelineContext,
    spec: &Arc<LiveBasemapSpec>,
    key: TileKey,
    id: QuadTileId,
    bytes: Vec<u8>,
    cancel: &CancelToken,
    guard: InFlightGuard,
) {
    let shared_bytes: Arc<[u8]> = Arc::from(bytes);
    let bytes_for_cache = Arc::clone(&shared_bytes);
    let Some((guard, outcome)) =
        submit_and_await_decode(ctx, key, shared_bytes, cancel, guard, false).await
    else {
        return;
    };
    match outcome {
        DecodeOutcome::Ok => {
            let transition =
                guard.finish(|st, key, generation| st.mark_completed(key, generation, false));
            if transition.is_applied()
                && let Some(gate) = &ctx.disk_cache
            {
                gate.persist_async(
                    Arc::clone(spec),
                    key,
                    id,
                    bytes_for_cache,
                    ctx.shared.shutdown.clone(),
                );
            }
        }
        DecodeOutcome::InvalidPayload(err) => {
            let safe_msg = spec.redact_secrets(&err);
            record_terminal_failure(ctx, cancel, guard, Some(safe_msg));
        }
    }
}

fn handle_http_transient_failure(
    ctx: &PipelineContext,
    spec: &LiveBasemapSpec,
    host: &str,
    key: TileKey,
    cancel: &CancelToken,
    guard: InFlightGuard,
    err: TransientFetchError,
) {
    if cancel.is_cancelled() {
        return;
    }
    let safe_msg = spec.redact_secrets(&err.msg);
    let (offline, rate_limited, delay) = match err.kind {
        TransientKind::Network { offline } => (offline, false, ctx.transient_retry_backoff),
        TransientKind::ServerError => (false, false, ctx.transient_retry_backoff),
        TransientKind::RateLimited { retry_after } => (
            false,
            true,
            retry_after.unwrap_or(ctx.transient_retry_backoff),
        ),
    };
    let until = Instant::now() + delay;
    tracing::debug!(
        ?key,
        offline,
        delay_s = delay.as_secs(),
        error = %safe_msg,
        "Transient HTTP tile failure"
    );
    finish_with_wakeup(ctx, guard, |st, key, generation| {
        let tr = st.mark_transient_backoff(key, generation, until, offline, Some(safe_msg));
        // Record host cooldown atomically under `state -> host_limiters` lock
        // order and only if this task still belongs to the active source &
        // generation, preventing a cancelled/stale task from poisoning a newly
        // switched basemap that shares the same host.
        if tr.is_applied() && rate_limited {
            ctx.shared.host_limiters.record_cooldown(host, until);
        }
        tr
    });
}

pub(super) async fn run_http_fetch(ctx: Arc<PipelineContext>, task: HttpTask) {
    let HttpTask {
        target:
            TileTarget {
                key,
                id,
                spec,
                cancel,
            },
        host,
        guard,
        permit_guard,
    } = task;

    if cancel.is_cancelled() {
        return;
    }

    // Expand full tile URL outside `PipelineState`'s mutex.
    let url = match spec.resolve_url(id) {
        Ok(url) => url,
        Err(err) => {
            drop(permit_guard);
            let safe_msg = spec.redact_secrets(&err.to_string());
            record_terminal_failure(&ctx, &cancel, guard, Some(safe_msg));
            return;
        }
    };

    let fetch_res = download_capped(&ctx.client, &url, ctx.max_tile_bytes, &cancel).await;

    // Release the per-host HTTP permit BEFORE submitting CPU decode / disk put.
    drop(permit_guard);

    if cancel.is_cancelled() {
        return;
    }

    match fetch_res {
        Ok(FetchOutcome::Payload(bytes)) => {
            handle_http_payload(&ctx, &spec, key, id, bytes, &cancel, guard).await;
        }
        Ok(FetchOutcome::Empty204) => {
            // Valid empty/nodata tile: suppress infinite refetch without
            // bumping `failed_count` or displaying an error badge.
            finish_with_wakeup(&ctx, guard, |st, key, generation| {
                st.mark_empty_204(key, generation)
            });
        }
        Ok(FetchOutcome::Cancelled) => {}
        Err(FetchFailure::Terminal(msg)) => {
            let safe_msg = spec.redact_secrets(&msg);
            tracing::debug!(?key, error = %safe_msg, "Terminal HTTP tile failure");
            record_terminal_failure(&ctx, &cancel, guard, Some(safe_msg));
        }
        Err(FetchFailure::Transient(err)) => {
            handle_http_transient_failure(&ctx, &spec, &host, key, &cancel, guard, err);
        }
    }
}

fn record_terminal_failure(
    ctx: &PipelineContext,
    cancel: &CancelToken,
    guard: InFlightGuard,
    error_msg: Option<String>,
) {
    if cancel.is_cancelled() {
        return;
    }
    finish_with_wakeup(ctx, guard, |st, key, generation| {
        st.mark_terminal_failure(key, generation, error_msg)
    });
}
