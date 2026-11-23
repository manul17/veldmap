//! Two-stage disk-lookup and per-host HTTP dispatcher loop for
//! [`super::LiveTilePipeline`].

use std::sync::Arc;
use std::time::Instant;

use tokio::sync::OwnedSemaphorePermit;
use veldmap_core::io::{CancelToken, lock_unpoisoned};
use veldmap_core::tile::TileKey;

use crate::TileBlobStore;
use crate::basemap::LiveBasemapSpec;
use crate::http::PermitDenied;
use crate::redact::extract_host;

use super::PipelineContext;
use super::guards::{DiskSlotGuard, InFlightGuard, PermitGuard};
use super::state::{FailureState, PipelineState};
use super::workers::{run_disk_lookup, run_http_fetch};

/// Maximum concurrent blocking disk-cache lookup tasks so cached tiles stay
/// responsive and never starve behind slow HTTP hosts.
const MAX_CONCURRENT_DISK_LOOKUPS: usize = 4;

pub(super) struct HttpFetchInput {
    pub(super) key: TileKey,
    pub(super) spec: Arc<LiveBasemapSpec>,
    pub(super) url: String,
    pub(super) host: String,
    pub(super) cancel: CancelToken,
}

struct DiskTask {
    key: TileKey,
    generation: u64,
    spec: Arc<LiveBasemapSpec>,
    cancel: CancelToken,
}

struct HttpTask {
    input: HttpFetchInput,
    generation: u64,
    permit: OwnedSemaphorePermit,
}

struct DispatchBatch {
    disk_tasks: Vec<DiskTask>,
    http_tasks: Vec<HttpTask>,
    earliest_wakeup: Option<Instant>,
}

fn min_instant(current: Option<Instant>, candidate: Instant) -> Option<Instant> {
    Some(match current {
        Some(cur) if cur < candidate => cur,
        _ => candidate,
    })
}

fn plan_disk_tasks(
    ctx: &PipelineContext,
    st: &mut PipelineState,
    spec: &Arc<LiveBasemapSpec>,
) -> Vec<DiskTask> {
    let active_source = spec.source_id();
    if ctx.disk_cache.is_none() {
        st.drain_disk_to_http(active_source);
        return Vec::new();
    }

    let mut tasks = Vec::new();
    while let Some((key, generation, cancel)) =
        st.pop_next_disk(active_source, MAX_CONCURRENT_DISK_LOOKUPS, &ctx.shutdown)
    {
        tasks.push(DiskTask {
            key,
            generation,
            spec: Arc::clone(spec),
            cancel,
        });
    }
    tasks
}

fn plan_http_tasks(
    ctx: &PipelineContext,
    st: &mut PipelineState,
    spec: &Arc<LiveBasemapSpec>,
    now: Instant,
    earliest_wakeup: &mut Option<Instant>,
) -> Vec<HttpTask> {
    let active_source = spec.source_id();
    let multi_host = spec.uses_subdomains();
    let mut deferred_http = Vec::new();
    let mut tasks = Vec::new();

    while let Some(key) = st.pop_next_http(active_source) {
        let Some(id) = key.quad_id() else {
            continue;
        };
        let Ok(url) = spec.resolve_url(id) else {
            st.insert_failure(key, FailureState::Terminal, now);
            continue;
        };
        let host = extract_host(&url);
        match ctx.host_limiters.try_acquire_permit(&host, now) {
            Ok(permit) => {
                let (generation, cancel) = st.register_http_in_flight(key, &ctx.shutdown);
                tasks.push(HttpTask {
                    input: HttpFetchInput {
                        key,
                        spec: Arc::clone(spec),
                        url,
                        host,
                        cancel,
                    },
                    generation,
                    permit,
                });
            }
            Err(PermitDenied::Cooldown(until)) => {
                *earliest_wakeup = min_instant(*earliest_wakeup, until);
                deferred_http.push(key);
                if !multi_host {
                    break;
                }
            }
            Err(PermitDenied::Saturated) => {
                deferred_http.push(key);
                if !multi_host {
                    break;
                }
            }
        }
    }

    st.restore_deferred_http(deferred_http, multi_host);
    tasks
}

fn plan_dispatch_batch(ctx: &PipelineContext, now: Instant) -> DispatchBatch {
    let mut st = lock_unpoisoned(&ctx.state);
    st.prune_expired_backoffs(now);

    let mut earliest_wakeup: Option<Instant> = None;
    let (disk_tasks, http_tasks) = if let Some(spec) = st.active_spec() {
        let disk_tasks = plan_disk_tasks(ctx, &mut st, &spec);
        let http_tasks = plan_http_tasks(ctx, &mut st, &spec, now, &mut earliest_wakeup);
        (disk_tasks, http_tasks)
    } else {
        st.clear_pending_queues();
        (Vec::new(), Vec::new())
    };

    if let Some(backoff_at) = st.earliest_wanted_backoff(now) {
        earliest_wakeup = min_instant(earliest_wakeup, backoff_at);
    }

    DispatchBatch {
        disk_tasks,
        http_tasks,
        earliest_wakeup,
    }
}

async fn wait_for_dispatch_event(ctx: &PipelineContext, wake_at: Option<Instant>) -> bool {
    if let Some(wake_at) = wake_at {
        let deadline = tokio::time::Instant::from_std(wake_at);
        tokio::select! {
            _ = ctx.shutdown.cancelled() => false,
            _ = ctx.notify.notified() => true,
            _ = tokio::time::sleep_until(deadline) => true,
        }
    } else {
        tokio::select! {
            _ = ctx.shutdown.cancelled() => false,
            _ = ctx.notify.notified() => true,
        }
    }
}

fn spawn_disk_task(ctx: &Arc<PipelineContext>, cache: &Arc<dyn TileBlobStore>, task: DiskTask) {
    let task_ctx = Arc::clone(ctx);
    let task_cache = Arc::clone(cache);
    tokio::spawn(async move {
        let guard = InFlightGuard::new(
            Arc::clone(&task_ctx.state),
            task.key,
            task.generation,
            Arc::clone(&task_ctx.notify),
        );
        let disk_slot = DiskSlotGuard {
            state: Arc::clone(&task_ctx.state),
            notify: Arc::clone(&task_ctx.notify),
        };
        run_disk_lookup(
            task_ctx,
            task_cache,
            task.key,
            task.spec,
            task.cancel,
            guard,
            disk_slot,
        )
        .await;
    });
}

fn spawn_http_task(ctx: &Arc<PipelineContext>, task: HttpTask) {
    let task_ctx = Arc::clone(ctx);
    tokio::spawn(async move {
        let guard = InFlightGuard::new(
            Arc::clone(&task_ctx.state),
            task.input.key,
            task.generation,
            Arc::clone(&task_ctx.notify),
        );
        // Release HTTP concurrency immediately after download completes.
        let permit_guard = PermitGuard {
            permit: Some(task.permit),
            notify: Arc::clone(&task_ctx.notify),
        };
        run_http_fetch(task_ctx, task.input, guard, permit_guard).await;
    });
}

pub(super) async fn dispatcher_loop(ctx: Arc<PipelineContext>) {
    while !ctx.shutdown.is_cancelled() {
        let batch = plan_dispatch_batch(&ctx, Instant::now());

        if batch.disk_tasks.is_empty() && batch.http_tasks.is_empty() {
            if !wait_for_dispatch_event(&ctx, batch.earliest_wakeup).await {
                break;
            }
            continue;
        }

        if let Some(cache) = &ctx.disk_cache {
            for task in batch.disk_tasks {
                spawn_disk_task(&ctx, cache, task);
            }
        }

        for task in batch.http_tasks {
            spawn_http_task(&ctx, task);
        }
    }
}
