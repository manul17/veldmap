//! Two-stage disk-lookup and per-host HTTP dispatcher loop for
//! [`super::LiveTilePipeline`].

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::OwnedSemaphorePermit;
use veldmap_core::sync::{CancelToken, lock_unpoisoned};
use veldmap_core::tile::TileKey;

use crate::basemap::LiveBasemapSpec;
use crate::http::PermitDenied;

use super::PipelineContext;
use super::disk::DiskCacheGate;
use super::failures::FailureMarker;
use super::guards::{DiskSlotGuard, InFlightGuard, PermitGuard};
use super::state::PipelineState;
use super::workers::{run_disk_lookup, run_http_fetch};

/// Maximum concurrent blocking disk-cache lookup tasks so cached tiles stay
/// responsive and never starve behind slow HTTP hosts.
const MAX_CONCURRENT_DISK_LOOKUPS: usize = 4;

pub(super) struct HttpFetchInput {
    pub(super) key: TileKey,
    pub(super) spec: Arc<LiveBasemapSpec>,
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
    while let Some((key, generation, cancel)) = st.pop_next_disk(
        active_source,
        MAX_CONCURRENT_DISK_LOOKUPS,
        &ctx.shared.shutdown,
    ) {
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
    let total_hosts = spec.host_count();
    let mut denied_hosts: HashSet<&str> = HashSet::with_capacity(total_hosts);
    let mut deferred_http = Vec::new();
    let mut tasks = Vec::new();

    while let Some(key) = st.pop_next_http(active_source) {
        let Some(id) = key.quad_id() else {
            continue;
        };
        if spec.validate_tile(id).is_err() {
            st.insert_failure(key, FailureMarker::Terminal);
            continue;
        }
        let host = spec.host_for(id);
        if denied_hosts.contains(host) {
            deferred_http.push(key);
            continue;
        }
        match ctx.shared.host_limiters.try_acquire_permit(host, now) {
            Ok(permit) => {
                let (generation, cancel) = st.register_http_in_flight(key, &ctx.shared.shutdown);
                tasks.push(HttpTask {
                    input: HttpFetchInput {
                        key,
                        spec: Arc::clone(spec),
                        host: host.to_owned(),
                        cancel,
                    },
                    generation,
                    permit,
                });
            }
            Err(PermitDenied::Cooldown(until)) => {
                *earliest_wakeup = min_instant(*earliest_wakeup, until);
                deferred_http.push(key);
                denied_hosts.insert(host);
                if denied_hosts.len() >= total_hosts {
                    break;
                }
            }
            Err(PermitDenied::Saturated) => {
                deferred_http.push(key);
                denied_hosts.insert(host);
                if denied_hosts.len() >= total_hosts {
                    break;
                }
            }
        }
    }

    st.push_deferred_http(deferred_http);
    tasks
}

fn plan_dispatch_batch(ctx: &PipelineContext, now: Instant) -> DispatchBatch {
    let mut st = lock_unpoisoned(&ctx.shared.state);
    let _ = st.prune_expired_backoffs(now);

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
            _ = ctx.shared.shutdown.cancelled() => false,
            _ = ctx.shared.notify.notified() => true,
            _ = tokio::time::sleep_until(deadline) => true,
        }
    } else {
        tokio::select! {
            _ = ctx.shared.shutdown.cancelled() => false,
            _ = ctx.shared.notify.notified() => true,
        }
    }
}

fn spawn_disk_task(ctx: &Arc<PipelineContext>, gate: &DiskCacheGate, task: DiskTask) {
    // Construct RAII guards synchronously BEFORE `tokio::spawn` (after
    // `PipelineState`'s mutex has been dropped) so even a task dropped prior to
    // its first poll cleanly releases its slot and `in_flight` registration.
    let guard = InFlightGuard::new(Arc::clone(&ctx.shared), task.key, task.generation);
    let disk_slot = DiskSlotGuard::new(Arc::clone(&ctx.shared));
    let task_ctx = Arc::clone(ctx);
    let task_gate = gate.clone();
    tokio::spawn(async move {
        run_disk_lookup(
            task_ctx,
            task_gate,
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
    let guard = InFlightGuard::new(Arc::clone(&ctx.shared), task.input.key, task.generation);
    let permit_guard = PermitGuard::new(task.permit, Arc::clone(&ctx.shared));
    let task_ctx = Arc::clone(ctx);
    tokio::spawn(async move {
        run_http_fetch(task_ctx, task.input, guard, permit_guard).await;
    });
}

pub(super) async fn dispatcher_loop(ctx: Arc<PipelineContext>) {
    while !ctx.shared.shutdown.is_cancelled() {
        let batch = plan_dispatch_batch(&ctx, Instant::now());

        if batch.disk_tasks.is_empty() && batch.http_tasks.is_empty() {
            if !wait_for_dispatch_event(&ctx, batch.earliest_wakeup).await {
                break;
            }
            continue;
        }

        if let Some(gate) = &ctx.disk_cache {
            for task in batch.disk_tasks {
                spawn_disk_task(&ctx, gate, task);
            }
        }

        for task in batch.http_tasks {
            spawn_http_task(&ctx, task);
        }
    }
}
