//! Two-stage disk-lookup and per-host HTTP dispatcher loop for
//! [`super::LiveTilePipeline`].

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::OwnedSemaphorePermit;
use veldmap_core::sync::{CancelToken, lock_unpoisoned};

use crate::basemap::LiveBasemapSpec;
use crate::http::{HostLimiters, PermitDenied};

use super::PipelineContext;
use super::disk::DiskCacheGate;
use super::guards::{DiskSlotGuard, InFlightGuard, PermitGuard};
use super::state::PipelineState;
use super::workers::{DiskTask, HttpTask, TileTarget, run_disk_lookup, run_http_fetch};

/// Maximum concurrent blocking disk-cache lookup tasks so cached tiles stay
/// responsive and never starve behind slow HTTP hosts.
const MAX_CONCURRENT_DISK_LOOKUPS: usize = 4;

/// Raw disk-lookup task planned under `PipelineState`'s mutex, to be wrapped
/// in RAII guards immediately upon unlocking `state` (preventing
/// Drop-on-unwind self-deadlock).
struct PlannedDiskTask {
    target: TileTarget,
    generation: u64,
}

/// Raw HTTP fetch task planned under `PipelineState`'s mutex, to be wrapped
/// in RAII guards immediately upon unlocking `state` (preventing
/// Drop-on-unwind self-deadlock).
struct PlannedHttpTask {
    target: TileTarget,
    generation: u64,
    host: String,
    permit: OwnedSemaphorePermit,
}

/// Dispatch batch planned under `PipelineState`'s mutex in
/// `state -> host_limiters` lock order.
struct PlannedDispatch {
    disk_tasks: Vec<PlannedDiskTask>,
    http_tasks: Vec<PlannedHttpTask>,
    earliest_wakeup: Option<Instant>,
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
    state: &mut PipelineState,
    spec: &Arc<LiveBasemapSpec>,
    has_disk_cache: bool,
    max_concurrent_disk: usize,
    shutdown: &CancelToken,
) -> Vec<PlannedDiskTask> {
    let active_source = spec.source_id();
    if !has_disk_cache {
        state.drain_disk_to_http(active_source);
        return Vec::new();
    }

    let mut tasks = Vec::new();
    while let Some((key, id, generation, cancel)) =
        state.pop_next_disk(active_source, max_concurrent_disk, shutdown)
    {
        tasks.push(PlannedDiskTask {
            target: TileTarget {
                key,
                id,
                spec: Arc::clone(spec),
                cancel,
            },
            generation,
        });
    }
    tasks
}

fn plan_http_tasks(
    state: &mut PipelineState,
    spec: &Arc<LiveBasemapSpec>,
    host_limiters: &HostLimiters,
    shutdown: &CancelToken,
    now: Instant,
    earliest_wakeup: &mut Option<Instant>,
) -> Vec<PlannedHttpTask> {
    let active_source = spec.source_id();
    let total_hosts = spec.host_count();
    let mut denied_hosts: HashSet<&str> = HashSet::with_capacity(total_hosts);
    let mut deferred_http = Vec::new();
    let mut tasks = Vec::new();

    while let Some(key) = state.pop_next_http(active_source) {
        let Some(id) = key.quad_id() else {
            continue;
        };
        let host = spec.host_for(id);
        if denied_hosts.contains(host) {
            deferred_http.push(key);
            continue;
        }
        match host_limiters.try_acquire_permit(host, now) {
            Ok(permit) => {
                let (generation, cancel) = state.register_http_in_flight(key, shutdown);
                tasks.push(PlannedHttpTask {
                    target: TileTarget {
                        key,
                        id,
                        spec: Arc::clone(spec),
                        cancel,
                    },
                    generation,
                    host: host.to_owned(),
                    permit,
                });
            }
            Err(denied) => {
                if let PermitDenied::Cooldown(until) = denied {
                    *earliest_wakeup = min_instant(*earliest_wakeup, until);
                }
                deferred_http.push(key);
                denied_hosts.insert(host);
                if denied_hosts.len() >= total_hosts {
                    break;
                }
            }
        }
    }

    state.push_deferred_http(deferred_http);
    tasks
}

/// Prune expired backoffs and plan the next disk + HTTP dispatch batch
/// while preserving `state -> host_limiters` lock order.
fn plan_dispatch(
    state: &mut PipelineState,
    has_disk_cache: bool,
    max_concurrent_disk: usize,
    host_limiters: &HostLimiters,
    shutdown: &CancelToken,
    now: Instant,
) -> PlannedDispatch {
    let mut earliest_wakeup: Option<Instant> = None;
    let (disk_tasks, http_tasks) = if let Some(spec) = state.prepare_dispatch(now) {
        let disk_tasks =
            plan_disk_tasks(state, &spec, has_disk_cache, max_concurrent_disk, shutdown);
        let http_tasks = plan_http_tasks(
            state,
            &spec,
            host_limiters,
            shutdown,
            now,
            &mut earliest_wakeup,
        );
        (disk_tasks, http_tasks)
    } else {
        (Vec::new(), Vec::new())
    };

    if let Some(backoff_at) = state.earliest_wanted_backoff(now) {
        earliest_wakeup = min_instant(earliest_wakeup, backoff_at);
    }

    PlannedDispatch {
        disk_tasks,
        http_tasks,
        earliest_wakeup,
    }
}

fn plan_dispatch_batch(ctx: &PipelineContext, now: Instant) -> DispatchBatch {
    let planned = {
        let mut st = lock_unpoisoned(&ctx.shared.state);
        plan_dispatch(
            &mut st,
            ctx.disk_cache.is_some(),
            MAX_CONCURRENT_DISK_LOOKUPS,
            &ctx.shared.host_limiters,
            &ctx.shared.shutdown,
            now,
        )
    };

    // Construct RAII guards immediately upon unlocking `state` so no unguarded
    // `in_flight` registration, `disk_in_flight` slot, or host permit ever
    // leaves `plan_dispatch_batch`.
    let disk_tasks = planned
        .disk_tasks
        .into_iter()
        .map(|task| {
            let key = task.target.key;
            DiskTask {
                target: task.target,
                guard: InFlightGuard::new(Arc::clone(&ctx.shared), key, task.generation),
                disk_slot: DiskSlotGuard::new(Arc::clone(&ctx.shared)),
            }
        })
        .collect();

    let http_tasks = planned
        .http_tasks
        .into_iter()
        .map(|task| {
            let key = task.target.key;
            HttpTask {
                target: task.target,
                host: task.host,
                guard: InFlightGuard::new(Arc::clone(&ctx.shared), key, task.generation),
                permit_guard: PermitGuard::new(task.permit, Arc::clone(&ctx.shared)),
            }
        })
        .collect();

    DispatchBatch {
        disk_tasks,
        http_tasks,
        earliest_wakeup: planned.earliest_wakeup,
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
    let task_ctx = Arc::clone(ctx);
    let task_gate = gate.clone();
    tokio::spawn(async move {
        run_disk_lookup(task_ctx, task_gate, task).await;
    });
}

fn spawn_http_task(ctx: &Arc<PipelineContext>, task: HttpTask) {
    let task_ctx = Arc::clone(ctx);
    tokio::spawn(async move {
        run_http_fetch(task_ctx, task).await;
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

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use veldmap_core::tile::{PlanetaryScheme, QuadTileId};

    use super::*;

    #[test]
    fn plan_dispatch_carries_quad_id_and_defers_cooled_down_hosts() {
        let spec = LiveBasemapSpec::eox_sentinel2_cloudless();
        let mut st = PipelineState::default();
        let now = Instant::now();
        let shutdown = CancelToken::new();
        let limiters = HostLimiters::new(2);

        let id0 = QuadTileId {
            scheme: PlanetaryScheme::WebMercator,
            level: 1,
            x: 0,
            y: 0,
        };
        let id1 = QuadTileId {
            scheme: PlanetaryScheme::WebMercator,
            level: 1,
            x: 1,
            y: 0,
        };
        let k0 = spec.tile_key(id0);
        let k1 = spec.tile_key(id1);
        st.set_active_spec(Some(spec.clone()));
        st.reconcile_wanted(&HashSet::from([k0, k1]), &|_| false, now);

        // With disk cache enabled, disk tasks are planned with unpacked QuadTileId.
        let disk_batch = plan_dispatch(&mut st, true, 1, &limiters, &shutdown, now);
        assert_eq!(disk_batch.disk_tasks.len(), 1);
        assert_eq!(disk_batch.disk_tasks[0].target.key, k0);
        assert_eq!(disk_batch.disk_tasks[0].target.id, id0);
        assert!(disk_batch.http_tasks.is_empty());

        // Without disk cache, remaining pending.disk drains to HTTP and respects host cooldown.
        let cooldown_until = now + Duration::from_secs(10);
        limiters.record_cooldown(spec.host_for(id1), cooldown_until);
        let deferred = plan_dispatch(&mut st, false, 4, &limiters, &shutdown, now);
        assert!(deferred.disk_tasks.is_empty());
        assert!(deferred.http_tasks.is_empty());
        assert_eq!(deferred.earliest_wakeup, Some(cooldown_until));
        assert!(st.is_http_pending(&k1));

        // After cooldown expires, HTTP task is planned with exact QuadTileId.
        let http_batch = plan_dispatch(&mut st, false, 4, &limiters, &shutdown, cooldown_until);
        assert_eq!(http_batch.http_tasks.len(), 1);
        assert_eq!(http_batch.http_tasks[0].target.key, k1);
        assert_eq!(http_batch.http_tasks[0].target.id, id1);
    }
}
