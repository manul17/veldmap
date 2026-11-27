//! Two-stage disk-lookup and per-host HTTP dispatcher loop for
//! [`crate::LiveTilePipeline`].

use std::sync::Arc;
use std::time::Instant;

use veldmap_core::sync::lock_unpoisoned;

use super::PipelineContext;
use super::disk::DiskCacheGate;
use super::guards::{DiskSlotGuard, InFlightGuard, PermitGuard};
use super::workers::{DiskTask, HttpTask, run_disk_lookup, run_http_fetch};

/// Maximum concurrent blocking disk-cache lookup tasks so cached tiles stay
/// responsive and never starve behind slow HTTP hosts.
const MAX_CONCURRENT_DISK_LOOKUPS: usize = 4;

struct DispatchBatch {
    disk_tasks: Vec<DiskTask>,
    http_tasks: Vec<HttpTask>,
    earliest_wakeup: Option<Instant>,
}

fn plan_dispatch_batch(ctx: &PipelineContext, now: Instant) -> DispatchBatch {
    let planned = {
        let mut st = lock_unpoisoned(&ctx.shared.state);
        st.plan_dispatch(
            MAX_CONCURRENT_DISK_LOOKUPS,
            &ctx.shared.shutdown,
            now,
            |host| ctx.shared.host_limiters.try_acquire_permit(host, now),
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
