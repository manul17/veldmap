//! Two-stage disk-lookup and per-host HTTP dispatcher loop for
//! [`super::LiveTilePipeline`].

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::OwnedSemaphorePermit;
use veldmap_core::io::{CancelToken, lock_unpoisoned};
use veldmap_core::tile::{PlanetaryScheme, TileKey};

use crate::basemap::LiveBasemapSpec;
use crate::http::PermitDenied;
use crate::redact::extract_host;

use super::guards::{DiskSlotGuard, InFlightGuard, PermitGuard};
use super::state::{FailureState, PipelineState};
use super::workers::{run_disk_lookup, run_http_fetch};
use super::{MAX_CONCURRENT_DISK_LOOKUPS, PipelineContext};

pub(super) struct HttpTask {
    pub(super) key: TileKey,
    pub(super) generation: u64,
    pub(super) spec: Arc<LiveBasemapSpec>,
    pub(super) url: String,
    pub(super) host: String,
    pub(super) permit: OwnedSemaphorePermit,
    pub(super) cancel: CancelToken,
}

pub(super) async fn dispatcher_loop(ctx: Arc<PipelineContext>) {
    loop {
        if ctx.shutdown.is_cancelled() {
            break;
        }

        let now = Instant::now();
        let mut disk_tasks: Vec<(TileKey, u64, Arc<LiveBasemapSpec>, CancelToken)> = Vec::new();
        let mut http_tasks: Vec<HttpTask> = Vec::new();
        let mut earliest_wakeup: Option<Instant> = None;

        {
            let mut st = lock_unpoisoned(&ctx.state);
            st.prune_expired_backoffs(now);

            if let Some(spec) = st.active_spec.clone() {
                let active_source = spec.source_id();
                // Stage 1: Pop from `pending_disk` (or batch-move straight to
                // `pending_http` when no disk cache is configured).
                if ctx.disk_cache.is_none() {
                    if !st.pending_disk.is_empty() {
                        let drained: Vec<TileKey> = st.pending_disk.drain(..).collect();
                        let mut existing: HashSet<TileKey> =
                            st.pending_http.iter().copied().collect();
                        let mut added = false;
                        for key in drained {
                            if key.source == active_source
                                && !st.in_flight.contains_key(&key)
                                && !st.failures.contains_key(&key)
                                && existing.insert(key)
                            {
                                st.pending_http.push(key);
                                added = true;
                            }
                        }
                        if added {
                            PipelineState::sort_queue_coarsest_at_end(&mut st.pending_http);
                        }
                    }
                } else {
                    while st.disk_in_flight < MAX_CONCURRENT_DISK_LOOKUPS {
                        let Some(key) = st.pending_disk.pop() else {
                            break;
                        };
                        if key.source != active_source
                            || st.in_flight.contains_key(&key)
                            || st.failures.contains_key(&key)
                        {
                            continue;
                        }
                        st.next_gen = st.next_gen.wrapping_add(1);
                        let gen_id = st.next_gen;
                        let tile_cancel = ctx.shutdown.child_token();
                        st.disk_in_flight += 1;
                        st.in_flight.insert(key, (gen_id, tile_cancel.clone()));
                        disk_tasks.push((key, gen_id, Arc::clone(&spec), tile_cancel));
                    }
                }

                // Stage 2: Pop from `pending_http` ONLY when a per-host permit
                // is immediately available and the host is not cooling down.
                // Short-circuits immediately once a single-host spec (no `{s}`)
                // is saturated or cooling down so remaining queued tiles are
                // preserved without redundant URL resolution or re-sorting.
                let multi_host = spec.uses_subdomains();
                let mut deferred_http = Vec::new();
                while let Some(key) = st.pending_http.pop() {
                    if key.source != active_source
                        || st.in_flight.contains_key(&key)
                        || st.failures.contains_key(&key)
                    {
                        continue;
                    }
                    let Some(id) = key.quad_id() else {
                        continue;
                    };
                    if matches!(id.scheme, PlanetaryScheme::PolarCap { .. }) {
                        continue;
                    }
                    let Ok(url) = spec.resolve_url(id) else {
                        st.insert_failure(key, FailureState::Terminal, now);
                        continue;
                    };
                    let host = extract_host(&url);
                    match ctx.host_limiters.try_acquire_permit(&host, now) {
                        Ok(permit) => {
                            st.next_gen = st.next_gen.wrapping_add(1);
                            let gen_id = st.next_gen;
                            let tile_cancel = ctx.shutdown.child_token();
                            st.in_flight.insert(key, (gen_id, tile_cancel.clone()));
                            http_tasks.push(HttpTask {
                                key,
                                generation: gen_id,
                                spec: Arc::clone(&spec),
                                url,
                                host,
                                permit,
                                cancel: tile_cancel,
                            });
                        }
                        Err(PermitDenied::Cooldown(until)) => {
                            earliest_wakeup = Some(match earliest_wakeup {
                                Some(cur) if cur < until => cur,
                                _ => until,
                            });
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
                if !deferred_http.is_empty() {
                    // `deferred_http` was popped coarsest-first from the back of
                    // `st.pending_http`; reversing restores ascending-priority
                    // order (coarsest at the end) cleanly without a full sort
                    // for single-host short-circuits, and sorting when mixed.
                    if multi_host {
                        st.pending_http.extend(deferred_http);
                        PipelineState::sort_queue_coarsest_at_end(&mut st.pending_http);
                    } else {
                        st.pending_http.extend(deferred_http.into_iter().rev());
                    }
                }
            } else {
                st.pending_disk.clear();
                st.pending_http.clear();
            }

            if let Some(backoff_at) = st.earliest_wanted_backoff(now) {
                earliest_wakeup = Some(match earliest_wakeup {
                    Some(cur) if cur < backoff_at => cur,
                    _ => backoff_at,
                });
            }
        }

        if disk_tasks.is_empty() && http_tasks.is_empty() {
            if let Some(wake_at) = earliest_wakeup {
                let deadline = tokio::time::Instant::from_std(wake_at);
                tokio::select! {
                    _ = ctx.shutdown.cancelled() => break,
                    _ = ctx.notify.notified() => continue,
                    _ = tokio::time::sleep_until(deadline) => continue,
                }
            } else {
                tokio::select! {
                    _ = ctx.shutdown.cancelled() => break,
                    _ = ctx.notify.notified() => continue,
                }
            }
        }

        for (key, generation, spec, tile_cancel) in disk_tasks {
            let task_ctx = Arc::clone(&ctx);
            tokio::spawn(async move {
                let guard = InFlightGuard::new(
                    Arc::clone(&task_ctx.state),
                    key,
                    generation,
                    Arc::clone(&task_ctx.notify),
                );
                let disk_slot = DiskSlotGuard {
                    state: Arc::clone(&task_ctx.state),
                    notify: Arc::clone(&task_ctx.notify),
                };
                run_disk_lookup(task_ctx, key, spec, tile_cancel, guard, disk_slot).await;
            });
        }

        for mut task in http_tasks {
            let task_ctx = Arc::clone(&ctx);
            tokio::spawn(async move {
                let guard = InFlightGuard::new(
                    Arc::clone(&task_ctx.state),
                    task.key,
                    task.generation,
                    Arc::clone(&task_ctx.notify),
                );
                // Extract the permit into `PermitGuard` so HTTP concurrency is
                // released immediately after download completes.
                let permit_guard = PermitGuard {
                    permit: Some(task.permit),
                    notify: Arc::clone(&task_ctx.notify),
                };
                // Reconstruct without the moved permit via a lightweight payload.
                run_http_fetch(
                    task_ctx,
                    HttpFetchInput {
                        key: task.key,
                        spec: task.spec,
                        url: std::mem::take(&mut task.url),
                        host: std::mem::take(&mut task.host),
                        cancel: task.cancel,
                    },
                    guard,
                    permit_guard,
                )
                .await;
            });
        }
    }
}

pub(super) struct HttpFetchInput {
    pub(super) key: TileKey,
    pub(super) spec: Arc<LiveBasemapSpec>,
    pub(super) url: String,
    pub(super) host: String,
    pub(super) cancel: CancelToken,
}
