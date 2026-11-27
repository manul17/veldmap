//! Atomic two-stage (disk-lookup + per-host HTTP) dispatch planning for
//! [`super::PipelineState`].

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use veldmap_core::sync::CancelToken;
use veldmap_core::tile::{QuadTileId, TileKey};

use crate::basemap::LiveBasemapSpec;
use crate::limiter::PermitDenied;

use super::{PipelineState, is_wanted_or_root};

/// Shared tile metadata passed from dispatch planning to disk/HTTP workers.
pub(in crate::fetcher) struct TileTarget {
    pub(in crate::fetcher) key: TileKey,
    pub(in crate::fetcher) id: QuadTileId,
    pub(in crate::fetcher) spec: Arc<LiveBasemapSpec>,
    pub(in crate::fetcher) cancel: CancelToken,
}

/// Raw disk-lookup task planned under `PipelineState`'s mutex, to be wrapped
/// in RAII guards immediately upon unlocking `state` (preventing
/// Drop-on-unwind self-deadlock).
pub(in crate::fetcher) struct PlannedDiskTask {
    pub(in crate::fetcher) target: TileTarget,
    pub(in crate::fetcher) generation: u64,
}

/// Raw HTTP fetch task planned under `PipelineState`'s mutex, to be wrapped
/// in RAII guards immediately upon unlocking `state` (preventing
/// Drop-on-unwind self-deadlock).
pub(in crate::fetcher) struct PlannedHttpTask<P> {
    pub(in crate::fetcher) target: TileTarget,
    pub(in crate::fetcher) generation: u64,
    pub(in crate::fetcher) host: String,
    pub(in crate::fetcher) permit: P,
}

/// Dispatch batch planned atomically under `PipelineState`'s mutex.
pub(in crate::fetcher) struct PlannedDispatch<P> {
    pub(in crate::fetcher) disk_tasks: Vec<PlannedDiskTask>,
    pub(in crate::fetcher) http_tasks: Vec<PlannedHttpTask<P>>,
    pub(in crate::fetcher) earliest_wakeup: Option<Instant>,
}

fn min_instant(current: Option<Instant>, candidate: Instant) -> Option<Instant> {
    Some(match current {
        Some(cur) if cur < candidate => cur,
        _ => candidate,
    })
}

impl PipelineState {
    /// Prune expired backoff entries and atomically plan the next disk-lookup
    /// and HTTP-fetch dispatch batch, enforcing `pending -> in_flight`
    /// transitions and `key.quad_id()` filtering inside one state-machine step.
    pub(in crate::fetcher) fn plan_dispatch<P>(
        &mut self,
        max_concurrent_disk: usize,
        shutdown: &CancelToken,
        now: Instant,
        mut try_acquire_permit: impl FnMut(&str) -> Result<P, PermitDenied>,
    ) -> PlannedDispatch<P> {
        self.prune_expired_backoffs(now);
        let Some(spec) = self.active_spec.clone() else {
            self.pending.clear();
            return PlannedDispatch {
                disk_tasks: Vec::new(),
                http_tasks: Vec::new(),
                earliest_wakeup: None,
            };
        };

        let mut earliest_wakeup = None;
        let disk_tasks = self.plan_disk_tasks(&spec, max_concurrent_disk, shutdown);
        let http_tasks = self.plan_http_tasks(
            &spec,
            shutdown,
            &mut try_acquire_permit,
            &mut earliest_wakeup,
        );

        let active_source = spec.source_id();
        if let Some(backoff_at) = self.failures.earliest_wanted_backoff(now, |k| {
            is_wanted_or_root(&self.last_wanted, k, active_source)
        }) {
            earliest_wakeup = min_instant(earliest_wakeup, backoff_at);
        }

        PlannedDispatch {
            disk_tasks,
            http_tasks,
            earliest_wakeup,
        }
    }

    fn plan_disk_tasks(
        &mut self,
        spec: &Arc<LiveBasemapSpec>,
        max_concurrent_disk: usize,
        shutdown: &CancelToken,
    ) -> Vec<PlannedDiskTask> {
        if !self.has_disk_cache {
            return Vec::new();
        }
        let active_source = spec.source_id();
        let mut tasks = Vec::new();
        while self.disk_in_flight < max_concurrent_disk {
            let Some(key) = self.pending.pop_disk() else {
                break;
            };
            if !self.is_queue_eligible(&key, active_source) {
                continue;
            }
            let Some(id) = key.quad_id() else {
                continue;
            };
            let (generation, cancel) = self.in_flight.register(key, shutdown);
            self.disk_in_flight += 1;
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

    fn plan_http_tasks<P>(
        &mut self,
        spec: &Arc<LiveBasemapSpec>,
        shutdown: &CancelToken,
        try_acquire_permit: &mut impl FnMut(&str) -> Result<P, PermitDenied>,
        earliest_wakeup: &mut Option<Instant>,
    ) -> Vec<PlannedHttpTask<P>> {
        let active_source = spec.source_id();
        let total_hosts = spec.host_count();
        let mut denied_hosts: HashSet<&str> = HashSet::with_capacity(total_hosts);
        let mut deferred_http = Vec::new();
        let mut tasks = Vec::new();

        while let Some(key) = self.pending.pop_http() {
            if !self.is_queue_eligible(&key, active_source) {
                continue;
            }
            let Some(id) = key.quad_id() else {
                continue;
            };
            let host = spec.host_for(id);
            if denied_hosts.contains(host) {
                deferred_http.push(key);
                continue;
            }
            match try_acquire_permit(host) {
                Ok(permit) => {
                    let (generation, cancel) = self.in_flight.register(key, shutdown);
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

        self.pending.push_http(deferred_http);
        tasks
    }
}
