//! Queue, in-flight tracking, and failure/backoff state machine for
//! [`super::LiveTilePipeline`].

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use veldmap_core::io::CancelToken;
use veldmap_core::tile::{SourceId, TileKey};

use crate::basemap::LiveBasemapSpec;

use super::NetworkStatus;

/// Upper bound on retained per-tile failure/empty markers to prevent unbounded
/// memory growth on sparse 404 layers over long sessions.
pub(super) const MAX_FAILURE_ENTRIES: usize = 4096;

#[derive(Clone, Copy, Debug)]
pub(super) enum FailureState {
    Terminal,
    /// HTTP `204 No Content`: suppresses infinite refetch without flagging an error.
    Empty,
    BackoffUntil {
        until: Instant,
        retry_disk: bool,
    },
}

#[derive(Default)]
pub(super) struct PipelineState {
    active_spec: Option<Arc<LiveBasemapSpec>>,
    /// Unready, active-source, contract-valid keys from the most recent
    /// `reconcile_wanted`, used by idle backoff retries and `retry_failed`.
    last_wanted: HashSet<TileKey>,
    /// Coarsest-first queue awaiting fast local disk-cache lookup (stored
    /// reverse-sorted so `pop()` yields the coarsest key in O(1)).
    pending_disk: Vec<TileKey>,
    /// Coarsest-first queue of disk misses awaiting an HTTP per-host slot.
    pending_http: Vec<TileKey>,
    /// Active blocking disk-lookup tasks (bounded by `MAX_CONCURRENT_DISK_LOOKUPS`).
    disk_in_flight: usize,
    /// Tracks active `(generation, CancelToken)` from dispatch all the way
    /// through CPU decode completion so stale guards never clobber a newer
    /// spawn and pan/switch cancels queued decode jobs immediately.
    in_flight: HashMap<TileKey, (u64, CancelToken)>,
    next_gen: u64,
    failures: HashMap<TileKey, FailureState>,
    completed_from_disk: u64,
    completed_from_net: u64,
    failed_count: u64,
    offline: bool,
    last_error: Option<String>,
}

impl PipelineState {
    pub(super) fn sort_queue_coarsest_at_end(queue: &mut [TileKey]) {
        queue.sort_by_key(|k| std::cmp::Reverse(k.coarsest_first_key()));
    }

    fn push_unique_sorted(
        queue: &mut Vec<TileKey>,
        keys: impl IntoIterator<Item = TileKey>,
    ) -> bool {
        let mut existing: HashSet<TileKey> = queue.iter().copied().collect();
        let mut added = false;
        for key in keys {
            if existing.insert(key) {
                queue.push(key);
                added = true;
            }
        }
        if added {
            Self::sort_queue_coarsest_at_end(queue);
        }
        added
    }

    fn prune_expired_backoff_markers(&mut self, now: Instant) {
        self.failures.retain(|_, f| match f {
            FailureState::Terminal | FailureState::Empty => true,
            FailureState::BackoffUntil { until, .. } => *until > now,
        });
    }

    fn is_active_source(&self, source: SourceId) -> bool {
        self.active_spec
            .as_ref()
            .is_some_and(|s| s.source_id() == source)
    }

    fn is_queue_eligible(&self, key: &TileKey, active_source: SourceId) -> bool {
        key.source == active_source
            && !self.in_flight.contains_key(key)
            && !self.failures.contains_key(key)
    }

    pub(super) fn active_spec(&self) -> Option<Arc<LiveBasemapSpec>> {
        self.active_spec.clone()
    }

    pub(super) fn set_active_spec(
        &mut self,
        spec: Option<LiveBasemapSpec>,
    ) -> Option<Vec<CancelToken>> {
        if self.active_spec.as_deref() == spec.as_ref() {
            return None;
        }
        self.active_spec = spec.map(Arc::new);
        self.last_wanted.clear();
        self.pending_disk.clear();
        self.pending_http.clear();
        self.failures.clear();
        self.failed_count = 0;
        self.offline = false;
        self.last_error = None;
        Some(self.in_flight.drain().map(|(_, (_, tok))| tok).collect())
    }

    pub(super) fn reconcile_wanted<F>(
        &mut self,
        wanted: &HashSet<TileKey>,
        is_ready: &F,
        now: Instant,
    ) -> Option<Vec<CancelToken>>
    where
        F: Fn(&TileKey) -> bool,
    {
        let Some(spec) = self.active_spec.clone() else {
            self.last_wanted.clear();
            self.pending_disk.clear();
            self.pending_http.clear();
            return None;
        };
        let active_source = spec.source_id();

        let mut cancelled = Vec::new();
        self.in_flight.retain(|key, (_, tok)| {
            let keep = key.source == active_source
                && !is_ready(key)
                && (wanted.contains(key) || key.is_root_of(active_source));
            if !keep {
                cancelled.push(tok.clone());
            }
            keep
        });

        self.prune_expired_backoff_markers(now);

        self.last_wanted.clear();
        for &key in wanted {
            if key.source == active_source
                && !is_ready(&key)
                && key
                    .quad_id()
                    .is_some_and(|id| spec.validate_tile(id).is_ok())
            {
                self.last_wanted.insert(key);
            }
        }

        // Retain still-wanted / active-root entries already in `pending_http`
        // so disk misses never bounce back to `pending_disk`.
        let mut already_in_http = HashSet::new();
        {
            let Self {
                last_wanted,
                failures,
                in_flight,
                pending_http,
                ..
            } = self;
            pending_http.retain(|key| {
                let keep = key.source == active_source
                    && !is_ready(key)
                    && !in_flight.contains_key(key)
                    && !failures.contains_key(key)
                    && (last_wanted.contains(key) || key.is_root_of(active_source));
                if keep {
                    already_in_http.insert(*key);
                }
                keep
            });
        }

        let mut next_disk = Vec::new();
        for &key in &self.last_wanted {
            if self.in_flight.contains_key(&key)
                || self.failures.contains_key(&key)
                || already_in_http.contains(&key)
            {
                continue;
            }
            next_disk.push(key);
        }
        Self::sort_queue_coarsest_at_end(&mut next_disk);
        Self::sort_queue_coarsest_at_end(&mut self.pending_http);
        self.pending_disk = next_disk;
        Some(cancelled)
    }

    pub(super) fn retry_failed(&mut self) {
        let cleared: Vec<TileKey> = self.failures.drain().map(|(k, _)| k).collect();
        self.failed_count = 0;
        self.offline = false;
        self.last_error = None;

        let Some(spec) = self.active_spec.as_ref() else {
            return;
        };
        let active_source = spec.source_id();
        let queued_http: HashSet<TileKey> = self.pending_http.iter().copied().collect();
        let to_requeue: Vec<TileKey> = cleared
            .into_iter()
            .filter(|key| {
                key.source == active_source
                    && self.last_wanted.contains(key)
                    && !self.in_flight.contains_key(key)
                    && !queued_http.contains(key)
            })
            .collect();
        Self::push_unique_sorted(&mut self.pending_disk, to_requeue);
    }

    pub(super) fn status(&self) -> NetworkStatus {
        NetworkStatus {
            in_flight: self.in_flight.len(),
            queued: self
                .pending_disk
                .len()
                .saturating_add(self.pending_http.len()),
            completed_from_disk: self.completed_from_disk,
            completed_from_net: self.completed_from_net,
            failed_count: self.failed_count,
            offline: self.offline,
            last_error: self.last_error.clone(),
        }
    }

    pub(super) fn signal_shutdown(&mut self) -> Vec<CancelToken> {
        self.last_wanted.clear();
        self.clear_pending_queues();
        self.in_flight.drain().map(|(_, (_, t))| t).collect()
    }

    pub(super) fn clear_pending_queues(&mut self) {
        self.pending_disk.clear();
        self.pending_http.clear();
    }

    pub(super) fn drain_disk_to_http(&mut self, active_source: SourceId) {
        if self.pending_disk.is_empty() {
            return;
        }
        let drained: Vec<TileKey> = self.pending_disk.drain(..).collect();
        let eligible: Vec<TileKey> = drained
            .into_iter()
            .filter(|key| self.is_queue_eligible(key, active_source))
            .collect();
        Self::push_unique_sorted(&mut self.pending_http, eligible);
    }

    pub(super) fn pop_next_disk(
        &mut self,
        active_source: SourceId,
        max_concurrent_disk: usize,
        shutdown: &CancelToken,
    ) -> Option<(TileKey, u64, CancelToken)> {
        if self.disk_in_flight >= max_concurrent_disk {
            return None;
        }
        while let Some(key) = self.pending_disk.pop() {
            if !self.is_queue_eligible(&key, active_source) {
                continue;
            }
            self.next_gen = self.next_gen.wrapping_add(1);
            let gen_id = self.next_gen;
            let tile_cancel = shutdown.child_token();
            self.disk_in_flight += 1;
            self.in_flight.insert(key, (gen_id, tile_cancel.clone()));
            return Some((key, gen_id, tile_cancel));
        }
        None
    }

    pub(super) fn pop_next_http(&mut self, active_source: SourceId) -> Option<TileKey> {
        while let Some(key) = self.pending_http.pop() {
            if self.is_queue_eligible(&key, active_source) {
                return Some(key);
            }
        }
        None
    }

    pub(super) fn register_http_in_flight(
        &mut self,
        key: TileKey,
        shutdown: &CancelToken,
    ) -> (u64, CancelToken) {
        self.next_gen = self.next_gen.wrapping_add(1);
        let gen_id = self.next_gen;
        let tile_cancel = shutdown.child_token();
        self.in_flight.insert(key, (gen_id, tile_cancel.clone()));
        (gen_id, tile_cancel)
    }

    pub(super) fn restore_deferred_http(&mut self, deferred: Vec<TileKey>, multi_host: bool) {
        if deferred.is_empty() {
            return;
        }
        // `deferred` was popped coarsest-first from the back of `pending_http`;
        // reversing restores ascending-priority order cleanly without a full sort
        // for single-host short-circuits, and sorting when mixed.
        if multi_host {
            self.pending_http.extend(deferred);
            Self::sort_queue_coarsest_at_end(&mut self.pending_http);
        } else {
            self.pending_http.extend(deferred.into_iter().rev());
        }
    }

    pub(super) fn remove_in_flight_if_gen(&mut self, key: &TileKey, generation: u64) -> bool {
        if self
            .in_flight
            .get(key)
            .is_some_and(|(gen_id, _)| *gen_id == generation)
        {
            self.in_flight.remove(key);
            true
        } else {
            false
        }
    }

    pub(super) fn promote_disk_miss_to_http(
        &mut self,
        key: TileKey,
        generation: u64,
        can_requeue: bool,
    ) -> bool {
        if !self.remove_in_flight_if_gen(&key, generation) {
            return false;
        }
        if can_requeue && self.is_active_source(key.source) && !self.failures.contains_key(&key) {
            Self::push_unique_sorted(&mut self.pending_http, [key]);
        }
        true
    }

    pub(super) fn release_disk_slot(&mut self) {
        self.disk_in_flight = self.disk_in_flight.saturating_sub(1);
    }

    pub(super) fn mark_completed(&mut self, key: TileKey, from_disk: bool) {
        if !self.is_active_source(key.source) {
            return;
        }
        self.last_wanted.remove(&key);
        if from_disk {
            self.completed_from_disk = self.completed_from_disk.saturating_add(1);
        } else {
            self.completed_from_net = self.completed_from_net.saturating_add(1);
            self.offline = false;
            self.last_error = None;
        }
    }

    pub(super) fn mark_empty_204(&mut self, key: TileKey, now: Instant) {
        if !self.is_active_source(key.source) {
            return;
        }
        self.last_wanted.remove(&key);
        self.insert_failure(key, FailureState::Empty, now);
        self.offline = false;
    }

    pub(super) fn mark_failure(
        &mut self,
        key: TileKey,
        failure: FailureState,
        offline: bool,
        error_msg: Option<String>,
        now: Instant,
    ) -> bool {
        if !self.is_active_source(key.source) {
            return false;
        }
        self.insert_failure(key, failure, now);
        self.failed_count = self.failed_count.saturating_add(1);
        if offline {
            self.offline = true;
        }
        if let Some(msg) = error_msg {
            self.last_error = Some(msg);
        }
        true
    }

    pub(super) fn schedule_decode_backpressure(
        &mut self,
        key: TileKey,
        backoff: Duration,
        retry_disk: bool,
        now: Instant,
    ) -> bool {
        if self.is_active_source(key.source)
            && self.last_wanted.contains(&key)
            && !self.in_flight.contains_key(&key)
            && !self.failures.contains_key(&key)
        {
            self.insert_failure(
                key,
                FailureState::BackoffUntil {
                    until: now + backoff,
                    retry_disk,
                },
                now,
            );
            true
        } else {
            false
        }
    }

    pub(super) fn insert_failure(&mut self, key: TileKey, failure: FailureState, now: Instant) {
        if self.failures.len() >= MAX_FAILURE_ENTRIES && !self.failures.contains_key(&key) {
            self.prune_expired_backoff_markers(now);
            if self.failures.len() >= MAX_FAILURE_ENTRIES {
                let wanted = &self.last_wanted;
                self.failures.retain(|k, _| wanted.contains(k));
            }
            if self.failures.len() >= MAX_FAILURE_ENTRIES
                && let Some(&evict_key) = self.failures.keys().next()
            {
                self.failures.remove(&evict_key);
            }
        }
        self.failures.insert(key, failure);
    }

    /// Prune all expired `BackoffUntil <= now` markers and re-enqueue any that
    /// are still in `last_wanted` into `pending_disk` or `pending_http`.
    /// Returns whether new work was queued.
    pub(super) fn prune_expired_backoffs(&mut self, now: Instant) -> bool {
        let Some(spec) = self.active_spec.as_ref() else {
            self.failures.clear();
            return false;
        };
        let active_source = spec.source_id();
        let mut expired = Vec::new();
        self.failures.retain(|&key, f| match f {
            FailureState::BackoffUntil { until, retry_disk } if *until <= now => {
                expired.push((key, *retry_disk));
                false
            }
            _ => true,
        });
        if expired.is_empty() {
            return false;
        }
        let queued_disk: HashSet<TileKey> = self.pending_disk.iter().copied().collect();
        let queued_http: HashSet<TileKey> = self.pending_http.iter().copied().collect();
        let mut disk_requeue = Vec::new();
        let mut http_requeue = Vec::new();
        for (key, retry_disk) in expired {
            if key.source == active_source
                && self.last_wanted.contains(&key)
                && !self.in_flight.contains_key(&key)
                && !queued_disk.contains(&key)
                && !queued_http.contains(&key)
            {
                if retry_disk {
                    disk_requeue.push(key);
                } else {
                    http_requeue.push(key);
                }
            }
        }
        let requeued_disk = Self::push_unique_sorted(&mut self.pending_disk, disk_requeue);
        let requeued_http = Self::push_unique_sorted(&mut self.pending_http, http_requeue);
        requeued_disk || requeued_http
    }

    /// Earliest `BackoffUntil` timestamp among currently wanted tiles (used by
    /// `dispatcher_loop` to self-wake for idle retries without busy-spinning).
    pub(super) fn earliest_wanted_backoff(&self, now: Instant) -> Option<Instant> {
        self.failures
            .iter()
            .filter_map(|(k, f)| match f {
                FailureState::BackoffUntil { until, .. }
                    if *until > now && self.last_wanted.contains(k) =>
                {
                    Some(*until)
                }
                _ => None,
            })
            .min()
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use veldmap_core::tile::{PlanetaryScheme, QuadTileId};

    use super::*;

    fn wm_key(spec: &LiveBasemapSpec, level: u8, x: u32, y: u32) -> TileKey {
        spec.tile_key(QuadTileId {
            scheme: PlanetaryScheme::WebMercator,
            level,
            x,
            y,
        })
    }

    #[test]
    fn pipeline_state_enforces_failure_cap_and_requeues_expired_backoffs() {
        let spec = Arc::new(LiveBasemapSpec::eox_sentinel2_cloudless());
        let mut st = PipelineState {
            active_spec: Some(Arc::clone(&spec)),
            ..PipelineState::default()
        };
        let now = Instant::now();
        let http_key = wm_key(&spec, 4, 1, 2);
        let disk_key = wm_key(&spec, 4, 2, 2);
        st.last_wanted.insert(http_key);
        st.last_wanted.insert(disk_key);
        st.insert_failure(
            http_key,
            FailureState::BackoffUntil {
                until: now - Duration::from_secs(1),
                retry_disk: false,
            },
            now,
        );
        st.insert_failure(
            disk_key,
            FailureState::BackoffUntil {
                until: now - Duration::from_secs(1),
                retry_disk: true,
            },
            now,
        );
        assert!(st.prune_expired_backoffs(now));
        assert!(st.pending_http.contains(&http_key));
        assert!(!st.pending_disk.contains(&http_key));
        assert!(st.pending_disk.contains(&disk_key));
        assert!(!st.pending_http.contains(&disk_key));

        // Filling beyond MAX_FAILURE_ENTRIES prunes non-wanted entries without exceeding cap.
        for i in 0..(MAX_FAILURE_ENTRIES as u32 + 16) {
            let k = wm_key(&spec, 12, i, 0);
            st.insert_failure(k, FailureState::Terminal, now);
        }
        assert!(st.failures.len() <= MAX_FAILURE_ENTRIES);
    }
}
