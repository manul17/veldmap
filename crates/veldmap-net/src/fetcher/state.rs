//! Queue, in-flight tracking, and failure/backoff state machine for
//! [`super::LiveTilePipeline`].

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use veldmap_core::sync::CancelToken;
use veldmap_core::tile::{SourceId, TileKey};

use crate::basemap::LiveBasemapSpec;

use super::NetworkStatus;
use super::queue::CoarsestTileQueue;

/// Upper bound on retained permanent/empty per-tile markers to prevent
/// unbounded memory growth on sparse 404/204 layers over long sessions.
pub(super) const MAX_FAILURE_ENTRIES: usize = 4096;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum FailureMarker {
    Terminal,
    /// HTTP `204 No Content`: suppresses infinite refetch without flagging an error.
    Empty,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct BackoffEntry {
    pub(super) until: Instant,
    pub(super) retry_disk: bool,
}

#[derive(Default)]
pub(super) struct PipelineState {
    active_spec: Option<Arc<LiveBasemapSpec>>,
    /// Unready, active-source, contract-valid keys from the most recent
    /// `reconcile_wanted`, used by idle backoff retries and `retry_failed`.
    last_wanted: HashSet<TileKey>,
    /// Coarsest-first queue awaiting fast local disk-cache lookup.
    pending_disk: CoarsestTileQueue,
    /// Coarsest-first queue of disk misses awaiting an HTTP per-host slot.
    pending_http: CoarsestTileQueue,
    /// Active blocking disk-lookup tasks (bounded by `MAX_CONCURRENT_DISK_LOOKUPS`).
    disk_in_flight: usize,
    /// Tracks active `(generation, CancelToken)` from dispatch all the way
    /// through CPU decode completion so stale guards never clobber a newer
    /// spawn and pan/switch cancels queued decode jobs immediately.
    in_flight: HashMap<TileKey, (u64, CancelToken)>,
    next_gen: u64,
    /// Permanent (`Terminal`) or valid-empty (`204 No Content`) tile markers.
    failures: HashMap<TileKey, FailureMarker>,
    /// Active retry cooldowns (kept separate from `failures` so cap pruning
    /// never drops retries and `retry_failed` preserves valid `204` markers).
    backoffs: HashMap<TileKey, BackoffEntry>,
    completed_from_disk: u64,
    completed_from_net: u64,
    failed_count: u64,
    offline: bool,
    last_error: Option<String>,
}

impl PipelineState {
    fn is_suppressed(&self, key: &TileKey) -> bool {
        self.failures.contains_key(key) || self.backoffs.contains_key(key)
    }

    fn is_active_source(&self, source: SourceId) -> bool {
        self.active_spec
            .as_ref()
            .is_some_and(|s| s.source_id() == source)
    }

    fn is_wanted_or_root(&self, key: &TileKey, active_source: SourceId) -> bool {
        self.last_wanted.contains(key) || key.is_root_of(active_source)
    }

    fn is_queue_eligible(&self, key: &TileKey, active_source: SourceId) -> bool {
        key.source == active_source && !self.in_flight.contains_key(key) && !self.is_suppressed(key)
    }

    fn drain_expired_backoffs(&mut self, now: Instant) -> Vec<(TileKey, bool)> {
        let mut expired = Vec::new();
        self.backoffs.retain(|&key, entry| {
            if entry.until <= now {
                expired.push((key, entry.retry_disk));
                false
            } else {
                true
            }
        });
        expired
    }

    fn requeue_eligible_entries(
        &mut self,
        active_source: SourceId,
        entries: Vec<(TileKey, bool)>,
    ) -> bool {
        let mut disk_requeue = Vec::new();
        let mut http_requeue = Vec::new();
        for (key, retry_disk) in entries {
            if self.is_queue_eligible(&key, active_source)
                && self.is_wanted_or_root(&key, active_source)
                && !self.pending_disk.contains(&key)
                && !self.pending_http.contains(&key)
            {
                if retry_disk {
                    disk_requeue.push(key);
                } else {
                    http_requeue.push(key);
                }
            }
        }
        let requeued_disk = self.pending_disk.push_unique(disk_requeue);
        let requeued_http = self.pending_http.push_unique(http_requeue);
        requeued_disk || requeued_http
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
        self.backoffs.clear();
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

        // Drain expired backoffs (`until <= now`), routing `retry_disk: false`
        // directly to `pending_http` (so expired HTTP retries never bounce back
        // through disk-cache lookups) while preserving `retry_disk: true`
        // entries for pinned active roots or wanted tiles.
        let mut expired_http = Vec::new();
        let mut expired_disk = Vec::new();
        for (key, retry_disk) in self.drain_expired_backoffs(now) {
            if retry_disk {
                expired_disk.push(key);
            } else {
                expired_http.push(key);
            }
        }

        // Retain still-wanted / active-root entries already in `pending_http`
        // so disk misses never bounce back to `pending_disk`.
        let mut already_in_http = {
            let Self {
                last_wanted,
                failures,
                backoffs,
                in_flight,
                pending_http,
                ..
            } = self;
            pending_http.retain_and_collect(|key| {
                key.source == active_source
                    && !is_ready(key)
                    && !in_flight.contains_key(key)
                    && !failures.contains_key(key)
                    && !backoffs.contains_key(key)
                    && (last_wanted.contains(key) || key.is_root_of(active_source))
            })
        };

        let mut promoted_http = Vec::new();
        for key in expired_http {
            if self.is_queue_eligible(&key, active_source)
                && self.is_wanted_or_root(&key, active_source)
                && !is_ready(&key)
                && already_in_http.insert(key)
            {
                promoted_http.push(key);
            }
        }
        self.pending_http.push_unique(promoted_http);

        // Preserve still-unready pinned active roots already queued in
        // `pending_disk` so `pending_disk` matches `in_flight` and
        // `pending_http` active-root pinning invariants.
        let retained_disk_roots = {
            let Self {
                failures,
                backoffs,
                in_flight,
                pending_disk,
                ..
            } = self;
            pending_disk.retain_and_collect(|key| {
                key.is_root_of(active_source)
                    && !is_ready(key)
                    && !in_flight.contains_key(key)
                    && !failures.contains_key(key)
                    && !backoffs.contains_key(key)
                    && !already_in_http.contains(key)
            })
        };

        let mut next_disk: Vec<TileKey> = retained_disk_roots.into_iter().collect();
        for key in expired_disk {
            if self.is_queue_eligible(&key, active_source)
                && self.is_wanted_or_root(&key, active_source)
                && !is_ready(&key)
                && !already_in_http.contains(&key)
            {
                next_disk.push(key);
            }
        }
        for &key in &self.last_wanted {
            if !self.is_queue_eligible(&key, active_source) || already_in_http.contains(&key) {
                continue;
            }
            next_disk.push(key);
        }
        self.pending_disk.replace_sorted(next_disk);
        Some(cancelled)
    }

    pub(super) fn retry_failed(&mut self) {
        // Preserve valid `204 Empty` markers across manual retries while
        // clearing all `Terminal` failures and active backoff timers.
        let mut entries: Vec<(TileKey, bool)> = Vec::new();
        self.failures.retain(|&key, marker| match marker {
            FailureMarker::Empty => true,
            FailureMarker::Terminal => {
                entries.push((key, true));
                false
            }
        });
        entries.extend(
            self.backoffs
                .drain()
                .map(|(key, entry)| (key, entry.retry_disk)),
        );

        self.failed_count = 0;
        self.offline = false;
        self.last_error = None;

        let Some(spec) = self.active_spec.as_ref() else {
            return;
        };
        let active_source = spec.source_id();
        self.requeue_eligible_entries(active_source, entries);
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
        let drained = self.pending_disk.drain();
        let eligible: Vec<TileKey> = drained
            .into_iter()
            .filter(|key| self.is_queue_eligible(key, active_source))
            .collect();
        self.pending_http.push_unique(eligible);
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

    pub(super) fn restore_deferred_http(&mut self, deferred: Vec<TileKey>) {
        self.pending_http.restore_deferred(deferred);
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
        if can_requeue && self.is_active_source(key.source) && !self.is_suppressed(&key) {
            self.pending_http.insert_unique(key);
        }
        true
    }

    pub(super) fn release_disk_slot(&mut self) {
        self.disk_in_flight = self.disk_in_flight.saturating_sub(1);
    }

    pub(super) fn mark_completed(
        &mut self,
        key: TileKey,
        generation: u64,
        from_disk: bool,
    ) -> bool {
        if !self.remove_in_flight_if_gen(&key, generation) {
            return false;
        }
        if !self.is_active_source(key.source) {
            return true;
        }
        self.last_wanted.remove(&key);
        if from_disk {
            self.completed_from_disk = self.completed_from_disk.saturating_add(1);
        } else {
            self.completed_from_net = self.completed_from_net.saturating_add(1);
            self.offline = false;
            self.last_error = None;
        }
        true
    }

    pub(super) fn mark_empty_204(&mut self, key: TileKey, generation: u64) -> bool {
        if !self.remove_in_flight_if_gen(&key, generation) {
            return false;
        }
        if !self.is_active_source(key.source) {
            return true;
        }
        self.last_wanted.remove(&key);
        self.insert_failure(key, FailureMarker::Empty);
        self.offline = false;
        true
    }

    pub(super) fn mark_terminal_failure(
        &mut self,
        key: TileKey,
        generation: u64,
        error_msg: Option<String>,
    ) -> bool {
        if !self.remove_in_flight_if_gen(&key, generation) {
            return false;
        }
        if !self.is_active_source(key.source) {
            return false;
        }
        self.insert_failure(key, FailureMarker::Terminal);
        self.failed_count = self.failed_count.saturating_add(1);
        if let Some(msg) = error_msg {
            self.last_error = Some(msg);
        }
        true
    }

    pub(super) fn mark_transient_backoff(
        &mut self,
        key: TileKey,
        generation: u64,
        until: Instant,
        offline: bool,
        error_msg: Option<String>,
    ) -> bool {
        if !self.remove_in_flight_if_gen(&key, generation) {
            return false;
        }
        if !self.is_active_source(key.source) {
            return false;
        }
        if !self.failures.contains_key(&key) {
            self.backoffs.insert(
                key,
                BackoffEntry {
                    until,
                    retry_disk: false,
                },
            );
        }
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
        generation: u64,
        backoff: Duration,
        retry_disk: bool,
        now: Instant,
    ) -> bool {
        if !self.remove_in_flight_if_gen(&key, generation) {
            return false;
        }
        let Some(active_source) = self.active_spec.as_ref().map(|s| s.source_id()) else {
            return false;
        };
        if self.is_queue_eligible(&key, active_source)
            && self.is_wanted_or_root(&key, active_source)
        {
            self.backoffs.insert(
                key,
                BackoffEntry {
                    until: now + backoff,
                    retry_disk,
                },
            );
            true
        } else {
            false
        }
    }

    pub(super) fn insert_failure(&mut self, key: TileKey, marker: FailureMarker) {
        self.backoffs.remove(&key);
        if self.failures.len() >= MAX_FAILURE_ENTRIES && !self.failures.contains_key(&key) {
            let wanted = &self.last_wanted;
            self.failures.retain(|k, _| wanted.contains(k));
            if self.failures.len() >= MAX_FAILURE_ENTRIES {
                let evict_key = self
                    .failures
                    .iter()
                    .find(|(_, m)| **m == FailureMarker::Terminal)
                    .map(|(k, _)| *k)
                    .or_else(|| self.failures.keys().next().copied());
                if let Some(evict_key) = evict_key {
                    self.failures.remove(&evict_key);
                }
            }
        }
        self.failures.insert(key, marker);
    }

    /// Prune all expired backoff entries (`until <= now`) and re-enqueue any
    /// that are still in `last_wanted` into `pending_disk` or `pending_http`.
    /// Returns whether new work was queued.
    pub(super) fn prune_expired_backoffs(&mut self, now: Instant) -> bool {
        let Some(spec) = self.active_spec.as_ref() else {
            self.backoffs.clear();
            return false;
        };
        let active_source = spec.source_id();
        let expired = self.drain_expired_backoffs(now);
        if expired.is_empty() {
            return false;
        }
        self.requeue_eligible_entries(active_source, expired)
    }

    /// Earliest backoff timestamp among currently wanted tiles and pinned
    /// active-source roots (used by `dispatcher_loop` to self-wake for idle
    /// retries without busy-spinning).
    pub(super) fn earliest_wanted_backoff(&self, now: Instant) -> Option<Instant> {
        let active_source = self.active_spec.as_ref()?.source_id();
        self.backoffs
            .iter()
            .filter_map(|(k, entry)| {
                (entry.until > now && self.is_wanted_or_root(k, active_source))
                    .then_some(entry.until)
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
    fn pipeline_state_separates_failures_and_preserves_204_empty_on_retry() {
        let spec = Arc::new(LiveBasemapSpec::eox_sentinel2_cloudless());
        let mut st = PipelineState {
            active_spec: Some(Arc::clone(&spec)),
            ..PipelineState::default()
        };
        let now = Instant::now();
        let http_key = wm_key(&spec, 4, 1, 2);
        let disk_key = wm_key(&spec, 4, 2, 2);
        let empty_key = wm_key(&spec, 4, 3, 2);
        let terminal_key = wm_key(&spec, 4, 4, 2);

        st.last_wanted.insert(http_key);
        st.last_wanted.insert(disk_key);
        st.last_wanted.insert(terminal_key);

        st.backoffs.insert(
            http_key,
            BackoffEntry {
                until: now - Duration::from_secs(1),
                retry_disk: false,
            },
        );
        st.backoffs.insert(
            disk_key,
            BackoffEntry {
                until: now - Duration::from_secs(1),
                retry_disk: true,
            },
        );
        assert!(st.prune_expired_backoffs(now));
        assert!(st.pending_http.contains(&http_key));
        assert!(!st.pending_disk.contains(&http_key));
        assert!(st.pending_disk.contains(&disk_key));
        assert!(!st.pending_http.contains(&disk_key));

        // 204 Empty markers survive retry_failed while Terminal markers re-queue.
        // Stale generations must NOT mutate failure or backoff state.
        let shutdown = CancelToken::new();
        let (empty_gen, _) = st.register_http_in_flight(empty_key, &shutdown);
        assert!(!st.mark_empty_204(empty_key, empty_gen.wrapping_add(99)));
        assert!(st.mark_empty_204(empty_key, empty_gen));
        st.last_wanted.insert(empty_key);
        let (term_gen, _) = st.register_http_in_flight(terminal_key, &shutdown);
        assert!(!st.mark_terminal_failure(
            terminal_key,
            term_gen.wrapping_add(99),
            Some("stale".to_owned())
        ));
        assert!(st.mark_terminal_failure(terminal_key, term_gen, Some("bad".to_owned())));
        st.retry_failed();
        assert_eq!(st.failures.get(&empty_key), Some(&FailureMarker::Empty));
        assert!(!st.failures.contains_key(&terminal_key));
        assert!(st.pending_disk.contains(&terminal_key));

        // Active-source coarsest root stays pinned in pending_disk and backoff wakeups
        // even when omitted from a subsequent reconcile_wanted frame.
        let root_key = wm_key(&spec, 0, 0, 0);
        let mut initial_wanted = HashSet::new();
        initial_wanted.insert(root_key);
        st.reconcile_wanted(&initial_wanted, &|_| false, now);
        assert!(st.pending_disk.contains(&root_key));
        let empty_wanted = HashSet::new();
        st.reconcile_wanted(&empty_wanted, &|_| false, now);
        assert!(
            st.pending_disk.contains(&root_key),
            "active root must remain pinned in pending_disk until ready"
        );
        // Active root in backoff still schedules an idle wakeup and re-queues on retry_disk expiry.
        st.backoffs.insert(
            root_key,
            BackoffEntry {
                until: now + Duration::from_secs(5),
                retry_disk: true,
            },
        );
        assert_eq!(
            st.earliest_wanted_backoff(now),
            Some(now + Duration::from_secs(5))
        );
        st.pending_disk.clear();
        st.reconcile_wanted(&empty_wanted, &|_| false, now + Duration::from_secs(6));
        assert!(
            st.pending_disk.contains(&root_key),
            "expired retry_disk backoff for pinned active root must re-queue"
        );

        // Filling beyond MAX_FAILURE_ENTRIES caps failures without evicting active backoffs.
        let active_backoff = wm_key(&spec, 5, 0, 0);
        st.last_wanted.insert(active_backoff);
        let (bo_gen, _) = st.register_http_in_flight(active_backoff, &shutdown);
        assert!(st.mark_transient_backoff(
            active_backoff,
            bo_gen,
            now + Duration::from_secs(10),
            false,
            None,
        ));
        for i in 0..(MAX_FAILURE_ENTRIES as u32 + 16) {
            let k = wm_key(&spec, 12, i, 0);
            st.insert_failure(k, FailureMarker::Terminal);
        }
        assert!(st.failures.len() <= MAX_FAILURE_ENTRIES);
        assert!(st.backoffs.contains_key(&active_backoff));
    }
}
