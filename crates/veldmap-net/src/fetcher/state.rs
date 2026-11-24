//! Queue, in-flight tracking, and failure/backoff state machine for
//! [`super::LiveTilePipeline`].

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use veldmap_core::sync::CancelToken;
use veldmap_core::tile::{SourceId, TileKey};

use crate::basemap::LiveBasemapSpec;

use super::NetworkStatus;
use super::failures::{BackoffEntry, FailureMarker, FailureTracker};
use super::queue::CoarsestTileQueue;

/// Explicit side-effect summary returned by in-flight state transitions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Transition {
    /// The tile's generation was already removed or no longer belonged to the
    /// active source; no state mutation or side effect applies.
    Stale,
    /// The tile's active generation was removed and the dispatcher loop should
    /// be notified.
    Applied {
        /// Whether observable HUD/UI status changed and the UI wakeup should fire.
        wakeup: bool,
    },
}

impl Transition {
    pub(super) const APPLIED_SILENT: Self = Self::Applied { wakeup: false };
    pub(super) const APPLIED_WAKEUP: Self = Self::Applied { wakeup: true };

    pub(super) const fn is_applied(self) -> bool {
        matches!(self, Self::Applied { .. })
    }

    pub(super) const fn should_wakeup(self) -> bool {
        matches!(self, Self::Applied { wakeup: true })
    }
}

/// Generation-checked table of active disk/HTTP/decode tasks plus bounded
/// disk-lookup slot accounting.
#[derive(Default)]
struct InFlightTable {
    entries: HashMap<TileKey, (u64, CancelToken)>,
    next_gen: u64,
    disk_in_flight: usize,
}

impl InFlightTable {
    fn len(&self) -> usize {
        self.entries.len()
    }

    fn contains(&self, key: &TileKey) -> bool {
        self.entries.contains_key(key)
    }

    fn register(&mut self, key: TileKey, shutdown: &CancelToken) -> (u64, CancelToken) {
        self.next_gen = self.next_gen.wrapping_add(1);
        let gen_id = self.next_gen;
        let tile_cancel = shutdown.child_token();
        self.entries.insert(key, (gen_id, tile_cancel.clone()));
        (gen_id, tile_cancel)
    }

    fn remove_if_gen(&mut self, key: &TileKey, generation: u64) -> bool {
        if self
            .entries
            .get(key)
            .is_some_and(|(gen_id, _)| *gen_id == generation)
        {
            self.entries.remove(key);
            true
        } else {
            false
        }
    }

    fn drain_tokens(&mut self) -> Vec<CancelToken> {
        self.entries.drain().map(|(_, (_, tok))| tok).collect()
    }

    fn retain_and_collect_cancelled(
        &mut self,
        mut keep: impl FnMut(&TileKey) -> bool,
    ) -> Vec<CancelToken> {
        let mut cancelled = Vec::new();
        self.entries.retain(|key, (_, tok)| {
            let ok = keep(key);
            if !ok {
                cancelled.push(tok.clone());
            }
            ok
        });
        cancelled
    }
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
    in_flight: InFlightTable,
    failures: FailureTracker,
    completed_from_disk: u64,
    completed_from_net: u64,
    failed_count: u64,
    offline: bool,
    last_error: Option<String>,
}

impl PipelineState {
    fn is_active_source(&self, source: SourceId) -> bool {
        self.active_spec
            .as_ref()
            .is_some_and(|s| s.source_id() == source)
    }

    fn is_wanted_or_root(&self, key: &TileKey, active_source: SourceId) -> bool {
        self.last_wanted.contains(key) || key.is_root_of(active_source)
    }

    fn is_queue_eligible(&self, key: &TileKey, active_source: SourceId) -> bool {
        key.source == active_source
            && !self.in_flight.contains(key)
            && !self.failures.is_suppressed(key)
    }

    fn finish_in_flight(&mut self, key: &TileKey, expected_gen: u64) -> bool {
        if !self.in_flight.remove_if_gen(key, expected_gen) {
            return false;
        }
        debug_assert!(self.is_active_source(key.source));
        self.is_active_source(key.source)
    }

    fn requeue_eligible_entries(
        &mut self,
        active_source: SourceId,
        entries: impl IntoIterator<Item = (TileKey, bool)>,
        is_ready: impl Fn(&TileKey) -> bool,
    ) {
        for (key, retry_disk) in entries {
            if self.is_queue_eligible(&key, active_source)
                && self.is_wanted_or_root(&key, active_source)
                && !is_ready(&key)
                && !self.pending_disk.contains(&key)
                && !self.pending_http.contains(&key)
            {
                if retry_disk {
                    self.pending_disk.insert_unique(key);
                } else {
                    self.pending_http.insert_unique(key);
                }
            }
        }
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
        Some(self.in_flight.drain_tokens())
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

        let cancelled = self.in_flight.retain_and_collect_cancelled(|key| {
            key.source == active_source
                && !is_ready(key)
                && (wanted.contains(key) || key.is_root_of(active_source))
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

        // Retain still-wanted / active-root entries already in `pending_http`
        // in place so disk misses never bounce back to `pending_disk`.
        {
            let Self {
                last_wanted,
                failures,
                in_flight,
                pending_http,
                ..
            } = self;
            pending_http.retain(|key| {
                key.source == active_source
                    && !is_ready(key)
                    && !in_flight.contains(key)
                    && !failures.is_suppressed(key)
                    && (last_wanted.contains(key) || key.is_root_of(active_source))
            });
        }

        // Preserve still-unready wanted tiles and pinned active roots already in
        // `pending_disk` in place before re-queueing expired backoffs and newly
        // wanted keys.
        {
            let Self {
                last_wanted,
                failures,
                in_flight,
                pending_http,
                pending_disk,
                ..
            } = self;
            pending_disk.retain(|key| {
                key.source == active_source
                    && !is_ready(key)
                    && !in_flight.contains(key)
                    && !failures.is_suppressed(key)
                    && !pending_http.contains(key)
                    && (last_wanted.contains(key) || key.is_root_of(active_source))
            });
        }

        // Drain expired backoffs (`until <= now`), routing `retry_disk: false`
        // directly to `pending_http` while keeping `retry_disk: true` on disk.
        let expired = self.failures.drain_expired_backoffs(now);
        self.requeue_eligible_entries(active_source, expired, is_ready);

        for &key in &self.last_wanted {
            if self.is_queue_eligible(&key, active_source) && !self.pending_http.contains(&key) {
                self.pending_disk.insert_unique(key);
            }
        }

        Some(cancelled)
    }

    pub(super) fn retry_failed(&mut self) {
        let entries = self.failures.drain_for_retry();
        self.failed_count = 0;
        self.offline = false;
        self.last_error = None;

        let Some(spec) = self.active_spec.as_ref() else {
            return;
        };
        let active_source = spec.source_id();
        self.requeue_eligible_entries(active_source, entries, |_| false);
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
        self.in_flight.drain_tokens()
    }

    pub(super) fn clear_pending_queues(&mut self) {
        self.pending_disk.clear();
        self.pending_http.clear();
    }

    pub(super) fn drain_disk_to_http(&mut self, active_source: SourceId) {
        if self.pending_disk.is_empty() {
            return;
        }
        for key in self.pending_disk.drain() {
            if self.is_queue_eligible(&key, active_source) {
                self.pending_http.insert_unique(key);
            }
        }
    }

    pub(super) fn pop_next_disk(
        &mut self,
        active_source: SourceId,
        max_concurrent_disk: usize,
        shutdown: &CancelToken,
    ) -> Option<(TileKey, u64, CancelToken)> {
        if self.in_flight.disk_in_flight >= max_concurrent_disk {
            return None;
        }
        while let Some(key) = self.pending_disk.pop() {
            if !self.is_queue_eligible(&key, active_source) {
                continue;
            }
            let (gen_id, tile_cancel) = self.in_flight.register(key, shutdown);
            self.in_flight.disk_in_flight += 1;
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
        self.in_flight.register(key, shutdown)
    }

    pub(super) fn push_deferred_http(&mut self, deferred: Vec<TileKey>) {
        self.pending_http.push_unique(deferred);
    }

    pub(super) fn remove_in_flight_if_gen(&mut self, key: &TileKey, generation: u64) -> bool {
        self.in_flight.remove_if_gen(key, generation)
    }

    pub(super) fn promote_disk_miss_to_http(
        &mut self,
        key: TileKey,
        generation: u64,
        can_requeue: bool,
    ) -> Transition {
        if !self.finish_in_flight(&key, generation) {
            return Transition::Stale;
        }
        if can_requeue && !self.failures.is_suppressed(&key) {
            self.pending_http.insert_unique(key);
        }
        Transition::APPLIED_SILENT
    }

    pub(super) fn release_disk_slot(&mut self) {
        self.in_flight.disk_in_flight = self.in_flight.disk_in_flight.saturating_sub(1);
    }

    pub(super) fn mark_completed(
        &mut self,
        key: TileKey,
        generation: u64,
        from_disk: bool,
    ) -> Transition {
        if !self.finish_in_flight(&key, generation) {
            return Transition::Stale;
        }
        self.last_wanted.remove(&key);
        if from_disk {
            self.completed_from_disk = self.completed_from_disk.saturating_add(1);
        } else {
            self.completed_from_net = self.completed_from_net.saturating_add(1);
            self.offline = false;
            self.last_error = None;
        }
        Transition::APPLIED_SILENT
    }

    pub(super) fn mark_empty_204(&mut self, key: TileKey, generation: u64) -> Transition {
        if !self.finish_in_flight(&key, generation) {
            return Transition::Stale;
        }
        self.last_wanted.remove(&key);
        self.failures
            .insert_failure(key, FailureMarker::Empty, &self.last_wanted);
        self.offline = false;
        Transition::APPLIED_WAKEUP
    }

    pub(super) fn mark_terminal_failure(
        &mut self,
        key: TileKey,
        generation: u64,
        error_msg: Option<String>,
    ) -> Transition {
        if !self.finish_in_flight(&key, generation) {
            return Transition::Stale;
        }
        self.failures
            .insert_failure(key, FailureMarker::Terminal, &self.last_wanted);
        self.failed_count = self.failed_count.saturating_add(1);
        if let Some(msg) = error_msg {
            self.last_error = Some(msg);
        }
        Transition::APPLIED_WAKEUP
    }

    pub(super) fn mark_transient_backoff(
        &mut self,
        key: TileKey,
        generation: u64,
        until: Instant,
        offline: bool,
        error_msg: Option<String>,
    ) -> Transition {
        if !self.finish_in_flight(&key, generation) {
            return Transition::Stale;
        }
        self.failures.insert_backoff(
            key,
            BackoffEntry {
                until,
                retry_disk: false,
            },
        );
        self.failed_count = self.failed_count.saturating_add(1);
        if offline {
            self.offline = true;
        }
        if let Some(msg) = error_msg {
            self.last_error = Some(msg);
        }
        Transition::APPLIED_WAKEUP
    }

    pub(super) fn schedule_decode_backpressure(
        &mut self,
        key: TileKey,
        generation: u64,
        backoff: Duration,
        retry_disk: bool,
        now: Instant,
    ) -> Transition {
        if !self.finish_in_flight(&key, generation) {
            return Transition::Stale;
        }
        let active_source = key.source;
        if self.is_queue_eligible(&key, active_source)
            && self.is_wanted_or_root(&key, active_source)
        {
            self.failures.insert_backoff(
                key,
                BackoffEntry {
                    until: now + backoff,
                    retry_disk,
                },
            );
        }
        Transition::APPLIED_SILENT
    }

    pub(super) fn insert_failure(&mut self, key: TileKey, marker: FailureMarker) {
        self.failures.insert_failure(key, marker, &self.last_wanted);
    }

    /// Prune all expired backoff entries (`until <= now`) and re-enqueue any
    /// that are still wanted or pinned active-source roots into `pending_disk`
    /// or `pending_http`.
    pub(super) fn prune_expired_backoffs(&mut self, now: Instant) {
        let Some(spec) = self.active_spec.as_ref() else {
            self.failures.clear_backoffs();
            return;
        };
        let active_source = spec.source_id();
        let expired = self.failures.drain_expired_backoffs(now);
        self.requeue_eligible_entries(active_source, expired, |_| false);
    }

    /// Earliest backoff timestamp among currently wanted tiles and pinned
    /// active-source roots (used by `dispatcher_loop` to self-wake for idle
    /// retries without busy-spinning).
    pub(super) fn earliest_wanted_backoff(&self, now: Instant) -> Option<Instant> {
        let active_source = self.active_spec.as_ref()?.source_id();
        self.failures
            .earliest_wanted_backoff(now, active_source, &self.last_wanted)
    }
}

#[cfg(test)]
#[path = "state_tests.rs"]
mod tests;
