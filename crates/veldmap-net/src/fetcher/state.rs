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
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Transition {
    /// Whether the tile's active generation was removed and the dispatcher loop
    /// should be notified.
    pub(super) notify: bool,
    /// Whether observable HUD/UI status changed and the UI wakeup should fire.
    pub(super) wakeup: bool,
}

impl Transition {
    pub(super) const NONE: Self = Self {
        notify: false,
        wakeup: false,
    };
    pub(super) const NOTIFY: Self = Self {
        notify: true,
        wakeup: false,
    };
    pub(super) const NOTIFY_AND_WAKEUP: Self = Self {
        notify: true,
        wakeup: true,
    };
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

    fn requeue_eligible_entries(
        &mut self,
        active_source: SourceId,
        entries: Vec<(TileKey, bool)>,
    ) -> bool {
        let mut requeued = false;
        for (key, retry_disk) in entries {
            if self.is_queue_eligible(&key, active_source)
                && self.is_wanted_or_root(&key, active_source)
                && !self.pending_disk.contains(&key)
                && !self.pending_http.contains(&key)
            {
                let added = if retry_disk {
                    self.pending_disk.insert_unique(key)
                } else {
                    self.pending_http.insert_unique(key)
                };
                requeued |= added;
            }
        }
        requeued
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

        // Drain expired backoffs (`until <= now`), routing `retry_disk: false`
        // directly to `pending_http` while keeping `retry_disk: true` on disk.
        let mut expired_http = Vec::new();
        let mut expired_disk = Vec::new();
        for (key, retry_disk) in self.failures.drain_expired_backoffs(now) {
            if retry_disk {
                expired_disk.push(key);
            } else {
                expired_http.push(key);
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

        for key in expired_http {
            if self.is_queue_eligible(&key, active_source)
                && self.is_wanted_or_root(&key, active_source)
                && !is_ready(&key)
            {
                self.pending_http.insert_unique(key);
            }
        }

        // Preserve still-unready wanted tiles and pinned active roots already in
        // `pending_disk` in place, then insert newly wanted / expired-disk keys.
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

        for key in expired_disk {
            if self.is_queue_eligible(&key, active_source)
                && self.is_wanted_or_root(&key, active_source)
                && !is_ready(&key)
                && !self.pending_http.contains(&key)
            {
                self.pending_disk.insert_unique(key);
            }
        }

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
        if !self.in_flight.remove_if_gen(&key, generation) {
            return Transition::NONE;
        }
        if can_requeue && self.is_active_source(key.source) && !self.failures.is_suppressed(&key) {
            self.pending_http.insert_unique(key);
        }
        Transition::NOTIFY
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
        if !self.in_flight.remove_if_gen(&key, generation) {
            return Transition::NONE;
        }
        if !self.is_active_source(key.source) {
            return Transition::NOTIFY;
        }
        self.last_wanted.remove(&key);
        if from_disk {
            self.completed_from_disk = self.completed_from_disk.saturating_add(1);
        } else {
            self.completed_from_net = self.completed_from_net.saturating_add(1);
            self.offline = false;
            self.last_error = None;
        }
        Transition::NOTIFY
    }

    pub(super) fn mark_empty_204(&mut self, key: TileKey, generation: u64) -> Transition {
        if !self.in_flight.remove_if_gen(&key, generation) {
            return Transition::NONE;
        }
        if !self.is_active_source(key.source) {
            return Transition::NOTIFY;
        }
        self.last_wanted.remove(&key);
        self.failures
            .insert_failure(key, FailureMarker::Empty, &self.last_wanted);
        self.offline = false;
        Transition::NOTIFY
    }

    pub(super) fn mark_terminal_failure(
        &mut self,
        key: TileKey,
        generation: u64,
        error_msg: Option<String>,
    ) -> Transition {
        if !self.in_flight.remove_if_gen(&key, generation) {
            return Transition::NONE;
        }
        if !self.is_active_source(key.source) {
            return Transition::NOTIFY;
        }
        self.failures
            .insert_failure(key, FailureMarker::Terminal, &self.last_wanted);
        self.failed_count = self.failed_count.saturating_add(1);
        if let Some(msg) = error_msg {
            self.last_error = Some(msg);
        }
        Transition::NOTIFY_AND_WAKEUP
    }

    pub(super) fn mark_transient_backoff(
        &mut self,
        key: TileKey,
        generation: u64,
        until: Instant,
        offline: bool,
        error_msg: Option<String>,
    ) -> Transition {
        if !self.in_flight.remove_if_gen(&key, generation) {
            return Transition::NONE;
        }
        if !self.is_active_source(key.source) {
            return Transition::NOTIFY;
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
        Transition::NOTIFY_AND_WAKEUP
    }

    pub(super) fn schedule_decode_backpressure(
        &mut self,
        key: TileKey,
        generation: u64,
        backoff: Duration,
        retry_disk: bool,
        now: Instant,
    ) -> Transition {
        if !self.in_flight.remove_if_gen(&key, generation) {
            return Transition::NONE;
        }
        if let Some(active_source) = self.active_spec.as_ref().map(|s| s.source_id())
            && self.is_queue_eligible(&key, active_source)
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
        Transition::NOTIFY
    }

    pub(super) fn insert_failure(&mut self, key: TileKey, marker: FailureMarker) {
        self.failures.insert_failure(key, marker, &self.last_wanted);
    }

    /// Prune all expired backoff entries (`until <= now`) and re-enqueue any
    /// that are still wanted or pinned active-source roots into `pending_disk`
    /// or `pending_http`. Returns whether new work was queued.
    pub(super) fn prune_expired_backoffs(&mut self, now: Instant) -> bool {
        let Some(spec) = self.active_spec.as_ref() else {
            self.failures.clear_backoffs();
            return false;
        };
        let active_source = spec.source_id();
        let expired = self.failures.drain_expired_backoffs(now);
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
        self.failures
            .earliest_wanted_backoff(now, active_source, &self.last_wanted)
    }
}

#[cfg(test)]
#[path = "state_tests.rs"]
mod tests;
