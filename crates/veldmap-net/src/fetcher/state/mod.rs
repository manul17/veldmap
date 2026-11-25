//! Pure synchronous queue, in-flight tracking, dispatch planning, and
//! failure/backoff state machine for [`crate::LiveTilePipeline`].

mod failures;
mod in_flight;
mod plan;
mod queue;

use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use veldmap_core::sync::CancelToken;
use veldmap_core::tile::{SourceId, TileKey};

use crate::basemap::LiveBasemapSpec;

use self::failures::{BackoffEntry, FailureMarker, FailureTracker};
use self::in_flight::InFlightTable;
pub(in crate::fetcher) use self::plan::TileTarget;
use self::queue::PendingQueues;

/// Observable status snapshot of the live tile pipeline for the GIS HUD / UI.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NetworkStatus {
    pub in_flight: usize,
    pub queued: usize,
    pub completed_from_disk: u64,
    pub completed_from_net: u64,
    pub failed_count: u64,
    pub offline: bool,
    pub last_error: Option<String>,
}

/// Returns `true` when `key` is currently requested in `wanted` or is a pinned
/// coarsest root tile of `active_source`.
fn is_wanted_or_root(wanted: &HashSet<TileKey>, key: &TileKey, active_source: SourceId) -> bool {
    wanted.contains(key) || key.is_root_of(active_source)
}

/// Explicit side-effect summary returned by in-flight state transitions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::fetcher) enum Transition {
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
    pub(in crate::fetcher) const APPLIED_SILENT: Self = Self::Applied { wakeup: false };
    pub(in crate::fetcher) const APPLIED_WAKEUP: Self = Self::Applied { wakeup: true };

    pub(in crate::fetcher) const fn is_applied(self) -> bool {
        matches!(self, Self::Applied { .. })
    }

    pub(in crate::fetcher) const fn should_wakeup(self) -> bool {
        matches!(self, Self::Applied { wakeup: true })
    }
}

#[derive(Default)]
struct StatusCounters {
    completed_from_disk: u64,
    completed_from_net: u64,
    failed_count: u64,
    offline: bool,
    last_error: Option<String>,
}

impl StatusCounters {
    fn reset_errors(&mut self) {
        self.failed_count = 0;
        self.offline = false;
        self.last_error = None;
    }

    fn record_failure(&mut self, offline: bool, error_msg: Option<String>) {
        self.failed_count = self.failed_count.saturating_add(1);
        if offline {
            self.offline = true;
        }
        if let Some(msg) = error_msg {
            self.last_error = Some(msg);
        }
    }
}

fn is_key_eligible(
    in_flight: &InFlightTable,
    failures: &FailureTracker,
    key: &TileKey,
    active_source: SourceId,
) -> bool {
    key.source == active_source && !in_flight.contains(key) && !failures.is_suppressed(key)
}

fn is_pending_eligible(
    in_flight: &InFlightTable,
    failures: &FailureTracker,
    last_wanted: &HashSet<TileKey>,
    key: &TileKey,
    active_source: SourceId,
    is_ready: &impl Fn(&TileKey) -> bool,
) -> bool {
    is_key_eligible(in_flight, failures, key, active_source)
        && is_wanted_or_root(last_wanted, key, active_source)
        && !is_ready(key)
}

#[derive(Default)]
pub(in crate::fetcher) struct PipelineState {
    has_disk_cache: bool,
    active_spec: Option<Arc<LiveBasemapSpec>>,
    /// Unready, active-source, contract-valid keys from the most recent
    /// `reconcile_wanted`, used by idle backoff retries and `retry_failed`.
    last_wanted: HashSet<TileKey>,
    pending: PendingQueues,
    in_flight: InFlightTable,
    disk_in_flight: usize,
    failures: FailureTracker,
    counters: StatusCounters,
}

impl PipelineState {
    pub(in crate::fetcher) fn new(has_disk_cache: bool) -> Self {
        Self {
            has_disk_cache,
            ..Self::default()
        }
    }

    fn is_active_source(&self, source: SourceId) -> bool {
        self.active_spec
            .as_ref()
            .is_some_and(|s| s.source_id() == source)
    }

    fn is_queue_eligible(&self, key: &TileKey, active_source: SourceId) -> bool {
        is_key_eligible(&self.in_flight, &self.failures, key, active_source)
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
        let has_disk_cache = self.has_disk_cache;
        for (key, retry_disk) in entries {
            if is_pending_eligible(
                &self.in_flight,
                &self.failures,
                &self.last_wanted,
                &key,
                active_source,
                &is_ready,
            ) && !self.pending.contains(&key)
            {
                self.pending
                    .insert_unique(key, retry_disk && has_disk_cache);
            }
        }
    }

    pub(in crate::fetcher) fn set_active_spec(
        &mut self,
        spec: Option<LiveBasemapSpec>,
    ) -> Option<Vec<CancelToken>> {
        if self.active_spec.as_deref() == spec.as_ref() {
            return None;
        }
        self.active_spec = spec.map(Arc::new);
        self.last_wanted.clear();
        self.pending.clear();
        self.failures.clear();
        self.counters.reset_errors();
        Some(self.in_flight.drain_tokens())
    }

    pub(in crate::fetcher) fn reconcile_wanted<F>(
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
            self.pending.clear();
            return None;
        };
        let active_source = spec.source_id();

        let cancelled = self.in_flight.retain_and_collect_cancelled(|key| {
            key.source == active_source
                && !is_ready(key)
                && is_wanted_or_root(wanted, key, active_source)
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

        // Retain still-wanted / active-root entries in place: keep `http`
        // entries out of `disk` so disk misses never bounce back.
        let in_flight = &self.in_flight;
        let failures = &self.failures;
        let last_wanted = &self.last_wanted;
        self.pending.retain(|key| {
            is_pending_eligible(
                in_flight,
                failures,
                last_wanted,
                key,
                active_source,
                is_ready,
            )
        });

        // Drain expired backoffs (`until <= now`), routing `retry_disk: false`
        // (or all entries when no disk cache is configured) directly to `http`.
        let expired = self.failures.drain_expired_backoffs(now);
        self.requeue_eligible_entries(active_source, expired, is_ready);

        let has_disk_cache = self.has_disk_cache;
        for &key in &self.last_wanted {
            if self.is_queue_eligible(&key, active_source) {
                self.pending.insert_unique(key, has_disk_cache);
            }
        }

        Some(cancelled)
    }

    pub(in crate::fetcher) fn retry_failed(&mut self) {
        let entries = self.failures.drain_for_retry();
        self.counters.reset_errors();

        let Some(spec) = self.active_spec.as_ref() else {
            return;
        };
        let active_source = spec.source_id();
        self.requeue_eligible_entries(active_source, entries, |_| false);
    }

    pub(in crate::fetcher) fn status(&self) -> NetworkStatus {
        NetworkStatus {
            in_flight: self.in_flight.len(),
            queued: self.pending.len(),
            completed_from_disk: self.counters.completed_from_disk,
            completed_from_net: self.counters.completed_from_net,
            failed_count: self.counters.failed_count,
            offline: self.counters.offline,
            last_error: self.counters.last_error.clone(),
        }
    }

    pub(in crate::fetcher) fn signal_shutdown(&mut self) -> Vec<CancelToken> {
        self.last_wanted.clear();
        self.pending.clear();
        self.in_flight.drain_tokens()
    }

    #[cfg(test)]
    pub(in crate::fetcher) fn register_in_flight_for_test(
        &mut self,
        key: TileKey,
        shutdown: &CancelToken,
    ) -> (u64, CancelToken) {
        self.in_flight.register(key, shutdown)
    }

    pub(in crate::fetcher) fn remove_in_flight_if_gen(
        &mut self,
        key: &TileKey,
        generation: u64,
    ) -> bool {
        self.in_flight.remove_if_gen(key, generation)
    }

    pub(in crate::fetcher) fn promote_disk_miss_to_http(
        &mut self,
        key: TileKey,
        generation: u64,
        can_requeue: bool,
    ) -> Transition {
        if !self.finish_in_flight(&key, generation) {
            return Transition::Stale;
        }
        if can_requeue && !self.failures.is_suppressed(&key) {
            self.pending.insert_http(key);
        }
        Transition::APPLIED_SILENT
    }

    pub(in crate::fetcher) fn release_disk_slot(&mut self) {
        self.disk_in_flight = self.disk_in_flight.saturating_sub(1);
    }

    pub(in crate::fetcher) fn mark_completed(
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
            self.counters.completed_from_disk = self.counters.completed_from_disk.saturating_add(1);
        } else {
            self.counters.completed_from_net = self.counters.completed_from_net.saturating_add(1);
            self.counters.offline = false;
            self.counters.last_error = None;
        }
        Transition::APPLIED_SILENT
    }

    pub(in crate::fetcher) fn mark_empty_204(
        &mut self,
        key: TileKey,
        generation: u64,
    ) -> Transition {
        if !self.finish_in_flight(&key, generation) {
            return Transition::Stale;
        }
        self.last_wanted.remove(&key);
        self.failures
            .insert_failure(key, FailureMarker::Empty, &self.last_wanted);
        self.counters.offline = false;
        Transition::APPLIED_WAKEUP
    }

    pub(in crate::fetcher) fn mark_terminal_failure(
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
        self.counters.record_failure(false, error_msg);
        Transition::APPLIED_WAKEUP
    }

    pub(in crate::fetcher) fn mark_transient_backoff(
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
        self.counters.record_failure(offline, error_msg);
        Transition::APPLIED_WAKEUP
    }

    pub(in crate::fetcher) fn schedule_decode_backpressure(
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
            && is_wanted_or_root(&self.last_wanted, &key, active_source)
        {
            self.failures.insert_backoff(
                key,
                BackoffEntry {
                    until: now + backoff,
                    retry_disk: retry_disk && self.has_disk_cache,
                },
            );
        }
        Transition::APPLIED_SILENT
    }

    #[cfg(test)]
    fn insert_failure(&mut self, key: TileKey, marker: FailureMarker) {
        self.failures.insert_failure(key, marker, &self.last_wanted);
    }

    /// Prune all expired backoff entries (`until <= now`) and re-enqueue any
    /// that are still wanted or pinned active-source roots into `pending.disk`
    /// or `pending.http`.
    fn prune_expired_backoffs(&mut self, now: Instant) {
        let Some(spec) = self.active_spec.as_ref() else {
            self.failures.clear_backoffs();
            return;
        };
        let active_source = spec.source_id();
        let expired = self.failures.drain_expired_backoffs(now);
        self.requeue_eligible_entries(active_source, expired, |_| false);
    }
}

#[cfg(test)]
#[path = "state_tests.rs"]
mod tests;
