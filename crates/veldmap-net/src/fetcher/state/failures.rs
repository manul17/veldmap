//! Bounded failure-marker and retry-backoff tracking for [`super::PipelineState`].

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use veldmap_core::tile::{SourceId, TileKey};

use super::is_wanted_or_root;

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
pub(super) struct FailureTracker {
    /// Permanent (`Terminal`) or valid-empty (`204 No Content`) tile markers.
    failures: HashMap<TileKey, FailureMarker>,
    /// Active retry cooldowns (kept separate from `failures` so cap pruning
    /// never drops retries and `retry_failed` preserves valid `204` markers).
    backoffs: HashMap<TileKey, BackoffEntry>,
}

impl FailureTracker {
    pub(super) fn clear(&mut self) {
        self.failures.clear();
        self.backoffs.clear();
    }

    pub(super) fn clear_backoffs(&mut self) {
        self.backoffs.clear();
    }

    pub(super) fn is_suppressed(&self, key: &TileKey) -> bool {
        self.failures.contains_key(key) || self.backoffs.contains_key(key)
    }

    #[cfg(test)]
    pub(super) fn has_failure(&self, key: &TileKey) -> bool {
        self.failures.contains_key(key)
    }

    pub(super) fn insert_failure(
        &mut self,
        key: TileKey,
        marker: FailureMarker,
        last_wanted: &HashSet<TileKey>,
    ) {
        self.backoffs.remove(&key);
        if self.failures.len() >= MAX_FAILURE_ENTRIES && !self.failures.contains_key(&key) {
            self.failures.retain(|k, _| last_wanted.contains(k));
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

    pub(super) fn insert_backoff(&mut self, key: TileKey, entry: BackoffEntry) {
        if !self.failures.contains_key(&key) {
            self.backoffs.insert(key, entry);
        }
    }

    pub(super) fn drain_expired_backoffs(&mut self, now: Instant) -> Vec<(TileKey, bool)> {
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

    /// Clear all `Terminal` failures and active backoffs for manual retry,
    /// preserving valid `204 Empty` markers and returning `(key, retry_disk)`
    /// candidates to re-queue.
    pub(super) fn drain_for_retry(&mut self) -> Vec<(TileKey, bool)> {
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
        entries
    }

    pub(super) fn earliest_wanted_backoff(
        &self,
        now: Instant,
        active_source: SourceId,
        last_wanted: &HashSet<TileKey>,
    ) -> Option<Instant> {
        self.backoffs
            .iter()
            .filter_map(|(k, entry)| {
                (entry.until > now && is_wanted_or_root(last_wanted, k, active_source))
                    .then_some(entry.until)
            })
            .min()
    }

    #[cfg(test)]
    pub(super) fn failure_marker(&self, key: &TileKey) -> Option<FailureMarker> {
        self.failures.get(key).copied()
    }

    #[cfg(test)]
    pub(super) fn failure_len(&self) -> usize {
        self.failures.len()
    }

    #[cfg(test)]
    pub(super) fn has_backoff(&self, key: &TileKey) -> bool {
        self.backoffs.contains_key(key)
    }
}
