//! Queue, in-flight tracking, and failure/backoff state machine for
//! [`super::LiveTilePipeline`].

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use veldmap_core::io::CancelToken;
use veldmap_core::tile::TileKey;

use crate::basemap::LiveBasemapSpec;

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
    pub(super) active_spec: Option<Arc<LiveBasemapSpec>>,
    /// Unready, active-source, contract-valid keys from the most recent
    /// `reconcile_wanted`, used by idle backoff retries and `retry_failed`.
    pub(super) last_wanted: HashSet<TileKey>,
    /// Coarsest-first queue awaiting fast local disk-cache lookup (stored
    /// reverse-sorted so `pop()` yields the coarsest key in O(1)).
    pub(super) pending_disk: Vec<TileKey>,
    /// Coarsest-first queue of disk misses awaiting an HTTP per-host slot.
    pub(super) pending_http: Vec<TileKey>,
    /// Active blocking disk-lookup tasks (bounded by `MAX_CONCURRENT_DISK_LOOKUPS`).
    pub(super) disk_in_flight: usize,
    /// Tracks active `(generation, CancelToken)` from dispatch all the way
    /// through CPU decode completion so stale guards never clobber a newer
    /// spawn and pan/switch cancels queued decode jobs immediately.
    pub(super) in_flight: HashMap<TileKey, (u64, CancelToken)>,
    pub(super) next_gen: u64,
    pub(super) failures: HashMap<TileKey, FailureState>,
    pub(super) completed_from_disk: u64,
    pub(super) completed_from_net: u64,
    pub(super) failed_count: u64,
    pub(super) offline: bool,
    pub(super) last_error: Option<String>,
}

impl PipelineState {
    pub(super) fn sort_queue_coarsest_at_end(queue: &mut [TileKey]) {
        queue.sort_by_key(|k| std::cmp::Reverse(k.coarsest_first_key()));
    }

    pub(super) fn insert_pending_http(&mut self, key: TileKey) {
        if !self.pending_http.contains(&key) {
            self.pending_http.push(key);
            Self::sort_queue_coarsest_at_end(&mut self.pending_http);
        }
    }

    pub(super) fn insert_failure(&mut self, key: TileKey, failure: FailureState, now: Instant) {
        if self.failures.len() >= MAX_FAILURE_ENTRIES && !self.failures.contains_key(&key) {
            self.failures.retain(|_, f| match f {
                FailureState::BackoffUntil { until, .. } => *until > now,
                FailureState::Terminal | FailureState::Empty => true,
            });
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
        let mut requeued_disk = false;
        let mut requeued_http = false;
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
        let mut queued_disk: HashSet<TileKey> = self.pending_disk.iter().copied().collect();
        let mut queued_http: HashSet<TileKey> = self.pending_http.iter().copied().collect();
        for (key, retry_disk) in expired {
            if key.source == active_source
                && self.last_wanted.contains(&key)
                && !self.in_flight.contains_key(&key)
                && !queued_disk.contains(&key)
                && !queued_http.contains(&key)
            {
                if retry_disk {
                    self.pending_disk.push(key);
                    queued_disk.insert(key);
                    requeued_disk = true;
                } else {
                    self.pending_http.push(key);
                    queued_http.insert(key);
                    requeued_http = true;
                }
            }
        }
        if requeued_disk {
            Self::sort_queue_coarsest_at_end(&mut self.pending_disk);
        }
        if requeued_http {
            Self::sort_queue_coarsest_at_end(&mut self.pending_http);
        }
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
