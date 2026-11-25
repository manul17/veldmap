//! Monotonic LRU timestamp generator and debounced touch buffer for
//! [`super::TileDiskCache`].

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicI64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use veldmap_core::sync::lock_unpoisoned;

/// Number of buffered `get()` LRU hits before attempting a best-effort batched
/// flush (avoids a WAL write transaction on every cache read).
const TOUCH_FLUSH_THRESHOLD: usize = 64;

/// Strictly monotonic millisecond clock seeded from SQLite's persisted
/// `MAX(last_used_ms)` so backward system-clock jumps never invert LRU order.
pub(super) struct MonotonicClock {
    last_ts_ms: AtomicI64,
}

impl MonotonicClock {
    pub(super) fn new(seed_ms: i64) -> Self {
        Self {
            last_ts_ms: AtomicI64::new(seed_ms),
        }
    }

    pub(super) fn next_ms(&self) -> i64 {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_millis().min(i64::MAX as u128) as i64);
        let mut prev = self.last_ts_ms.load(Ordering::Relaxed);
        loop {
            let next = if now > prev {
                now
            } else {
                prev.saturating_add(1)
            };
            match self.last_ts_ms.compare_exchange_weak(
                prev,
                next,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => return next,
                Err(actual) => prev = actual,
            }
        }
    }

    #[cfg(test)]
    pub(super) fn current_ms(&self) -> i64 {
        self.last_ts_ms.load(Ordering::Acquire)
    }
}

/// Debounced in-memory map of `tile_id -> last_used_ms` read hits flushed
/// opportunistically or on the next writer transaction.
///
/// Lock ordering: whenever both `writer` and `pending` are held simultaneously,
/// `writer` MUST be acquired first. `pending` may be locked briefly as a leaf
/// lock (for example when recording a touch before calling `writer.try_lock()`).
pub(super) struct TouchBuffer {
    pending: Mutex<HashMap<i64, i64>>,
}

impl TouchBuffer {
    pub(super) fn new() -> Self {
        Self {
            pending: Mutex::new(HashMap::new()),
        }
    }

    /// Record an LRU read hit for `tile_id` at `ts`, returning `true` when the
    /// buffered entry count has reached [`TOUCH_FLUSH_THRESHOLD`].
    ///
    /// Releases `pending`'s mutex before returning so callers may follow up
    /// with a non-blocking `writer.try_lock()` without violating
    /// `writer -> pending_touches` lock ordering.
    pub(super) fn record(&self, tile_id: i64, ts: i64) -> bool {
        let mut touches = lock_unpoisoned(&self.pending);
        touches
            .entry(tile_id)
            .and_modify(|prev| *prev = (*prev).max(ts))
            .or_insert(ts);
        touches.len() >= TOUCH_FLUSH_THRESHOLD
    }

    pub(super) fn take_all(&self) -> HashMap<i64, i64> {
        std::mem::take(&mut *lock_unpoisoned(&self.pending))
    }

    /// Merge drained touches back into `pending` (keeping the newest timestamp
    /// per `tile_id`) when a write transaction fails so buffered LRU recency
    /// hits are not lost on transient SQLite errors.
    pub(super) fn restore(&self, touches: HashMap<i64, i64>) {
        if touches.is_empty() {
            return;
        }
        let mut pending = lock_unpoisoned(&self.pending);
        for (tile_id, ts) in touches {
            pending
                .entry(tile_id)
                .and_modify(|prev| *prev = (*prev).max(ts))
                .or_insert(ts);
        }
    }

    pub(super) fn prune(&self, tile_ids: &[i64]) {
        if tile_ids.is_empty() {
            return;
        }
        let mut touches = lock_unpoisoned(&self.pending);
        for id in tile_ids {
            touches.remove(id);
        }
    }

    pub(super) fn clear(&self) {
        lock_unpoisoned(&self.pending).clear();
    }

    pub(super) fn is_empty(&self) -> bool {
        lock_unpoisoned(&self.pending).is_empty()
    }

    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        lock_unpoisoned(&self.pending).len()
    }
}
