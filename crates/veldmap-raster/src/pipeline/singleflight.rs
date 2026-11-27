//! Shared cancellable, panic-safe singleflight orchestration used by
//! [`super::chunk_cache`] and lazy display-mapping stretch computation.

use std::collections::HashMap;
use std::hash::Hash;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Condvar, Mutex};

use veldmap_core::sync::{CancelToken, lock_unpoisoned, wait_unpoisoned};

use crate::RasterError;

/// Resolution outcome published through a [`SingleflightSlot`].
#[derive(Clone)]
enum SingleflightOutcome<T> {
    Ready(T),
    /// The leader's own [`CancelToken`] fired during execution; uncancelled
    /// followers should loop and elect a new leader instead of failing.
    LeaderCancelled,
    Failed(RasterError),
}

/// Classification returned by a cache when entering [`run_singleflight`].
pub(super) enum SingleflightRole<T> {
    /// Cached value already available.
    Ready(T),
    /// Caller became the leader responsible for computing and publishing into
    /// this slot.
    Leader(Arc<SingleflightSlot<T>>),
    /// Another leader is already computing this key; wait on this slot.
    Follower(Arc<SingleflightSlot<T>>),
}

/// One-shot cancellable singleflight slot storing a resolved outcome for `T`.
pub(super) struct SingleflightSlot<T> {
    state: Mutex<Option<SingleflightOutcome<T>>>,
    cvar: Condvar,
}

impl<T> SingleflightSlot<T> {
    pub(super) fn new() -> Self {
        Self {
            state: Mutex::new(None),
            cvar: Condvar::new(),
        }
    }
}

impl<T: Clone + Send + 'static> SingleflightSlot<T> {
    /// Resolve the slot if still pending and wake all waiting followers.
    fn resolve(&self, value: SingleflightOutcome<T>) {
        let mut guard = lock_unpoisoned(&self.state);
        if guard.is_none() {
            *guard = Some(value);
            self.cvar.notify_all();
        }
    }

    /// Wait until this slot resolves or `cancel` fires.
    fn wait(self: &Arc<Self>, cancel: &CancelToken) -> Result<SingleflightOutcome<T>, RasterError> {
        if cancel.is_cancelled() {
            return Err(RasterError::Cancelled);
        }
        let wake_slot = Arc::clone(self);
        let _cancel_guard = cancel.on_cancel_guarded(move || {
            let _lock = lock_unpoisoned(&wake_slot.state);
            wake_slot.cvar.notify_all();
        });

        let mut guard = lock_unpoisoned(&self.state);
        loop {
            if let Some(value) = guard.as_ref() {
                return Ok(value.clone());
            }
            if cancel.is_cancelled() {
                return Err(RasterError::Cancelled);
            }
            guard = wait_unpoisoned(&self.cvar, guard);
        }
    }
}

struct LeaderGuard<'a, T: Clone + Send + 'static, F: FnOnce(&Arc<SingleflightSlot<T>>)> {
    slot: &'a Arc<SingleflightSlot<T>>,
    abort: Option<F>,
    resolved: bool,
    panic_message: &'static str,
}

impl<T: Clone + Send + 'static, F: FnOnce(&Arc<SingleflightSlot<T>>)> LeaderGuard<'_, T, F> {
    fn mark_resolved(&mut self) {
        self.abort = None;
        self.resolved = true;
    }

    fn abort_now(&mut self) {
        if let Some(abort) = self.abort.take() {
            abort(self.slot);
        }
    }
}

impl<T: Clone + Send + 'static, F: FnOnce(&Arc<SingleflightSlot<T>>)> Drop
    for LeaderGuard<'_, T, F>
{
    fn drop(&mut self) {
        if let Some(abort) = self.abort.take() {
            abort(self.slot);
        }
        if !self.resolved {
            self.slot
                .resolve(SingleflightOutcome::Failed(RasterError::DecoderPanicked(
                    self.panic_message.to_string(),
                )));
        }
    }
}

/// Execute a cancellable, panic-safe singleflight lookup/compute loop.
///
/// Checks `cancel` at loop entry before calling `acquire`. When the caller is
/// elected leader, runs `compute` inside `catch_unwind` with a panic-safe
/// `Drop` guard that invokes `abort` and wakes waiting followers with
/// [`RasterError::DecoderPanicked`].
///
/// Only emits retryable leader cancellation when both `err.is_cancelled()` and
/// `cancel.is_cancelled()` are true; an uncancelled leader returning
/// `Err(RasterError::Cancelled)` propagates once to followers to avoid
/// livelock.
pub(super) fn run_singleflight<T, Acquire, Compute, Commit, Abort>(
    cancel: &CancelToken,
    panic_message: &'static str,
    mut acquire: Acquire,
    compute: Compute,
    commit: Commit,
    abort: Abort,
) -> Result<T, RasterError>
where
    T: Clone + Send + 'static,
    Acquire: FnMut() -> SingleflightRole<T>,
    Compute: FnOnce() -> Result<T, RasterError>,
    Commit: FnOnce(&Arc<SingleflightSlot<T>>, &T),
    Abort: FnOnce(&Arc<SingleflightSlot<T>>),
{
    loop {
        if cancel.is_cancelled() {
            return Err(RasterError::Cancelled);
        }

        match acquire() {
            SingleflightRole::Ready(value) => return Ok(value),
            SingleflightRole::Follower(slot) => match slot.wait(cancel)? {
                SingleflightOutcome::Ready(value) => return Ok(value),
                SingleflightOutcome::LeaderCancelled => continue,
                SingleflightOutcome::Failed(_) if cancel.is_cancelled() => {
                    return Err(RasterError::Cancelled);
                }
                SingleflightOutcome::Failed(err) => return Err(err),
            },
            SingleflightRole::Leader(slot) => {
                let mut guard = LeaderGuard {
                    slot: &slot,
                    abort: Some(abort),
                    resolved: false,
                    panic_message,
                };

                return match catch_unwind(AssertUnwindSafe(compute)) {
                    Ok(Ok(value)) => {
                        commit(&slot, &value);
                        slot.resolve(SingleflightOutcome::Ready(value.clone()));
                        guard.mark_resolved();
                        Ok(value)
                    }
                    Ok(Err(err)) => {
                        guard.abort_now();
                        let published = if err.is_cancelled() && cancel.is_cancelled() {
                            SingleflightOutcome::LeaderCancelled
                        } else {
                            SingleflightOutcome::Failed(err.clone())
                        };
                        slot.resolve(published);
                        guard.mark_resolved();
                        Err(err)
                    }
                    Err(_) => {
                        drop(guard);
                        Err(RasterError::DecoderPanicked(panic_message.to_string()))
                    }
                };
            }
        }
    }
}

enum SlotEntry<V> {
    InFlight(Arc<SingleflightSlot<V>>),
    Ready(V),
}

/// Keyed lazy singleflight cache deduplicating concurrent computations per key
/// while allowing uncancelled followers to take over cleanly if a leader cancels.
pub(crate) struct SingleflightMap<K, V> {
    entries: Mutex<HashMap<K, SlotEntry<V>>>,
}

impl<K: Copy + Eq + Hash, V: Clone + Send + 'static> SingleflightMap<K, V> {
    pub(crate) fn new() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
        }
    }

    pub(crate) fn with_ready(key: K, value: V) -> Self {
        let mut map = HashMap::with_capacity(1);
        map.insert(key, SlotEntry::Ready(value));
        Self {
            entries: Mutex::new(map),
        }
    }

    pub(crate) fn is_ready(&self, key: &K) -> bool {
        matches!(
            lock_unpoisoned(&self.entries).get(key),
            Some(SlotEntry::Ready(_))
        )
    }

    pub(crate) fn get_or_compute(
        &self,
        key: K,
        cancel: &CancelToken,
        panic_label: &'static str,
        compute: impl FnOnce() -> Result<V, RasterError>,
    ) -> Result<V, RasterError> {
        run_singleflight(
            cancel,
            panic_label,
            || {
                let mut guard = lock_unpoisoned(&self.entries);
                match guard.get(&key) {
                    Some(SlotEntry::Ready(val)) => SingleflightRole::Ready(val.clone()),
                    Some(SlotEntry::InFlight(slot)) => SingleflightRole::Follower(Arc::clone(slot)),
                    None => {
                        let slot = Arc::new(SingleflightSlot::new());
                        guard.insert(key, SlotEntry::InFlight(Arc::clone(&slot)));
                        SingleflightRole::Leader(slot)
                    }
                }
            },
            compute,
            |_slot, computed| {
                let mut guard = lock_unpoisoned(&self.entries);
                guard.insert(key, SlotEntry::Ready(computed.clone()));
            },
            |slot| {
                let mut guard = lock_unpoisoned(&self.entries);
                if let Some(SlotEntry::InFlight(cur)) = guard.get(&key)
                    && Arc::ptr_eq(cur, slot)
                {
                    guard.remove(&key);
                }
            },
        )
    }
}
