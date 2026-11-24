//! Shared cancellable singleflight slot primitive used by [`super::chunk_cache`]
//! and lazy display-mapping stretch computation.

use std::sync::{Arc, Condvar, Mutex};

use veldmap_core::sync::{CancelToken, lock_unpoisoned, wait_unpoisoned};

use crate::RasterError;

/// One-shot cancellable singleflight slot storing a resolved value `T`.
pub(crate) struct SingleflightSlot<T> {
    state: Mutex<Option<T>>,
    cvar: Condvar,
}

impl<T: Clone + Send + 'static> SingleflightSlot<T> {
    pub(crate) fn new() -> Self {
        Self {
            state: Mutex::new(None),
            cvar: Condvar::new(),
        }
    }

    /// Resolve the slot if still pending and wake all waiting followers.
    pub(crate) fn resolve(&self, value: T) {
        let mut guard = lock_unpoisoned(&self.state);
        if guard.is_none() {
            *guard = Some(value);
            self.cvar.notify_all();
        }
    }

    /// Wait until this slot resolves or `cancel` fires.
    pub(crate) fn wait(self: &Arc<Self>, cancel: &CancelToken) -> Result<T, RasterError> {
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
