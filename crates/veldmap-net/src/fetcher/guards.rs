//! RAII concurrency guards preserving `in_flight`, disk-slot, and per-host HTTP
//! permit invariants across cancellation, early returns, and unpolled task
//! drops.

use std::sync::Arc;

use tokio::sync::OwnedSemaphorePermit;
use veldmap_core::sync::lock_unpoisoned;
use veldmap_core::tile::TileKey;

use super::PipelineShared;
use super::state::{PipelineState, Transition};

/// Generation-checked RAII guard keeping `key` registered in `state.in_flight`
/// across async fetch, disk lookup/delete, and CPU decode submission.
pub(super) struct InFlightGuard {
    shared: Arc<PipelineShared>,
    key: TileKey,
    generation: u64,
    armed: bool,
}

impl InFlightGuard {
    pub(super) fn new(shared: Arc<PipelineShared>, key: TileKey, generation: u64) -> Self {
        Self {
            shared,
            key,
            generation,
            armed: true,
        }
    }

    /// Atomically apply a terminal state transition under `state`'s mutex,
    /// unconditionally disarm this guard (preventing any duplicate lock/notify
    /// on drop), and notify the dispatcher iff the transition was applied.
    pub(super) fn finish(
        mut self,
        f: impl FnOnce(&mut PipelineState, TileKey, u64) -> Transition,
    ) -> Transition {
        let transition = {
            let mut st = lock_unpoisoned(&self.shared.state);
            f(&mut st, self.key, self.generation)
        };
        self.armed = false;
        if transition.is_applied() {
            self.shared.notify.notify_one();
        }
        transition
    }
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let removed = {
            let mut st = lock_unpoisoned(&self.shared.state);
            st.remove_in_flight_if_gen(&self.key, self.generation)
        };
        if removed {
            self.shared.notify.notify_one();
        }
    }
}

/// RAII guard for the bounded disk-lookup lane: decrements `disk_in_flight`
/// under lock BEFORE notifying `dispatcher_loop`.
pub(super) struct DiskSlotGuard {
    shared: Arc<PipelineShared>,
}

impl DiskSlotGuard {
    pub(super) fn new(shared: Arc<PipelineShared>) -> Self {
        Self { shared }
    }
}

impl Drop for DiskSlotGuard {
    fn drop(&mut self) {
        {
            let mut st = lock_unpoisoned(&self.shared.state);
            st.release_disk_slot();
        }
        self.shared.notify.notify_one();
    }
}

/// RAII guard for a per-host HTTP semaphore permit.
///
/// Explicitly releases `permit` BEFORE calling `notify.notify_one()` so
/// `dispatcher_loop`'s `try_acquire_owned()` never races ahead of permit drop.
pub(super) struct PermitGuard {
    permit: Option<OwnedSemaphorePermit>,
    shared: Arc<PipelineShared>,
}

impl PermitGuard {
    pub(super) fn new(permit: OwnedSemaphorePermit, shared: Arc<PipelineShared>) -> Self {
        Self {
            permit: Some(permit),
            shared,
        }
    }
}

impl Drop for PermitGuard {
    fn drop(&mut self) {
        drop(self.permit.take());
        self.shared.notify.notify_one();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::sync::atomic::AtomicBool;

    use tokio::sync::Notify;
    use veldmap_core::sync::CancelToken;

    use crate::basemap::LiveBasemapSpec;
    use crate::http::HostLimiters;

    use super::*;

    #[test]
    fn stale_inflight_guard_never_removes_newer_generation() {
        let shared = Arc::new(PipelineShared {
            shutdown: CancelToken::new(),
            shutdown_signaled: AtomicBool::new(false),
            state: Mutex::new(PipelineState::default()),
            host_limiters: Arc::new(HostLimiters::new(4)),
            notify: Notify::new(),
        });
        let spec = LiveBasemapSpec::eox_sentinel2_cloudless();
        let key = spec.wm_key(0, 0, 0);

        let (gen1, _) = {
            let mut st = lock_unpoisoned(&shared.state);
            st.register_in_flight_for_test(key, &shared.shutdown)
        };
        let guard1 = InFlightGuard::new(Arc::clone(&shared), key, gen1);
        let (gen2, _) = {
            let mut st = lock_unpoisoned(&shared.state);
            st.register_in_flight_for_test(key, &shared.shutdown)
        };
        drop(guard1);

        let mut st = lock_unpoisoned(&shared.state);
        assert!(
            !st.remove_in_flight_if_gen(&key, gen1),
            "stale generation 1 guard must not remove generation 2 entry"
        );
        assert!(
            st.remove_in_flight_if_gen(&key, gen2),
            "generation 2 entry must still be present"
        );
    }
}
