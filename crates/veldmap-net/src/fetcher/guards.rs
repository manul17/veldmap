//! RAII concurrency guards preserving `in_flight`, disk-slot, and per-host HTTP
//! permit invariants across cancellation and early returns.

use std::sync::{Arc, Mutex};

use tokio::sync::{Notify, OwnedSemaphorePermit};
use veldmap_core::io::lock_unpoisoned;
use veldmap_core::tile::TileKey;

use super::state::PipelineState;

/// Generation-checked RAII guard keeping `key` registered in `state.in_flight`
/// across async fetch, disk lookup/delete, and CPU decode submission.
pub(super) struct InFlightGuard {
    state: Arc<Mutex<PipelineState>>,
    key: TileKey,
    pub(super) generation: u64,
    notify: Arc<Notify>,
    armed: bool,
}

impl InFlightGuard {
    pub(super) fn new(
        state: Arc<Mutex<PipelineState>>,
        key: TileKey,
        generation: u64,
        notify: Arc<Notify>,
    ) -> Self {
        Self {
            state,
            key,
            generation,
            notify,
            armed: true,
        }
    }

    /// Disarm this guard after manually removing `key` from `in_flight` under
    /// `PipelineState`'s lock.
    pub(super) fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        {
            let mut st = lock_unpoisoned(&self.state);
            st.remove_in_flight_if_gen(&self.key, self.generation);
        }
        self.notify.notify_one();
    }
}

/// RAII guard for the bounded disk-lookup lane: decrements `disk_in_flight`
/// under lock BEFORE notifying `dispatcher_loop`.
pub(super) struct DiskSlotGuard {
    pub(super) state: Arc<Mutex<PipelineState>>,
    pub(super) notify: Arc<Notify>,
}

impl Drop for DiskSlotGuard {
    fn drop(&mut self) {
        {
            let mut st = lock_unpoisoned(&self.state);
            st.release_disk_slot();
        }
        self.notify.notify_one();
    }
}

/// RAII guard for a per-host HTTP semaphore permit.
///
/// Explicitly releases `permit` BEFORE calling `notify.notify_one()` so
/// `dispatcher_loop`'s `try_acquire_owned()` never races ahead of permit drop.
pub(super) struct PermitGuard {
    pub(super) permit: Option<OwnedSemaphorePermit>,
    pub(super) notify: Arc<Notify>,
}

impl Drop for PermitGuard {
    fn drop(&mut self) {
        drop(self.permit.take());
        self.notify.notify_one();
    }
}

#[cfg(test)]
mod tests {
    use veldmap_core::io::CancelToken;
    use veldmap_core::tile::{PlanetaryScheme, QuadTileId};

    use crate::basemap::LiveBasemapSpec;

    use super::*;

    #[test]
    fn stale_inflight_guard_never_removes_newer_generation() {
        let state = Arc::new(Mutex::new(PipelineState::default()));
        let notify = Arc::new(Notify::new());
        let spec = LiveBasemapSpec::eox_sentinel2_cloudless();
        let key = spec.tile_key(QuadTileId {
            scheme: PlanetaryScheme::WebMercator,
            level: 0,
            x: 0,
            y: 0,
        });

        let shut = CancelToken::new();
        let (gen1, _) = {
            let mut st = lock_unpoisoned(&state);
            st.register_http_in_flight(key, &shut)
        };
        let guard1 = InFlightGuard::new(Arc::clone(&state), key, gen1, Arc::clone(&notify));
        let (gen2, _) = {
            let mut st = lock_unpoisoned(&state);
            st.register_http_in_flight(key, &shut)
        };
        drop(guard1);

        let mut st = lock_unpoisoned(&state);
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
