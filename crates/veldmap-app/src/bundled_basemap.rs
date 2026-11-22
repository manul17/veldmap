//! Virtual offline `z = 0..3` Geodetic/WebMercator + `z = 0` PolarCap basemap
//! worker (`BundledBasemapProvider`) serving the embedded public-domain NASA
//! Blue Marble Next Generation asset.

use std::collections::HashSet;
use std::sync::mpsc::Sender;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

use veldmap_core::io::{Wakeup, lock_unpoisoned};
use veldmap_core::tile::{
    DecodedTile, GlobeScheme, PlanetaryScheme, QuadTileId, SourceId, TileKey,
};
use veldmap_raster::BundledBasemap;

/// Embedded public-domain NASA Blue Marble Next Generation equirectangular
/// global basemap (`3600 x 1800` JPEG, see `assets/README.md`).
pub const BUNDLED_BASEMAP_JPEG: &[u8] =
    include_bytes!("../../../assets/basemap/blue_marble_ng.jpg");

#[derive(Default)]
struct WorkerQueue {
    pending: Vec<TileKey>,
    failed: HashSet<TileKey>,
    shutdown: bool,
}

/// Virtual offline `z = 0..3` Geodetic/WebMercator + `z = 0` PolarCap basemap
/// tile provider backed by a dedicated background worker thread, coarsest-first
/// cancellable queue, and injected [`Wakeup`].
pub struct BundledBasemapProvider {
    shared: Arc<(Mutex<WorkerQueue>, Condvar)>,
    in_flight: HashSet<TileKey>,
    worker: Option<JoinHandle<()>>,
}

impl BundledBasemapProvider {
    /// Spawn the bundled offline basemap provider pushing mipmapped decoded
    /// tiles into `done_tx` and invoking `wakeup` on each arrival.
    pub fn spawn(done_tx: Sender<(TileKey, DecodedTile)>, wakeup: Wakeup) -> Self {
        let shared = Arc::new((Mutex::new(WorkerQueue::default()), Condvar::new()));
        let worker_shared = Arc::clone(&shared);

        let worker = match thread::Builder::new()
            .name("veldmap-bundled-basemap".to_owned())
            .spawn(move || {
                let basemap = match BundledBasemap::from_encoded_bytes(BUNDLED_BASEMAP_JPEG) {
                    Ok(b) => b,
                    Err(err) => {
                        tracing::error!(error = %err, "Failed to decode bundled offline basemap");
                        let (lock, _) = &*worker_shared;
                        let mut state = lock_unpoisoned(lock);
                        state.shutdown = true;
                        state.pending.clear();
                        return;
                    }
                };
                loop {
                    let key = {
                        let (lock, cvar) = &*worker_shared;
                        let mut state = lock_unpoisoned(lock);
                        while state.pending.is_empty() && !state.shutdown {
                            state = match cvar.wait(state) {
                                Ok(guard) => guard,
                                Err(poisoned) => poisoned.into_inner(),
                            };
                        }
                        if state.shutdown {
                            break;
                        }
                        let best_idx = state
                            .pending
                            .iter()
                            .enumerate()
                            .min_by_key(|(_, k)| k.coarsest_first_key())
                            .map(|(i, _)| i);
                        match best_idx {
                            Some(i) => state.pending.swap_remove(i),
                            None => continue,
                        }
                    };

                    let Some(quad_id) = key.quad_id() else {
                        let (lock, _) = &*worker_shared;
                        lock_unpoisoned(lock).failed.insert(key);
                        continue;
                    };
                    match basemap.render_tile_with_mipmaps(quad_id) {
                        Ok(tile) => {
                            if done_tx.send((key, tile)).is_err() {
                                break;
                            }
                            (wakeup)();
                        }
                        Err(err) => {
                            tracing::debug!(?key, error = %err, "Skipping out-of-range bundled tile");
                            let (lock, _) = &*worker_shared;
                            lock_unpoisoned(lock).failed.insert(key);
                        }
                    }
                }
            }) {
            Ok(handle) => Some(handle),
            Err(err) => {
                tracing::error!(error = %err, "Failed to spawn bundled basemap worker thread");
                None
            }
        };

        Self {
            shared,
            in_flight: HashSet::new(),
            worker,
        }
    }

    pub fn in_flight_count(&self) -> usize {
        self.in_flight.len()
    }

    pub fn is_in_flight(&self, key: &TileKey) -> bool {
        self.in_flight.contains(key)
    }

    /// Enqueue `key` for background tile rendering if not already in flight or
    /// terminally failed. Returns `true` if a new request was submitted.
    pub fn request(&mut self, key: TileKey) -> bool {
        if self.worker.is_none() || self.in_flight.contains(&key) {
            return false;
        }
        let (lock, cvar) = &*self.shared;
        let mut state = lock_unpoisoned(lock);
        if state.shutdown || state.failed.contains(&key) {
            return false;
        }
        self.in_flight.insert(key);
        state.pending.push(key);
        cvar.notify_one();
        true
    }

    /// Clear in-flight markers and cancel unstarted queued requests for tiles
    /// that are already satisfied (`is_satisfied`), failed terminally, or are
    /// no longer requested.
    pub fn reconcile<F>(&mut self, wanted: &HashSet<TileKey>, is_satisfied: F)
    where
        F: Fn(&TileKey) -> bool,
    {
        let (lock, _) = &*self.shared;
        let mut state = lock_unpoisoned(lock);
        let WorkerQueue {
            pending,
            failed,
            shutdown,
        } = &mut *state;
        if *shutdown {
            self.in_flight.clear();
            pending.clear();
            return;
        }
        let keep = |k: &TileKey| {
            !is_satisfied(k)
                && !failed.contains(k)
                && (wanted.contains(k) || k.is_root_of(SourceId::BUNDLED_BASEMAP))
        };
        self.in_flight.retain(keep);
        pending.retain(keep);
    }

    /// Pre-seed the coarsest root tiles (`scheme z = 0` + North/South
    /// `PolarCap z = 0`) so the entire planet renders immediately on startup
    /// and offline fallback stays warm.
    pub fn seed_roots(&mut self, scheme: GlobeScheme) {
        let planetary = scheme.into_planetary();
        if let Some((cols, rows)) = planetary.grid_dims(0) {
            for y in 0..rows {
                for x in 0..cols {
                    self.request(TileKey::display_quad(
                        SourceId::BUNDLED_BASEMAP,
                        QuadTileId {
                            scheme: planetary,
                            level: 0,
                            x,
                            y,
                        },
                    ));
                }
            }
        }
        for north in [true, false] {
            self.request(TileKey::display_quad(
                SourceId::BUNDLED_BASEMAP,
                QuadTileId {
                    scheme: PlanetaryScheme::PolarCap { north },
                    level: 0,
                    x: 0,
                    y: 0,
                },
            ));
        }
    }

    /// Signal shutdown without joining yet (paired with [`Self::join`]).
    pub fn signal_shutdown(&mut self) {
        let (lock, cvar) = &*self.shared;
        let mut state = lock_unpoisoned(lock);
        state.shutdown = true;
        state.pending.clear();
        cvar.notify_all();
    }

    /// Join the worker thread deterministically.
    pub fn join(&mut self) {
        self.signal_shutdown();
        if let Some(handle) = self.worker.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for BundledBasemapProvider {
    fn drop(&mut self) {
        self.join();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reconcile_clears_satisfied_and_terminal_failures_from_in_flight() {
        let k_satisfied = TileKey::display_quad(
            SourceId::BUNDLED_BASEMAP,
            QuadTileId {
                scheme: PlanetaryScheme::Geodetic,
                level: 1,
                x: 0,
                y: 0,
            },
        );
        let k_failed = TileKey::display_quad(
            SourceId::BUNDLED_BASEMAP,
            QuadTileId {
                scheme: PlanetaryScheme::Geodetic,
                level: 9,
                x: 0,
                y: 0,
            },
        );
        let k_active = TileKey::display_quad(
            SourceId::BUNDLED_BASEMAP,
            QuadTileId {
                scheme: PlanetaryScheme::Geodetic,
                level: 1,
                x: 1,
                y: 0,
            },
        );

        let shared = Arc::new((Mutex::new(WorkerQueue::default()), Condvar::new()));
        {
            let mut st = lock_unpoisoned(&shared.0);
            st.pending.extend([k_satisfied, k_failed, k_active]);
            st.failed.insert(k_failed);
        }
        let mut provider = BundledBasemapProvider {
            shared,
            in_flight: HashSet::from([k_satisfied, k_failed, k_active]),
            worker: None,
        };

        let wanted = HashSet::from([k_satisfied, k_failed, k_active]);
        provider.reconcile(&wanted, |k| *k == k_satisfied);

        assert!(!provider.is_in_flight(&k_satisfied));
        assert!(!provider.is_in_flight(&k_failed));
        assert!(provider.is_in_flight(&k_active));
        assert_eq!(provider.in_flight_count(), 1);
    }
}
