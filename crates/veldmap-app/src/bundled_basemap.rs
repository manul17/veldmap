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
use veldmap_raster::{BundledBasemap, BundledBasemapConfig};

/// Default tile size in pixels for the bundled offline global basemap.
pub const BUNDLED_TILE_PX: u32 = 256;

/// Maximum `Geodetic` / `WebMercator` zoom level served by the bundled offline
/// NASA Blue Marble Next Generation asset (`3600 x 1800`).
pub const BUNDLED_MAX_LEVEL: u8 = 3;

/// Maximum `PolarCap` zoom level served by the bundled offline basemap.
pub const BUNDLED_POLAR_MAX_LEVEL: u8 = 0;

/// Tiling configuration matching [`BUNDLED_BASEMAP_JPEG`].
pub const BUNDLED_BASEMAP_CONFIG: BundledBasemapConfig = BundledBasemapConfig {
    tile_px: BUNDLED_TILE_PX,
    max_level: BUNDLED_MAX_LEVEL,
    polar_max_level: BUNDLED_POLAR_MAX_LEVEL,
};

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
                let basemap = match BundledBasemap::from_encoded_bytes_with_config(
                    BUNDLED_BASEMAP_JPEG,
                    BUNDLED_BASEMAP_CONFIG,
                ) {
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

    #[test]
    fn bundled_blue_marble_asset_decodes_and_preserves_geographic_landmarks() {
        let basemap = BundledBasemap::from_encoded_bytes_with_config(
            BUNDLED_BASEMAP_JPEG,
            BUNDLED_BASEMAP_CONFIG,
        )
        .expect("decode embedded blue marble");
        assert_eq!(basemap.dimensions(), (3600, 1800));

        // Sample z=0 Geodetic tiles and confirm polar Antarctica, Sahara, and
        // central Pacific landmarks preserve expected colors.
        let west = basemap
            .render_tile(QuadTileId {
                scheme: PlanetaryScheme::Geodetic,
                level: 0,
                x: 0,
                y: 0,
            })
            .expect("west hemisphere");
        let east = basemap
            .render_tile(QuadTileId {
                scheme: PlanetaryScheme::Geodetic,
                level: 0,
                x: 1,
                y: 0,
            })
            .expect("east hemisphere");
        let px = |tile: &DecodedTile, lat: f64, lon_start: f64| -> [u8; 4] {
            let u = ((lon_start) / 180.0).clamp(0.0, 0.999);
            let v = ((90.0 - lat) / 180.0).clamp(0.0, 0.999);
            let x = (u * f64::from(tile.width())) as usize;
            let y = (v * f64::from(tile.height())) as usize;
            let off = (y * tile.width() as usize + x) * 4;
            let d = tile.data();
            [d[off], d[off + 1], d[off + 2], d[off + 3]]
        };

        let antarctica = px(&east, -85.0, 0.0);
        assert!(antarctica[0] > 200 && antarctica[1] > 200 && antarctica[2] > 200);
        let sahara = px(&east, 23.0, 13.0);
        assert!(sahara[0] > 150 && sahara[0] > sahara[2].saturating_add(40));
        let pacific = px(&west, 0.0, 30.0); // -150 deg relative to -180
        assert!(pacific[2] > pacific[0].saturating_add(20) && pacific[0] < 40);
    }
}
