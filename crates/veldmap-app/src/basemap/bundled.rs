//! Virtual offline `z = 0..3` Geodetic/WebMercator + `z = 0` PolarCap basemap
//! provider ([`BundledBasemapProvider`]) serving the embedded public-domain
//! NASA Blue Marble Next Generation asset on the shared `DecodePool`.
//!
//! The embedded JPEG (`3600 x 1800`, ~24.6 MiB decoded) is initialized lazily
//! via `OnceLock` on the first requested tile so `EllipsoidOnly` sessions pay
//! zero decode cost and zero dedicated OS thread overhead.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, OnceLock};

use veldmap_core::sync::{CancelToken, Wakeup, lock_unpoisoned};
use veldmap_core::tile::{GlobeScheme, PlanetaryScheme, QuadTileId, SourceId, TileKey};
use veldmap_raster::{
    BundledBasemap, BundledBasemapConfig, DecodeJob, DecodeJobKey, DecodePoolHandle,
};

use crate::gpu_tiles::UploadSender;

/// Tiling configuration matching [`BUNDLED_BASEMAP_JPEG`].
pub const BUNDLED_BASEMAP_CONFIG: BundledBasemapConfig = BundledBasemapConfig::DEFAULT;

/// Default tile size in pixels for the bundled offline global basemap.
pub(crate) const BUNDLED_TILE_PX: u32 = BUNDLED_BASEMAP_CONFIG.tile_px;

/// Maximum `Geodetic` / `WebMercator` zoom level served by the bundled offline
/// NASA Blue Marble Next Generation asset (`3600 x 1800`).
pub const BUNDLED_MAX_LEVEL: u8 = BUNDLED_BASEMAP_CONFIG.max_level;

/// Maximum `PolarCap` zoom level served by the bundled offline basemap.
pub(crate) const BUNDLED_POLAR_MAX_LEVEL: u8 = BUNDLED_BASEMAP_CONFIG.polar_max_level;

/// Embedded public-domain NASA Blue Marble Next Generation equirectangular
/// global basemap (`3600 x 1800` JPEG, see `assets/README.md`).
pub const BUNDLED_BASEMAP_JPEG: &[u8] =
    include_bytes!("../../../../assets/basemap/blue_marble_ng.jpg");

#[derive(Default)]
struct BundledState {
    next_job_id: u64,
    in_flight: HashMap<TileKey, (u64, CancelToken)>,
    failed: HashSet<TileKey>,
    decode_failed: bool,
}

struct InFlightGuard {
    state: Arc<Mutex<BundledState>>,
    key: TileKey,
    job_id: u64,
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        let mut st = lock_unpoisoned(&self.state);
        if st
            .in_flight
            .get(&self.key)
            .is_some_and(|(id, _)| *id == self.job_id)
        {
            st.in_flight.remove(&self.key);
        }
    }
}

/// Virtual offline `z = 0..3` Geodetic/WebMercator + `z = 0` PolarCap basemap
/// tile provider scheduled onto the shared `veldmap-raster` [`DecodePoolHandle`].
pub(crate) struct BundledBasemapProvider {
    pool: DecodePoolHandle,
    done_tx: UploadSender,
    wakeup: Wakeup,
    shutdown: CancelToken,
    basemap: Arc<OnceLock<Result<BundledBasemap, String>>>,
    state: Arc<Mutex<BundledState>>,
}

impl BundledBasemapProvider {
    /// Create a lazy bundled offline basemap provider backed by `pool`.
    pub(crate) fn new(
        pool: DecodePoolHandle,
        done_tx: UploadSender,
        wakeup: Wakeup,
        shutdown: CancelToken,
    ) -> Self {
        Self {
            pool,
            done_tx,
            wakeup,
            shutdown,
            basemap: Arc::new(OnceLock::new()),
            state: Arc::new(Mutex::new(BundledState::default())),
        }
    }

    pub(crate) fn in_flight_count(&self) -> usize {
        lock_unpoisoned(&self.state).in_flight.len()
    }

    pub(crate) fn is_in_flight(&self, key: &TileKey) -> bool {
        lock_unpoisoned(&self.state).in_flight.contains_key(key)
    }

    #[cfg(test)]
    pub(crate) fn is_basemap_decoded(&self) -> bool {
        self.basemap.get().is_some()
    }

    /// Submit `key` to the shared decode pool if not already in flight or
    /// terminally failed. Returns `true` if a new job was admitted.
    pub(crate) fn request(&self, key: TileKey) -> bool {
        if self.shutdown.is_cancelled() {
            return false;
        }
        let (job_id, job_cancel) = {
            let mut st = lock_unpoisoned(&self.state);
            if st.decode_failed || st.failed.contains(&key) || st.in_flight.contains_key(&key) {
                return false;
            }
            st.next_job_id = st.next_job_id.wrapping_add(1);
            let id = st.next_job_id;
            let cancel = self.shutdown.child_token();
            st.in_flight.insert(key, (id, cancel.clone()));
            (id, cancel)
        };

        let state = Arc::clone(&self.state);
        let basemap = Arc::clone(&self.basemap);
        let done_tx = self.done_tx.clone();
        let wakeup = Arc::clone(&self.wakeup);
        let work_cancel = job_cancel.clone();
        // Construct `InFlightGuard` BEFORE `pool.submit` and move it into the
        // closure so rejection, queue eviction, cancellation, panic, or normal
        // completion all clean up `in_flight` on drop (outside `DecodeQueue`'s
        // mutex).
        let guard = InFlightGuard {
            state: Arc::clone(&state),
            key,
            job_id,
        };

        self.pool.submit(DecodeJob {
            key: DecodeJobKey::Tile(key),
            queued_bytes: 0,
            cancel: job_cancel,
            work: Box::new(move || {
                let _guard = guard;
                if work_cancel.is_cancelled() {
                    return;
                }
                let Some(quad_id) = key.quad_id() else {
                    lock_unpoisoned(&state).failed.insert(key);
                    return;
                };
                let decoded_basemap = basemap.get_or_init(|| {
                    BundledBasemap::from_encoded_bytes_with_config(
                        BUNDLED_BASEMAP_JPEG,
                        BUNDLED_BASEMAP_CONFIG,
                    )
                    .map_err(|err| {
                        tracing::error!(error = %err, "Failed to decode bundled offline basemap");
                        err.to_string()
                    })
                });
                let Ok(map) = decoded_basemap else {
                    lock_unpoisoned(&state).decode_failed = true;
                    return;
                };
                if work_cancel.is_cancelled() {
                    return;
                }
                match map.render_tile_with_mipmaps(quad_id) {
                    Ok(tile) => {
                        if work_cancel.is_cancelled() {
                            return;
                        }
                        if done_tx.send((key, tile)).is_ok() {
                            (wakeup)();
                        }
                    }
                    Err(err) => {
                        tracing::debug!(?key, error = %err, "Skipping out-of-range bundled tile");
                        lock_unpoisoned(&state).failed.insert(key);
                    }
                }
            }),
        })
    }

    /// Cancel and clear in-flight markers for tiles that are already satisfied
    /// (`is_satisfied`), failed terminally, or are no longer requested.
    pub(crate) fn reconcile<F>(&self, wanted: &HashSet<TileKey>, is_satisfied: F)
    where
        F: Fn(&TileKey) -> bool,
    {
        let mut to_cancel = Vec::new();
        {
            let mut st = lock_unpoisoned(&self.state);
            if st.decode_failed {
                for (_, (_, tok)) in st.in_flight.drain() {
                    to_cancel.push(tok);
                }
            } else {
                let failed = &st.failed;
                let remove_keys: Vec<TileKey> = st
                    .in_flight
                    .keys()
                    .copied()
                    .filter(|k| {
                        is_satisfied(k)
                            || failed.contains(k)
                            || (!wanted.contains(k) && !k.is_root_of(SourceId::BUNDLED_BASEMAP))
                    })
                    .collect();
                for k in remove_keys {
                    if let Some((_, tok)) = st.in_flight.remove(&k) {
                        to_cancel.push(tok);
                    }
                }
            }
        }
        for tok in to_cancel {
            tok.cancel();
        }
    }

    /// Pre-seed the coarsest root tiles (`scheme z = 0` + North/South
    /// `PolarCap z = 0`) so the entire planet renders immediately on startup
    /// and offline fallback stays warm.
    pub(crate) fn seed_roots(&self, scheme: GlobeScheme) {
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

    /// Cancel all currently in-flight bundled tile jobs.
    pub(crate) fn cancel_all(&self) {
        let tokens: Vec<CancelToken> = {
            let mut st = lock_unpoisoned(&self.state);
            st.in_flight.drain().map(|(_, (_, tok))| tok).collect()
        };
        for tok in tokens {
            tok.cancel();
        }
    }
}

#[cfg(test)]
#[path = "bundled_tests.rs"]
mod tests;
