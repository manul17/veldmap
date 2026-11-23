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
mod tests {
    use veldmap_core::tile::DecodedTile;
    use veldmap_raster::{DecodePool, DecodePoolConfig};

    use crate::gpu_tiles::PendingUploads;

    use super::*;

    #[test]
    fn reconcile_cancels_and_clears_satisfied_and_terminal_failures() {
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

        let shutdown = CancelToken::new();
        let pool = DecodePool::spawn(
            DecodePoolConfig {
                worker_count: 1,
                ..DecodePoolConfig::default()
            },
            shutdown.clone(),
        );
        let uploads = PendingUploads::new();
        let tx = uploads.sender().expect("sender");
        let provider =
            BundledBasemapProvider::new(pool.handle(), tx, Arc::new(|| {}), shutdown.clone());

        let tok_sat = shutdown.child_token();
        let tok_fail = shutdown.child_token();
        let tok_act = shutdown.child_token();
        {
            let mut st = lock_unpoisoned(&provider.state);
            st.in_flight.insert(k_satisfied, (1, tok_sat.clone()));
            st.in_flight.insert(k_failed, (2, tok_fail.clone()));
            st.in_flight.insert(k_active, (3, tok_act.clone()));
            st.failed.insert(k_failed);
        }

        let wanted = HashSet::from([k_satisfied, k_failed, k_active]);
        provider.reconcile(&wanted, |k| *k == k_satisfied);

        assert!(!provider.is_in_flight(&k_satisfied));
        assert!(tok_sat.is_cancelled());
        assert!(!provider.is_in_flight(&k_failed));
        assert!(tok_fail.is_cancelled());
        assert!(provider.is_in_flight(&k_active));
        assert!(!tok_act.is_cancelled());
        assert_eq!(provider.in_flight_count(), 1);

        shutdown.cancel();
        pool.shutdown();
    }

    #[test]
    fn evicted_bundled_job_clears_in_flight_via_captured_guard() {
        use std::sync::Barrier;

        let shutdown = CancelToken::new();
        let pool = DecodePool::spawn(
            DecodePoolConfig {
                worker_count: 1,
                max_pending_jobs: 1,
                max_pending_bytes: 1024,
            },
            shutdown.clone(),
        );
        let uploads = PendingUploads::new();
        let tx = uploads.sender().expect("sender");
        let provider =
            BundledBasemapProvider::new(pool.handle(), tx, Arc::new(|| {}), shutdown.clone());

        // Block the single worker thread so the next queued job sits in `DecodeQueue`.
        let entered = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));
        let entered_w = Arc::clone(&entered);
        let release_w = Arc::clone(&release);
        let blocker_key = TileKey::display_quad(
            SourceId::new(999),
            QuadTileId {
                scheme: PlanetaryScheme::WebMercator,
                level: 0,
                x: 0,
                y: 0,
            },
        );
        assert!(pool.handle().submit(DecodeJob {
            key: DecodeJobKey::Tile(blocker_key),
            queued_bytes: 0,
            cancel: shutdown.child_token(),
            work: Box::new(move || {
                entered_w.wait();
                release_w.wait();
            }),
        }));
        entered.wait();

        // Submit a finer non-root bundled job (`level = 3`), then evict it with a
        // coarser job (`level = 0`) filling the 1-slot queue.
        let fine_bundled = TileKey::display_quad(
            SourceId::BUNDLED_BASEMAP,
            QuadTileId {
                scheme: PlanetaryScheme::Geodetic,
                level: 3,
                x: 1,
                y: 1,
            },
        );
        assert!(provider.request(fine_bundled));
        assert!(provider.is_in_flight(&fine_bundled));

        let coarse_evictor = TileKey::display_quad(
            SourceId::new(998),
            QuadTileId {
                scheme: PlanetaryScheme::WebMercator,
                level: 0,
                x: 0,
                y: 0,
            },
        );
        assert!(pool.handle().submit(DecodeJob {
            key: DecodeJobKey::Tile(coarse_evictor),
            queued_bytes: 0,
            cancel: shutdown.child_token(),
            work: Box::new(|| {}),
        }));
        assert!(
            !provider.is_in_flight(&fine_bundled),
            "evicted bundled job must clean up in_flight on closure drop"
        );

        release.wait();
        shutdown.cancel();
        pool.shutdown();
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
