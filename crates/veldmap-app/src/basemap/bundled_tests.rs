use std::sync::Barrier;

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
    let mut uploads = PendingUploads::new();
    let tx = uploads.sender().expect("sender");
    let provider = BundledBasemapProvider::new(
        pool.handle(),
        tx,
        Arc::new(|| {}),
        shutdown.clone(),
        Arc::new(PolarRimTracker::new()),
    );

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
    uploads.close();
    pool.shutdown();
}

#[test]
fn evicted_bundled_job_clears_in_flight_via_captured_guard() {
    let shutdown = CancelToken::new();
    let pool = DecodePool::spawn(
        DecodePoolConfig {
            worker_count: 1,
            max_pending_jobs: 1,
            max_pending_bytes: 1024,
        },
        shutdown.clone(),
    );
    let mut uploads = PendingUploads::new();
    let tx = uploads.sender().expect("sender");
    let provider = BundledBasemapProvider::new(
        pool.handle(),
        tx,
        Arc::new(|| {}),
        shutdown.clone(),
        Arc::new(PolarRimTracker::new()),
    );

    // Block the single worker thread so the next queued job sits in `DecodeQueue`.
    // Guard the release barrier with `Drop` so an assertion failure never hangs
    // `DecodePool::drop` while joining the blocked worker.
    struct BarrierReleaseGuard(Option<Arc<Barrier>>);
    impl Drop for BarrierReleaseGuard {
        fn drop(&mut self) {
            if let Some(barrier) = self.0.take() {
                barrier.wait();
            }
        }
    }

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
    let release_guard = BarrierReleaseGuard(Some(release));

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

    drop(release_guard);
    shutdown.cancel();
    uploads.close();
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
        let u = (lon_start / 180.0).clamp(0.0, 0.999);
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
