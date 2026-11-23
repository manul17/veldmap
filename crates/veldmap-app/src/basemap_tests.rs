use veldmap_core::geo::camera::Camera;
use veldmap_core::tile::{TileScheme, traverse};
use veldmap_storage::StoragePaths;

use super::*;
use crate::gpu_tiles::source_tile_key;
use crate::storage::AppStorage;
use crate::test_support::TempTestRoot;

#[test]
fn ellipsoid_only_does_not_seed_bundled_roots_and_invalid_configs_fall_back() {
    let root = TempTestRoot::new("ellipsoid-and-errors");
    let paths = StoragePaths::with_root(root.path());
    let initial = AppSettings {
        active_basemap: BasemapSelection::EllipsoidOnly,
        ..AppSettings::default()
    };
    initial.save(&paths).expect("save ellipsoid settings");

    let wakeup: Wakeup = Arc::new(|| {});
    let mut storage = AppStorage::open(Arc::clone(&wakeup), Some(paths));
    let mut pending = PendingUploads::new();
    let tx = pending.sender().expect("sender");
    let mut gpu_cache = GpuTileCache::new(64 * 1024 * 1024);
    let mut manager = BasemapManager::new(storage.settings(), storage.disk_cache(), tx, wakeup);

    assert!(!manager.is_textured());
    assert_eq!(manager.attribution(), None);
    assert_eq!(
        manager.bundled_provider.in_flight_count(),
        0,
        "EllipsoidOnly must not seed bundled roots"
    );

    let mut bad_date = initial.clone();
    bad_date.active_basemap = BasemapSelection::NasaGibsViirs;
    bad_date.gibs_date = "2024-02-30".to_owned();
    storage.update_settings(bad_date);
    manager.sync_settings(storage.settings(), &mut gpu_cache, &mut pending);
    assert!(manager.status_snapshot().config_error.is_some());
    assert_eq!(manager.active_source(), SourceId::BUNDLED_BASEMAP);
    assert_eq!(manager.attribution(), Some(BUNDLED_ATTRIBUTION));

    let mut bad_custom = initial;
    bad_custom.active_basemap = BasemapSelection::CustomXyz;
    bad_custom.custom_xyz.url_template = "ftp://bad/{z}/{x}/{y}".to_owned();
    storage.update_settings(bad_custom);
    manager.sync_settings(storage.settings(), &mut gpu_cache, &mut pending);
    assert!(manager.status_snapshot().config_error.is_some());
    assert_eq!(manager.active_source(), SourceId::BUNDLED_BASEMAP);

    manager.begin_shutdown();
    storage.begin_shutdown();
    pending.close();
    manager.finish_shutdown();
    storage.finish_shutdown();
}

#[test]
fn schedule_for_tiles_enforces_polar_exclusion_max_level_and_source_switch_flush() {
    let root = TempTestRoot::new("schedule");
    let paths = StoragePaths::with_root(root.path());
    let wakeup: Wakeup = Arc::new(|| {});
    let mut storage = AppStorage::open(Arc::clone(&wakeup), Some(paths));
    let mut pending = PendingUploads::new();
    let tx = pending.sender().expect("sender");
    let mut gpu_cache = GpuTileCache::new(64 * 1024 * 1024);
    let mut manager =
        BasemapManager::new(storage.settings(), storage.disk_cache(), tx.clone(), wakeup);

    storage.mutate_settings(|s| s.active_basemap = BasemapSelection::NasaGibsBlueMarble);
    manager.sync_settings(storage.settings(), &mut gpu_cache, &mut pending);
    let gibs_source = manager.active_source();
    assert_ne!(gibs_source, SourceId::BUNDLED_BASEMAP);

    let qcfg = manager.quadtree_config();
    let visible = traverse(&Camera::at(-86.0, 0.0, 3_000_000.0), 800, 600, &qcfg);
    manager.schedule_for_tiles(visible.iter().map(|v| v.tex_id), &gpu_cache, &pending, 1);

    assert!(
        manager
            .wanted_live
            .iter()
            .all(|k| !matches!(k.scheme, TileScheme::PolarCap { .. }) && k.level <= 8),
        "wanted_live must never include PolarCap or levels above max_level"
    );
    assert!(
        manager
            .wanted_bundled
            .iter()
            .any(|k| matches!(k.scheme, TileScheme::PolarCap { north: false })),
        "South PolarCap must be scheduled on bundled offline provider"
    );

    let stale_key = source_tile_key(
        gibs_source,
        QuadTileId {
            scheme: PlanetaryScheme::WebMercator,
            level: 1,
            x: 0,
            y: 0,
        },
    );
    tx.send((
        stale_key,
        DecodedTile::try_new(1, 1, PixelFormat::Rgba8UnormSrgb, vec![0, 0, 0, 255])
            .expect("valid tile"),
    ))
    .expect("send stale");
    manager.record_upload_report(&UploadBatchReport {
        uploaded: 1,
        max_active_tile_bytes: 2 * 1024 * 1024,
        ..UploadBatchReport::default()
    });
    assert_eq!(manager.observed_live_tile_bytes, 2 * 1024 * 1024);

    storage.mutate_settings(|s| s.active_basemap = BasemapSelection::EoxSentinel2Cloudless);
    manager.sync_settings(storage.settings(), &mut gpu_cache, &mut pending);
    assert!(!pending.contains(&stale_key));
    assert_eq!(manager.observed_live_tile_bytes, 0);

    manager.begin_shutdown();
    storage.begin_shutdown();
    pending.close();
    manager.finish_shutdown();
    storage.finish_shutdown();
}

#[test]
fn rejected_or_cancelled_decode_request_does_not_report_panic() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::mpsc;

    let (done_tx, _done_rx) = mpsc::channel();
    let shutdown = CancelToken::new();
    let pool = DecodePool::spawn(
        DecodePoolConfig {
            worker_count: 1,
            max_pending_jobs: 1,
            max_pending_bytes: 1024,
        },
        shutdown.clone(),
    );
    let executor = RasterDecodeExecutor {
        pool: pool.handle(),
        done_tx,
        wakeup: Arc::new(|| {}),
        shutdown: shutdown.clone(),
    };

    let cancel = CancelToken::new();
    cancel.cancel();
    let invoked = Arc::new(AtomicBool::new(false));
    let invoked_cb = Arc::clone(&invoked);
    let key = bundled_tile_key(QuadTileId {
        scheme: PlanetaryScheme::WebMercator,
        level: 0,
        x: 0,
        y: 0,
    });
    let accepted = executor.submit(DecodeRequest {
        key,
        bytes: Arc::from([0u8; 4]),
        cancel,
        on_decoded: Box::new(move |_| {
            invoked_cb.store(true, Ordering::SeqCst);
        }),
    });
    assert!(!accepted);
    assert!(
        !invoked.load(Ordering::SeqCst),
        "rejected decode job must not fire on_decoded"
    );
    shutdown.cancel();
    pool.shutdown();
}
