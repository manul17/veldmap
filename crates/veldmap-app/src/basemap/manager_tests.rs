use veldmap_core::budget::MemoryBudget;
use veldmap_core::geo::Camera;
use veldmap_core::tile::{DecodedTile, GlobeScheme, PixelFormat, TileScheme, traverse};
use veldmap_net::test_support::decode_completion_channel;
use veldmap_net::{DecodeFailure, DecodeRequest, NetConfig, NetContext};
use veldmap_raster::{DecodePool, DecodePoolConfig, TwoLaneDecodePool, TwoLaneDecodePoolConfig};
use veldmap_storage::StoragePaths;
use veldmap_storage::test_support::TestTempDir;

use super::super::adapters::RasterDecodeExecutor;
use super::super::policy::{bundled_tile_key, source_tile_key};
use super::*;
use crate::storage::AppStorage;
use crate::storage::settings::AppSettings;

struct BasemapTestHarness {
    _root: Option<TestTempDir>,
    storage: Option<AppStorage>,
    net: Option<Arc<NetContext>>,
    pending: PendingUploads,
    tx: UploadSender,
    decode_shutdown: CancelToken,
    decode_pool: Option<TwoLaneDecodePool>,
    manager: BasemapManager,
}

impl BasemapTestHarness {
    fn with_storage(label: &str, initial: Option<&AppSettings>) -> Self {
        let root = TestTempDir::new(label);
        let paths = StoragePaths::with_root(root.path());
        if let Some(settings) = initial {
            settings
                .save_for_test(&paths)
                .expect("save initial settings");
        }
        let wakeup: Wakeup = Arc::new(|| {});
        let storage = AppStorage::open(Arc::clone(&wakeup), Some(paths));
        let net = Arc::new(NetContext::new(NetConfig::default()).expect("net context"));
        let pending = PendingUploads::new();
        let tx = pending.sender().expect("sender");
        let decode_shutdown = CancelToken::new();
        let decode_pool = TwoLaneDecodePool::spawn(
            TwoLaneDecodePoolConfig::from_budget(&MemoryBudget::default()),
            decode_shutdown.clone(),
        );
        let manager = BasemapManager::new(BasemapManagerDeps {
            config: storage.settings().basemap_config(),
            disk_cache: storage.disk_cache(),
            net: Some(Arc::clone(&net)),
            net_init_error: None,
            done_tx: tx.clone(),
            wakeup,
            decode_handle: decode_pool.fast_handle(),
            decode_shutdown: decode_shutdown.clone(),
        });
        Self {
            _root: Some(root),
            storage: Some(storage),
            net: Some(net),
            pending,
            tx,
            decode_shutdown,
            decode_pool: Some(decode_pool),
            manager,
        }
    }

    fn without_storage(settings: &AppSettings) -> Self {
        let wakeup: Wakeup = Arc::new(|| {});
        let net = Arc::new(NetContext::new(NetConfig::default()).expect("net context"));
        let pending = PendingUploads::new();
        let tx = pending.sender().expect("sender");
        let decode_shutdown = CancelToken::new();
        let decode_pool = TwoLaneDecodePool::spawn(
            TwoLaneDecodePoolConfig::from_budget(&MemoryBudget::default()),
            decode_shutdown.clone(),
        );
        let manager = BasemapManager::new(BasemapManagerDeps {
            config: settings.basemap_config(),
            disk_cache: None,
            net: Some(Arc::clone(&net)),
            net_init_error: None,
            done_tx: tx.clone(),
            wakeup,
            decode_handle: decode_pool.fast_handle(),
            decode_shutdown: decode_shutdown.clone(),
        });
        Self {
            _root: None,
            storage: None,
            net: Some(net),
            pending,
            tx,
            decode_shutdown,
            decode_pool: Some(decode_pool),
            manager,
        }
    }
}

impl Drop for BasemapTestHarness {
    fn drop(&mut self) {
        self.manager.begin_shutdown();
        if let Some(net) = &self.net {
            net.signal_shutdown();
        }
        if let Some(storage) = &mut self.storage {
            storage.begin_shutdown();
        }
        self.decode_shutdown.cancel();
        self.pending.close();
        if let Some(pool) = self.decode_pool.take() {
            pool.shutdown();
        }
        self.manager.finish_shutdown();
        if let Some(net) = self.net.take() {
            net.shutdown();
        }
        if let Some(storage) = &mut self.storage {
            storage.finish_shutdown();
        }
    }
}

#[test]
fn ellipsoid_only_avoids_bundled_decode_and_invalid_configs_fall_back() {
    let initial = AppSettings {
        active_basemap: BasemapSelection::EllipsoidOnly,
        ..AppSettings::default()
    };
    let mut h = BasemapTestHarness::with_storage("ellipsoid-and-errors", Some(&initial));
    let storage = h.storage.as_mut().expect("storage");

    assert!(!h.manager.is_textured());
    assert_eq!(h.manager.attribution(), None);
    assert_eq!(
        h.manager.bundled_provider.in_flight_count(),
        0,
        "EllipsoidOnly must not seed bundled roots"
    );
    assert!(
        !h.manager.bundled_provider.is_basemap_decoded(),
        "EllipsoidOnly must not eagerly decode bundled JPEG"
    );

    storage.mutate_settings(|s| {
        *s = initial.clone();
        s.active_basemap = BasemapSelection::NasaGibsViirs;
        s.gibs_date = "2024-02-30".to_owned();
    });
    let outcome = h.manager.sync_settings(storage.settings().basemap_config());
    assert!(outcome.changed);
    assert_eq!(outcome.evict_source, None);
    assert!(h.manager.status_snapshot().config_error.is_some());
    assert_eq!(h.manager.active_source(), SourceId::BUNDLED_BASEMAP);
    assert_eq!(h.manager.attribution(), Some(BUNDLED_ATTRIBUTION));

    storage.mutate_settings(|s| {
        *s = initial;
        s.active_basemap = BasemapSelection::CustomXyz;
        s.custom_xyz.url_template = "ftp://bad/{z}/{x}/{y}".to_owned();
    });
    let outcome = h.manager.sync_settings(storage.settings().basemap_config());
    assert!(outcome.changed);
    assert!(h.manager.status_snapshot().config_error.is_some());
    assert_eq!(h.manager.active_source(), SourceId::BUNDLED_BASEMAP);
}

#[test]
fn unavailable_live_pipeline_degrades_cleanly_to_full_bundled_basemap() {
    let decode_shutdown = CancelToken::new();
    let decode_pool = DecodePool::spawn(
        DecodePoolConfig {
            worker_count: 1,
            ..DecodePoolConfig::default()
        },
        decode_shutdown.clone(),
    );
    let mut pending = PendingUploads::new();
    let tx = pending.sender().expect("sender");
    let bundled_provider = BundledBasemapProvider::new(
        decode_pool.handle(),
        tx,
        Arc::new(|| {}),
        decode_shutdown.clone(),
    );
    let mut manager = BasemapManager {
        active_basemap: BasemapSelection::EoxSentinel2Cloudless,
        bundled_provider,
        live_pipeline: None,
        live_pipeline_error: Some("simulated TLS init failure".to_owned()),
        active_live_spec: None,
        config_error: None,
        observed_live_tile_bytes: 0,
        wanted_bundled: HashSet::new(),
        wanted_live: HashSet::new(),
    };

    let settings = AppSettings {
        active_basemap: BasemapSelection::EoxSentinel2Cloudless,
        ..AppSettings::default()
    };
    manager.sync_active_spec(settings.basemap_config());

    assert_eq!(manager.active_source(), SourceId::BUNDLED_BASEMAP);
    assert_eq!(manager.attribution(), Some(BUNDLED_ATTRIBUTION));
    let qcfg = manager.quadtree_config();
    assert_eq!(qcfg.scheme, GlobeScheme::Geodetic);
    assert_eq!(qcfg.max_tex_level, BUNDLED_MAX_LEVEL);
    assert!(manager.status_snapshot().live_pipeline_error.is_some());

    let gpu_cache = GpuTileCache::new(64 * 1024 * 1024);
    let visible = traverse(&Camera::at(0.0, 0.0, 1_500_000.0), 800, 600, &qcfg);
    let weight = manager.appetite_weight_bytes();
    let appetite = gpu_cache.proportional_pyramid_byte_cap(weight, weight);
    manager.schedule_for_tiles(
        visible.iter().map(|v| v.tex_id),
        &gpu_cache,
        &pending,
        appetite,
    );
    assert!(manager.wanted_live.is_empty());
    assert!(
        manager.wanted_bundled.iter().any(|k| k.level >= 2),
        "unavailable live pipeline must allow full z=0..3 bundled fallback instead of z<=1 cap"
    );

    manager.begin_shutdown();
    decode_shutdown.cancel();
    pending.close();
    decode_pool.shutdown();
    manager.finish_shutdown();
}

#[test]
fn schedule_for_tiles_enforces_polar_exclusion_max_level_and_source_switch_eviction_intent() {
    let mut h = BasemapTestHarness::with_storage("schedule", None);
    let gpu_cache = GpuTileCache::new(64 * 1024 * 1024);

    let storage = h.storage.as_mut().expect("storage");
    storage.mutate_settings(|s| s.active_basemap = BasemapSelection::NasaGibsBlueMarble);
    let first_switch = h.manager.sync_settings(storage.settings().basemap_config());
    assert!(first_switch.changed);
    assert_eq!(first_switch.evict_source, None);
    let gibs_source = h.manager.active_source();
    assert_ne!(gibs_source, SourceId::BUNDLED_BASEMAP);

    let qcfg = h.manager.quadtree_config();
    let visible = traverse(&Camera::at(-86.0, 0.0, 3_000_000.0), 800, 600, &qcfg);
    let weight = h.manager.appetite_weight_bytes();
    let appetite = gpu_cache.proportional_pyramid_byte_cap(weight, weight);
    h.manager.schedule_for_tiles(
        visible.iter().map(|v| v.tex_id),
        &gpu_cache,
        &h.pending,
        appetite,
    );

    assert!(
        h.manager
            .wanted_live
            .iter()
            .all(|k| !matches!(k.scheme, TileScheme::PolarCap { .. }) && k.level <= 8),
        "wanted_live must never include PolarCap or levels above max_level"
    );
    assert!(
        h.manager
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
    h.tx.send((
        stale_key,
        DecodedTile::try_new(1, 1, PixelFormat::Rgba8UnormSrgb, vec![0, 0, 0, 255])
            .expect("valid tile"),
    ))
    .expect("send stale");
    h.manager.record_upload_report(&UploadBatchReport {
        uploaded: 1,
        max_tile_bytes_by_stream: vec![(StreamKey::display(gibs_source), 2 * 1024 * 1024)],
        ..UploadBatchReport::default()
    });
    assert_eq!(h.manager.observed_live_tile_bytes, 2 * 1024 * 1024);

    storage.mutate_settings(|s| s.active_basemap = BasemapSelection::EoxSentinel2Cloudless);
    let second_switch = h.manager.sync_settings(storage.settings().basemap_config());
    assert!(second_switch.changed);
    assert_eq!(second_switch.evict_source, Some(gibs_source));
    if let Some(evict) = second_switch.evict_source {
        h.pending.clear_source(evict);
    }
    assert!(!h.pending.contains(&stale_key));
    assert_eq!(h.manager.observed_live_tile_bytes, 0);
}

#[test]
fn rejected_or_cancelled_decode_request_signals_aborted() {
    let uploads = PendingUploads::new();
    let done_tx = uploads.sender().expect("sender");
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
    let (on_decoded, mut rx) = decode_completion_channel();
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
        on_decoded,
    });
    assert!(!accepted);
    assert_eq!(
        rx.try_recv(),
        Some(Err(DecodeFailure::Aborted)),
        "rejected decode job must drop its completion handle as Aborted"
    );
    shutdown.cancel();
    pool.shutdown();
}

#[test]
fn in_place_custom_xyz_spec_edit_marks_changed_without_evicting_source() {
    let mut settings = AppSettings {
        active_basemap: BasemapSelection::CustomXyz,
        custom_xyz: crate::storage::settings::CustomXyzSettings {
            name: "Custom Test".to_string(),
            url_template: "https://tiles.example.test/{z}/{x}/{y}.png?key={key}".to_string(),
            max_level: 10,
            api_key: "old-secret".to_string(),
            attribution: "Example v1".to_string(),
            tms_flip_y: false,
        },
        ..AppSettings::default()
    };

    let mut h = BasemapTestHarness::without_storage(&settings);
    let initial_source = h.manager.active_source();
    assert_ne!(initial_source, SourceId::BUNDLED_BASEMAP);

    // Editing max_level / api_key / attribution preserves `SourceId` (derived
    // from template + tms_flip_y) while mutating `LiveBasemapSpec`. Sync must
    // report `changed = true` so `SceneState` invalidates its cached snapshot
    // and re-schedules tiles, without evicting cached tiles for the same source.
    settings.custom_xyz.max_level = 14;
    settings.custom_xyz.api_key = "new-secret".to_string();
    settings.custom_xyz.attribution = "Example v2".to_string();
    let outcome = h.manager.sync_settings(settings.basemap_config());
    assert_eq!(h.manager.active_source(), initial_source);
    assert!(
        outcome.changed,
        "in-place CustomXyz spec edit must mark scene dirty"
    );
    assert_eq!(
        outcome.evict_source, None,
        "same SourceId must not evict cached tiles"
    );
    assert_eq!(h.manager.quadtree_config().max_tex_level, 14);

    // Repeating sync with identical settings must be a no-op.
    let noop = h.manager.sync_settings(settings.basemap_config());
    assert!(!noop.changed);
    assert_eq!(noop.evict_source, None);
}
