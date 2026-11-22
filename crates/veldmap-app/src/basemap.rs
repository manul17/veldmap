//! Live + offline basemap orchestration extracted from `app.rs`.
//!
//! Coordinates the bundled offline NASA Blue Marble worker
//! ([`BundledBasemapProvider`]) and the async HTTP streaming + CPU decode
//! pipeline (`LiveTilePipeline`), enforcing:
//! - `0` live HTTP requests for `PolarCap` keys (always routed to bundled
//!   `SourceId::BUNDLED_BASEMAP` offline polar caps)
//! - `0` live HTTP requests above per-source `max_level`
//! - Seamless Frame-1 & offline fallback to bundled Blue Marble tiles
//! - Shared `< 50%` VRAM appetite capping across bundled + live tiles
//! - Deterministic shutdown (`CancelToken` -> close `PendingUploads` rx -> join
//!   bundled/decode threads -> shut down `tokio::Runtime`)

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::mpsc::Sender;

use veldmap_core::tile::{
    DecodedTile, PixelFormat, QuadtreeConfig, SourceId, TileKey, TileScheme, max_mip_levels,
};
use veldmap_net::{
    CustomBasemapConfig, FetcherConfig, LiveBasemapSpec, LiveTilePipeline, NetError, NetworkStatus,
    TileBlobStore, Wakeup,
};
use veldmap_raster::{BUNDLED_MAX_LEVEL, BUNDLED_POLAR_MAX_LEVEL, BUNDLED_TILE_PX};
use veldmap_storage::{AppSettings, BasemapSelection};

use crate::bundled_basemap::BundledBasemapProvider;
use crate::gpu_tiles::{
    GpuTileCache, PendingUploads, UploadBatchReport, bundled_tile_key, source_tile_key,
    texture_byte_size,
};
use crate::render::DrawSnapshot;

/// Attribution string displayed for the bundled public-domain Blue Marble
/// offline basemap.
pub const BUNDLED_ATTRIBUTION: &str = "Imagery: NASA Blue Marble Next Generation (Public Domain)";

/// Resolve the active [`LiveBasemapSpec`] from persisted [`AppSettings`],
/// returning `Ok(None)` for offline (`BundledBlueMarble`) or `EllipsoidOnly`
/// modes.
pub fn spec_from_settings(settings: &AppSettings) -> Result<Option<LiveBasemapSpec>, NetError> {
    match settings.active_basemap {
        BasemapSelection::BundledBlueMarble | BasemapSelection::EllipsoidOnly => Ok(None),
        BasemapSelection::NasaGibsBlueMarble => Ok(Some(LiveBasemapSpec::nasa_gibs_blue_marble())),
        BasemapSelection::NasaGibsViirs => {
            LiveBasemapSpec::nasa_gibs_viirs(&settings.gibs_date).map(Some)
        }
        BasemapSelection::EoxSentinel2Cloudless => {
            Ok(Some(LiveBasemapSpec::eox_sentinel2_cloudless()))
        }
        BasemapSelection::CustomXyz => LiveBasemapSpec::custom(CustomBasemapConfig {
            name: &settings.custom_xyz.name,
            url_template: &settings.custom_xyz.url_template,
            api_key: &settings.custom_xyz.api_key,
            tms_flip_y: settings.custom_xyz.tms_flip_y,
            max_level: settings.custom_xyz.max_level,
            attribution: &settings.custom_xyz.attribution,
        })
        .map(Some),
    }
}

/// Coordinates offline bundled + live HTTP basemap streaming without owning
/// application-wide storage or settings state.
pub struct BasemapManager {
    active_basemap: BasemapSelection,
    bundled_provider: BundledBasemapProvider,
    live_pipeline: Option<LiveTilePipeline>,
    live_pipeline_error: Option<String>,
    active_live_spec: Option<LiveBasemapSpec>,
    config_error: Option<String>,
    observed_live_tile_bytes: usize,
    wanted_bundled: HashSet<TileKey>,
    wanted_live: HashSet<TileKey>,
    wanted_all: HashSet<TileKey>,
}

impl BasemapManager {
    /// Initialize basemap orchestration using initial `settings` and an
    /// optional shared [`TileBlobStore`] from [`crate::storage::AppStorage`].
    pub fn new(
        settings: &AppSettings,
        blob_store: Option<Arc<dyn TileBlobStore>>,
        done_tx: Sender<(TileKey, DecodedTile)>,
        wakeup: Wakeup,
    ) -> Self {
        let bundled_provider = BundledBasemapProvider::spawn(done_tx.clone(), Arc::clone(&wakeup));

        let (live_pipeline, live_pipeline_error) = match LiveTilePipeline::spawn(
            FetcherConfig::default(),
            blob_store,
            done_tx,
            wakeup,
        ) {
            Ok(pipeline) => (Some(pipeline), None),
            Err(err) => {
                tracing::error!(error = %err, "Failed to start live HTTP tile pipeline; falling back to offline basemap");
                (None, Some(err.to_string()))
            }
        };

        let mut manager = Self {
            active_basemap: settings.active_basemap,
            bundled_provider,
            live_pipeline,
            live_pipeline_error,
            active_live_spec: None,
            config_error: None,
            observed_live_tile_bytes: 0,
            wanted_bundled: HashSet::new(),
            wanted_live: HashSet::new(),
            wanted_all: HashSet::new(),
        };

        manager.sync_active_spec(settings);
        if manager.is_textured() {
            let scheme = manager.quadtree_config().scheme;
            manager.bundled_provider.seed_roots(scheme);
        }
        manager
    }

    /// Whether the globe should render textured imagery (`false` for
    /// `EllipsoidOnly`).
    pub fn is_textured(&self) -> bool {
        self.active_basemap != BasemapSelection::EllipsoidOnly
    }

    /// Active primary [`SourceId`] passed into [`DrawSnapshot::build`] and
    /// [`PendingUploads::drain_and_upload`].
    pub fn active_source(&self) -> SourceId {
        match &self.active_live_spec {
            Some(spec) => spec.source_id,
            None => SourceId::BUNDLED_BASEMAP,
        }
    }

    /// Active planetary [`QuadtreeConfig`] matched to the selected basemap.
    pub fn quadtree_config(&self) -> QuadtreeConfig {
        match &self.active_live_spec {
            Some(spec) => QuadtreeConfig::web_mercator(spec.max_level),
            None => QuadtreeConfig::bundled_geodetic(),
        }
    }

    /// Visible attribution string for the active basemap (`None` for
    /// `EllipsoidOnly`).
    pub fn attribution(&self) -> Option<&str> {
        match self.active_basemap {
            BasemapSelection::EllipsoidOnly => None,
            BasemapSelection::BundledBlueMarble => Some(BUNDLED_ATTRIBUTION),
            _ => match &self.active_live_spec {
                Some(spec) => Some(spec.attribution.as_str()),
                None => Some(BUNDLED_ATTRIBUTION),
            },
        }
    }

    pub fn config_error(&self) -> Option<&str> {
        self.config_error.as_deref()
    }

    /// Error recorded if the live HTTP pipeline failed to initialize at startup.
    pub fn live_pipeline_error(&self) -> Option<&str> {
        self.live_pipeline_error.as_deref()
    }

    /// Combined wanted set (active live + bundled/polar fallback) for
    /// [`PendingUploads::drain_and_upload`].
    pub fn wanted_tiles(&self) -> &HashSet<TileKey> {
        &self.wanted_all
    }

    /// Observable live streaming status snapshot.
    pub fn network_status(&self) -> NetworkStatus {
        self.live_pipeline
            .as_ref()
            .map(LiveTilePipeline::status)
            .unwrap_or_default()
    }

    /// Update observed live tile byte size from an upload batch so VRAM
    /// appetite budgeting accounts for 512 px / `@2x` Custom XYZ tiles cleanly.
    pub fn record_upload_report(&mut self, report: &UploadBatchReport) {
        if report.max_active_tile_bytes > 0 && self.active_live_spec.is_some() {
            self.observed_live_tile_bytes = self
                .observed_live_tile_bytes
                .max(report.max_active_tile_bytes);
        }
    }

    /// Retry failed/offline live tile requests immediately.
    pub fn retry_live(&self) {
        if let Some(pipeline) = &self.live_pipeline {
            pipeline.retry_failed();
        }
    }

    /// Synchronize active basemap providers with updated [`AppSettings`],
    /// evicting GPU tiles from the previous live source if changed and
    /// switching the live pipeline. Returns `true` when the active basemap
    /// source or mode changed.
    pub fn sync_settings(
        &mut self,
        settings: &AppSettings,
        gpu_cache: &mut GpuTileCache,
        pending_uploads: &mut PendingUploads,
    ) -> bool {
        let prev_source = self.active_source();
        let prev_mode = self.active_basemap;
        self.active_basemap = settings.active_basemap;

        self.sync_active_spec(settings);
        let next_source = self.active_source();

        if prev_source != next_source {
            // Keep bundled fallback tiles warm in VRAM; purge stale tiles from
            // the previous live source so VRAM stays clean across switches.
            if prev_source != SourceId::BUNDLED_BASEMAP {
                gpu_cache.remove_source(prev_source);
                pending_uploads.clear_source(prev_source);
            }
            self.observed_live_tile_bytes = 0;
            self.wanted_live.clear();
            self.wanted_bundled.clear();
            self.wanted_all.clear();
            if self.is_textured() {
                self.bundled_provider
                    .seed_roots(self.quadtree_config().scheme);
            }
        } else if prev_mode != self.active_basemap && self.is_textured() {
            self.bundled_provider
                .seed_roots(self.quadtree_config().scheme);
        }

        prev_source != next_source || prev_mode != self.active_basemap
    }

    fn sync_active_spec(&mut self, settings: &AppSettings) {
        match spec_from_settings(settings) {
            Ok(spec) => {
                self.config_error = None;
                self.active_live_spec = spec.clone();
                if let Some(pipeline) = &self.live_pipeline {
                    pipeline.set_active_spec(spec);
                }
            }
            Err(err) => {
                self.config_error = Some(err.to_string());
                self.active_live_spec = None;
                if let Some(pipeline) = &self.live_pipeline {
                    pipeline.set_active_spec(None);
                }
            }
        }
    }

    /// Schedule missing visible tiles for both the bundled offline provider
    /// (polar caps + Frame-1/offline fallback) and the active live HTTP
    /// pipeline, sharing this basemap's `< 50%` VRAM byte appetite allocation
    /// (out of `active_pyramids`) and enforcing `0` HTTP requests for
    /// `PolarCap` and `0` HTTP requests above `max_level`.
    pub fn schedule_for_snapshot(
        &mut self,
        snapshot: &DrawSnapshot,
        gpu_cache: &GpuTileCache,
        pending_uploads: &PendingUploads,
        active_pyramids: usize,
    ) {
        self.wanted_bundled.clear();
        self.wanted_live.clear();
        self.wanted_all.clear();

        if !self.is_textured() {
            self.reconcile_providers(gpu_cache, pending_uploads);
            return;
        }

        let mips = max_mip_levels(BUNDLED_TILE_PX, BUNDLED_TILE_PX);
        let default_tile_bytes = texture_byte_size(
            BUNDLED_TILE_PX,
            BUNDLED_TILE_PX,
            PixelFormat::Rgba8UnormSrgb,
            mips,
        )
        .max(1);
        let live_tile_bytes = default_tile_bytes.max(self.observed_live_tile_bytes);
        let total_appetite_bytes = gpu_cache.pyramid_byte_cap(active_pyramids.max(1));

        let mut missing_bundled = Vec::new();
        let mut cached_bundled_wanted = 0usize;
        let mut live_candidates = Vec::new();
        let mut cached_live_wanted = 0usize;
        let mut queued_live_wanted = 0usize;

        if let Some(spec) = &self.active_live_spec {
            let mut seen_live = HashSet::new();

            for draw in &snapshot.tiles {
                let tex_id = draw.visible.tex_id;
                if matches!(tex_id.scheme, TileScheme::PolarCap { .. }) {
                    let (polar_id, _) = tex_id.clamp_to_level(BUNDLED_POLAR_MAX_LEVEL);
                    let key = bundled_tile_key(polar_id);
                    if self.wanted_bundled.insert(key) {
                        if gpu_cache.contains(&key) {
                            cached_bundled_wanted += 1;
                        } else if !pending_uploads.contains(&key)
                            && !self.bundled_provider.is_in_flight(&key)
                        {
                            missing_bundled.push(key);
                        }
                    }
                } else {
                    let (clamped_live, _) = tex_id.clamp_to_level(spec.max_level);
                    let live_key = source_tile_key(spec.source_id, clamped_live);
                    if seen_live.insert(live_key) {
                        if gpu_cache.contains(&live_key) {
                            cached_live_wanted += 1;
                            self.wanted_live.insert(live_key);
                        } else if pending_uploads.contains(&live_key) {
                            queued_live_wanted += 1;
                            self.wanted_live.insert(live_key);
                        } else {
                            live_candidates.push(live_key);
                        }
                    }

                    if !gpu_cache.has_carrier(&live_key) {
                        let (fb_id, _) = tex_id.clamp_to_level(BUNDLED_MAX_LEVEL.min(1));
                        let fb_key = bundled_tile_key(fb_id);
                        if self.wanted_bundled.insert(fb_key) {
                            if gpu_cache.contains(&fb_key) {
                                cached_bundled_wanted += 1;
                            } else if !pending_uploads.contains(&fb_key)
                                && !self.bundled_provider.is_in_flight(&fb_key)
                            {
                                missing_bundled.push(fb_key);
                            }
                        }
                    }
                }
            }
        } else {
            for draw in &snapshot.tiles {
                let key = bundled_tile_key(draw.visible.tex_id);
                if self.wanted_bundled.insert(key) {
                    if gpu_cache.contains(&key) {
                        cached_bundled_wanted += 1;
                    } else if !pending_uploads.contains(&key)
                        && !self.bundled_provider.is_in_flight(&key)
                    {
                        missing_bundled.push(key);
                    }
                }
            }
        }

        let bundled_queued = pending_uploads.count_for_source(SourceId::BUNDLED_BASEMAP);
        let bundled_committed_tiles = cached_bundled_wanted
            .saturating_add(self.bundled_provider.in_flight_count())
            .saturating_add(bundled_queued);
        let bundled_committed_bytes = bundled_committed_tiles.saturating_mul(default_tile_bytes);
        let bundled_room_tiles =
            total_appetite_bytes.saturating_sub(bundled_committed_bytes) / default_tile_bytes;

        missing_bundled.sort_by_key(TileKey::coarsest_first_key);
        let bundled_to_request: Vec<TileKey> = missing_bundled
            .into_iter()
            .take(bundled_room_tiles)
            .collect();
        let bundled_total_bytes = bundled_committed_tiles
            .saturating_add(bundled_to_request.len())
            .saturating_mul(default_tile_bytes);

        if self.active_live_spec.is_some() {
            let live_appetite_bytes = total_appetite_bytes.saturating_sub(bundled_total_bytes);
            let live_tile_cap = live_appetite_bytes / live_tile_bytes;
            let live_budget = live_tile_cap
                .saturating_sub(cached_live_wanted)
                .saturating_sub(queued_live_wanted);
            live_candidates.sort_by_key(TileKey::coarsest_first_key);
            for k in live_candidates.into_iter().take(live_budget) {
                self.wanted_live.insert(k);
            }
        }

        self.wanted_all.extend(self.wanted_live.iter().copied());
        self.wanted_all.extend(self.wanted_bundled.iter().copied());

        self.reconcile_providers(gpu_cache, pending_uploads);

        for key in bundled_to_request {
            self.bundled_provider.request(key);
        }
    }

    /// Reconcile both background providers after GPU tile uploads complete.
    pub fn reconcile_providers(
        &mut self,
        gpu_cache: &GpuTileCache,
        pending_uploads: &PendingUploads,
    ) {
        self.bundled_provider
            .reconcile(gpu_cache, pending_uploads, &self.wanted_bundled);
        if let Some(pipeline) = &self.live_pipeline {
            pipeline.reconcile_wanted(&self.wanted_live, |k| {
                gpu_cache.contains(k) || pending_uploads.contains(k)
            });
        }
    }

    /// Phase 1 of deterministic shutdown: cancel `CancelToken`s immediately
    /// and signal background workers BEFORE dropping `PendingUploads`'s rx.
    pub fn begin_shutdown(&mut self) {
        self.bundled_provider.signal_shutdown();
        if let Some(pipeline) = &self.live_pipeline {
            pipeline.signal_shutdown();
        }
    }

    /// Phase 2 of deterministic shutdown (after `PendingUploads::close()`):
    /// join CPU decode/bundled threads and shut down `tokio::Runtime`.
    pub fn finish_shutdown(&mut self) {
        self.bundled_provider.join();
        if let Some(pipeline) = self.live_pipeline.take() {
            pipeline.shutdown();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    use veldmap_core::geo::camera::Camera;
    use veldmap_core::tile::QuadTileId;
    use veldmap_storage::StoragePaths;

    use super::*;
    use crate::storage::AppStorage;

    struct TempRoot(PathBuf);

    impl TempRoot {
        fn new(label: &str) -> Self {
            static SEQ: AtomicU64 = AtomicU64::new(0);
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos());
            let seq = SEQ.fetch_add(1, Ordering::Relaxed);
            let dir = std::env::temp_dir().join(format!(
                "veldmap-basemap-test-{label}-{}-{nanos}-{seq}",
                std::process::id()
            ));
            std::fs::create_dir_all(&dir).expect("mkdir");
            Self(dir)
        }
    }

    impl Drop for TempRoot {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn ellipsoid_only_does_not_seed_bundled_roots_and_invalid_configs_fall_back() {
        let root = TempRoot::new("ellipsoid-and-errors");
        let paths = StoragePaths::with_root(&root.0);
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
        let mut manager =
            BasemapManager::new(storage.settings(), storage.tile_blob_store(), tx, wakeup);

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
        assert!(manager.config_error().is_some());
        assert_eq!(manager.active_source(), SourceId::BUNDLED_BASEMAP);
        assert_eq!(manager.attribution(), Some(BUNDLED_ATTRIBUTION));

        let mut bad_custom = initial;
        bad_custom.active_basemap = BasemapSelection::CustomXyz;
        bad_custom.custom_xyz.url_template = "ftp://bad/{z}/{x}/{y}".to_owned();
        storage.update_settings(bad_custom);
        manager.sync_settings(storage.settings(), &mut gpu_cache, &mut pending);
        assert!(manager.config_error().is_some());
        assert_eq!(manager.active_source(), SourceId::BUNDLED_BASEMAP);

        manager.begin_shutdown();
        storage.begin_shutdown();
        pending.close();
        manager.finish_shutdown();
        storage.finish_shutdown();
    }

    #[test]
    fn schedule_for_snapshot_enforces_polar_exclusion_max_level_and_source_switch_flush() {
        let root = TempRoot::new("schedule");
        let paths = StoragePaths::with_root(&root.0);
        let wakeup: Wakeup = Arc::new(|| {});
        let mut storage = AppStorage::open(Arc::clone(&wakeup), Some(paths));
        let mut pending = PendingUploads::new();
        let tx = pending.sender().expect("sender");
        let mut gpu_cache = GpuTileCache::new(64 * 1024 * 1024);
        let mut manager = BasemapManager::new(
            storage.settings(),
            storage.tile_blob_store(),
            tx.clone(),
            wakeup,
        );

        storage.mutate_settings(|s| s.active_basemap = BasemapSelection::NasaGibsBlueMarble);
        manager.sync_settings(storage.settings(), &mut gpu_cache, &mut pending);
        let gibs_source = manager.active_source();
        assert_ne!(gibs_source, SourceId::BUNDLED_BASEMAP);

        let qcfg = manager.quadtree_config();
        let snap = DrawSnapshot::build(
            Camera::at(-86.0, 0.0, 3_000_000.0),
            800,
            600,
            1.0,
            false,
            true,
            gibs_source,
            1,
            &qcfg,
            Some(&mut gpu_cache),
        );
        manager.schedule_for_snapshot(&snap, &gpu_cache, &pending, 1);

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
                scheme: TileScheme::WebMercator,
                level: 1,
                x: 0,
                y: 0,
            },
        );
        tx.send((
            stale_key,
            DecodedTile::new(1, 1, PixelFormat::Rgba8UnormSrgb, vec![0, 0, 0, 255]),
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
}
