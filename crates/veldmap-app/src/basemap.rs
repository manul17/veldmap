//! Live + offline basemap orchestration extracted from `app.rs`.
//!
//! Coordinates durable user settings (`AppSettings`), crash-safe SQLite WAL
//! disk caching (`TileDiskCache`, accessed asynchronously off the UI thread via
//! `DiskCacheWorker`), the bundled offline NASA Blue Marble worker
//! (`BundledBasemapProvider`), and the async HTTP streaming + CPU decode
//! pipeline (`LiveTilePipeline`), enforcing:
//! - `0` live HTTP requests for `PolarCap` keys (always routed to bundled
//!   `SourceId::BUNDLED_BASEMAP` offline polar caps)
//! - `0` live HTTP requests above per-source `max_level`
//! - Seamless Frame-1 & offline fallback to bundled Blue Marble tiles
//! - Shared `< 50%` VRAM appetite capping across bundled + live tiles
//! - Deterministic shutdown (`CancelToken` -> close `PendingUploads` rx -> join
//!   bundled/decode/disk-cache threads -> shut down `tokio::Runtime`)

use std::collections::HashSet;
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use veldmap_core::io::lock_unpoisoned;
use veldmap_core::tile::{
    DecodedTile, PixelFormat, QuadTileId, QuadtreeConfig, SourceId, TileKey, TileScheme,
    max_mip_levels,
};
use veldmap_net::{
    CustomBasemapConfig, FetcherConfig, LiveBasemapSpec, LiveTilePipeline, NetError, NetworkStatus,
    TileBlobStore,
};
use veldmap_raster::{BUNDLED_MAX_LEVEL, BUNDLED_POLAR_MAX_LEVEL, BUNDLED_TILE_PX};
use veldmap_storage::{
    AppSettings, BasemapSelection, DiskCacheStats, DiskTileKey, StoragePaths, TileDiskCache,
};

use crate::gpu_tiles::{
    BundledBasemapProvider, GpuTileCache, PendingUploads, UploadBatchReport, Wakeup,
    bundled_tile_key, source_tile_key, texture_byte_size,
};
use crate::render::DrawSnapshot;

/// Attribution string displayed for the bundled public-domain Blue Marble
/// offline basemap.
pub const BUNDLED_ATTRIBUTION: &str = "Imagery: NASA Blue Marble Next Generation (Public Domain)";

/// Minimum interval between event-driven disk-cache stats refreshes while the
/// Settings window is open and new tiles are streaming in.
const DISK_STATS_REFRESH_INTERVAL: Duration = Duration::from_secs(1);

/// Adapter bridging `veldmap-storage`'s [`TileDiskCache`] into `veldmap-net`'s
/// storage-agnostic [`TileBlobStore`] trait.
struct SqliteBlobStore(Arc<TileDiskCache>);

impl TileBlobStore for SqliteBlobStore {
    fn get(&self, source_key: &str, id: QuadTileId) -> Result<Option<Vec<u8>>, String> {
        let key = DiskTileKey::new(source_key, id.scheme, id.level, id.x, id.y);
        self.0.get(&key).map_err(|e| e.to_string())
    }

    fn put(&self, source_key: &str, id: QuadTileId, payload: &[u8]) -> Result<(), String> {
        let key = DiskTileKey::new(source_key, id.scheme, id.level, id.x, id.y);
        self.0
            .put(&key, payload)
            .map(|_| ())
            .map_err(|e| e.to_string())
    }

    fn delete(&self, source_key: &str, id: QuadTileId) -> Result<(), String> {
        let key = DiskTileKey::new(source_key, id.scheme, id.level, id.x, id.y);
        self.0.delete(&key).map(|_| ()).map_err(|e| e.to_string())
    }
}

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

/// UI-readable snapshot of asynchronous disk cache stats and maintenance state.
#[derive(Clone, Debug, Default)]
pub struct DiskCacheUiState {
    pub available: bool,
    pub stats: Option<DiskCacheStats>,
    pub busy: bool,
    pub notice: Option<String>,
    pub clear_gen: u64,
}

enum DiskCacheCmd {
    RefreshStats,
    SetMaxBytes(u64),
    Clear,
}

/// Serialized off-UI-thread worker handling SQLite stats, cap changes, and
/// full clears in FIFO order (never blocking the egui frame loop and joined
/// deterministically on shutdown).
struct DiskCacheWorker {
    tx: Option<Sender<DiskCacheCmd>>,
    state: Arc<Mutex<DiskCacheUiState>>,
    handle: Option<JoinHandle<()>>,
}

impl DiskCacheWorker {
    fn spawn(cache: Arc<TileDiskCache>, wakeup: Wakeup) -> Self {
        let (tx, rx) = mpsc::channel::<DiskCacheCmd>();
        let state = Arc::new(Mutex::new(DiskCacheUiState {
            available: true,
            stats: None,
            busy: false,
            notice: None,
            clear_gen: 0,
        }));
        let worker_state = Arc::clone(&state);
        let handle = match thread::Builder::new()
            .name("veldmap-disk-cache-ui".to_owned())
            .spawn(move || {
                while let Ok(cmd) = rx.recv() {
                    match cmd {
                        DiskCacheCmd::RefreshStats => {
                            if let Ok(stats) = cache.stats() {
                                lock_unpoisoned(&worker_state).stats = Some(stats);
                                (wakeup)();
                            }
                        }
                        DiskCacheCmd::SetMaxBytes(max_bytes) => {
                            if let Err(err) = cache.set_max_bytes(max_bytes) {
                                tracing::warn!(
                                    error = %err,
                                    "Failed to update SQLite disk cache budget"
                                );
                            }
                            if let Ok(stats) = cache.stats() {
                                lock_unpoisoned(&worker_state).stats = Some(stats);
                            }
                            (wakeup)();
                        }
                        DiskCacheCmd::Clear => {
                            let res = cache.clear();
                            let stats = cache.stats().ok();
                            {
                                let mut st = lock_unpoisoned(&worker_state);
                                st.busy = false;
                                if let Some(s) = stats {
                                    st.stats = Some(s);
                                }
                                match res {
                                    Ok(()) => {
                                        st.notice = Some("Disk cache cleared.".to_owned());
                                        st.clear_gen = st.clear_gen.wrapping_add(1);
                                    }
                                    Err(err) => {
                                        st.notice = Some(format!("Clear failed: {err}"));
                                    }
                                }
                            }
                            (wakeup)();
                        }
                    }
                }
            }) {
            Ok(h) => Some(h),
            Err(err) => {
                tracing::warn!(error = %err, "Failed to spawn disk cache maintenance worker");
                None
            }
        };

        Self {
            tx: Some(tx),
            state,
            handle,
        }
    }

    fn snapshot(&self) -> DiskCacheUiState {
        lock_unpoisoned(&self.state).clone()
    }

    fn clear_notice(&self) {
        lock_unpoisoned(&self.state).notice = None;
    }

    fn request_refresh(&self) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(DiskCacheCmd::RefreshStats);
        }
    }

    fn request_set_max_bytes(&self, max_bytes: u64) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(DiskCacheCmd::SetMaxBytes(max_bytes));
        }
    }

    fn request_clear(&self) {
        {
            let mut st = lock_unpoisoned(&self.state);
            if st.busy {
                return;
            }
            st.busy = true;
            st.notice = None;
        }
        if let Some(tx) = &self.tx {
            let _ = tx.send(DiskCacheCmd::Clear);
        }
    }

    fn close_channel(&mut self) {
        self.tx = None;
    }

    fn join(&mut self) {
        self.tx = None;
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for DiskCacheWorker {
    fn drop(&mut self) {
        self.join();
    }
}

/// Coordinates offline bundled + live HTTP basemap streaming, disk caching,
/// and durable user settings.
pub struct BasemapManager {
    paths: Option<StoragePaths>,
    settings: AppSettings,
    disk_worker: Option<DiskCacheWorker>,
    bundled_provider: BundledBasemapProvider,
    live_pipeline: Option<LiveTilePipeline>,
    live_pipeline_error: Option<String>,
    active_live_spec: Option<LiveBasemapSpec>,
    config_error: Option<String>,
    observed_live_tile_bytes: usize,
    last_settings_open: bool,
    last_stats_completed_net: u64,
    last_stats_refresh: Option<Instant>,
    last_handled_clear_gen: u64,
    wanted_bundled: HashSet<TileKey>,
    wanted_live: HashSet<TileKey>,
    wanted_all: HashSet<TileKey>,
}

impl BasemapManager {
    /// Initialize basemap orchestration, loading persisted user settings and
    /// opening the SQLite WAL disk cache under `paths` (or platform defaults
    /// when `None`).
    pub fn new(
        done_tx: Sender<(TileKey, DecodedTile)>,
        wakeup: Wakeup,
        paths: Option<StoragePaths>,
    ) -> Self {
        let resolved_paths = paths.or_else(|| match StoragePaths::system_default() {
            Ok(p) => Some(p),
            Err(err) => {
                tracing::warn!(error = %err, "Platform storage directories unavailable; running without disk persistence");
                None
            }
        });

        let settings = resolved_paths
            .as_ref()
            .map(AppSettings::load_or_default)
            .unwrap_or_default();

        let disk_cache = resolved_paths.as_ref().and_then(|p| {
            match TileDiskCache::open(p.tile_cache_db_path(), settings.disk_cache_max_bytes()) {
                Ok(cache) => Some(Arc::new(cache)),
                Err(err) => {
                    tracing::warn!(
                        error = %err,
                        "Failed to open SQLite tile disk cache; continuing without disk cache"
                    );
                    None
                }
            }
        });

        let disk_worker = disk_cache
            .as_ref()
            .map(|cache| DiskCacheWorker::spawn(Arc::clone(cache), Arc::clone(&wakeup)));
        let blob_store: Option<Arc<dyn TileBlobStore>> = disk_cache
            .as_ref()
            .map(|cache| Arc::new(SqliteBlobStore(Arc::clone(cache))) as Arc<dyn TileBlobStore>);

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
            paths: resolved_paths,
            settings,
            disk_worker,
            bundled_provider,
            live_pipeline,
            live_pipeline_error,
            active_live_spec: None,
            config_error: None,
            observed_live_tile_bytes: 0,
            last_settings_open: false,
            last_stats_completed_net: 0,
            last_stats_refresh: None,
            last_handled_clear_gen: 0,
            wanted_bundled: HashSet::new(),
            wanted_live: HashSet::new(),
            wanted_all: HashSet::new(),
        };

        manager.sync_active_spec();
        if manager.is_textured() {
            let scheme = manager.quadtree_config().scheme;
            manager.bundled_provider.seed_roots(scheme);
        }
        manager
    }

    pub fn settings(&self) -> &AppSettings {
        &self.settings
    }

    /// Whether the globe should render textured imagery (`false` for
    /// `EllipsoidOnly`).
    pub fn is_textured(&self) -> bool {
        self.settings.active_basemap != BasemapSelection::EllipsoidOnly
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
        match self.settings.active_basemap {
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

    /// Cached off-UI-thread disk cache status & stats snapshot (zero SQLite
    /// blocking on the UI thread).
    pub fn disk_cache_ui_state(&self) -> DiskCacheUiState {
        self.disk_worker
            .as_ref()
            .map(DiskCacheWorker::snapshot)
            .unwrap_or_default()
    }

    /// Clear transient disk cache UI notice (e.g. when reopening Settings).
    pub fn clear_disk_cache_notice(&self) {
        if let Some(worker) = &self.disk_worker {
            worker.clear_notice();
        }
    }

    /// Event-driven disk-cache stats refresh: triggers an async stats query
    /// only when Settings opens or when new live tiles have landed and at least
    /// `DISK_STATS_REFRESH_INTERVAL` has elapsed (zero idle queries/repaints).
    pub fn maybe_refresh_disk_stats(&mut self, settings_open: bool) {
        let just_opened = settings_open && !self.last_settings_open;
        self.last_settings_open = settings_open;
        if !settings_open {
            return;
        }
        let Some(worker) = &self.disk_worker else {
            return;
        };
        let completed_net = self.network_status().completed_from_net;
        let now = Instant::now();
        let interval_elapsed = self
            .last_stats_refresh
            .is_none_or(|t| now.duration_since(t) >= DISK_STATS_REFRESH_INTERVAL);
        if just_opened || (completed_net != self.last_stats_completed_net && interval_elapsed) {
            self.last_stats_completed_net = completed_net;
            self.last_stats_refresh = Some(now);
            worker.request_refresh();
        }
    }

    /// Enqueue an asynchronous SQLite disk cache clear on the maintenance
    /// worker thread (never stalls the UI thread).
    pub fn request_clear_disk_cache(&self) {
        if let Some(worker) = &self.disk_worker {
            worker.request_clear();
        }
    }

    /// Poll background maintenance completions (e.g. after an async disk clear
    /// finishes, reset live failure markers and re-reconcile cleanly). Returns
    /// `true` if the scene/UI should repaint.
    pub fn poll_background_events(
        &mut self,
        gpu_cache: &GpuTileCache,
        pending_uploads: &PendingUploads,
    ) -> bool {
        let Some(worker) = &self.disk_worker else {
            return false;
        };
        let clear_gen = worker.snapshot().clear_gen;
        if clear_gen != self.last_handled_clear_gen {
            self.last_handled_clear_gen = clear_gen;
            self.retry_live();
            self.reconcile_providers(gpu_cache, pending_uploads);
            return true;
        }
        false
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

    /// Apply updated [`AppSettings`], persisting durably via
    /// `write_atomic_durable_private()`, updating the disk cache cap
    /// asynchronously, evicting GPU tiles from the previous live source if
    /// changed, and switching the live pipeline. Returns `true` when visual
    /// state changed.
    pub fn apply_settings(
        &mut self,
        mut next: AppSettings,
        gpu_cache: &mut GpuTileCache,
        pending_uploads: &mut PendingUploads,
    ) -> bool {
        next.normalize();
        let prev_source = self.active_source();
        let prev_mode = self.settings.active_basemap;
        let prev_cache_bytes = self.settings.disk_cache_max_bytes();
        let changed = next != self.settings;

        self.settings = next;
        if let Some(paths) = &self.paths
            && let Err(err) = self.settings.save(paths)
        {
            tracing::warn!(error = %err, "Failed to persist user settings");
        }

        let next_cache_bytes = self.settings.disk_cache_max_bytes();
        if next_cache_bytes != prev_cache_bytes
            && let Some(worker) = &self.disk_worker
        {
            worker.request_set_max_bytes(next_cache_bytes);
        }

        self.sync_active_spec();
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
        } else if prev_mode != self.settings.active_basemap && self.is_textured() {
            self.bundled_provider
                .seed_roots(self.quadtree_config().scheme);
        }

        changed || prev_source != next_source
    }

    fn sync_active_spec(&mut self) {
        match spec_from_settings(&self.settings) {
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
    /// pipeline, sharing a single `< 50%` VRAM byte appetite cap across bundled
    /// and live tiles and enforcing `0` HTTP requests for `PolarCap` and `0`
    /// HTTP requests above `max_level`.
    pub fn schedule_for_snapshot(
        &mut self,
        snapshot: &DrawSnapshot,
        gpu_cache: &GpuTileCache,
        pending_uploads: &PendingUploads,
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
        let total_appetite_bytes = gpu_cache.pyramid_byte_cap(1);

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
                    // Hard contract: PolarCap tiles ALWAYS route to bundled offline basemap.
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
                    // Non-polar tile: request live tile strictly <= spec.max_level.
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

                    // Only request bundled coarse fallback when NO live ancestor
                    // carrier exists in VRAM yet (avoiding wasted Blue Marble
                    // renders once live z=0 is already covering the viewport).
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
            // Bundled offline basemap mode (`Geodetic z = 0..3` + `PolarCap z = 0`).
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

        // Budget bundled tiles first (typically just 2 polar caps + <= 4 coarse
        // fallbacks in live mode) so live tiles never starve polar caps, then
        // allocate the remaining `< 50%` byte appetite to live tiles.
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

        // Build `wanted_all` from the final appetite-bounded live + bundled sets.
        self.wanted_all.extend(self.wanted_live.iter().copied());
        self.wanted_all.extend(self.wanted_bundled.iter().copied());

        // Reconcile providers before submitting new bundled jobs so in-flight
        // and queued counts stay exact.
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
        if let Some(worker) = &mut self.disk_worker {
            worker.close_channel();
        }
        if let Some(pipeline) = &self.live_pipeline {
            pipeline.signal_shutdown();
        }
    }

    /// Phase 2 of deterministic shutdown (after `PendingUploads::close()`):
    /// join CPU decode/bundled/disk-cache threads and shut down `tokio::Runtime`.
    pub fn finish_shutdown(&mut self) {
        self.bundled_provider.join();
        if let Some(worker) = &mut self.disk_worker {
            worker.join();
        }
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

    use super::*;

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

        let mut pending = PendingUploads::new();
        let tx = pending.sender().expect("sender");
        let mut gpu_cache = GpuTileCache::new(64 * 1024 * 1024);
        let mut manager = BasemapManager::new(tx, Arc::new(|| {}), Some(paths.clone()));

        assert!(!manager.is_textured());
        assert_eq!(manager.attribution(), None);
        assert_eq!(
            manager.bundled_provider.in_flight_count(),
            0,
            "EllipsoidOnly must not seed bundled roots"
        );

        // Invalid GIBS date surfaces config_error and falls back cleanly to Blue Marble attribution.
        let mut bad_date = initial.clone();
        bad_date.active_basemap = BasemapSelection::NasaGibsViirs;
        bad_date.gibs_date = "2024-02-30".to_owned();
        manager.apply_settings(bad_date, &mut gpu_cache, &mut pending);
        assert!(manager.config_error().is_some());
        assert_eq!(manager.active_source(), SourceId::BUNDLED_BASEMAP);
        assert_eq!(manager.attribution(), Some(BUNDLED_ATTRIBUTION));

        // Invalid Custom XYZ surfaces config_error and falls back cleanly.
        let mut bad_custom = initial;
        bad_custom.active_basemap = BasemapSelection::CustomXyz;
        bad_custom.custom_xyz.url_template = "ftp://bad/{z}/{x}/{y}".to_owned();
        manager.apply_settings(bad_custom, &mut gpu_cache, &mut pending);
        assert!(manager.config_error().is_some());
        assert_eq!(manager.active_source(), SourceId::BUNDLED_BASEMAP);

        manager.begin_shutdown();
        pending.close();
        manager.finish_shutdown();
    }

    #[test]
    fn schedule_for_snapshot_enforces_polar_exclusion_max_level_and_source_switch_flush() {
        let root = TempRoot::new("schedule");
        let paths = StoragePaths::with_root(&root.0);
        let mut pending = PendingUploads::new();
        let tx = pending.sender().expect("sender");
        let mut gpu_cache = GpuTileCache::new(64 * 1024 * 1024);
        let mut manager = BasemapManager::new(tx.clone(), Arc::new(|| {}), Some(paths));

        // Switch to NASA GIBS Blue Marble (max_level = 8).
        let mut live_cfg = manager.settings().clone();
        live_cfg.active_basemap = BasemapSelection::NasaGibsBlueMarble;
        manager.apply_settings(live_cfg, &mut gpu_cache, &mut pending);
        let gibs_source = manager.active_source();
        assert_ne!(gibs_source, SourceId::BUNDLED_BASEMAP);

        // View centered near the South Pole so PolarCap + WebMercator tiles both appear.
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
        manager.schedule_for_snapshot(&snap, &gpu_cache, &pending);

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

        // Put a live tile into PendingUploads and record observed size, then switch source:
        // stale live tile must be cleared and observed_live_tile_bytes must reset.
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

        let mut eox_cfg = manager.settings().clone();
        eox_cfg.active_basemap = BasemapSelection::EoxSentinel2Cloudless;
        manager.apply_settings(eox_cfg, &mut gpu_cache, &mut pending);
        assert!(!pending.contains(&stale_key));
        assert_eq!(manager.observed_live_tile_bytes, 0);

        manager.begin_shutdown();
        pending.close();
        manager.finish_shutdown();
    }
}
