//! Live + offline basemap orchestration extracted from `app.rs`.
//!
//! Coordinates the bundled offline NASA Blue Marble worker
//! ([`BundledBasemapProvider`]), the shared `veldmap-raster` CPU decode pool
//! ([`DecodePool`]), and the async HTTP streaming pipeline
//! (`LiveTilePipeline`), enforcing:
//! - `0` live HTTP requests for `PolarCap` keys (always routed to bundled
//!   `SourceId::BUNDLED_BASEMAP` offline polar caps)
//! - `0` live HTTP requests above per-source `max_level`
//! - Seamless Frame-1 & offline fallback to bundled Blue Marble tiles
//! - Shared `< 50%` VRAM appetite capping across bundled + live tiles
//! - Deterministic shutdown (`CancelToken` -> close `PendingUploads` rx -> join
//!   bundled/decode threads -> shut down `tokio::Runtime`)

use std::collections::HashSet;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::mpsc::Sender;

use veldmap_core::io::{CancelToken, Wakeup};
use veldmap_core::tile::{
    DecodedTile, PixelFormat, PlanetaryScheme, QuadTileId, QuadtreeConfig, SourceId, TileKey,
    max_mip_levels, mip_chain_byte_size,
};
use veldmap_net::{
    DecodeRequest, FetcherConfig, LiveBasemapSpec, LiveTilePipeline, NetError, NetworkStatus,
    TileBlobStore, TileDecodeExecutor,
};
use veldmap_raster::{
    BUNDLED_MAX_LEVEL, BUNDLED_POLAR_MAX_LEVEL, BUNDLED_TILE_PX, DecodeJob, DecodePool,
    DecodePoolConfig, DecodePoolHandle, decode_map_tile_with_mipmaps,
};
use veldmap_storage::{DiskTileKey, TileDiskCache};

use crate::bundled_basemap::BundledBasemapProvider;
use crate::gpu_tiles::{GpuTileCache, PendingUploads, UploadBatchReport, bundled_tile_key};
use crate::render::BasemapCarrierPolicy;
use crate::settings::{AppSettings, BasemapSelection};

/// Attribution string displayed for the bundled public-domain Blue Marble
/// offline basemap.
pub const BUNDLED_ATTRIBUTION: &str = "Imagery: NASA Blue Marble Next Generation (Public Domain)";

/// Plain read-only basemap status view model passed to UI toolbar controls so
/// `basemap_ui` never depends directly on [`BasemapManager`].
#[derive(Clone, Debug, Default)]
pub struct BasemapStatusSnapshot {
    pub network: NetworkStatus,
    pub live_pipeline_error: Option<String>,
    pub config_error: Option<String>,
}

/// Adapter bridging `veldmap-storage`'s [`TileDiskCache`] into `veldmap-net`'s
/// storage-agnostic [`TileBlobStore`] trait.
struct SqliteBlobStore(Arc<TileDiskCache>);

impl TileBlobStore for SqliteBlobStore {
    fn get(&self, source_key: &str, id: QuadTileId) -> Result<Option<Vec<u8>>, String> {
        let key = DiskTileKey::new(
            source_key,
            id.scheme.into_tile_scheme(),
            id.level,
            id.x,
            id.y,
        );
        self.0.get(&key).map_err(|e| e.to_string())
    }

    fn put(&self, source_key: &str, id: QuadTileId, payload: &[u8]) -> Result<(), String> {
        let key = DiskTileKey::new(
            source_key,
            id.scheme.into_tile_scheme(),
            id.level,
            id.x,
            id.y,
        );
        self.0
            .put(&key, payload)
            .map(|_| ())
            .map_err(|e| e.to_string())
    }

    fn delete(&self, source_key: &str, id: QuadTileId) -> Result<(), String> {
        let key = DiskTileKey::new(
            source_key,
            id.scheme.into_tile_scheme(),
            id.level,
            id.x,
            id.y,
        );
        self.0.delete(&key).map(|_| ()).map_err(|e| e.to_string())
    }
}

/// Adapter bridging `veldmap-raster`'s generic [`DecodePoolHandle`] into
/// `veldmap-net`'s codec-agnostic [`TileDecodeExecutor`] trait.
struct RasterDecodeExecutor {
    pool: DecodePoolHandle,
    done_tx: Sender<(TileKey, DecodedTile)>,
    wakeup: Wakeup,
    shutdown: CancelToken,
}

impl TileDecodeExecutor for RasterDecodeExecutor {
    fn submit(&self, req: DecodeRequest) -> bool {
        let key = req.key;
        let queued_bytes = req.bytes.len();
        let cancel = req.cancel.clone();
        let done_tx = self.done_tx.clone();
        let wakeup = Arc::clone(&self.wakeup);
        let shutdown = self.shutdown.clone();
        let bytes = req.bytes;
        let on_decoded = req.on_decoded;

        self.pool.submit(DecodeJob {
            key,
            queued_bytes,
            cancel: req.cancel,
            work: Box::new(move || {
                let decoded =
                    catch_unwind(AssertUnwindSafe(|| decode_map_tile_with_mipmaps(&bytes)));
                match decoded {
                    Ok(Ok(tile)) => {
                        if cancel.is_cancelled() || shutdown.is_cancelled() {
                            return;
                        }
                        if done_tx.send((key, tile)).is_err() {
                            return;
                        }
                        on_decoded(Ok(()));
                        (wakeup)();
                    }
                    Ok(Err(err)) => {
                        tracing::debug!(
                            ?key,
                            error = %err,
                            "Failed to decode map tile payload"
                        );
                        on_decoded(Err(err.to_string()));
                    }
                    Err(_) => {
                        tracing::error!(?key, "Tile decoder panicked; isolated worker continues");
                        on_decoded(Err("tile decoder panicked".to_owned()));
                    }
                }
            }),
        })
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
        BasemapSelection::CustomXyz => settings.custom_xyz.to_live_spec().map(Some),
    }
}

/// Coordinates offline bundled + live HTTP basemap streaming without owning
/// application-wide storage or settings state.
pub struct BasemapManager {
    active_basemap: BasemapSelection,
    bundled_provider: BundledBasemapProvider,
    decode_shutdown: CancelToken,
    decode_pool: Option<DecodePool>,
    live_pipeline: Option<LiveTilePipeline>,
    live_pipeline_error: Option<String>,
    active_live_spec: Option<LiveBasemapSpec>,
    config_error: Option<String>,
    observed_live_tile_bytes: usize,
    wanted_bundled: HashSet<TileKey>,
    wanted_live: HashSet<TileKey>,
    wanted_all: HashSet<TileKey>,
}

impl Drop for BasemapManager {
    fn drop(&mut self) {
        self.begin_shutdown();
        self.finish_shutdown();
    }
}

impl BasemapManager {
    /// Initialize basemap orchestration using initial `settings` and an
    /// optional shared SQLite [`TileDiskCache`] from [`crate::storage::AppStorage`].
    pub fn new(
        settings: &AppSettings,
        disk_cache: Option<Arc<TileDiskCache>>,
        done_tx: Sender<(TileKey, DecodedTile)>,
        wakeup: Wakeup,
    ) -> Self {
        let bundled_provider = BundledBasemapProvider::spawn(done_tx.clone(), Arc::clone(&wakeup));

        let decode_threads = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(2)
            .clamp(2, 4);
        let decode_shutdown = CancelToken::new();
        let decode_pool = DecodePool::spawn(
            DecodePoolConfig {
                worker_count: decode_threads,
                ..DecodePoolConfig::default()
            },
            decode_shutdown.clone(),
        );
        let decode_executor: Arc<dyn TileDecodeExecutor> = Arc::new(RasterDecodeExecutor {
            pool: decode_pool.handle(),
            done_tx,
            wakeup: Arc::clone(&wakeup),
            shutdown: decode_shutdown.clone(),
        });
        let blob_store: Option<Arc<dyn TileBlobStore>> =
            disk_cache.map(|cache| Arc::new(SqliteBlobStore(cache)) as Arc<dyn TileBlobStore>);

        let (live_pipeline, live_pipeline_error) = match LiveTilePipeline::spawn(
            FetcherConfig::default(),
            blob_store,
            decode_executor,
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
            decode_shutdown,
            decode_pool: Some(decode_pool),
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

    /// Active primary [`SourceId`] passed into `DrawSnapshot::build` and
    /// [`PendingUploads::drain_and_upload`].
    pub fn active_source(&self) -> SourceId {
        match &self.active_live_spec {
            Some(spec) => spec.source_id(),
            None => SourceId::BUNDLED_BASEMAP,
        }
    }

    /// Pure carrier-routing policy matching the active basemap source.
    pub fn carrier_policy(&self) -> BasemapCarrierPolicy {
        BasemapCarrierPolicy::new(self.active_source())
    }

    /// Active planetary [`QuadtreeConfig`] matched to the selected basemap.
    pub fn quadtree_config(&self) -> QuadtreeConfig {
        let mut cfg = match &self.active_live_spec {
            Some(spec) => QuadtreeConfig::for_scheme(spec.scheme(), spec.max_level()),
            None => QuadtreeConfig::geodetic(BUNDLED_MAX_LEVEL),
        };
        cfg.polar_max_tex_level = BUNDLED_POLAR_MAX_LEVEL;
        cfg.tile_px = BUNDLED_TILE_PX;
        cfg
    }

    /// Visible attribution string for the active basemap (`None` for
    /// `EllipsoidOnly`).
    pub fn attribution(&self) -> Option<&str> {
        match self.active_basemap {
            BasemapSelection::EllipsoidOnly => None,
            BasemapSelection::BundledBlueMarble => Some(BUNDLED_ATTRIBUTION),
            _ => match &self.active_live_spec {
                Some(spec) => Some(spec.attribution()),
                None => Some(BUNDLED_ATTRIBUTION),
            },
        }
    }

    #[cfg(test)]
    pub(crate) fn config_error(&self) -> Option<&str> {
        self.config_error.as_deref()
    }

    /// Snapshot toolbar-visible live streaming / config error state.
    pub fn status_snapshot(&self) -> BasemapStatusSnapshot {
        BasemapStatusSnapshot {
            network: self
                .live_pipeline
                .as_ref()
                .map(LiveTilePipeline::status)
                .unwrap_or_default(),
            live_pipeline_error: self.live_pipeline_error.clone(),
            config_error: self.config_error.clone(),
        }
    }

    /// Combined wanted set (active live + bundled/polar fallback) for
    /// [`PendingUploads::drain_and_upload`].
    pub fn wanted_tiles(&self) -> &HashSet<TileKey> {
        &self.wanted_all
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

    /// Schedule missing visible tiles (identified purely by their planetary
    /// `QuadTileId`s, keeping `basemap` decoupled from `render::DrawSnapshot`)
    /// for both the bundled offline provider and the active live HTTP pipeline.
    pub fn schedule_for_tiles(
        &mut self,
        visible_tex_ids: impl IntoIterator<Item = QuadTileId>,
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
        let default_tile_bytes = mip_chain_byte_size(
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

        let policy = self.carrier_policy();
        let wanted_bundled = &mut self.wanted_bundled;
        let bundled_provider = &self.bundled_provider;
        let mut record_bundled_candidate = |key: TileKey| {
            if wanted_bundled.insert(key) {
                if gpu_cache.contains(&key) {
                    cached_bundled_wanted += 1;
                } else if !pending_uploads.contains(&key) && !bundled_provider.is_in_flight(&key) {
                    missing_bundled.push(key);
                }
            }
        };

        if let Some(spec) = &self.active_live_spec {
            let mut seen_live = HashSet::new();

            for tex_id in visible_tex_ids {
                if matches!(tex_id.scheme, PlanetaryScheme::PolarCap { .. }) {
                    let (polar_id, _) = tex_id.clamp_to_level(BUNDLED_POLAR_MAX_LEVEL);
                    record_bundled_candidate(policy.primary_key(polar_id));
                } else {
                    let (clamped_live, _) = tex_id.clamp_to_level(spec.max_level());
                    let live_key = spec.tile_key(clamped_live);
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
                        if let Some(fb_key) = policy.fallback_key(fb_id) {
                            record_bundled_candidate(fb_key);
                        }
                    }
                }
            }
        } else {
            for tex_id in visible_tex_ids {
                record_bundled_candidate(bundled_tile_key(tex_id));
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
        let is_ready = |k: &TileKey| gpu_cache.contains(k) || pending_uploads.contains(k);
        self.bundled_provider
            .reconcile(&self.wanted_bundled, is_ready);
        if let Some(pipeline) = &self.live_pipeline {
            pipeline.reconcile_wanted(&self.wanted_live, is_ready);
        }
    }

    /// Phase 1 of deterministic shutdown: cancel `CancelToken`s immediately
    /// and signal background workers BEFORE dropping `PendingUploads`'s rx.
    pub fn begin_shutdown(&mut self) {
        self.bundled_provider.signal_shutdown();
        if let Some(pipeline) = &self.live_pipeline {
            pipeline.signal_shutdown();
        }
        self.decode_shutdown.cancel();
    }

    /// Phase 2 of deterministic shutdown (after `PendingUploads::close()`):
    /// join CPU decode & bundled threads, then shut down `tokio::Runtime`.
    pub fn finish_shutdown(&mut self) {
        self.bundled_provider.join();
        if let Some(pool) = self.decode_pool.take() {
            pool.shutdown();
        }
        if let Some(pipeline) = self.live_pipeline.take() {
            pipeline.shutdown();
        }
    }
}

#[cfg(test)]
#[path = "basemap_tests.rs"]
mod tests;
