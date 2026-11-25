//! Live + offline basemap coordinator ([`BasemapManager`]) and visible-tile
//! scheduling helpers over [`BundledBasemapProvider`] and `LiveTilePipeline`.

use std::collections::HashSet;
use std::sync::Arc;

use veldmap_core::sync::{CancelToken, Wakeup};
use veldmap_core::tile::{PlanetaryScheme, QuadTileId, QuadtreeConfig, SourceId, TileKey};
use veldmap_net::{
    FetcherConfig, LiveBasemapSpec, LiveTilePipeline, NetError, TileBlobStore, TileDecodeExecutor,
};
use veldmap_raster::DecodePoolHandle;
use veldmap_storage::TileDiskCache;

use super::adapters::{RasterDecodeExecutor, SqliteBlobStore};
use super::bundled::{
    BUNDLED_MAX_LEVEL, BUNDLED_POLAR_MAX_LEVEL, BUNDLED_TILE_PX, BundledBasemapProvider,
};
use super::policy::BasemapCarrierPolicy;
use super::{BasemapStatusSnapshot, BasemapSyncOutcome, LiveStreamStatus};
use crate::gpu_tiles::{
    GpuTileCache, PendingUploads, UploadBatchReport, UploadSender, display_tile_mip_bytes,
};
use crate::storage::settings::{AppSettings, BasemapSelection};

/// Attribution string displayed for the bundled public-domain Blue Marble
/// offline basemap.
const BUNDLED_ATTRIBUTION: &str = "Imagery: NASA Blue Marble Next Generation (Public Domain)";

/// Resolve the active [`LiveBasemapSpec`] from persisted [`AppSettings`],
/// returning `Ok(None)` for offline (`BundledBlueMarble`) or `EllipsoidOnly`
/// modes.
fn spec_from_settings(settings: &AppSettings) -> Result<Option<LiveBasemapSpec>, NetError> {
    match settings.active_basemap {
        BasemapSelection::BundledBlueMarble | BasemapSelection::EllipsoidOnly => Ok(None),
        BasemapSelection::NasaGibsBlueMarble => Ok(Some(LiveBasemapSpec::nasa_gibs_blue_marble())),
        BasemapSelection::NasaGibsViirs => {
            LiveBasemapSpec::nasa_gibs_viirs(&settings.gibs_date).map(Some)
        }
        BasemapSelection::EoxSentinel2Cloudless => {
            Ok(Some(LiveBasemapSpec::eox_sentinel2_cloudless()))
        }
        BasemapSelection::CustomXyz => {
            LiveBasemapSpec::custom(settings.custom_xyz.as_net_config()).map(Some)
        }
    }
}

/// Coordinates offline bundled + live HTTP basemap streaming without owning
/// application-wide storage, shared CPU decode threads, GPU cache, or upload
/// queue state.
pub(crate) struct BasemapManager {
    active_basemap: BasemapSelection,
    bundled_provider: BundledBasemapProvider,
    live_pipeline: Option<LiveTilePipeline>,
    live_pipeline_error: Option<String>,
    active_live_spec: Option<LiveBasemapSpec>,
    config_error: Option<String>,
    observed_live_tile_bytes: usize,
    wanted_bundled: HashSet<TileKey>,
    wanted_live: HashSet<TileKey>,
}

impl Drop for BasemapManager {
    fn drop(&mut self) {
        self.begin_shutdown();
        self.finish_shutdown();
    }
}

impl BasemapManager {
    /// Initialize basemap orchestration using initial `settings`, an injected
    /// fast-lane [`DecodePoolHandle`] + `decode_shutdown` token from
    /// [`crate::scene::SceneState`], and an optional shared SQLite
    /// [`TileDiskCache`].
    pub(crate) fn new(
        settings: &AppSettings,
        disk_cache: Option<Arc<TileDiskCache>>,
        done_tx: UploadSender,
        wakeup: Wakeup,
        decode_handle: DecodePoolHandle,
        decode_shutdown: CancelToken,
    ) -> Self {
        let bundled_provider = BundledBasemapProvider::new(
            decode_handle.clone(),
            done_tx.clone(),
            Arc::clone(&wakeup),
            decode_shutdown.clone(),
        );
        let decode_executor: Arc<dyn TileDecodeExecutor> = Arc::new(RasterDecodeExecutor {
            pool: decode_handle,
            done_tx,
            wakeup: Arc::clone(&wakeup),
            shutdown: decode_shutdown,
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
            live_pipeline,
            live_pipeline_error,
            active_live_spec: None,
            config_error: None,
            observed_live_tile_bytes: 0,
            wanted_bundled: HashSet::new(),
            wanted_live: HashSet::new(),
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
    pub(crate) fn is_textured(&self) -> bool {
        self.active_basemap != BasemapSelection::EllipsoidOnly
    }

    /// Relative VRAM appetite weight in bytes for proportional budget sharing
    /// in [`crate::scene::SceneState`] (`0` when untextured).
    pub(crate) fn appetite_weight_bytes(&self) -> u64 {
        if !self.is_textured() {
            0
        } else {
            Self::bundled_tile_byte_size().max(self.observed_live_tile_bytes) as u64
        }
    }

    /// Active primary [`SourceId`] passed into `DrawSnapshot::build` and
    /// pinned-root configuration.
    pub(crate) fn active_source(&self) -> SourceId {
        match &self.active_live_spec {
            Some(spec) => spec.source_id(),
            None => SourceId::BUNDLED_BASEMAP,
        }
    }

    /// Pure carrier-routing policy matching the active basemap source.
    pub(crate) fn carrier_policy(&self) -> BasemapCarrierPolicy {
        BasemapCarrierPolicy::new(self.active_source())
    }

    /// Active planetary [`QuadtreeConfig`] matched to the selected basemap.
    pub(crate) fn quadtree_config(&self) -> QuadtreeConfig {
        match &self.active_live_spec {
            Some(spec) => {
                // Live basemaps stream 256px WebMercator/Geodetic tiles up to
                // `spec.max_level()` and route polar caps to the bundled
                // offline polar cap (`BUNDLED_POLAR_MAX_LEVEL`).
                let mut cfg = QuadtreeConfig::for_scheme(spec.scheme(), spec.max_level());
                cfg.polar_max_tex_level = BUNDLED_POLAR_MAX_LEVEL;
                cfg
            }
            None => {
                let mut cfg = QuadtreeConfig::geodetic(BUNDLED_MAX_LEVEL);
                cfg.polar_max_tex_level = BUNDLED_POLAR_MAX_LEVEL;
                cfg.tile_px = BUNDLED_TILE_PX;
                cfg
            }
        }
    }

    /// Visible attribution string for the active basemap (`None` for
    /// `EllipsoidOnly`).
    pub(crate) fn attribution(&self) -> Option<&str> {
        match self.active_basemap {
            BasemapSelection::EllipsoidOnly => None,
            BasemapSelection::BundledBlueMarble => Some(BUNDLED_ATTRIBUTION),
            _ => match &self.active_live_spec {
                Some(spec) => Some(spec.attribution()),
                None => Some(BUNDLED_ATTRIBUTION),
            },
        }
    }

    /// Snapshot toolbar-visible live streaming / config error state.
    pub(crate) fn status_snapshot(&self) -> BasemapStatusSnapshot {
        BasemapStatusSnapshot {
            stream: self
                .live_pipeline
                .as_ref()
                .map(LiveTilePipeline::status)
                .map(LiveStreamStatus::from)
                .unwrap_or_default(),
            live_pipeline_error: self.live_pipeline_error.clone(),
            config_error: self.config_error.clone(),
        }
    }

    /// Borrow both active wanted sets (live + bundled/polar fallback) without
    /// duplicating them into a merged `HashSet`.
    pub(crate) fn wanted_tile_sets(&self) -> [&HashSet<TileKey>; 2] {
        [&self.wanted_live, &self.wanted_bundled]
    }

    /// Update observed live tile byte size from an upload batch so VRAM
    /// appetite budgeting accounts for 512 px / `@2x` Custom XYZ tiles cleanly.
    pub(crate) fn record_upload_report(&mut self, report: &UploadBatchReport) {
        if let Some(spec) = &self.active_live_spec {
            let max_bytes = report.max_tile_bytes_for_stream(veldmap_core::tile::StreamKey::new(
                spec.source_id(),
                veldmap_core::tile::BandKey::Display,
                veldmap_core::tile::PixelFormat::Rgba8UnormSrgb,
                0,
            ));
            if max_bytes > 0 {
                self.observed_live_tile_bytes = self.observed_live_tile_bytes.max(max_bytes);
            }
        }
    }

    /// Retry failed/offline live tile requests immediately.
    pub(crate) fn retry_live(&self) {
        if let Some(pipeline) = &self.live_pipeline {
            pipeline.retry_failed();
        }
    }

    /// Synchronize active basemap providers with updated [`AppSettings`],
    /// returning explicit eviction intent so the caller (`SceneState`) owns
    /// GPU cache and upload queue purging.
    pub(crate) fn sync_settings(&mut self, settings: &AppSettings) -> BasemapSyncOutcome {
        let prev_source = self.active_source();
        let prev_mode = self.active_basemap;
        let prev_spec = self.active_live_spec.clone();
        self.active_basemap = settings.active_basemap;

        self.sync_active_spec(settings);
        let next_source = self.active_source();

        let mut evict_source = None;
        if prev_source != next_source {
            // Keep bundled fallback tiles warm in VRAM; instruct caller to
            // purge stale tiles from the previous live source.
            if prev_source != SourceId::BUNDLED_BASEMAP {
                evict_source = Some(prev_source);
            }
            self.observed_live_tile_bytes = 0;
            self.wanted_live.clear();
            self.wanted_bundled.clear();
            if self.is_textured() {
                self.bundled_provider
                    .seed_roots(self.quadtree_config().scheme);
            }
        } else if prev_mode != self.active_basemap && self.is_textured() {
            self.bundled_provider
                .seed_roots(self.quadtree_config().scheme);
        }

        BasemapSyncOutcome {
            changed: prev_source != next_source
                || prev_mode != self.active_basemap
                || prev_spec != self.active_live_spec,
            evict_source,
        }
    }

    fn sync_active_spec(&mut self, settings: &AppSettings) {
        match spec_from_settings(settings) {
            Ok(spec) => {
                self.config_error = None;
                // Only activate a live spec when `live_pipeline` is actually
                // running; otherwise cleanly fall back to full `z = 0..3`
                // bundled Blue Marble while `live_pipeline_error` surfaces the
                // `"Live unavailable"` toolbar badge.
                if let Some(pipeline) = &self.live_pipeline {
                    self.active_live_spec = spec.clone();
                    pipeline.set_active_spec(spec);
                } else {
                    self.active_live_spec = None;
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
    /// for both the bundled offline provider and the active live HTTP pipeline
    /// within the caller-provided VRAM `appetite_bytes` ceiling.
    pub(crate) fn schedule_for_tiles(
        &mut self,
        visible_tex_ids: impl IntoIterator<Item = QuadTileId>,
        gpu_cache: &GpuTileCache,
        pending_uploads: &PendingUploads,
        appetite_bytes: usize,
    ) {
        self.wanted_bundled.clear();
        self.wanted_live.clear();

        if !self.is_textured() {
            self.reconcile_providers(gpu_cache, pending_uploads);
            return;
        }

        let candidates = self.collect_tile_candidates(visible_tex_ids, gpu_cache, pending_uploads);
        let default_tile_bytes = Self::bundled_tile_byte_size();

        let (bundled_to_request, bundled_total_bytes) = self.plan_bundled_requests(
            candidates.missing_bundled,
            candidates.cached_bundled_wanted,
            pending_uploads,
            default_tile_bytes,
            appetite_bytes,
        );

        if self.active_live_spec.is_some() {
            self.admit_live_candidates(
                candidates.live_candidates,
                candidates.cached_live_wanted,
                candidates.queued_live_wanted,
                default_tile_bytes,
                appetite_bytes.saturating_sub(bundled_total_bytes),
            );
        }

        self.reconcile_providers(gpu_cache, pending_uploads);

        for key in bundled_to_request {
            self.bundled_provider.request(key);
        }
    }

    fn bundled_tile_byte_size() -> usize {
        display_tile_mip_bytes(BUNDLED_TILE_PX)
    }

    fn collect_tile_candidates(
        &mut self,
        visible_tex_ids: impl IntoIterator<Item = QuadTileId>,
        gpu_cache: &GpuTileCache,
        pending_uploads: &PendingUploads,
    ) -> ScheduledCandidates {
        let mut out = ScheduledCandidates::default();
        let policy = self.carrier_policy();
        let wanted_bundled = &mut self.wanted_bundled;
        let bundled_provider = &self.bundled_provider;
        let mut record_bundled_candidate = |key: TileKey| {
            if wanted_bundled.insert(key) {
                if gpu_cache.contains(&key) {
                    out.cached_bundled_wanted += 1;
                } else if !pending_uploads.contains(&key) && !bundled_provider.is_in_flight(&key) {
                    out.missing_bundled.push(key);
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
                    let live_key = policy.primary_key(clamped_live);
                    if seen_live.insert(live_key) {
                        if gpu_cache.contains(&live_key) {
                            out.cached_live_wanted += 1;
                            self.wanted_live.insert(live_key);
                        } else if pending_uploads.contains(&live_key) {
                            out.queued_live_wanted += 1;
                            self.wanted_live.insert(live_key);
                        } else {
                            out.live_candidates.push(live_key);
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
                record_bundled_candidate(policy.primary_key(tex_id));
            }
        }

        out
    }

    fn plan_bundled_requests(
        &self,
        mut missing_bundled: Vec<TileKey>,
        cached_bundled_wanted: usize,
        pending_uploads: &PendingUploads,
        default_tile_bytes: usize,
        total_appetite_bytes: usize,
    ) -> (Vec<TileKey>, usize) {
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
        (bundled_to_request, bundled_total_bytes)
    }

    fn admit_live_candidates(
        &mut self,
        mut live_candidates: Vec<TileKey>,
        cached_live_wanted: usize,
        queued_live_wanted: usize,
        default_tile_bytes: usize,
        live_appetite_bytes: usize,
    ) {
        let live_tile_bytes = default_tile_bytes.max(self.observed_live_tile_bytes);
        let live_tile_cap = live_appetite_bytes / live_tile_bytes;
        let live_budget = live_tile_cap
            .saturating_sub(cached_live_wanted)
            .saturating_sub(queued_live_wanted);
        live_candidates.sort_by_key(TileKey::coarsest_first_key);
        for k in live_candidates.into_iter().take(live_budget) {
            self.wanted_live.insert(k);
        }
    }

    /// Reconcile both background providers after GPU tile uploads complete.
    pub(crate) fn reconcile_providers(
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

    /// Phase 1 of deterministic shutdown: cancel bundled requests and signal
    /// `LiveTilePipeline` BEFORE closing `PendingUploads` and joining the
    /// shared decode pool.
    pub(crate) fn begin_shutdown(&mut self) {
        self.bundled_provider.cancel_all();
        if let Some(pipeline) = &self.live_pipeline {
            pipeline.signal_shutdown();
        }
    }

    /// Phase 2 of deterministic shutdown (after `PendingUploads::close()` and
    /// joining the shared decode pool): shut down `tokio::Runtime`.
    pub(crate) fn finish_shutdown(&mut self) {
        if let Some(pipeline) = self.live_pipeline.take() {
            pipeline.shutdown();
        }
    }
}

#[derive(Default)]
struct ScheduledCandidates {
    missing_bundled: Vec<TileKey>,
    cached_bundled_wanted: usize,
    live_candidates: Vec<TileKey>,
    cached_live_wanted: usize,
    queued_live_wanted: usize,
}

#[cfg(test)]
#[path = "manager_tests.rs"]
mod tests;
