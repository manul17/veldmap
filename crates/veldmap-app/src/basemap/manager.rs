//! Live + offline basemap coordinator ([`BasemapManager`]) and visible-tile
//! scheduling helpers over [`BundledBasemapProvider`] and `LiveTilePipeline`.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use veldmap_core::sync::{CancelToken, Wakeup};
use veldmap_core::tile::{
    PlanetaryScheme, QuadTileId, QuadtreeConfig, SourceId, StreamKey, TileKey, TileScheme,
};
use veldmap_net::{
    FetcherConfig, LiveBasemapSpec, LiveTilePipeline, NetContext, NetError, TileBlobStore,
    TileDecodeExecutor,
};
use veldmap_raster::DecodePoolHandle;
use veldmap_storage::TileDiskCache;

use super::adapters::{RasterDecodeExecutor, SqliteBlobStore};
use super::bundled::{
    BUNDLED_MAX_LEVEL, BUNDLED_POLAR_MAX_LEVEL, BUNDLED_TILE_PX, BundledBasemapProvider,
};
use super::polar_rim::PolarRimTracker;
use super::policy::BasemapCarrierPolicy;
use super::{BasemapStatusSnapshot, BasemapSyncOutcome, LiveStreamStatus};
use crate::gpu_tiles::{
    GpuTileCache, PendingUploads, UploadBatchReport, UploadSender, display_tile_mip_bytes,
};
use crate::storage::settings::{BasemapConfigView, BasemapSelection};

/// Attribution string displayed for the bundled public-domain Blue Marble
/// offline basemap.
const BUNDLED_ATTRIBUTION: &str = "Imagery: NASA Blue Marble Next Generation (Public Domain)";

/// Resolve the active [`LiveBasemapSpec`] from a borrowed [`BasemapConfigView`],
/// returning `Ok(None)` for offline (`BundledBlueMarble`) or `EllipsoidOnly`
/// modes.
fn spec_from_config(config: BasemapConfigView<'_>) -> Result<Option<LiveBasemapSpec>, NetError> {
    match config.active_basemap {
        BasemapSelection::BundledBlueMarble | BasemapSelection::EllipsoidOnly => Ok(None),
        BasemapSelection::NasaGibsBlueMarble => Ok(Some(LiveBasemapSpec::nasa_gibs_blue_marble())),
        BasemapSelection::NasaGibsViirs => {
            LiveBasemapSpec::nasa_gibs_viirs(config.gibs_date).map(Some)
        }
        BasemapSelection::EoxSentinel2Cloudless => {
            Ok(Some(LiveBasemapSpec::eox_sentinel2_cloudless()))
        }
        BasemapSelection::CustomXyz => {
            LiveBasemapSpec::custom(config.custom_xyz.as_net_config()).map(Some)
        }
    }
}

/// Initial configuration and shared runtime subsystems injected into
/// [`BasemapManager::new`].
pub(crate) struct BasemapManagerDeps<'a> {
    pub(crate) config: BasemapConfigView<'a>,
    pub(crate) disk_cache: Option<Arc<TileDiskCache>>,
    pub(crate) net: Option<Arc<NetContext>>,
    pub(crate) net_init_error: Option<String>,
    pub(crate) done_tx: UploadSender,
    pub(crate) wakeup: Wakeup,
    pub(crate) decode_handle: DecodePoolHandle,
    pub(crate) decode_shutdown: CancelToken,
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
    polar_rim: Arc<PolarRimTracker>,
    polar_revs: [u32; 2],
    polar_fallback_revs: [u32; 2],
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
    /// Initialize basemap orchestration using initial `deps.config`, an
    /// injected fast-lane [`DecodePoolHandle`] + `decode_shutdown` token from
    /// [`crate::scene::SceneState`], and an optional shared SQLite
    /// [`TileDiskCache`].
    pub(crate) fn new(deps: BasemapManagerDeps<'_>) -> Self {
        let BasemapManagerDeps {
            config,
            disk_cache,
            net,
            net_init_error,
            done_tx,
            wakeup,
            decode_handle,
            decode_shutdown,
        } = deps;
        let polar_rim = Arc::new(PolarRimTracker::new());
        let bundled_provider = BundledBasemapProvider::new(
            decode_handle.clone(),
            done_tx.clone(),
            Arc::clone(&wakeup),
            decode_shutdown.clone(),
            Arc::clone(&polar_rim),
        );
        let decode_executor: Arc<dyn TileDecodeExecutor> = Arc::new(RasterDecodeExecutor {
            pool: decode_handle,
            done_tx,
            wakeup: Arc::clone(&wakeup),
            shutdown: decode_shutdown,
            polar_rim: Arc::clone(&polar_rim),
        });
        let blob_store: Option<Arc<dyn TileBlobStore>> =
            disk_cache.map(|cache| Arc::new(SqliteBlobStore(cache)) as Arc<dyn TileBlobStore>);

        let (live_pipeline, live_pipeline_error) = match net {
            Some(net_ctx) => (
                Some(LiveTilePipeline::spawn_with_context(
                    net_ctx,
                    FetcherConfig::default(),
                    blob_store,
                    decode_executor,
                    wakeup,
                )),
                None,
            ),
            None => (
                None,
                Some(net_init_error.unwrap_or_else(|| "network context unavailable".to_owned())),
            ),
        };

        let mut manager = Self {
            active_basemap: config.active_basemap,
            bundled_provider,
            live_pipeline,
            live_pipeline_error,
            active_live_spec: None,
            config_error: None,
            observed_live_tile_bytes: 0,
            polar_rim,
            polar_revs: [0, 0],
            polar_fallback_revs: [0, 0],
            wanted_bundled: HashSet::new(),
            wanted_live: HashSet::new(),
        };

        manager.sync_active_spec(config);
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

    /// Pure carrier-routing policy matching the active basemap source and
    /// latest live polar-rim harmonic revisions.
    pub(crate) fn carrier_policy(&self) -> BasemapCarrierPolicy {
        BasemapCarrierPolicy::with_polar_revs(
            self.active_source(),
            self.polar_revs,
            self.polar_fallback_revs,
        )
    }

    /// Poll live polar-rim tracker updates and promote fallback revisions once
    /// the latest harmonic polar root tile is cached in VRAM.
    ///
    /// Returns `(snapshot_dirty, evict_old_polar)`:
    /// - `snapshot_dirty` is `true` when a pole's active revision advanced and
    ///   `DrawSnapshot` should rebuild.
    /// - `evict_old_polar` is `true` when active or fallback revisions changed
    ///   and unreachable `PolarCap` `decode_rev > 0` entries can be purged from
    ///   `GpuTileCache` / `PendingUploads`.
    pub(crate) fn sync_polar_rim(&mut self, gpu_cache: &GpuTileCache) -> (bool, bool) {
        if self.active_live_spec.is_none() {
            return (false, false);
        }
        let now = Instant::now();
        let mut changed = false;
        let mut evict_old_polar = false;
        for (idx, north) in [(0usize, true), (1usize, false)] {
            let rev = self.polar_rim.poll_revision(north, now);
            if rev != self.polar_revs[idx] && rev > 0 {
                self.polar_revs[idx] = rev;
                changed = true;
                evict_old_polar = true;
            }
            if self.polar_fallback_revs[idx] != self.polar_revs[idx] && self.polar_revs[idx] > 0 {
                let root_key = super::policy::bundled_polar_tile_key(
                    QuadTileId {
                        scheme: PlanetaryScheme::PolarCap { north },
                        level: 0,
                        x: 0,
                        y: 0,
                    },
                    self.polar_revs[idx],
                );
                if gpu_cache.contains(&root_key) {
                    self.polar_fallback_revs[idx] = self.polar_revs[idx];
                    evict_old_polar = true;
                }
            }
        }
        (changed, evict_old_polar)
    }

    /// Remaining debounce delay before pending polar-rim updates should flush
    /// (`None` when idle or in offline/ellipsoid mode).
    pub(crate) fn polar_rim_retry_delay(&self) -> Option<Duration> {
        self.active_live_spec.as_ref()?;
        self.polar_rim.next_flush_delay(Instant::now())
    }

    /// Return whether `key` is a currently reachable `PolarCap` revision
    /// (base `decode_rev == 0`, active `polar_revs`, or warm `polar_fallback_revs`).
    pub(crate) fn is_retained_polar_rev(&self, key: &TileKey) -> bool {
        let TileScheme::PolarCap { north } = key.scheme else {
            return true;
        };
        if key.decode_rev == 0 {
            return true;
        }
        let idx = usize::from(!north);
        key.decode_rev == self.polar_revs[idx] || key.decode_rev == self.polar_fallback_revs[idx]
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
            let max_bytes = report.max_tile_bytes_for_stream(StreamKey::display(spec.source_id()));
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

    /// Synchronize active basemap providers with updated [`BasemapConfigView`],
    /// returning explicit eviction intent so the caller (`SceneState`) owns
    /// GPU cache and upload queue purging.
    pub(crate) fn sync_settings(&mut self, config: BasemapConfigView<'_>) -> BasemapSyncOutcome {
        let prev_source = self.active_source();
        let prev_mode = self.active_basemap;
        let prev_spec = self.active_live_spec.clone();
        self.active_basemap = config.active_basemap;

        self.sync_active_spec(config);
        let next_source = self.active_source();

        let mut evict_source = None;
        let mut evict_polar_revs = false;
        if prev_source != next_source {
            // Keep bundled fallback tiles warm in VRAM; instruct caller to
            // purge stale tiles from the previous live source and any
            // source-specific harmonic polar-cap revisions.
            if prev_source != SourceId::BUNDLED_BASEMAP {
                evict_source = Some(prev_source);
            }
            evict_polar_revs = true;
            self.bundled_provider.cancel_polar_revs();
            self.observed_live_tile_bytes = 0;
            self.polar_revs = [0, 0];
            self.polar_fallback_revs = [0, 0];
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
            evict_polar_revs,
        }
    }

    fn sync_active_spec(&mut self, config: BasemapConfigView<'_>) {
        match spec_from_config(config) {
            Ok(spec) => {
                self.config_error = None;
                // Only activate a live spec when `live_pipeline` is actually
                // running; otherwise cleanly fall back to full `z = 0..3`
                // bundled Blue Marble while `live_pipeline_error` surfaces the
                // `"Live unavailable"` toolbar badge.
                if let Some(pipeline) = &self.live_pipeline {
                    let prev_id = self
                        .active_live_spec
                        .as_ref()
                        .map(LiveBasemapSpec::source_id);
                    let next_id = spec.as_ref().map(LiveBasemapSpec::source_id);
                    self.active_live_spec = spec.clone();
                    if prev_id != next_id {
                        self.polar_rim.reset_for_source(next_id);
                    }
                    pipeline.set_active_spec(spec);
                } else {
                    self.active_live_spec = None;
                    self.polar_rim.reset_for_source(None);
                }
            }
            Err(err) => {
                self.config_error = Some(err.to_string());
                self.active_live_spec = None;
                self.polar_rim.reset_for_source(None);
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
    #[cfg(test)]
    pub(crate) fn schedule_for_tiles(
        &mut self,
        visible_tex_ids: impl IntoIterator<Item = QuadTileId>,
        gpu_cache: &GpuTileCache,
        pending_uploads: &PendingUploads,
        appetite_bytes: usize,
    ) {
        self.schedule_for_prioritized_tiles(
            visible_tex_ids.into_iter().map(|id| (id, 0.0)),
            gpu_cache,
            pending_uploads,
            appetite_bytes,
        );
    }

    /// Schedule missing visible tiles prioritized by squared ECEF distance from
    /// the camera eye (closest / screen-center tiles admitted before distant
    /// horizon tiles whenever VRAM appetite is constrained).
    pub(crate) fn schedule_for_prioritized_tiles(
        &mut self,
        visible_tiles: impl IntoIterator<Item = (QuadTileId, f64)>,
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

        let candidates = self.collect_tile_candidates(visible_tiles, gpu_cache);
        let default_tile_bytes = Self::bundled_tile_byte_size();

        let (bundled_to_request, bundled_total_bytes) = self.plan_bundled_requests(
            candidates.bundled_candidates,
            gpu_cache,
            pending_uploads,
            default_tile_bytes,
            appetite_bytes,
        );

        if self.active_live_spec.is_some() {
            self.admit_live_candidates(
                candidates.live_candidates,
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
        &self,
        visible_tiles: impl IntoIterator<Item = (QuadTileId, f64)>,
        gpu_cache: &GpuTileCache,
    ) -> ScheduledCandidates {
        let mut out = ScheduledCandidates::default();
        let policy = self.carrier_policy();
        let mut seen_bundled: HashSet<TileKey> = HashSet::new();
        let mut record_bundled_candidate = |key: TileKey, is_polar: bool| {
            if seen_bundled.insert(key) {
                out.bundled_candidates.push((key, is_polar));
            }
        };

        if let Some(spec) = &self.active_live_spec {
            let mut live_dist_by_key: HashMap<TileKey, f64> = HashMap::new();
            let mut record_live_key = |live_key: TileKey, dist_sq: f64| {
                let d = if dist_sq.is_finite() {
                    dist_sq
                } else {
                    f64::INFINITY
                };
                live_dist_by_key
                    .entry(live_key)
                    .and_modify(|prev| *prev = prev.min(d))
                    .or_insert(d);
            };

            for (tex_id, dist_sq) in visible_tiles {
                if matches!(tex_id.scheme, PlanetaryScheme::PolarCap { .. }) {
                    let (polar_id, _) = tex_id.clamp_to_level(BUNDLED_POLAR_MAX_LEVEL);
                    let primary_polar = policy.primary_key(polar_id);
                    record_bundled_candidate(primary_polar, true);
                    if polar_id.level > 0 && primary_polar.decode_rev > 0 {
                        let (polar_root, _) = polar_id.clamp_to_level(0);
                        record_bundled_candidate(policy.primary_key(polar_root), true);
                    }
                } else {
                    let (clamped_live, _) = tex_id.clamp_to_level(spec.max_level());
                    let live_key = policy.primary_key(clamped_live);
                    record_live_key(live_key, dist_sq);

                    let carrier_level = gpu_cache.carrier_level(&live_key);
                    if carrier_level.is_none() {
                        // No live carrier yet: request the single z=0 live root
                        // plus an offline bundled fallback while streaming.
                        if clamped_live.level > 0 {
                            let (root_id, _) = clamped_live.clamp_to_level(0);
                            record_live_key(policy.primary_key(root_id), 0.0);
                        }
                        let (fb_id, _) = tex_id.clamp_to_level(BUNDLED_MAX_LEVEL.min(2));
                        if let Some(fb_key) = policy.fallback_key(fb_id) {
                            record_bundled_candidate(fb_key, false);
                        }
                    }
                    // Keep requesting a coarse z=2 progressive live carrier
                    // until at least a z>=2 live ancestor is cached (so z=0
                    // uploading never cancels z=2 mid-flight).
                    if clamped_live.level > 2 && carrier_level.is_none_or(|lvl| lvl < 2) {
                        let (coarse_id, _) = clamped_live.clamp_to_level(2);
                        record_live_key(policy.primary_key(coarse_id), dist_sq);
                    }
                }
            }

            out.live_candidates = live_dist_by_key.into_iter().collect();
        } else {
            for (tex_id, _) in visible_tiles {
                let is_polar = matches!(tex_id.scheme, PlanetaryScheme::PolarCap { .. });
                record_bundled_candidate(policy.primary_key(tex_id), is_polar);
            }
        }

        out
    }

    fn plan_bundled_requests(
        &mut self,
        mut bundled_candidates: Vec<(TileKey, bool)>,
        gpu_cache: &GpuTileCache,
        pending_uploads: &PendingUploads,
        default_tile_bytes: usize,
        total_appetite_bytes: usize,
    ) -> (Vec<TileKey>, usize) {
        if bundled_candidates.is_empty() {
            return (Vec::new(), 0);
        }
        // Under a live basemap, reserve at least 75% of basemap VRAM appetite
        // for live tiles so bundled fallbacks can never deadlock live streaming
        // under tight budgets; prioritize primary PolarCap tiles ahead of
        // temporary non-polar fallbacks.
        let has_polar = bundled_candidates.iter().any(|(_, is_polar)| *is_polar);
        let bundled_budget_bytes = if self.active_live_spec.is_some() {
            (total_appetite_bytes / 4).max(if has_polar { default_tile_bytes } else { 0 })
        } else {
            total_appetite_bytes
        };
        let bundled_cap_tiles = bundled_budget_bytes / default_tile_bytes;

        bundled_candidates.sort_by(|(ka, pa), (kb, pb)| {
            // Primary PolarCap tiles first, then coarsest-first.
            pb.cmp(pa)
                .then_with(|| ka.coarsest_first_key().cmp(&kb.coarsest_first_key()))
        });

        let mut to_request = Vec::new();
        let mut admitted_count = 0usize;
        for (key, _) in bundled_candidates.into_iter().take(bundled_cap_tiles) {
            self.wanted_bundled.insert(key);
            admitted_count += 1;
            if !gpu_cache.contains(&key)
                && !pending_uploads.contains(&key)
                && !self.bundled_provider.is_in_flight(&key)
            {
                to_request.push(key);
            }
        }

        let bundled_total_bytes = admitted_count.saturating_mul(default_tile_bytes);
        (to_request, bundled_total_bytes)
    }

    fn admit_live_candidates(
        &mut self,
        mut live_candidates: Vec<(TileKey, f64)>,
        default_tile_bytes: usize,
        live_appetite_bytes: usize,
    ) {
        let live_tile_bytes = default_tile_bytes.max(self.observed_live_tile_bytes);
        let live_tile_cap = live_appetite_bytes / live_tile_bytes;
        // Sort ALL visible live tiles (cached, queued, or uncached) together:
        // pinned z=0 root first, then closest-to-camera (screen center) before
        // distant horizon tiles, breaking ties coarsest-first. Sorting cached
        // and uncached together ensures cached horizon tiles from prior frames
        // never starve newly exposed screen-center tiles under tight caps.
        live_candidates.sort_by(|(ka, da), (kb, db)| {
            let ra = u8::from(ka.level > 0);
            let rb = u8::from(kb.level > 0);
            ra.cmp(&rb)
                .then_with(|| da.total_cmp(db))
                .then_with(|| ka.coarsest_first_key().cmp(&kb.coarsest_first_key()))
        });
        for (k, _) in live_candidates.into_iter().take(live_tile_cap) {
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

    /// Phase 3 of deterministic shutdown (after `PendingUploads::close()` and
    /// joining the shared decode pool): drop `LiveTilePipeline` ahead of
    /// `NetContext::shutdown()`.
    pub(crate) fn finish_shutdown(&mut self) {
        if let Some(pipeline) = self.live_pipeline.take() {
            pipeline.shutdown();
        }
    }
}

#[derive(Default)]
struct ScheduledCandidates {
    bundled_candidates: Vec<(TileKey, bool)>,
    live_candidates: Vec<(TileKey, f64)>,
}

#[cfg(test)]
#[path = "manager_tests.rs"]
mod tests;
