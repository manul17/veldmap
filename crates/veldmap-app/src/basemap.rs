//! Live + offline basemap orchestration extracted from `app.rs`.
//!
//! Coordinates durable user settings (`AppSettings`), crash-safe SQLite WAL
//! disk caching (`TileDiskCache`), the bundled offline NASA Blue Marble worker
//! (`BundledBasemapProvider`), and the async HTTP streaming + CPU decode
//! pipeline (`LiveTilePipeline`), enforcing:
//! - `0` live HTTP requests for `PolarCap` keys (always routed to bundled
//!   `SourceId::BUNDLED_BASEMAP` offline polar caps)
//! - `0` live HTTP requests above per-source `max_level`
//! - Seamless Frame-1 & offline fallback to bundled Blue Marble tiles
//! - `< 50%` VRAM appetite capping across visible + in-flight + queued tiles
//! - Deterministic shutdown (`CancelToken` -> drop completion rx / join decode
//!   pool -> drop `tokio::Runtime`)

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::mpsc::Sender;

use veldmap_core::tile::{
    DecodedTile, PixelFormat, QuadtreeConfig, SourceId, TileKey, TileScheme, max_mip_levels,
};
use veldmap_net::{FetcherConfig, LiveBasemapSpec, LiveTilePipeline, NetworkStatus};
use veldmap_raster::{BUNDLED_MAX_LEVEL, BUNDLED_POLAR_MAX_LEVEL, BUNDLED_TILE_PX};
use veldmap_storage::{AppSettings, BasemapSelection, DiskCacheStats, StoragePaths, TileDiskCache};

use crate::gpu_tiles::{
    BundledBasemapProvider, GpuTileCache, PendingUploads, Wakeup, bundled_tile_key,
    source_tile_key, texture_byte_size,
};
use crate::render::DrawSnapshot;

/// Attribution string displayed for the bundled public-domain Blue Marble
/// offline basemap.
pub const BUNDLED_ATTRIBUTION: &str = "Imagery: NASA Blue Marble Next Generation (Public Domain)";

/// Coordinates offline bundled + live HTTP basemap streaming, disk caching,
/// and durable user settings.
pub struct BasemapManager {
    paths: Option<StoragePaths>,
    settings: AppSettings,
    disk_cache: Option<Arc<TileDiskCache>>,
    bundled_provider: BundledBasemapProvider,
    live_pipeline: Option<LiveTilePipeline>,
    active_live_spec: Option<LiveBasemapSpec>,
    config_error: Option<String>,
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

        let bundled_provider = BundledBasemapProvider::spawn(done_tx.clone(), Arc::clone(&wakeup));

        let live_pipeline = match LiveTilePipeline::spawn(
            FetcherConfig::default(),
            disk_cache.clone(),
            done_tx,
            wakeup,
        ) {
            Ok(pipeline) => Some(pipeline),
            Err(err) => {
                tracing::error!(error = %err, "Failed to start live HTTP tile pipeline; falling back to offline basemap");
                None
            }
        };

        let mut manager = Self {
            paths: resolved_paths,
            settings,
            disk_cache,
            bundled_provider,
            live_pipeline,
            active_live_spec: None,
            config_error: None,
            wanted_bundled: HashSet::new(),
            wanted_live: HashSet::new(),
            wanted_all: HashSet::new(),
        };

        manager.sync_active_spec();
        let scheme = manager.quadtree_config().scheme;
        manager.bundled_provider.seed_roots(scheme);
        manager
    }

    /// Current persisted settings.
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

    /// Validation or configuration error (e.g. invalid Custom XYZ URL).
    pub fn config_error(&self) -> Option<&str> {
        self.config_error.as_deref()
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

    /// Observable SQLite disk cache stats, if disk caching is active.
    pub fn disk_cache_stats(&self) -> Option<DiskCacheStats> {
        self.disk_cache.as_ref().and_then(|c| c.stats().ok())
    }

    /// Clear all cached tiles from disk and reset transient failure markers.
    pub fn clear_disk_cache(&mut self) -> Result<(), String> {
        if let Some(cache) = &self.disk_cache {
            cache.clear().map_err(|e| e.to_string())?;
        }
        if let Some(pipeline) = &self.live_pipeline {
            pipeline.retry_failed();
        }
        Ok(())
    }

    /// Retry failed/offline live tile requests immediately.
    pub fn retry_live(&self) {
        if let Some(pipeline) = &self.live_pipeline {
            pipeline.retry_failed();
        }
    }

    /// Apply updated [`AppSettings`], persisting durably via
    /// `write_atomic_durable()`, updating the disk cache cap, evicting GPU
    /// tiles from the previous live source if changed, and switching the live
    /// pipeline. Returns `true` when the active basemap or visual state changed.
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
            && let Some(cache) = &self.disk_cache
            && let Err(err) = cache.set_max_bytes(next_cache_bytes)
        {
            tracing::warn!(error = %err, "Failed to update SQLite disk cache budget");
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
            self.wanted_live.clear();
            self.wanted_bundled.clear();
            self.wanted_all.clear();
            self.bundled_provider
                .seed_roots(self.quadtree_config().scheme);
        } else if prev_mode != self.settings.active_basemap {
            self.bundled_provider
                .seed_roots(self.quadtree_config().scheme);
        }

        changed || prev_source != next_source
    }

    fn sync_active_spec(&mut self) {
        match LiveBasemapSpec::from_settings(&self.settings) {
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
    /// pipeline, respecting `< 50%` VRAM appetite caps and enforcing `0` HTTP
    /// requests for `PolarCap` and `0` HTTP requests above `max_level`.
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
        let tile_bytes = texture_byte_size(
            BUNDLED_TILE_PX,
            BUNDLED_TILE_PX,
            PixelFormat::Rgba8UnormSrgb,
            mips,
        );
        let appetite_cap = gpu_cache.max_tiles_for_pyramid(1, tile_bytes);

        let mut missing_bundled = Vec::new();
        let mut cached_bundled_wanted = 0usize;

        if let Some(spec) = &self.active_live_spec {
            let mut live_candidates = Vec::new();
            let mut cached_live_wanted = 0usize;
            let mut queued_live_wanted = 0usize;
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
                    // Non-polar tile: request live tile (strictly <= spec.max_level)
                    // AND ensure bundled coarse fallback root stays warm if missing.
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

                    // Keep a bundled coarse fallback tile warm so offline/loading
                    // tiles always render Blue Marble instead of blank ellipsoid.
                    if !gpu_cache.contains(&live_key) {
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

            // Cap live wanted set to the < 50% VRAM appetite budget (coarsest first),
            // accounting for both already-cached and queued-for-upload live tiles.
            live_candidates.sort_by_key(TileKey::coarsest_first_key);
            let live_budget = appetite_cap
                .saturating_sub(cached_live_wanted)
                .saturating_sub(queued_live_wanted);
            for k in live_candidates.into_iter().take(live_budget) {
                self.wanted_live.insert(k);
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

        // Build `wanted_all` from the final appetite-bounded live + bundled sets.
        self.wanted_all.extend(self.wanted_live.iter().copied());
        self.wanted_all.extend(self.wanted_bundled.iter().copied());

        // Reconcile providers before submitting new bundled jobs so in-flight
        // and queued counts never double-count against appetite_cap.
        self.reconcile_providers(gpu_cache, pending_uploads);

        let committed_bundled = cached_bundled_wanted
            .saturating_add(self.bundled_provider.in_flight_count())
            .saturating_add(pending_uploads.len());
        let remaining_bundled = appetite_cap.saturating_sub(committed_bundled);
        missing_bundled.sort_by_key(TileKey::coarsest_first_key);
        for key in missing_bundled.into_iter().take(remaining_bundled) {
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
    /// join CPU decode/bundled worker threads and shut down `tokio::Runtime`.
    pub fn finish_shutdown(&mut self) {
        self.bundled_provider.join();
        if let Some(pipeline) = self.live_pipeline.take() {
            pipeline.shutdown();
        }
    }
}
