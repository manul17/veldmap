//! Scene/upload/decode-pool orchestration (`SceneState`) extracted from `app.rs`.
//!
//! Owns the GPU tile cache, bounded upload queue, shared two-lane CPU decode
//! pool (`TwoLaneDecodePool`), basemap coordinator, ordered raster overlay
//! `LayerStack` / `OverlayManager`, cached immutable `DrawSnapshot`, and
//! monotonic `scene_generation`.

use std::path::Path;
use std::sync::Arc;

use veldmap_core::budget::MemoryBudget;
use veldmap_core::geo::camera::Camera;
use veldmap_core::geo::footprint::FootprintBounds;
use veldmap_core::sync::{CancelToken, Wakeup};
use veldmap_core::tile::SourceId;
use veldmap_raster::{TwoLaneDecodePool, TwoLaneDecodePoolConfig};
use veldmap_storage::TileDiskCache;

use crate::basemap::{BasemapManager, BasemapStatusSnapshot};
use crate::gpu_tiles::{
    DEFAULT_UPLOAD_BYTE_BUDGET, DEFAULT_UPLOAD_TIME_BUDGET, GpuTileCache, PendingUploads,
    UploadBatchReport, UploadChannelClosed, UploadPassParams,
};
use crate::layers::{CursorReadout, LayerId, LayerSummary, OverlayManager};
use crate::render::{DrawSnapshot, SnapshotBuildParams, TileCarrierPolicy};
use crate::settings::AppSettings;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SnapshotKey {
    eye_bits: [u64; 3],
    up_bits: [u64; 3],
    width_px: u32,
    height_px: u32,
    ppp_bits: u32,
    show_graticule: bool,
    scene_generation: u64,
}

impl SnapshotKey {
    fn new(
        camera: &Camera,
        viewport_px: [u32; 2],
        pixels_per_point: f32,
        show_graticule: bool,
        scene_generation: u64,
    ) -> Self {
        let eye = camera.eye_ecef();
        let (_, up, _) = camera.basis();
        Self {
            eye_bits: eye.to_array().map(f64::to_bits),
            up_bits: up.to_array().map(f64::to_bits),
            width_px: viewport_px[0],
            height_px: viewport_px[1],
            ppp_bits: pixels_per_point.to_bits(),
            show_graticule,
            scene_generation,
        }
    }
}

/// Coordinates tile caches, bounded GPU uploads, shared two-lane CPU decode
/// threads, active basemap + raster overlay scheduling, and cached frame
/// snapshots.
pub(crate) struct SceneState {
    gpu_tile_cache: GpuTileCache,
    pending_uploads: PendingUploads,
    decode_shutdown: CancelToken,
    decode_pool: Option<TwoLaneDecodePool>,
    basemap: BasemapManager,
    overlays: OverlayManager,
    cached_snapshot: Option<(SnapshotKey, Arc<DrawSnapshot>)>,
    scene_generation: u64,
}

impl SceneState {
    /// Initialize GPU/upload caches, the shared two-lane CPU decode pool,
    /// basemap streaming, and local raster overlay management from startup
    /// `budget` and persisted `settings`.
    pub(crate) fn new(
        settings: &AppSettings,
        disk_cache: Option<Arc<TileDiskCache>>,
        wakeup: Wakeup,
        budget: &MemoryBudget,
    ) -> Result<Self, UploadChannelClosed> {
        let pending_uploads = PendingUploads::with_max_queued_bytes(budget.queued_upload_bytes);
        let done_tx = pending_uploads.sender().ok_or(UploadChannelClosed)?;

        let decode_shutdown = CancelToken::new();
        let mut decode_cfg = TwoLaneDecodePoolConfig::from_budget(budget);
        decode_cfg.fast.worker_count = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(2)
            .clamp(2, 4);
        let decode_pool = TwoLaneDecodePool::spawn(decode_cfg, decode_shutdown.clone());

        let basemap = BasemapManager::new(
            settings,
            disk_cache.clone(),
            done_tx.clone(),
            Arc::clone(&wakeup),
            decode_pool.handle().fast().clone(),
            decode_shutdown.clone(),
        );
        let overlays = OverlayManager::new(
            budget,
            disk_cache,
            decode_pool.handle().fast().clone(),
            decode_pool.handle().bulk().clone(),
            done_tx,
            wakeup,
            decode_shutdown.clone(),
        );

        let scene = Self {
            gpu_tile_cache: GpuTileCache::new(budget.gpu_tile_cache_bytes),
            pending_uploads,
            decode_shutdown,
            decode_pool: Some(decode_pool),
            basemap,
            overlays,
            cached_snapshot: None,
            scene_generation: 1,
        };
        scene.sync_pinned_root_sources();
        Ok(scene)
    }

    fn pinned_root_sources(&self) -> Vec<SourceId> {
        let mut sources = vec![self.basemap.active_source(), SourceId::BUNDLED_BASEMAP];
        self.overlays
            .stack()
            .extend_pinned_root_sources(&mut sources);
        sources
    }

    fn sync_pinned_root_sources(&self) {
        let sources = self.pinned_root_sources();
        self.pending_uploads.set_pinned_root_sources(&sources);
    }

    /// Bump `scene_generation` after internal basemap, layer, or upload state
    /// changes.
    fn mark_dirty(&mut self) {
        self.scene_generation = self.scene_generation.wrapping_add(1);
    }

    /// Release cached `Arc<DrawSnapshot>` references when the viewport is
    /// collapsed/minimized so evicted `GpuTile` textures are not pinned.
    pub(crate) fn clear_cached_snapshot(&mut self) {
        self.cached_snapshot = None;
    }

    /// Visible basemap attribution pill text (`None` for `EllipsoidOnly`).
    pub(crate) fn attribution(&self) -> Option<&str> {
        self.basemap.attribution()
    }

    /// Snapshot toolbar-visible live streaming / config error status.
    pub(crate) fn basemap_status(&self) -> BasemapStatusSnapshot {
        self.basemap.status_snapshot()
    }

    /// Snapshot Layer List UI summaries for all loaded raster overlays.
    pub(crate) fn layer_summaries(&self) -> Vec<LayerSummary> {
        self.overlays.layer_summaries()
    }

    /// Open a local GeoTIFF/BigTIFF/COG overlay layer and return its ID and
    /// geographic footprint bounds.
    pub(crate) fn open_overlay_file(
        &mut self,
        path: &Path,
    ) -> Result<(LayerId, Option<FootprintBounds>), String> {
        let result = self.overlays.open_local_file(path, false)?;
        self.sync_pinned_root_sources();
        self.mark_dirty();
        Ok(result)
    }

    /// Reload an existing overlay layer from disk, bypassing both the stat
    /// fingerprint cache and any cached QOI disk tiles.
    pub(crate) fn reload_overlay_bypass_cache(
        &mut self,
        id: LayerId,
    ) -> Result<Option<FootprintBounds>, String> {
        let bounds = self.overlays.reload_layer_bypass_cache(
            id,
            &mut self.gpu_tile_cache,
            &mut self.pending_uploads,
        )?;
        self.sync_pinned_root_sources();
        self.mark_dirty();
        Ok(bounds)
    }

    /// Remove an overlay layer and reclaim its VRAM/pending/chunk state.
    pub(crate) fn remove_overlay(&mut self, id: LayerId) -> bool {
        let removed =
            self.overlays
                .remove_layer(id, &mut self.gpu_tile_cache, &mut self.pending_uploads);
        if removed {
            self.sync_pinned_root_sources();
            self.mark_dirty();
        }
        removed
    }

    /// Toggle visibility of an overlay layer.
    pub(crate) fn set_overlay_visible(&mut self, id: LayerId, visible: bool) -> bool {
        let changed = self.overlays.set_visible(id, visible);
        if changed {
            self.sync_pinned_root_sources();
            self.mark_dirty();
        }
        changed
    }

    /// Set opacity (`0.0..=1.0`) of an overlay layer.
    pub(crate) fn set_overlay_opacity(&mut self, id: LayerId, opacity: f32) -> bool {
        let changed = self.overlays.set_opacity(id, opacity);
        if changed {
            self.mark_dirty();
        }
        changed
    }

    /// Query the zoom-invariant native CPU display RGBA readout under
    /// `(lat_deg, lon_deg)` on the topmost visible overlay.
    pub(crate) fn cursor_overlay_readout(
        &self,
        lat_deg: f64,
        lon_deg: f64,
    ) -> Option<CursorReadout> {
        self.overlays.query_cursor_readout(lat_deg, lon_deg)
    }

    /// Cancel any in-flight cursor readout probe (e.g. when moving off-globe).
    pub(crate) fn cancel_cursor_readout(&self) {
        self.overlays.cancel_cursor_readout();
    }

    /// Return the native source pixel size `(m_x, m_y)` of the topmost visible
    /// overlay (used in the HUD when the cursor is not over a specific overlay).
    pub(crate) fn topmost_visible_source_pixel_size_m(&self) -> Option<(f64, f64)> {
        self.overlays
            .stack()
            .layers()
            .iter()
            .rev()
            .find(|l| l.visible)
            .and_then(|l| l.source_pixel_size_m)
    }

    /// Synchronize basemap providers with updated [`AppSettings`], purging any
    /// superseded live source from VRAM / pending uploads and updating pinned
    /// roots. Returns `true` if visible scene state changed.
    pub(crate) fn sync_basemap_settings(&mut self, settings: &AppSettings) -> bool {
        let outcome = self.basemap.sync_settings(settings);
        if let Some(evict) = outcome.evict_source {
            self.gpu_tile_cache.remove_source(evict);
            self.pending_uploads.clear_source(evict);
        }
        if outcome.changed {
            self.sync_pinned_root_sources();
            self.mark_dirty();
        }
        outcome.changed
    }

    /// Retry failed/offline live tile requests and reconcile providers.
    pub(crate) fn retry_live_and_reconcile(&mut self) {
        self.basemap.retry_live();
        self.reconcile_providers();
    }

    /// Reconcile background tile providers against current GPU + pending state.
    pub(crate) fn reconcile_providers(&mut self) {
        self.basemap
            .reconcile_providers(&self.gpu_tile_cache, &self.pending_uploads);
    }

    /// Drain background tile completions and upload coarsest-first within the
    /// per-frame VRAM/time budget while protecting active-frame carriers.
    pub(crate) fn drain_tile_uploads(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> UploadBatchReport {
        let mut wanted = self.basemap.wanted_tile_sets().to_vec();
        self.overlays.stack().extend_wanted_tile_sets(&mut wanted);
        let protected = self
            .cached_snapshot
            .as_ref()
            .map(|(_, snap)| &snap.carrier_keys);
        let report = self.pending_uploads.drain_and_upload(
            device,
            queue,
            &mut self.gpu_tile_cache,
            UploadPassParams {
                wanted: &wanted,
                protected,
                byte_budget: DEFAULT_UPLOAD_BYTE_BUDGET,
                time_budget: DEFAULT_UPLOAD_TIME_BUDGET,
            },
        );
        self.basemap.record_upload_report(&report);
        self.overlays.record_upload_report(&report);
        if report.uploaded > 0 {
            self.mark_dirty();
        }
        report
    }

    /// Return the cached immutable `Arc<DrawSnapshot>` when `(camera, viewport,
    /// show_graticule, scene_generation)` is unchanged (avoiding quadtree
    /// traversal on 120 Hz mouse hover), or rebuild and schedule missing tiles.
    /// Returns `(snapshot, reconciled_providers)`.
    pub(crate) fn snapshot_for_frame(
        &mut self,
        camera: Camera,
        width_px: u32,
        height_px: u32,
        pixels_per_point: f32,
        show_graticule: bool,
    ) -> (Arc<DrawSnapshot>, bool) {
        let key = SnapshotKey::new(
            &camera,
            [width_px, height_px],
            pixels_per_point,
            show_graticule,
            self.scene_generation,
        );
        if let Some((cached_key, snapshot)) = &self.cached_snapshot
            && *cached_key == key
        {
            return (Arc::clone(snapshot), false);
        }

        let textured = self.basemap.is_textured();
        let quadtree_config = self.basemap.quadtree_config();
        let policy = self.basemap.carrier_policy();
        let carrier =
            textured.then_some((&policy as &dyn TileCarrierPolicy, &mut self.gpu_tile_cache));
        let mut snapshot_inner = DrawSnapshot::build(
            SnapshotBuildParams {
                camera,
                width_px,
                height_px,
                pixels_per_point,
                show_graticule,
                config: &quadtree_config,
            },
            carrier,
        );

        let basemap_weight = if textured {
            self.basemap.appetite_weight_bytes()
        } else {
            0
        };
        let overlay_weight = self.overlays.stack().total_appetite_weight_bytes();
        let total_weight = basemap_weight.saturating_add(overlay_weight).max(1);

        let overlay_layers = self.overlays.build_and_schedule(
            &camera,
            width_px,
            height_px,
            &mut self.gpu_tile_cache,
            &self.pending_uploads,
            total_weight,
            &mut snapshot_inner.carrier_keys,
        );
        snapshot_inner.overlays = overlay_layers;

        let snapshot = Arc::new(snapshot_inner);

        let basemap_appetite_bytes = self
            .gpu_tile_cache
            .proportional_pyramid_byte_cap(basemap_weight, total_weight);
        self.basemap.schedule_for_tiles(
            snapshot.visible_tex_ids(),
            &self.gpu_tile_cache,
            &self.pending_uploads,
            basemap_appetite_bytes,
        );

        self.cached_snapshot = Some((key, Arc::clone(&snapshot)));
        (snapshot, true)
    }

    /// Phase 1 of deterministic shutdown: cancel all worker tokens and signal
    /// live/bundled/overlay pipelines BEFORE closing `PendingUploads`.
    pub(crate) fn begin_shutdown(&mut self) {
        self.overlays.begin_shutdown();
        self.basemap.begin_shutdown();
        self.decode_shutdown.cancel();
    }

    /// Phase 2 of deterministic shutdown: close bounded upload queue so
    /// producers exit promptly on send.
    pub(crate) fn close_uploads(&mut self) {
        self.pending_uploads.close();
    }

    /// Phase 3 of deterministic shutdown: join the shared CPU decode pool
    /// BEFORE shutting down `LiveTilePipeline`'s `tokio::Runtime`.
    pub(crate) fn finish_shutdown(&mut self) {
        if let Some(pool) = self.decode_pool.take() {
            pool.shutdown();
        }
        self.basemap.finish_shutdown();
    }
}

impl Drop for SceneState {
    fn drop(&mut self) {
        self.begin_shutdown();
        self.close_uploads();
        self.finish_shutdown();
    }
}
