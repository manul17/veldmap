//! Scene, tile-upload, and two-lane decode-pool orchestration (`SceneState`).
//!
//! Owns the GPU tile cache, bounded upload queue, shared two-lane CPU decode
//! pool (`TwoLaneDecodePool`), shared [`NetContext`], [`BasemapManager`],
//! raster [`OverlayManager`], GeoJSON [`VectorManager`], interactive geodesic
//! [`MeasureSession`], Cloud/STAC [`StacSession`], cached immutable
//! [`DrawSnapshot`], and monotonic `scene_generation`.

use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use veldmap_core::budget::MemoryBudget;
use veldmap_core::geo::Camera;
use veldmap_core::sync::{CancelToken, Wakeup};
use veldmap_core::tile::{SourceId, StreamKey, TileKey, TileStreamSpec};
use veldmap_net::{NetConfig, NetContext};
use veldmap_raster::{TwoLaneDecodePool, TwoLaneDecodePoolConfig};
use veldmap_storage::TileDiskCache;

use crate::basemap::{BasemapManager, BasemapManagerDeps};
use crate::gpu_tiles::{
    DEFAULT_UPLOAD_BYTE_BUDGET, DEFAULT_UPLOAD_TIME_BUDGET, GpuTileCache, PendingUploads,
    UploadBatchReport, UploadChannelClosed, UploadPassParams,
};
use crate::layers::{
    CrsOverride, CursorReadout, LayerId, OpenOverlayRequest, OverlayEdit, OverlayFrameCtx,
    OverlayIoEvent, OverlayManager, OverlayManagerDeps, OverlayMutation, RemoteCogRequest,
};
use crate::measure::MeasureSession;
use crate::render::{DrawSnapshot, SnapshotBuildParams, TileCarrierPolicy};
use crate::stac::StacSession;
use crate::storage::settings::BasemapConfigView;
use crate::vectors::{VectorEdit, VectorIoEvent, VectorLayerId, VectorManager, is_geojson_path};

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

/// Repaint scheduling hint returned by [`SceneState::end_frame`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SceneRepaintHint {
    /// No scene-driven repaint needed.
    None,
    /// Schedule a delayed retry after an overlay decode lane saturation backoff.
    After(Duration),
    /// Request an immediate follow-up frame (budget-deferred uploads remain or
    /// partial upload progress may have unprotected a deferred tile).
    Immediate,
}

/// Unified background I/O event drained from both raster and vector managers at
/// the start of each frame.
#[derive(Clone, Debug)]
pub(crate) enum SceneIoEvent {
    Raster(OverlayIoEvent),
    Vector(VectorIoEvent),
}

#[derive(Clone, Copy, Debug, Default)]
struct FrameState {
    reconciled: bool,
    uploaded: usize,
    remaining_ready: usize,
    deferred_protected: usize,
}

struct CachedFrameSnapshot {
    key: SnapshotKey,
    snapshot: Arc<DrawSnapshot>,
    /// Exact GPU carrier tile keys referenced by `snapshot`'s basemap and
    /// overlay layers, retained outside [`DrawSnapshot`] for active-frame
    /// upload eviction protection.
    carrier_keys: HashSet<TileKey>,
}

/// Coordinates tile caches, bounded GPU uploads, shared two-lane CPU decode
/// threads, active basemap + raster/vector overlay scheduling, cursor probes,
/// and cached frame snapshots.
pub(crate) struct SceneState {
    gpu_tile_cache: GpuTileCache,
    default_gpu_tile_cache_bytes: usize,
    pending_uploads: PendingUploads,
    decode_shutdown: CancelToken,
    decode_pool: Option<TwoLaneDecodePool>,
    net: Option<Arc<NetContext>>,
    basemap: BasemapManager,
    overlays: OverlayManager,
    vectors: VectorManager,
    measure: MeasureSession,
    stac: StacSession,
    cached_snapshot: Option<CachedFrameSnapshot>,
    scene_generation: u64,
    active_frame: Option<FrameState>,
}

impl SceneState {
    /// Initialize GPU/upload caches, the shared [`NetContext`], the shared
    /// two-lane CPU decode pool, basemap streaming, local + remote raster
    /// overlay management, STAC controller, and the GeoJSON vector manager from
    /// startup `budget` and persisted `settings`.
    pub(crate) fn new(
        basemap_config: BasemapConfigView<'_>,
        gpu_tile_cache_override_bytes: Option<usize>,
        disk_cache: Option<Arc<TileDiskCache>>,
        wakeup: Wakeup,
        budget: &MemoryBudget,
    ) -> Result<Self, UploadChannelClosed> {
        let pending_uploads = PendingUploads::with_max_queued_bytes(budget.queued_upload_bytes);
        let done_tx = pending_uploads.sender().ok_or(UploadChannelClosed)?;

        let (net, net_init_error) = match NetContext::new(NetConfig::from_budget(budget)) {
            Ok(ctx) => (Some(Arc::new(ctx)), None),
            Err(err) => {
                tracing::warn!(error = %err, "Failed to initialize shared NetContext");
                (None, Some(err.to_string()))
            }
        };

        let decode_shutdown = CancelToken::new();
        let mut decode_cfg = TwoLaneDecodePoolConfig::from_budget(budget);
        decode_cfg.fast.worker_count = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(2)
            .clamp(2, 4);
        let decode_pool = TwoLaneDecodePool::spawn(decode_cfg, decode_shutdown.clone());

        let basemap = BasemapManager::new(BasemapManagerDeps {
            config: basemap_config,
            disk_cache: disk_cache.clone(),
            net: net.clone(),
            net_init_error,
            done_tx: done_tx.clone(),
            wakeup: Arc::clone(&wakeup),
            decode_handle: decode_pool.fast_handle(),
            decode_shutdown: decode_shutdown.clone(),
        });
        let vectors = VectorManager::new(Arc::clone(&wakeup), decode_shutdown.clone());
        let stac = StacSession::new(net.clone(), Arc::clone(&wakeup), decode_shutdown.clone());
        let overlays = OverlayManager::new(OverlayManagerDeps {
            budget: *budget,
            disk_cache,
            net: net.clone(),
            fast_pool: decode_pool.fast_handle(),
            bulk_pool: decode_pool.bulk_handle(),
            done_tx,
            wakeup,
            shutdown: decode_shutdown.clone(),
        });

        let gpu_cap = gpu_tile_cache_override_bytes.unwrap_or(budget.gpu_tile_cache_bytes);
        let scene = Self {
            gpu_tile_cache: GpuTileCache::new(gpu_cap),
            default_gpu_tile_cache_bytes: budget.gpu_tile_cache_bytes,
            pending_uploads,
            decode_shutdown,
            decode_pool: Some(decode_pool),
            net,
            basemap,
            overlays,
            vectors,
            measure: MeasureSession::default(),
            stac,
            cached_snapshot: None,
            scene_generation: 1,
            active_frame: None,
        };
        scene.sync_pinned_root_streams();
        Ok(scene)
    }

    fn pinned_root_streams(&self) -> Vec<StreamKey> {
        let mut streams = vec![
            StreamKey::display(self.basemap.active_source()),
            StreamKey::display(SourceId::BUNDLED_BASEMAP),
        ];
        self.overlays.extend_pinned_root_streams(&mut streams);
        streams
    }

    fn sync_pinned_root_streams(&self) {
        let streams = self.pinned_root_streams();
        self.pending_uploads.set_pinned_root_streams(&streams);
    }

    /// Bump `scene_generation` after internal basemap, layer, or upload state
    /// changes.
    fn mark_dirty(&mut self) {
        self.scene_generation = self.scene_generation.wrapping_add(1);
    }

    fn mark_dirty_if(&mut self, changed: bool) -> bool {
        if changed {
            self.mark_dirty();
        }
        changed
    }

    /// Purge a superseded/removed source from both GPU VRAM and pending uploads.
    fn purge_source_caches(&mut self, src: SourceId) {
        self.gpu_tile_cache.remove_source(src);
        self.pending_uploads.clear_source(src);
    }

    /// Re-sync pinned root streams and bump `scene_generation` after basemap or
    /// overlay topology changes.
    fn mark_topology_changed(&mut self) {
        self.sync_pinned_root_streams();
        self.mark_dirty();
    }

    /// Release cached `Arc<DrawSnapshot>` references when the viewport is
    /// collapsed/minimized so evicted `GpuTile` textures are not pinned.
    pub(crate) fn clear_cached_snapshot(&mut self) {
        self.cached_snapshot = None;
    }

    /// Read-only access to the basemap coordinator (all mutations stay on
    /// `SceneState` to preserve `scene_generation` and cache invariants).
    pub(crate) fn basemap(&self) -> &BasemapManager {
        &self.basemap
    }

    /// Read-only access to the raster overlay manager.
    pub(crate) fn overlays(&self) -> &OverlayManager {
        &self.overlays
    }

    /// Read-only access to the GeoJSON vector overlay manager.
    pub(crate) fn vectors(&self) -> &VectorManager {
        &self.vectors
    }

    /// Read-only access to the interactive geodesic measurement session.
    pub(crate) fn measure(&self) -> &MeasureSession {
        &self.measure
    }

    /// Read-only access to the STAC catalog & AOI controller.
    pub(crate) fn stac(&self) -> &StacSession {
        &self.stac
    }

    /// Mutate the STAC catalog & AOI controller and mark the scene dirty
    /// automatically whenever its visual overlay revision advances.
    pub(crate) fn mutate_stac<R>(&mut self, f: impl FnOnce(&mut StacSession) -> R) -> R {
        let rev_before = self.stac.rev();
        let out = f(&mut self.stac);
        self.mark_dirty_if(self.stac.rev() != rev_before);
        out
    }

    /// Queue a non-blocking remote Cloud-Optimized GeoTIFF open.
    pub(crate) fn begin_open_remote_cog(&self, request: RemoteCogRequest) -> Result<(), String> {
        self.overlays.begin_open_remote_cog(request)
    }

    /// Mutate the geodesic measurement session and mark the scene dirty
    /// automatically whenever its revision advances.
    pub(crate) fn mutate_measure(&mut self, f: impl FnOnce(&mut MeasureSession)) -> bool {
        let rev_before = self.measure.rev();
        f(&mut self.measure);
        self.mark_dirty_if(self.measure.rev() != rev_before)
    }

    /// Total count of loaded raster + vector layers (O(1), lock-free).
    pub(crate) fn total_layer_count(&self) -> usize {
        self.overlays.layer_count() + self.vectors.len()
    }

    /// Update the zoom-invariant CPU cursor readout for `cursor_geo`, cancelling
    /// any in-flight probe when the cursor leaves the globe.
    pub(crate) fn update_cursor_readout(
        &self,
        cursor_geo: Option<(f64, f64)>,
    ) -> Option<CursorReadout> {
        match cursor_geo {
            Some((lat, lon)) => self.overlays.query_cursor_readout(lat, lon),
            None => {
                self.overlays.cancel_cursor_readout();
                None
            }
        }
    }

    /// Queue a non-blocking local raster or GeoJSON vector open based on path
    /// extension.
    pub(crate) fn begin_open_local_path(
        &self,
        path: &Path,
        fly_to: bool,
        crs_override: Option<CrsOverride>,
    ) -> Result<(), String> {
        if is_geojson_path(path) {
            self.begin_open_vector(path, fly_to, true, 1.0)
        } else {
            self.begin_open_overlay(OpenOverlayRequest {
                path,
                fly_to,
                visible: true,
                opacity: 1.0,
                crs_override,
                style: None,
            })
        }
    }

    /// Queue a non-blocking local raster overlay open with explicit initial
    /// display/style/CRS options (used by session restoration and local open).
    pub(crate) fn begin_open_overlay(&self, request: OpenOverlayRequest<'_>) -> Result<(), String> {
        self.overlays.begin_open_file(request)
    }

    /// Queue a non-blocking GeoJSON vector layer open with explicit initial
    /// visibility and opacity.
    pub(crate) fn begin_open_vector(
        &self,
        path: &Path,
        fly_to: bool,
        visible: bool,
        opacity: f32,
    ) -> Result<(), String> {
        self.vectors.begin_open(path, fly_to, visible, opacity)
    }

    /// Apply an infallible [`VectorEdit`] to `id` and mark the scene dirty
    /// whenever layer state changed.
    pub(crate) fn edit_vector(&mut self, id: VectorLayerId, edit: VectorEdit) -> bool {
        let changed = self.vectors.edit(id, edit);
        self.mark_dirty_if(changed)
    }

    fn apply_overlay_mutation(&mut self, mutation: OverlayMutation) -> bool {
        for src in mutation.purge_sources {
            self.purge_source_caches(src);
        }
        if mutation.topology_changed {
            self.mark_topology_changed();
        } else if mutation.changed {
            self.mark_dirty();
        }
        mutation.changed
    }

    /// Apply an infallible [`OverlayEdit`] to `id` and synchronize caches and
    /// `scene_generation`.
    pub(crate) fn edit_overlay(&mut self, id: LayerId, edit: OverlayEdit) -> bool {
        let mutation = self.overlays.edit_layer(id, edit);
        self.apply_overlay_mutation(mutation)
    }

    pub(crate) fn override_overlay_crs(
        &mut self,
        id: LayerId,
        crs_override: Option<CrsOverride>,
    ) -> Result<Option<veldmap_core::geo::FootprintBounds>, String> {
        let (bounds, mutation) = self.overlays.override_layer_crs(id, crs_override)?;
        let _ = self.apply_overlay_mutation(mutation);
        Ok(bounds)
    }

    /// Queue a non-blocking reload of an existing overlay layer from disk,
    /// bypassing both the stat fingerprint cache and any cached disk tiles.
    pub(crate) fn begin_reload_overlay_bypass_cache(&mut self, id: LayerId) -> Result<(), String> {
        let mutation = self.overlays.begin_reload_bypass_cache(id)?;
        let _ = self.apply_overlay_mutation(mutation);
        Ok(())
    }

    /// Drain completed background raster, vector, and STAC I/O events in a
    /// single batch.
    pub(crate) fn poll_io_events(&mut self) -> (Vec<SceneIoEvent>, bool) {
        let (raster_events, mutation) = self.overlays.poll_io_events();
        let _ = self.apply_overlay_mutation(mutation);

        let vector_events = self.vectors.poll_events();
        if vector_events
            .iter()
            .any(|ev| matches!(ev, VectorIoEvent::Opened { .. }))
        {
            self.mark_dirty();
        }

        let stac_rev_before = self.stac.rev();
        let stac_changed = self.stac.poll_results();
        if self.stac.rev() != stac_rev_before {
            self.mark_dirty();
        }

        let mut events = Vec::with_capacity(raster_events.len() + vector_events.len());
        events.extend(raster_events.into_iter().map(SceneIoEvent::Raster));
        events.extend(vector_events.into_iter().map(SceneIoEvent::Vector));
        (events, stac_changed)
    }

    pub(crate) fn set_overlay_stream(
        &mut self,
        id: LayerId,
        spec: TileStreamSpec,
    ) -> Result<bool, String> {
        let mutation = self.overlays.set_layer_stream(id, spec)?;
        Ok(self.apply_overlay_mutation(mutation))
    }

    /// Synchronize basemap providers and GPU tile cache cap with updated
    /// settings slices, purging any superseded live source from VRAM / pending
    /// uploads and updating pinned roots.
    pub(crate) fn sync_settings(
        &mut self,
        basemap_config: BasemapConfigView<'_>,
        gpu_tile_cache_override_bytes: Option<usize>,
    ) -> bool {
        let next_gpu_cap =
            gpu_tile_cache_override_bytes.unwrap_or(self.default_gpu_tile_cache_bytes);
        let mut gpu_changed = false;
        if self.gpu_tile_cache.max_bytes() != next_gpu_cap {
            self.gpu_tile_cache.set_max_bytes(next_gpu_cap);
            gpu_changed = true;
        }
        let outcome = self.basemap.sync_settings(basemap_config);
        if let Some(evict) = outcome.evict_source {
            self.purge_source_caches(evict);
        }
        if outcome.changed {
            self.mark_topology_changed();
        } else if gpu_changed {
            self.mark_dirty();
        }
        outcome.changed || gpu_changed
    }

    /// Retry failed/offline live tile requests and reconcile providers.
    pub(crate) fn retry_live_and_reconcile(&mut self) {
        self.basemap.retry_live();
        self.reconcile_providers();
    }

    /// Reconcile background tile providers against current GPU + pending state.
    fn reconcile_providers(&mut self) {
        self.basemap
            .reconcile_providers(&self.gpu_tile_cache, &self.pending_uploads);
        self.overlays
            .reconcile(&self.gpu_tile_cache, &self.pending_uploads);
        if let Some(frame) = &mut self.active_frame {
            frame.reconciled = true;
        }
    }

    /// Begin a UI frame by draining background tile completions into VRAM
    /// within the per-frame byte/time budget while protecting active-frame
    /// carriers. Call [`Self::end_frame`] once the frame's UI/viewport pass
    /// completes.
    pub(crate) fn begin_frame(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let report = self.drain_tile_uploads(device, queue);
        self.active_frame = Some(FrameState {
            reconciled: false,
            uploaded: report.uploaded,
            remaining_ready: report.remaining_ready,
            deferred_protected: report.deferred_protected,
        });
    }

    /// Drain background tile completions and upload coarsest-first within the
    /// per-frame VRAM/time budget while protecting active-frame carriers.
    fn drain_tile_uploads(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> UploadBatchReport {
        let mut wanted = self.basemap.wanted_tile_sets().to_vec();
        self.overlays.extend_wanted_tile_sets(&mut wanted);
        let protected = self
            .cached_snapshot
            .as_ref()
            .map(|cached| &cached.carrier_keys);
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
        if report.uploaded > 0 || self.overlays.take_reschedule_needed() {
            self.mark_dirty();
        }
        report
    }

    /// Return the cached immutable `Arc<DrawSnapshot>` when `(camera, viewport,
    /// show_graticule, scene_generation)` is unchanged (avoiding quadtree
    /// traversal on 120 Hz mouse hover), or rebuild and schedule missing tiles.
    pub(crate) fn snapshot_for_frame(
        &mut self,
        camera: Camera,
        width_px: u32,
        height_px: u32,
        pixels_per_point: f32,
        show_graticule: bool,
    ) -> Arc<DrawSnapshot> {
        let key = SnapshotKey::new(
            &camera,
            [width_px, height_px],
            pixels_per_point,
            show_graticule,
            self.scene_generation,
        );
        if let Some(cached) = &self.cached_snapshot
            && cached.key == key
        {
            return Arc::clone(&cached.snapshot);
        }

        let textured = self.basemap.is_textured();
        let quadtree_config = self.basemap.quadtree_config();
        let policy = self.basemap.carrier_policy();
        let carrier =
            textured.then_some((&policy as &dyn TileCarrierPolicy, &mut self.gpu_tile_cache));
        let (mut snapshot_inner, mut carrier_keys) = DrawSnapshot::build(
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
        let overlay_weight = self.overlays.total_appetite_weight_bytes();
        let total_weight = basemap_weight.saturating_add(overlay_weight).max(1);

        let overlay_layers = self.overlays.build_and_schedule(&mut OverlayFrameCtx {
            camera: &camera,
            width_px,
            height_px,
            gpu_cache: &mut self.gpu_tile_cache,
            pending: &self.pending_uploads,
            total_appetite_weight: total_weight,
            carrier_keys: &mut carrier_keys,
        });
        snapshot_inner.overlays = overlay_layers;
        let mut draw_vectors = self.vectors.draw_layers();
        self.stac.push_draw_layers(&mut draw_vectors);
        if let Some(measure_layer) = self.measure.draw_layer() {
            draw_vectors.push(measure_layer);
        }
        snapshot_inner.vectors = draw_vectors;

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

        self.cached_snapshot = Some(CachedFrameSnapshot {
            key,
            snapshot: Arc::clone(&snapshot),
            carrier_keys,
        });
        if let Some(frame) = &mut self.active_frame {
            frame.reconciled = true;
        }
        snapshot
    }

    /// Finalize the active UI frame: reconcile background providers if uploads
    /// progressed or remained deferred without a snapshot rebuild (e.g. a
    /// collapsed window), and return the scene-driven [`SceneRepaintHint`].
    pub(crate) fn end_frame(&mut self) -> SceneRepaintHint {
        let frame = self.active_frame.take().unwrap_or_default();
        if !frame.reconciled
            && (frame.uploaded > 0 || frame.remaining_ready > 0 || frame.deferred_protected > 0)
        {
            self.reconcile_providers();
        }

        if frame.remaining_ready > 0 || (frame.uploaded > 0 && frame.deferred_protected > 0) {
            SceneRepaintHint::Immediate
        } else if let Some(delay) = self.overlays.retry_delay() {
            SceneRepaintHint::After(delay)
        } else {
            SceneRepaintHint::None
        }
    }

    /// Phase 1 of deterministic shutdown: cancel all worker tokens and signal
    /// live/bundled/overlay/vector/STAC pipelines and `NetContext` BEFORE
    /// closing `PendingUploads`.
    pub(crate) fn begin_shutdown(&mut self) {
        self.overlays.begin_shutdown();
        self.vectors.begin_shutdown();
        self.basemap.begin_shutdown();
        self.stac.cancel();
        self.decode_shutdown.cancel();
        if let Some(net) = &self.net {
            net.signal_shutdown();
        }
    }

    /// Phase 2 of deterministic shutdown: close bounded upload queue so
    /// producers exit promptly on send.
    pub(crate) fn close_uploads(&mut self) {
        self.pending_uploads.close();
    }

    /// Phase 3 of deterministic shutdown: join the shared CPU decode pool,
    /// remote I/O pool, local I/O + tile-write workers, and basemap pipeline
    /// BEFORE shutting down the shared `NetContext` `tokio::Runtime`.
    pub(crate) fn finish_shutdown(&mut self) {
        if let Some(pool) = self.decode_pool.take() {
            pool.shutdown();
        }
        self.overlays.finish_shutdown();
        self.vectors.finish_shutdown();
        self.basemap.finish_shutdown();
        if let Some(net) = self.net.take() {
            net.shutdown();
        }
    }
}

impl Drop for SceneState {
    fn drop(&mut self) {
        self.begin_shutdown();
        self.close_uploads();
        self.finish_shutdown();
    }
}
