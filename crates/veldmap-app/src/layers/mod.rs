//! Ordered raster overlay [`LayerStack`], `PassKey`-scoped two-lane
//! [`OverlayManager`] with QOI disk caching, and zoom-invariant CPU cursor
//! readout.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use veldmap_core::budget::{DecodeBudgetTracker, MemoryBudget};
use veldmap_core::geo::camera::Camera;
use veldmap_core::geo::footprint::FootprintBounds;
use veldmap_core::geo::overlay::{Frame, OverlayCullTree, build_warp_patch};
use veldmap_core::glam::DVec3;
use veldmap_core::io::ByteSource;
use veldmap_core::sync::{CancelToken, Wakeup, lock_unpoisoned};
use veldmap_core::tile::{
    DecodedTile, PixelFormat, RASTER_TILE_SIZE, SourceId, TileKey, cell_image_rect, cell_uv,
};
use veldmap_raster::{
    DecodeContext, DecodeJob, DecodeJobKey, DecodePoolHandle, PassKey, PyramidMeta, RasterSource,
    ServeMode, SharedChunkCache, TiffSource, TileScheduler, frame_from_description,
};
use veldmap_storage::{DiskTileKey, LocalFileSource, LocalFingerprintCache, TileDiskCache};

use crate::gpu_tiles::{GpuTileCache, PendingUploads, UploadBatchReport, UploadSender};
use crate::render::{DrawOverlayLayer, DrawOverlayPatch, OverlayMeshKey};

/// Default unpacked VRAM byte weight for an 8-bit `512 x 512` display tile
/// (`1.0 MiB`).
const DEFAULT_DISPLAY_TILE_BYTES: usize = (RASTER_TILE_SIZE * RASTER_TILE_SIZE * 4) as usize;

/// Unique identifier for a loaded overlay layer in [`LayerStack`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct LayerId(pub(crate) u64);

/// Read-only summary of an overlay layer rendered in the Layer List UI.
#[derive(Clone, Debug)]
pub(crate) struct LayerSummary {
    pub(crate) id: LayerId,
    pub(crate) name: String,
    pub(crate) path: PathBuf,
    pub(crate) visible: bool,
    pub(crate) opacity: f32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) source_pixel_size_m: Option<(f64, f64)>,
    pub(crate) footprint: Option<FootprintBounds>,
    pub(crate) in_flight: usize,
    pub(crate) pass_active: bool,
}

/// Zoom-invariant CPU cursor readout for the topmost visible overlay covering
/// the cursor position.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CursorReadout {
    pub(crate) layer_name: String,
    pub(crate) pixel: (u32, u32),
    pub(crate) source_pixel_size_m: Option<(f64, f64)>,
    pub(crate) rgba: Option<[u8; 4]>,
    pub(crate) pending: bool,
}

#[derive(Clone)]
struct CachedWarpCell {
    origin_ecef_m: DVec3,
    segments: u32,
    flip_winding: bool,
    rel_positions: Arc<[[f32; 3]]>,
}

/// Single georeferenced raster overlay inside [`LayerStack`].
pub(crate) struct OverlayLayer {
    pub(crate) id: LayerId,
    pub(crate) name: String,
    pub(crate) path: PathBuf,
    pub(crate) source: Arc<dyn RasterSource>,
    pub(crate) meta: PyramidMeta,
    pub(crate) fingerprint_hex: String,
    pub(crate) frame: Frame,
    pub(crate) relaid: u64,
    pub(crate) cull: OverlayCullTree,
    pub(crate) visible: bool,
    pub(crate) opacity: f32,
    pub(crate) source_pixel_size_m: Option<(f64, f64)>,
    pub(crate) footprint: Option<FootprintBounds>,
    pub(crate) wanted: HashSet<TileKey>,
    pub(crate) observed_tile_bytes: usize,
    warp_cache: HashMap<(u8, u32, u32), CachedWarpCell>,
}

impl OverlayLayer {
    fn from_source(
        id: LayerId,
        path: PathBuf,
        source: Arc<dyn RasterSource>,
        frame: Frame,
        visible: bool,
        opacity: f32,
        relaid: u64,
    ) -> Self {
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .map_or_else(|| path.display().to_string(), str::to_owned);
        let meta = source.pyramid_meta();
        let fingerprint_hex = source.fingerprint().to_hex();
        let desc = source.description();
        let source_pixel_size_m = frame.pixel_size_m(desc.width, desc.height);
        let footprint = frame.footprint_bounds();
        let cull = OverlayCullTree::new(desc.width, desc.height, meta.max_level());
        Self {
            id,
            name,
            path,
            source,
            meta,
            fingerprint_hex,
            frame,
            relaid,
            cull,
            visible,
            opacity: opacity.clamp(0.0, 1.0),
            source_pixel_size_m,
            footprint,
            wanted: HashSet::new(),
            observed_tile_bytes: DEFAULT_DISPLAY_TILE_BYTES,
            warp_cache: HashMap::new(),
        }
    }

    pub(crate) fn source_id(&self) -> SourceId {
        self.meta.source
    }

    pub(crate) fn appetite_weight_bytes(&self) -> u64 {
        if self.visible {
            (self.observed_tile_bytes as u64).max(1024)
        } else {
            0
        }
    }
}

/// Ordered collection of draped raster overlays in painter's order
/// (index `0` drawn right above the basemap, last index drawn on top beneath
/// the procedural graticule).
#[derive(Default)]
pub(crate) struct LayerStack {
    layers: Vec<OverlayLayer>,
}

impl LayerStack {
    pub(crate) fn push(&mut self, layer: OverlayLayer) {
        self.layers.push(layer);
    }

    pub(crate) fn remove(&mut self, id: LayerId) -> Option<OverlayLayer> {
        let idx = self.layers.iter().position(|l| l.id == id)?;
        Some(self.layers.remove(idx))
    }

    pub(crate) fn get(&self, id: LayerId) -> Option<&OverlayLayer> {
        self.layers.iter().find(|l| l.id == id)
    }

    pub(crate) fn get_mut(&mut self, id: LayerId) -> Option<&mut OverlayLayer> {
        self.layers.iter_mut().find(|l| l.id == id)
    }

    pub(crate) fn layers(&self) -> &[OverlayLayer] {
        &self.layers
    }

    pub(crate) fn layers_mut(&mut self) -> &mut [OverlayLayer] {
        &mut self.layers
    }

    pub(crate) fn total_appetite_weight_bytes(&self) -> u64 {
        self.layers
            .iter()
            .fold(0u64, |acc, l| acc.saturating_add(l.appetite_weight_bytes()))
    }

    pub(crate) fn extend_pinned_root_sources(&self, out: &mut Vec<SourceId>) {
        for layer in &self.layers {
            if layer.visible {
                let src = layer.source_id();
                if !out.contains(&src) {
                    out.push(src);
                }
            }
        }
    }

    pub(crate) fn extend_wanted_tile_sets<'a>(&'a self, out: &mut Vec<&'a HashSet<TileKey>>) {
        for layer in &self.layers {
            if layer.visible && !layer.wanted.is_empty() {
                out.push(&layer.wanted);
            }
        }
    }
}

/// RAII guard guaranteeing a pointwise tile never remains stuck in-flight in
/// [`TileScheduler`] if its `DecodeJob` is evicted from the queue, cancelled,
/// or panics inside a worker thread.
///
/// Queue evictions and cancellations clear in-flight state via
/// [`TileScheduler::cancel_in_flight`] without penalizing the tile's failure
/// retry budget; only actual decode errors call `on_tile_failed`.
struct PointwiseFlightGuard {
    scheduler: Arc<Mutex<TileScheduler>>,
    key: TileKey,
    wakeup: Wakeup,
    completed: bool,
}

impl PointwiseFlightGuard {
    fn new(scheduler: Arc<Mutex<TileScheduler>>, key: TileKey, wakeup: Wakeup) -> Self {
        Self {
            scheduler,
            key,
            wakeup,
            completed: false,
        }
    }

    fn mark_arrived(mut self) {
        self.completed = true;
        lock_unpoisoned(&self.scheduler).on_tile_arrived(&self.key);
        (self.wakeup)();
    }

    fn mark_failed(mut self, terminal: bool) {
        self.completed = true;
        lock_unpoisoned(&self.scheduler).on_tile_failed(&self.key, terminal);
        (self.wakeup)();
    }

    fn mark_cancelled(mut self) {
        self.completed = true;
        lock_unpoisoned(&self.scheduler).cancel_in_flight(&self.key);
        (self.wakeup)();
    }
}

impl Drop for PointwiseFlightGuard {
    fn drop(&mut self) {
        if !self.completed {
            lock_unpoisoned(&self.scheduler).cancel_in_flight(&self.key);
            (self.wakeup)();
        }
    }
}

/// RAII guard guaranteeing an active sequential pass in [`TileScheduler`] is
/// always closed even if evicted from `DecodeQueue`, rejected on submit, or
/// unwound by a worker panic.
///
/// Cancellation, eviction, and rejected submits abort via
/// [`TileScheduler::abort_pass`] without penalizing undelivered tiles; only
/// genuine pass errors call `finish_pass(..., false)`.
struct PassFlightGuard {
    scheduler: Arc<Mutex<TileScheduler>>,
    pass_key: PassKey,
    pass_id: u64,
    wakeup: Wakeup,
    finished: bool,
}

impl PassFlightGuard {
    fn new(
        scheduler: Arc<Mutex<TileScheduler>>,
        pass_key: PassKey,
        pass_id: u64,
        wakeup: Wakeup,
    ) -> Self {
        Self {
            scheduler,
            pass_key,
            pass_id,
            wakeup,
            finished: false,
        }
    }

    fn finish(mut self, succeeded: bool) {
        self.finished = true;
        lock_unpoisoned(&self.scheduler).finish_pass(self.pass_key, self.pass_id, succeeded);
        (self.wakeup)();
    }

    fn abort(mut self) {
        self.finished = true;
        lock_unpoisoned(&self.scheduler).abort_pass(self.pass_key, self.pass_id);
        (self.wakeup)();
    }
}

impl Drop for PassFlightGuard {
    fn drop(&mut self) {
        if !self.finished {
            lock_unpoisoned(&self.scheduler).abort_pass(self.pass_key, self.pass_id);
            (self.wakeup)();
        }
    }
}

/// RAII guard ensuring an evicted, cancelled, or panicked cursor readout job
/// never leaves `CursorProbeCache` permanently stuck in `pending: true`.
struct CursorProbeGuard {
    cache: Arc<Mutex<CursorProbeCache>>,
    probe_key: (SourceId, u32, u32),
    wakeup: Wakeup,
    settled: bool,
}

impl CursorProbeGuard {
    fn new(
        cache: Arc<Mutex<CursorProbeCache>>,
        probe_key: (SourceId, u32, u32),
        wakeup: Wakeup,
    ) -> Self {
        Self {
            cache,
            probe_key,
            wakeup,
            settled: false,
        }
    }

    fn resolve(mut self, rgba: Option<[u8; 4]>) {
        self.settled = true;
        let should_wake = {
            let mut guard = lock_unpoisoned(&self.cache);
            if guard.active_key == Some(self.probe_key) {
                guard.resolved_key = Some(self.probe_key);
                guard.resolved_rgba = rgba;
                guard.pending = false;
                guard.active_cancel = None;
                true
            } else {
                false
            }
        };
        if should_wake {
            (self.wakeup)();
        }
    }

    fn clear_transient(mut self) {
        self.settled = true;
        let should_wake = {
            let mut guard = lock_unpoisoned(&self.cache);
            if guard.active_key == Some(self.probe_key) {
                guard.active_key = None;
                guard.active_cancel = None;
                guard.pending = false;
                true
            } else {
                false
            }
        };
        if should_wake {
            (self.wakeup)();
        }
    }
}

impl Drop for CursorProbeGuard {
    fn drop(&mut self) {
        if !self.settled {
            let should_wake = {
                let mut guard = lock_unpoisoned(&self.cache);
                if guard.active_key == Some(self.probe_key) {
                    guard.active_key = None;
                    guard.active_cancel = None;
                    guard.pending = false;
                    true
                } else {
                    false
                }
            };
            if should_wake {
                (self.wakeup)();
            }
        }
    }
}

#[derive(Default)]
struct CursorProbeCache {
    active_key: Option<(SourceId, u32, u32)>,
    active_cancel: Option<CancelToken>,
    resolved_key: Option<(SourceId, u32, u32)>,
    resolved_rgba: Option<[u8; 4]>,
    pending: bool,
}

/// Coordinates local raster overlays: opening/reloading via `RasterSource`,
/// `PassKey`-scoped two-lane scheduling (`fast` pointwise / `bulk` pass),
/// 8-bit QOI SQLite disk caching, and zoom-invariant CPU cursor readouts.
pub(crate) struct OverlayManager {
    stack: LayerStack,
    next_layer_id: u64,
    budget: MemoryBudget,
    chunk_cache: Arc<SharedChunkCache>,
    budget_tracker: Arc<DecodeBudgetTracker>,
    fingerprint_cache: LocalFingerprintCache,
    scheduler: Arc<Mutex<TileScheduler>>,
    disk_cache: Option<Arc<TileDiskCache>>,
    fast_pool: DecodePoolHandle,
    bulk_pool: DecodePoolHandle,
    done_tx: UploadSender,
    wakeup: Wakeup,
    shutdown: CancelToken,
    cursor_cache: Arc<Mutex<CursorProbeCache>>,
}

impl OverlayManager {
    pub(crate) fn new(
        budget: &MemoryBudget,
        disk_cache: Option<Arc<TileDiskCache>>,
        fast_pool: DecodePoolHandle,
        bulk_pool: DecodePoolHandle,
        done_tx: UploadSender,
        wakeup: Wakeup,
        shutdown: CancelToken,
    ) -> Self {
        Self {
            stack: LayerStack::default(),
            next_layer_id: 1,
            budget: *budget,
            chunk_cache: Arc::new(SharedChunkCache::new(budget.decoded_chunk_cache_bytes)),
            budget_tracker: Arc::new(DecodeBudgetTracker::new(budget.decode_working_bytes)),
            fingerprint_cache: LocalFingerprintCache::new(),
            scheduler: Arc::new(Mutex::new(TileScheduler::default())),
            disk_cache,
            fast_pool,
            bulk_pool,
            done_tx,
            wakeup,
            shutdown,
            cursor_cache: Arc::new(Mutex::new(CursorProbeCache::default())),
        }
    }

    pub(crate) fn stack(&self) -> &LayerStack {
        &self.stack
    }

    /// Return immutable UI summaries of all loaded overlay layers.
    pub(crate) fn layer_summaries(&self) -> Vec<LayerSummary> {
        let sched = lock_unpoisoned(&self.scheduler);
        self.stack
            .layers()
            .iter()
            .map(|layer| {
                let desc = layer.source.description();
                let pass_key = layer.meta.pass_key();
                LayerSummary {
                    id: layer.id,
                    name: layer.name.clone(),
                    path: layer.path.clone(),
                    visible: layer.visible,
                    opacity: layer.opacity,
                    width: desc.width,
                    height: desc.height,
                    source_pixel_size_m: layer.source_pixel_size_m,
                    footprint: layer.footprint,
                    in_flight: sched.in_flight_count_for_source(layer.source_id()),
                    pass_active: sched.has_active_pass(pass_key),
                }
            })
            .collect()
    }

    /// Open a local GeoTIFF/BigTIFF/COG from `path`, adding it to the top of
    /// [`LayerStack`].
    pub(crate) fn open_local_file(
        &mut self,
        path: &Path,
        bypass_cache: bool,
    ) -> Result<(LayerId, Option<FootprintBounds>), String> {
        let (source, frame) = self.open_source_and_frame(path, bypass_cache)?;
        let id = LayerId(self.next_layer_id);
        self.next_layer_id = self.next_layer_id.saturating_add(1);
        let layer = OverlayLayer::from_source(id, path.to_path_buf(), source, frame, true, 1.0, 1);
        let bounds = layer.footprint;
        self.stack.push(layer);
        Ok((id, bounds))
    }

    fn open_source_and_frame(
        &self,
        path: &Path,
        bypass_cache: bool,
    ) -> Result<(Arc<dyn RasterSource>, Frame), String> {
        let fingerprint = self
            .fingerprint_cache
            .get_or_compute_with_bypass(path, bypass_cache, &self.shutdown)
            .map_err(|e| format!("failed to fingerprint '{}': {e}", path.display()))?;
        let file_source: Arc<dyn ByteSource> = Arc::new(
            LocalFileSource::open(path)
                .map_err(|e| format!("failed to open '{}': {e}", path.display()))?,
        );
        let tiff = TiffSource::open(file_source, fingerprint, &self.budget, &self.shutdown)
            .map_err(|e| format!("failed to read GeoTIFF '{}': {e}", path.display()))?;
        let frame = frame_from_description(tiff.description())
            .map_err(|e| format!("'{}': {e}", path.display()))?;
        Ok((Arc::new(tiff), frame))
    }

    /// Remove a layer from [`LayerStack`], cancelling active jobs and purging
    /// VRAM/pending/chunk caches for its source (unless another layer shares
    /// the exact same content fingerprint).
    pub(crate) fn remove_layer(
        &mut self,
        id: LayerId,
        gpu_cache: &mut GpuTileCache,
        pending: &mut PendingUploads,
    ) -> bool {
        let Some(removed) = self.stack.remove(id) else {
            return false;
        };
        let src = removed.source_id();
        let still_shared = self
            .stack
            .layers()
            .iter()
            .any(|other| other.source_id() == src);
        if !still_shared {
            self.purge_source_runtime_state(src, gpu_cache, pending);
        }
        true
    }

    /// Reload a layer from disk, bypassing both [`LocalFingerprintCache`] and
    /// any cached QOI disk tiles for its previous fingerprint.
    pub(crate) fn reload_layer_bypass_cache(
        &mut self,
        id: LayerId,
        gpu_cache: &mut GpuTileCache,
        pending: &mut PendingUploads,
    ) -> Result<Option<FootprintBounds>, String> {
        let Some(existing) = self.stack.get(id) else {
            return Err("layer no longer exists".to_owned());
        };
        let path = existing.path.clone();
        let visible = existing.visible;
        let opacity = existing.opacity;
        let next_relaid = existing.relaid.saturating_add(1);
        let old_src = existing.source_id();
        let old_fp = existing.fingerprint_hex.clone();

        self.fingerprint_cache.invalidate(&path);
        if let Some(disk) = &self.disk_cache {
            let _ = disk.clear_source(&old_fp);
        }
        self.purge_source_runtime_state(old_src, gpu_cache, pending);

        let (source, frame) = self.open_source_and_frame(&path, true)?;
        let new_fp = source.fingerprint().to_hex();
        if new_fp != old_fp
            && let Some(disk) = &self.disk_cache
        {
            let _ = disk.clear_source(&new_fp);
        }
        let replacement =
            OverlayLayer::from_source(id, path, source, frame, visible, opacity, next_relaid);
        let bounds = replacement.footprint;
        if let Some(slot) = self.stack.get_mut(id) {
            *slot = replacement;
        }
        Ok(bounds)
    }

    fn purge_source_runtime_state(
        &self,
        source_id: SourceId,
        gpu_cache: &mut GpuTileCache,
        pending: &mut PendingUploads,
    ) {
        lock_unpoisoned(&self.scheduler).reset_source(source_id);
        self.chunk_cache.invalidate_source(source_id);
        gpu_cache.remove_source(source_id);
        pending.clear_source(source_id);
        let mut cursor = lock_unpoisoned(&self.cursor_cache);
        if cursor
            .active_key
            .is_some_and(|(src, _, _)| src == source_id)
            || cursor
                .resolved_key
                .is_some_and(|(src, _, _)| src == source_id)
        {
            if let Some(tok) = cursor.active_cancel.take() {
                tok.cancel();
            }
            *cursor = CursorProbeCache::default();
        }
    }

    pub(crate) fn set_visible(&mut self, id: LayerId, visible: bool) -> bool {
        let Some(layer) = self.stack.get_mut(id) else {
            return false;
        };
        if layer.visible == visible {
            return false;
        }
        layer.visible = visible;
        if !visible {
            layer.wanted.clear();
        }
        true
    }

    pub(crate) fn set_opacity(&mut self, id: LayerId, opacity: f32) -> bool {
        let clamped = opacity.clamp(0.0, 1.0);
        let Some(layer) = self.stack.get_mut(id) else {
            return false;
        };
        if (layer.opacity - clamped).abs() <= f32::EPSILON {
            return false;
        }
        layer.opacity = clamped;
        true
    }

    pub(crate) fn record_upload_report(&mut self, report: &UploadBatchReport) {
        for &(src, max_bytes) in &report.max_tile_bytes_by_source {
            for layer in self.stack.layers_mut() {
                if layer.source_id() == src {
                    layer.observed_tile_bytes = layer.observed_tile_bytes.max(max_bytes);
                }
            }
        }
    }

    /// Build immutable [`DrawOverlayLayer`]s for the current frame, update
    /// per-layer wanted sets, and schedule missing tiles onto the two-lane
    /// decode pool (`fast` for pointwise / QOI cache, `bulk` for sequential
    /// passes).
    #[expect(
        clippy::too_many_arguments,
        reason = "frame orchestration passes camera, viewport, caches, and budget weights"
    )]
    pub(crate) fn build_and_schedule(
        &mut self,
        camera: &Camera,
        width_px: u32,
        height_px: u32,
        gpu_cache: &mut GpuTileCache,
        pending: &PendingUploads,
        total_appetite_weight: u64,
        carrier_keys: &mut HashSet<TileKey>,
    ) -> Vec<DrawOverlayLayer> {
        let aspect = f64::from(width_px.max(1)) / f64::from(height_px.max(1));
        let vp = camera.view_projection(aspect);
        let eye = camera.eye_ecef();
        let frustum = camera.frustum(aspect);
        let viewport_mpp = camera.metres_per_pixel(f64::from(height_px.max(1)));

        let mut draw_layers = Vec::new();
        for layer in self.stack.layers_mut() {
            if !layer.visible {
                layer.wanted.clear();
                continue;
            }
            let desc = layer.source.description();
            let (img_w, img_h) = (desc.width, desc.height);
            let max_level = layer.meta.max_level();

            let weight = layer.appetite_weight_bytes();
            let appetite_bytes =
                gpu_cache.proportional_pyramid_byte_cap(weight, total_appetite_weight);
            let cap_tiles = (appetite_bytes / layer.observed_tile_bytes.max(1)).max(1);

            let sharpest = layer.frame.sharpest_level(img_w, viewport_mpp, max_level);
            let (target_level, visible_cells) =
                select_target_rung(layer, sharpest, cap_tiles, &frustum, &vp, eye);

            // Track wanted keys for upload prioritization and root pinning.
            layer.wanted.clear();
            let top_w = veldmap_core::tile::grid_dim(veldmap_core::tile::level_size(
                img_w,
                u32::from(max_level),
            ));
            let top_h = veldmap_core::tile::grid_dim(veldmap_core::tile::level_size(
                img_h,
                u32::from(max_level),
            ));
            for ry in 0..top_h {
                for rx in 0..top_w {
                    layer.wanted.insert(layer.meta.tile_key(max_level, rx, ry));
                }
            }
            for &(x, y) in &visible_cells {
                layer.wanted.insert(layer.meta.tile_key(target_level, x, y));
            }

            // Retain only visible cells in the CPU warp-patch cache.
            let visible_set: HashSet<(u8, u32, u32)> = visible_cells
                .iter()
                .map(|&(x, y)| (target_level, x, y))
                .collect();
            layer.warp_cache.retain(|key, _| visible_set.contains(key));

            let mut patches = Vec::with_capacity(visible_cells.len());
            for &(x, y) in &visible_cells {
                let want_key = layer.meta.tile_key(target_level, x, y);
                let Some((carrier_key, gpu_tile)) = gpu_cache.find_carrier(&want_key) else {
                    continue;
                };
                carrier_keys.insert(carrier_key);

                let cell_rect = cell_image_rect(x, y, u32::from(target_level), img_w, img_h);
                let uv_corners = cell_uv(
                    cell_rect,
                    &carrier_key,
                    gpu_tile.width,
                    gpu_tile.height,
                    gpu_tile.valid_w,
                    gpu_tile.valid_h,
                    gpu_tile.gutter,
                );
                let uv_rect = [
                    uv_corners[0],
                    uv_corners[1],
                    uv_corners[2] - uv_corners[0],
                    uv_corners[3] - uv_corners[1],
                ];
                let source_id = layer.source_id();
                let relaid = layer.relaid;
                let cached = layer
                    .warp_cache
                    .entry((target_level, x, y))
                    .or_insert_with(|| {
                        let patch = build_warp_patch(&layer.frame, img_w, img_h, cell_rect);
                        CachedWarpCell {
                            origin_ecef_m: patch.origin_ecef_m,
                            segments: patch.segments,
                            flip_winding: patch.flip_winding,
                            rel_positions: Arc::from(patch.rel_positions),
                        }
                    });
                patches.push(DrawOverlayPatch {
                    mesh_key: OverlayMeshKey {
                        source: source_id,
                        relaid,
                        level: target_level,
                        x,
                        y,
                    },
                    origin_ecef_m: cached.origin_ecef_m,
                    segments: cached.segments,
                    flip_winding: cached.flip_winding,
                    rel_positions: Arc::clone(&cached.rel_positions),
                    uv_rect,
                    texture: gpu_tile,
                });
            }
            if !patches.is_empty() {
                draw_layers.push(DrawOverlayLayer {
                    opacity: layer.opacity,
                    patches,
                });
            }

            // Schedule missing rungs coarsest-first onto the two-lane decode pool.
            schedule_layer_rungs(
                layer,
                target_level,
                &visible_cells,
                &frustum,
                &vp,
                eye,
                gpu_cache,
                pending,
                &self.scheduler,
                &self.chunk_cache,
                &self.budget_tracker,
                &self.budget,
                self.disk_cache.as_ref(),
                &self.fast_pool,
                &self.bulk_pool,
                &self.done_tx,
                &self.wakeup,
                &self.shutdown,
            );
        }

        draw_layers
    }

    /// Cancel any in-flight cursor readout probe (e.g. when the pointer leaves
    /// the globe).
    pub(crate) fn cancel_cursor_readout(&self) {
        let mut state = lock_unpoisoned(&self.cursor_cache);
        if let Some(cancel) = state.active_cancel.take() {
            cancel.cancel();
        }
        state.active_key = None;
        state.pending = false;
    }

    /// Query the zoom-invariant native CPU display RGBA readout under the
    /// cursor `(lat_deg, lon_deg)` on the topmost visible overlay covering that
    /// point. Never blocks the UI thread: misses dispatch a single cancelled-on-
    /// move job onto `fast_pool` and wake the UI when resolved.
    pub(crate) fn query_cursor_readout(&self, lat_deg: f64, lon_deg: f64) -> Option<CursorReadout> {
        let topmost = self.stack.layers().iter().rev().find_map(|layer| {
            if !layer.visible {
                return None;
            }
            let (fx, fy) = layer.frame.fraction_at(lat_deg, lon_deg)?;
            let desc = layer.source.description();
            let px = ((fx * f64::from(desc.width)).floor() as i64)
                .clamp(0, i64::from(desc.width.saturating_sub(1))) as u32;
            let py = ((fy * f64::from(desc.height)).floor() as i64)
                .clamp(0, i64::from(desc.height.saturating_sub(1))) as u32;
            Some((layer, px, py))
        });

        let Some((layer, px, py)) = topmost else {
            self.cancel_cursor_readout();
            return None;
        };

        let probe_key = (layer.source_id(), px, py);
        let submit_job = {
            let mut state = lock_unpoisoned(&self.cursor_cache);
            if state.resolved_key == Some(probe_key) {
                return Some(CursorReadout {
                    layer_name: layer.name.clone(),
                    pixel: (px, py),
                    source_pixel_size_m: layer.source_pixel_size_m,
                    rgba: state.resolved_rgba,
                    pending: false,
                });
            }

            if state.active_key != Some(probe_key) {
                if let Some(prev) = state.active_cancel.take() {
                    prev.cancel();
                }
                let cancel = self.shutdown.child_token();
                state.active_key = Some(probe_key);
                state.active_cancel = Some(cancel.clone());
                state.pending = true;
                Some(cancel)
            } else {
                None
            }
        };

        if let Some(cancel) = submit_job {
            let source = Arc::clone(&layer.source);
            let chunk_cache = Arc::clone(&self.chunk_cache);
            let budget_tracker = Arc::clone(&self.budget_tracker);
            let budget = self.budget;
            let probe_guard = CursorProbeGuard::new(
                Arc::clone(&self.cursor_cache),
                probe_key,
                Arc::clone(&self.wakeup),
            );
            // Use `max_level()` so the single 4 KiB interactive cursor readout
            // gets coarse priority in `DecodeQueue` and is not immediately
            // evicted behind a burst of level-0 tile jobs.
            let job_key = DecodeJobKey::Tile(TileKey {
                source: layer.source_id(),
                band: veldmap_core::tile::BandKey::Display,
                scheme: layer.meta.scheme(),
                format: PixelFormat::Rgba8UnormSrgb,
                gutter: 0,
                decode_rev: u32::MAX,
                level: layer.meta.max_level(),
                x: px,
                y: py,
            });
            let job = DecodeJob {
                key: job_key,
                queued_bytes: 4096,
                cancel: cancel.clone(),
                work: Box::new(move || {
                    let ctx = DecodeContext {
                        chunk_cache: &chunk_cache,
                        budget_tracker: &budget_tracker,
                        budget: &budget,
                        cancel: &cancel,
                    };
                    match source.sample_display_rgba(px, py, &ctx) {
                        Ok(rgba) => probe_guard.resolve(rgba),
                        Err(err) if err.is_cancelled() => probe_guard.clear_transient(),
                        Err(_) => probe_guard.resolve(None),
                    }
                }),
            };
            // Drop on rejected submit automatically triggers `CursorProbeGuard::drop`,
            // clearing `active_key` and `pending` cleanly.
            let _ = self.fast_pool.submit(job);
        }

        let pending = lock_unpoisoned(&self.cursor_cache).pending;
        Some(CursorReadout {
            layer_name: layer.name.clone(),
            pixel: (px, py),
            source_pixel_size_m: layer.source_pixel_size_m,
            rgba: None,
            pending,
        })
    }

    /// Cancel active overlay passes and cursor probes on shutdown.
    pub(crate) fn begin_shutdown(&self) {
        let mut sched = lock_unpoisoned(&self.scheduler);
        for layer in self.stack.layers() {
            sched.reset_source(layer.source_id());
        }
        drop(sched);
        let mut cursor = lock_unpoisoned(&self.cursor_cache);
        if let Some(tok) = cursor.active_cancel.take() {
            tok.cancel();
        }
    }
}

fn select_target_rung(
    layer: &mut OverlayLayer,
    sharpest: u8,
    cap_tiles: usize,
    frustum: &veldmap_core::geo::CullFrustum,
    vp: &veldmap_core::glam::DMat4,
    eye: DVec3,
) -> (u8, Vec<(u32, u32)>) {
    let max_level = layer.meta.max_level();
    let finest = layer
        .meta
        .finest_fitting_level()
        .map_or(max_level, |fit| sharpest.max(fit));

    let mut level = finest.min(max_level);
    loop {
        let cells = layer
            .cull
            .visible_cells(&layer.frame, level, frustum, vp, eye);
        if cells.len() <= cap_tiles || level >= max_level {
            return (level, cells);
        }
        level = level.saturating_add(1);
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "internal two-lane overlay scheduling helper"
)]
fn schedule_layer_rungs(
    layer: &mut OverlayLayer,
    target_level: u8,
    target_cells: &[(u32, u32)],
    frustum: &veldmap_core::geo::CullFrustum,
    vp: &veldmap_core::glam::DMat4,
    eye: DVec3,
    gpu_cache: &GpuTileCache,
    pending: &PendingUploads,
    scheduler: &Arc<Mutex<TileScheduler>>,
    chunk_cache: &Arc<SharedChunkCache>,
    budget_tracker: &Arc<DecodeBudgetTracker>,
    budget: &MemoryBudget,
    disk_cache: Option<&Arc<TileDiskCache>>,
    fast_pool: &DecodePoolHandle,
    bulk_pool: &DecodePoolHandle,
    done_tx: &UploadSender,
    wakeup: &Wakeup,
    shutdown: &CancelToken,
) {
    if target_cells.is_empty() {
        return;
    }

    let ladder = layer.meta.ladder(target_level);
    let is_cached = |key: &TileKey| gpu_cache.contains(key) || pending.contains(key);

    // Walk the ladder coarsest-first so coarse root coverage appears
    // immediately before fine tiles queue up.
    for &rung in &ladder {
        let Some(plan) = layer.meta.level(rung) else {
            continue;
        };
        if !plan.fits {
            continue;
        }
        let rung_cells: Vec<(u32, u32)> = if rung == target_level {
            target_cells.to_vec()
        } else {
            layer
                .cull
                .visible_cells(&layer.frame, rung, frustum, vp, eye)
        };
        if rung_cells.is_empty() {
            continue;
        }
        let candidates: Vec<TileKey> = rung_cells
            .iter()
            .map(|&(x, y)| layer.meta.tile_key(rung, x, y))
            .collect();

        match plan.serve {
            ServeMode::Pointwise => {
                let admitted = {
                    let mut sched = lock_unpoisoned(scheduler);
                    sched.admit_pointwise(&layer.meta, rung, &candidates, is_cached)
                };
                if admitted.is_empty() {
                    continue;
                }
                for tile_key in admitted {
                    submit_pointwise_job(
                        tile_key,
                        &layer.source,
                        &layer.fingerprint_hex,
                        scheduler,
                        chunk_cache,
                        budget_tracker,
                        *budget,
                        disk_cache.cloned(),
                        fast_pool,
                        done_tx,
                        wakeup,
                        shutdown,
                    );
                }
                // Schedule one unfinished rung at a time per frame to keep
                // coarsest-first convergence snappy.
                break;
            }
            ServeMode::Pass { from } => {
                let pass_key = layer.meta.pass_key();
                let (pass_info, pass_active) = {
                    let mut sched = lock_unpoisoned(scheduler);
                    let _ = sched.update_target(pass_key, rung, Some(from));
                    let missing =
                        sched.check_pass_candidates(&layer.meta, rung, &candidates, is_cached);
                    if missing.is_empty() {
                        (None, sched.has_active_pass(pass_key))
                    } else {
                        let pass_cancel = shutdown.child_token();
                        match sched.begin_pass(
                            pass_key,
                            rung,
                            Some(from),
                            pass_cancel.clone(),
                            missing.clone(),
                        ) {
                            Ok(pass_id) => (Some((pass_id, pass_cancel, missing)), true),
                            Err(_) => (None, true),
                        }
                    }
                };
                if let Some((pass_id, pass_cancel, missing)) = pass_info {
                    submit_pass_job(
                        pass_key,
                        pass_id,
                        pass_cancel,
                        missing,
                        layer.wanted.clone(),
                        &layer.source,
                        &layer.meta,
                        &layer.fingerprint_hex,
                        scheduler,
                        chunk_cache,
                        budget_tracker,
                        *budget,
                        disk_cache.cloned(),
                        bulk_pool,
                        done_tx,
                        wakeup,
                    );
                }
                if pass_active {
                    break;
                }
            }
        }
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "submits self-contained pointwise decode closure to fast lane"
)]
fn submit_pointwise_job(
    tile_key: TileKey,
    source: &Arc<dyn RasterSource>,
    fingerprint_hex: &str,
    scheduler: &Arc<Mutex<TileScheduler>>,
    chunk_cache: &Arc<SharedChunkCache>,
    budget_tracker: &Arc<DecodeBudgetTracker>,
    budget: MemoryBudget,
    disk_cache: Option<Arc<TileDiskCache>>,
    fast_pool: &DecodePoolHandle,
    done_tx: &UploadSender,
    wakeup: &Wakeup,
    shutdown: &CancelToken,
) {
    let source = Arc::clone(source);
    let fp = fingerprint_hex.to_owned();
    let sched = Arc::clone(scheduler);
    let chunk_cache = Arc::clone(chunk_cache);
    let budget_tracker = Arc::clone(budget_tracker);
    let done_tx = done_tx.clone();
    let cancel = shutdown.child_token();
    let guard = PointwiseFlightGuard::new(Arc::clone(&sched), tile_key, Arc::clone(wakeup));

    let job = DecodeJob {
        key: DecodeJobKey::Tile(tile_key),
        queued_bytes: DEFAULT_DISPLAY_TILE_BYTES,
        cancel: cancel.clone(),
        work: Box::new(move || {
            let disk_key = DiskTileKey::from_tile_key(fp, &tile_key);
            if let Some(ref disk) = disk_cache
                && let Ok(Some((w, h, rgba))) = disk.get_qoi_rgba8(&disk_key)
                && let Ok(tile) = DecodedTile::try_new(w, h, PixelFormat::Rgba8UnormSrgb, rgba)
            {
                let _ = done_tx.send((tile_key, tile));
                guard.mark_arrived();
                return;
            }

            let ctx = DecodeContext {
                chunk_cache: &chunk_cache,
                budget_tracker: &budget_tracker,
                budget: &budget,
                cancel: &cancel,
            };
            match source.decode_pointwise_tile(tile_key.level, tile_key.x, tile_key.y, &ctx) {
                Ok(tile) => {
                    if let Some(ref disk) = disk_cache {
                        let _ =
                            disk.put_qoi_rgba8(&disk_key, tile.width(), tile.height(), tile.data());
                    }
                    let _ = done_tx.send((tile_key, tile));
                    guard.mark_arrived();
                }
                Err(err) if err.is_cancelled() => {
                    guard.mark_cancelled();
                }
                Err(err) => {
                    tracing::warn!(
                        source = ?tile_key.source,
                        level = tile_key.level,
                        x = tile_key.x,
                        y = tile_key.y,
                        error = %err,
                        "Pointwise overlay tile decode failed"
                    );
                    let terminal = matches!(
                        err,
                        veldmap_raster::RasterError::UnsupportedFormat
                            | veldmap_raster::RasterError::InvalidDimensions { .. }
                            | veldmap_raster::RasterError::LimitExceeded { .. }
                            | veldmap_raster::RasterError::TileOutOfRange { .. }
                    );
                    guard.mark_failed(terminal);
                }
            }
        }),
    };
    let _ = fast_pool.submit(job);
}

#[expect(
    clippy::too_many_arguments,
    reason = "submits self-contained sequential pass closure to bulk lane"
)]
fn submit_pass_job(
    pass_key: PassKey,
    pass_id: u64,
    pass_cancel: CancelToken,
    missing: Vec<TileKey>,
    layer_wanted: HashSet<TileKey>,
    source: &Arc<dyn RasterSource>,
    meta: &PyramidMeta,
    fingerprint_hex: &str,
    scheduler: &Arc<Mutex<TileScheduler>>,
    chunk_cache: &Arc<SharedChunkCache>,
    budget_tracker: &Arc<DecodeBudgetTracker>,
    budget: MemoryBudget,
    disk_cache: Option<Arc<TileDiskCache>>,
    bulk_pool: &DecodePoolHandle,
    done_tx: &UploadSender,
    wakeup: &Wakeup,
) {
    let source = Arc::clone(source);
    let meta = meta.clone();
    let fp = fingerprint_hex.to_owned();
    let sched = Arc::clone(scheduler);
    let chunk_cache = Arc::clone(chunk_cache);
    let budget_tracker = Arc::clone(budget_tracker);
    let done_tx = done_tx.clone();
    let wakeup_cb = Arc::clone(wakeup);
    let queued_bytes = source
        .pass_peak_bytes()
        .min(budget.bulk_lane_pending_bytes as u64) as usize;
    let guard = PassFlightGuard::new(Arc::clone(&sched), pass_key, pass_id, Arc::clone(wakeup));

    let job = DecodeJob {
        key: DecodeJobKey::Pass(pass_key),
        queued_bytes: queued_bytes.max(DEFAULT_DISPLAY_TILE_BYTES),
        cancel: pass_cancel.clone(),
        work: Box::new(move || {
            // Satisfy from QOI disk cache first if every missing tile is
            // already cached on disk (avoiding a full sequential file scan).
            if let Some(ref disk) = disk_cache {
                let mut all_hits = true;
                let mut cached_tiles = Vec::with_capacity(missing.len());
                for &tk in &missing {
                    if pass_cancel.is_cancelled() {
                        guard.abort();
                        return;
                    }
                    let dk = DiskTileKey::from_tile_key(fp.clone(), &tk);
                    if let Ok(Some((w, h, rgba))) = disk.get_qoi_rgba8(&dk)
                        && let Ok(tile) =
                            DecodedTile::try_new(w, h, PixelFormat::Rgba8UnormSrgb, rgba)
                    {
                        cached_tiles.push((tk, tile));
                    } else {
                        all_hits = false;
                        break;
                    }
                }
                if all_hits {
                    for (tk, tile) in cached_tiles {
                        let _ = done_tx.send((tk, tile));
                        lock_unpoisoned(&sched).on_tile_arrived(&tk);
                    }
                    guard.finish(true);
                    return;
                }
            }

            let mut upload_set = layer_wanted;
            upload_set.extend(missing);
            let max_level = meta.max_level();
            let ctx = DecodeContext {
                chunk_cache: &chunk_cache,
                budget_tracker: &budget_tracker,
                budget: &budget,
                cancel: &pass_cancel,
            };

            let result = source.produce_pass(&ctx, &mut |level, tx, ty, tw, th, rgba| {
                let tk = meta.tile_key(level, tx, ty);
                if let Some(ref disk) = disk_cache {
                    let dk = DiskTileKey::from_tile_key(fp.clone(), &tk);
                    let _ = disk.put_qoi_rgba8(&dk, tw, th, &rgba);
                }
                if upload_set.contains(&tk) || level == max_level {
                    let tile = DecodedTile::try_new(tw, th, PixelFormat::Rgba8UnormSrgb, rgba)
                        .map_err(|e| veldmap_raster::RasterError::Decode(e.to_string()))?;
                    let _ = done_tx.send((tk, tile));
                    lock_unpoisoned(&sched).on_tile_arrived(&tk);
                    wakeup_cb();
                }
                Ok(())
            });

            match result {
                Ok(()) => {
                    guard.finish(true);
                }
                Err(err) if err.is_cancelled() => {
                    guard.abort();
                }
                Err(err) => {
                    tracing::warn!(
                        source = ?meta.source,
                        error = %err,
                        "Sequential overlay pass failed"
                    );
                    guard.finish(false);
                }
            }
        }),
    };
    let _ = bulk_pool.submit(job);
}
