//! High-level [`OverlayManager`] facade coordinating [`LayerStack`],
//! [`OverlayRuntime`], [`CursorProber`], and the overlay I/O worker.
//!
//! Returns [`OverlayMutation`] records rather than mutating `GpuTileCache` /
//! `PendingUploads` directly, matching `BasemapManager`'s clean boundary with
//! `SceneState`.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use veldmap_core::budget::MemoryBudget;
use veldmap_core::geo::{Camera, Frame};
use veldmap_core::sync::{CancelToken, Wakeup};
use veldmap_core::tile::{SourceId, TileKey};
use veldmap_raster::{DecodePoolHandle, RasterSource};
use veldmap_storage::TileDiskCache;

use crate::clamp_opacity;
use crate::gpu_tiles::{GpuTileCache, PendingUploads, UploadBatchReport, UploadSender};
use crate::render::DrawOverlayLayer;
use crate::worker::SingleThreadWorker;

use super::cursor::CursorProber;
use super::frame::{LayerFrameCtx, build_and_schedule_layer};
use super::io_worker::{
    OverlayIoCmd, OverlayIoResult, ReloadRequest, ReloadedPayload, spawn_overlay_io_worker,
};
use super::opener::{OpenRasterError, bind_frame_with_override};
use super::runtime::OverlayRuntime;
use super::stack::{LayerStack, OverlayLayer};
use super::{
    CrsOverride, CursorReadout, LayerId, LayerInitOptions, LayerSummary, OverlayIoEvent,
    OverlayPersistEntry,
};

fn alloc_seq(counter: &mut u64) -> u64 {
    let value = *counter;
    *counter = counter.saturating_add(1);
    value
}

/// Per-frame camera, viewport, cache, and appetite inputs for
/// [`OverlayManager::build_and_schedule`].
pub(crate) struct OverlayFrameCtx<'a> {
    pub(crate) camera: &'a Camera,
    pub(crate) width_px: u32,
    pub(crate) height_px: u32,
    pub(crate) gpu_cache: &'a mut GpuTileCache,
    pub(crate) pending: &'a PendingUploads,
    pub(crate) total_appetite_weight: u64,
    pub(crate) carrier_keys: &'a mut HashSet<TileKey>,
}

/// Cache-purge side effects returned by [`OverlayManager`] mutations for
/// `SceneState` to apply cleanly to `GpuTileCache` and `PendingUploads`.
#[derive(Clone, Debug, Default)]
pub(crate) struct OverlayMutation {
    pub(crate) changed: bool,
    pub(crate) topology_changed: bool,
    pub(crate) purge_sources: Vec<SourceId>,
}

/// Coordinates local raster overlays: non-blocking open/reload via the overlay
/// I/O worker, `StreamKey`-scoped two-lane scheduling, lossless QOI disk
/// caching, and zoom-invariant CPU cursor readouts.
pub(crate) struct OverlayManager {
    stack: LayerStack,
    next_layer_id: u64,
    next_relaid: u64,
    runtime: Arc<OverlayRuntime>,
    cursor: CursorProber,
    io_worker: SingleThreadWorker<OverlayIoCmd, OverlayIoResult>,
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
        let runtime = Arc::new(OverlayRuntime::new(
            budget, disk_cache, fast_pool, bulk_pool, done_tx, wakeup, shutdown,
        ));
        let io_worker = spawn_overlay_io_worker(Arc::clone(&runtime));
        Self {
            stack: LayerStack::default(),
            next_layer_id: 1,
            next_relaid: 1,
            runtime,
            cursor: CursorProber::new(),
            io_worker,
        }
    }

    pub(crate) fn extend_pinned_root_sources(&self, out: &mut Vec<SourceId>) {
        self.stack.extend_pinned_root_sources(out);
    }

    pub(crate) fn extend_wanted_tile_sets<'a>(&'a self, out: &mut Vec<&'a HashSet<TileKey>>) {
        self.stack.extend_wanted_tile_sets(out);
    }

    pub(crate) fn total_appetite_weight_bytes(&self) -> u64 {
        self.stack.total_appetite_weight_bytes()
    }

    pub(crate) fn topmost_visible_source_pixel_size_m(&self) -> Option<(f64, f64)> {
        self.stack.topmost_visible_source_pixel_size_m()
    }

    fn purge_source(&self, src: SourceId) {
        self.runtime.purge_source_runtime(src);
        self.cursor.invalidate_source(src);
    }

    /// Return immutable UI summaries of all loaded overlay layers.
    pub(crate) fn layer_summaries(&self) -> Vec<LayerSummary> {
        let layers = self.stack.layers();
        let stats = self.runtime.layer_flight_stats(
            layers
                .iter()
                .map(|l| (l.source_id(), l.meta().stream_key())),
        );
        let len = layers.len();
        layers
            .iter()
            .zip(stats)
            .enumerate()
            .map(|(idx, (layer, (in_flight, pass_active)))| {
                layer.to_summary(in_flight, pass_active, idx + 1 < len, idx > 0)
            })
            .collect()
    }

    /// Number of loaded raster layers (O(1), lock-free).
    pub(crate) fn layer_count(&self) -> usize {
        self.stack.layers().len()
    }

    /// Look up the current CRS override for `id`, if any.
    pub(crate) fn layer_crs_override(&self, id: LayerId) -> Option<CrsOverride> {
        self.stack.get(id).and_then(|l| l.crs_override().cloned())
    }

    /// Borrowed view of loaded raster layers in painter's order for session
    /// persistence.
    pub(crate) fn saved_entries_iter(&self) -> impl Iterator<Item = OverlayPersistEntry<'_>> {
        self.stack.layers().iter().map(OverlayLayer::persist_entry)
    }

    /// Queue a non-blocking local raster open onto the overlay I/O worker.
    #[cfg(test)]
    fn begin_open_file(&self, path: &Path, fly_to: bool) -> Result<(), String> {
        self.begin_open_file_with_options(path, fly_to, true, 1.0, None)
    }

    /// Queue a non-blocking local raster open with explicit initial visibility,
    /// opacity, and optional CRS override (used by session restore and CRS
    /// picker re-open flows).
    pub(crate) fn begin_open_file_with_options(
        &self,
        path: &Path,
        fly_to: bool,
        visible: bool,
        opacity: f32,
        crs_override: Option<CrsOverride>,
    ) -> Result<(), String> {
        if self.io_worker.send(OverlayIoCmd::Open {
            path: path.to_path_buf(),
            fly_to,
            options: LayerInitOptions {
                visible,
                opacity: clamp_opacity(opacity),
                crs_override,
            },
        }) {
            Ok(())
        } else {
            Err("overlay I/O worker is unavailable".to_owned())
        }
    }

    /// Queue a non-blocking reload of `id` onto the overlay I/O worker,
    /// cancelling active tile jobs for `old_src` immediately so stale tiles
    /// cannot race the reload, while preserving the layer if reopen fails.
    pub(crate) fn begin_reload_bypass_cache(
        &mut self,
        id: LayerId,
    ) -> Result<OverlayMutation, String> {
        let cancel = self.runtime.shutdown.child_token();
        let (request, prep) = {
            let Some(layer) = self.stack.get_mut(id) else {
                return Err("layer no longer exists".to_owned());
            };
            let prep = layer.begin_reload(cancel.clone());
            let request = ReloadRequest {
                id,
                reload_seq: prep.reload_seq,
                crs_override_epoch: layer.crs_override_epoch(),
                path: layer.path().to_path_buf(),
                old_fingerprint_hex: layer.fingerprint_hex().to_owned(),
                crs_override: layer.crs_override().cloned(),
                cancel,
            };
            (request, prep)
        };
        if let Some(prev) = prep.old_cancel {
            prev.cancel();
        }

        // Cancel in-flight jobs and clear runtime caches for `old_source` immediately
        // so no worker can write stale pixels while reload runs.
        self.purge_source(prep.old_source);

        if !self.io_worker.send(OverlayIoCmd::Reload(request)) {
            if let Some(layer) = self.stack.get_mut(id) {
                layer.abort_reload();
            }
            self.runtime.request_reschedule();
            return Err("overlay I/O worker is unavailable".to_owned());
        }

        Ok(OverlayMutation {
            changed: true,
            topology_changed: true,
            purge_sources: vec![prep.old_source],
        })
    }

    /// Drain completed background raster open/reload operations from the
    /// overlay I/O worker.
    pub(crate) fn poll_io_events(&mut self) -> (Vec<OverlayIoEvent>, OverlayMutation) {
        let mut events = Vec::new();
        let mut mutation = OverlayMutation::default();

        while let Some(res) = self.io_worker.try_recv() {
            match res {
                OverlayIoResult::Opened {
                    path,
                    fly_to,
                    options,
                    outcome,
                } => self.handle_opened_result(
                    path,
                    fly_to,
                    options,
                    outcome,
                    &mut events,
                    &mut mutation,
                ),
                OverlayIoResult::Reloaded(payload) => {
                    self.handle_reloaded_result(payload, &mut events, &mut mutation);
                }
            }
        }

        (events, mutation)
    }

    fn handle_opened_result(
        &mut self,
        path: PathBuf,
        fly_to: bool,
        options: LayerInitOptions,
        outcome: Result<(Arc<dyn RasterSource>, Frame), OpenRasterError>,
        events: &mut Vec<OverlayIoEvent>,
        mutation: &mut OverlayMutation,
    ) {
        match outcome {
            Ok((source, frame)) => {
                let id = LayerId(alloc_seq(&mut self.next_layer_id));
                let relaid = alloc_seq(&mut self.next_relaid);
                let event_path = path.clone();
                let layer = OverlayLayer::from_source(id, path, source, frame, options, relaid);
                let bounds = layer.footprint();
                self.stack.push(layer);
                mutation.changed = true;
                mutation.topology_changed = true;
                events.push(OverlayIoEvent::Opened {
                    id,
                    path: event_path,
                    bounds,
                    fly_to,
                });
            }
            Err(err) => {
                events.push(OverlayIoEvent::Failed {
                    path,
                    error: err.message,
                    can_assign_crs: err.can_assign_crs,
                });
            }
        }
    }

    fn handle_reloaded_result(
        &mut self,
        reloaded: ReloadedPayload,
        events: &mut Vec<OverlayIoEvent>,
        mutation: &mut OverlayMutation,
    ) {
        let ReloadedPayload {
            id,
            reload_seq,
            crs_override_epoch,
            path,
            outcome,
        } = reloaded;
        let (old_src, new_src) = {
            let Some(slot) = self.stack.get_mut(id) else {
                return;
            };
            if !slot.matches_reload_seq(reload_seq) {
                // Layer was superseded by a newer reload trigger.
                return;
            }
            match outcome {
                Ok((source, mut frame)) => {
                    // If `override_layer_crs` ran while this reload was in flight,
                    // re-bind the reloaded source against the live `crs_override`
                    // so it never silently reverts.
                    if slot.crs_override_epoch() != crs_override_epoch {
                        match bind_frame_with_override(source.description(), slot.crs_override()) {
                            Ok(rebound) => frame = rebound,
                            Err(error) => {
                                slot.fail_reload(error.clone());
                                mutation.changed = true;
                                mutation.topology_changed = true;
                                events.push(OverlayIoEvent::Failed {
                                    path,
                                    error,
                                    can_assign_crs: true,
                                });
                                return;
                            }
                        }
                    }
                    let new_src = source.source_id();
                    let relaid = alloc_seq(&mut self.next_relaid);
                    let old_src = slot.replace_reloaded(path, source, frame, relaid);
                    (old_src, new_src)
                }
                Err(err) => {
                    slot.fail_reload(err.message.clone());
                    // Reload failed: leave the existing layer intact and
                    // resume normal rendering on the next frame.
                    mutation.changed = true;
                    mutation.topology_changed = true;
                    events.push(OverlayIoEvent::Failed {
                        path,
                        error: err.message,
                        can_assign_crs: err.can_assign_crs,
                    });
                    return;
                }
            }
        };

        // Purge runtime/GPU state for both `old_src` and `new_src`
        // (in case a sibling layer decoded tiles during reload).
        self.purge_source(old_src);
        mutation.purge_sources.push(old_src);
        if new_src != old_src {
            self.purge_source(new_src);
            mutation.purge_sources.push(new_src);
        }
        mutation.changed = true;
        mutation.topology_changed = true;
        events.push(OverlayIoEvent::Reloaded { id });
    }

    /// Move layer toward the top of painter's order.
    pub(crate) fn move_layer_up(&mut self, id: LayerId) -> bool {
        self.stack.move_up(id)
    }

    /// Move layer toward the bottom of painter's order.
    pub(crate) fn move_layer_down(&mut self, id: LayerId) -> bool {
        self.stack.move_down(id)
    }

    /// Rebind an existing layer's affine placement to `crs_override` (or reset
    /// to its embedded/sidecar CRS when `None`), bumping `relaid` to rebuild
    /// warp meshes without re-decoding image tiles.
    pub(crate) fn override_layer_crs(
        &mut self,
        id: LayerId,
        crs_override: Option<CrsOverride>,
    ) -> Result<Option<veldmap_core::geo::FootprintBounds>, String> {
        let Some(layer) = self.stack.get_mut(id) else {
            return Err("layer no longer exists".to_owned());
        };
        let frame = bind_frame_with_override(layer.source().description(), crs_override.as_ref())?;
        let relaid = alloc_seq(&mut self.next_relaid);
        layer.apply_rebound_frame(frame, crs_override, relaid);
        Ok(layer.footprint())
    }

    /// Remove a layer from [`LayerStack`], cancelling active jobs and returning
    /// its `SourceId` for GPU/pending purge unless another layer still shares
    /// that exact content fingerprint.
    pub(crate) fn remove_layer(&mut self, id: LayerId) -> OverlayMutation {
        let Some(mut removed) = self.stack.remove(id) else {
            return OverlayMutation::default();
        };
        if let Some(tok) = removed.take_reload_cancel() {
            tok.cancel();
        }
        let src = removed.source_id();
        let mut mutation = OverlayMutation {
            changed: true,
            topology_changed: true,
            purge_sources: Vec::new(),
        };
        if self.stack.count_sharing_source(src) == 0 {
            self.purge_source(src);
            mutation.purge_sources.push(src);
        } else if !self.stack.has_visible_source(src) {
            self.purge_source(src);
        }
        mutation
    }

    pub(crate) fn set_visible(&mut self, id: LayerId, visible: bool) -> OverlayMutation {
        let src = {
            let Some(layer) = self.stack.get_mut(id) else {
                return OverlayMutation::default();
            };
            if !layer.set_visible(visible) {
                return OverlayMutation::default();
            }
            layer.source_id()
        };
        if !visible && !self.stack.has_visible_source(src) {
            self.purge_source(src);
        }
        OverlayMutation {
            changed: true,
            topology_changed: true,
            purge_sources: Vec::new(),
        }
    }

    pub(crate) fn set_opacity(&mut self, id: LayerId, opacity: f32) -> bool {
        let Some(layer) = self.stack.get_mut(id) else {
            return false;
        };
        layer.set_opacity(opacity)
    }

    pub(crate) fn record_upload_report(&mut self, report: &UploadBatchReport) {
        for &(src, max_bytes) in &report.max_tile_bytes_by_source {
            for layer in self.stack.layers_mut() {
                if layer.source_id() == src {
                    layer.observe_tile_bytes(max_bytes);
                }
            }
        }
    }

    /// Return `true` if a background overlay event (eviction, retryable/terminal
    /// failure, pass finish, or elapsed backoff) requires re-running scheduling.
    pub(crate) fn take_reschedule_needed(&self) -> bool {
        self.runtime.take_reschedule_needed()
    }

    /// Optional backoff duration if a decode lane was recently saturated.
    pub(crate) fn retry_delay(&self) -> Option<Duration> {
        self.runtime.retry_delay()
    }

    /// Reconcile in-flight pointwise overlay jobs against current GPU/pending
    /// cache state (called when `SceneState::reconcile_providers` runs outside a
    /// full snapshot rebuild).
    pub(crate) fn reconcile(&self, gpu_cache: &GpuTileCache, pending: &PendingUploads) {
        self.runtime.reconcile_pointwise_streams(
            self.stack
                .layers()
                .iter()
                .filter(|l| l.is_active())
                .map(|l| (l.meta().stream_key(), l.wanted())),
            gpu_cache,
            pending,
        );
    }

    /// Build immutable [`DrawOverlayLayer`]s for the current frame, update
    /// per-layer wanted sets across all ladder rungs, and schedule missing
    /// tiles onto the two-lane decode pool.
    pub(crate) fn build_and_schedule(
        &mut self,
        ctx: &mut OverlayFrameCtx<'_>,
    ) -> Vec<DrawOverlayLayer> {
        let aspect = f64::from(ctx.width_px.max(1)) / f64::from(ctx.height_px.max(1));
        let frustum = ctx.camera.frustum(aspect);
        let viewport_mpp = ctx.camera.metres_per_pixel(f64::from(ctx.height_px.max(1)));
        let mut layer_ctx = LayerFrameCtx {
            runtime: &self.runtime,
            frustum: &frustum,
            viewport_mpp,
            gpu_cache: ctx.gpu_cache,
            pending: ctx.pending,
            total_appetite_weight: ctx.total_appetite_weight,
            carrier_keys: ctx.carrier_keys,
        };

        let mut draw_layers = Vec::new();
        for layer in self.stack.layers_mut() {
            if let Some(draw_layer) = build_and_schedule_layer(layer, &mut layer_ctx) {
                draw_layers.push(draw_layer);
            }
        }
        draw_layers
    }

    pub(crate) fn cancel_cursor_readout(&self) {
        self.cursor.cancel();
    }

    pub(crate) fn query_cursor_readout(&self, lat_deg: f64, lon_deg: f64) -> Option<CursorReadout> {
        self.cursor
            .query(&self.stack, &self.runtime, lat_deg, lon_deg)
    }

    /// Phase 1 shutdown: cancel active overlay passes, pointwise jobs, reloads,
    /// cursor probes, and close the I/O worker channel.
    pub(crate) fn begin_shutdown(&mut self) {
        self.runtime
            .cancel_sources(self.stack.layers().iter().map(OverlayLayer::source_id));
        for layer in self.stack.layers_mut() {
            if let Some(tok) = layer.take_reload_cancel() {
                tok.cancel();
            }
        }
        self.cursor.cancel();
        self.io_worker.close_channel();
    }

    /// Phase 3 shutdown: deterministically join the overlay I/O worker.
    pub(crate) fn finish_shutdown(&mut self) {
        self.io_worker.join();
    }
}

impl Drop for OverlayManager {
    fn drop(&mut self) {
        self.begin_shutdown();
        self.finish_shutdown();
    }
}

#[cfg(test)]
#[path = "manager_tests.rs"]
mod tests;
