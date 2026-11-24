//! High-level [`OverlayManager`] facade coordinating [`LayerStack`],
//! [`OverlayRuntime`], [`CursorProber`], and [`OverlayIoWorker`].
//!
//! Returns [`OverlayMutation`] records rather than mutating `GpuTileCache` /
//! `PendingUploads` directly, matching `BasemapManager`'s clean boundary with
//! `SceneState`.

use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use veldmap_core::budget::MemoryBudget;
use veldmap_core::geo::Camera;
use veldmap_core::sync::{CancelToken, Wakeup, lock_unpoisoned};
use veldmap_core::tile::{SourceId, TileKey};
use veldmap_raster::DecodePoolHandle;
use veldmap_storage::TileDiskCache;

use crate::gpu_tiles::{GpuTileCache, PendingUploads, UploadBatchReport, UploadSender};
use crate::render::DrawOverlayLayer;

use super::cursor::CursorProber;
use super::frame::{LayerFrameCtx, build_and_schedule_layer};
use super::io_worker::{OverlayIoCmd, OverlayIoResult, OverlayIoWorker};
use super::runtime::OverlayRuntime;
use super::stack::{LayerStack, OverlayLayer};
use super::{CursorReadout, LayerId, LayerSummary, OverlayIoEvent};

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
    pub(crate) purge_sources: Vec<SourceId>,
}

/// Coordinates local raster overlays: non-blocking open/reload via
/// [`OverlayIoWorker`], `PassKey`-scoped two-lane scheduling, lossless QOI
/// disk caching, and zoom-invariant CPU cursor readouts.
pub(crate) struct OverlayManager {
    stack: LayerStack,
    next_layer_id: u64,
    next_relaid: u64,
    runtime: Arc<OverlayRuntime>,
    cursor: CursorProber,
    io_worker: OverlayIoWorker,
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
        let io_worker = OverlayIoWorker::spawn(Arc::clone(&runtime));
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

    fn alloc_relaid(&mut self) -> u64 {
        let r = self.next_relaid;
        self.next_relaid = self.next_relaid.saturating_add(1);
        r
    }

    fn purge_source(&self, src: SourceId) {
        self.runtime.purge_source_runtime(src);
        self.cursor.invalidate_source(src);
    }

    /// Return immutable UI summaries of all loaded overlay layers.
    pub(crate) fn layer_summaries(&self) -> Vec<LayerSummary> {
        let sched = lock_unpoisoned(&self.runtime.scheduler);
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
                    reloading: layer.is_reloading(),
                }
            })
            .collect()
    }

    /// Queue a non-blocking local GeoTIFF/BigTIFF/COG open onto
    /// [`OverlayIoWorker`]. Completion arrives via [`Self::poll_io_events`].
    pub(crate) fn begin_open_file(&self, path: &Path, fly_to: bool) -> Result<(), String> {
        let cancel = self.runtime.shutdown.child_token();
        if self.io_worker.send(OverlayIoCmd::Open {
            path: path.to_path_buf(),
            fly_to,
            cancel,
        }) {
            Ok(())
        } else {
            Err("overlay I/O worker is unavailable".to_owned())
        }
    }

    /// Queue a non-blocking reload of `id` onto [`OverlayIoWorker`], cancelling
    /// active tile jobs for `old_src` immediately so stale tiles cannot race
    /// the reload, while preserving the layer if reopen fails.
    pub(crate) fn begin_reload_bypass_cache(
        &mut self,
        id: LayerId,
    ) -> Result<OverlayMutation, String> {
        let (path, old_src, old_fp, reload_seq, reload_cancel, prev_cancel) = {
            let Some(layer) = self.stack.get_mut(id) else {
                return Err("layer no longer exists".to_owned());
            };
            let cancel = self.runtime.shutdown.child_token();
            let (reload_seq, prev) = layer.begin_reload(cancel.clone());
            (
                layer.path.clone(),
                layer.source_id(),
                layer.fingerprint_hex.clone(),
                reload_seq,
                cancel,
                prev,
            )
        };
        if let Some(prev) = prev_cancel {
            prev.cancel();
        }

        // Cancel in-flight jobs and clear runtime caches for `old_src` immediately
        // so no worker can write stale pixels while reload runs.
        self.purge_source(old_src);

        if !self.io_worker.send(OverlayIoCmd::Reload {
            id,
            reload_seq,
            path,
            old_fingerprint_hex: old_fp,
            cancel: reload_cancel,
        }) {
            if let Some(layer) = self.stack.get_mut(id) {
                layer.abort_reload();
            }
            self.runtime.request_reschedule();
            return Err("overlay I/O worker is unavailable".to_owned());
        }

        Ok(OverlayMutation {
            changed: true,
            purge_sources: vec![old_src],
        })
    }

    /// Drain completed background open/reload operations from
    /// [`OverlayIoWorker`].
    pub(crate) fn poll_io_events(&mut self) -> (Vec<OverlayIoEvent>, OverlayMutation) {
        let mut events = Vec::new();
        let mut mutation = OverlayMutation::default();

        while let Some(res) = self.io_worker.try_recv() {
            match res {
                OverlayIoResult::Opened {
                    path,
                    fly_to,
                    outcome,
                } => match outcome {
                    Ok((source, frame)) => {
                        let id = LayerId(self.next_layer_id);
                        self.next_layer_id = self.next_layer_id.saturating_add(1);
                        let relaid = self.alloc_relaid();
                        let layer =
                            OverlayLayer::from_source(id, path, source, frame, true, 1.0, relaid);
                        let bounds = layer.footprint;
                        self.stack.push(layer);
                        mutation.changed = true;
                        events.push(OverlayIoEvent::Opened { id, bounds, fly_to });
                    }
                    Err(error) => {
                        events.push(OverlayIoEvent::Failed { path, error });
                    }
                },
                OverlayIoResult::Reloaded {
                    id,
                    reload_seq,
                    path,
                    outcome,
                } => {
                    if !self
                        .stack
                        .get(id)
                        .is_some_and(|slot| slot.matches_reload_seq(reload_seq))
                    {
                        // Layer was removed or superseded by a newer reload trigger.
                        continue;
                    }
                    match outcome {
                        Ok((source, frame)) => {
                            let new_src = source.source_id();
                            let relaid = self.alloc_relaid();
                            let Some(old_src) = self.stack.get_mut(id).and_then(|slot| {
                                slot.replace_reloaded(reload_seq, path, source, frame, relaid)
                            }) else {
                                continue;
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
                            events.push(OverlayIoEvent::Reloaded { id });
                        }
                        Err(error) => {
                            if let Some(slot) = self.stack.get_mut(id)
                                && slot.finish_reload_err(reload_seq)
                            {
                                // Reload failed: leave the existing layer intact and
                                // resume normal rendering on the next frame.
                                mutation.changed = true;
                                events.push(OverlayIoEvent::Failed { path, error });
                            }
                        }
                    }
                }
            }
        }

        (events, mutation)
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

    pub(crate) fn set_visible(&mut self, id: LayerId, visible: bool) -> bool {
        let src = {
            let Some(layer) = self.stack.get_mut(id) else {
                return false;
            };
            if layer.visible == visible {
                return false;
            }
            layer.visible = visible;
            if !visible {
                layer.clear_wanted();
            }
            layer.source_id()
        };
        if !visible && !self.stack.has_visible_source(src) {
            self.purge_source(src);
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
        self.runtime
            .reconcile_pointwise_layers(self.stack.layers(), gpu_cache, pending);
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

    /// Phase 3 shutdown: deterministically join [`OverlayIoWorker`].
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
