//! High-level [`OverlayManager`] facade coordinating `LayerStack`,
//! `OverlayRuntime`, `CursorProber`, `BandStatsTracker`, isolated remote I/O
//! (`veldmap-remote-io`), offloaded tile persistence (`veldmap-tile-write`),
//! and the local overlay I/O worker.
//!
//! Asynchronous open, reload, and completion reconciliation methods live in the
//! private [`lifecycle`] submodule. Returns [`OverlayMutation`] records rather
//! than mutating `GpuTileCache` / `PendingUploads` directly, matching
//! `BasemapManager`'s clean boundary with `SceneState`.

mod lifecycle;

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use veldmap_core::budget::MemoryBudget;
use veldmap_core::geo::{Camera, FootprintBounds};
use veldmap_core::sync::{CancelToken, Wakeup};
use veldmap_core::tile::{SourceId, StreamKey, TileKey, TileStreamSpec};
use veldmap_net::NetContext;
use veldmap_raster::{DecodePool, DecodePoolConfig, DecodePoolHandle};
use veldmap_storage::TileDiskCache;

use crate::colormap::{CategoricalPaletteId, ColormapId};
use crate::gpu_tiles::{GpuTileCache, PendingUploads, UploadBatchReport, UploadSender};
use crate::render::{DrawOverlayLayer, ScratchDemand};
use crate::worker::SingleThreadWorker;

use super::cursor::CursorProber;
use super::disk_cache::TileWriteWorker;
use super::frame::{LayerFrameCtx, PrepareFrameCtx, finish_scheduled_layer, prepare_layer_wanted};
use super::io_worker::{OverlayIoCmd, OverlayIoResult, spawn_overlay_io_worker};
use super::jobs::band_stats::BandStatsTracker;
use super::opener::bind_frame_with_override;
use super::remote_opener::RemoteOpenTracker;
use super::runtime::{OverlayRuntime, OverlayRuntimeDeps};
use super::stack::{LayerStack, OverlayLayer};
use super::{
    ColorbarInfo, CrsOverride, CursorReadout, LayerId, LayerSummary, OverlayPersistEntry,
    StretchPreset,
};

fn alloc_seq(counter: &mut u64) -> u64 {
    let value = *counter;
    *counter = counter.saturating_add(1);
    value
}

/// Shared runtime subsystems injected into [`OverlayManager::new`].
pub(crate) struct OverlayManagerDeps {
    pub(crate) budget: MemoryBudget,
    pub(crate) disk_cache: Option<Arc<TileDiskCache>>,
    pub(crate) net: Option<Arc<NetContext>>,
    pub(crate) fast_pool: DecodePoolHandle,
    pub(crate) bulk_pool: DecodePoolHandle,
    pub(crate) done_tx: UploadSender,
    pub(crate) wakeup: Wakeup,
    pub(crate) shutdown: CancelToken,
}

/// Typed infallible edit operation applied to a single raster or band-math
/// overlay layer via [`OverlayManager::edit_layer`] / `SceneState::edit_overlay`.
#[derive(Clone, Copy, Debug)]
pub(crate) enum OverlayEdit {
    SetVisible(bool),
    SetOpacity(f32),
    SetStretchPreset(StretchPreset),
    SetStretchRange {
        min: f64,
        max: f64,
    },
    SetColormap {
        colormap: ColormapId,
        invert: bool,
    },
    SetCategoricalPalette(CategoricalPaletteId),
    SetBandMathRgbStretch {
        min: [f64; 3],
        max: [f64; 3],
        gamma: f32,
    },
    ResetBandMathStretch,
    SetBandMathSclEnabled(bool),
    SetBandMathSclMaskBits(u32),
    MoveUp,
    MoveDown,
    Remove,
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

/// Coordinates local and remote COG raster overlays: non-blocking local open/reload
/// via the overlay I/O worker, isolated remote HTTP Range open/prefetch via
/// `NetContext` + `veldmap-remote-io`, `StreamKey`-scoped two-lane scheduling,
/// offloaded lossless QOI & `Lz4Raw` disk caching, and zoom-invariant CPU cursor readouts.
pub(crate) struct OverlayManager {
    stack: LayerStack,
    next_layer_id: u64,
    next_relaid: u64,
    workspace_epoch: u64,
    workspace_cancel: CancelToken,
    runtime: Arc<OverlayRuntime>,
    remote_pool: Option<DecodePool>,
    tile_writer: Option<TileWriteWorker>,
    remote_opener: RemoteOpenTracker,
    cursor: CursorProber,
    io_worker: SingleThreadWorker<OverlayIoCmd, OverlayIoResult>,
    band_stats: BandStatsTracker,
}

impl OverlayManager {
    pub(crate) fn new(deps: OverlayManagerDeps) -> Self {
        let OverlayManagerDeps {
            budget,
            disk_cache,
            net,
            fast_pool,
            bulk_pool,
            done_tx,
            wakeup,
            shutdown,
        } = deps;
        let remote_pool = DecodePool::spawn_named(
            "veldmap-remote-io",
            DecodePoolConfig {
                worker_count: 2,
                max_pending_jobs: 64,
                max_pending_bytes: 32 * 1024 * 1024,
            },
            shutdown.clone(),
        );
        let (tile_writer, tile_write_handle) = match &disk_cache {
            Some(disk) => {
                let (worker, handle) = TileWriteWorker::spawn(Arc::clone(disk), shutdown.clone());
                (Some(worker), Some(handle))
            }
            None => (None, None),
        };
        let runtime = Arc::new(OverlayRuntime::new(OverlayRuntimeDeps {
            budget,
            disk_cache,
            tile_writer: tile_write_handle,
            net,
            fast_pool,
            bulk_pool,
            remote_pool: remote_pool.handle(),
            done_tx,
            wakeup,
            shutdown,
        }));
        let workspace_cancel = runtime.shutdown.child_token();
        let io_worker = spawn_overlay_io_worker(Arc::clone(&runtime));
        Self {
            stack: LayerStack::default(),
            next_layer_id: 1,
            next_relaid: 1,
            workspace_epoch: 1,
            workspace_cancel,
            runtime,
            remote_pool: Some(remote_pool),
            tile_writer,
            remote_opener: RemoteOpenTracker::new(),
            cursor: CursorProber::new(),
            io_worker,
            band_stats: BandStatsTracker::new(),
        }
    }

    pub(crate) fn extend_pinned_root_streams(&self, out: &mut Vec<StreamKey>) {
        self.stack.extend_pinned_root_streams(out);
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

    pub(crate) fn topmost_colorbar_info(&self) -> Option<ColorbarInfo> {
        self.stack.topmost_colorbar_info()
    }

    /// Return the combined viewport [`ScratchDemand`] across all currently
    /// active layers.
    pub(crate) fn max_scratch_requirements(&self) -> ScratchDemand {
        self.stack.max_scratch_requirements()
    }

    fn request_band_stats_if_needed(&mut self, id: LayerId) {
        if let Some(layer) = self.stack.get_mut(id) {
            self.band_stats.request_if_needed(layer, &self.runtime);
        }
    }

    fn purge_source(&self, src: SourceId) {
        self.runtime.purge_source_runtime(src);
        self.cursor.invalidate_source(src);
    }

    /// Release runtime/GPU caches and cancel orphaned streams for `sources` and
    /// `stream_keys` after a layer is removed, hidden, or reloaded.
    ///
    /// - When `gpu_purge_max_refs` is `Some(max_refs)` and
    ///   `count_sharing_source(src) <= max_refs`, purges runtime state and
    ///   appends `src` to `purge_out` so `SceneState` clears VRAM/pending uploads.
    /// - Otherwise, when no active visible layer references `src`, purges
    ///   runtime state only (keeping GPU tiles warm across hide/show toggles).
    /// - For every stream whose source was not purged, cancels its job if no
    ///   other active layer uses that `StreamKey`.
    fn release_sources_and_streams(
        &self,
        sources: &[SourceId],
        stream_keys: &[StreamKey],
        gpu_purge_max_refs: Option<usize>,
        purge_out: &mut Vec<SourceId>,
    ) {
        let mut purged_sources = Vec::new();
        for &src in sources {
            if gpu_purge_max_refs
                .is_some_and(|max_refs| self.stack.count_sharing_source(src) <= max_refs)
            {
                self.purge_source(src);
                purge_out.push(src);
                purged_sources.push(src);
            } else if !self.stack.has_visible_source(src) {
                self.purge_source(src);
                purged_sources.push(src);
            }
        }
        for &sk in stream_keys {
            if !purged_sources.contains(&sk.source) {
                self.cancel_stream_if_unused(sk);
            }
        }
    }

    /// Return immutable UI summaries of all loaded overlay layers.
    pub(crate) fn layer_summaries(&self) -> Vec<LayerSummary> {
        let layers = self.stack.layers();
        let all_stream_keys: Vec<StreamKey> = layers
            .iter()
            .flat_map(|l| l.streams().map(|s| s.stream_key()))
            .collect();
        let flat_stats = self.runtime.layer_flight_stats(all_stream_keys);
        let mut stats_iter = flat_stats.into_iter();
        let len = layers.len();
        let summaries = layers
            .iter()
            .enumerate()
            .map(|(idx, layer)| {
                let count = layer.active_stream_count();
                let mut in_flight = 0usize;
                let mut pass_active = false;
                let mut consumed = 0usize;
                for (f, p) in stats_iter.by_ref().take(count) {
                    in_flight = in_flight.saturating_add(f);
                    pass_active |= p;
                    consumed += 1;
                }
                debug_assert_eq!(
                    consumed, count,
                    "layer active_stream_count() must match streams().count()"
                );
                layer.to_summary(in_flight, pass_active, idx + 1 < len, idx > 0)
            })
            .collect();
        debug_assert!(
            stats_iter.next().is_none(),
            "all layer stream flight stats should be consumed"
        );
        summaries
    }

    /// Number of loaded raster layers (O(1), lock-free).
    pub(crate) fn layer_count(&self) -> usize {
        self.stack.layers().len()
    }

    /// Ordered live raster layer IDs in painter's order (bottom-to-top).
    pub(crate) fn layer_ids(&self) -> Vec<LayerId> {
        self.stack.layers().iter().map(OverlayLayer::id).collect()
    }

    /// Look up the current CRS override for a single-raster layer `id`, if any.
    pub(crate) fn layer_crs_override(&self, id: LayerId) -> Option<CrsOverride> {
        self.stack
            .get(id)
            .and_then(OverlayLayer::single_asset)
            .and_then(|a| a.geo().crs_override().cloned())
    }

    /// Owned persistence snapshots of all loaded raster layers paired with
    /// their [`LayerId`] in painter's order (bottom-to-top).
    pub(crate) fn saved_entries_iter(
        &self,
    ) -> impl Iterator<Item = (LayerId, OverlayPersistEntry)> + '_ {
        self.stack
            .layers()
            .iter()
            .map(|layer| (layer.id(), layer.persist_entry()))
    }

    /// Move an existing layer to painter's-order index `target_idx` (used when
    /// asynchronously restored layers settle out of order).
    pub(crate) fn move_layer_to_index(
        &mut self,
        id: LayerId,
        target_idx: usize,
    ) -> OverlayMutation {
        OverlayMutation {
            changed: self.stack.move_to_index(id, target_idx),
            ..OverlayMutation::default()
        }
    }

    /// Remove every loaded raster layer and cancel/fence any in-flight local or
    /// remote opens from the prior workspace (e.g., before loading a
    /// `.veldmap.json` project), returning purged `SourceId`s.
    pub(crate) fn clear_all(&mut self) -> OverlayMutation {
        self.workspace_cancel.cancel();
        self.workspace_cancel = self.runtime.shutdown.child_token();
        self.workspace_epoch = self.workspace_epoch.saturating_add(1);
        let ids: Vec<LayerId> = self.layer_ids();
        let mut combined = OverlayMutation::default();
        for id in ids {
            let m = self.remove_layer(id);
            combined.changed |= m.changed;
            combined.topology_changed |= m.topology_changed;
            combined.purge_sources.extend(m.purge_sources);
        }
        combined
    }

    fn mutate_layer(
        &mut self,
        id: LayerId,
        f: impl FnOnce(&mut OverlayLayer) -> bool,
    ) -> OverlayMutation {
        OverlayMutation {
            changed: self.stack.get_mut(id).is_some_and(f),
            ..OverlayMutation::default()
        }
    }

    /// Apply an infallible [`OverlayEdit`] to `id`, returning cache-purge and
    /// topology side effects for `SceneState`.
    pub(crate) fn edit_layer(&mut self, id: LayerId, edit: OverlayEdit) -> OverlayMutation {
        match edit {
            OverlayEdit::SetVisible(visible) => self.set_visible(id, visible),
            OverlayEdit::Remove => self.remove_layer(id),
            OverlayEdit::MoveUp => OverlayMutation {
                changed: self.stack.move_up(id),
                ..OverlayMutation::default()
            },
            OverlayEdit::MoveDown => OverlayMutation {
                changed: self.stack.move_down(id),
                ..OverlayMutation::default()
            },
            OverlayEdit::SetOpacity(opacity) => {
                self.mutate_layer(id, |layer| layer.set_opacity(opacity))
            }
            OverlayEdit::SetStretchPreset(preset) => {
                self.mutate_layer(id, |layer| layer.set_stretch_preset(preset))
            }
            OverlayEdit::SetStretchRange { min, max } => {
                self.mutate_layer(id, |layer| layer.set_stretch_range(min, max))
            }
            OverlayEdit::SetColormap { colormap, invert } => {
                self.mutate_layer(id, |layer| layer.set_colormap(colormap, invert))
            }
            OverlayEdit::SetCategoricalPalette(palette) => {
                self.mutate_layer(id, |layer| layer.set_categorical_palette(palette))
            }
            OverlayEdit::SetBandMathRgbStretch { min, max, gamma } => {
                self.mutate_layer(id, |layer| layer.set_band_math_rgb_stretch(min, max, gamma))
            }
            OverlayEdit::ResetBandMathStretch => {
                self.mutate_layer(id, OverlayLayer::reset_band_math_stretch)
            }
            OverlayEdit::SetBandMathSclEnabled(enabled) => {
                let (changed, disabled_scl_stream) =
                    self.stack.get_mut(id).map_or((false, None), |layer| {
                        layer.set_band_math_scl_enabled(enabled)
                    });
                if let Some(sk) = disabled_scl_stream {
                    self.cancel_stream_if_unused(sk);
                }
                OverlayMutation {
                    changed,
                    topology_changed: changed,
                    purge_sources: Vec::new(),
                }
            }
            OverlayEdit::SetBandMathSclMaskBits(bits) => {
                self.mutate_layer(id, |layer| layer.set_band_math_scl_mask_bits(bits))
            }
        }
    }

    /// Rebind an existing layer's affine placement to `crs_override` (or reset
    /// to its embedded/sidecar CRS when `None`), bumping `relaid` to rebuild
    /// warp meshes without re-decoding image tiles.
    pub(crate) fn override_layer_crs(
        &mut self,
        id: LayerId,
        crs_override: Option<CrsOverride>,
    ) -> Result<(Option<FootprintBounds>, OverlayMutation), String> {
        let Some(layer) = self.stack.get_mut(id) else {
            return Err("layer no longer exists".to_owned());
        };
        let Some(asset) = layer.single_asset() else {
            return Err("band-math layers do not support per-layer CRS overrides".to_owned());
        };
        let frame = bind_frame_with_override(asset.source().description(), crs_override.as_ref())?;
        let relaid = alloc_seq(&mut self.next_relaid);
        layer.apply_rebound_frame(frame, crs_override, relaid);
        Ok((
            layer.footprint(),
            OverlayMutation {
                changed: true,
                ..OverlayMutation::default()
            },
        ))
    }

    fn cancel_stream_if_unused(&self, stream_key: StreamKey) {
        let still_used = self
            .stack
            .layers()
            .iter()
            .any(|l| l.is_active() && l.streams().any(|s| s.stream_key() == stream_key));
        if !still_used {
            self.runtime.cancel_stream(stream_key);
        }
    }

    pub(crate) fn has_layer(&self, id: LayerId) -> bool {
        self.stack.get(id).is_some()
    }

    /// Remove a layer from [`LayerStack`], cancelling active jobs and returning
    /// its `SourceId`(s) for GPU/pending purge unless another layer still shares
    /// that exact content fingerprint.
    fn remove_layer(&mut self, id: LayerId) -> OverlayMutation {
        let Some(mut removed) = self.stack.remove(id) else {
            return OverlayMutation::default();
        };
        removed.cancel_stats_job();
        if let Some(tok) = removed.take_reload_cancel() {
            tok.cancel();
        }
        let sources = removed.all_source_ids();
        let stream_keys = removed.all_stream_keys();
        let mut mutation = OverlayMutation {
            changed: true,
            topology_changed: true,
            purge_sources: Vec::new(),
        };
        self.release_sources_and_streams(
            &sources,
            &stream_keys,
            Some(0),
            &mut mutation.purge_sources,
        );
        mutation
    }

    fn set_visible(&mut self, id: LayerId, visible: bool) -> OverlayMutation {
        let (sources, stream_keys) = {
            let Some(layer) = self.stack.get_mut(id) else {
                return OverlayMutation::default();
            };
            if !layer.set_visible(visible) {
                return OverlayMutation::default();
            }
            (layer.all_source_ids(), layer.all_stream_keys())
        };
        if !visible {
            let mut unused_gpu_purge = Vec::new();
            self.release_sources_and_streams(&sources, &stream_keys, None, &mut unused_gpu_purge);
        }
        OverlayMutation {
            changed: true,
            topology_changed: true,
            purge_sources: Vec::new(),
        }
    }

    pub(crate) fn set_layer_stream(
        &mut self,
        id: LayerId,
        spec: TileStreamSpec,
    ) -> Result<OverlayMutation, String> {
        let (changed, old_stream, source_id) = {
            let Some(layer) = self.stack.get_mut(id) else {
                return Err("layer no longer exists".to_owned());
            };
            let Some(asset) = layer.single_asset() else {
                return Err(
                    "band-math layers do not support switching single-band streams".to_owned(),
                );
            };
            let prev_stream = asset.meta().stream_key();
            let source_id = asset.source_id();
            let changed = layer.set_active_stream(spec)?;
            (changed, prev_stream, source_id)
        };
        if !changed {
            return Ok(OverlayMutation::default());
        }
        self.cancel_stream_if_unused(old_stream);
        self.cursor.invalidate_source(source_id);
        self.request_band_stats_if_needed(id);
        self.runtime.request_reschedule();
        Ok(OverlayMutation {
            changed: true,
            topology_changed: true,
            purge_sources: Vec::new(),
        })
    }

    pub(crate) fn record_upload_report(&mut self, report: &UploadBatchReport) {
        for &(stream_key, max_bytes) in &report.max_tile_bytes_by_stream {
            let mut is_remote = false;
            for layer in self.stack.layers_mut() {
                layer.observe_stream_tile_bytes(stream_key, max_bytes);
                if layer.has_remote_source(stream_key.source) {
                    is_remote = true;
                }
            }
            if is_remote {
                self.cursor
                    .invalidate_unavailable_for_source(stream_key.source);
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
        let streams = self
            .stack
            .layers()
            .iter()
            .filter(|l| l.is_active())
            .flat_map(|l| l.streams().map(|s| (s.stream_key(), s.wanted())));
        self.runtime
            .reconcile_pointwise_streams(streams, gpu_cache, pending);
    }

    /// Build immutable [`DrawOverlayLayer`]s for the current frame, update
    /// per-layer wanted sets across all ladder rungs, reconcile stale pointwise
    /// jobs once across unioned active streams, and schedule missing tiles onto
    /// the two-lane decode pool.
    pub(crate) fn build_and_schedule(
        &mut self,
        ctx: &mut OverlayFrameCtx<'_>,
    ) -> Vec<DrawOverlayLayer> {
        let aspect = f64::from(ctx.width_px.max(1)) / f64::from(ctx.height_px.max(1));
        let frustum = ctx.camera.frustum(aspect);
        let viewport_mpp = ctx.camera.metres_per_pixel(f64::from(ctx.height_px.max(1)));

        // Phase 1: compute visible target/ladder rungs and replace `wanted` on
        // every active asset before reconciling so shared `StreamKey`s union.
        let prep_ctx = PrepareFrameCtx {
            frustum: &frustum,
            viewport_mpp,
            gpu_cache: ctx.gpu_cache,
            total_appetite_weight: ctx.total_appetite_weight,
        };
        let mut stats_needed_ids = Vec::new();
        let mut prepared = Vec::with_capacity(self.stack.layers().len());
        for layer in self.stack.layers_mut() {
            if layer.needs_band_stats().is_some() {
                stats_needed_ids.push(layer.id());
            }
            prepared.push(prepare_layer_wanted(layer, &prep_ctx));
        }

        // Phase 2: reconcile in-flight pointwise jobs once across the union of
        // all active layers' wanted sets.
        self.reconcile(ctx.gpu_cache, ctx.pending);

        // Phase 3: build draw patches from cached carriers and schedule missing
        // tiles onto `fast_pool` / `bulk_pool`.
        let mut layer_ctx = LayerFrameCtx {
            runtime: &self.runtime,
            gpu_cache: ctx.gpu_cache,
            pending: ctx.pending,
            carrier_keys: ctx.carrier_keys,
        };
        let mut draw_layers = Vec::new();
        for (layer, plan) in self.stack.layers().iter().zip(prepared) {
            if let Some(plan) = plan
                && let Some(draw_layer) = finish_scheduled_layer(layer, plan, &mut layer_ctx)
            {
                draw_layers.push(draw_layer);
            }
        }
        for id in stats_needed_ids {
            self.request_band_stats_if_needed(id);
        }
        draw_layers
    }

    pub(crate) fn cancel_cursor_readout(&self) {
        self.cursor.cancel();
    }

    pub(crate) fn query_cursor_readout(
        &self,
        lat_deg: f64,
        lon_deg: f64,
        excluded_layer: Option<LayerId>,
    ) -> Option<CursorReadout> {
        self.cursor
            .query(&self.stack, &self.runtime, lat_deg, lon_deg, excluded_layer)
    }

    /// Phase 1 shutdown: cancel active overlay passes, pointwise jobs, reloads,
    /// cursor probes, and close the I/O and tile-write worker channels.
    pub(crate) fn begin_shutdown(&mut self) {
        let mut all_sources = Vec::new();
        for layer in self.stack.layers() {
            for src in layer.all_source_ids() {
                if !all_sources.contains(&src) {
                    all_sources.push(src);
                }
            }
        }
        self.runtime.cancel_sources(all_sources);
        for layer in self.stack.layers_mut() {
            layer.cancel_stats_job();
            if let Some(tok) = layer.take_reload_cancel() {
                tok.cancel();
            }
        }
        self.cursor.cancel();
        self.io_worker.close_channel();
        if let Some(writer) = &self.tile_writer {
            writer.close();
        }
    }

    /// Phase 3 shutdown: deterministically join the remote I/O pool, overlay
    /// I/O worker, and offloaded tile-write worker.
    pub(crate) fn finish_shutdown(&mut self) {
        if let Some(pool) = self.remote_pool.take() {
            pool.shutdown();
        }
        self.io_worker.join();
        if let Some(mut writer) = self.tile_writer.take() {
            writer.join();
        }
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
