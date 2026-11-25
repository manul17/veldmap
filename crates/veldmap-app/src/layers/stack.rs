//! Ordered [`LayerStack`] and per-layer metadata/state (`OverlayLayer`).

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use veldmap_core::geo::{CullFrustum, FootprintBounds, Frame};
use veldmap_core::sync::{CancelToken, lock_unpoisoned};
use veldmap_core::tile::{OverlayCullTree, SourceId, TileKey};
use veldmap_raster::{PyramidMeta, RasterSource};

use crate::{clamp_opacity, display_name_from_path};

use super::{
    CrsOverride, LayerId, LayerInitOptions, LayerSummary, OverlayPersistEntry,
    default_display_tile_bytes,
};

/// Layer-domain state transition record returned by [`OverlayLayer::begin_reload`].
pub(super) struct ReloadPrep {
    pub(super) reload_seq: u64,
    pub(super) old_source: SourceId,
    pub(super) old_cancel: Option<CancelToken>,
}

/// Cohesive per-layer reload lifecycle state.
#[derive(Default)]
struct ReloadState {
    in_progress: bool,
    seq: u64,
    cancel: Option<CancelToken>,
}

/// Single georeferenced raster overlay inside [`LayerStack`].
///
/// Mutable fields stay private behind transition methods so visibility,
/// opacity, wanted-set synchronization, reload lifecycle, and CRS rebinding
/// invariants cannot be broken by callers.
pub(super) struct OverlayLayer {
    id: LayerId,
    name: String,
    path: PathBuf,
    source: Arc<dyn RasterSource>,
    meta: PyramidMeta,
    fingerprint_hex: String,
    frame: Arc<Frame>,
    crs_override: Option<CrsOverride>,
    crs_override_epoch: u64,
    last_error: Option<String>,
    relaid: u64,
    cull: OverlayCullTree,
    visible: bool,
    opacity: f32,
    source_pixel_size_m: Option<(f64, f64)>,
    footprint: Option<FootprintBounds>,
    /// Current wanted tile set (shared with active `ServeMode::Pass` jobs so
    /// mid-pass zooms deliver tiles matching the live viewport).
    wanted_shared: Arc<Mutex<HashSet<TileKey>>>,
    /// Snapshot of `wanted_shared` retained on the UI thread for
    /// `PendingUploads` retention without acquiring a lock inside upload drain.
    wanted: HashSet<TileKey>,
    observed_tile_bytes: usize,
    reload: ReloadState,
}

impl OverlayLayer {
    pub(super) fn from_source(
        id: LayerId,
        path: PathBuf,
        source: Arc<dyn RasterSource>,
        frame: Frame,
        options: LayerInitOptions,
        relaid: u64,
    ) -> Self {
        let name = display_name_from_path(&path);
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
            frame: Arc::new(frame),
            crs_override: options.crs_override,
            crs_override_epoch: 0,
            last_error: None,
            relaid,
            cull,
            visible: options.visible,
            opacity: clamp_opacity(options.opacity),
            source_pixel_size_m,
            footprint,
            wanted_shared: Arc::new(Mutex::new(HashSet::new())),
            wanted: HashSet::new(),
            observed_tile_bytes: default_display_tile_bytes(),
            reload: ReloadState::default(),
        }
    }

    pub(super) fn name(&self) -> &str {
        &self.name
    }

    pub(super) fn source(&self) -> &Arc<dyn RasterSource> {
        &self.source
    }

    pub(super) fn meta(&self) -> &PyramidMeta {
        &self.meta
    }

    pub(super) fn fingerprint_hex(&self) -> &str {
        &self.fingerprint_hex
    }

    pub(super) fn frame(&self) -> &Arc<Frame> {
        &self.frame
    }

    pub(super) fn crs_override(&self) -> Option<&CrsOverride> {
        self.crs_override.as_ref()
    }

    pub(super) fn crs_override_epoch(&self) -> u64 {
        self.crs_override_epoch
    }

    pub(super) fn relaid(&self) -> u64 {
        self.relaid
    }

    pub(super) fn opacity(&self) -> f32 {
        self.opacity
    }

    pub(super) fn source_pixel_size_m(&self) -> Option<(f64, f64)> {
        self.source_pixel_size_m
    }

    pub(super) fn footprint(&self) -> Option<FootprintBounds> {
        self.footprint
    }

    pub(super) fn observed_tile_bytes(&self) -> usize {
        self.observed_tile_bytes
    }

    pub(super) fn persist_entry(&self) -> OverlayPersistEntry<'_> {
        OverlayPersistEntry {
            path: &self.path,
            visible: self.visible,
            opacity: self.opacity,
            crs_override: self.crs_override.as_ref(),
        }
    }

    pub(super) fn visible_cells(&mut self, level: u8, frustum: &CullFrustum) -> Vec<(u32, u32)> {
        self.cull.visible_cells(&self.frame, level, frustum)
    }

    pub(super) fn set_visible(&mut self, visible: bool) -> bool {
        if self.visible == visible {
            return false;
        }
        self.visible = visible;
        if !visible {
            self.clear_wanted();
        }
        true
    }

    pub(super) fn set_opacity(&mut self, opacity: f32) -> bool {
        let clamped = clamp_opacity(opacity);
        if (self.opacity - clamped).abs() <= f32::EPSILON {
            return false;
        }
        self.opacity = clamped;
        true
    }

    pub(super) fn observe_tile_bytes(&mut self, bytes: usize) {
        self.observed_tile_bytes = self.observed_tile_bytes.max(bytes);
    }

    pub(super) fn to_summary(
        &self,
        in_flight: usize,
        pass_active: bool,
        can_move_up: bool,
        can_move_down: bool,
    ) -> LayerSummary {
        let desc = self.source.description();
        let sample_info = desc.bands.first().map_or_else(
            || "Display RGBA".to_owned(),
            |b| {
                let nodata_str = b
                    .nodata
                    .map(|v| format!(", nodata={v}"))
                    .unwrap_or_default();
                format!("{} ch, {:?}{nodata_str}", b.channels, b.sample_type)
            },
        );
        LayerSummary {
            id: self.id,
            name: self.name.clone(),
            path: self.path.clone(),
            visible: self.visible,
            opacity: self.opacity,
            width: desc.width,
            height: desc.height,
            source_pixel_size_m: self.source_pixel_size_m,
            footprint: self.footprint,
            in_flight,
            pass_active,
            reloading: self.is_reloading(),
            crs_label: self.frame.crs_label(),
            datum_warning: self.frame.datum_warning().map(|w| w.message()),
            sample_info,
            detail_limit_level: self
                .meta
                .finest_fitting_level()
                .unwrap_or(self.meta.max_level()),
            can_override_crs: desc.can_override_crs(),
            has_crs_override: self.crs_override.is_some(),
            last_error: self.last_error.clone(),
            can_move_up,
            can_move_down,
        }
    }

    pub(super) fn apply_rebound_frame(
        &mut self,
        frame: Frame,
        crs_override: Option<CrsOverride>,
        relaid: u64,
    ) {
        let desc = self.source.description();
        self.source_pixel_size_m = frame.pixel_size_m(desc.width, desc.height);
        self.footprint = frame.footprint_bounds();
        self.frame = Arc::new(frame);
        self.crs_override = crs_override;
        self.crs_override_epoch = self.crs_override_epoch.saturating_add(1);
        self.last_error = None;
        self.relaid = relaid;
        self.cull = OverlayCullTree::new(desc.width, desc.height, self.meta.max_level());
    }

    pub(super) fn source_id(&self) -> SourceId {
        self.meta.source
    }

    pub(super) fn is_active(&self) -> bool {
        self.visible && !self.reload.in_progress
    }

    pub(super) fn is_reloading(&self) -> bool {
        self.reload.in_progress
    }

    pub(super) fn begin_reload(&mut self, cancel: CancelToken) -> ReloadPrep {
        let old_cancel = self.reload.cancel.replace(cancel);
        self.reload.seq = self.reload.seq.saturating_add(1);
        self.reload.in_progress = true;
        self.clear_wanted();
        ReloadPrep {
            reload_seq: self.reload.seq,
            old_source: self.source_id(),
            old_cancel,
        }
    }

    pub(super) fn abort_reload(&mut self) {
        self.reload.in_progress = false;
        self.reload.cancel = None;
    }

    pub(super) fn matches_reload_seq(&self, seq: u64) -> bool {
        self.reload.seq == seq
    }

    pub(super) fn fail_reload(&mut self, error: String) {
        self.reload.in_progress = false;
        self.reload.cancel = None;
        self.last_error = Some(error);
    }

    pub(super) fn replace_reloaded(
        &mut self,
        path: PathBuf,
        source: Arc<dyn RasterSource>,
        frame: Frame,
        relaid: u64,
    ) -> SourceId {
        let old_src = self.source_id();
        let mut replacement = Self::from_source(
            self.id,
            path,
            source,
            frame,
            LayerInitOptions {
                crs_override: self.crs_override.clone(),
                visible: self.visible,
                opacity: self.opacity,
            },
            relaid,
        );
        replacement.reload.seq = self.reload.seq;
        replacement.crs_override_epoch = self.crs_override_epoch;
        *self = replacement;
        old_src
    }

    pub(super) fn take_reload_cancel(&mut self) -> Option<CancelToken> {
        self.reload.cancel.take()
    }

    pub(super) fn appetite_weight_bytes(&self) -> u64 {
        if self.is_active() {
            (self.observed_tile_bytes as u64).max(1024)
        } else {
            0
        }
    }

    /// Borrow the UI-thread snapshot of this layer's currently wanted tiles.
    pub(super) fn wanted(&self) -> &HashSet<TileKey> {
        &self.wanted
    }

    /// Clone the shared wanted-tile handle passed to sequential `ServeMode::Pass` jobs.
    pub(super) fn wanted_shared(&self) -> Arc<Mutex<HashSet<TileKey>>> {
        Arc::clone(&self.wanted_shared)
    }

    /// Atomically replace this layer's wanted tile set and synchronize its
    /// worker-shared mirror so the two can never drift out of sync.
    pub(super) fn replace_wanted(&mut self, tiles: impl IntoIterator<Item = TileKey>) {
        self.wanted.clear();
        self.wanted.extend(tiles);
        lock_unpoisoned(&self.wanted_shared).clone_from(&self.wanted);
    }

    pub(super) fn clear_wanted(&mut self) {
        self.wanted.clear();
        lock_unpoisoned(&self.wanted_shared).clear();
    }

    pub(super) fn path(&self) -> &Path {
        &self.path
    }
}

/// Ordered collection of draped raster overlays in painter's order
/// (index `0` drawn right above the basemap, last index drawn on top beneath
/// the procedural graticule).
#[derive(Default)]
pub(super) struct LayerStack {
    layers: Vec<OverlayLayer>,
}

impl LayerStack {
    pub(super) fn push(&mut self, layer: OverlayLayer) {
        self.layers.push(layer);
    }

    pub(super) fn remove(&mut self, id: LayerId) -> Option<OverlayLayer> {
        let idx = self.layers.iter().position(|l| l.id == id)?;
        Some(self.layers.remove(idx))
    }

    /// Move layer toward the top of painter's order (higher index, drawn later).
    pub(super) fn move_up(&mut self, id: LayerId) -> bool {
        let Some(idx) = self.layers.iter().position(|l| l.id == id) else {
            return false;
        };
        if idx + 1 >= self.layers.len() {
            return false;
        }
        self.layers.swap(idx, idx + 1);
        true
    }

    /// Move layer toward the bottom of painter's order (lower index, drawn earlier).
    pub(super) fn move_down(&mut self, id: LayerId) -> bool {
        let Some(idx) = self.layers.iter().position(|l| l.id == id) else {
            return false;
        };
        if idx == 0 {
            return false;
        }
        self.layers.swap(idx - 1, idx);
        true
    }

    pub(super) fn get(&self, id: LayerId) -> Option<&OverlayLayer> {
        self.layers.iter().find(|l| l.id == id)
    }

    pub(super) fn get_mut(&mut self, id: LayerId) -> Option<&mut OverlayLayer> {
        self.layers.iter_mut().find(|l| l.id == id)
    }

    pub(super) fn layers(&self) -> &[OverlayLayer] {
        &self.layers
    }

    pub(super) fn layers_mut(&mut self) -> &mut [OverlayLayer] {
        &mut self.layers
    }

    pub(super) fn count_sharing_source(&self, source_id: SourceId) -> usize {
        self.layers
            .iter()
            .filter(|l| l.source_id() == source_id)
            .count()
    }

    pub(super) fn has_visible_source(&self, source_id: SourceId) -> bool {
        self.layers
            .iter()
            .any(|l| l.is_active() && l.source_id() == source_id)
    }

    pub(super) fn total_appetite_weight_bytes(&self) -> u64 {
        self.layers
            .iter()
            .fold(0u64, |acc, l| acc.saturating_add(l.appetite_weight_bytes()))
    }

    pub(super) fn topmost_visible_source_pixel_size_m(&self) -> Option<(f64, f64)> {
        self.layers
            .iter()
            .rev()
            .find(|l| l.is_active())
            .and_then(|l| l.source_pixel_size_m)
    }

    pub(super) fn extend_pinned_root_sources(&self, out: &mut Vec<SourceId>) {
        for layer in &self.layers {
            if layer.is_active() {
                let src = layer.source_id();
                if !out.contains(&src) {
                    out.push(src);
                }
            }
        }
    }

    pub(super) fn extend_wanted_tile_sets<'a>(&'a self, out: &mut Vec<&'a HashSet<TileKey>>) {
        for layer in &self.layers {
            if layer.is_active() && !layer.wanted().is_empty() {
                out.push(layer.wanted());
            }
        }
    }
}

#[cfg(test)]
#[path = "stack_tests.rs"]
mod tests;
