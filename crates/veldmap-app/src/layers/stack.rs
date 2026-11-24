//! Ordered [`LayerStack`] and per-layer metadata/state (`OverlayLayer`).

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use veldmap_core::geo::{FootprintBounds, Frame};
use veldmap_core::sync::{CancelToken, lock_unpoisoned};
use veldmap_core::tile::{OverlayCullTree, SourceId, TileKey};
use veldmap_raster::{PyramidMeta, RasterSource};

use crate::session::SavedCrsOverride;

use super::{LayerId, LayerSummary, default_display_tile_bytes};

/// Initial display and georeference override options when inserting an opened
/// raster layer into [`LayerStack`].
#[derive(Clone, Debug)]
pub(super) struct LayerInitOptions {
    pub(super) visible: bool,
    pub(super) opacity: f32,
    pub(super) crs_override: Option<SavedCrsOverride>,
}

/// Single georeferenced raster overlay inside [`LayerStack`].
pub(super) struct OverlayLayer {
    pub(super) id: LayerId,
    pub(super) name: String,
    pub(super) path: PathBuf,
    pub(super) source: Arc<dyn RasterSource>,
    pub(super) meta: PyramidMeta,
    pub(super) fingerprint_hex: String,
    pub(super) frame: Arc<Frame>,
    pub(super) crs_override: Option<SavedCrsOverride>,
    pub(super) crs_override_epoch: u64,
    pub(super) last_error: Option<String>,
    pub(super) relaid: u64,
    pub(super) cull: OverlayCullTree,
    pub(super) visible: bool,
    pub(super) opacity: f32,
    pub(super) source_pixel_size_m: Option<(f64, f64)>,
    pub(super) footprint: Option<FootprintBounds>,
    /// Current wanted tile set (shared with active `ServeMode::Pass` jobs so
    /// mid-pass zooms deliver tiles matching the live viewport).
    pub(super) wanted_shared: Arc<Mutex<HashSet<TileKey>>>,
    /// Snapshot of `wanted_shared` retained on the UI thread for
    /// `PendingUploads` retention without acquiring a lock inside upload drain.
    pub(super) wanted: HashSet<TileKey>,
    pub(super) observed_tile_bytes: usize,
    reloading: bool,
    reload_seq: u64,
    reload_cancel: Option<CancelToken>,
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
            frame: Arc::new(frame),
            crs_override: options.crs_override,
            crs_override_epoch: 0,
            last_error: None,
            relaid,
            cull,
            visible: options.visible,
            opacity: options.opacity.clamp(0.0, 1.0),
            source_pixel_size_m,
            footprint,
            wanted_shared: Arc::new(Mutex::new(HashSet::new())),
            wanted: HashSet::new(),
            observed_tile_bytes: default_display_tile_bytes(),
            reloading: false,
            reload_seq: 0,
            reload_cancel: None,
        }
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
            can_override_crs: desc.placement.is_some() || !desc.ties.is_empty(),
            has_crs_override: self.crs_override.is_some(),
            last_error: self.last_error.clone(),
            can_move_up,
            can_move_down,
        }
    }

    pub(super) fn apply_rebound_frame(
        &mut self,
        frame: Frame,
        crs_override: Option<SavedCrsOverride>,
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
        self.visible && !self.reloading
    }

    pub(super) fn is_reloading(&self) -> bool {
        self.reloading
    }

    pub(super) fn begin_reload(&mut self, cancel: CancelToken) -> (u64, Option<CancelToken>) {
        let prev = self.reload_cancel.replace(cancel);
        self.reload_seq = self.reload_seq.saturating_add(1);
        self.reloading = true;
        self.clear_wanted();
        (self.reload_seq, prev)
    }

    pub(super) fn abort_reload(&mut self) {
        self.reloading = false;
        self.reload_cancel = None;
    }

    pub(super) fn matches_reload_seq(&self, seq: u64) -> bool {
        self.reload_seq == seq
    }

    pub(super) fn finish_reload_err(&mut self, seq: u64) -> bool {
        if self.reload_seq != seq {
            return false;
        }
        self.reloading = false;
        self.reload_cancel = None;
        true
    }

    pub(super) fn replace_reloaded(
        &mut self,
        seq: u64,
        path: PathBuf,
        source: Arc<dyn RasterSource>,
        frame: Frame,
        relaid: u64,
    ) -> Option<SourceId> {
        if self.reload_seq != seq {
            return None;
        }
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
        replacement.reload_seq = seq;
        replacement.crs_override_epoch = self.crs_override_epoch;
        *self = replacement;
        Some(old_src)
    }

    pub(super) fn take_reload_cancel(&mut self) -> Option<CancelToken> {
        self.reload_cancel.take()
    }

    pub(super) fn appetite_weight_bytes(&self) -> u64 {
        if self.is_active() {
            (self.observed_tile_bytes as u64).max(1024)
        } else {
            0
        }
    }

    pub(super) fn sync_wanted_shared(&self) {
        let mut shared = lock_unpoisoned(&self.wanted_shared);
        shared.clone_from(&self.wanted);
    }

    pub(super) fn clear_wanted(&mut self) {
        self.wanted.clear();
        lock_unpoisoned(&self.wanted_shared).clear();
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
            if layer.is_active() && !layer.wanted.is_empty() {
                out.push(&layer.wanted);
            }
        }
    }
}
