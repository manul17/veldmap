//! Ordered [`LayerStack`] and per-layer metadata/state (`OverlayLayer`).

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use veldmap_core::geo::Frame;
use veldmap_core::geo::footprint::FootprintBounds;
use veldmap_core::sync::{CancelToken, lock_unpoisoned};
use veldmap_core::tile::{OverlayCullTree, SourceId, TileKey};
use veldmap_raster::{PyramidMeta, RasterSource};

use super::{LayerId, default_display_tile_bytes};

/// Single georeferenced raster overlay inside [`LayerStack`].
pub(crate) struct OverlayLayer {
    pub(crate) id: LayerId,
    pub(crate) name: String,
    pub(crate) path: PathBuf,
    pub(crate) source: Arc<dyn RasterSource>,
    pub(crate) meta: PyramidMeta,
    pub(crate) fingerprint_hex: String,
    pub(crate) frame: Arc<Frame>,
    pub(crate) relaid: u64,
    pub(crate) cull: OverlayCullTree,
    pub(crate) visible: bool,
    pub(crate) opacity: f32,
    pub(crate) source_pixel_size_m: Option<(f64, f64)>,
    pub(crate) footprint: Option<FootprintBounds>,
    /// Current wanted tile set (shared with active `ServeMode::Pass` jobs so
    /// mid-pass zooms deliver tiles matching the live viewport).
    pub(crate) wanted_shared: Arc<Mutex<HashSet<TileKey>>>,
    /// Snapshot of `wanted_shared` retained on the UI thread for
    /// `PendingUploads` retention without acquiring a lock inside upload drain.
    pub(crate) wanted: HashSet<TileKey>,
    pub(crate) observed_tile_bytes: usize,
    pub(crate) reloading: bool,
    pub(crate) reload_seq: u64,
    pub(crate) reload_cancel: Option<CancelToken>,
}

impl OverlayLayer {
    pub(crate) fn from_source(
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
            frame: Arc::new(frame),
            relaid,
            cull,
            visible,
            opacity: opacity.clamp(0.0, 1.0),
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

    pub(crate) fn source_id(&self) -> SourceId {
        self.meta.source
    }

    pub(crate) fn appetite_weight_bytes(&self) -> u64 {
        if self.visible && !self.reloading {
            (self.observed_tile_bytes as u64).max(1024)
        } else {
            0
        }
    }

    pub(crate) fn sync_wanted_shared(&self) {
        let mut shared = lock_unpoisoned(&self.wanted_shared);
        shared.clone_from(&self.wanted);
    }

    pub(crate) fn clear_wanted(&mut self) {
        self.wanted.clear();
        lock_unpoisoned(&self.wanted_shared).clear();
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

    pub(crate) fn get_mut(&mut self, id: LayerId) -> Option<&mut OverlayLayer> {
        self.layers.iter_mut().find(|l| l.id == id)
    }

    pub(crate) fn layers(&self) -> &[OverlayLayer] {
        &self.layers
    }

    pub(crate) fn layers_mut(&mut self) -> &mut [OverlayLayer] {
        &mut self.layers
    }

    pub(crate) fn count_sharing_source(&self, source_id: SourceId) -> usize {
        self.layers
            .iter()
            .filter(|l| l.source_id() == source_id)
            .count()
    }

    pub(crate) fn has_visible_source(&self, source_id: SourceId) -> bool {
        self.layers
            .iter()
            .any(|l| l.visible && !l.reloading && l.source_id() == source_id)
    }

    pub(crate) fn total_appetite_weight_bytes(&self) -> u64 {
        self.layers
            .iter()
            .fold(0u64, |acc, l| acc.saturating_add(l.appetite_weight_bytes()))
    }

    pub(crate) fn extend_pinned_root_sources(&self, out: &mut Vec<SourceId>) {
        for layer in &self.layers {
            if layer.visible && !layer.reloading {
                let src = layer.source_id();
                if !out.contains(&src) {
                    out.push(src);
                }
            }
        }
    }

    pub(crate) fn extend_wanted_tile_sets<'a>(&'a self, out: &mut Vec<&'a HashSet<TileKey>>) {
        for layer in &self.layers {
            if layer.visible && !layer.reloading && !layer.wanted.is_empty() {
                out.push(&layer.wanted);
            }
        }
    }
}
