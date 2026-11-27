//! Ordered [`LayerStack`] collection of draped raster overlays in painter's
//! order.

use std::collections::HashSet;

use veldmap_core::tile::{SourceId, StreamKey, TileKey};

use super::LayerId;
pub(super) use super::layer::OverlayLayer;
use super::style::ColorbarInfo;

/// Ordered collection of draped raster overlays in painter's order.
#[derive(Default)]
pub(super) struct LayerStack {
    layers: Vec<OverlayLayer>,
}

impl LayerStack {
    pub(super) fn push(&mut self, layer: OverlayLayer) {
        self.layers.push(layer);
    }

    pub(super) fn remove(&mut self, id: LayerId) -> Option<OverlayLayer> {
        let idx = self.layers.iter().position(|l| l.id() == id)?;
        Some(self.layers.remove(idx))
    }

    pub(super) fn move_up(&mut self, id: LayerId) -> bool {
        let Some(idx) = self.layers.iter().position(|l| l.id() == id) else {
            return false;
        };
        if idx + 1 >= self.layers.len() {
            return false;
        }
        self.layers.swap(idx, idx + 1);
        true
    }

    pub(super) fn move_down(&mut self, id: LayerId) -> bool {
        let Some(idx) = self.layers.iter().position(|l| l.id() == id) else {
            return false;
        };
        if idx == 0 {
            return false;
        }
        self.layers.swap(idx - 1, idx);
        true
    }

    pub(super) fn get(&self, id: LayerId) -> Option<&OverlayLayer> {
        self.layers.iter().find(|l| l.id() == id)
    }

    pub(super) fn get_mut(&mut self, id: LayerId) -> Option<&mut OverlayLayer> {
        self.layers.iter_mut().find(|l| l.id() == id)
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
            .filter(|l| l.has_source(source_id))
            .count()
    }

    pub(super) fn has_visible_source(&self, source_id: SourceId) -> bool {
        self.layers
            .iter()
            .any(|l| l.is_active() && l.has_source(source_id))
    }

    pub(super) fn max_scratch_requirements(&self) -> (usize, bool) {
        let mut max_float_bands = 0usize;
        let mut needs_mask = false;
        for layer in &self.layers {
            let (fb, mask) = layer.scratch_requirements();
            max_float_bands = max_float_bands.max(fb);
            needs_mask |= mask;
        }
        (max_float_bands, needs_mask)
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
            .and_then(|l| l.geo().source_pixel_size_m())
    }

    pub(super) fn topmost_colorbar_info(&self) -> Option<ColorbarInfo> {
        self.layers
            .iter()
            .rev()
            .find_map(OverlayLayer::colorbar_info)
    }

    pub(super) fn extend_pinned_root_streams(&self, out: &mut Vec<StreamKey>) {
        for layer in &self.layers {
            if layer.is_active() && layer.pin_root() {
                for stream in layer.streams() {
                    let key = stream.stream_key();
                    if !out.contains(&key) {
                        out.push(key);
                    }
                }
            }
        }
    }

    pub(super) fn extend_wanted_tile_sets<'a>(&'a self, out: &mut Vec<&'a HashSet<TileKey>>) {
        for layer in &self.layers {
            if layer.is_active() {
                for stream in layer.streams() {
                    if !stream.wanted().is_empty() {
                        out.push(stream.wanted());
                    }
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "stack_tests.rs"]
mod tests;
