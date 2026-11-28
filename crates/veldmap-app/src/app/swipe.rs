//! Live split-screen swipe curtain state owned by `VeldmapApp` (kept outside
//! `DrawSnapshot` so dragging the curtain at 120 Hz never rebuilds the globe
//! quadtree or overlay cull lists).

use crate::layers::{LayerId, LayerSummary};
use crate::render::SwipeRenderState;

/// Interactive split-screen swipe curtain state comparing any two loaded raster
/// layers (or one raster layer against the underlying stack / basemap).
#[derive(Clone, Debug)]
pub(super) struct SwipeState {
    active: bool,
    split_t: f32,
    left_layer: Option<LayerId>,
    right_layer: Option<LayerId>,
}

impl Default for SwipeState {
    fn default() -> Self {
        Self {
            active: false,
            split_t: 0.5,
            left_layer: None,
            right_layer: None,
        }
    }
}

impl SwipeState {
    pub(super) const fn is_active(&self) -> bool {
        self.active
    }

    pub(super) const fn split_t(&self) -> f32 {
        self.split_t
    }

    pub(super) const fn left_layer(&self) -> Option<LayerId> {
        self.left_layer
    }

    pub(super) const fn right_layer(&self) -> Option<LayerId> {
        self.right_layer
    }

    pub(super) fn set_split_t(&mut self, t: f32) -> bool {
        let clamped = if t.is_finite() {
            t.clamp(0.0, 1.0)
        } else {
            0.5
        };
        if (self.split_t - clamped).abs() <= f32::EPSILON {
            return false;
        }
        self.split_t = clamped;
        true
    }

    pub(super) fn set_left_layer(&mut self, id: Option<LayerId>) -> bool {
        if self.left_layer == id {
            return false;
        }
        self.left_layer = id;
        true
    }

    pub(super) fn set_right_layer(&mut self, id: Option<LayerId>) -> bool {
        if self.right_layer == id {
            return false;
        }
        self.right_layer = id;
        true
    }

    pub(super) fn swap_sides(&mut self) {
        std::mem::swap(&mut self.left_layer, &mut self.right_layer);
    }

    pub(super) fn activate(&mut self, layers: &[LayerSummary]) {
        self.active = true;
        self.reconcile_layers(layers);
        if self.left_layer.is_none() && self.right_layer.is_none() {
            // Auto-select top two visible layers (or top two layers, or top
            // layer vs basemap when only one raster is loaded).
            let mut visible_top: Vec<LayerId> = layers
                .iter()
                .rev()
                .filter(|l| l.visible)
                .map(|l| l.id)
                .collect();
            if visible_top.is_empty() {
                visible_top = layers.iter().rev().map(|l| l.id).collect();
            }
            self.left_layer = visible_top.first().copied();
            self.right_layer = visible_top.get(1).copied();
        }
    }

    pub(super) fn deactivate(&mut self) {
        self.active = false;
    }

    pub(super) fn restore(
        &mut self,
        split_t: f32,
        left_layer: Option<LayerId>,
        right_layer: Option<LayerId>,
    ) {
        self.active = true;
        self.split_t = if split_t.is_finite() {
            split_t.clamp(0.0, 1.0)
        } else {
            0.5
        };
        self.left_layer = left_layer;
        self.right_layer = right_layer;
    }

    pub(super) fn reset(&mut self) {
        self.active = false;
        self.split_t = 0.5;
        self.left_layer = None;
        self.right_layer = None;
    }

    /// Reconcile selected layer IDs against the currently loaded raster stack,
    /// clearing removed layers and exiting swipe cleanly if no raster layers
    /// remain or if removing selected layer(s) leaves both sides unselected.
    /// Returns `true` when active swipe state changed.
    pub(super) fn reconcile_layers(&mut self, layers: &[LayerSummary]) -> bool {
        let mut removed_any = false;
        if let Some(id) = self.left_layer
            && !layers.iter().any(|l| l.id == id)
        {
            self.left_layer = None;
            removed_any = true;
        }
        if let Some(id) = self.right_layer
            && !layers.iter().any(|l| l.id == id)
        {
            self.right_layer = None;
            removed_any = true;
        }
        if self.active
            && (layers.is_empty()
                || (removed_any && self.left_layer.is_none() && self.right_layer.is_none()))
        {
            self.active = false;
            return true;
        }
        removed_any
    }

    /// Return the layer ID clipped away by the active swipe curtain at cursor
    /// logical X coordinate `cursor_x` (matching `ResolvedSwipe::scissor_for_layer`
    /// identically, including `split_x == 0`, `split_x >= width_px`, and
    /// same-layer full-viewport collapse).
    pub(super) fn excluded_layer_for_cursor(
        &self,
        cursor_x: f32,
        div_x: f32,
        split_x: u32,
        width_px: u32,
    ) -> Option<LayerId> {
        if !self.active || self.left_layer == self.right_layer {
            return None;
        }
        if split_x == 0 {
            self.left_layer
        } else if split_x >= width_px || cursor_x < div_x {
            self.right_layer
        } else {
            self.left_layer
        }
    }

    /// Produce the renderer-facing [`SwipeRenderState`] when Swipe is active.
    pub(super) fn to_render_state(&self) -> Option<SwipeRenderState> {
        self.active.then(|| SwipeRenderState {
            split_t: self.split_t,
            left_layer: self.left_layer.map(LayerId::raw),
            right_layer: self.right_layer.map(LayerId::raw),
        })
    }

    /// Resolve human-readable labels for the left and right sides of the
    /// divider.
    pub(super) fn side_labels(&self, layers: &[LayerSummary]) -> (String, String) {
        let lookup = |id: Option<LayerId>| -> String {
            id.and_then(|target| layers.iter().find(|l| l.id == target))
                .map_or_else(
                    || "Basemap".to_owned(),
                    |l| {
                        if l.visible {
                            l.name.clone()
                        } else {
                            format!("{} (hidden)", l.name)
                        }
                    },
                )
        };
        (lookup(self.left_layer), lookup(self.right_layer))
    }
}
