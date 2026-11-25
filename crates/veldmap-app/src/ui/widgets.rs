//! Reusable `egui` widgets and interaction helpers shared across HUD and side
//! panels (colormap ramps, drag-commit tracking, and opacity sliders).

use std::fmt::Debug;
use std::hash::Hash;

use eframe::egui;

use crate::colormap::{ColormapId, sample_colormap_srgb};

/// Outcome of tracking a drag-editable control (`Slider`, `DragValue`, etc.)
/// so dragging previews continuously without thrashing `session.json` and a
/// no-op click/release without value changes never emits a spurious commit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DragCommit {
    None,
    Preview,
    Commit,
}

/// Track whether a drag gesture actually mutated a value before committing on
/// release. Persists a single `bool` in `egui` temporary memory keyed by
/// `dirty_id`.
pub(crate) fn track_drag_commit(
    ui: &mut egui::Ui,
    dirty_id: egui::Id,
    changed: bool,
    dragged: bool,
    drag_stopped: bool,
) -> DragCommit {
    if changed {
        if dragged {
            ui.data_mut(|d| d.insert_temp(dirty_id, true));
            DragCommit::Preview
        } else {
            ui.data_mut(|d| d.remove::<bool>(dirty_id));
            DragCommit::Commit
        }
    } else if drag_stopped {
        let was_dirty = ui.data(|d| d.get_temp::<bool>(dirty_id)).unwrap_or(false);
        ui.data_mut(|d| d.remove::<bool>(dirty_id));
        if was_dirty {
            DragCommit::Commit
        } else {
            DragCommit::None
        }
    } else {
        DragCommit::None
    }
}

/// Paint a horizontal linear colormap gradient strip cleanly clipped to `rect`.
pub(crate) fn paint_colormap_ramp(
    painter: &egui::Painter,
    rect: egui::Rect,
    colormap: ColormapId,
    invert: bool,
    steps: usize,
) {
    let steps = steps.max(2);
    let seg_w = rect.width() / (steps as f32);
    for i in 0..steps {
        let t = (i as f32) / ((steps - 1) as f32);
        let [r, g, b, _] = sample_colormap_srgb(colormap, t, invert);
        let x0 = rect.left() + (i as f32) * seg_w;
        let x1 = (rect.left() + ((i + 1) as f32) * seg_w + 0.5).min(rect.right());
        let seg =
            egui::Rect::from_min_max(egui::pos2(x0, rect.top()), egui::pos2(x1, rect.bottom()));
        painter.rect_filled(seg, 0.0, egui::Color32::from_rgb(r, g, b));
    }
}

/// Render a compact opacity slider returning `Some((opacity, commit))` only
/// when the value changed or a mutating drag finished.
pub(crate) fn show_opacity_slider(
    ui: &mut egui::Ui,
    id_salt: impl Hash + Debug,
    current: f32,
) -> Option<(f32, bool)> {
    let mut opacity = current;
    let mut update = None;
    let dirty_id = ui.scope_id().with(("opacity_drag_dirty", id_salt));
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("Opacity").small());
        let resp = ui.add(egui::Slider::new(&mut opacity, 0.0..=1.0).show_value(false));
        match track_drag_commit(
            ui,
            dirty_id,
            resp.changed(),
            resp.dragged(),
            resp.drag_stopped(),
        ) {
            DragCommit::None => {}
            DragCommit::Preview => update = Some((opacity, false)),
            DragCommit::Commit => update = Some((opacity, true)),
        }
        ui.monospace(egui::RichText::new(format!("{:.0}%", opacity * 100.0)).small());
    });
    update
}
