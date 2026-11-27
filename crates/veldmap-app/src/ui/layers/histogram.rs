//! Interactive scientific band histogram widget with draggable min/max stretch
//! handles and live colormap ramp preview.

use eframe::egui;

use crate::colormap::sample_colormap_srgb;
use crate::layers::{LayerId, OverlayEdit, ScientificSummary};
use crate::ui::widgets::paint_colormap_ramp;

use super::LayerUiAction;

#[derive(Clone, Copy)]
enum HistHandle {
    Min,
    Max,
}

#[derive(Clone, Copy)]
struct HistDragState {
    handle: HistHandle,
    changed: bool,
}

fn nearest_hist_handle(val: f64, sci_min: f64, sci_max: f64) -> HistHandle {
    if (val - sci_min).abs() <= (val - sci_max).abs() {
        HistHandle::Min
    } else {
        HistHandle::Max
    }
}

struct HistogramDomain<'a> {
    bins: &'a [u32],
    max_count: u32,
    data_min: f64,
    data_span: f64,
}

fn paint_histogram_canvas(
    painter: &egui::Painter,
    hist_rect: egui::Rect,
    ramp_rect: egui::Rect,
    domain: &HistogramDomain<'_>,
    sci: &ScientificSummary,
) {
    painter.rect_filled(hist_rect, 2.0, egui::Color32::from_gray(24));

    let val_to_x = |v: f64| -> f32 {
        let f = ((v - domain.data_min) / domain.data_span).clamp(0.0, 1.0) as f32;
        hist_rect.left() + f * hist_rect.width()
    };

    let hist_h = hist_rect.height();
    let bin_w = hist_rect.width() / (domain.bins.len() as f32);
    let stretch_span = (sci.max - sci.min).max(1e-12);
    for (i, &count) in domain.bins.iter().enumerate() {
        if count == 0 {
            continue;
        }
        let h_frac = (f64::from(count) / f64::from(domain.max_count)).sqrt() as f32;
        let bar_h = (h_frac * (hist_h - 2.0)).max(1.0);
        let x0 = hist_rect.left() + (i as f32) * bin_w;
        let x1 = (x0 + bin_w - 0.5).max(x0 + 1.0);
        let bin_val =
            domain.data_min + ((i as f64) + 0.5) / (domain.bins.len() as f64) * domain.data_span;
        let t = ((bin_val - sci.min) / stretch_span).clamp(0.0, 1.0) as f32;
        let [r, g, b, _] = sample_colormap_srgb(sci.colormap, t, sci.invert_colormap);
        let bar_rect = egui::Rect::from_min_max(
            egui::pos2(x0, hist_rect.bottom() - bar_h),
            egui::pos2(x1, hist_rect.bottom()),
        );
        painter.rect_filled(bar_rect, 0.0, egui::Color32::from_rgb(r, g, b));
    }

    let x_lo = val_to_x(sci.min);
    let x_hi = val_to_x(sci.max);
    let marker_stroke = egui::Stroke::new(1.5, egui::Color32::from_rgb(245, 245, 245));
    painter.line_segment(
        [
            egui::pos2(x_lo, hist_rect.top()),
            egui::pos2(x_lo, hist_rect.bottom()),
        ],
        marker_stroke,
    );
    painter.line_segment(
        [
            egui::pos2(x_hi, hist_rect.top()),
            egui::pos2(x_hi, hist_rect.bottom()),
        ],
        marker_stroke,
    );

    paint_colormap_ramp(painter, ramp_rect, sci.colormap, sci.invert_colormap, 48);
    painter.rect_stroke(
        hist_rect,
        2.0,
        egui::Stroke::new(1.0, egui::Color32::from_gray(70)),
        egui::StrokeKind::Inside,
    );
}

fn handle_histogram_interaction(
    ui: &mut egui::Ui,
    response: egui::Response,
    hist_rect: egui::Rect,
    layer_id: LayerId,
    domain: &HistogramDomain<'_>,
    sci: &ScientificSummary,
    actions: &mut Vec<LayerUiAction>,
) {
    let state_id = response.id.with("hist_drag_state");
    let x_to_val = |x: f32| -> f64 {
        let f = f64::from(((x - hist_rect.left()) / hist_rect.width()).clamp(0.0, 1.0));
        domain.data_min + f * domain.data_span
    };

    if response.drag_started()
        && let Some(pos) = response.interact_pointer_pos()
    {
        let handle = nearest_hist_handle(x_to_val(pos.x), sci.min, sci.max);
        ui.data_mut(|d| {
            d.insert_temp(
                state_id,
                HistDragState {
                    handle,
                    changed: false,
                },
            )
        });
    }

    if (response.dragged() || response.clicked())
        && let Some(pos) = response.interact_pointer_pos()
    {
        let val = x_to_val(pos.x);
        let handle = ui
            .data(|d| d.get_temp::<HistDragState>(state_id).map(|s| s.handle))
            .unwrap_or_else(|| nearest_hist_handle(val, sci.min, sci.max));
        let (next_lo, next_hi) = match handle {
            HistHandle::Min => (val.min(sci.max - domain.data_span * 1e-4), sci.max),
            HistHandle::Max => (sci.min, val.max(sci.min + domain.data_span * 1e-4)),
        };
        let commit = response.clicked() || response.drag_stopped();
        if response.dragged() {
            ui.data_mut(|d| {
                d.insert_temp(
                    state_id,
                    HistDragState {
                        handle,
                        changed: true,
                    },
                )
            });
        }
        actions.push(LayerUiAction::edit_raster_drag(
            layer_id,
            OverlayEdit::SetStretchRange {
                min: next_lo,
                max: next_hi,
            },
            commit,
        ));
    } else if response.drag_stopped() {
        let was_changed = ui.data(|d| {
            d.get_temp::<HistDragState>(state_id)
                .is_some_and(|s| s.changed)
        });
        if was_changed {
            actions.push(LayerUiAction::edit_raster_drag(
                layer_id,
                OverlayEdit::SetStretchRange {
                    min: sci.min,
                    max: sci.max,
                },
                true,
            ));
        }
    }

    if response.drag_stopped() {
        ui.data_mut(|d| d.remove::<HistDragState>(state_id));
    }
    response.on_hover_text("Click or drag to adjust stretch min/max");
}

pub(super) fn paint_interactive_histogram(
    ui: &mut egui::Ui,
    layer_id: LayerId,
    sci: &ScientificSummary,
    actions: &mut Vec<LayerUiAction>,
) {
    let Some(ref stats) = sci.stats else {
        if sci.stats_loading {
            ui.weak(egui::RichText::new("Sampling band statistics…").small());
        }
        return;
    };

    let (bins, data_min, data_max) = if sci.preset.is_log10_db() {
        let (Some(lo), Some(hi)) = (stats.db_min, stats.db_max) else {
            ui.weak(egui::RichText::new("No positive samples for dB histogram").small());
            return;
        };
        (&stats.db_histogram, lo, hi)
    } else {
        (&stats.histogram, stats.min, stats.max)
    };
    let max_count = bins.iter().copied().max().unwrap_or(0);
    if max_count == 0 {
        return;
    }

    let width = ui.available_width().max(120.0);
    let total_h = 54.0;
    let ramp_h = 8.0;
    let hist_h = total_h - ramp_h - 2.0;
    let (rect, response) =
        ui.allocate_exact_size(egui::vec2(width, total_h), egui::Sense::click_and_drag());

    let hist_rect = egui::Rect::from_min_size(rect.min, egui::vec2(width, hist_h));
    let ramp_rect = egui::Rect::from_min_size(
        egui::pos2(rect.left(), hist_rect.bottom() + 2.0),
        egui::vec2(width, ramp_h),
    );
    let domain = HistogramDomain {
        bins,
        max_count,
        data_min,
        data_span: (data_max - data_min).max(1e-12),
    };

    paint_histogram_canvas(ui.painter(), hist_rect, ramp_rect, &domain, sci);
    handle_histogram_interaction(ui, response, hist_rect, layer_id, &domain, sci, actions);
}
