//! Raster overlay layer card rendering inside the Layer List panel.

use eframe::egui;

use crate::layers::{LayerSummary, OverlayEdit};
use crate::ui::widgets::show_opacity_slider;
use crate::ui::{ERROR_TEXT_COLOR, WARN_TEXT_COLOR, format};

use super::LayerUiAction;
use super::style_controls::{show_categorical_controls, show_scientific_controls};

fn paint_loading_shimmer(ui: &mut egui::Ui) {
    let width = ui.available_width().max(40.0);
    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, 3.0), egui::Sense::hover());
    let t = ui.input(|i| i.time) as f32;
    let phase = (t * 0.9) % 1.0;
    let bar_w = (rect.width() * 0.35).max(16.0);
    let x0 = rect.left() - bar_w + phase * (rect.width() + bar_w);
    let painter = ui.painter();
    painter.rect_filled(rect, 1.5, egui::Color32::from_gray(50));
    let highlight = egui::Rect::from_min_max(
        egui::pos2(x0.max(rect.left()), rect.top()),
        egui::pos2((x0 + bar_w).min(rect.right()), rect.bottom()),
    );
    if highlight.width() > 0.0 {
        painter.rect_filled(highlight, 1.5, egui::Color32::from_rgb(80, 175, 245));
    }
    ui.ctx()
        .request_repaint_after(std::time::Duration::from_millis(33));
}

fn show_layer_status_banners(ui: &mut egui::Ui, layer: &LayerSummary) {
    if layer.detail_limit_level > 0 {
        ui.colored_label(
            WARN_TEXT_COLOR,
            egui::RichText::new(format!(
                "Detail limited to level {} (memory budget)",
                layer.detail_limit_level
            ))
            .small(),
        );
    }

    if let Some(warn) = &layer.datum_warning {
        ui.colored_label(WARN_TEXT_COLOR, egui::RichText::new(warn).small());
    }

    if let Some(err) = &layer.last_error {
        ui.colored_label(
            ERROR_TEXT_COLOR,
            egui::RichText::new(format!("Partial/reload error: {err}")).small(),
        );
    }
}

fn show_layer_controls(ui: &mut egui::Ui, layer: &LayerSummary, actions: &mut Vec<LayerUiAction>) {
    if let Some((opacity, commit)) = show_opacity_slider(ui, ("raster", layer.id), layer.opacity) {
        actions.push(LayerUiAction::edit_raster_drag(
            layer.id,
            OverlayEdit::SetOpacity(opacity),
            commit,
        ));
    }

    if layer.band_options.len() > 1 {
        let current_label = layer
            .band_options
            .iter()
            .find(|opt| opt.spec == layer.active_spec)
            .map_or("Display (RGB)", |opt| opt.label.as_str());
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Band").small());
            egui::ComboBox::from_id_salt(("band_select", layer.id))
                .selected_text(egui::RichText::new(current_label).small())
                .width(172.0)
                .show_ui(ui, |ui| {
                    for opt in &layer.band_options {
                        let selected = opt.spec == layer.active_spec;
                        if ui.selectable_label(selected, &opt.label).clicked() && !selected {
                            actions.push(LayerUiAction::SetStream(layer.id, opt.spec));
                        }
                    }
                });
        });
    }

    if let Some(ref sci) = layer.scientific {
        show_scientific_controls(ui, layer.id, sci, actions);
    } else if let Some(cat) = layer.categorical {
        show_categorical_controls(ui, layer.id, cat, actions);
    }

    egui::CollapsingHeader::new(egui::RichText::new("CRS & Metadata").small())
        .id_salt(("meta", layer.id))
        .default_open(false)
        .show(ui, |ui| {
            let crs_tag = if layer.has_crs_override {
                format!("CRS: {} (override)", layer.crs_label)
            } else {
                format!("CRS: {}", layer.crs_label)
            };
            ui.small(crs_tag);
            ui.small(format!("Radiometry: {}", layer.sample_info));
            if let Some(ref sci) = layer.scientific {
                if let Some(ref units) = sci.units {
                    ui.small(format!("Units: {units}"));
                }
                if (sci.scale - 1.0).abs() > 1e-12 || sci.offset.abs() > 1e-12 {
                    ui.small(format!("Scale: {} · Offset: {}", sci.scale, sci.offset));
                }
            }
            if let Some(ref url) = layer.remote_url {
                ui.small(format!("URL: {url}"));
            } else {
                ui.small(format!("Path: {}", layer.path.display()));
            }
        });
}

fn show_layer_actions(ui: &mut egui::Ui, layer: &LayerSummary, actions: &mut Vec<LayerUiAction>) {
    ui.horizontal_wrapped(|ui| {
        if let Some(bounds) = layer.footprint
            && ui
                .small_button("Zoom")
                .on_hover_text("Fly camera to layer extent")
                .clicked()
        {
            actions.push(LayerUiAction::ZoomToExtent(bounds));
        }
        if !layer.is_remote
            && layer.can_override_crs
            && ui
                .small_button("CRS…")
                .on_hover_text("Override CRS or custom Transverse Mercator parameters")
                .clicked()
        {
            actions.push(LayerUiAction::OpenCrsPicker(layer.id, layer.name.clone()));
        }
        if !layer.is_remote
            && ui
                .add_enabled(!layer.reloading, egui::Button::new("Reload").small())
                .on_hover_text("Reload from disk (bypass fingerprint & tile cache)")
                .clicked()
        {
            actions.push(LayerUiAction::ReloadBypassCache(layer.id));
        }
        if ui
            .small_button("Remove")
            .on_hover_text("Remove layer and free VRAM")
            .clicked()
        {
            actions.push(LayerUiAction::edit_raster(layer.id, OverlayEdit::Remove));
        }
    });
}

pub(super) fn show_raster_card(
    ui: &mut egui::Ui,
    layer: &LayerSummary,
    actions: &mut Vec<LayerUiAction>,
) {
    ui.push_id(layer.id, |ui| {
        ui.group(|ui| {
            ui.horizontal(|ui| {
                let mut visible = layer.visible;
                if ui.checkbox(&mut visible, "").changed() {
                    actions.push(LayerUiAction::edit_raster(
                        layer.id,
                        OverlayEdit::SetVisible(visible),
                    ));
                }
                let hover_tip = layer
                    .remote_url
                    .clone()
                    .unwrap_or_else(|| layer.path.display().to_string());
                ui.add(egui::Label::new(egui::RichText::new(&layer.name).strong()).truncate())
                    .on_hover_text(hover_tip);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .add_enabled(layer.can_move_down, egui::Button::new("↓").small())
                        .on_hover_text("Move layer down")
                        .clicked()
                    {
                        actions.push(LayerUiAction::edit_raster(layer.id, OverlayEdit::MoveDown));
                    }
                    if ui
                        .add_enabled(layer.can_move_up, egui::Button::new("↑").small())
                        .on_hover_text("Move layer up")
                        .clicked()
                    {
                        actions.push(LayerUiAction::edit_raster(layer.id, OverlayEdit::MoveUp));
                    }
                });
            });

            let px_label = match layer.source_pixel_size_m {
                Some((mx, my)) => format!(
                    "{}×{} · {}",
                    layer.width,
                    layer.height,
                    format::format_source_pixel_size(mx, my)
                ),
                None => format!("{}×{}", layer.width, layer.height),
            };
            let is_loading =
                layer.reloading || (layer.visible && (layer.pass_active || layer.in_flight > 0));
            ui.horizontal(|ui| {
                if layer.is_remote {
                    ui.colored_label(
                        egui::Color32::from_rgb(100, 195, 255),
                        egui::RichText::new("Cloud COG").small().strong(),
                    );
                }
                ui.weak(egui::RichText::new(px_label).small());
                if layer.reloading {
                    ui.weak(egui::RichText::new("· reloading…").small());
                } else if is_loading {
                    ui.weak(egui::RichText::new("· loading…").small());
                }
            });

            if is_loading {
                paint_loading_shimmer(ui);
            }

            show_layer_status_banners(ui, layer);
            show_layer_controls(ui, layer, actions);
            show_layer_actions(ui, layer, actions);
        });
    });
    ui.add_space(4.0);
}
