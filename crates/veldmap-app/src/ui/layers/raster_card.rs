//! Raster overlay layer card rendering inside the Layer List panel.

use eframe::egui;

use crate::format;
use crate::layers::{LayerSummary, OverlayEdit};
use crate::ui::widgets::{paint_loading_shimmer, show_opacity_slider};
use crate::ui::{ERROR_TEXT_COLOR, WARN_TEXT_COLOR};

use super::LayerUiAction;
use super::band_math_card::show_band_math_controls;
use super::style_controls::{show_categorical_controls, show_scientific_controls};

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

    if let Some(ref bm) = layer.band_math {
        show_band_math_controls(ui, layer.id, bm, actions);
    } else if let Some(ref sci) = layer.scientific {
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
                if sci.has_affine_scaling() {
                    ui.small(format!("Scale: {} · Offset: {}", sci.scale, sci.offset));
                }
            }
            if let Some(ref url) = layer.remote_url {
                ui.small(format!("URL: {url}"));
            } else if !layer.path.as_os_str().is_empty() {
                ui.small(format!("Path: {}", layer.path.display()));
            } else if let Some(ref bm) = layer.band_math {
                let inputs = bm
                    .inputs
                    .iter()
                    .map(|i| i.label.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                ui.small(format!("Sources: {inputs}"));
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
            && layer.band_math.is_none()
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
                    .or_else(|| {
                        (!layer.path.as_os_str().is_empty())
                            .then(|| layer.path.display().to_string())
                    })
                    .unwrap_or_else(|| layer.sample_info.clone());
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
                if let Some(ref bm) = layer.band_math {
                    ui.colored_label(
                        crate::ui::BADGE_RECIPE_COLOR,
                        egui::RichText::new(&bm.recipe_label).small().strong(),
                    );
                }
                if layer.is_remote {
                    ui.colored_label(
                        crate::ui::BADGE_CLOUD_COLOR,
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
