//! Raster overlay layer card rendering inside the Layer List panel.

use eframe::egui;

use crate::format;
use crate::layers::{LayerSummary, LayerSummaryKind, OverlayEdit};
use crate::ui::widgets::{paint_loading_shimmer, show_opacity_slider};
use crate::ui::{ERROR_TEXT_COLOR, WARN_TEXT_COLOR};

use super::LayerUiAction;
use super::band_math_card::show_band_math_controls;
use super::style_controls::{show_categorical_controls, show_scientific_controls};
use super::timeline_card::show_timeline_controls;

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

    match &layer.kind {
        LayerSummaryKind::SingleRaster(single) => {
            if single.band_options.len() > 1 {
                let current_label = single
                    .band_options
                    .iter()
                    .find(|opt| opt.spec == single.active_spec)
                    .map_or("Display (RGB)", |opt| opt.label.as_str());
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Band").small());
                    egui::ComboBox::from_id_salt(("band_select", layer.id))
                        .selected_text(egui::RichText::new(current_label).small())
                        .width(172.0)
                        .show_ui(ui, |ui| {
                            for opt in &single.band_options {
                                let selected = opt.spec == single.active_spec;
                                if ui.selectable_label(selected, &opt.label).clicked() && !selected
                                {
                                    actions.push(LayerUiAction::SetStream(layer.id, opt.spec));
                                }
                            }
                        });
                });
            }

            if let Some(ref sci) = single.scientific {
                show_scientific_controls(ui, layer.id, sci, actions);
            } else if let Some(cat) = single.categorical {
                show_categorical_controls(ui, layer.id, cat, actions);
            }
        }
        LayerSummaryKind::BandMath(bm) => {
            show_band_math_controls(ui, layer.id, bm, actions);
        }
        LayerSummaryKind::Timeline(tl) => {
            show_timeline_controls(ui, layer.id, tl, actions);
        }
    }

    egui::CollapsingHeader::new(egui::RichText::new("CRS & Metadata").small())
        .id_salt(("meta", layer.id))
        .default_open(false)
        .show(ui, |ui| {
            let has_crs_override = matches!(
                &layer.kind,
                LayerSummaryKind::SingleRaster(s) if s.has_crs_override
            );
            let crs_tag = if has_crs_override {
                format!("CRS: {} (override)", layer.crs_label)
            } else {
                format!("CRS: {}", layer.crs_label)
            };
            ui.small(crs_tag);
            ui.small(format!("Radiometry: {}", layer.sample_info));
            if let LayerSummaryKind::SingleRaster(single) = &layer.kind
                && let Some(ref sci) = single.scientific
            {
                if let Some(ref units) = sci.units {
                    ui.small(format!("Units: {units}"));
                }
                if sci.has_affine_scaling() {
                    ui.small(format!("Scale: {} · Offset: {}", sci.scale, sci.offset));
                }
            }
            if let Some(ref url) = layer.remote_url {
                ui.small(format!("URL: {url}"));
            } else {
                match &layer.kind {
                    LayerSummaryKind::SingleRaster(single)
                        if !single.path.as_os_str().is_empty() =>
                    {
                        ui.small(format!("Path: {}", single.path.display()));
                    }
                    LayerSummaryKind::BandMath(bm) => {
                        let inputs = bm
                            .inputs
                            .iter()
                            .map(|i| i.label.as_str())
                            .collect::<Vec<_>>()
                            .join(", ");
                        ui.small(format!("Sources: {inputs}"));
                    }
                    LayerSummaryKind::Timeline(tl) => {
                        ui.small(format!("Epochs: {}", tl.epochs.len()));
                    }
                    LayerSummaryKind::SingleRaster(_) => {}
                }
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
        if let LayerSummaryKind::SingleRaster(single) = &layer.kind {
            if !layer.is_remote
                && single.can_override_crs
                && ui
                    .small_button("CRS…")
                    .on_hover_text("Override CRS or custom Transverse Mercator parameters")
                    .clicked()
            {
                actions.push(LayerUiAction::OpenCrsPicker(layer.id, layer.name.clone()));
            }
            if !layer.is_remote
                && ui
                    .add_enabled(!single.reloading, egui::Button::new("Reload").small())
                    .on_hover_text("Reload from disk (bypass fingerprint & tile cache)")
                    .clicked()
            {
                actions.push(LayerUiAction::ReloadBypassCache(layer.id));
            }
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
                    .or_else(|| match &layer.kind {
                        LayerSummaryKind::SingleRaster(s) if !s.path.as_os_str().is_empty() => {
                            Some(s.path.display().to_string())
                        }
                        _ => None,
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

            let px_label = match (layer.dimensions(), layer.source_pixel_size_m) {
                (Some((w, h)), Some((mx, my))) => {
                    format!("{w}×{h} · {}", format::format_source_pixel_size(mx, my))
                }
                (Some((w, h)), None) => format!("{w}×{h}"),
                (None, Some((mx, my))) => format::format_source_pixel_size(mx, my),
                (None, None) => String::new(),
            };
            let reloading = layer.is_reloading();
            let is_loading =
                reloading || (layer.visible && (layer.pass_active || layer.in_flight > 0));
            ui.horizontal(|ui| {
                if let LayerSummaryKind::BandMath(ref bm) = layer.kind {
                    ui.colored_label(
                        crate::ui::BADGE_RECIPE_COLOR,
                        egui::RichText::new(&bm.recipe_label).small().strong(),
                    );
                } else if let LayerSummaryKind::Timeline(ref tl) = layer.kind {
                    ui.colored_label(
                        crate::ui::BADGE_RECIPE_COLOR,
                        egui::RichText::new(format!("Timeline ({})", tl.epochs.len()))
                            .small()
                            .strong(),
                    );
                }
                if layer.is_remote {
                    ui.colored_label(
                        crate::ui::BADGE_CLOUD_COLOR,
                        egui::RichText::new("Cloud COG").small().strong(),
                    );
                }
                if !px_label.is_empty() {
                    ui.weak(egui::RichText::new(px_label).small());
                }
                if reloading {
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
