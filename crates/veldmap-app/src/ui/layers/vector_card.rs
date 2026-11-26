//! GeoJSON vector overlay card rendering inside the Layer List panel.

use eframe::egui;

use crate::ui::widgets::show_opacity_slider;
use crate::vectors::VectorLayerSummary;

use super::LayerUiAction;

pub(super) fn show_vector_card(
    ui: &mut egui::Ui,
    layer: &VectorLayerSummary,
    actions: &mut Vec<LayerUiAction>,
) {
    ui.push_id(("vec", layer.id), |ui| {
        ui.group(|ui| {
            ui.horizontal(|ui| {
                let mut visible = layer.visible;
                if ui.checkbox(&mut visible, "").changed() {
                    actions.push(LayerUiAction::ToggleVectorVisibility(layer.id, visible));
                }
                let swatch = egui::Color32::from_rgb(
                    layer.color_rgb[0],
                    layer.color_rgb[1],
                    layer.color_rgb[2],
                );
                let (rect, _) =
                    ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
                ui.painter().rect_filled(rect, 2.0, swatch);
                ui.add(egui::Label::new(egui::RichText::new(&layer.name).strong()).truncate())
                    .on_hover_text(layer.path.display().to_string());
            });

            let summary_text = if layer.point_count > 0 {
                format!(
                    "GeoJSON · {} features ({} tris, {} segs, {} pts)",
                    layer.feature_count,
                    layer.triangle_count,
                    layer.line_segment_count,
                    layer.point_count
                )
            } else {
                format!(
                    "GeoJSON · {} features ({} tris, {} segs)",
                    layer.feature_count, layer.triangle_count, layer.line_segment_count
                )
            };
            ui.weak(egui::RichText::new(summary_text).small());

            if let Some((opacity, commit)) =
                show_opacity_slider(ui, ("vector", layer.id), layer.opacity)
            {
                actions.push(LayerUiAction::SetVectorOpacity(layer.id, opacity, commit));
            }

            ui.horizontal_wrapped(|ui| {
                if let Some(bounds) = layer.footprint
                    && ui
                        .small_button("Zoom")
                        .on_hover_text("Fly camera to vector layer extent")
                        .clicked()
                {
                    actions.push(LayerUiAction::ZoomToExtent(bounds));
                }
                if layer.has_polygons
                    && ui
                        .small_button("STAC AOI")
                        .on_hover_text("Use this polygon layer as the Cloud / STAC search AOI")
                        .clicked()
                {
                    actions.push(LayerUiAction::UseVectorAsStacAoi(layer.id));
                }
                if ui.small_button("Remove").clicked() {
                    actions.push(LayerUiAction::RemoveVector(layer.id));
                }
            });
        });
    });
    ui.add_space(4.0);
}
