//! Area of Interest (AOI) capture, draw-box toggle, GeoJSON import, and
//! GeoJSON/WKT export controls inside the Cloud / STAC panel.

use eframe::egui;

use crate::stac::StacViewSnapshot;
use crate::ui::WARN_TEXT_COLOR;
use crate::vectors::VectorLayerSummary;

use super::{StacUiAction, StacUiState};

pub(super) fn show_aoi_section(
    ui: &mut egui::Ui,
    stac: &StacViewSnapshot<'_>,
    vectors: &[VectorLayerSummary],
    draw_aoi_active: bool,
    state: &mut StacUiState,
    actions: &mut Vec<StacUiAction>,
) {
    ui.strong("Area of Interest (AOI)");
    ui.horizontal_wrapped(|ui| {
        if ui
            .small_button("Viewport")
            .on_hover_text("Use current visible globe extent as STAC search AOI")
            .clicked()
        {
            state.clear_copy_notice();
            actions.push(StacUiAction::CaptureViewportAoi);
        }
        if ui
            .selectable_label(draw_aoi_active, "Draw Box")
            .on_hover_text(
                "Click and drag on the globe to draw a search AOI rectangle (Esc cancels)",
            )
            .clicked()
        {
            state.clear_copy_notice();
            actions.push(StacUiAction::ToggleDrawAoiBox);
        }
        if vectors.iter().any(|v| v.has_polygons) {
            ui.menu_button("From GeoJSON", |ui| {
                for vec_layer in vectors.iter().filter(|v| v.has_polygons) {
                    if ui.button(&vec_layer.name).clicked() {
                        state.clear_copy_notice();
                        actions.push(StacUiAction::UseVectorAoi(vec_layer.id));
                        ui.close();
                    }
                }
            });
        }
        if stac.has_aoi
            && ui
                .small_button("Clear")
                .on_hover_text("Clear active search AOI")
                .clicked()
        {
            state.clear_copy_notice();
            actions.push(StacUiAction::ClearAoi);
        }
    });

    if let Some(bbox) = stac.aoi_bbox {
        let label = stac.aoi_label.unwrap_or("Custom AOI");
        ui.small(format!(
            "{label}: [{:.3}, {:.3}] → [{:.3}, {:.3}]",
            bbox[0], bbox[1], bbox[2], bbox[3]
        ));
        ui.horizontal(|ui| {
            if ui
                .small_button("Copy GeoJSON")
                .on_hover_text("Copy active AOI polygon as RFC 7946 GeoJSON")
                .clicked()
            {
                actions.push(StacUiAction::CopyAoiGeoJson);
            }
            if ui
                .small_button("Copy WKT")
                .on_hover_text("Copy active AOI polygon as Well-Known Text (WKT)")
                .clicked()
            {
                actions.push(StacUiAction::CopyAoiWkt);
            }
        });
        if let Some(notice) = &state.copy_notice {
            ui.weak(egui::RichText::new(notice).small());
        }
    } else if draw_aoi_active {
        ui.colored_label(
            WARN_TEXT_COLOR,
            egui::RichText::new("Drag on the globe to draw an AOI box (Esc to cancel).").small(),
        );
    } else {
        ui.weak(
            egui::RichText::new("No AOI set — click Viewport or Draw Box to bound search.").small(),
        );
    }
}
