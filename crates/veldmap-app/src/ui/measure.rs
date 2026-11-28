//! Presentation-only floating Geodesic Measure window driven by
//! [`crate::measure::MeasureSummary`].
//!
//! Strictly action-only: emits [`MeasureAction`] values back to `VeldmapApp`
//! without mutating or borrowing `MeasureSession` directly from UI layout code.

use eframe::egui;

use crate::measure::{MeasureMode, MeasureSummary};

use super::format;

/// Action emitted by the Measure tool window and applied by `VeldmapApp`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MeasureAction {
    SetMode(MeasureMode),
    Undo,
    Clear,
    Close,
    ExportGeoJson,
    UseAsStacAoi,
}

/// Transient UI state for the Geodesic Measure window (export status notice).
#[derive(Default)]
pub(crate) struct MeasureUiState {
    notice: Option<String>,
}

impl MeasureUiState {
    pub(crate) fn set_notice(&mut self, msg: String) {
        self.notice = Some(msg);
    }

    pub(crate) fn clear_notice(&mut self) {
        self.notice = None;
    }
}

/// Render the floating Geodesic Measure window when `summary.active` is `true`.
pub(crate) fn show_measure_window(
    ctx: &egui::Context,
    summary: MeasureSummary,
    state: &MeasureUiState,
) -> Option<MeasureAction> {
    if !summary.active {
        return None;
    }
    let mut action = None;
    let mut open = true;
    egui::Window::new("Geodesic Measure (WGS84 Karney)")
        .open(&mut open)
        .resizable(false)
        .default_width(300.0)
        .show(ctx, |ui| {
            let mode = summary.mode;
            ui.horizontal(|ui| {
                if ui
                    .selectable_label(mode == MeasureMode::Distance, "Distance")
                    .clicked()
                    && mode != MeasureMode::Distance
                {
                    action = Some(MeasureAction::SetMode(MeasureMode::Distance));
                }
                if ui
                    .selectable_label(mode == MeasureMode::Area, "Area")
                    .clicked()
                    && mode != MeasureMode::Area
                {
                    action = Some(MeasureAction::SetMode(MeasureMode::Area));
                }
            });
            ui.separator();
            ui.weak("Click on the globe to place vertices.");
            let point_count = summary.point_count;
            ui.monospace(format!("Vertices: {point_count}"));

            let metrics = summary.metrics.unwrap_or_default();
            match mode {
                MeasureMode::Distance => {
                    ui.strong(format!(
                        "Length: {}",
                        format::format_distance_m(metrics.length_m)
                    ));
                }
                MeasureMode::Area => {
                    if point_count >= 3 {
                        ui.strong(format!(
                            "Area: {}",
                            format::format_area_m2(metrics.polygon.area_m2)
                        ));
                        ui.monospace(format!(
                            "Perimeter: {}",
                            format::format_distance_m(metrics.polygon.perimeter_m)
                        ));
                    } else {
                        ui.monospace(format!(
                            "Length: {}",
                            format::format_distance_m(metrics.length_m)
                        ));
                        ui.weak("Add at least 3 points for area.");
                    }
                }
                MeasureMode::Off => {}
            }

            ui.add_space(6.0);
            ui.horizontal(|ui| {
                if ui
                    .add_enabled(point_count > 0, egui::Button::new("Undo"))
                    .clicked()
                {
                    action = Some(MeasureAction::Undo);
                }
                if ui
                    .add_enabled(point_count > 0, egui::Button::new("Clear"))
                    .clicked()
                {
                    action = Some(MeasureAction::Clear);
                }
                if ui
                    .add_enabled(point_count > 0, egui::Button::new("Export GeoJSON…"))
                    .clicked()
                {
                    action = Some(MeasureAction::ExportGeoJson);
                }
            });
            if mode == MeasureMode::Area && point_count >= 3 {
                ui.add_space(4.0);
                if ui
                    .button("Use as STAC AOI")
                    .on_hover_text("Promote this measured polygon into the Cloud / STAC search AOI")
                    .clicked()
                {
                    action = Some(MeasureAction::UseAsStacAoi);
                }
            }

            if let Some(note) = &state.notice {
                ui.separator();
                ui.weak(note);
            }
        });

    if !open {
        action = Some(MeasureAction::Close);
    }
    action
}
