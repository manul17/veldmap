//! Presentation-only floating Geodesic Measure window bound to
//! [`crate::layers::MeasureSession`].

use eframe::egui;

use crate::layers::{MeasureMode, MeasureSession};

use super::format;

/// Action emitted by the Measure tool window.
#[derive(Clone, Debug)]
pub(crate) enum MeasureAction {
    ExportGeoJson,
}

/// UI wrapper pairing the domain [`MeasureSession`] with transient export
/// status copy.
#[derive(Default)]
pub(crate) struct MeasureUiState {
    pub(crate) session: MeasureSession,
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

/// Render the floating Geodesic Measure window when active.
pub(crate) fn show_measure_window(
    ctx: &egui::Context,
    state: &mut MeasureUiState,
) -> Option<MeasureAction> {
    if !state.session.is_active() {
        return None;
    }
    let mut action = None;
    let mut open = true;
    egui::Window::new("Geodesic Measure (WGS84 Karney)")
        .open(&mut open)
        .resizable(false)
        .default_width(300.0)
        .show(ctx, |ui| {
            let mode = state.session.mode();
            ui.horizontal(|ui| {
                if ui
                    .selectable_label(mode == MeasureMode::Distance, "Distance")
                    .clicked()
                {
                    state.session.set_mode(MeasureMode::Distance);
                }
                if ui
                    .selectable_label(mode == MeasureMode::Area, "Area")
                    .clicked()
                {
                    state.session.set_mode(MeasureMode::Area);
                }
            });
            ui.separator();
            ui.weak("Click on the globe to place vertices.");
            let point_count = state.session.points().len();
            ui.monospace(format!("Vertices: {point_count}"));

            let metrics = state.session.metrics();
            match state.session.mode() {
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
                    state.session.undo();
                    state.clear_notice();
                }
                if ui
                    .add_enabled(point_count > 0, egui::Button::new("Clear"))
                    .clicked()
                {
                    state.session.clear();
                    state.clear_notice();
                }
                if ui
                    .add_enabled(point_count > 0, egui::Button::new("Export GeoJSON…"))
                    .clicked()
                {
                    action = Some(MeasureAction::ExportGeoJson);
                }
            });

            if let Some(note) = &state.notice {
                ui.separator();
                ui.weak(note);
            }
        });

    if !open {
        state.session.set_mode(MeasureMode::Off);
    }
    action
}
