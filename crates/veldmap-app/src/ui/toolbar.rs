//! Top workstation toolbar (Go-To navigation, basemap selector, graticule
//! toggle, GPU adapter badge) and bottom GIS status bar extracted from `app.rs`.
//!
//! Strictly action-only: emits [`ToolbarAction`] values back to `VeldmapApp`
//! without mutating `Camera`, storage, or scene state directly from UI layout
//! code.

use eframe::egui;
use veldmap_core::geo::camera::Camera;
use veldmap_core::geo::coords::{ParsedLocation, parse_location};

use crate::basemap::BasemapStatusSnapshot;
use crate::settings::AppSettings;

use super::ERROR_TEXT_COLOR;
use super::basemap::{BasemapUiAction, BasemapUiState, show_toolbar_controls};
use super::hud;

/// User intent emitted by the top workstation toolbar and applied by
/// `VeldmapApp`.
#[derive(Clone, Debug)]
pub(crate) enum ToolbarAction {
    FocusLocation(ParsedLocation),
    Basemap(BasemapUiAction),
    SetGraticule(bool),
}

/// Read-only inputs rendered by the top workstation toolbar.
#[derive(Clone, Copy)]
pub(crate) struct ToolbarInputs<'a> {
    pub(crate) settings: &'a AppSettings,
    pub(crate) basemap_status: &'a BasemapStatusSnapshot,
    pub(crate) adapter_label: &'a str,
}

/// Transient UI state for the Go-To coordinate input field in the top toolbar.
#[derive(Clone, Debug, Default)]
pub(crate) struct GoToBarState {
    input: String,
    error: Option<String>,
}

impl GoToBarState {
    fn trigger_goto(&mut self) -> Option<ParsedLocation> {
        match parse_location(&self.input) {
            Ok(loc) => {
                self.error = None;
                Some(loc)
            }
            Err(err) => {
                self.error = Some(err.to_string());
                None
            }
        }
    }
}

/// Render the top toolbar and return user actions to apply in `VeldmapApp`.
pub(crate) fn show_top_toolbar(
    ui: &mut egui::Ui,
    inputs: ToolbarInputs<'_>,
    goto: &mut GoToBarState,
    basemap_ui: &mut BasemapUiState,
) -> Vec<ToolbarAction> {
    let mut actions = Vec::new();
    egui::Panel::top("top_toolbar").show(ui, |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.strong(veldmap_core::APP_NAME);
            ui.separator();

            ui.label("Go to:");
            let text_edit = egui::TextEdit::singleline(&mut goto.input)
                .desired_width(170.0)
                .hint_text("DD, DMS, or MGRS (e.g. 47.23, 39.70 / 37UDQ)");
            let edit_resp = ui.add(text_edit);
            if edit_resp.changed() {
                goto.error = None;
            }
            let enter_pressed =
                edit_resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
            if (ui.button("Fly").clicked() || enter_pressed)
                && let Some(loc) = goto.trigger_goto()
            {
                actions.push(ToolbarAction::FocusLocation(loc));
            }

            ui.separator();
            actions.extend(
                show_toolbar_controls(ui, inputs.settings, inputs.basemap_status, basemap_ui)
                    .into_iter()
                    .map(ToolbarAction::Basemap),
            );

            ui.separator();
            let mut show_graticule = inputs.settings.show_graticule;
            if ui.checkbox(&mut show_graticule, "Graticule").changed() {
                actions.push(ToolbarAction::SetGraticule(show_graticule));
            }

            if let Some(err) = &goto.error {
                ui.separator();
                ui.add(
                    egui::Label::new(egui::RichText::new(err).color(ERROR_TEXT_COLOR).small())
                        .truncate(),
                )
                .on_hover_text(err);
            }

            ui.separator();
            ui.add(egui::Label::new(egui::RichText::new(inputs.adapter_label).weak()).truncate())
                .on_hover_text(inputs.adapter_label);
        });
    });
    actions
}

/// Render the bottom GIS status bar (cursor position, camera center, altitude, GSD).
pub(crate) fn show_status_bar(
    ui: &mut egui::Ui,
    cursor_geo: Option<(f64, f64)>,
    camera: &Camera,
    viewport_height_px: f64,
) {
    egui::Panel::bottom("gis_status_bar").show(ui, |ui| {
        ui.horizontal_wrapped(|ui| {
            let cursor_text = match cursor_geo {
                Some((lat, lon)) => format!("Cursor: {}", hud::format_lat_lon(lat, lon)),
                None => "Cursor: Off globe".to_owned(),
            };
            ui.monospace(cursor_text);
            ui.separator();

            let (clat, clon) = camera.center();
            ui.monospace(format!("Center: {}", hud::format_lat_lon(clat, clon)));
            ui.separator();

            ui.monospace(format!(
                "Alt: {}",
                hud::format_distance_m(camera.height_m())
            ));
            ui.separator();

            let mpp = camera.metres_per_pixel(viewport_height_px);
            ui.monospace(format!("GSD: {}", hud::format_resolution(mpp)));
        });
    });
}
