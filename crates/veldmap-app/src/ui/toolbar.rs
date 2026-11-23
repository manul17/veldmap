//! Top workstation toolbar (Go-To navigation, basemap selector, graticule
//! toggle, GPU adapter badge) and bottom GIS status bar extracted from `app.rs`.

use eframe::egui;
use veldmap_core::geo::camera::Camera;
use veldmap_core::geo::coords::parse_location;

use crate::basemap::BasemapStatusSnapshot;
use crate::settings::AppSettings;

use super::ERROR_TEXT_COLOR;
use super::basemap::{BasemapUiAction, BasemapUiState, show_toolbar_controls};
use super::hud;

/// Transient UI state for the Go-To coordinate input field in the top toolbar.
#[derive(Clone, Debug, Default)]
pub(crate) struct GoToBarState {
    input: String,
    error: Option<String>,
}

impl GoToBarState {
    fn trigger_goto(&mut self, camera: &mut Camera) {
        match parse_location(&self.input) {
            Ok(loc) => {
                tracing::info!(
                    lat = loc.lat_deg,
                    lon = loc.lon_deg,
                    radius_deg = loc.radius_deg,
                    kind = ?loc.kind,
                    "Flying camera to parsed location"
                );
                camera.focus(loc.lat_deg, loc.lon_deg, loc.radius_deg);
                self.error = None;
            }
            Err(err) => {
                self.error = Some(err.to_string());
            }
        }
    }
}

/// Render the top toolbar and return `(basemap_actions, graticule_toggled)`.
pub(crate) fn show_top_toolbar(
    ui: &mut egui::Ui,
    camera: &mut Camera,
    goto: &mut GoToBarState,
    settings: &AppSettings,
    basemap_status: &BasemapStatusSnapshot,
    basemap_ui: &mut BasemapUiState,
    adapter_label: &str,
) -> (Vec<BasemapUiAction>, Option<bool>) {
    let mut toolbar_actions = Vec::new();
    let mut graticule_toggled = None;
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
            if ui.button("Fly").clicked() || enter_pressed {
                goto.trigger_goto(camera);
            }

            ui.separator();
            toolbar_actions = show_toolbar_controls(ui, settings, basemap_status, basemap_ui);

            ui.separator();
            let mut show_graticule = settings.show_graticule;
            if ui.checkbox(&mut show_graticule, "Graticule").changed() {
                graticule_toggled = Some(show_graticule);
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
            ui.add(egui::Label::new(egui::RichText::new(adapter_label).weak()).truncate())
                .on_hover_text(adapter_label);
        });
    });
    (toolbar_actions, graticule_toggled)
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
