//! `egui` UI chrome, floating GIS HUD widgets, and basemap settings panels.

pub(crate) mod basemap;
pub(crate) mod crs_picker;
pub(crate) mod format;
pub(crate) mod hud;
pub(crate) mod layers;
pub(crate) mod measure;
pub(crate) mod status_bar;
pub(crate) mod toolbar;

use eframe::egui;

/// Validation / fatal badge text color (`#EB6E5F`).
const ERROR_TEXT_COLOR: egui::Color32 = egui::Color32::from_rgb(235, 110, 95);
/// Warning / offline fallback badge text color (`#F0B955`).
const WARN_TEXT_COLOR: egui::Color32 = egui::Color32::from_rgb(240, 185, 85);
/// Transient tile-error badge text color (`#EB915F`).
const TILE_ERROR_TEXT_COLOR: egui::Color32 = egui::Color32::from_rgb(235, 145, 95);

/// Return `true` when a single-line `egui::TextEdit` was submitted by pressing
/// `Enter`.
pub(super) fn submitted_with_enter(ui: &egui::Ui, resp: &egui::Response) -> bool {
    resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter))
}
