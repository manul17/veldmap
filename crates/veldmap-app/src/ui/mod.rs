//! `egui` UI chrome, floating GIS HUD widgets, and basemap settings panels.

pub(crate) mod basemap;
pub(crate) mod crs_picker;
pub(crate) mod hud;
pub(crate) mod layers;
pub(crate) mod measure;
mod settings_window;
pub(crate) mod status_bar;
pub(crate) mod toolbar;

pub(crate) use settings_window::show_settings_window;

use eframe::egui;

/// Validation / fatal badge text color (`#EB6E5F`).
const ERROR_TEXT_COLOR: egui::Color32 = egui::Color32::from_rgb(235, 110, 95);
/// Warning / offline fallback badge text color (`#F0B955`).
const WARN_TEXT_COLOR: egui::Color32 = egui::Color32::from_rgb(240, 185, 85);
/// Transient tile-error badge text color (`#EB915F`).
const TILE_ERROR_TEXT_COLOR: egui::Color32 = egui::Color32::from_rgb(235, 145, 95);
