//! `egui` UI chrome, floating GIS HUD widgets, and basemap settings panels.

pub(crate) mod basemap;
pub(crate) mod hud;
pub(crate) mod toolbar;

use eframe::egui;

/// Validation / fatal badge text color (`#EB6E5F`).
pub(crate) const ERROR_TEXT_COLOR: egui::Color32 = egui::Color32::from_rgb(235, 110, 95);
/// Warning / offline fallback badge text color (`#F0B955`).
pub(crate) const WARN_TEXT_COLOR: egui::Color32 = egui::Color32::from_rgb(240, 185, 85);
/// Transient tile-error badge text color (`#EB915F`).
pub(crate) const TILE_ERROR_TEXT_COLOR: egui::Color32 = egui::Color32::from_rgb(235, 145, 95);
