//! `egui` workstation UI chrome: top toolbar, collapsible `Layers` | `Cloud /
//! STAC` workbench, 3-tab Settings modal, CRS picker, geodesic measurement
//! panel, status bar, and floating GIS HUD widgets.

pub(crate) mod basemap;
pub(crate) mod crs_picker;
pub(crate) mod export_png;
pub(crate) mod format;
pub(crate) mod hud;
pub(crate) mod layers;
pub(crate) mod measure;
pub(crate) mod settings;
pub(crate) mod stac;
pub(crate) mod status_bar;
pub(crate) mod toolbar;
mod widgets;
pub(crate) mod workbench;

use eframe::egui;

/// Validation / fatal badge text color (`#EB6E5F`).
const ERROR_TEXT_COLOR: egui::Color32 = egui::Color32::from_rgb(235, 110, 95);
/// Warning / offline fallback badge text color (`#F0B955`).
const WARN_TEXT_COLOR: egui::Color32 = egui::Color32::from_rgb(240, 185, 85);
/// Positive / Band Math recipe / L2A BOA badge text color (`#82DC91`).
const BADGE_RECIPE_COLOR: egui::Color32 = egui::Color32::from_rgb(130, 220, 145);
/// Cloud COG / L1C TOA badge text color (`#64C3FF`).
const BADGE_CLOUD_COLOR: egui::Color32 = egui::Color32::from_rgb(100, 195, 255);
/// Selected STAC item highlight border color (`#FFC840`).
const SELECTION_ACCENT_COLOR: egui::Color32 = egui::Color32::from_rgb(255, 200, 64);

/// User dismissal intents returned by [`show_system_fault_banner`].
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct FaultBannerAction {
    pub(crate) dismiss_gpu: bool,
    pub(crate) dismiss_storage: bool,
}

fn show_fault_item(ui: &mut egui::Ui, msg: &str, color: egui::Color32) -> bool {
    ui.add(egui::Label::new(egui::RichText::new(msg).color(color).small()).truncate())
        .on_hover_text(msg);
    ui.small_button("Dismiss").clicked()
}

/// Render a slim top warning/error banner when a GPU or persistence fault is
/// active (renders zero widgets when both inputs are `None`).
pub(crate) fn show_system_fault_banner(
    ui: &mut egui::Ui,
    gpu_fault: Option<(&str, bool)>,
    storage_fault: Option<&str>,
) -> FaultBannerAction {
    let mut action = FaultBannerAction::default();
    if gpu_fault.is_none() && storage_fault.is_none() {
        return action;
    }
    egui::Panel::top("system_fault_banner").show(ui, |ui| {
        ui.horizontal(|ui| {
            if let Some((msg, is_lost)) = gpu_fault {
                let color = if is_lost {
                    ERROR_TEXT_COLOR
                } else {
                    WARN_TEXT_COLOR
                };
                action.dismiss_gpu = show_fault_item(ui, msg, color);
            }
            if gpu_fault.is_some() && storage_fault.is_some() {
                ui.separator();
            }
            if let Some(msg) = storage_fault {
                action.dismiss_storage = show_fault_item(ui, msg, WARN_TEXT_COLOR);
            }
        });
    });
    action
}
