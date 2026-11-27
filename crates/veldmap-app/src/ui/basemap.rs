//! Slim toolbar basemap selector dropdown, Settings trigger, and live
//! streaming / error / offline status pill.
//!
//! Strictly action-only: emits [`BasemapUiAction`] values back to `VeldmapApp`
//! without borrowing `SettingsWindowState`, `AppSettings`, `AppStorage`, or
//! `veldmap-net`.

use eframe::egui;

use crate::basemap::BasemapStatusSnapshot;
use crate::storage::settings::BasemapSelection;

use super::{ERROR_TEXT_COLOR, WARN_TEXT_COLOR};

/// Transient tile-error badge text color (`#EB915F`).
const TILE_ERROR_TEXT_COLOR: egui::Color32 = egui::Color32::from_rgb(235, 145, 95);

/// User intent emitted by the toolbar basemap controls and applied by
/// `VeldmapApp`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BasemapUiAction {
    /// Switch the active basemap from the toolbar without clobbering unsaved
    /// draft edits in an open Settings window.
    SelectBasemap(BasemapSelection),
    /// Open Settings on the Basemap tab pre-selecting Custom XYZ when no
    /// Custom XYZ URL template is configured yet.
    ConfigureCustomXyz,
    /// Toggle the unified Settings modal open or closed.
    ToggleSettings,
    /// Clear transient failure / offline state and retry visible live tiles.
    RetryLive,
}

fn show_live_status_badge(
    ui: &mut egui::Ui,
    current: BasemapSelection,
    status: &BasemapStatusSnapshot,
    actions: &mut Vec<BasemapUiAction>,
) {
    if !current.is_live() {
        return;
    }
    let net = &status.stream;
    if let Some(init_err) = status.live_pipeline_error.as_deref() {
        ui.label(
            egui::RichText::new("Live unavailable")
                .color(WARN_TEXT_COLOR)
                .small(),
        )
        .on_hover_text(init_err);
    } else if let Some(cfg_err) = status.config_error.as_deref() {
        let badge = match current {
            BasemapSelection::NasaGibsViirs => "Invalid date",
            BasemapSelection::CustomXyz => "Invalid URL",
            _ => "Config error",
        };
        ui.label(egui::RichText::new(badge).color(ERROR_TEXT_COLOR).small())
            .on_hover_text(cfg_err);
    } else if net.offline {
        let hover = net
            .last_error
            .as_deref()
            .unwrap_or("Network unreachable; showing offline Blue Marble fallback");
        ui.label(
            egui::RichText::new("Offline (fallback)")
                .color(WARN_TEXT_COLOR)
                .small(),
        )
        .on_hover_text(hover);
        if ui.small_button("Retry").clicked() {
            actions.push(BasemapUiAction::RetryLive);
        }
    } else if net.failed_count > 0
        && let Some(err) = net.last_error.as_deref()
    {
        ui.label(
            egui::RichText::new("Tile error")
                .color(TILE_ERROR_TEXT_COLOR)
                .small(),
        )
        .on_hover_text(err);
        if ui.small_button("Retry").clicked() {
            actions.push(BasemapUiAction::RetryLive);
        }
    } else if net.in_flight > 0 || net.queued > 0 {
        ui.label(
            egui::RichText::new(format!("Streaming ({})", net.in_flight + net.queued))
                .weak()
                .small(),
        );
    }
}

/// Render toolbar basemap selector, live streaming / error / offline pill, and
/// Settings button, returning pure actions to apply in `VeldmapApp`.
pub(super) fn show_toolbar_controls(
    ui: &mut egui::Ui,
    active_basemap: BasemapSelection,
    custom_xyz_configured: bool,
    status: &BasemapStatusSnapshot,
) -> Vec<BasemapUiAction> {
    let mut actions = Vec::new();

    egui::ComboBox::from_id_salt("basemap_selector")
        .selected_text(active_basemap.label())
        .width(175.0)
        .show_ui(ui, |ui| {
            for option in BasemapSelection::ALL {
                if ui
                    .selectable_label(option == active_basemap, option.label())
                    .clicked()
                    && option != active_basemap
                {
                    if option == BasemapSelection::CustomXyz && !custom_xyz_configured {
                        actions.push(BasemapUiAction::ConfigureCustomXyz);
                    } else {
                        actions.push(BasemapUiAction::SelectBasemap(option));
                    }
                }
            }
        });

    if ui
        .button("Settings…")
        .on_hover_text("Configure basemaps, disk/GPU cache budgets, and custom STAC endpoints")
        .clicked()
    {
        actions.push(BasemapUiAction::ToggleSettings);
    }

    show_live_status_badge(ui, active_basemap, status, &mut actions);
    actions
}
