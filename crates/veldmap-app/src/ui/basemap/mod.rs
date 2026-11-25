//! Basemap selector toolbar controls, settings window, and action types for
//! `VeldmapApp` (`app/mod.rs`).
//!
//! Strictly action-only: emits [`BasemapUiAction`] values back to `VeldmapApp`
//! without mutating managers, storage, GPU caches, or upload channels directly
//! from UI layout code, and without importing `veldmap-net` or `AppStorage`.

mod settings_window;

pub(crate) use settings_window::show_settings_window;

use eframe::egui;

use super::{ERROR_TEXT_COLOR, TILE_ERROR_TEXT_COLOR, WARN_TEXT_COLOR};
use crate::basemap::BasemapStatusSnapshot;
use crate::storage::settings::{
    AppSettings, BasemapSelection, BasemapSettingsDraft, validate_custom_url_template,
};

/// User intent emitted by basemap toolbar / settings widgets and applied by
/// `VeldmapApp`.
#[derive(Clone, Debug)]
pub(crate) enum BasemapUiAction {
    /// Switch the active basemap from the toolbar without clobbering unsaved
    /// draft edits in an open Settings window.
    SelectBasemap(BasemapSelection),
    /// Commit edited basemap/cache settings from the Settings window.
    ApplyDraft(BasemapSettingsDraft),
    /// Settings window opened this frame (triggers event-driven cache stats refresh).
    SettingsOpened,
    /// Request an asynchronous SQLite disk cache clear off the UI thread.
    ClearDiskCache,
    /// Dismiss any transient disk-cache status notice when opening Settings.
    ClearDiskCacheNotice,
    /// Clear transient failure / offline state and retry visible live tiles.
    RetryLive,
}

/// Transient UI editing state for the Basemap & Cache Settings window.
#[derive(Clone, Debug)]
pub(crate) struct BasemapUiState {
    settings_open: bool,
    draft: BasemapSettingsDraft,
    cached_custom_validation: Option<(String, Result<(), String>)>,
}

impl BasemapUiState {
    pub(crate) fn new(initial: &AppSettings) -> Self {
        Self {
            settings_open: false,
            draft: BasemapSettingsDraft::from_settings(initial),
            cached_custom_validation: None,
        }
    }

    /// Whether the Basemap & Cache Settings window is currently open.
    pub(crate) fn is_settings_open(&self) -> bool {
        self.settings_open
    }

    /// Update only the draft's `active_basemap` selection (preserving unsaved
    /// edits in an open Settings window).
    pub(crate) fn set_draft_active_basemap(&mut self, selection: BasemapSelection) {
        self.draft.active_basemap = selection;
    }

    /// Synchronize the Settings draft from persisted `AppSettings` and clear
    /// cached validation.
    pub(crate) fn sync_from_settings(&mut self, settings: &AppSettings) {
        self.draft = BasemapSettingsDraft::from_settings(settings);
        self.cached_custom_validation = None;
    }

    fn open_settings(&mut self, settings: &AppSettings, actions: &mut Vec<BasemapUiAction>) {
        let was_open = self.settings_open;
        if !was_open {
            self.sync_from_settings(settings);
            self.settings_open = true;
            actions.push(BasemapUiAction::SettingsOpened);
        }
        actions.push(BasemapUiAction::ClearDiskCacheNotice);
    }

    /// Return cached validation result for `draft.custom_xyz.url_template`,
    /// recomputing only when the URL template changes.
    fn custom_validation(&mut self) -> Result<(), &str> {
        let needs_recompute = self
            .cached_custom_validation
            .as_ref()
            .is_none_or(|(prev, _)| prev != &self.draft.custom_xyz.url_template);
        if needs_recompute {
            let res = validate_custom_url_template(&self.draft.custom_xyz.url_template);
            self.cached_custom_validation = Some((self.draft.custom_xyz.url_template.clone(), res));
        }
        match self.cached_custom_validation.as_ref() {
            Some((_, Ok(()))) => Ok(()),
            Some((_, Err(err))) => Err(err.as_str()),
            None => Ok(()),
        }
    }
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
/// Settings button, returning any user actions to apply in `VeldmapApp`.
pub(super) fn show_toolbar_controls(
    ui: &mut egui::Ui,
    settings: &AppSettings,
    status: &BasemapStatusSnapshot,
    ui_state: &mut BasemapUiState,
) -> Vec<BasemapUiAction> {
    let mut actions = Vec::new();
    let current = settings.active_basemap;

    egui::ComboBox::from_id_salt("basemap_selector")
        .selected_text(current.label())
        .width(175.0)
        .show_ui(ui, |ui| {
            for option in BasemapSelection::ALL {
                if ui
                    .selectable_label(option == current, option.label())
                    .clicked()
                    && option != current
                {
                    if option == BasemapSelection::CustomXyz
                        && settings.custom_xyz.url_template.trim().is_empty()
                    {
                        ui_state.open_settings(settings, &mut actions);
                        ui_state.draft.active_basemap = BasemapSelection::CustomXyz;
                    } else {
                        actions.push(BasemapUiAction::SelectBasemap(option));
                    }
                }
            }
        });

    if ui
        .button("Basemap…")
        .on_hover_text("Configure NASA GIBS date, Custom XYZ/TMS URL & API key, and disk cache")
        .clicked()
    {
        if !ui_state.settings_open {
            ui_state.open_settings(settings, &mut actions);
        } else {
            ui_state.settings_open = false;
        }
    }

    show_live_status_badge(ui, current, status, &mut actions);
    actions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn custom_validation_caches_and_recomputes_on_url_change() {
        let settings = AppSettings::default();
        let mut ui_state = BasemapUiState::new(&settings);

        assert!(ui_state.custom_validation().is_err());
        ui_state.draft.custom_xyz.url_template =
            "https://tiles.example.org/{z}/{x}/{y}.png".to_owned();
        assert!(ui_state.custom_validation().is_ok());
    }
}
