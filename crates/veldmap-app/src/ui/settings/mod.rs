//! Unified 3-tab Settings modal window (Basemap, Cache & GPU, and STAC / Cloud).
//!
//! Strictly action-only: emits [`SettingsAction`] values back to `VeldmapApp`
//! without mutating storage, managers, or GPU caches directly.

mod basemap_tab;
mod cache_tab;
mod stac_tab;

use eframe::egui;

use super::ERROR_TEXT_COLOR;
use crate::storage::DiskCacheStatus;
use crate::storage::settings::{
    AppSettings, AppSettingsDraft, BasemapSelection, validate_custom_url_template,
    validate_gibs_date,
};

use basemap_tab::show_basemap_tab;
use cache_tab::show_cache_tab;
use stac_tab::show_stac_tab;

/// Active tab inside the unified Settings modal.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum SettingsTab {
    #[default]
    Basemap,
    CacheGpu,
    StacCloud,
}

/// User intent emitted by the Settings modal and applied by `VeldmapApp`.
#[derive(Clone, Debug)]
pub(crate) enum SettingsAction {
    /// Commit edited basemap, cache/GPU, and STAC settings from the Settings window.
    ApplyDraft(Box<AppSettingsDraft>),
    /// Request an asynchronous SQLite disk cache clear off the UI thread.
    ClearDiskCache,
}

/// Transient UI editing state for the unified Settings modal.
#[derive(Clone, Debug)]
pub(crate) struct SettingsWindowState {
    open: bool,
    active_tab: SettingsTab,
    pub(super) draft: AppSettingsDraft,
    cached_custom_validation: Option<(String, Result<(), String>)>,
}

impl SettingsWindowState {
    pub(crate) fn new(initial: &AppSettings) -> Self {
        Self {
            open: false,
            active_tab: SettingsTab::Basemap,
            draft: AppSettingsDraft::from_settings(initial),
            cached_custom_validation: None,
        }
    }

    /// Whether the Settings modal is currently open.
    pub(crate) fn is_open(&self) -> bool {
        self.open
    }

    /// Currently selected Settings tab.
    pub(crate) fn active_tab(&self) -> SettingsTab {
        self.active_tab
    }

    /// Open the Settings modal focused on `tab`, synchronizing the draft from
    /// persisted `settings` if the modal was previously closed.
    pub(crate) fn open_tab(&mut self, settings: &AppSettings, tab: SettingsTab) {
        if !self.open {
            self.sync_from_settings(settings);
            self.open = true;
        }
        self.active_tab = tab;
    }

    /// Close the Settings modal.
    pub(crate) fn close(&mut self) {
        self.open = false;
    }

    /// Update only the draft's `active_basemap` selection (preserving unsaved
    /// edits in an open Settings window).
    pub(crate) fn set_draft_active_basemap(&mut self, selection: BasemapSelection) {
        self.draft.active_basemap = selection;
    }

    /// Synchronize the Settings draft from persisted [`AppSettings`] and clear
    /// cached URL validation.
    pub(crate) fn sync_from_settings(&mut self, settings: &AppSettings) {
        self.draft = AppSettingsDraft::from_settings(settings);
        self.cached_custom_validation = None;
    }

    /// Return cached validation result for `draft.custom_xyz.url_template`,
    /// recomputing only when the URL template changes.
    pub(super) fn custom_validation(&mut self) -> Result<(), &str> {
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

/// Render the floating unified Settings modal when open, returning any user
/// actions to apply in `VeldmapApp`.
pub(crate) fn show_settings_window(
    ctx: &egui::Context,
    settings: &AppSettings,
    disk_status: &DiskCacheStatus,
    state: &mut SettingsWindowState,
) -> Vec<SettingsAction> {
    if !state.open {
        return Vec::new();
    }

    let mut actions = Vec::new();
    let mut open = state.open;

    egui::Window::new("Settings")
        .open(&mut open)
        .resizable(false)
        .default_width(450.0)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut state.active_tab, SettingsTab::Basemap, "Basemap");
                ui.selectable_value(&mut state.active_tab, SettingsTab::CacheGpu, "Cache & GPU");
                ui.selectable_value(
                    &mut state.active_tab,
                    SettingsTab::StacCloud,
                    "STAC / Cloud",
                );
            });
            ui.separator();

            match state.active_tab {
                SettingsTab::Basemap => {
                    show_basemap_tab(ui, settings, state);
                }
                SettingsTab::CacheGpu => {
                    show_cache_tab(
                        ui,
                        disk_status,
                        &mut state.draft.disk_cache_max_mb,
                        &mut state.draft.gpu_tile_cache_max_mb,
                        &mut actions,
                    );
                }
                SettingsTab::StacCloud => {
                    show_stac_tab(ui, &mut state.draft.stac);
                }
            }

            // Evaluate validation AFTER tab widgets mutate `state.draft` so
            // button enablement is always fresh on the same frame.
            let date_valid = validate_gibs_date(&state.draft.gibs_date);
            let custom_valid = state.custom_validation().is_ok();
            let basemap_valid = (state.draft.active_basemap != BasemapSelection::NasaGibsViirs
                || date_valid)
                && (state.draft.active_basemap != BasemapSelection::CustomXyz || custom_valid);
            let stac_valid = state.draft.stac.validation_error().is_none();
            let can_apply = basemap_valid && stac_valid;

            ui.separator();
            ui.horizontal(|ui| {
                if ui
                    .add_enabled(can_apply, egui::Button::new("Apply & Save"))
                    .clicked()
                {
                    actions.push(SettingsAction::ApplyDraft(Box::new(state.draft.clone())));
                }
                if state.active_tab == SettingsTab::Basemap
                    && ui
                        .add_enabled(
                            custom_valid && stac_valid,
                            egui::Button::new("Use Custom XYZ"),
                        )
                        .clicked()
                {
                    let mut next = state.draft.clone();
                    next.active_basemap = BasemapSelection::CustomXyz;
                    actions.push(SettingsAction::ApplyDraft(Box::new(next)));
                }
                if !basemap_valid && state.active_tab != SettingsTab::Basemap {
                    ui.colored_label(
                        ERROR_TEXT_COLOR,
                        egui::RichText::new("Fix invalid Basemap settings before saving.").small(),
                    );
                } else if !stac_valid && state.active_tab != SettingsTab::StacCloud {
                    ui.colored_label(
                        ERROR_TEXT_COLOR,
                        egui::RichText::new("Fix invalid STAC / Cloud settings before saving.")
                            .small(),
                    );
                }
            });
        });

    state.open = open;
    actions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn custom_validation_caches_and_recomputes_on_url_change() {
        let settings = AppSettings::default();
        let mut state = SettingsWindowState::new(&settings);

        assert!(state.custom_validation().is_err());
        state.draft.custom_xyz.url_template =
            "https://tiles.example.org/{z}/{x}/{y}.png".to_owned();
        assert!(state.custom_validation().is_ok());
    }
}
