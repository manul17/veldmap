//! Basemap selector toolbar controls and Basemap & Cache Settings window
//! extracted out of `app.rs`.
//!
//! Strictly action-only: emits [`BasemapUiAction`] values back to `VeldmapApp`
//! without mutating managers, storage, GPU caches, or upload channels directly
//! from UI layout code.

use eframe::egui;
use veldmap_net::validate_custom_url_template;

use super::{ERROR_TEXT_COLOR, TILE_ERROR_TEXT_COLOR, WARN_TEXT_COLOR};
use crate::basemap::BasemapStatusSnapshot;
use crate::settings::{
    AppSettings, BasemapSelection, CUSTOM_MAX_LEVEL_RANGE, CustomXyzSettings,
    DISK_CACHE_MAX_MB_RANGE, validate_gibs_date,
};
use crate::storage::{DiskCacheNotice, DiskCacheStatus};

/// Short human-readable label for toolbar dropdowns and settings UI.
pub const fn basemap_label(selection: BasemapSelection) -> &'static str {
    match selection {
        BasemapSelection::BundledBlueMarble => "Offline Blue Marble",
        BasemapSelection::NasaGibsBlueMarble => "NASA GIBS Blue Marble",
        BasemapSelection::NasaGibsViirs => "NASA GIBS VIIRS True Color",
        BasemapSelection::EoxSentinel2Cloudless => "EOX Sentinel-2 Cloudless",
        BasemapSelection::CustomXyz => "Custom XYZ / TMS",
        BasemapSelection::EllipsoidOnly => "Ellipsoid-only",
    }
}

fn format_disk_cache_notice(notice: &DiskCacheNotice) -> String {
    match notice {
        DiskCacheNotice::Cleared => "Disk cache cleared.".to_owned(),
        DiskCacheNotice::ClearFailed(err) => format!("Clear failed: {err}"),
        DiskCacheNotice::WorkerUnavailable => "Disk cache worker unavailable.".to_owned(),
        DiskCacheNotice::WorkerDisconnected => {
            "Clear failed: maintenance worker disconnected.".to_owned()
        }
    }
}

/// Scoped editable slice of [`AppSettings`] owned by the Basemap & Tile Cache
/// Settings window.
///
/// Keeping this slice separate from unrelated `AppSettings` fields (such as
/// `show_graticule` or future workspace preferences) guarantees saving basemap
/// settings can never silently clobber other user settings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BasemapSettingsDraft {
    pub active_basemap: BasemapSelection,
    pub gibs_date: String,
    pub custom_xyz: CustomXyzSettings,
    pub disk_cache_max_mb: u64,
}

impl BasemapSettingsDraft {
    pub fn from_settings(settings: &AppSettings) -> Self {
        Self {
            active_basemap: settings.active_basemap,
            gibs_date: settings.gibs_date.clone(),
            custom_xyz: settings.custom_xyz.clone(),
            disk_cache_max_mb: settings.disk_cache_max_mb,
        }
    }

    pub fn apply_into(&self, target: &mut AppSettings) {
        target.active_basemap = self.active_basemap;
        target.gibs_date = self.gibs_date.clone();
        target.custom_xyz = self.custom_xyz.clone();
        target.disk_cache_max_mb = self.disk_cache_max_mb;
    }
}

/// User intent emitted by basemap toolbar / settings widgets and applied by
/// `VeldmapApp`.
#[derive(Clone, Debug)]
pub enum BasemapUiAction {
    /// Switch the active basemap from the toolbar without clobbering unsaved
    /// draft edits in an open Settings window.
    SelectBasemap(BasemapSelection),
    /// Commit edited basemap/cache settings from the Settings window.
    ApplyDraft(BasemapSettingsDraft),
    /// Settings window opened this frame (triggers event-driven cache stats refresh).
    SettingsOpened,
    /// Settings window closed this frame.
    SettingsClosed,
    /// Request an asynchronous SQLite disk cache clear off the UI thread.
    ClearDiskCache,
    /// Dismiss any transient disk-cache status notice when opening Settings.
    ClearDiskCacheNotice,
    /// Clear transient failure / offline state and retry visible live tiles.
    RetryLive,
}

/// Transient UI editing state for the Basemap & Cache Settings window.
#[derive(Clone, Debug)]
pub struct BasemapUiState {
    settings_open: bool,
    draft: BasemapSettingsDraft,
    cached_custom_validation: Option<(String, Result<(), String>)>,
}

impl BasemapUiState {
    pub fn new(initial: &AppSettings) -> Self {
        Self {
            settings_open: false,
            draft: BasemapSettingsDraft::from_settings(initial),
            cached_custom_validation: None,
        }
    }

    /// Whether the Basemap & Cache Settings window is currently open.
    pub fn is_settings_open(&self) -> bool {
        self.settings_open
    }

    /// Update only the draft's `active_basemap` selection (preserving unsaved
    /// edits in an open Settings window).
    pub fn set_draft_active_basemap(&mut self, selection: BasemapSelection) {
        self.draft.active_basemap = selection;
    }

    /// Synchronize the Settings draft from persisted `AppSettings` and clear
    /// cached validation.
    pub fn sync_from_settings(&mut self, settings: &AppSettings) {
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
            let res = validate_custom_url_template(&self.draft.custom_xyz.url_template)
                .map(|_| ())
                .map_err(|err| err.to_string());
            self.cached_custom_validation = Some((self.draft.custom_xyz.url_template.clone(), res));
        }
        match self.cached_custom_validation.as_ref() {
            Some((_, Ok(()))) => Ok(()),
            Some((_, Err(err))) => Err(err.as_str()),
            None => Ok(()),
        }
    }
}

const ALL_BASEMAPS: [BasemapSelection; 6] = [
    BasemapSelection::BundledBlueMarble,
    BasemapSelection::NasaGibsBlueMarble,
    BasemapSelection::NasaGibsViirs,
    BasemapSelection::EoxSentinel2Cloudless,
    BasemapSelection::CustomXyz,
    BasemapSelection::EllipsoidOnly,
];

fn show_live_status_badge(
    ui: &mut egui::Ui,
    current: BasemapSelection,
    status: &BasemapStatusSnapshot,
    actions: &mut Vec<BasemapUiAction>,
) {
    if !current.is_live() {
        return;
    }
    let net = &status.network;
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
pub fn show_toolbar_controls(
    ui: &mut egui::Ui,
    settings: &AppSettings,
    status: &BasemapStatusSnapshot,
    ui_state: &mut BasemapUiState,
) -> Vec<BasemapUiAction> {
    let mut actions = Vec::new();
    let current = settings.active_basemap;

    egui::ComboBox::from_id_salt("basemap_selector")
        .selected_text(basemap_label(current))
        .width(175.0)
        .show_ui(ui, |ui| {
            for option in ALL_BASEMAPS {
                if ui
                    .selectable_label(option == current, basemap_label(option))
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
            actions.push(BasemapUiAction::SettingsClosed);
        }
    }

    show_live_status_badge(ui, current, status, &mut actions);
    actions
}

fn show_gibs_section(ui: &mut egui::Ui, ui_state: &mut BasemapUiState) -> bool {
    ui.heading("NASA GIBS Daily Imagery");
    ui.horizontal(|ui| {
        ui.label("VIIRS Date (UTC YYYY-MM-DD):");
        ui.add(
            egui::TextEdit::singleline(&mut ui_state.draft.gibs_date)
                .desired_width(110.0)
                .hint_text("2024-07-01"),
        );
    });
    let date_valid = validate_gibs_date(&ui_state.draft.gibs_date);
    if !date_valid {
        ui.label(
            egui::RichText::new("Enter a valid YYYY-MM-DD date (e.g. 2024-07-01).")
                .color(WARN_TEXT_COLOR)
                .small(),
        );
    }
    date_valid
}

fn show_custom_xyz_section(
    ui: &mut egui::Ui,
    settings: &AppSettings,
    ui_state: &mut BasemapUiState,
) -> bool {
    ui.heading("Custom XYZ / TMS Tile Source");
    if ui_state.draft.active_basemap == BasemapSelection::CustomXyz
        && settings.active_basemap != BasemapSelection::CustomXyz
    {
        ui.label(
            egui::RichText::new(
                "Enter a valid URL template below, then click Apply & Save or Use Custom XYZ.",
            )
            .weak()
            .small(),
        );
    }
    egui::Grid::new("custom_xyz_grid")
        .num_columns(2)
        .spacing([8.0, 6.0])
        .show(ui, |ui| {
            ui.label("Name:");
            ui.add(
                egui::TextEdit::singleline(&mut ui_state.draft.custom_xyz.name)
                    .desired_width(260.0),
            );
            ui.end_row();

            ui.label("URL template:");
            ui.add(
                egui::TextEdit::singleline(&mut ui_state.draft.custom_xyz.url_template)
                    .desired_width(260.0)
                    .hint_text("https://{s}.host/{z}/{x}/{y}.png?key={key}"),
            );
            ui.end_row();

            ui.label("API key (optional):");
            ui.add(
                egui::TextEdit::singleline(&mut ui_state.draft.custom_xyz.api_key)
                    .password(true)
                    .desired_width(260.0)
                    .hint_text("Substituted for {key} / {api_key}"),
            );
            ui.end_row();

            ui.label("Attribution:");
            ui.add(
                egui::TextEdit::singleline(&mut ui_state.draft.custom_xyz.attribution)
                    .desired_width(260.0)
                    .hint_text("Data provider credit"),
            );
            ui.end_row();

            ui.label("Max zoom:");
            ui.add(egui::Slider::new(
                &mut ui_state.draft.custom_xyz.max_level,
                CUSTOM_MAX_LEVEL_RANGE,
            ));
            ui.end_row();
        });

    ui.checkbox(
        &mut ui_state.draft.custom_xyz.tms_flip_y,
        "TMS Y-axis flip (invert {y} from south-up)",
    );

    let custom_err = ui_state.custom_validation().err().map(str::to_owned);
    let custom_valid = custom_err.is_none();
    if let Some(err) = &custom_err
        && (ui_state.draft.active_basemap == BasemapSelection::CustomXyz
            || !ui_state.draft.custom_xyz.url_template.trim().is_empty())
    {
        ui.label(egui::RichText::new(err).color(ERROR_TEXT_COLOR).small());
    }
    custom_valid
}

fn show_disk_cache_section(
    ui: &mut egui::Ui,
    disk_status: &DiskCacheStatus,
    disk_cache_max_mb: &mut u64,
    actions: &mut Vec<BasemapUiAction>,
) {
    ui.heading("Tile Disk Cache");
    if disk_status.available {
        if let Some(stats) = disk_status.stats {
            let used_mb = (stats.payload_bytes as f64) / (1024.0 * 1024.0);
            let cap_mb = (stats.max_bytes as f64) / (1024.0 * 1024.0);
            ui.monospace(format!(
                "Cached: {} tiles · {:.1} MiB / {:.0} MiB",
                stats.tile_count, used_mb, cap_mb
            ));
        } else {
            ui.label(egui::RichText::new("Loading cache statistics…").weak());
        }
    } else {
        ui.label(egui::RichText::new("Disk cache unavailable in this session.").weak());
    }

    ui.horizontal(|ui| {
        ui.label("Disk cap (MiB):");
        ui.add(egui::Slider::new(disk_cache_max_mb, DISK_CACHE_MAX_MB_RANGE).logarithmic(true));
    });

    ui.horizontal(|ui| {
        let clear_label = if disk_status.busy {
            "Clearing…"
        } else {
            "Clear Disk Cache"
        };
        if ui
            .add_enabled(
                disk_status.available && !disk_status.busy,
                egui::Button::new(clear_label),
            )
            .clicked()
        {
            actions.push(BasemapUiAction::ClearDiskCache);
        }
        if let Some(notice) = &disk_status.notice {
            ui.label(
                egui::RichText::new(format_disk_cache_notice(notice))
                    .weak()
                    .small(),
            );
        }
    });
}

/// Render the floating Basemap & Disk Cache Settings window when open,
/// returning any user actions to apply in `VeldmapApp`.
pub fn show_settings_window(
    ctx: &egui::Context,
    settings: &AppSettings,
    disk_status: &DiskCacheStatus,
    ui_state: &mut BasemapUiState,
) -> Vec<BasemapUiAction> {
    if !ui_state.settings_open {
        return Vec::new();
    }

    let mut actions = Vec::new();
    let mut open = ui_state.settings_open;

    egui::Window::new("Basemap & Tile Cache Settings")
        .open(&mut open)
        .resizable(false)
        .default_width(420.0)
        .show(ctx, |ui| {
            let date_valid = show_gibs_section(ui, ui_state);

            ui.separator();
            let custom_valid = show_custom_xyz_section(ui, settings, ui_state);

            ui.separator();
            show_disk_cache_section(
                ui,
                disk_status,
                &mut ui_state.draft.disk_cache_max_mb,
                &mut actions,
            );

            ui.separator();
            let can_apply = date_valid
                && (ui_state.draft.active_basemap != BasemapSelection::CustomXyz || custom_valid);
            ui.horizontal(|ui| {
                if ui
                    .add_enabled(can_apply, egui::Button::new("Apply & Save"))
                    .clicked()
                {
                    actions.push(BasemapUiAction::ApplyDraft(ui_state.draft.clone()));
                }
                if ui
                    .add_enabled(
                        date_valid && custom_valid,
                        egui::Button::new("Use Custom XYZ"),
                    )
                    .clicked()
                {
                    let mut next = ui_state.draft.clone();
                    next.active_basemap = BasemapSelection::CustomXyz;
                    actions.push(BasemapUiAction::ApplyDraft(next));
                }
            });
        });

    ui_state.settings_open = open;
    if !open {
        actions.push(BasemapUiAction::SettingsClosed);
    }
    actions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basemap_draft_preserves_unrelated_app_settings_and_caches_validation() {
        let mut settings = AppSettings {
            show_graticule: false,
            ..AppSettings::default()
        };
        let mut ui_state = BasemapUiState::new(&settings);
        ui_state.draft.active_basemap = BasemapSelection::EoxSentinel2Cloudless;
        ui_state.draft.disk_cache_max_mb = 1024;

        // Applying basemap draft must leave unrelated `show_graticule` intact.
        ui_state.draft.apply_into(&mut settings);
        assert!(!settings.show_graticule);
        assert_eq!(
            settings.active_basemap,
            BasemapSelection::EoxSentinel2Cloudless
        );
        assert_eq!(settings.disk_cache_max_mb, 1024);

        // Cached custom validation fails on empty URL, then updates when edited.
        assert!(ui_state.custom_validation().is_err());
        ui_state.draft.custom_xyz.url_template =
            "https://tiles.example.org/{z}/{x}/{y}.png".to_owned();
        assert!(ui_state.custom_validation().is_ok());
    }
}
