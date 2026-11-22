//! Basemap selector toolbar controls, Basemap & Cache Settings window, and
//! viewport attribution badge extracted out of `app.rs`.
//!
//! Strictly action-only: emits [`BasemapUiAction`] values back to `VeldmapApp`
//! without mutating managers, storage, GPU caches, or upload channels directly
//! from UI layout code.

use eframe::egui;
use veldmap_net::{CustomBasemapConfig, LiveBasemapSpec, is_valid_iso_date};
use veldmap_storage::{
    AppSettings, BasemapSelection, CustomXyzSettings, MAX_DISK_CACHE_MAX_MB, MIN_DISK_CACHE_MAX_MB,
};

use crate::basemap::BasemapManager;
use crate::hud;
use crate::storage::DiskCacheUiState;

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
    pub settings_open: bool,
    pub draft: BasemapSettingsDraft,
    cached_custom_validation: Option<(CustomXyzSettings, Result<(), String>)>,
}

impl BasemapUiState {
    pub fn new(initial: &AppSettings) -> Self {
        Self {
            settings_open: false,
            draft: BasemapSettingsDraft::from_settings(initial),
            cached_custom_validation: None,
        }
    }

    /// Synchronize the Settings draft from persisted `AppSettings` and clear
    /// cached validation.
    pub fn sync_from_settings(&mut self, settings: &AppSettings) {
        self.draft = BasemapSettingsDraft::from_settings(settings);
        self.cached_custom_validation = None;
    }

    /// Return cached validation result for `draft.custom_xyz`, recomputing
    /// only when any field of `custom_xyz` changes.
    fn custom_validation(&mut self) -> Result<(), &str> {
        let needs_recompute = self
            .cached_custom_validation
            .as_ref()
            .is_none_or(|(prev, _)| prev != &self.draft.custom_xyz);
        if needs_recompute {
            let res = validate_custom_xyz(&self.draft.custom_xyz);
            self.cached_custom_validation = Some((self.draft.custom_xyz.clone(), res));
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

fn validate_custom_xyz(custom: &CustomXyzSettings) -> Result<(), String> {
    LiveBasemapSpec::custom(CustomBasemapConfig {
        name: &custom.name,
        url_template: &custom.url_template,
        api_key: &custom.api_key,
        tms_flip_y: custom.tms_flip_y,
        max_level: custom.max_level,
        attribution: &custom.attribution,
    })
    .map(|_| ())
    .map_err(|err| err.to_string())
}

/// Render toolbar basemap selector, live streaming / error / offline pill, and
/// Settings button, returning any user actions to apply in `VeldmapApp`.
pub fn show_toolbar_controls(
    ui: &mut egui::Ui,
    settings: &AppSettings,
    manager: &BasemapManager,
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
                        if !ui_state.settings_open {
                            ui_state.sync_from_settings(settings);
                        }
                        ui_state.draft.active_basemap = BasemapSelection::CustomXyz;
                        ui_state.settings_open = true;
                        actions.push(BasemapUiAction::ClearDiskCacheNotice);
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
        ui_state.settings_open = !ui_state.settings_open;
        if ui_state.settings_open {
            ui_state.sync_from_settings(settings);
            actions.push(BasemapUiAction::ClearDiskCacheNotice);
        }
    }

    // Live streaming / error / offline indicator badge.
    if current.is_live() {
        let status = manager.network_status();
        if let Some(init_err) = manager.live_pipeline_error() {
            ui.label(
                egui::RichText::new("Live unavailable")
                    .color(egui::Color32::from_rgb(240, 185, 85))
                    .small(),
            )
            .on_hover_text(init_err);
        } else if let Some(cfg_err) = manager.config_error() {
            let badge = match current {
                BasemapSelection::NasaGibsViirs => "Invalid date",
                BasemapSelection::CustomXyz => "Invalid URL",
                _ => "Config error",
            };
            ui.label(
                egui::RichText::new(badge)
                    .color(egui::Color32::from_rgb(235, 110, 95))
                    .small(),
            )
            .on_hover_text(cfg_err);
        } else if status.offline {
            let hover = status
                .last_error
                .as_deref()
                .unwrap_or("Network unreachable; showing offline Blue Marble fallback");
            ui.label(
                egui::RichText::new("Offline (fallback)")
                    .color(egui::Color32::from_rgb(240, 185, 85))
                    .small(),
            )
            .on_hover_text(hover);
            if ui.small_button("Retry").clicked() {
                actions.push(BasemapUiAction::RetryLive);
            }
        } else if status.failed_count > 0
            && let Some(err) = status.last_error.as_deref()
        {
            ui.label(
                egui::RichText::new("Tile error")
                    .color(egui::Color32::from_rgb(235, 145, 95))
                    .small(),
            )
            .on_hover_text(err);
            if ui.small_button("Retry").clicked() {
                actions.push(BasemapUiAction::RetryLive);
            }
        } else if status.in_flight > 0 || status.queued > 0 {
            ui.label(
                egui::RichText::new(format!("Streaming ({})", status.in_flight + status.queued))
                    .weak()
                    .small(),
            );
        }
    }

    actions
}

/// Render the floating Basemap & Disk Cache Settings window when open,
/// returning any user actions to apply in `VeldmapApp`.
pub fn show_settings_window(
    ctx: &egui::Context,
    settings: &AppSettings,
    disk_ui: &DiskCacheUiState,
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
            ui.heading("NASA GIBS Daily Imagery");
            ui.horizontal(|ui| {
                ui.label("VIIRS Date (UTC YYYY-MM-DD):");
                ui.add(
                    egui::TextEdit::singleline(&mut ui_state.draft.gibs_date)
                        .desired_width(110.0)
                        .hint_text("2024-07-01"),
                );
            });
            let date_valid = is_valid_iso_date(ui_state.draft.gibs_date.trim());
            if !date_valid {
                ui.label(
                    egui::RichText::new("Enter a valid YYYY-MM-DD date (e.g. 2024-07-01).")
                        .color(egui::Color32::from_rgb(240, 185, 85))
                        .small(),
                );
            }

            ui.separator();
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
                        1..=20,
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
                ui.label(
                    egui::RichText::new(err)
                        .color(egui::Color32::from_rgb(235, 110, 95))
                        .small(),
                );
            }

            ui.separator();
            ui.heading("SQLite WAL Tile Disk Cache");
            if disk_ui.available {
                if let Some(stats) = disk_ui.stats {
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
                ui.add(
                    egui::Slider::new(
                        &mut ui_state.draft.disk_cache_max_mb,
                        MIN_DISK_CACHE_MAX_MB..=MAX_DISK_CACHE_MAX_MB,
                    )
                    .logarithmic(true),
                );
            });

            ui.horizontal(|ui| {
                let clear_label = if disk_ui.busy {
                    "Clearing…"
                } else {
                    "Clear Disk Cache"
                };
                if ui
                    .add_enabled(
                        disk_ui.available && !disk_ui.busy,
                        egui::Button::new(clear_label),
                    )
                    .clicked()
                {
                    actions.push(BasemapUiAction::ClearDiskCache);
                }
                if let Some(notice) = &disk_ui.notice {
                    ui.label(egui::RichText::new(notice).weak().small());
                }
            });

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
    actions
}

/// Paint a compact, high-contrast attribution pill in the bottom-right corner
/// of the globe viewport so NASA GIBS and EOX CC BY 4.0 credits remain visible
/// without overlapping the bottom-left metric scale bar at narrow widths.
pub fn paint_attribution_badge(painter: &egui::Painter, rect: egui::Rect, attribution: &str) {
    if attribution.is_empty() || rect.height() < 60.0 {
        return;
    }

    let scale_bar_reserve = hud::scale_bar_reserved_width(rect.width());
    let avail = rect.width() - scale_bar_reserve - 26.0; // 16 pad_x*2 + 10 right margin
    if avail < 100.0 {
        return;
    }

    let font = egui::FontId::proportional(11.0);
    let max_width = avail.min(560.0);
    let galley = painter.layout(
        attribution.to_owned(),
        font,
        egui::Color32::from_rgba_unmultiplied(235, 240, 246, 235),
        max_width,
    );

    let pad_x = 8.0;
    let pad_y = 4.0;
    let size = galley.size() + egui::vec2(pad_x * 2.0, pad_y * 2.0);
    let margin = 10.0;
    let badge_rect = egui::Rect::from_min_size(
        egui::pos2(
            rect.right() - margin - size.x,
            rect.bottom() - margin - size.y,
        ),
        size,
    );

    painter.rect_filled(
        badge_rect,
        4.0,
        egui::Color32::from_rgba_unmultiplied(10, 16, 26, 175),
    );
    painter.galley(
        badge_rect.min + egui::vec2(pad_x, pad_y),
        galley,
        egui::Color32::WHITE,
    );
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
