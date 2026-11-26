//! Unified Settings modal window (Basemap, Cache & GPU, and STAC / Cloud tabs).
//!
//! Strictly action-only: emits [`BasemapUiAction`] values back to `VeldmapApp`
//! without mutating storage, managers, or GPU caches directly.

use eframe::egui;

use super::super::{ERROR_TEXT_COLOR, WARN_TEXT_COLOR};
use super::{BasemapUiAction, BasemapUiState, SettingsTab};
use crate::storage::settings::{
    AppSettings, BasemapSelection, CUSTOM_MAX_LEVEL_RANGE, DEFAULT_CUSTOM_GPU_CACHE_MAX_MB,
    DISK_CACHE_MAX_MB_RANGE, GPU_CACHE_MAX_MB_RANGE, StacAuthKind, validate_gibs_date,
};
use crate::storage::{DiskCacheNotice, DiskCacheStatus};

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
    gpu_tile_cache_max_mb: &mut Option<u64>,
    actions: &mut Vec<BasemapUiAction>,
) {
    ui.heading("Tile Disk & GPU Cache");
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
        let mut custom_vram = gpu_tile_cache_max_mb.is_some();
        if ui
            .checkbox(&mut custom_vram, "Custom GPU VRAM cap (MiB):")
            .changed()
        {
            *gpu_tile_cache_max_mb = if custom_vram {
                Some(DEFAULT_CUSTOM_GPU_CACHE_MAX_MB)
            } else {
                None
            };
        }
        if let Some(mb) = gpu_tile_cache_max_mb {
            ui.add(egui::Slider::new(mb, GPU_CACHE_MAX_MB_RANGE).logarithmic(true));
        } else {
            ui.weak("Auto (host RAM scaled)");
        }
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

fn show_stac_section(ui: &mut egui::Ui, ui_state: &mut BasemapUiState) {
    ui.heading("Custom STAC API Endpoint & Auth");
    ui.label(
        egui::RichText::new(
            "Built-in AWS Earth Search Sentinel-2 L2A and COP-DEM GLO-30 work with zero auth. \
             Configure a custom STAC API endpoint and optional credentials below.",
        )
        .weak()
        .small(),
    );
    ui.add_space(4.0);

    egui::Grid::new("custom_stac_grid")
        .num_columns(2)
        .spacing([8.0, 6.0])
        .show(ui, |ui| {
            ui.label("STAC API URL:");
            ui.add(
                egui::TextEdit::singleline(&mut ui_state.draft.stac.custom_endpoint)
                    .desired_width(270.0)
                    .hint_text("https://earth-search.aws.element84.com/v1"),
            );
            ui.end_row();

            ui.label("Collection ID:");
            ui.add(
                egui::TextEdit::singleline(&mut ui_state.draft.stac.custom_collection)
                    .desired_width(270.0)
                    .hint_text("sentinel-2-l2a (optional)"),
            );
            ui.end_row();

            ui.label("Auth mode:");
            let prev_auth_kind = ui_state.draft.stac.auth_kind;
            egui::ComboBox::from_id_salt("stac_auth_kind")
                .selected_text(ui_state.draft.stac.auth_kind.label())
                .width(170.0)
                .show_ui(ui, |ui| {
                    for kind in StacAuthKind::ALL {
                        ui.selectable_value(&mut ui_state.draft.stac.auth_kind, kind, kind.label());
                    }
                });
            if ui_state.draft.stac.auth_kind != prev_auth_kind {
                // Scrub the hidden secret whenever auth mode changes so a Bearer
                // token can never leak into a SAS query string or linger on disk
                // after switching to `None`.
                ui_state.draft.stac.auth_secret.clear();
            }
            ui.end_row();

            if ui_state.draft.stac.auth_kind == StacAuthKind::ApiKey {
                ui.label("Header name:");
                ui.add(
                    egui::TextEdit::singleline(&mut ui_state.draft.stac.auth_header_name)
                        .desired_width(270.0)
                        .hint_text("x-api-key"),
                );
                ui.end_row();
            }

            if ui_state.draft.stac.auth_kind != StacAuthKind::None {
                let label = match ui_state.draft.stac.auth_kind {
                    StacAuthKind::Bearer => "Bearer token:",
                    StacAuthKind::ApiKey => "API key value:",
                    StacAuthKind::SasQuery => "SAS query:",
                    StacAuthKind::None => "Secret:",
                };
                let hint = match ui_state.draft.stac.auth_kind {
                    StacAuthKind::Bearer => "Paste bearer token",
                    StacAuthKind::ApiKey => "Paste API key",
                    StacAuthKind::SasQuery => "sv=2022-...&sig=...",
                    StacAuthKind::None => "",
                };
                ui.label(label);
                ui.add(
                    egui::TextEdit::singleline(&mut ui_state.draft.stac.auth_secret)
                        .password(true)
                        .desired_width(270.0)
                        .hint_text(hint),
                );
                ui.end_row();
            }
        });

    if let Some(err) = ui_state.draft.stac.validation_error() {
        ui.colored_label(ERROR_TEXT_COLOR, egui::RichText::new(err).small());
    } else if ui_state.draft.stac.auth_kind != StacAuthKind::None
        && ui_state.draft.stac.auth_secret.trim().is_empty()
    {
        ui.colored_label(
            crate::ui::WARN_TEXT_COLOR,
            egui::RichText::new("Secret is empty — requests will be sent without auth.").small(),
        );
    }
}

/// Render the floating unified Settings modal when open, returning any user
/// actions to apply in `VeldmapApp`.
pub(crate) fn show_settings_window(
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

    egui::Window::new("Settings")
        .open(&mut open)
        .resizable(false)
        .default_width(450.0)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut ui_state.active_tab, SettingsTab::Basemap, "Basemap");
                ui.selectable_value(
                    &mut ui_state.active_tab,
                    SettingsTab::CacheGpu,
                    "Cache & GPU",
                );
                ui.selectable_value(
                    &mut ui_state.active_tab,
                    SettingsTab::StacCloud,
                    "STAC / Cloud",
                );
            });
            ui.separator();

            match ui_state.active_tab {
                SettingsTab::Basemap => {
                    let _ = show_gibs_section(ui, ui_state);
                    ui.separator();
                    let _ = show_custom_xyz_section(ui, settings, ui_state);
                }
                SettingsTab::CacheGpu => {
                    show_disk_cache_section(
                        ui,
                        disk_status,
                        &mut ui_state.draft.disk_cache_max_mb,
                        &mut ui_state.draft.gpu_tile_cache_max_mb,
                        &mut actions,
                    );
                }
                SettingsTab::StacCloud => {
                    show_stac_section(ui, ui_state);
                }
            }

            // Evaluate validation AFTER tab widgets mutate `ui_state.draft` so
            // button enablement is always fresh on the same frame.
            let date_valid = validate_gibs_date(&ui_state.draft.gibs_date);
            let custom_valid = ui_state.custom_validation().is_ok();
            let basemap_valid = (ui_state.draft.active_basemap != BasemapSelection::NasaGibsViirs
                || date_valid)
                && (ui_state.draft.active_basemap != BasemapSelection::CustomXyz || custom_valid);
            let stac_valid = ui_state.draft.stac.validation_error().is_none();
            let can_apply = basemap_valid && stac_valid;

            ui.separator();
            ui.horizontal(|ui| {
                if ui
                    .add_enabled(can_apply, egui::Button::new("Apply & Save"))
                    .clicked()
                {
                    actions.push(BasemapUiAction::ApplyDraft(Box::new(
                        ui_state.draft.clone(),
                    )));
                }
                if ui_state.active_tab == SettingsTab::Basemap
                    && ui
                        .add_enabled(
                            custom_valid && stac_valid,
                            egui::Button::new("Use Custom XYZ"),
                        )
                        .clicked()
                {
                    let mut next = ui_state.draft.clone();
                    next.active_basemap = BasemapSelection::CustomXyz;
                    actions.push(BasemapUiAction::ApplyDraft(Box::new(next)));
                }
                if !basemap_valid && ui_state.active_tab != SettingsTab::Basemap {
                    ui.colored_label(
                        ERROR_TEXT_COLOR,
                        egui::RichText::new("Fix invalid Basemap settings before saving.").small(),
                    );
                } else if !stac_valid && ui_state.active_tab != SettingsTab::StacCloud {
                    ui.colored_label(
                        ERROR_TEXT_COLOR,
                        egui::RichText::new("Fix invalid STAC / Cloud settings before saving.")
                            .small(),
                    );
                }
            });
        });

    ui_state.settings_open = open;
    actions
}
