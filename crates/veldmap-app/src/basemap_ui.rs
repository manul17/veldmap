//! Basemap selector toolbar controls, Basemap & Cache Settings window, and
//! viewport attribution badge extracted out of `app.rs`.

use eframe::egui;
use veldmap_net::is_valid_iso_date;
use veldmap_storage::{AppSettings, BasemapSelection};

use crate::basemap::BasemapManager;
use crate::gpu_tiles::{GpuTileCache, PendingUploads};

/// Transient UI editing state for the Basemap & Cache Settings window.
#[derive(Clone, Debug)]
pub struct BasemapUiState {
    pub settings_open: bool,
    pub draft: AppSettings,
    pub cache_notice: Option<String>,
}

impl BasemapUiState {
    pub fn new(initial: &AppSettings) -> Self {
        Self {
            settings_open: false,
            draft: initial.clone(),
            cache_notice: None,
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

/// Render toolbar basemap selector, live streaming / offline pill, and
/// Settings button. Returns `true` if the active basemap or scene changed.
pub fn show_toolbar_controls(
    ui: &mut egui::Ui,
    manager: &mut BasemapManager,
    ui_state: &mut BasemapUiState,
    gpu_cache: &mut GpuTileCache,
    pending_uploads: &mut PendingUploads,
) -> bool {
    let mut changed = false;
    let current = manager.settings().active_basemap;

    egui::ComboBox::from_id_salt("basemap_selector")
        .selected_text(current.label())
        .width(175.0)
        .show_ui(ui, |ui| {
            for option in ALL_BASEMAPS {
                if ui
                    .selectable_label(option == current, option.label())
                    .clicked()
                    && option != current
                {
                    let mut next = manager.settings().clone();
                    next.active_basemap = option;
                    if option == BasemapSelection::CustomXyz
                        && next.custom_xyz.url_template.trim().is_empty()
                    {
                        ui_state.settings_open = true;
                    }
                    changed |= manager.apply_settings(next, gpu_cache, pending_uploads);
                    ui_state.draft = manager.settings().clone();
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
            ui_state.draft = manager.settings().clone();
            ui_state.cache_notice = None;
        }
    }

    // Live streaming / offline indicator badge.
    if current.is_live() {
        let status = manager.network_status();
        if let Some(cfg_err) = manager.config_error() {
            ui.label(
                egui::RichText::new("Invalid URL")
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
                manager.retry_live();
                changed = true;
            }
        } else if status.in_flight > 0 || status.queued > 0 {
            ui.label(
                egui::RichText::new(format!("Streaming ({})", status.in_flight + status.queued))
                    .weak()
                    .small(),
            );
        }
    }

    changed
}

/// Render the floating Basemap & Disk Cache Settings window when open.
/// Returns `true` if settings were applied and changed the scene.
pub fn show_settings_window(
    ctx: &egui::Context,
    manager: &mut BasemapManager,
    ui_state: &mut BasemapUiState,
    gpu_cache: &mut GpuTileCache,
    pending_uploads: &mut PendingUploads,
) -> bool {
    if !ui_state.settings_open {
        return false;
    }

    let mut changed = false;
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
            if !is_valid_iso_date(ui_state.draft.gibs_date.trim()) {
                ui.label(
                    egui::RichText::new("Enter a valid YYYY-MM-DD date (e.g. 2024-07-01).")
                        .color(egui::Color32::from_rgb(240, 185, 85))
                        .small(),
                );
            }

            ui.separator();
            ui.heading("Custom XYZ / TMS Tile Source");
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
                            .hint_text("https://host/{z}/{x}/{y}.png?key={key}"),
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

            if let Some(err) = manager.config_error() {
                ui.label(
                    egui::RichText::new(err)
                        .color(egui::Color32::from_rgb(235, 110, 95))
                        .small(),
                );
            }

            ui.separator();
            ui.heading("SQLite WAL Tile Disk Cache");
            if let Some(stats) = manager.disk_cache_stats() {
                let used_mb = (stats.payload_bytes as f64) / (1024.0 * 1024.0);
                let cap_mb = (stats.max_bytes as f64) / (1024.0 * 1024.0);
                ui.monospace(format!(
                    "Cached: {} tiles · {:.1} MiB / {:.0} MiB",
                    stats.tile_count, used_mb, cap_mb
                ));
            } else {
                ui.label(egui::RichText::new("Disk cache unavailable in this session.").weak());
            }

            ui.horizontal(|ui| {
                ui.label("Disk cap (MiB):");
                ui.add(egui::Slider::new(
                    &mut ui_state.draft.disk_cache_max_mb,
                    16..=4096,
                ));
            });

            ui.horizontal(|ui| {
                if ui.button("Clear Disk Cache").clicked() {
                    match manager.clear_disk_cache() {
                        Ok(()) => {
                            ui_state.cache_notice = Some("Disk cache cleared.".to_owned());
                            changed = true;
                        }
                        Err(err) => {
                            ui_state.cache_notice = Some(format!("Clear failed: {err}"));
                        }
                    }
                }
                if let Some(notice) = &ui_state.cache_notice {
                    ui.label(egui::RichText::new(notice).weak().small());
                }
            });

            ui.separator();
            ui.horizontal(|ui| {
                if ui.button("Apply & Save").clicked() {
                    let mut next = ui_state.draft.clone();
                    next.active_basemap = manager.settings().active_basemap;
                    next.show_graticule = manager.settings().show_graticule;
                    changed |= manager.apply_settings(next, gpu_cache, pending_uploads);
                    ui_state.draft = manager.settings().clone();
                }
                if ui.button("Use Custom XYZ").clicked() {
                    let mut next = ui_state.draft.clone();
                    next.active_basemap = BasemapSelection::CustomXyz;
                    next.show_graticule = manager.settings().show_graticule;
                    changed |= manager.apply_settings(next, gpu_cache, pending_uploads);
                    ui_state.draft = manager.settings().clone();
                }
            });
        });

    ui_state.settings_open = open;
    changed
}

/// Paint a compact, high-contrast attribution pill in the bottom-right corner
/// of the globe viewport so NASA GIBS and EOX CC BY 4.0 credits remain visible.
pub fn paint_attribution_badge(painter: &egui::Painter, rect: egui::Rect, attribution: &str) {
    if attribution.is_empty() || rect.width() < 160.0 || rect.height() < 60.0 {
        return;
    }

    let font = egui::FontId::proportional(11.0);
    let max_width = (rect.width() - 28.0).clamp(120.0, 560.0);
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
