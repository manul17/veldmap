//! Top workstation toolbar (native File Open, Recent files, Bookmarks,
//! Layers toggle, Geodesic Measure toggle, Go-To navigation, basemap selector,
//! graticule toggle, and GPU adapter badge).
//!
//! Strictly action-only: emits [`ToolbarAction`] values back to `VeldmapApp`
//! without mutating `Camera`, storage, or scene state directly from UI layout
//! code.

use std::path::PathBuf;

use eframe::egui;
use veldmap_core::geo::{ParsedLocation, parse_location};

use crate::basemap::BasemapStatusSnapshot;
use crate::session::{CameraBookmark, MAX_BOOKMARKS};
use crate::settings::AppSettings;

use super::ERROR_TEXT_COLOR;
use super::basemap::{BasemapUiAction, BasemapUiState, show_toolbar_controls};
use super::format;

/// User intent emitted by the top workstation toolbar and applied by
/// `VeldmapApp`.
#[derive(Clone, Debug)]
pub(crate) enum ToolbarAction {
    OpenFileDialog,
    OpenPathPrompt,
    OpenRecentFile(PathBuf),
    ClearRecentFiles,
    ToggleLayersPanel,
    ToggleMeasureTool,
    SaveBookmark(String),
    GoToBookmark(CameraBookmark),
    DeleteBookmark(usize),
    FocusLocation(ParsedLocation),
    Basemap(BasemapUiAction),
    SetGraticule(bool),
}

/// Read-only inputs rendered by the top workstation toolbar.
#[derive(Clone, Copy)]
pub(crate) struct ToolbarInputs<'a> {
    pub(crate) settings: &'a AppSettings,
    pub(crate) basemap_status: &'a BasemapStatusSnapshot,
    pub(crate) adapter_label: &'a str,
    pub(crate) layer_count: usize,
    pub(crate) layers_open: bool,
    pub(crate) measure_active: bool,
    pub(crate) recent_files: &'a [PathBuf],
    pub(crate) bookmarks: &'a [CameraBookmark],
}

/// Transient UI state for the Go-To coordinate input and bookmark name input.
#[derive(Clone, Debug, Default)]
pub(crate) struct GoToBarState {
    input: String,
    error: Option<String>,
    bookmark_name: String,
}

impl GoToBarState {
    fn trigger_goto(&mut self) -> Option<ParsedLocation> {
        match parse_location(&self.input) {
            Ok(loc) => {
                self.error = None;
                Some(loc)
            }
            Err(err) => {
                self.error = Some(err.to_string());
                None
            }
        }
    }
}

/// Render the top toolbar and return user actions to apply in `VeldmapApp`.
pub(crate) fn show_top_toolbar(
    ui: &mut egui::Ui,
    inputs: ToolbarInputs<'_>,
    goto: &mut GoToBarState,
    basemap_ui: &mut BasemapUiState,
) -> Vec<ToolbarAction> {
    let mut actions = Vec::new();
    egui::Panel::top("top_toolbar").show(ui, |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.strong(veldmap_core::APP_NAME).on_hover_text(format!(
                "GPU: {}\nControls: Drag orbit · Right/Shift+drag or Q/E rotate · WASD/Arrows pan · Scroll/+/- zoom · N reset North",
                inputs.adapter_label
            ));
            ui.separator();

            if ui
                .button("Open…")
                .on_hover_text("Open local GeoTIFF / PNG / JPEG / GeoJSON via native picker")
                .clicked()
            {
                actions.push(ToolbarAction::OpenFileDialog);
            }

            ui.menu_button("Recent", |ui| {
                if ui.button("Open by path…").clicked() {
                    actions.push(ToolbarAction::OpenPathPrompt);
                    ui.close();
                }
                if !inputs.recent_files.is_empty() {
                    ui.separator();
                    for path in inputs.recent_files {
                        let label = path
                            .file_name()
                            .and_then(|n| n.to_str())
                            .map_or_else(|| path.display().to_string(), str::to_owned);
                        if ui
                            .button(&label)
                            .on_hover_text(path.display().to_string())
                            .clicked()
                        {
                            actions.push(ToolbarAction::OpenRecentFile(path.clone()));
                            ui.close();
                        }
                    }
                    ui.separator();
                    if ui.button("Clear Recent").clicked() {
                        actions.push(ToolbarAction::ClearRecentFiles);
                        ui.close();
                    }
                } else {
                    ui.separator();
                    ui.weak("No recent files");
                }
            });

            let layers_label = if inputs.layer_count > 0 {
                format!("Layers ({})", inputs.layer_count)
            } else {
                "Layers".to_owned()
            };
            if ui
                .selectable_label(inputs.layers_open, layers_label)
                .on_hover_text("Toggle Layer List panel")
                .clicked()
            {
                actions.push(ToolbarAction::ToggleLayersPanel);
            }

            if ui
                .selectable_label(inputs.measure_active, "Measure")
                .on_hover_text("WGS84 Karney geodesic distance & area tool")
                .clicked()
            {
                actions.push(ToolbarAction::ToggleMeasureTool);
            }

            ui.menu_button("Bookmarks", |ui| {
                let at_cap = inputs.bookmarks.len() >= MAX_BOOKMARKS;
                ui.horizontal(|ui| {
                    let bm_resp = ui.add(
                        egui::TextEdit::singleline(&mut goto.bookmark_name)
                            .desired_width(130.0)
                            .hint_text("Bookmark name"),
                    );
                    let enter =
                        bm_resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                    let save_clicked = ui
                        .add_enabled(!at_cap, egui::Button::new("Save View"))
                        .on_disabled_hover_text(format!(
                            "Bookmark limit reached ({MAX_BOOKMARKS}); delete one to add more."
                        ))
                        .clicked();
                    if !at_cap && (save_clicked || enter) {
                        let name = goto.bookmark_name.trim().to_owned();
                        actions.push(ToolbarAction::SaveBookmark(name));
                        goto.bookmark_name.clear();
                    }
                });
                if !inputs.bookmarks.is_empty() {
                    ui.separator();
                    for (idx, bm) in inputs.bookmarks.iter().enumerate() {
                        ui.horizontal(|ui| {
                            let tip = format!(
                                "{} · Alt {} · Hdg {:.1}°",
                                format::format_lat_lon(bm.pose.lat_deg, bm.pose.lon_deg),
                                format::format_distance_m(bm.pose.height_m),
                                bm.pose.heading_deg
                            );
                            if ui.button(&bm.name).on_hover_text(tip).clicked() {
                                actions.push(ToolbarAction::GoToBookmark(bm.clone()));
                                ui.close();
                            }
                            if ui
                                .small_button("×")
                                .on_hover_text("Delete bookmark")
                                .clicked()
                            {
                                actions.push(ToolbarAction::DeleteBookmark(idx));
                            }
                        });
                    }
                } else {
                    ui.separator();
                    ui.weak("No saved camera bookmarks");
                }
            });

            ui.separator();

            ui.label("Go to:");
            let text_edit = egui::TextEdit::singleline(&mut goto.input)
                .desired_width(160.0)
                .hint_text("DD, DMS, or MGRS (e.g. 47.23, 39.70 / 37UDQ)");
            let edit_resp = ui.add(text_edit);
            if edit_resp.changed() {
                goto.error = None;
            }
            let enter_pressed =
                edit_resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
            if (ui.button("Fly").clicked() || enter_pressed)
                && let Some(loc) = goto.trigger_goto()
            {
                actions.push(ToolbarAction::FocusLocation(loc));
            }

            ui.separator();
            actions.extend(
                show_toolbar_controls(ui, inputs.settings, inputs.basemap_status, basemap_ui)
                    .into_iter()
                    .map(ToolbarAction::Basemap),
            );

            ui.separator();
            let mut show_graticule = inputs.settings.show_graticule;
            if ui.checkbox(&mut show_graticule, "Graticule").changed() {
                actions.push(ToolbarAction::SetGraticule(show_graticule));
            }

            if let Some(err) = &goto.error {
                ui.separator();
                ui.add(
                    egui::Label::new(egui::RichText::new(err).color(ERROR_TEXT_COLOR).small())
                        .truncate(),
                )
                .on_hover_text(err);
            }
        });
    });
    actions
}
