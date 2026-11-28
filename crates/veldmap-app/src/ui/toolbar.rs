//! Top workstation toolbar (native File Open, Project save/open & Export PNG
//! menu, Recent files, Bookmarks, `Layers` and `Cloud / STAC` workbench
//! toggles, Geodesic Measure & Split-Screen Swipe toggles, Go-To navigation,
//! basemap selector, graticule toggle, and GPU adapter badge).
//!
//! Strictly action-only: emits [`ToolbarAction`] values back to `VeldmapApp`
//! without mutating `Camera`, storage, or scene state directly from UI layout
//! code.

use std::path::PathBuf;

use eframe::egui;
use veldmap_core::geo::{ParsedLocation, parse_location};

use crate::basemap::BasemapStatusSnapshot;
use crate::display_name_from_path;
use crate::storage::session::{CameraBookmark, MAX_BOOKMARKS};
use crate::storage::settings::BasemapSelection;

use super::basemap::{BasemapUiAction, show_toolbar_controls};
use super::widgets::submitted_with_enter;
use super::{ERROR_TEXT_COLOR, WARN_TEXT_COLOR, format};

/// User intent emitted by the top workstation toolbar and applied by
/// `VeldmapApp`.
#[derive(Clone, Debug)]
pub(crate) enum ToolbarAction {
    OpenFileDialog,
    OpenPathPrompt,
    OpenRecentFile(PathBuf),
    ClearRecentFiles,
    SaveProjectDialog,
    OpenProjectDialog,
    OpenExportPngModal,
    ToggleLayersPanel,
    ToggleCloudPanel,
    ShowUnavailableSessionLayers,
    ToggleMeasureTool,
    ToggleSwipeTool,
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
    pub(crate) active_basemap: BasemapSelection,
    pub(crate) custom_xyz_configured: bool,
    pub(crate) show_graticule: bool,
    pub(crate) basemap_status: &'a BasemapStatusSnapshot,
    pub(crate) adapter_label: &'a str,
    pub(crate) layer_count: usize,
    pub(crate) raster_layer_count: usize,
    pub(crate) unavailable_layer_count: usize,
    pub(crate) layers_open: bool,
    pub(crate) cloud_open: bool,
    pub(crate) measure_active: bool,
    pub(crate) swipe_active: bool,
    pub(crate) recent_files: &'a [PathBuf],
    pub(crate) bookmarks: &'a [CameraBookmark],
}

/// Transient UI state for the toolbar Go-To coordinate input and bookmark name
/// input.
#[derive(Clone, Debug, Default)]
pub(crate) struct ToolbarState {
    input: String,
    error: Option<String>,
    bookmark_name: String,
}

impl ToolbarState {
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

fn show_recent_menu(ui: &mut egui::Ui, recent_files: &[PathBuf], actions: &mut Vec<ToolbarAction>) {
    ui.menu_button("Recent", |ui| {
        if ui.button("Open by path…").clicked() {
            actions.push(ToolbarAction::OpenPathPrompt);
            ui.close();
        }
        if !recent_files.is_empty() {
            ui.separator();
            for path in recent_files {
                let label = display_name_from_path(path);
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
}

fn show_bookmarks_menu(
    ui: &mut egui::Ui,
    bookmarks: &[CameraBookmark],
    state: &mut ToolbarState,
    actions: &mut Vec<ToolbarAction>,
) {
    ui.menu_button("Bookmarks", |ui| {
        let at_cap = bookmarks.len() >= MAX_BOOKMARKS;
        ui.horizontal(|ui| {
            let bm_resp = ui.add(
                egui::TextEdit::singleline(&mut state.bookmark_name)
                    .desired_width(130.0)
                    .hint_text("Bookmark name"),
            );
            let enter = submitted_with_enter(ui, &bm_resp);
            let save_clicked = ui
                .add_enabled(!at_cap, egui::Button::new("Save View"))
                .on_disabled_hover_text(format!(
                    "Bookmark limit reached ({MAX_BOOKMARKS}); delete one to add more."
                ))
                .clicked();
            if !at_cap && (save_clicked || enter) {
                let name = state.bookmark_name.trim().to_owned();
                actions.push(ToolbarAction::SaveBookmark(name));
                state.bookmark_name.clear();
            }
        });
        if !bookmarks.is_empty() {
            ui.separator();
            for (idx, bm) in bookmarks.iter().enumerate() {
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
}

fn show_goto_controls(
    ui: &mut egui::Ui,
    state: &mut ToolbarState,
    actions: &mut Vec<ToolbarAction>,
) {
    ui.label("Go to:");
    let text_edit = egui::TextEdit::singleline(&mut state.input)
        .desired_width(160.0)
        .hint_text("DD, DMS, or MGRS (e.g. 47.23, 39.70 / 37UDQ)");
    let edit_resp = ui.add(text_edit);
    if edit_resp.changed() {
        state.error = None;
    }
    let enter_pressed = submitted_with_enter(ui, &edit_resp);
    if (ui.button("Fly").clicked() || enter_pressed)
        && let Some(loc) = state.trigger_goto()
    {
        actions.push(ToolbarAction::FocusLocation(loc));
    }
}

/// Render the top toolbar and return user actions to apply in `VeldmapApp`.
pub(crate) fn show_top_toolbar(
    ui: &mut egui::Ui,
    inputs: ToolbarInputs<'_>,
    state: &mut ToolbarState,
) -> Vec<ToolbarAction> {
    let mut actions = Vec::new();
    egui::Panel::top("top_toolbar").show(ui, |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.strong(veldmap_core::APP_NAME).on_hover_text(format!(
                "GPU: {}\nControls: Drag orbit · Right/Shift+drag or Q/E rotate · WASD/Arrows pan · Scroll/+/- zoom · N reset North · Right-click or Ctrl/Cmd+Shift+C copy coords · Esc exit Swipe/AOI",
                inputs.adapter_label
            ));
            ui.separator();

            if ui
                .button("Open…")
                .on_hover_text(
                    "Open local GeoTIFF / world-file PNG or JPEG / GeoJSON / .veldmap.json project via native picker",
                )
                .clicked()
            {
                actions.push(ToolbarAction::OpenFileDialog);
            }

            ui.menu_button("Project", |ui| {
                if ui
                    .button("Save Project (.veldmap.json)…")
                    .on_hover_text(
                        "Save layers, sanitized cloud COGs, Band Math recipes, swipe & camera pose",
                    )
                    .clicked()
                {
                    actions.push(ToolbarAction::SaveProjectDialog);
                    ui.close();
                }
                if ui
                    .button("Open Project (.veldmap.json)…")
                    .on_hover_text("Load a saved .veldmap.json workspace project")
                    .clicked()
                {
                    actions.push(ToolbarAction::OpenProjectDialog);
                    ui.close();
                }
                ui.separator();
                if ui
                    .button("Export PNG…")
                    .on_hover_text(
                        "Export the current globe viewport as a cartographic PNG with optional overlays",
                    )
                    .clicked()
                {
                    actions.push(ToolbarAction::OpenExportPngModal);
                    ui.close();
                }
            });

            show_recent_menu(ui, inputs.recent_files, &mut actions);

            let layers_label = super::workbench::format_layers_tab_label(inputs.layer_count);
            if ui
                .selectable_label(inputs.layers_open, layers_label)
                .on_hover_text("Toggle Layers workbench tab")
                .clicked()
            {
                actions.push(ToolbarAction::ToggleLayersPanel);
            }

            if ui
                .selectable_label(inputs.cloud_open, "Cloud / STAC")
                .on_hover_text("Browse public STAC catalogs (Sentinel-2, COP-DEM) and stream COGs")
                .clicked()
            {
                actions.push(ToolbarAction::ToggleCloudPanel);
            }

            if inputs.unavailable_layer_count > 0 {
                let badge = format!("{} missing", inputs.unavailable_layer_count);
                if ui
                    .button(egui::RichText::new(badge).color(WARN_TEXT_COLOR).small())
                    .on_hover_text(
                        "Saved workspace layers could not be reopened (preserved in session.json). Click to view.",
                    )
                    .clicked()
                {
                    actions.push(ToolbarAction::ShowUnavailableSessionLayers);
                }
            }

            if ui
                .selectable_label(inputs.measure_active, "Measure")
                .on_hover_text("WGS84 Karney geodesic distance & area tool")
                .clicked()
            {
                actions.push(ToolbarAction::ToggleMeasureTool);
            }

            let can_swipe = inputs.raster_layer_count > 0 || inputs.swipe_active;
            if ui
                .add_enabled(
                    can_swipe,
                    egui::Button::selectable(inputs.swipe_active, "Swipe"),
                )
                .on_hover_text("Interactive split-screen swipe curtain across any two raster layers")
                .on_disabled_hover_text("Load at least one raster layer to compare with Swipe")
                .clicked()
            {
                actions.push(ToolbarAction::ToggleSwipeTool);
            }

            show_bookmarks_menu(ui, inputs.bookmarks, state, &mut actions);

            ui.separator();
            show_goto_controls(ui, state, &mut actions);

            ui.separator();
            actions.extend(
                show_toolbar_controls(
                    ui,
                    inputs.active_basemap,
                    inputs.custom_xyz_configured,
                    inputs.basemap_status,
                )
                .into_iter()
                .map(ToolbarAction::Basemap),
            );

            ui.separator();
            let mut show_graticule = inputs.show_graticule;
            if ui.checkbox(&mut show_graticule, "Graticule").changed() {
                actions.push(ToolbarAction::SetGraticule(show_graticule));
            }

            if let Some(err) = &state.error {
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
