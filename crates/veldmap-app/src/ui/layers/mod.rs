//! Collapsible Layer List side panel for local raster overlays (GeoTIFF/COG,
//! PNG/JPEG + world files) and read-only GeoJSON vector overlays, with layer
//! reordering, CRS/radiometry metadata inspection, diagnostic banners, detail
//! limit captions, and loading shimmer indicators.

mod raster_card;
mod style_controls;
mod vector_card;

use std::path::PathBuf;

use eframe::egui;
use veldmap_core::geo::FootprintBounds;
use veldmap_core::tile::TileStreamSpec;

use crate::app::UnavailableLayerNotice;
use crate::colormap::{CategoricalPaletteId, ColormapId};
use crate::layers::{LayerId, LayerSummary, StretchPreset};
use crate::vectors::{VectorLayerId, VectorLayerSummary};

use super::{ERROR_TEXT_COLOR, WARN_TEXT_COLOR, submitted_with_enter};

use raster_card::show_raster_card;
use vector_card::show_vector_card;

/// User intent emitted by the Layer List panel, applied by `VeldmapApp`.
#[derive(Clone, Debug)]
pub(crate) enum LayerUiAction {
    OpenFileDialog,
    OpenPath(PathBuf),
    RetryUnavailableSessionLayers,
    HideUnavailableSessionNotice,
    ForgetUnavailableSessionLayers,
    ToggleVisibility(LayerId, bool),
    SetOpacity(LayerId, f32, bool),
    SetStream(LayerId, TileStreamSpec),
    SetStretchPreset(LayerId, StretchPreset),
    SetStretchRange(LayerId, f64, f64, bool),
    SetColormap(LayerId, ColormapId, bool),
    SetCategoricalPalette(LayerId, CategoricalPaletteId),
    MoveUp(LayerId),
    MoveDown(LayerId),
    OpenCrsPicker(LayerId, String),
    OpenCrsPickerForPendingPath(PathBuf),
    ZoomToExtent(FootprintBounds),
    ReloadBypassCache(LayerId),
    Remove(LayerId),
    ToggleVectorVisibility(VectorLayerId, bool),
    SetVectorOpacity(VectorLayerId, f32, bool),
    RemoveVector(VectorLayerId),
}

/// Transient UI state for the Layer List side panel and File Open input.
#[derive(Clone, Debug, Default)]
pub(crate) struct LayersPanelState {
    open: bool,
    open_prompt_visible: bool,
    focus_path_input: bool,
    path_input: String,
    error: Option<String>,
    failed_unbound_path: Option<PathBuf>,
}

impl LayersPanelState {
    pub(crate) fn is_open(&self) -> bool {
        self.open
    }

    pub(crate) fn toggle_open(&mut self) {
        self.open = !self.open;
    }

    pub(crate) fn open_panel(&mut self) {
        self.open = true;
    }

    pub(crate) fn on_layer_opened(&mut self) {
        self.open = true;
        self.path_input.clear();
        self.open_prompt_visible = false;
        self.failed_unbound_path = None;
    }

    pub(crate) fn show_open_prompt(&mut self) {
        self.open = true;
        self.open_prompt_visible = true;
        self.focus_path_input = true;
        self.error = None;
    }

    pub(crate) fn push_error(&mut self, msg: String) {
        self.open = true;
        match &mut self.error {
            Some(existing) if !existing.is_empty() => {
                existing.push('\n');
                existing.push_str(&msg);
            }
            slot => *slot = Some(msg),
        }
    }

    pub(crate) fn note_failed_path(&mut self, path: PathBuf, msg: String, can_assign_crs: bool) {
        if can_assign_crs {
            self.failed_unbound_path = Some(path);
        }
        self.push_error(msg);
    }

    pub(crate) fn clear_error(&mut self) {
        self.error = None;
        self.failed_unbound_path = None;
    }
}

fn show_panel_header(
    ui: &mut egui::Ui,
    layer_count: usize,
    state: &mut LayersPanelState,
    actions: &mut Vec<LayerUiAction>,
) {
    ui.horizontal(|ui| {
        ui.strong(format!("Layers ({layer_count})"));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.small_button("×").on_hover_text("Hide Layers").clicked() {
                state.open = false;
            }
            if ui
                .small_button("Path…")
                .on_hover_text("Type a local file path (fallback for minimal WMs)")
                .clicked()
            {
                state.open_prompt_visible = !state.open_prompt_visible;
                if state.open_prompt_visible {
                    state.focus_path_input = true;
                }
            }
            if ui
                .small_button("Open…")
                .on_hover_text("Open raster or GeoJSON via native file dialog")
                .clicked()
            {
                actions.push(LayerUiAction::OpenFileDialog);
            }
        });
    });
    ui.separator();
}

fn show_open_prompt(
    ui: &mut egui::Ui,
    state: &mut LayersPanelState,
    actions: &mut Vec<LayerUiAction>,
) {
    ui.label(
        egui::RichText::new("Local GeoTIFF / PNG / JPEG / GeoJSON path (or drag & drop):").small(),
    );
    ui.horizontal(|ui| {
        let resp = ui.add(
            egui::TextEdit::singleline(&mut state.path_input)
                .desired_width((ui.available_width() - 56.0).max(120.0))
                .hint_text("/path/to/orthophoto.tif"),
        );
        if state.focus_path_input {
            resp.request_focus();
            state.focus_path_input = false;
        }
        if resp.changed() {
            state.clear_error();
        }
        let enter = submitted_with_enter(ui, &resp);
        if ui.button("Load").clicked() || enter {
            let trimmed = state.path_input.trim().trim_matches(['"', '\'']);
            if !trimmed.is_empty() {
                actions.push(LayerUiAction::OpenPath(PathBuf::from(trimmed)));
            }
        }
    });
    ui.separator();
}

fn show_unavailable_session_banner(
    ui: &mut egui::Ui,
    unavailable: &[UnavailableLayerNotice],
    actions: &mut Vec<LayerUiAction>,
) {
    if unavailable.is_empty() {
        return;
    }
    let count = unavailable.len();
    let noun = if count == 1 { "layer" } else { "layers" };
    egui::Frame::group(ui.style()).show(ui, |ui| {
        ui.colored_label(
            WARN_TEXT_COLOR,
            format!("{count} saved session {noun} unavailable"),
        );
        ui.small(
            egui::RichText::new("Preserved in session.json with style/CRS settings intact.").weak(),
        );
        for item in unavailable.iter().take(4) {
            let kind_tag = if item.is_vector { "Vector" } else { "Raster" };
            ui.add(
                egui::Label::new(
                    egui::RichText::new(format!("• [{kind_tag}] {}", item.name)).small(),
                )
                .truncate(),
            )
            .on_hover_text(format!("{}\n{}", item.path.display(), item.error));
        }
        if count > 4 {
            ui.small(egui::RichText::new(format!("… and {} more", count - 4)).weak());
        }
        ui.horizontal(|ui| {
            if ui
                .small_button("Retry")
                .on_hover_text("Retry opening unavailable layers from the previous session")
                .clicked()
            {
                actions.push(LayerUiAction::RetryUnavailableSessionLayers);
            }
            if ui
                .small_button("Hide")
                .on_hover_text("Hide this notice for now (keeps layers saved in session.json)")
                .clicked()
            {
                actions.push(LayerUiAction::HideUnavailableSessionNotice);
            }
            if ui
                .small_button("Forget")
                .on_hover_text("Remove unavailable layers from saved session.json")
                .clicked()
            {
                actions.push(LayerUiAction::ForgetUnavailableSessionLayers);
            }
        });
    });
    ui.separator();
}

/// Render the collapsible left Layer List panel and return user actions.
pub(crate) fn show_layers_panel(
    ui: &mut egui::Ui,
    layers: &[LayerSummary],
    vectors: &[VectorLayerSummary],
    unavailable_session_layers: &[UnavailableLayerNotice],
    state: &mut LayersPanelState,
) -> Vec<LayerUiAction> {
    let mut actions = Vec::new();
    if !state.open {
        return actions;
    }

    let total = layers.len() + vectors.len();
    egui::Panel::left("layer_list_panel")
        .resizable(true)
        .default_size(285.0)
        .min_size(220.0)
        .max_size(440.0)
        .show(ui, |ui| {
            show_panel_header(ui, total, state, &mut actions);

            if state.open_prompt_visible {
                show_open_prompt(ui, state, &mut actions);
            }

            if let Some(err) = &state.error {
                ui.horizontal_wrapped(|ui| {
                    ui.colored_label(ERROR_TEXT_COLOR, err);
                });
                ui.horizontal(|ui| {
                    if let Some(path) = state.failed_unbound_path.clone()
                        && ui
                            .small_button("Assign CRS…")
                            .on_hover_text("Pick a CRS and retry opening this file")
                            .clicked()
                    {
                        actions.push(LayerUiAction::OpenCrsPickerForPendingPath(path));
                    }
                    if ui.small_button("Dismiss").clicked() {
                        state.clear_error();
                    }
                });
                ui.separator();
            }

            show_unavailable_session_banner(ui, unavailable_session_layers, &mut actions);

            if total == 0 {
                ui.weak("No raster or GeoJSON layers loaded.");
                ui.weak(
                    "Drop a GeoTIFF/PNG/JPEG/GeoJSON onto the globe, pass a path on the CLI, or click Open….",
                );
                return;
            }

            egui::ScrollArea::vertical().show(ui, |ui| {
                if !vectors.is_empty() {
                    ui.small(egui::RichText::new("Vector Overlays (GeoJSON)").strong());
                    ui.add_space(2.0);
                    for vec_layer in vectors.iter().rev() {
                        show_vector_card(ui, vec_layer, &mut actions);
                    }
                    if !layers.is_empty() {
                        ui.separator();
                    }
                }
                if !layers.is_empty() {
                    ui.small(egui::RichText::new("Raster Overlays").strong());
                    ui.add_space(2.0);
                    // Display top-to-bottom (reverse of painter's order), keyed
                    // strictly by `LayerId` so removals/reorders never shift widget state.
                    for layer in layers.iter().rev() {
                        show_raster_card(ui, layer, &mut actions);
                    }
                }
            });
        });

    actions
}
