//! `Layers` workbench tab for local and remote HTTP Range COG raster overlays
//! (GeoTIFF/BigTIFF/COG, PNG/JPEG + world files) and read-only GeoJSON vector
//! overlays, with layer reordering, analytical track/stretch/palette controls,
//! CRS/radiometry metadata inspection, diagnostic banners, and loading shimmer
//! indicators.

mod band_math_builder;
mod band_math_card;
mod histogram;
mod raster_card;
mod style_controls;
mod vector_card;

use std::path::PathBuf;

use eframe::egui;
use veldmap_core::geo::FootprintBounds;
use veldmap_core::tile::TileStreamSpec;

use crate::layers::{LayerId, LayerSummary, LocalBandMathSpec, OverlayEdit};
use crate::vectors::{VectorEdit, VectorLayerId, VectorLayerSummary};

use super::widgets::submitted_with_enter;
use super::{ERROR_TEXT_COLOR, WARN_TEXT_COLOR};

use band_math_builder::{BandMathBuilderState, has_band_math_inputs, show_band_math_builder};
use raster_card::show_raster_card;
use vector_card::show_vector_card;

/// Immutable summary of a layer from the previous `session.json` that could not
/// be reopened at startup (kept in `session.json` until reopened or explicitly
/// forgotten so custom CRS overrides and stretch styles are never lost).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct UnavailableLayerNotice {
    pub(crate) name: String,
    pub(crate) path: PathBuf,
    pub(crate) error: String,
    pub(crate) is_vector: bool,
}

/// Session persistence policy for an infallible [`OverlayEdit`] or [`VectorEdit`]
/// emitted from UI.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EditPersist {
    /// Persist `session.json` whenever the edit actually changed layer state.
    OnChange,
    /// Persist `session.json` only on explicit commit (e.g. slider/histogram
    /// drag end, where the final frame may have `changed == false`).
    OnCommit(bool),
}

/// User intent emitted by the Layer List panel, applied by `VeldmapApp`.
#[derive(Clone, Debug)]
pub(crate) enum LayerUiAction {
    OpenFileDialog,
    OpenPath(PathBuf),
    RetryUnavailableSessionLayers,
    HideUnavailableSessionNotice,
    ForgetUnavailableSessionLayers,
    EditRaster {
        id: LayerId,
        edit: OverlayEdit,
        persist: EditPersist,
    },
    SetStream(LayerId, TileStreamSpec),
    OpenCrsPicker(LayerId, String),
    OpenCrsPickerForPendingPath(PathBuf),
    ZoomToExtent(FootprintBounds),
    ReloadBypassCache(LayerId),
    EditVector {
        id: VectorLayerId,
        edit: VectorEdit,
        persist: EditPersist,
    },
    UseVectorAsStacAoi(VectorLayerId),
    CreateBandMath(LocalBandMathSpec),
}

impl LayerUiAction {
    fn edit_raster(id: LayerId, edit: OverlayEdit) -> Self {
        Self::EditRaster {
            id,
            edit,
            persist: EditPersist::OnChange,
        }
    }

    fn edit_raster_drag(id: LayerId, edit: OverlayEdit, commit: bool) -> Self {
        Self::EditRaster {
            id,
            edit,
            persist: EditPersist::OnCommit(commit),
        }
    }

    fn edit_vector(id: VectorLayerId, edit: VectorEdit) -> Self {
        Self::EditVector {
            id,
            edit,
            persist: EditPersist::OnChange,
        }
    }

    fn edit_vector_drag(id: VectorLayerId, edit: VectorEdit, commit: bool) -> Self {
        Self::EditVector {
            id,
            edit,
            persist: EditPersist::OnCommit(commit),
        }
    }
}

/// Transient UI state for the Layer List tab and File Open input.
#[derive(Clone, Debug, Default)]
pub(crate) struct LayersPanelState {
    open_prompt_visible: bool,
    focus_path_input: bool,
    path_input: String,
    error: Option<String>,
    failed_unbound_path: Option<PathBuf>,
    band_math_builder: BandMathBuilderState,
}

impl LayersPanelState {
    pub(crate) fn on_layer_opened(&mut self) {
        self.path_input.clear();
        self.open_prompt_visible = false;
        self.failed_unbound_path = None;
    }

    pub(crate) fn on_band_math_created(&mut self) {
        self.clear_error();
        self.band_math_builder.on_created();
    }

    pub(crate) fn show_open_prompt(&mut self) {
        self.open_prompt_visible = true;
        self.focus_path_input = true;
        self.error = None;
    }

    pub(crate) fn push_error(&mut self, msg: String) {
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
    has_raster_sources: bool,
    state: &mut LayersPanelState,
    actions: &mut Vec<LayerUiAction>,
) {
    ui.horizontal(|ui| {
        ui.strong(format!("Loaded ({layer_count})"));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
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
            if has_raster_sources
                && ui
                    .selectable_label(
                        state.band_math_builder.open,
                        egui::RichText::new("+ Math").small(),
                    )
                    .on_hover_text("Build a Band Math / RGB composite from loaded raster layers")
                    .clicked()
            {
                state.band_math_builder.open = !state.band_math_builder.open;
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

/// Render the Layers tab contents inside the left workbench panel and return
/// user actions.
pub(crate) fn show_layers_tab_contents(
    ui: &mut egui::Ui,
    layers: &[LayerSummary],
    vectors: &[VectorLayerSummary],
    unavailable_session_layers: &[UnavailableLayerNotice],
    state: &mut LayersPanelState,
) -> Vec<LayerUiAction> {
    let mut actions = Vec::new();
    let total = layers.len() + vectors.len();
    let can_build_band_math = has_band_math_inputs(layers);
    if !can_build_band_math {
        state.band_math_builder.open = false;
    }

    show_panel_header(ui, total, can_build_band_math, state, &mut actions);

    if state.open_prompt_visible {
        show_open_prompt(ui, state, &mut actions);
    }

    if state.band_math_builder.open {
        show_band_math_builder(ui, layers, &mut state.band_math_builder, &mut actions);
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
            "Drop a GeoTIFF/PNG/JPEG/GeoJSON onto the globe, click Open…, or stream a public COG from the Cloud / STAC tab.",
        );
        return actions;
    }

    egui::ScrollArea::vertical()
        .id_salt("layers_tab_scroll")
        .show(ui, |ui| {
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

    actions
}
