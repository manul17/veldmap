//! Collapsible Layer List side panel for local raster overlays (GeoTIFF/COG,
//! PNG/JPEG + world files) and read-only GeoJSON vector overlays, with layer
//! reordering, CRS/radiometry metadata inspection, diagnostic banners, detail
//! limit captions, and loading shimmer indicators.

use std::path::PathBuf;

use eframe::egui;
use veldmap_core::geo::FootprintBounds;

use crate::layers::{LayerId, LayerSummary};
use crate::vectors::{VectorLayerId, VectorLayerSummary};

use super::{ERROR_TEXT_COLOR, WARN_TEXT_COLOR, format, submitted_with_enter};

/// User intent emitted by the Layer List panel, applied by `VeldmapApp`.
#[derive(Clone, Debug)]
pub(crate) enum LayerUiAction {
    OpenFileDialog,
    OpenPath(PathBuf),
    ToggleVisibility(LayerId, bool),
    SetOpacity(LayerId, f32, bool),
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

fn paint_loading_shimmer(ui: &mut egui::Ui) {
    let width = ui.available_width().max(40.0);
    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, 3.0), egui::Sense::hover());
    let t = ui.input(|i| i.time) as f32;
    let phase = (t * 0.9) % 1.0;
    let bar_w = (rect.width() * 0.35).max(16.0);
    let x0 = rect.left() - bar_w + phase * (rect.width() + bar_w);
    let painter = ui.painter();
    painter.rect_filled(rect, 1.5, egui::Color32::from_gray(50));
    let highlight = egui::Rect::from_min_max(
        egui::pos2(x0.max(rect.left()), rect.top()),
        egui::pos2((x0 + bar_w).min(rect.right()), rect.bottom()),
    );
    if highlight.width() > 0.0 {
        painter.rect_filled(highlight, 1.5, egui::Color32::from_rgb(80, 175, 245));
    }
    ui.ctx()
        .request_repaint_after(std::time::Duration::from_millis(33));
}

fn show_opacity_slider(ui: &mut egui::Ui, current: f32) -> Option<(f32, bool)> {
    let mut opacity = current;
    let mut update = None;
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("Opacity").small());
        let resp = ui.add(egui::Slider::new(&mut opacity, 0.0..=1.0).show_value(false));
        let commit = resp.drag_stopped() || (resp.changed() && !resp.dragged());
        if resp.changed() || commit {
            update = Some((opacity, commit));
        }
        ui.monospace(egui::RichText::new(format!("{:.0}%", opacity * 100.0)).small());
    });
    update
}

fn show_layer_status_banners(ui: &mut egui::Ui, layer: &LayerSummary) {
    if layer.detail_limit_level > 0 {
        ui.colored_label(
            WARN_TEXT_COLOR,
            egui::RichText::new(format!(
                "Detail limited to level {} (memory budget)",
                layer.detail_limit_level
            ))
            .small(),
        );
    }

    if let Some(warn) = &layer.datum_warning {
        ui.colored_label(WARN_TEXT_COLOR, egui::RichText::new(warn).small());
    }

    if let Some(err) = &layer.last_error {
        ui.colored_label(
            ERROR_TEXT_COLOR,
            egui::RichText::new(format!("Partial/reload error: {err}")).small(),
        );
    }
}

fn show_layer_controls(ui: &mut egui::Ui, layer: &LayerSummary, actions: &mut Vec<LayerUiAction>) {
    if let Some((opacity, commit)) = show_opacity_slider(ui, layer.opacity) {
        actions.push(LayerUiAction::SetOpacity(layer.id, opacity, commit));
    }

    egui::CollapsingHeader::new(egui::RichText::new("CRS & Metadata").small())
        .id_salt(("meta", layer.id))
        .default_open(false)
        .show(ui, |ui| {
            let crs_tag = if layer.has_crs_override {
                format!("CRS: {} (override)", layer.crs_label)
            } else {
                format!("CRS: {}", layer.crs_label)
            };
            ui.small(crs_tag);
            ui.small(format!("Radiometry: {}", layer.sample_info));
            ui.small(format!("Path: {}", layer.path.display()));
        });
}

fn show_layer_actions(ui: &mut egui::Ui, layer: &LayerSummary, actions: &mut Vec<LayerUiAction>) {
    ui.horizontal_wrapped(|ui| {
        if let Some(bounds) = layer.footprint
            && ui
                .small_button("Zoom")
                .on_hover_text("Fly camera to layer extent")
                .clicked()
        {
            actions.push(LayerUiAction::ZoomToExtent(bounds));
        }
        if layer.can_override_crs
            && ui
                .small_button("CRS…")
                .on_hover_text("Override CRS or custom Transverse Mercator parameters")
                .clicked()
        {
            actions.push(LayerUiAction::OpenCrsPicker(layer.id, layer.name.clone()));
        }
        if ui
            .add_enabled(!layer.reloading, egui::Button::new("Reload").small())
            .on_hover_text("Reload from disk (bypass fingerprint & tile cache)")
            .clicked()
        {
            actions.push(LayerUiAction::ReloadBypassCache(layer.id));
        }
        if ui
            .small_button("Remove")
            .on_hover_text("Remove layer and free VRAM")
            .clicked()
        {
            actions.push(LayerUiAction::Remove(layer.id));
        }
    });
}

fn show_layer_card(ui: &mut egui::Ui, layer: &LayerSummary, actions: &mut Vec<LayerUiAction>) {
    ui.push_id(layer.id, |ui| {
        ui.group(|ui| {
            ui.horizontal(|ui| {
                let mut visible = layer.visible;
                if ui.checkbox(&mut visible, "").changed() {
                    actions.push(LayerUiAction::ToggleVisibility(layer.id, visible));
                }
                ui.add(egui::Label::new(egui::RichText::new(&layer.name).strong()).truncate())
                    .on_hover_text(layer.path.display().to_string());
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .add_enabled(layer.can_move_down, egui::Button::new("↓").small())
                        .on_hover_text("Move layer down")
                        .clicked()
                    {
                        actions.push(LayerUiAction::MoveDown(layer.id));
                    }
                    if ui
                        .add_enabled(layer.can_move_up, egui::Button::new("↑").small())
                        .on_hover_text("Move layer up")
                        .clicked()
                    {
                        actions.push(LayerUiAction::MoveUp(layer.id));
                    }
                });
            });

            let px_label = match layer.source_pixel_size_m {
                Some((mx, my)) => format!(
                    "{}×{} · {}",
                    layer.width,
                    layer.height,
                    format::format_source_pixel_size(mx, my)
                ),
                None => format!("{}×{}", layer.width, layer.height),
            };
            let is_loading =
                layer.reloading || (layer.visible && (layer.pass_active || layer.in_flight > 0));
            ui.horizontal(|ui| {
                ui.weak(egui::RichText::new(px_label).small());
                if layer.reloading {
                    ui.weak(egui::RichText::new("· reloading…").small());
                } else if is_loading {
                    ui.weak(egui::RichText::new("· loading…").small());
                }
            });

            if is_loading {
                paint_loading_shimmer(ui);
            }

            show_layer_status_banners(ui, layer);
            show_layer_controls(ui, layer, actions);
            show_layer_actions(ui, layer, actions);
        });
    });
    ui.add_space(4.0);
}

fn show_vector_card(
    ui: &mut egui::Ui,
    layer: &VectorLayerSummary,
    actions: &mut Vec<LayerUiAction>,
) {
    ui.push_id(("vec", layer.id), |ui| {
        ui.group(|ui| {
            ui.horizontal(|ui| {
                let mut visible = layer.visible;
                if ui.checkbox(&mut visible, "").changed() {
                    actions.push(LayerUiAction::ToggleVectorVisibility(layer.id, visible));
                }
                let swatch = egui::Color32::from_rgb(
                    layer.color_rgb[0],
                    layer.color_rgb[1],
                    layer.color_rgb[2],
                );
                let (rect, _) =
                    ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
                ui.painter().rect_filled(rect, 2.0, swatch);
                ui.add(egui::Label::new(egui::RichText::new(&layer.name).strong()).truncate())
                    .on_hover_text(layer.path.display().to_string());
            });

            let summary_text = if layer.point_count > 0 {
                format!(
                    "GeoJSON · {} features ({} tris, {} segs, {} pts)",
                    layer.feature_count,
                    layer.triangle_count,
                    layer.line_segment_count,
                    layer.point_count
                )
            } else {
                format!(
                    "GeoJSON · {} features ({} tris, {} segs)",
                    layer.feature_count, layer.triangle_count, layer.line_segment_count
                )
            };
            ui.weak(egui::RichText::new(summary_text).small());

            if let Some((opacity, commit)) = show_opacity_slider(ui, layer.opacity) {
                actions.push(LayerUiAction::SetVectorOpacity(layer.id, opacity, commit));
            }

            ui.horizontal(|ui| {
                if let Some(bounds) = layer.footprint
                    && ui
                        .small_button("Zoom")
                        .on_hover_text("Fly camera to vector layer extent")
                        .clicked()
                {
                    actions.push(LayerUiAction::ZoomToExtent(bounds));
                }
                if ui.small_button("Remove").clicked() {
                    actions.push(LayerUiAction::RemoveVector(layer.id));
                }
            });
        });
    });
    ui.add_space(4.0);
}

/// Render the collapsible left Layer List panel and return user actions.
pub(crate) fn show_layers_panel(
    ui: &mut egui::Ui,
    layers: &[LayerSummary],
    vectors: &[VectorLayerSummary],
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
                        show_layer_card(ui, layer, &mut actions);
                    }
                }
            });
        });

    actions
}
