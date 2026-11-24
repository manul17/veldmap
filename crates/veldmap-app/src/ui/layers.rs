//! Collapsible Layer List side panel and minimal File Open trigger for local
//! GeoTIFF/BigTIFF/COG overlays.

use std::path::PathBuf;

use eframe::egui;
use veldmap_core::geo::footprint::FootprintBounds;

use crate::layers::{LayerId, LayerSummary};

use super::{ERROR_TEXT_COLOR, hud};

/// User intent emitted by the Layer List panel and File Open trigger, applied
/// by `VeldmapApp`.
#[derive(Clone, Debug)]
pub(crate) enum LayerUiAction {
    OpenPath(PathBuf),
    ToggleVisibility(LayerId, bool),
    SetOpacity(LayerId, f32),
    ZoomToExtent(FootprintBounds),
    ReloadBypassCache(LayerId),
    Remove(LayerId),
}

/// Transient UI state for the Layer List side panel and File Open input.
#[derive(Clone, Debug, Default)]
pub(crate) struct LayersPanelState {
    pub(crate) open: bool,
    pub(crate) open_prompt_visible: bool,
    focus_path_input: bool,
    pub(crate) path_input: String,
    pub(crate) error: Option<String>,
}

impl LayersPanelState {
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

    pub(crate) fn clear_error(&mut self) {
        self.error = None;
    }
}

/// Render the collapsible left Layer List panel and return user actions.
pub(crate) fn show_layers_panel(
    ui: &mut egui::Ui,
    layers: &[LayerSummary],
    state: &mut LayersPanelState,
) -> Vec<LayerUiAction> {
    let mut actions = Vec::new();
    if !state.open {
        return actions;
    }

    egui::Panel::left("layer_list_panel")
        .resizable(true)
        .default_size(270.0)
        .min_size(210.0)
        .max_size(420.0)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.strong(format!("Layers ({})", layers.len()));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.small_button("×").on_hover_text("Hide Layers").clicked() {
                        state.open = false;
                    }
                    if ui
                        .small_button("Open…")
                        .on_hover_text("Open local GeoTIFF / BigTIFF / COG")
                        .clicked()
                    {
                        state.open_prompt_visible = !state.open_prompt_visible;
                        if state.open_prompt_visible {
                            state.focus_path_input = true;
                        }
                    }
                });
            });
            ui.separator();

            if state.open_prompt_visible {
                ui.label(egui::RichText::new("Local GeoTIFF / COG path (or drag & drop):").small());
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
                        state.error = None;
                    }
                    let enter = resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                    if ui.button("Load").clicked() || enter {
                        let trimmed = state.path_input.trim().trim_matches(['"', '\'']);
                        if !trimmed.is_empty() {
                            actions.push(LayerUiAction::OpenPath(PathBuf::from(trimmed)));
                        }
                    }
                });
                ui.separator();
            }

            if let Some(err) = &state.error {
                ui.horizontal_wrapped(|ui| {
                    ui.colored_label(ERROR_TEXT_COLOR, err);
                });
                ui.separator();
            }

            if layers.is_empty() {
                ui.weak("No raster overlays loaded.");
                ui.weak(
                    "Drop a GeoTIFF/COG onto the globe, pass a path on the CLI, or click Open….",
                );
                return;
            }

            egui::ScrollArea::vertical().show(ui, |ui| {
                // Display top-to-bottom (reverse of painter's order), keyed
                // strictly by `LayerId` so removals never shift widget state.
                for layer in layers.iter().rev() {
                    ui.push_id(layer.id, |ui| {
                        ui.group(|ui| {
                            ui.horizontal(|ui| {
                                let mut visible = layer.visible;
                                if ui.checkbox(&mut visible, "").changed() {
                                    actions
                                        .push(LayerUiAction::ToggleVisibility(layer.id, visible));
                                }
                                ui.add(
                                    egui::Label::new(egui::RichText::new(&layer.name).strong())
                                        .truncate(),
                                )
                                .on_hover_text(layer.path.display().to_string());
                            });

                            let px_label = match layer.source_pixel_size_m {
                                Some((mx, my)) => format!(
                                    "{}×{} · {}",
                                    layer.width,
                                    layer.height,
                                    hud::format_source_pixel_size(mx, my)
                                ),
                                None => format!("{}×{}", layer.width, layer.height),
                            };
                            ui.horizontal(|ui| {
                                ui.weak(egui::RichText::new(px_label).small());
                                if layer.reloading {
                                    ui.weak(egui::RichText::new("· reloading…").small());
                                } else if layer.visible
                                    && (layer.pass_active || layer.in_flight > 0)
                                {
                                    ui.weak(egui::RichText::new("· loading…").small());
                                }
                            });

                            let mut opacity = layer.opacity;
                            ui.horizontal(|ui| {
                                ui.label(egui::RichText::new("Opacity").small());
                                if ui
                                    .add(
                                        egui::Slider::new(&mut opacity, 0.0..=1.0)
                                            .show_value(false),
                                    )
                                    .changed()
                                {
                                    actions.push(LayerUiAction::SetOpacity(layer.id, opacity));
                                }
                                ui.monospace(
                                    egui::RichText::new(format!("{:.0}%", opacity * 100.0)).small(),
                                );
                            });

                            ui.horizontal(|ui| {
                                if let Some(bounds) = layer.footprint
                                    && ui
                                        .small_button("Zoom")
                                        .on_hover_text("Fly camera to layer extent")
                                        .clicked()
                                {
                                    actions.push(LayerUiAction::ZoomToExtent(bounds));
                                }
                                if ui
                                    .add_enabled(
                                        !layer.reloading,
                                        egui::Button::new("Reload").small(),
                                    )
                                    .on_hover_text(
                                        "Reload from disk (bypass fingerprint & tile cache)",
                                    )
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
                        });
                    });
                    ui.add_space(4.0);
                }
            });
        });

    actions
}
