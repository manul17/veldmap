//! Band Math / RGB Composite layer-card controls (colormap, reflectance stretch,
//! `SCL` cloud/shadow mask class toggles) and the Layers-panel Band Math
//! builder for local multi-band GeoTIFFs and loaded layers.

use eframe::egui;
use veldmap_core::tile::{BandKey, PixelFormat};

use crate::colormap::{ColormapId, DEFAULT_SCL_MASK_BITS, SENTINEL2_SCL_NAMES};
use crate::layers::{
    BandMathFormula, BandMathSummary, LayerId, LayerSummary, LocalBandMathInput, LocalBandMathSpec,
    OverlayEdit,
};
use crate::ui::format;
use crate::ui::widgets::{DragCommit, track_drag_commit};

use super::LayerUiAction;

/// Quick-toggle ESA Sentinel-2 `SCL` classes (0, 1, 3, 8, 9, 10).
const QUICK_SCL_CLASSES: [u8; 6] = [0, 1, 3, 8, 9, 10];

/// Render live Band Math styling, stretch, and `SCL` mask controls inside a
/// raster layer card.
pub(super) fn show_band_math_controls(
    ui: &mut egui::Ui,
    layer_id: LayerId,
    bm: &BandMathSummary,
    actions: &mut Vec<LayerUiAction>,
) {
    // Constituent band summary.
    for input in &bm.inputs {
        let gsd = input
            .source_pixel_size_m
            .map(|(mx, my)| format!(" · {}", format::format_source_pixel_size(mx, my)))
            .unwrap_or_default();
        ui.weak(egui::RichText::new(format!("{}: {}{gsd}", input.slot_name, input.label)).small());
    }
    if let Some(scl_label) = &bm.scl_label {
        ui.weak(egui::RichText::new(format!("Mask: {scl_label}")).small());
    }

    if bm.formula.is_scalar() {
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Ramp").small());
            let mut cmap = bm.colormap;
            egui::ComboBox::from_id_salt(("bm_colormap", layer_id))
                .selected_text(egui::RichText::new(cmap.label()).small())
                .width(120.0)
                .show_ui(ui, |ui| {
                    for opt in ColormapId::ALL {
                        if ui.selectable_value(&mut cmap, opt, opt.label()).clicked() {
                            actions.push(LayerUiAction::edit_raster(
                                layer_id,
                                OverlayEdit::SetColormap {
                                    colormap: opt,
                                    invert: bm.invert_colormap,
                                },
                            ));
                        }
                    }
                });
            let mut invert = bm.invert_colormap;
            if ui
                .checkbox(&mut invert, egui::RichText::new("Inv").small())
                .changed()
            {
                actions.push(LayerUiAction::edit_raster(
                    layer_id,
                    OverlayEdit::SetColormap {
                        colormap: bm.colormap,
                        invert,
                    },
                ));
            }
        });

        let mut lo = bm.stretch_min;
        let mut hi = bm.stretch_max;
        let speed = ((hi - lo).abs() * 0.005).max(1e-3);
        let drag_id = ui.scope_id().with(("bm_stretch_drag", layer_id));
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Min").small());
            let r_lo = ui.add(egui::DragValue::new(&mut lo).speed(speed).max_decimals(3));
            ui.label(egui::RichText::new("Max").small());
            let r_hi = ui.add(egui::DragValue::new(&mut hi).speed(speed).max_decimals(3));
            if ui
                .small_button("Reset")
                .on_hover_text("Reset stretch to recipe default")
                .clicked()
            {
                actions.push(LayerUiAction::edit_raster(
                    layer_id,
                    OverlayEdit::ResetBandMathStretch,
                ));
            }
            match track_drag_commit(
                ui,
                drag_id,
                r_lo.changed() || r_hi.changed(),
                r_lo.dragged() || r_hi.dragged(),
                r_lo.drag_stopped() || r_hi.drag_stopped(),
            ) {
                DragCommit::None => {}
                DragCommit::Preview => {
                    actions.push(LayerUiAction::edit_raster_drag(
                        layer_id,
                        OverlayEdit::SetStretchRange { min: lo, max: hi },
                        false,
                    ));
                }
                DragCommit::Commit => {
                    actions.push(LayerUiAction::edit_raster_drag(
                        layer_id,
                        OverlayEdit::SetStretchRange { min: lo, max: hi },
                        true,
                    ));
                }
            }
        });
    } else {
        // RGB Composite stretch (linked min/max + gamma).
        let mut lo = bm.rgb_min[0];
        let mut hi = bm.rgb_max[0];
        let mut gamma = bm.rgb_gamma;
        let speed = ((hi - lo).abs() * 0.005).max(1e-3);
        let drag_id = ui.scope_id().with(("bm_rgb_drag", layer_id));
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Min").small());
            let r_lo = ui.add(egui::DragValue::new(&mut lo).speed(speed).max_decimals(3));
            ui.label(egui::RichText::new("Max").small());
            let r_hi = ui.add(egui::DragValue::new(&mut hi).speed(speed).max_decimals(3));
            if ui
                .small_button("Reset")
                .on_hover_text("Reset RGB stretch and gamma to default")
                .clicked()
            {
                actions.push(LayerUiAction::edit_raster(
                    layer_id,
                    OverlayEdit::ResetBandMathStretch,
                ));
            }
            match track_drag_commit(
                ui,
                drag_id,
                r_lo.changed() || r_hi.changed(),
                r_lo.dragged() || r_hi.dragged(),
                r_lo.drag_stopped() || r_hi.drag_stopped(),
            ) {
                DragCommit::None => {}
                DragCommit::Preview => {
                    actions.push(LayerUiAction::edit_raster_drag(
                        layer_id,
                        OverlayEdit::SetBandMathRgbStretch {
                            min: [lo; 3],
                            max: [hi; 3],
                            gamma,
                        },
                        false,
                    ));
                }
                DragCommit::Commit => {
                    actions.push(LayerUiAction::edit_raster_drag(
                        layer_id,
                        OverlayEdit::SetBandMathRgbStretch {
                            min: [lo; 3],
                            max: [hi; 3],
                            gamma,
                        },
                        true,
                    ));
                }
            }
        });
        let gamma_id = ui.scope_id().with(("bm_gamma_drag", layer_id));
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Gamma").small());
            let r_g = ui.add(
                egui::Slider::new(&mut gamma, 0.4..=2.5)
                    .fixed_decimals(2)
                    .show_value(true),
            );
            match track_drag_commit(
                ui,
                gamma_id,
                r_g.changed(),
                r_g.dragged(),
                r_g.drag_stopped(),
            ) {
                DragCommit::None => {}
                DragCommit::Preview => {
                    actions.push(LayerUiAction::edit_raster_drag(
                        layer_id,
                        OverlayEdit::SetBandMathRgbStretch {
                            min: bm.rgb_min,
                            max: bm.rgb_max,
                            gamma,
                        },
                        false,
                    ));
                }
                DragCommit::Commit => {
                    actions.push(LayerUiAction::edit_raster_drag(
                        layer_id,
                        OverlayEdit::SetBandMathRgbStretch {
                            min: bm.rgb_min,
                            max: bm.rgb_max,
                            gamma,
                        },
                        true,
                    ));
                }
            }
        });
    }

    if bm.has_scl {
        ui.separator();
        ui.horizontal(|ui| {
            let mut enabled = bm.scl_mask_enabled;
            if ui
                .checkbox(
                    &mut enabled,
                    egui::RichText::new("SCL Cloud/Shadow Mask")
                        .small()
                        .strong(),
                )
                .changed()
            {
                actions.push(LayerUiAction::edit_raster(
                    layer_id,
                    OverlayEdit::SetBandMathSclEnabled(enabled),
                ));
            }
            if enabled
                && bm.scl_mask_bits != DEFAULT_SCL_MASK_BITS
                && ui.small_button("Default").clicked()
            {
                actions.push(LayerUiAction::edit_raster(
                    layer_id,
                    OverlayEdit::SetBandMathSclMaskBits(DEFAULT_SCL_MASK_BITS),
                ));
            }
        });
        if bm.scl_mask_enabled {
            let mut bits = bm.scl_mask_bits;
            let mut changed = false;
            ui.horizontal_wrapped(|ui| {
                for cls in QUICK_SCL_CLASSES {
                    let bit = 1u32 << cls;
                    let mut checked = (bits & bit) != 0;
                    let short = match cls {
                        0 => "0 NoData",
                        1 => "1 Defective",
                        3 => "3 Shadow",
                        8 => "8 Cloud Med",
                        9 => "9 Cloud High",
                        10 => "10 Cirrus",
                        _ => "",
                    };
                    if ui
                        .checkbox(&mut checked, egui::RichText::new(short).small())
                        .on_hover_text(format!("SCL {cls}: {}", SENTINEL2_SCL_NAMES[cls as usize]))
                        .changed()
                    {
                        if checked {
                            bits |= bit;
                        } else {
                            bits &= !bit;
                        }
                        changed = true;
                    }
                }
            });
            egui::CollapsingHeader::new(egui::RichText::new("All 12 SCL Classes").small())
                .id_salt(("bm_scl_all", layer_id))
                .default_open(false)
                .show(ui, |ui| {
                    for (cls, name) in SENTINEL2_SCL_NAMES.iter().enumerate() {
                        let bit = 1u32 << cls;
                        let mut checked = (bits & bit) != 0;
                        if ui
                            .checkbox(
                                &mut checked,
                                egui::RichText::new(format!("{cls}: {name}")).small(),
                            )
                            .changed()
                        {
                            if checked {
                                bits |= bit;
                            } else {
                                bits &= !bit;
                            }
                            changed = true;
                        }
                    }
                });
            if changed {
                actions.push(LayerUiAction::edit_raster(
                    layer_id,
                    OverlayEdit::SetBandMathSclMaskBits(bits),
                ));
            }
        }
    }
}

/// Selectable `(LayerId, band_index)` candidate collected from loaded
/// `SingleRaster` layers for the Band Math builder.
#[derive(Clone, Debug, PartialEq, Eq)]
struct BandChoice {
    layer_id: LayerId,
    band_index: u16,
    label: String,
    short_label: String,
    supports_u8_mask: bool,
}

fn collect_band_choices(layers: &[LayerSummary]) -> Vec<BandChoice> {
    let mut choices = Vec::new();
    for layer in layers {
        if layer.band_math.is_some() {
            continue;
        }
        let mut seen_bands = Vec::new();
        for opt in &layer.band_options {
            let BandKey::Index(idx) = opt.spec.band else {
                continue;
            };
            if seen_bands.contains(&idx) {
                continue;
            }
            seen_bands.push(idx);
            let supports_u8 = layer.band_options.iter().any(|o| {
                o.spec.band == BandKey::Index(idx) && o.spec.format == PixelFormat::R8Uint
            });
            let band_tag = opt
                .label
                .split(" · ")
                .next()
                .unwrap_or(&opt.label)
                .to_owned();
            choices.push(BandChoice {
                layer_id: layer.id,
                band_index: idx,
                label: format!("{} — {band_tag}", layer.name),
                short_label: band_tag,
                supports_u8_mask: supports_u8,
            });
        }
    }
    choices
}

/// Transient UI state for the Layers-panel Band Math / RGB Composite builder.
#[derive(Clone, Debug)]
pub(crate) struct BandMathBuilderState {
    pub(crate) open: bool,
    formula: BandMathFormula,
    title: String,
    slots: [usize; 3],
    scl_slot: Option<usize>,
    initialized_for_count: usize,
}

impl Default for BandMathBuilderState {
    fn default() -> Self {
        Self {
            open: false,
            formula: BandMathFormula::NormalizedDiff,
            title: String::new(),
            slots: [0, 0, 0],
            scl_slot: None,
            initialized_for_count: 0,
        }
    }
}

impl BandMathBuilderState {
    fn auto_populate(&mut self, choices: &[BandChoice]) {
        if choices.is_empty() {
            return;
        }
        // Smart default: if a 4-band RGB+NIR raster is loaded (Red=0, Green=1,
        // Blue=2, NIR=3), pre-populate NDVI as (Band 4 NIR, Band 1 Red).
        if choices.len() >= 4
            && choices[0].layer_id == choices[3].layer_id
            && choices[3].band_index == 3
        {
            self.slots = [3, 0, 1];
        } else if choices.len() >= 3 {
            self.slots = [0, 1, 2];
        } else if choices.len() == 2 {
            self.slots = [0, 1, 0];
        } else {
            self.slots = [0, 0, 0];
        }
        self.scl_slot = choices
            .iter()
            .position(|c| c.supports_u8_mask && c.label.to_ascii_lowercase().contains("scl"));
        self.initialized_for_count = choices.len();
    }
}

/// Render the collapsible Band Math / RGB Composite builder inside the Layers
/// panel so analysts can combine bands from local multi-band GeoTIFFs (e.g.
/// 4-band RGB+NIR orthophotos) or distinct loaded raster layers.
pub(super) fn show_band_math_builder(
    ui: &mut egui::Ui,
    layers: &[LayerSummary],
    state: &mut BandMathBuilderState,
    actions: &mut Vec<LayerUiAction>,
) {
    let choices = collect_band_choices(layers);
    if choices.is_empty() {
        ui.weak(egui::RichText::new("Load at least one raster layer to build Band Math.").small());
        return;
    }
    if state.initialized_for_count != choices.len() {
        state.auto_populate(&choices);
    }
    for slot in &mut state.slots {
        if *slot >= choices.len() {
            *slot = 0;
        }
    }
    if let Some(s) = state.scl_slot
        && (s >= choices.len() || !choices[s].supports_u8_mask)
    {
        state.scl_slot = None;
    }

    egui::Frame::group(ui.style()).show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.strong(egui::RichText::new("Band Math / RGB Builder").small());
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.small_button("✕").clicked() {
                    state.open = false;
                }
            });
        });

        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Formula").small());
            egui::ComboBox::from_id_salt("bm_builder_formula")
                .selected_text(egui::RichText::new(state.formula.label()).small())
                .width(185.0)
                .show_ui(ui, |ui| {
                    for opt in BandMathFormula::ALL {
                        ui.selectable_value(&mut state.formula, opt, opt.label());
                    }
                });
        });

        let slot_names: &[&str] = match state.formula {
            BandMathFormula::RgbComposite => &["Red", "Green", "Blue"],
            _ => &["Band A", "Band B"],
        };
        for (i, slot_name) in slot_names.iter().enumerate() {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(*slot_name).small());
                let cur_label = choices
                    .get(state.slots[i])
                    .map_or("—", |c| c.label.as_str());
                egui::ComboBox::from_id_salt(("bm_builder_slot", i))
                    .selected_text(egui::RichText::new(cur_label).small())
                    .width(185.0)
                    .show_ui(ui, |ui| {
                        for (idx, choice) in choices.iter().enumerate() {
                            ui.selectable_value(&mut state.slots[i], idx, &choice.label);
                        }
                    });
            });
        }

        let u8_choices: Vec<(usize, &BandChoice)> = choices
            .iter()
            .enumerate()
            .filter(|(_, c)| c.supports_u8_mask)
            .collect();
        if !u8_choices.is_empty() {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("SCL Mask").small());
                let cur = state
                    .scl_slot
                    .and_then(|idx| choices.get(idx))
                    .map_or("None", |c| c.label.as_str());
                egui::ComboBox::from_id_salt("bm_builder_scl")
                    .selected_text(egui::RichText::new(cur).small())
                    .width(175.0)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut state.scl_slot, None, "None");
                        for (idx, choice) in &u8_choices {
                            ui.selectable_value(&mut state.scl_slot, Some(*idx), &choice.label);
                        }
                    });
            });
        }

        ui.horizontal(|ui| {
            ui.add(
                egui::TextEdit::singleline(&mut state.title)
                    .desired_width(145.0)
                    .hint_text("Layer title (optional)"),
            );
            if ui.small_button("Create Layer").clicked() {
                let req = state.formula.required_bands();
                let inputs: Vec<LocalBandMathInput> = (0..req)
                    .map(|i| {
                        let c = &choices[state.slots[i]];
                        LocalBandMathInput {
                            layer_id: c.layer_id,
                            band_index: c.band_index,
                            label: c.short_label.clone(),
                        }
                    })
                    .collect();
                let scl = state.scl_slot.map(|idx| {
                    let c = &choices[idx];
                    LocalBandMathInput {
                        layer_id: c.layer_id,
                        band_index: c.band_index,
                        label: c.short_label.clone(),
                    }
                });
                let title = if state.title.trim().is_empty() {
                    format!(
                        "{} ({})",
                        state.formula.short_label(),
                        inputs
                            .iter()
                            .map(|b| b.label.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                } else {
                    state.title.trim().to_owned()
                };
                actions.push(LayerUiAction::CreateBandMath(LocalBandMathSpec {
                    title,
                    formula: state.formula,
                    inputs,
                    scl,
                }));
                state.title.clear();
                state.open = false;
            }
        });
    });
    ui.separator();
}
