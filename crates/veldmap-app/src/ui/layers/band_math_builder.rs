//! Collapsible Layers-panel Band Math / RGB Composite builder for combining
//! bands from local multi-band GeoTIFFs (e.g. 4-band RGB+NIR orthophotos) or
//! distinct loaded single-raster layers.

use eframe::egui;
use veldmap_core::tile::{BandKey, PixelFormat};

use crate::layers::{
    BandMathFormula, LayerId, LayerSummary, LayerSummaryKind, LocalBandMathInput, LocalBandMathSpec,
};
use crate::ui::WARN_TEXT_COLOR;

use super::LayerUiAction;

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

impl BandChoice {
    fn key(&self) -> (LayerId, u16) {
        (self.layer_id, self.band_index)
    }
}

pub(super) fn has_band_math_inputs(layers: &[LayerSummary]) -> bool {
    layers.iter().any(|layer| {
        matches!(
            &layer.kind,
            LayerSummaryKind::SingleRaster(single)
                if single
                    .band_options
                    .iter()
                    .any(|opt| matches!(opt.spec.band, BandKey::Index(_)))
        )
    })
}

fn collect_band_choices(layers: &[LayerSummary]) -> Vec<BandChoice> {
    let mut choices = Vec::new();
    for layer in layers {
        let LayerSummaryKind::SingleRaster(single) = &layer.kind else {
            continue;
        };
        let mut seen_bands = Vec::new();
        for opt in &single.band_options {
            let BandKey::Index(idx) = opt.spec.band else {
                continue;
            };
            if seen_bands.contains(&idx) {
                continue;
            }
            seen_bands.push(idx);
            let supports_u8 = single.band_options.iter().any(|o| {
                o.spec.band == BandKey::Index(idx) && o.spec.format == PixelFormat::R8Uint
            });
            choices.push(BandChoice {
                layer_id: layer.id,
                band_index: idx,
                label: format!("{} — {}", layer.name, opt.short_label),
                short_label: opt.short_label.clone(),
                supports_u8_mask: supports_u8,
            });
        }
    }
    choices
}

/// Transient UI state for the Layers-panel Band Math / RGB Composite builder.
#[derive(Clone, Debug)]
pub(super) struct BandMathBuilderState {
    pub(super) open: bool,
    formula: BandMathFormula,
    title: String,
    slot_keys: [Option<(LayerId, u16)>; 3],
    scl_key: Option<(LayerId, u16)>,
    user_customized_slots: bool,
    user_customized_scl: bool,
}

impl Default for BandMathBuilderState {
    fn default() -> Self {
        Self {
            open: false,
            formula: BandMathFormula::NormalizedDiff,
            title: String::new(),
            slot_keys: [None; 3],
            scl_key: None,
            user_customized_slots: false,
            user_customized_scl: false,
        }
    }
}

impl BandMathBuilderState {
    pub(super) fn on_created(&mut self) {
        self.title.clear();
        self.open = false;
    }

    fn apply_smart_defaults(&mut self, choices: &[BandChoice]) {
        if choices.is_empty() {
            self.slot_keys = [None; 3];
            self.scl_key = None;
            return;
        }
        let key_at = |idx: usize| {
            choices
                .get(idx)
                .or_else(|| choices.first())
                .map(BandChoice::key)
        };
        // Smart default: if a 4-band RGB+NIR raster is loaded (Red=0, Green=1,
        // Blue=2, NIR=3), pre-populate scalar indices as (NIR, Red) while
        // keeping RGB Composite ordered as (Red, Green, Blue).
        let is_4band_same_layer = choices.len() >= 4
            && choices[0].layer_id == choices[3].layer_id
            && choices[3].band_index == 3;
        if is_4band_same_layer && self.formula.is_scalar() {
            self.slot_keys = [key_at(3), key_at(0), key_at(1)];
        } else if choices.len() >= 3 {
            self.slot_keys = [key_at(0), key_at(1), key_at(2)];
        } else if choices.len() == 2 {
            self.slot_keys = [key_at(0), key_at(1), key_at(0)];
        } else {
            self.slot_keys = [key_at(0), key_at(0), key_at(0)];
        }
        if !self.user_customized_scl {
            self.scl_key = choices
                .iter()
                .find(|c| c.supports_u8_mask && c.label.to_ascii_lowercase().contains("scl"))
                .map(BandChoice::key);
        }
    }

    fn reconcile_choices(&mut self, choices: &[BandChoice]) {
        let missing_slot = self
            .slot_keys
            .iter()
            .any(|k| k.is_none_or(|key| !choices.iter().any(|c| c.key() == key)));
        if missing_slot && !self.user_customized_slots {
            self.apply_smart_defaults(choices);
        } else {
            for (i, slot) in self.slot_keys.iter_mut().enumerate() {
                let valid = slot.is_some_and(|key| choices.iter().any(|c| c.key() == key));
                if !valid {
                    *slot = choices
                        .get(i)
                        .or_else(|| choices.first())
                        .map(BandChoice::key);
                }
            }
        }
        if let Some(key) = self.scl_key
            && !choices.iter().any(|c| c.key() == key && c.supports_u8_mask)
        {
            self.scl_key = None;
        }
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
        state.open = false;
        return;
    }
    state.reconcile_choices(&choices);

    egui::Frame::group(ui.style()).show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.strong(egui::RichText::new("Band Math / RGB Builder").small());
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.small_button("×").clicked() {
                    state.open = false;
                }
            });
        });

        let combo_w = (ui.available_width() - 62.0).max(110.0);

        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Formula").small());
            let prev_formula = state.formula;
            egui::ComboBox::from_id_salt("bm_builder_formula")
                .selected_text(egui::RichText::new(state.formula.label()).small())
                .width(combo_w)
                .show_ui(ui, |ui| {
                    for opt in BandMathFormula::ALL {
                        ui.selectable_value(&mut state.formula, opt, opt.label());
                    }
                });
            if state.formula != prev_formula && !state.user_customized_slots {
                state.apply_smart_defaults(&choices);
            }
        });

        let slot_names = state.formula.slot_names();
        for (i, slot_name) in slot_names.iter().enumerate() {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(*slot_name).small());
                let cur_label = state.slot_keys[i]
                    .and_then(|key| choices.iter().find(|c| c.key() == key))
                    .map_or("—", |c| c.label.as_str());
                let prev = state.slot_keys[i];
                egui::ComboBox::from_id_salt(("bm_builder_slot", i))
                    .selected_text(egui::RichText::new(cur_label).small())
                    .width(combo_w)
                    .show_ui(ui, |ui| {
                        for choice in &choices {
                            ui.selectable_value(
                                &mut state.slot_keys[i],
                                Some(choice.key()),
                                &choice.label,
                            );
                        }
                    });
                if state.slot_keys[i] != prev {
                    state.user_customized_slots = true;
                }
            });
        }

        let u8_choices: Vec<&BandChoice> = choices.iter().filter(|c| c.supports_u8_mask).collect();
        if !u8_choices.is_empty() {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("SCL Mask").small());
                let cur = state
                    .scl_key
                    .and_then(|key| choices.iter().find(|c| c.key() == key))
                    .map_or("None", |c| c.label.as_str());
                let prev = state.scl_key;
                egui::ComboBox::from_id_salt("bm_builder_scl")
                    .selected_text(egui::RichText::new(cur).small())
                    .width(combo_w)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut state.scl_key, None, "None");
                        for choice in &u8_choices {
                            ui.selectable_value(
                                &mut state.scl_key,
                                Some(choice.key()),
                                &choice.label,
                            );
                        }
                    });
                if state.scl_key != prev {
                    state.user_customized_scl = true;
                }
            });
        }

        let duplicate_scalar = state.formula.is_scalar()
            && state.slot_keys[0].is_some()
            && state.slot_keys[0] == state.slot_keys[1];
        if duplicate_scalar {
            ui.colored_label(
                WARN_TEXT_COLOR,
                egui::RichText::new("Select two different bands for Band A and Band B.").small(),
            );
        }

        ui.horizontal(|ui| {
            let title_w = (ui.available_width() - 92.0).max(96.0);
            ui.add(
                egui::TextEdit::singleline(&mut state.title)
                    .desired_width(title_w)
                    .hint_text("Layer title (optional)"),
            );
            if ui
                .add_enabled(!duplicate_scalar, egui::Button::new("Create Layer").small())
                .clicked()
            {
                let req = state.formula.required_bands();
                let inputs: Vec<LocalBandMathInput> = (0..req)
                    .filter_map(|i| {
                        let key = state.slot_keys[i]?;
                        let c = choices.iter().find(|c| c.key() == key)?;
                        Some(LocalBandMathInput {
                            layer_id: c.layer_id,
                            band_index: c.band_index,
                            label: c.short_label.clone(),
                        })
                    })
                    .collect();
                if inputs.len() == req {
                    let scl = state.scl_key.and_then(|key| {
                        let c = choices.iter().find(|c| c.key() == key)?;
                        Some(LocalBandMathInput {
                            layer_id: c.layer_id,
                            band_index: c.band_index,
                            label: c.short_label.clone(),
                        })
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
                }
            }
        });
    });
    ui.separator();
}
