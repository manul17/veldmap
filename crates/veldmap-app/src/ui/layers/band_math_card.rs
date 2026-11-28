//! Band Math / RGB Composite layer-card controls (colormap, reflectance stretch,
//! gamma, and `SCL` cloud/shadow mask class toggles).

use eframe::egui;

use crate::colormap::{DEFAULT_SCL_MASK_BITS, SENTINEL2_SCL_NAMES};
use crate::format;
use crate::layers::{BandMathEdit, BandMathSummary, LayerId, OverlayEdit};
use crate::ui::widgets::{DragCommit, track_drag_commit};

use super::LayerUiAction;
use super::style_controls::{MinMaxEvent, show_colormap_picker, show_min_max_drag};

/// Quick-toggle ESA Sentinel-2 `SCL` classes `(class_id, compact_label)`.
const QUICK_SCL_CLASSES: [(u8, &str); 6] = [
    (0, "0 NoData"),
    (1, "1 Defective"),
    (3, "3 Shadow"),
    (8, "8 Cloud Med"),
    (9, "9 Cloud High"),
    (10, "10 Cirrus"),
];
/// Remaining non-quick ESA Sentinel-2 `SCL` classes shown under "Other SCL Classes".
const OTHER_SCL_CLASSES: [u8; 6] = [2, 4, 5, 6, 7, 11];

fn toggle_scl_checkbox(
    ui: &mut egui::Ui,
    cls: u8,
    label: &str,
    hover: Option<String>,
    bits: &mut u32,
) -> bool {
    let bit = 1u32 << cls;
    let mut checked = (*bits & bit) != 0;
    let resp = ui.checkbox(&mut checked, egui::RichText::new(label).small());
    let resp = if let Some(tip) = hover {
        resp.on_hover_text(tip)
    } else {
        resp
    };
    if resp.changed() {
        if checked {
            *bits |= bit;
        } else {
            *bits &= !bit;
        }
        true
    } else {
        false
    }
}

/// Render live Band Math styling, stretch, and `SCL` mask controls inside a
/// raster layer card.
pub(super) fn show_band_math_controls(
    ui: &mut egui::Ui,
    layer_id: LayerId,
    bm: &BandMathSummary,
    actions: &mut Vec<LayerUiAction>,
) {
    show_band_math_controls_with_options(ui, layer_id, bm, true, actions);
}

/// Render Band Math input summaries, optional single-epoch colormap/stretch
/// controls (`show_style_controls = false` suppresses redundant single-epoch
/// ramps when a Timeline layer is in `ScalarDiff` mode), and `SCL` mask toggles.
pub(super) fn show_band_math_controls_with_options(
    ui: &mut egui::Ui,
    layer_id: LayerId,
    bm: &BandMathSummary,
    show_style_controls: bool,
    actions: &mut Vec<LayerUiAction>,
) {
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

    if show_style_controls && bm.formula.is_scalar() {
        show_colormap_picker(
            ui,
            ("bm_colormap", layer_id),
            bm.colormap,
            bm.invert_colormap,
            |colormap, invert| {
                actions.push(LayerUiAction::edit_raster(
                    layer_id,
                    OverlayEdit::BandMath(BandMathEdit::SetColormap { colormap, invert }),
                ));
            },
        );
        match show_min_max_drag(
            ui,
            ("bm_stretch_drag", layer_id),
            bm.stretch_min,
            bm.stretch_max,
            Some("Reset stretch to recipe default"),
        ) {
            MinMaxEvent::None => {}
            MinMaxEvent::Drag { min, max, commit } => {
                actions.push(LayerUiAction::edit_raster_drag(
                    layer_id,
                    OverlayEdit::BandMath(BandMathEdit::SetStretchRange { min, max }),
                    commit,
                ));
            }
            MinMaxEvent::Reset => {
                actions.push(LayerUiAction::edit_raster(
                    layer_id,
                    OverlayEdit::BandMath(BandMathEdit::ResetStretch),
                ));
            }
        }
    } else if show_style_controls {
        // RGB Composite stretch (linked min/max + gamma).
        let mut gamma = bm.rgb_gamma;
        match show_min_max_drag(
            ui,
            ("bm_rgb_drag", layer_id),
            bm.rgb_min[0],
            bm.rgb_max[0],
            Some("Reset RGB stretch and gamma to default"),
        ) {
            MinMaxEvent::None => {}
            MinMaxEvent::Drag {
                min: lo,
                max: hi,
                commit,
            } => {
                actions.push(LayerUiAction::edit_raster_drag(
                    layer_id,
                    OverlayEdit::BandMath(BandMathEdit::SetRgbStretch {
                        min: [lo; 3],
                        max: [hi; 3],
                        gamma,
                    }),
                    commit,
                ));
            }
            MinMaxEvent::Reset => {
                actions.push(LayerUiAction::edit_raster(
                    layer_id,
                    OverlayEdit::BandMath(BandMathEdit::ResetStretch),
                ));
            }
        }
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
                state @ (DragCommit::Preview | DragCommit::Commit) => {
                    let commit = matches!(state, DragCommit::Commit);
                    actions.push(LayerUiAction::edit_raster_drag(
                        layer_id,
                        OverlayEdit::BandMath(BandMathEdit::SetRgbStretch {
                            min: bm.rgb_min,
                            max: bm.rgb_max,
                            gamma,
                        }),
                        commit,
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
                    OverlayEdit::BandMath(BandMathEdit::SetSclEnabled(enabled)),
                ));
            }
            if enabled
                && bm.scl_mask_bits != DEFAULT_SCL_MASK_BITS
                && ui.small_button("Default").clicked()
            {
                actions.push(LayerUiAction::edit_raster(
                    layer_id,
                    OverlayEdit::BandMath(BandMathEdit::SetSclMaskBits(DEFAULT_SCL_MASK_BITS)),
                ));
            }
        });
        if bm.scl_mask_enabled {
            let mut bits = bm.scl_mask_bits;
            let mut changed = false;
            ui.horizontal_wrapped(|ui| {
                for (cls, short) in QUICK_SCL_CLASSES {
                    changed |= toggle_scl_checkbox(
                        ui,
                        cls,
                        short,
                        Some(format!("SCL {cls}: {}", SENTINEL2_SCL_NAMES[cls as usize])),
                        &mut bits,
                    );
                }
            });
            egui::CollapsingHeader::new(egui::RichText::new("Other SCL Classes").small())
                .id_salt(("bm_scl_other", layer_id))
                .default_open(false)
                .show(ui, |ui| {
                    for cls in OTHER_SCL_CLASSES {
                        let name = SENTINEL2_SCL_NAMES[cls as usize];
                        changed |= toggle_scl_checkbox(
                            ui,
                            cls,
                            &format!("{cls}: {name}"),
                            None,
                            &mut bits,
                        );
                    }
                });
            if changed {
                actions.push(LayerUiAction::edit_raster(
                    layer_id,
                    OverlayEdit::BandMath(BandMathEdit::SetSclMaskBits(bits)),
                ));
            }
        }
    }
}
