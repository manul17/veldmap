//! Scientific (`R32Float` stretch, colormap) and categorical
//! (`R8Uint`/`R16Uint` palette) styling controls for raster layer cards.

use eframe::egui;

use crate::colormap::{CategoricalPaletteId, ColormapId};
use crate::layers::{
    CategoricalSummary, LayerId, OverlayEdit, ScientificSummary, SingleRasterEdit, StretchPreset,
};
use crate::ui::widgets::{DragCommit, track_drag_commit};

use super::LayerUiAction;
use super::histogram::paint_interactive_histogram;

/// Shared `Ramp` ComboBox + `Inv` checkbox reused across scientific single-band,
/// scalar Band Math, and temporal diff layer cards.
pub(super) fn show_colormap_picker(
    ui: &mut egui::Ui,
    id_salt: (&str, LayerId),
    colormap: ColormapId,
    invert_colormap: bool,
    mut on_change: impl FnMut(ColormapId, bool),
) {
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("Ramp").small());
        let mut cmap = colormap;
        egui::ComboBox::from_id_salt(id_salt)
            .selected_text(egui::RichText::new(cmap.label()).small())
            .width(114.0)
            .show_ui(ui, |ui| {
                for opt in ColormapId::ALL {
                    if ui.selectable_value(&mut cmap, opt, opt.label()).clicked() {
                        on_change(opt, invert_colormap);
                    }
                }
            });
        let mut invert = invert_colormap;
        if ui
            .checkbox(&mut invert, egui::RichText::new("Inv").small())
            .changed()
        {
            on_change(colormap, invert);
        }
    });
}

pub(super) enum MinMaxEvent {
    None,
    Drag { min: f64, max: f64, commit: bool },
    Reset,
}

/// Shared `Min`/`Max` `DragValue` row with `track_drag_commit` and optional
/// `Reset` button reused across scientific and Band Math layer cards.
pub(super) fn show_min_max_drag(
    ui: &mut egui::Ui,
    drag_salt: (&str, LayerId),
    mut lo: f64,
    mut hi: f64,
    reset_tooltip: Option<&str>,
) -> MinMaxEvent {
    let speed = ((hi - lo).abs() * 0.005).max(1e-4);
    let drag_id = ui.scope_id().with(drag_salt);
    let mut event = MinMaxEvent::None;
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("Min").small());
        let r_lo = ui.add(egui::DragValue::new(&mut lo).speed(speed).max_decimals(4));
        ui.label(egui::RichText::new("Max").small());
        let r_hi = ui.add(egui::DragValue::new(&mut hi).speed(speed).max_decimals(4));
        if let Some(tip) = reset_tooltip
            && ui.small_button("Reset").on_hover_text(tip).clicked()
        {
            event = MinMaxEvent::Reset;
            return;
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
                event = MinMaxEvent::Drag {
                    min: lo,
                    max: hi,
                    commit: false,
                };
            }
            DragCommit::Commit => {
                event = MinMaxEvent::Drag {
                    min: lo,
                    max: hi,
                    commit: true,
                };
            }
        }
    });
    event
}

pub(super) fn show_scientific_controls(
    ui: &mut egui::Ui,
    layer_id: LayerId,
    sci: &ScientificSummary,
    actions: &mut Vec<LayerUiAction>,
) {
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("Stretch").small());
        let mut preset = sci.preset;
        egui::ComboBox::from_id_salt(("stretch_preset", layer_id))
            .selected_text(egui::RichText::new(preset.label()).small())
            .width(128.0)
            .show_ui(ui, |ui| {
                for opt in StretchPreset::ALL {
                    if ui.selectable_value(&mut preset, opt, opt.label()).clicked() {
                        actions.push(LayerUiAction::edit_raster(
                            layer_id,
                            OverlayEdit::SingleRaster(SingleRasterEdit::StretchPreset(opt)),
                        ));
                    }
                }
            });
    });

    show_colormap_picker(
        ui,
        ("colormap", layer_id),
        sci.colormap,
        sci.invert_colormap,
        |colormap, invert| {
            actions.push(LayerUiAction::edit_raster(
                layer_id,
                OverlayEdit::SingleRaster(SingleRasterEdit::Colormap { colormap, invert }),
            ));
        },
    );

    if let MinMaxEvent::Drag { min, max, commit } =
        show_min_max_drag(ui, ("stretch_drag_dirty", layer_id), sci.min, sci.max, None)
    {
        actions.push(LayerUiAction::edit_raster_drag(
            layer_id,
            OverlayEdit::SingleRaster(SingleRasterEdit::StretchRange { min, max }),
            commit,
        ));
    }

    paint_interactive_histogram(ui, layer_id, sci, actions);
}

pub(super) fn show_categorical_controls(
    ui: &mut egui::Ui,
    layer_id: LayerId,
    cat: CategoricalSummary,
    actions: &mut Vec<LayerUiAction>,
) {
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("Palette").small());
        let mut pal = cat.palette;
        egui::ComboBox::from_id_salt(("cat_palette", layer_id))
            .selected_text(egui::RichText::new(pal.label()).small())
            .width(136.0)
            .show_ui(ui, |ui| {
                for opt in CategoricalPaletteId::ALL {
                    if opt == CategoricalPaletteId::SourceColorMap && !cat.has_source_color_map {
                        continue;
                    }
                    if ui.selectable_value(&mut pal, opt, opt.label()).clicked() {
                        actions.push(LayerUiAction::edit_raster(
                            layer_id,
                            OverlayEdit::SingleRaster(SingleRasterEdit::CategoricalPalette(opt)),
                        ));
                    }
                }
            });
    });
    if let Some(nd) = cat.nodata_value {
        ui.weak(egui::RichText::new(format!("Nodata class: {nd}")).small());
    }
}
