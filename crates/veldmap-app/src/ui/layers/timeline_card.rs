//! Multi-temporal `Timeline` overlay controls inside the Layer List panel:
//! date scrubber, prev/next stepping, cloud-cover filter, Single / Diff /
//! Cross-Fade comparison mode selector, reference epoch picker, cross-fade
//! `blend_t` slider, diff colormap/stretch controls, and shared Band Math /
//! `SCL` controls.

use eframe::egui;

use crate::colormap::ColormapId;
use crate::layers::{
    BandMathEdit, LayerId, OverlayEdit, TimelineCompareMode, TimelineEdit, TimelineEpochLoadState,
    TimelineSummary,
};
use crate::ui::ERROR_TEXT_COLOR;
use crate::ui::widgets::paint_colormap_ramp;

use super::LayerUiAction;
use super::band_math_card::show_band_math_controls;

fn show_diff_colormap_and_stretch(
    ui: &mut egui::Ui,
    id: LayerId,
    tl: &TimelineSummary,
    actions: &mut Vec<LayerUiAction>,
) {
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("Diff Ramp").small());
        egui::ComboBox::from_id_salt(("tl_diff_cmap", id))
            .selected_text(egui::RichText::new(tl.diff_colormap.label()).small())
            .width(104.0)
            .show_ui(ui, |ui| {
                for cm in ColormapId::ALL {
                    let selected = cm == tl.diff_colormap;
                    if ui.selectable_label(selected, cm.label()).clicked() && !selected {
                        actions.push(LayerUiAction::edit_raster(
                            id,
                            OverlayEdit::Timeline(TimelineEdit::SetDiffColormap {
                                colormap: cm,
                                invert: tl.diff_invert_colormap,
                            }),
                        ));
                    }
                }
            });
        let mut invert = tl.diff_invert_colormap;
        if ui
            .checkbox(&mut invert, egui::RichText::new("Inv").small())
            .changed()
        {
            actions.push(LayerUiAction::edit_raster(
                id,
                OverlayEdit::Timeline(TimelineEdit::SetDiffColormap {
                    colormap: tl.diff_colormap,
                    invert,
                }),
            ));
        }
    });

    let (rect, _) = ui.allocate_exact_size(
        egui::vec2(ui.available_width().max(80.0), 8.0),
        egui::Sense::hover(),
    );
    paint_colormap_ramp(
        ui.painter(),
        rect,
        tl.diff_colormap,
        tl.diff_invert_colormap,
        32,
    );

    let mut min_v = tl.diff_stretch_min;
    let mut max_v = tl.diff_stretch_max;
    let step = ((max_v - min_v).abs() * 0.01).clamp(1e-4, 1.0);
    let mut changed = false;
    let mut commit = false;
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("Δ Range").small());
        let r_min = ui.add(
            egui::DragValue::new(&mut min_v)
                .speed(step)
                .max_decimals(3)
                .prefix("min "),
        );
        let r_max = ui.add(
            egui::DragValue::new(&mut max_v)
                .speed(step)
                .max_decimals(3)
                .prefix("max "),
        );
        changed |= r_min.changed() || r_max.changed();
        commit |= r_min.drag_stopped()
            || r_max.drag_stopped()
            || r_min.lost_focus()
            || r_max.lost_focus();
    });
    if (changed || commit) && (max_v - min_v).abs() > 1e-9 {
        actions.push(LayerUiAction::edit_raster_drag(
            id,
            OverlayEdit::Timeline(TimelineEdit::SetDiffStretchRange {
                min: min_v,
                max: max_v,
            }),
            commit,
        ));
    }
}

pub(super) fn show_timeline_controls(
    ui: &mut egui::Ui,
    id: LayerId,
    tl: &TimelineSummary,
    actions: &mut Vec<LayerUiAction>,
) {
    let active_epoch = tl.epochs.get(tl.active_idx);
    let filtered_pos = tl
        .filtered_indices
        .iter()
        .position(|&i| i == tl.active_idx)
        .unwrap_or(0);
    let filtered_len = tl.filtered_indices.len();

    // Active date badge + cloud % + load state.
    ui.horizontal(|ui| {
        if let Some(ep) = active_epoch {
            ui.strong(egui::RichText::new(&ep.date_label).small())
                .on_hover_text(&ep.label);
            if let Some(cc) = ep.cloud_cover {
                ui.weak(egui::RichText::new(format!("· {cc:.0}% cloud")).small());
            }
            match ep.load_state {
                TimelineEpochLoadState::Loading => {
                    ui.spinner();
                }
                TimelineEpochLoadState::Failed => {
                    ui.colored_label(ERROR_TEXT_COLOR, egui::RichText::new("failed").small())
                        .on_hover_text(ep.error.as_deref().unwrap_or("epoch failed to open"));
                }
                TimelineEpochLoadState::Unloaded | TimelineEpochLoadState::Ready => {}
            }
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.weak(egui::RichText::new(format!("{}/{filtered_len}", filtered_pos + 1)).small());
        });
    });

    // Prev / Scrubber / Next bar.
    if filtered_len >= 2 {
        ui.horizontal(|ui| {
            if ui
                .add_enabled(filtered_pos > 0, egui::Button::new("◀").small())
                .on_hover_text("Previous date")
                .clicked()
            {
                actions.push(LayerUiAction::edit_raster(
                    id,
                    OverlayEdit::Timeline(TimelineEdit::StepEpoch(-1)),
                ));
            }
            let mut pos = filtered_pos;
            let slider = ui.add(
                egui::Slider::new(&mut pos, 0..=filtered_len - 1)
                    .show_value(false)
                    .trailing_fill(true),
            );
            if slider.changed()
                && let Some(&target_idx) = tl.filtered_indices.get(pos)
            {
                actions.push(LayerUiAction::edit_raster_drag(
                    id,
                    OverlayEdit::Timeline(TimelineEdit::SetActiveEpoch(target_idx)),
                    slider.drag_stopped(),
                ));
            } else if slider.drag_stopped() {
                actions.push(LayerUiAction::edit_raster_drag(
                    id,
                    OverlayEdit::Timeline(TimelineEdit::SetActiveEpoch(tl.active_idx)),
                    true,
                ));
            }
            if ui
                .add_enabled(
                    filtered_pos + 1 < filtered_len,
                    egui::Button::new("▶").small(),
                )
                .on_hover_text("Next date")
                .clicked()
            {
                actions.push(LayerUiAction::edit_raster(
                    id,
                    OverlayEdit::Timeline(TimelineEdit::StepEpoch(1)),
                ));
            }
        });
    }

    // Cloud-cover filter (only shown when epochs carry STAC cloud_cover metadata).
    if tl.has_cloud_cover_data {
        ui.horizontal(|ui| {
            let mut enabled = tl.max_cloud_cover.is_some();
            if ui
                .checkbox(&mut enabled, egui::RichText::new("Max cloud").small())
                .changed()
            {
                actions.push(LayerUiAction::edit_raster(
                    id,
                    OverlayEdit::Timeline(TimelineEdit::SetCloudFilter(enabled.then_some(25.0))),
                ));
            }
            if let Some(mut limit) = tl.max_cloud_cover {
                let r = ui.add(
                    egui::Slider::new(&mut limit, 0.0..=100.0)
                        .suffix("%")
                        .fixed_decimals(0),
                );
                if r.changed() || r.drag_stopped() {
                    actions.push(LayerUiAction::edit_raster_drag(
                        id,
                        OverlayEdit::Timeline(TimelineEdit::SetCloudFilter(Some(limit))),
                        r.drag_stopped(),
                    ));
                }
            }
        });
    }

    // Comparison mode selector (Single | Diff | Cross-Fade).
    if tl.epochs.len() >= 2 && (tl.supports_scalar_diff || tl.supports_cross_fade) {
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Mode").small());
            let is_single = matches!(tl.compare, TimelineCompareMode::Single);
            if ui
                .selectable_label(is_single, egui::RichText::new("Single").small())
                .clicked()
                && !is_single
            {
                actions.push(LayerUiAction::edit_raster(
                    id,
                    OverlayEdit::Timeline(TimelineEdit::SetCompareMode(
                        TimelineCompareMode::Single,
                    )),
                ));
            }
            if tl.supports_scalar_diff {
                let is_diff = matches!(tl.compare, TimelineCompareMode::ScalarDiff { .. });
                if ui
                    .selectable_label(is_diff, egui::RichText::new("Diff (B−A)").small())
                    .on_hover_text("Dual-epoch GPU scalar difference (Active B minus Reference A)")
                    .clicked()
                    && !is_diff
                {
                    let ref_idx = tl.compare.ref_idx().unwrap_or(tl.active_idx);
                    actions.push(LayerUiAction::edit_raster(
                        id,
                        OverlayEdit::Timeline(TimelineEdit::SetCompareMode(
                            TimelineCompareMode::ScalarDiff { ref_idx },
                        )),
                    ));
                }
            }
            if tl.supports_cross_fade {
                let is_cf = matches!(tl.compare, TimelineCompareMode::CrossFade { .. });
                if ui
                    .selectable_label(is_cf, egui::RichText::new("Cross-Fade").small())
                    .on_hover_text(
                        "0%-bleed-through GPU RGB cross-fade between Reference A and Active B",
                    )
                    .clicked()
                    && !is_cf
                {
                    let ref_idx = tl.compare.ref_idx().unwrap_or(tl.active_idx);
                    actions.push(LayerUiAction::edit_raster(
                        id,
                        OverlayEdit::Timeline(TimelineEdit::SetCompareMode(
                            TimelineCompareMode::CrossFade {
                                ref_idx,
                                blend_t: 0.5,
                            },
                        )),
                    ));
                }
            }
        });

        // Reference epoch (Epoch A) selector when comparing.
        if let Some(ref_idx) = tl.compare.ref_idx() {
            let ref_label = tl
                .epochs
                .get(ref_idx)
                .map_or("Epoch A", |e| e.date_label.as_str());
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("Ref (A)").small());
                egui::ComboBox::from_id_salt(("tl_ref_epoch", id))
                    .selected_text(egui::RichText::new(ref_label).small())
                    .width(136.0)
                    .show_ui(ui, |ui| {
                        for &ep_idx in &tl.filtered_indices {
                            if let Some(ep) = tl.epochs.get(ep_idx) {
                                let label = match ep.cloud_cover {
                                    Some(cc) => format!("{} ({cc:.0}%)", ep.date_label),
                                    None => ep.date_label.clone(),
                                };
                                let selected = ep_idx == ref_idx;
                                if ui.selectable_label(selected, label).clicked() && !selected {
                                    let next_mode = match tl.compare {
                                        TimelineCompareMode::ScalarDiff { .. } => {
                                            TimelineCompareMode::ScalarDiff { ref_idx: ep_idx }
                                        }
                                        TimelineCompareMode::CrossFade { blend_t, .. } => {
                                            TimelineCompareMode::CrossFade {
                                                ref_idx: ep_idx,
                                                blend_t,
                                            }
                                        }
                                        TimelineCompareMode::Single => TimelineCompareMode::Single,
                                    };
                                    actions.push(LayerUiAction::edit_raster(
                                        id,
                                        OverlayEdit::Timeline(TimelineEdit::SetCompareMode(
                                            next_mode,
                                        )),
                                    ));
                                }
                            }
                        }
                    });
            });
        }

        // Cross-fade blend slider.
        if let TimelineCompareMode::CrossFade { mut blend_t, .. } = tl.compare {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("A ↔ B").small());
                let r = ui.add(
                    egui::Slider::new(&mut blend_t, 0.0..=1.0)
                        .fixed_decimals(2)
                        .trailing_fill(true),
                );
                if r.changed() || r.drag_stopped() {
                    actions.push(LayerUiAction::edit_raster_drag(
                        id,
                        OverlayEdit::Timeline(TimelineEdit::SetBlendT(blend_t)),
                        r.drag_stopped(),
                    ));
                }
            });
        }

        // Diff colormap & stretch range.
        if matches!(tl.compare, TimelineCompareMode::ScalarDiff { .. }) {
            show_diff_colormap_and_stretch(ui, id, tl, actions);
        }
    }

    // Shared BandMath / SCL controls when BandMath-backed (remapping emitted
    // `OverlayEdit::BandMath` into `OverlayEdit::Timeline(TimelineEdit::BandMath)`).
    if let Some(ref bm) = tl.band_math {
        let mut sub_actions = Vec::new();
        show_band_math_controls(ui, id, bm, &mut sub_actions);
        for act in sub_actions {
            if let LayerUiAction::EditRaster {
                id: act_id,
                edit: OverlayEdit::BandMath(bm_edit),
                persist,
            } = act
            {
                // When in ScalarDiff mode, suppress redundant single-epoch colormap/stretch
                // edits if needed, or let them update the underlying BandMath style cleanly.
                let _ = BandMathEdit::ResetStretch;
                actions.push(LayerUiAction::EditRaster {
                    id: act_id,
                    edit: OverlayEdit::Timeline(TimelineEdit::BandMath(bm_edit)),
                    persist,
                });
            } else {
                actions.push(act);
            }
        }
    }
}
