//! Multi-temporal `Timeline` overlay controls inside the Layer List panel:
//! date scrubber, prev/next stepping, cloud-cover filter, Single / Diff /
//! Cross-Fade comparison mode selector, reference epoch picker, cross-fade
//! `blend_t` slider, diff colormap/stretch controls, and shared Band Math /
//! Single-raster scientific & categorical / `SCL` controls.

use eframe::egui;

use crate::layers::{
    LayerId, OverlayEdit, TimelineCompareMode, TimelineEdit, TimelineEpochLoadState,
    TimelineSummary,
};
use crate::ui::ERROR_TEXT_COLOR;
use crate::ui::widgets::{DragCommit, paint_colormap_ramp, track_drag_commit};

use super::LayerUiAction;
use super::band_math_card::show_band_math_controls_with_options;
use super::style_controls::{
    MinMaxEvent, show_categorical_controls, show_colormap_picker, show_min_max_drag,
    show_scientific_controls,
};

fn show_diff_colormap_and_stretch(
    ui: &mut egui::Ui,
    id: LayerId,
    tl: &TimelineSummary,
    actions: &mut Vec<LayerUiAction>,
) {
    show_colormap_picker(
        ui,
        ("tl_diff_cmap", id),
        tl.diff_colormap,
        tl.diff_invert_colormap,
        |colormap, invert| {
            actions.push(LayerUiAction::edit_raster(
                id,
                OverlayEdit::Timeline(TimelineEdit::SetDiffColormap { colormap, invert }),
            ));
        },
    );

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

    if let MinMaxEvent::Drag { min, max, commit } = show_min_max_drag(
        ui,
        ("tl_diff_stretch", id),
        tl.diff_stretch_min,
        tl.diff_stretch_max,
        None,
    ) && (max - min).abs() > 1e-9
    {
        actions.push(LayerUiAction::edit_raster_drag(
            id,
            OverlayEdit::Timeline(TimelineEdit::SetDiffStretchRange { min, max }),
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
    let is_scalar_diff = matches!(tl.compare, TimelineCompareMode::ScalarDiff { .. });

    // Active date badge + cloud % + load state (+ Retry on failure).
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
                    if ui.small_button("Retry").clicked() {
                        actions.push(LayerUiAction::edit_raster(
                            id,
                            OverlayEdit::Timeline(TimelineEdit::RetryEpoch(tl.active_idx)),
                        ));
                    }
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
            let scrub_id = ui.scope_id().with(("tl_scrub", id));
            let slider = ui.add(
                egui::Slider::new(&mut pos, 0..=filtered_len - 1)
                    .show_value(false)
                    .trailing_fill(true),
            );
            match track_drag_commit(
                ui,
                scrub_id,
                slider.changed(),
                slider.dragged(),
                slider.drag_stopped(),
            ) {
                DragCommit::None => {}
                state @ (DragCommit::Preview | DragCommit::Commit) => {
                    let target_idx = tl
                        .filtered_indices
                        .get(pos)
                        .copied()
                        .unwrap_or(tl.active_idx);
                    actions.push(LayerUiAction::edit_raster_drag(
                        id,
                        OverlayEdit::Timeline(TimelineEdit::SetActiveEpoch(target_idx)),
                        matches!(state, DragCommit::Commit),
                    ));
                }
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
                let cloud_id = ui.scope_id().with(("tl_cloud", id));
                let r = ui.add(
                    egui::Slider::new(&mut limit, 0.0..=100.0)
                        .suffix("%")
                        .fixed_decimals(0),
                );
                match track_drag_commit(ui, cloud_id, r.changed(), r.dragged(), r.drag_stopped()) {
                    DragCommit::None => {}
                    state @ (DragCommit::Preview | DragCommit::Commit) => {
                        actions.push(LayerUiAction::edit_raster_drag(
                            id,
                            OverlayEdit::Timeline(TimelineEdit::SetCloudFilter(Some(limit))),
                            matches!(state, DragCommit::Commit),
                        ));
                    }
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
            if tl.supports_scalar_diff
                && ui
                    .selectable_label(is_scalar_diff, egui::RichText::new("Diff (B−A)").small())
                    .on_hover_text("Dual-epoch GPU scalar difference (Active B minus Reference A)")
                    .clicked()
                && !is_scalar_diff
            {
                let ref_idx = tl.compare.ref_idx().unwrap_or(tl.active_idx);
                actions.push(LayerUiAction::edit_raster(
                    id,
                    OverlayEdit::Timeline(TimelineEdit::SetCompareMode(
                        TimelineCompareMode::ScalarDiff { ref_idx },
                    )),
                ));
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
            let ref_epoch = tl.epochs.get(ref_idx);
            let ref_label = ref_epoch.map_or("Epoch A", |e| e.date_label.as_str());
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
                                let is_active = ep_idx == tl.active_idx;
                                ui.add_enabled_ui(!is_active, |ui| {
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
                                            TimelineCompareMode::Single => {
                                                TimelineCompareMode::Single
                                            }
                                        };
                                        actions.push(LayerUiAction::edit_raster(
                                            id,
                                            OverlayEdit::Timeline(TimelineEdit::SetCompareMode(
                                                next_mode,
                                            )),
                                        ));
                                    }
                                });
                            }
                        }
                    });
                if let Some(ep) = ref_epoch {
                    match ep.load_state {
                        TimelineEpochLoadState::Loading => {
                            ui.spinner();
                        }
                        TimelineEpochLoadState::Failed => {
                            ui.colored_label(
                                ERROR_TEXT_COLOR,
                                egui::RichText::new("failed").small(),
                            )
                            .on_hover_text(
                                ep.error
                                    .as_deref()
                                    .unwrap_or("reference epoch failed to open"),
                            );
                            if ui.small_button("Retry").clicked() {
                                actions.push(LayerUiAction::edit_raster(
                                    id,
                                    OverlayEdit::Timeline(TimelineEdit::RetryEpoch(ref_idx)),
                                ));
                            }
                        }
                        TimelineEpochLoadState::Unloaded | TimelineEpochLoadState::Ready => {}
                    }
                }
            });
        }

        // Cross-fade blend slider.
        if let TimelineCompareMode::CrossFade { mut blend_t, .. } = tl.compare {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("A ↔ B").small());
                let blend_id = ui.scope_id().with(("tl_blend_t", id));
                let r = ui.add(
                    egui::Slider::new(&mut blend_t, 0.0..=1.0)
                        .fixed_decimals(2)
                        .trailing_fill(true),
                );
                match track_drag_commit(ui, blend_id, r.changed(), r.dragged(), r.drag_stopped()) {
                    DragCommit::None => {}
                    state @ (DragCommit::Preview | DragCommit::Commit) => {
                        actions.push(LayerUiAction::edit_raster_drag(
                            id,
                            OverlayEdit::Timeline(TimelineEdit::SetBlendT(blend_t)),
                            matches!(state, DragCommit::Commit),
                        ));
                    }
                }
            });
        }

        // Diff colormap & stretch range.
        if is_scalar_diff {
            show_diff_colormap_and_stretch(ui, id, tl, actions);
        }
    }

    // Shared BandMath / SCL controls when BandMath-backed (suppressing
    // redundant single-epoch colormap/stretch controls while in ScalarDiff
    // mode, while keeping band labels and SCL cloud/shadow masking visible).
    if let Some(ref bm) = tl.band_math {
        let mut sub_actions = Vec::new();
        show_band_math_controls_with_options(ui, id, bm, !is_scalar_diff, &mut sub_actions);
        for act in sub_actions {
            if let LayerUiAction::EditRaster {
                id: act_id,
                edit: OverlayEdit::BandMath(bm_edit),
                persist,
            } = act
            {
                actions.push(LayerUiAction::EditRaster {
                    id: act_id,
                    edit: OverlayEdit::Timeline(TimelineEdit::BandMath(bm_edit)),
                    persist,
                });
            } else {
                actions.push(act);
            }
        }
    } else if !is_scalar_diff {
        // Shared Single-raster scientific / categorical controls for non-BandMath
        // timelines (e.g. local dated float DEM / SAR / NDVI GeoTIFF folders).
        let mut sub_actions = Vec::new();
        if let Some(ref sci) = tl.scientific {
            show_scientific_controls(ui, id, sci, &mut sub_actions);
        } else if let Some(cat) = tl.categorical {
            show_categorical_controls(ui, id, cat, &mut sub_actions);
        }
        for act in sub_actions {
            if let LayerUiAction::EditRaster {
                id: act_id,
                edit: OverlayEdit::SingleRaster(sr_edit),
                persist,
            } = act
            {
                actions.push(LayerUiAction::EditRaster {
                    id: act_id,
                    edit: OverlayEdit::Timeline(TimelineEdit::SingleRaster(sr_edit)),
                    persist,
                });
            } else {
                actions.push(act);
            }
        }
    }
}
