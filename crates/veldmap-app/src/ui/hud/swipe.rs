//! Split-screen Swipe curtain divider, side-label badges, and floating
//! top-center selector bar.

use eframe::egui;

use crate::layers::{LayerId, LayerSummary};

use super::{HUD_BORDER_STRONG, HUD_INK_PRIMARY, HUD_PANEL_BG, HUD_PANEL_BG_HOVER, paint_hud_card};

/// User action emitted by the floating split-screen Swipe HUD controls.
#[derive(Clone, Copy, Debug)]
pub(crate) enum SwipeHudAction {
    SetLeft(Option<LayerId>),
    SetRight(Option<LayerId>),
    Swap,
    Close,
}

/// Paint the HiDPI-snapped split-screen swipe curtain divider line and
/// draggable grip handle onto `painter`.
pub(crate) fn paint_swipe_curtain(
    painter: &egui::Painter,
    viewport_rect: egui::Rect,
    divider_x: f32,
    left_label: &str,
    right_label: &str,
    hovered_or_dragging: bool,
) {
    let x = divider_x.clamp(viewport_rect.left(), viewport_rect.right());
    // Subtle dark outer halo for contrast over bright clouds/snow.
    painter.line_segment(
        [
            egui::pos2(x, viewport_rect.top()),
            egui::pos2(x, viewport_rect.bottom()),
        ],
        egui::Stroke::new(3.5, egui::Color32::from_black_alpha(150)),
    );
    painter.line_segment(
        [
            egui::pos2(x, viewport_rect.top()),
            egui::pos2(x, viewport_rect.bottom()),
        ],
        egui::Stroke::new(1.5, HUD_INK_PRIMARY),
    );

    // Draggable circular grip handle at vertical midpoint.
    let center = egui::pos2(x, viewport_rect.center().y);
    let handle_r = if hovered_or_dragging { 16.0 } else { 14.0 };
    painter.circle_filled(
        center,
        handle_r,
        if hovered_or_dragging {
            HUD_PANEL_BG_HOVER
        } else {
            HUD_PANEL_BG
        },
    );
    painter.circle_stroke(center, handle_r, egui::Stroke::new(1.5, HUD_INK_PRIMARY));
    painter.text(
        center,
        egui::Align2::CENTER_CENTER,
        "◀ ▶",
        egui::FontId::proportional(10.0),
        HUD_INK_PRIMARY,
    );

    // Compact Left / Right side label badges flanking the divider below the
    // top Swipe bar and North compass.
    let label_y = viewport_rect.top() + 68.0;
    let font = egui::FontId::proportional(11.0);
    let pad = egui::vec2(6.0, 3.0);
    let paint_side = |text: &str, anchor_x: f32, align_right: bool| {
        if text.is_empty() {
            return;
        }
        let clipped = truncate_badge_label(text, 28);
        let galley = painter.layout_no_wrap(clipped, font.clone(), HUD_INK_PRIMARY);
        let size = galley.size() + pad * 2.0;
        let min_x = if align_right {
            anchor_x - size.x - 10.0
        } else {
            anchor_x + 10.0
        };
        let pill = egui::Rect::from_min_size(egui::pos2(min_x, label_y), size);
        if pill.left() >= viewport_rect.left() + 8.0 && pill.right() <= viewport_rect.right() - 8.0
        {
            paint_hud_card(painter, pill);
            painter.galley(pill.min + pad, galley, egui::Color32::WHITE);
        }
    };
    paint_side(left_label, x, true);
    paint_side(right_label, x, false);
}

fn truncate_badge_label(text: &str, max_chars: usize) -> String {
    let mut chars = text.chars();
    let head: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        let trimmed: String = head.chars().take(max_chars.saturating_sub(1)).collect();
        format!("{trimmed}…")
    } else {
        head
    }
}

fn format_swipe_layer_label(layer: &LayerSummary) -> String {
    if layer.visible {
        layer.name.clone()
    } else {
        format!("{} (hidden)", layer.name)
    }
}

fn show_swipe_side_combo(
    ui: &mut egui::Ui,
    id_salt: &str,
    width: f32,
    selected: Option<LayerId>,
    layers: &[LayerSummary],
    on_select: impl FnOnce(Option<LayerId>) -> SwipeHudAction,
) -> Option<SwipeHudAction> {
    let mut action = None;
    let selected_text = selected
        .and_then(|target| layers.iter().find(|l| l.id == target))
        .map_or_else(
            || "Basemap".to_owned(),
            |l| truncate_badge_label(&format_swipe_layer_label(l), 24),
        );
    let mut picked: Option<Option<LayerId>> = None;
    egui::ComboBox::from_id_salt(id_salt)
        .width(width)
        .selected_text(selected_text)
        .show_ui(ui, |ui| {
            if ui
                .selectable_label(selected.is_none(), "Basemap")
                .on_hover_text("Show underlying basemap and unswiped layers")
                .clicked()
            {
                picked = Some(None);
            }
            for layer in layers.iter().rev() {
                let label = format_swipe_layer_label(layer);
                if ui
                    .selectable_label(selected == Some(layer.id), label)
                    .clicked()
                {
                    picked = Some(Some(layer.id));
                }
            }
        });
    if let Some(choice) = picked {
        action = Some(on_select(choice));
    }
    action
}

/// Interactive floating top-center Swipe selector bar (`Left`, `Swap`, `Right`,
/// `Close`) hosted in a `Background`-ordered `egui::Area`, clamped left of the
/// top-right North compass on narrow viewports.
pub(crate) fn show_swipe_controls(
    ctx: &egui::Context,
    viewport_rect: egui::Rect,
    layers: &[LayerSummary],
    left_layer: Option<LayerId>,
    right_layer: Option<LayerId>,
) -> (Option<SwipeHudAction>, egui::Rect) {
    let mut action = None;
    // Keep at least an 8 pt margin on the left and clear the top-right North
    // compass (`right - 64..right - 16`) by 8 pt (`right - 72`).
    let min_left = viewport_rect.left() + 8.0;
    let max_right = (viewport_rect.right() - 72.0).max(min_left + 180.0);
    let avail_w = (max_right - min_left).max(180.0);
    let show_prefix = avail_w >= 390.0;
    let chrome_w = if show_prefix { 170.0 } else { 118.0 };
    let combo_w = ((avail_w - chrome_w) * 0.5).clamp(72.0, 140.0);
    let est_bar_w = chrome_w + combo_w * 2.0;
    let half_w = est_bar_w * 0.5;
    let anchor_x = viewport_rect
        .center()
        .x
        .min(max_right - half_w)
        .max(min_left + half_w);
    let anchor_y = viewport_rect.top() + 12.0;

    let area_resp = egui::Area::new(egui::Id::unique("swipe_hud_bar"))
        .order(egui::Order::Background)
        .pivot(egui::Align2::CENTER_TOP)
        .fixed_pos(egui::pos2(anchor_x, anchor_y))
        .constrain_to(viewport_rect)
        .interactable(true)
        .show(ctx, |ui| {
            let frame = egui::Frame::NONE
                .fill(HUD_PANEL_BG)
                .stroke(egui::Stroke::new(1.0, HUD_BORDER_STRONG))
                .corner_radius(6.0)
                .inner_margin(egui::Margin::symmetric(8, 5));
            frame
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        if show_prefix {
                            ui.label(egui::RichText::new("Swipe").strong().color(HUD_INK_PRIMARY));
                            ui.separator();
                        }
                        ui.label(egui::RichText::new("L:").small().color(HUD_INK_PRIMARY));
                        if let Some(a) = show_swipe_side_combo(
                            ui,
                            "swipe_left_combo",
                            combo_w,
                            left_layer,
                            layers,
                            SwipeHudAction::SetLeft,
                        ) {
                            action = Some(a);
                        }
                        if ui
                            .small_button("⇄")
                            .on_hover_text("Swap Left and Right layers")
                            .clicked()
                        {
                            action = Some(SwipeHudAction::Swap);
                        }
                        ui.label(egui::RichText::new("R:").small().color(HUD_INK_PRIMARY));
                        if let Some(a) = show_swipe_side_combo(
                            ui,
                            "swipe_right_combo",
                            combo_w,
                            right_layer,
                            layers,
                            SwipeHudAction::SetRight,
                        ) {
                            action = Some(a);
                        }
                        if ui
                            .small_button("×")
                            .on_hover_text("Exit Swipe mode (Esc)")
                            .clicked()
                        {
                            action = Some(SwipeHudAction::Close);
                        }
                    });
                })
                .response
                .rect
        });

    (action, area_resp.inner)
}
