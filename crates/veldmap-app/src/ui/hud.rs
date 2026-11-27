//! Baseline GIS HUD widgets: dynamic metric scale bar, North compass /
//! Reset-North control, scientific colorbar legend, and basemap attribution badge.

use eframe::egui;

use crate::layers::ColorbarInfo;
use crate::ui::widgets::paint_colormap_ramp;

const HUD_PANEL_BG: egui::Color32 = egui::Color32::from_rgba_unmultiplied_const(11, 19, 32, 196);
const HUD_PANEL_BG_HOVER: egui::Color32 =
    egui::Color32::from_rgba_unmultiplied_const(26, 40, 62, 224);
const HUD_ATTRIBUTION_BG: egui::Color32 =
    egui::Color32::from_rgba_unmultiplied_const(10, 16, 26, 175);
const HUD_BORDER_SUBTLE: egui::Color32 =
    egui::Color32::from_rgba_unmultiplied_const(168, 185, 155, 110);
const HUD_BORDER_STRONG: egui::Color32 =
    egui::Color32::from_rgba_unmultiplied_const(168, 185, 155, 140);
const HUD_INK_PRIMARY: egui::Color32 = egui::Color32::from_rgb(245, 242, 232);
const HUD_ATTRIBUTION_TEXT: egui::Color32 =
    egui::Color32::from_rgba_unmultiplied_const(235, 240, 246, 235);
const COMPASS_NEEDLE_NORTH: egui::Color32 = egui::Color32::from_rgb(220, 78, 66);
const COMPASS_NEEDLE_SOUTH: egui::Color32 = egui::Color32::from_rgb(210, 215, 220);

/// Choose a clean `1–2–5` metric scale-bar distance and its exact screen width
/// in logical points for `m_per_pt` ground metres per logical point.
///
/// The returned width always equals `bar_m / m_per_pt` (falling in
/// `[0.4 * max_width_pt, max_width_pt]`), so the visual bar never distorts
/// metric scale.
fn compute_scale_bar(m_per_pt: f64, max_width_pt: f32) -> (String, f32) {
    let max_w = f64::from(max_width_pt).max(1.0);
    if !m_per_pt.is_finite() || m_per_pt <= 0.0 {
        return ("0 m".to_owned(), 0.0);
    }
    let target_m = m_per_pt * max_w;
    let exponent = target_m.log10().floor();
    let decade = 10_f64.powf(exponent);
    let mantissa = target_m / decade;
    let nice = if mantissa >= 5.0 {
        5.0
    } else if mantissa >= 2.0 {
        2.0
    } else {
        1.0
    };
    let bar_m = nice * decade;
    let width_pt = (bar_m / m_per_pt) as f32;

    let label = if bar_m >= 1_000.0 {
        format!("{} km", (bar_m / 1_000.0).round() as u64)
    } else if bar_m >= 1.0 {
        format!("{} m", bar_m.round() as u64)
    } else if bar_m >= 0.01 {
        format!("{} cm", (bar_m * 100.0).round() as u64)
    } else if bar_m >= 0.001 {
        format!("{} mm", (bar_m * 1_000.0).round() as u64)
    } else {
        format!("{:.2} mm", bar_m * 1_000.0)
    };
    (label, width_pt)
}

const SCALE_BAR_MARGIN_PT: f32 = 16.0;
const SCALE_BAR_PAD_X_PT: f32 = 10.0;
const SCALE_BAR_GAP_PT: f32 = 16.0;
const SCALE_BAR_MIN_WIDTH_PT: f32 = 48.0;
const SCALE_BAR_MAX_WIDTH_PT: f32 = 160.0;

fn scale_bar_max_width(available_width: f32) -> f32 {
    (available_width * 0.35).clamp(SCALE_BAR_MIN_WIDTH_PT, SCALE_BAR_MAX_WIDTH_PT)
}

fn paint_hud_card(painter: &egui::Painter, rect: egui::Rect) {
    painter.rect_filled(rect, 6.0, HUD_PANEL_BG);
    painter.rect_stroke(
        rect,
        6.0,
        egui::Stroke::new(1.0, HUD_BORDER_SUBTLE),
        egui::StrokeKind::Inside,
    );
}

/// Horizontal screen width in logical points reserved on the bottom-left for
/// the metric scale bar plus spacing, so right-aligned viewport overlays (e.g.
/// the basemap attribution pill) never collide with it.
fn scale_bar_reserved_width(viewport_width_pt: f32) -> f32 {
    let max_bar_w = scale_bar_max_width(viewport_width_pt);
    SCALE_BAR_MARGIN_PT + max_bar_w + SCALE_BAR_PAD_X_PT * 2.0 + SCALE_BAR_GAP_PT
}

/// Remaining horizontal width in logical points available to right-aligned
/// bottom-edge viewport overlays (e.g. attribution pill) without colliding
/// with the bottom-left scale bar.
fn available_right_overlay_width(viewport_width_pt: f32, pad_x: f32, margin: f32) -> f32 {
    viewport_width_pt - scale_bar_reserved_width(viewport_width_pt) - (pad_x * 2.0 + margin)
}

/// Paint a crisp floating dynamic metric scale bar in the bottom-left corner
/// of `viewport_rect`, returning its painted bounding rectangle if visible.
pub(crate) fn paint_scale_bar(
    painter: &egui::Painter,
    viewport_rect: egui::Rect,
    m_per_pt: f64,
) -> Option<egui::Rect> {
    if !m_per_pt.is_finite()
        || m_per_pt <= 0.0
        || viewport_rect.width() < 120.0
        || viewport_rect.height() < 80.0
    {
        return None;
    }
    let max_width_pt = scale_bar_max_width(viewport_rect.width());
    // Hide the tangent-plane scale bar at extreme orbital zoom where the bar
    // would exceed 10,000 km and no longer reflects curved planetary scale.
    if m_per_pt * f64::from(max_width_pt) > 10_000_000.0 {
        return None;
    }
    let (label, bar_w) = compute_scale_bar(m_per_pt, max_width_pt);
    let margin = SCALE_BAR_MARGIN_PT;
    let pad_x = SCALE_BAR_PAD_X_PT;
    let pad_y = 6.0;
    let box_w = bar_w + pad_x * 2.0;
    let box_h = 34.0;

    let box_rect = egui::Rect::from_min_size(
        egui::pos2(
            viewport_rect.left() + margin,
            viewport_rect.bottom() - margin - box_h,
        ),
        egui::vec2(box_w, box_h),
    );

    paint_hud_card(painter, box_rect);

    let y_bar = box_rect.bottom() - pad_y - 3.0;
    let x0 = box_rect.left() + pad_x;
    let x1 = x0 + bar_w;
    let stroke = egui::Stroke::new(2.0, HUD_INK_PRIMARY);

    painter.line_segment([egui::pos2(x0, y_bar), egui::pos2(x1, y_bar)], stroke);
    painter.line_segment(
        [egui::pos2(x0, y_bar - 5.0), egui::pos2(x0, y_bar + 1.0)],
        stroke,
    );
    painter.line_segment(
        [egui::pos2(x1, y_bar - 5.0), egui::pos2(x1, y_bar + 1.0)],
        stroke,
    );

    painter.text(
        egui::pos2(x0 + bar_w * 0.5, box_rect.top() + 11.0),
        egui::Align2::CENTER_CENTER,
        label,
        egui::FontId::monospace(12.0),
        HUD_INK_PRIMARY,
    );
    Some(box_rect)
}

/// Interactive floating North compass dial in the top-right corner of
/// `viewport_rect`. Hosted in a `Background`-ordered `egui::Area` so it sits
/// cleanly above the `CentralPanel` globe (capturing clicks/hovers) while
/// staying strictly beneath floating `Order::Middle` windows.
pub(crate) fn show_north_compass(
    ctx: &egui::Context,
    viewport_rect: egui::Rect,
    heading_deg: f64,
) -> (bool, egui::Rect) {
    let size = 48.0;
    let margin = 16.0;
    let pos = egui::pos2(
        viewport_rect.right() - margin - size,
        viewport_rect.top() + margin,
    );

    let area_resp = egui::Area::new(egui::Id::unique("north_compass_overlay"))
        .order(egui::Order::Background)
        .fixed_pos(pos)
        .interactable(true)
        .show(ctx, |ui| {
            let (compass_rect, response) =
                ui.allocate_exact_size(egui::vec2(size, size), egui::Sense::click());
            let response = response.on_hover_cursor(egui::CursorIcon::PointingHand);
            let painter = ui.painter();
            let center = compass_rect.center();
            let radius = size * 0.5;
            let bg = if response.hovered() {
                HUD_PANEL_BG_HOVER
            } else {
                HUD_PANEL_BG
            };
            painter.circle_filled(center, radius, bg);
            painter.circle_stroke(center, radius, egui::Stroke::new(1.0, HUD_BORDER_STRONG));

            // Needle points toward geographic North (rotated counter-clockwise by heading).
            let angle = -(heading_deg as f32).to_radians();
            let (sin_a, cos_a) = angle.sin_cos();
            let rot = |dx: f32, dy: f32| {
                egui::pos2(
                    center.x + dx * cos_a - dy * sin_a,
                    center.y + dx * sin_a + dy * cos_a,
                )
            };

            let tip_n = rot(0.0, -radius * 0.48);
            let tip_s = rot(0.0, radius * 0.52);
            let left = rot(-4.0, 0.0);
            let right = rot(4.0, 0.0);

            painter.add(egui::Shape::convex_polygon(
                vec![tip_n, right, left],
                COMPASS_NEEDLE_NORTH,
                egui::Stroke::NONE,
            ));
            painter.add(egui::Shape::convex_polygon(
                vec![tip_s, left, right],
                COMPASS_NEEDLE_SOUTH,
                egui::Stroke::NONE,
            ));
            painter.text(
                rot(0.0, -radius * 0.72),
                egui::Align2::CENTER_CENTER,
                "N",
                egui::FontId::proportional(9.5),
                egui::Color32::WHITE,
            );

            let clicked = response.clicked();
            response.on_hover_text(format!(
                "Heading: {heading_deg:.1}°\nClick or press N to reset North"
            ));
            (clicked, compass_rect)
        });

    area_resp.inner
}

/// Paint a compact, high-contrast attribution pill in the bottom-right corner
/// of the globe viewport so NASA GIBS and EOX CC BY 4.0 credits remain visible
/// without overlapping the bottom-left metric scale bar at narrow widths.
/// Returns its painted bounding rectangle if visible.
pub(crate) fn paint_attribution_badge(
    painter: &egui::Painter,
    rect: egui::Rect,
    attribution: &str,
) -> Option<egui::Rect> {
    if attribution.is_empty() || rect.height() < 60.0 {
        return None;
    }

    let pad_x = 8.0;
    let pad_y = 4.0;
    let margin = 10.0;
    let avail = available_right_overlay_width(rect.width(), pad_x, margin);
    if avail < 100.0 {
        return None;
    }

    let font = egui::FontId::proportional(11.0);
    let max_width = avail.min(560.0);
    let galley = painter.layout(
        attribution.to_owned(),
        font,
        HUD_ATTRIBUTION_TEXT,
        max_width,
    );

    let size = galley.size() + egui::vec2(pad_x * 2.0, pad_y * 2.0);
    let badge_rect = egui::Rect::from_min_size(
        egui::pos2(
            rect.right() - margin - size.x,
            rect.bottom() - margin - size.y,
        ),
        size,
    );

    painter.rect_filled(badge_rect, 4.0, HUD_ATTRIBUTION_BG);
    painter.galley(
        badge_rect.min + egui::vec2(pad_x, pad_y),
        galley,
        egui::Color32::WHITE,
    );
    Some(badge_rect)
}

/// Paint a floating horizontal colorbar HUD above the bottom-left scale bar
/// (or anchored directly to the bottom-left margin when the scale bar is hidden
/// at extreme orbital zoom) whenever the topmost active overlay is on a
/// scientific (`R32Float`) track.
pub(crate) fn paint_colorbar(
    painter: &egui::Painter,
    viewport_rect: egui::Rect,
    scale_rect: Option<egui::Rect>,
    info: &ColorbarInfo,
) -> Option<egui::Rect> {
    if viewport_rect.width() < 260.0 || viewport_rect.height() < 150.0 {
        return None;
    }
    let box_w = (viewport_rect.width() * 0.28).clamp(180.0, 260.0);
    let box_h = 54.0;
    let margin_x = SCALE_BAR_MARGIN_PT;
    let bottom_y = scale_rect.map_or(viewport_rect.bottom() - SCALE_BAR_MARGIN_PT, |sr| {
        sr.top() - 8.0
    });
    let box_rect = egui::Rect::from_min_size(
        egui::pos2(viewport_rect.left() + margin_x, bottom_y - box_h),
        egui::vec2(box_w, box_h),
    );

    paint_hud_card(painter, box_rect);

    let pad_x = 8.0;
    let unit_suffix = if info.log10_db {
        " [dB]".to_owned()
    } else if let Some(ref u) = info.units {
        format!(" [{u}]")
    } else {
        String::new()
    };
    let header = format!("{}{unit_suffix}", info.title);
    let galley = painter.layout_no_wrap(header, egui::FontId::proportional(10.5), HUD_INK_PRIMARY);
    painter.with_clip_rect(box_rect.shrink(4.0)).galley(
        egui::pos2(box_rect.left() + pad_x, box_rect.top() + 5.0),
        galley,
        egui::Color32::WHITE,
    );

    let bar_rect = egui::Rect::from_min_size(
        egui::pos2(box_rect.left() + pad_x, box_rect.top() + 22.0),
        egui::vec2(box_w - pad_x * 2.0, 12.0),
    );
    paint_colormap_ramp(painter, bar_rect, info.colormap, info.invert_colormap, 64);
    painter.rect_stroke(
        bar_rect,
        1.0,
        egui::Stroke::new(0.8, HUD_BORDER_SUBTLE),
        egui::StrokeKind::Inside,
    );

    let mid = 0.5 * (info.min + info.max);
    let tick_y = bar_rect.bottom() + 9.0;
    let tick_font = egui::FontId::monospace(10.0);
    painter.text(
        egui::pos2(bar_rect.left(), tick_y),
        egui::Align2::LEFT_CENTER,
        super::format::format_scalar(info.min),
        tick_font.clone(),
        HUD_INK_PRIMARY,
    );
    painter.text(
        egui::pos2(bar_rect.center().x, tick_y),
        egui::Align2::CENTER_CENTER,
        super::format::format_scalar(mid),
        tick_font.clone(),
        HUD_INK_PRIMARY,
    );
    painter.text(
        egui::pos2(bar_rect.right(), tick_y),
        egui::Align2::RIGHT_CENTER,
        super::format::format_scalar(info.max),
        tick_font,
        HUD_INK_PRIMARY,
    );

    Some(box_rect)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_bar_selects_nice_1_2_5_increments_and_stays_metrically_accurate() {
        let (label, w) = compute_scale_bar(10.0, 150.0);
        assert_eq!(label, "1 km");
        assert!((w - 100.0).abs() < 1e-3);

        let (label_m, w_m) = compute_scale_bar(0.02, 150.0);
        assert_eq!(label_m, "2 m");
        assert!((w_m - 100.0).abs() < 1e-3);

        let (label_cm, w_cm) = compute_scale_bar(0.001_5, 160.0);
        assert_eq!(label_cm, "20 cm");
        assert!((w_cm - (0.20 / 0.001_5) as f32).abs() < 1e-3);

        // Extreme close-up in a narrow pane must use mm rather than distorting width.
        let (label_mm, w_mm) = compute_scale_bar(0.000_5, 12.0);
        assert_eq!(label_mm, "5 mm");
        assert!((w_mm - 10.0).abs() < 1e-3);

        // Non-finite or non-positive resolutions degrade gracefully.
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(compute_scale_bar(bad, 160.0), ("0 m".to_owned(), 0.0));
        }
    }
}
