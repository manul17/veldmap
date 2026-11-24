//! Baseline GIS HUD widgets: dynamic metric scale bar, North compass /
//! Reset-North control, and coordinate/resolution formatting.

use eframe::egui;

/// Format a latitude/longitude pair as human-readable decimal degrees with
/// hemisphere indicators.
pub(super) fn format_lat_lon(lat_deg: f64, lon_deg: f64) -> String {
    let lat_hem = if lat_deg >= 0.0 { 'N' } else { 'S' };
    let lon_hem = if lon_deg >= 0.0 { 'E' } else { 'W' };
    format!(
        "{:.5}° {}, {:.5}° {}",
        lat_deg.abs(),
        lat_hem,
        lon_deg.abs(),
        lon_hem
    )
}

/// Format an altitude or ground distance in metres/kilometres.
pub(super) fn format_distance_m(metres: f64) -> String {
    if metres >= 100_000.0 {
        format!("{:.0} km", metres / 1_000.0)
    } else if metres >= 10_000.0 {
        format!("{:.1} km", metres / 1_000.0)
    } else if metres >= 1_000.0 {
        format!("{:.2} km", metres / 1_000.0)
    } else if metres >= 10.0 {
        format!("{metres:.1} m")
    } else {
        format!("{metres:.2} m")
    }
}

/// Format ground resolution in `mm/px`, `cm/px`, `m/px`, or `km/px`.
pub(super) fn format_resolution(m_per_px: f64) -> String {
    if m_per_px >= 1_000.0 {
        format!("{:.2} km/px", m_per_px / 1_000.0)
    } else if m_per_px >= 1.0 {
        format!("{m_per_px:.2} m/px")
    } else if m_per_px >= 0.01 {
        format!("{:.1} cm/px", m_per_px * 100.0)
    } else {
        format!("{:.1} mm/px", m_per_px * 1_000.0)
    }
}

/// Format an `(x, y)` native source pixel size, collapsing nearly isotropic
/// pixels (`<= 2%` difference) to a single value and formatting anisotropic
/// pixels as `mx × my`.
pub(super) fn format_source_pixel_size(mx: f64, my: f64) -> String {
    if (mx - my).abs() / mx.max(1e-9) <= 0.02 {
        format_resolution(mx)
    } else {
        format!("{} × {}", format_resolution(mx), format_resolution(my))
    }
}

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

/// Horizontal screen width in logical points reserved on the bottom-left for
/// the metric scale bar plus spacing, so right-aligned viewport overlays (e.g.
/// the basemap attribution pill) never collide with it.
fn scale_bar_reserved_width(viewport_width_pt: f32) -> f32 {
    let max_bar_w =
        (viewport_width_pt * 0.35).clamp(SCALE_BAR_MIN_WIDTH_PT, SCALE_BAR_MAX_WIDTH_PT);
    SCALE_BAR_MARGIN_PT + max_bar_w + SCALE_BAR_PAD_X_PT * 2.0 + SCALE_BAR_GAP_PT
}

/// Remaining horizontal width in logical points available to right-aligned
/// bottom-edge viewport overlays (e.g. attribution pill) without colliding
/// with the bottom-left scale bar.
fn available_right_overlay_width(viewport_width_pt: f32, pad_x: f32, margin: f32) -> f32 {
    viewport_width_pt - scale_bar_reserved_width(viewport_width_pt) - (pad_x * 2.0 + margin)
}

/// Paint a crisp floating dynamic metric scale bar in the bottom-left corner
/// of `viewport_rect`.
pub(crate) fn paint_scale_bar(painter: &egui::Painter, viewport_rect: egui::Rect, m_per_pt: f64) {
    if !m_per_pt.is_finite()
        || m_per_pt <= 0.0
        || viewport_rect.width() < 120.0
        || viewport_rect.height() < 80.0
    {
        return;
    }
    let max_width_pt =
        (viewport_rect.width() * 0.35).clamp(SCALE_BAR_MIN_WIDTH_PT, SCALE_BAR_MAX_WIDTH_PT);
    // Hide the tangent-plane scale bar at extreme orbital zoom where the bar
    // would exceed 10,000 km and no longer reflects curved planetary scale.
    if m_per_pt * f64::from(max_width_pt) > 10_000_000.0 {
        return;
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

    painter.rect_filled(
        box_rect,
        6.0,
        egui::Color32::from_rgba_unmultiplied(11, 19, 32, 196),
    );
    painter.rect_stroke(
        box_rect,
        6.0,
        egui::Stroke::new(
            1.0,
            egui::Color32::from_rgba_unmultiplied(168, 185, 155, 110),
        ),
        egui::StrokeKind::Inside,
    );

    let y_bar = box_rect.bottom() - pad_y - 3.0;
    let x0 = box_rect.left() + pad_x;
    let x1 = x0 + bar_w;
    let stroke = egui::Stroke::new(2.0, egui::Color32::from_rgb(245, 242, 232));

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
        egui::Color32::from_rgb(245, 242, 232),
    );
}

/// Interactive floating North compass dial in the top-right corner of
/// `viewport_rect`. Hosted in a foreground `egui::Area` so clicks and hovers
/// never bleed into the globe viewport underneath. Returns `(clicked, rect)`.
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
        .order(egui::Order::Foreground)
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
                egui::Color32::from_rgba_unmultiplied(26, 40, 62, 224)
            } else {
                egui::Color32::from_rgba_unmultiplied(11, 19, 32, 196)
            };
            painter.circle_filled(center, radius, bg);
            painter.circle_stroke(
                center,
                radius,
                egui::Stroke::new(
                    1.0,
                    egui::Color32::from_rgba_unmultiplied(168, 185, 155, 140),
                ),
            );

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
                egui::Color32::from_rgb(220, 78, 66),
                egui::Stroke::NONE,
            ));
            painter.add(egui::Shape::convex_polygon(
                vec![tip_s, left, right],
                egui::Color32::from_rgb(210, 215, 220),
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
pub(crate) fn paint_attribution_badge(
    painter: &egui::Painter,
    rect: egui::Rect,
    attribution: &str,
) {
    if attribution.is_empty() || rect.height() < 60.0 {
        return;
    }

    let pad_x = 8.0;
    let pad_y = 4.0;
    let margin = 10.0;
    let avail = available_right_overlay_width(rect.width(), pad_x, margin);
    if avail < 100.0 {
        return;
    }

    let font = egui::FontId::proportional(11.0);
    let max_width = avail.min(560.0);
    let galley = painter.layout(
        attribution.to_owned(),
        font,
        egui::Color32::from_rgba_unmultiplied(235, 240, 246, 235),
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

    painter.rect_filled(
        badge_rect,
        4.0,
        egui::Color32::from_rgba_unmultiplied(10, 16, 26, 175),
    );
    painter.galley(
        badge_rect.min + egui::vec2(pad_x, pad_y),
        galley,
        egui::Color32::WHITE,
    );
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

    #[test]
    fn formats_coordinates_distances_and_resolutions_cleanly() {
        assert_eq!(format_lat_lon(47.2357, 39.7015), "47.23570° N, 39.70150° E");
        assert_eq!(
            format_lat_lon(-33.8688, -70.6693),
            "33.86880° S, 70.66930° W"
        );
        assert_eq!(format_distance_m(2.5), "2.50 m");
        assert_eq!(format_distance_m(42.3), "42.3 m");
        assert_eq!(format_distance_m(1_500.0), "1.50 km");
        assert_eq!(format_distance_m(25_400.0), "25.4 km");
        assert_eq!(format_distance_m(14_000_000.0), "14000 km");
        assert_eq!(format_resolution(0.002), "2.0 mm/px");
        assert_eq!(format_resolution(0.15), "15.0 cm/px");
        assert_eq!(format_resolution(10.0), "10.00 m/px");
        assert_eq!(format_resolution(2_500.0), "2.50 km/px");
    }
}
