//! Baseline GIS HUD widgets: dynamic metric scale bar, North compass /
//! Reset-North control, and coordinate/resolution formatting.

use eframe::egui;

/// Format a latitude/longitude pair as human-readable decimal degrees with
/// hemisphere indicators.
pub fn format_lat_lon(lat_deg: f64, lon_deg: f64) -> String {
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
pub fn format_distance_m(metres: f64) -> String {
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
pub fn format_resolution(m_per_px: f64) -> String {
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

/// Choose a clean `1–2–5` metric scale-bar distance and its exact screen width
/// in logical points for `m_per_pt` ground metres per logical point.
pub fn compute_scale_bar(m_per_pt: f64, max_width_pt: f32) -> (String, f32) {
    let max_w = f64::from(max_width_pt).max(1.0);
    let min_w = 24.0_f64.min(max_w);
    let target_m = (m_per_pt * max_w).max(0.01);
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
    let width_pt = (bar_m / m_per_pt).clamp(min_w, max_w) as f32;

    let label = if bar_m >= 1_000.0 {
        format!("{} km", (bar_m / 1_000.0).round() as u64)
    } else if bar_m >= 1.0 {
        format!("{} m", bar_m.round() as u64)
    } else {
        format!("{} cm", (bar_m * 100.0).round() as u64)
    };
    (label, width_pt)
}

/// Paint a crisp floating dynamic metric scale bar in the bottom-left corner
/// of `viewport_rect`.
pub fn paint_scale_bar(painter: &egui::Painter, viewport_rect: egui::Rect, m_per_pt: f64) {
    if !m_per_pt.is_finite() || m_per_pt <= 0.0 {
        return;
    }
    let (label, bar_w) = compute_scale_bar(m_per_pt, 160.0);
    let margin = 16.0;
    let pad_x = 10.0;
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
/// never bleed into the globe viewport underneath. Returns `true` when clicked
/// to reset North.
pub fn show_north_compass(
    ctx: &egui::Context,
    viewport_rect: egui::Rect,
    heading_deg: f64,
) -> bool {
    let size = 46.0;
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

            let tip_n = rot(0.0, -radius * 0.58);
            let tip_s = rot(0.0, radius * 0.58);
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
                rot(0.0, -radius * 0.78),
                egui::Align2::CENTER_CENTER,
                "N",
                egui::FontId::proportional(9.5),
                egui::Color32::WHITE,
            );

            let clicked = response.clicked();
            response.on_hover_text(format!(
                "Heading: {heading_deg:.1}°\nClick or press N to reset North"
            ));
            clicked
        });

    area_resp.inner
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_bar_selects_nice_1_2_5_increments_and_handles_narrow_bounds() {
        let (label, w) = compute_scale_bar(10.0, 150.0);
        assert_eq!(label, "1 km");
        assert!((w - 100.0).abs() < 1e-3);

        let (label_close, w_close) = compute_scale_bar(0.02, 150.0);
        assert_eq!(label_close, "2 m");
        assert!((w_close - 100.0).abs() < 1e-3);

        // Must never panic even when max_width_pt < 24.0.
        let (_, narrow_w) = compute_scale_bar(10.0, 12.0);
        assert!((narrow_w - 12.0).abs() < 1e-3);
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
