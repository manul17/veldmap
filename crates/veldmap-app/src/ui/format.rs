//! Shared GIS display formatters for coordinates, metric distances,
//! ground resolutions, source pixel sizes, and geodesic areas.

/// Format a latitude/longitude pair as human-readable decimal degrees with
/// hemisphere indicators.
pub(crate) fn format_lat_lon(lat_deg: f64, lon_deg: f64) -> String {
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
pub(crate) fn format_distance_m(metres: f64) -> String {
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
pub(crate) fn format_resolution(m_per_px: f64) -> String {
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
pub(crate) fn format_source_pixel_size(mx: f64, my: f64) -> String {
    if (mx - my).abs() / mx.max(1e-9) <= 0.02 {
        format_resolution(mx)
    } else {
        format!("{} × {}", format_resolution(mx), format_resolution(my))
    }
}

/// Format a scientific scalar cleanly (switching to scientific notation for
/// very large or very small magnitudes so values like `1.5e-6` never display
/// as `0.0000`).
pub(crate) fn format_scalar(v: f64) -> String {
    if !v.is_finite() {
        return "—".to_owned();
    }
    let abs = v.abs();
    if abs >= 10_000.0 || (abs > 0.0 && abs < 0.01) {
        format!("{v:.2e}")
    } else if abs >= 100.0 {
        format!("{v:.1}")
    } else {
        format!("{v:.3}")
    }
}

/// Format area cleanly in `m²`, `ha`, or `km²`.
pub(crate) fn format_area_m2(area_m2: f64) -> String {
    if !area_m2.is_finite() || area_m2 <= 0.0 {
        "0 m²".to_owned()
    } else if area_m2 >= 1_000_000.0 {
        format!("{:.3} km²", area_m2 / 1_000_000.0)
    } else if area_m2 >= 10_000.0 {
        format!("{:.3} ha", area_m2 / 10_000.0)
    } else {
        format!("{area_m2:.1} m²")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_coordinates_distances_resolutions_and_areas_cleanly() {
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
        assert_eq!(format_source_pixel_size(10.0, 10.1), "10.00 m/px");
        assert_eq!(
            format_source_pixel_size(10.0, 20.0),
            "10.00 m/px × 20.00 m/px"
        );
        assert_eq!(format_area_m2(0.0), "0 m²");
        assert_eq!(format_area_m2(-5.0), "0 m²");
        assert_eq!(format_area_m2(f64::NAN), "0 m²");
        assert_eq!(format_area_m2(450.5), "450.5 m²");
        assert_eq!(format_area_m2(25_000.0), "2.500 ha");
        assert_eq!(format_area_m2(3_500_000.0), "3.500 km²");
        assert_eq!(format_scalar(0.0), "0.000");
        assert_eq!(format_scalar(1.5e-6), "1.50e-6");
        assert_eq!(format_scalar(42.125), "42.125");
        assert_eq!(format_scalar(250.4), "250.4");
        assert_eq!(format_scalar(25_000.0), "2.50e4");
        assert_eq!(format_scalar(f64::NAN), "—");
    }
}
