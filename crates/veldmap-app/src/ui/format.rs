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

/// Format a latitude/longitude pair as signed decimal degrees (`lat, lon`)
/// suitable for clipboard pasting into GIS tools and Go-To search boxes.
pub(crate) fn format_signed_dd(lat_deg: f64, lon_deg: f64) -> String {
    format!("{lat_deg:.6}, {lon_deg:.6}")
}

fn split_dms(abs_deg: f64) -> (u32, u32, u32, u32) {
    if !abs_deg.is_finite() {
        return (0, 0, 0, 0);
    }
    let total_cs = (abs_deg.clamp(0.0, 360.0) * 360_000.0).round() as u64;
    let deg = (total_cs / 360_000) as u32;
    let rem_deg = total_cs % 360_000;
    let min = (rem_deg / 6_000) as u32;
    let rem_min = rem_deg % 6_000;
    let sec = (rem_min / 100) as u32;
    let cs = (rem_min % 100) as u32;
    (deg, min, sec, cs)
}

/// Format a latitude/longitude pair in Degrees Minutes Seconds (`DD°MM'SS.SS" N`).
pub(crate) fn format_dms(lat_deg: f64, lon_deg: f64) -> String {
    if !lat_deg.is_finite() || !lon_deg.is_finite() {
        return "—".to_owned();
    }
    let lat_hem = if lat_deg >= 0.0 { 'N' } else { 'S' };
    let lon_hem = if lon_deg >= 0.0 { 'E' } else { 'W' };
    let (ld, lm, ls, lcs) = split_dms(lat_deg.abs());
    let (od, om, os, ocs) = split_dms(lon_deg.abs());
    format!("{ld}°{lm:02}'{ls:02}.{lcs:02}\" {lat_hem}, {od}°{om:02}'{os:02}.{ocs:02}\" {lon_hem}")
}

fn polar_mgrs_fallback(lat_deg: f64, lon_deg: f64) -> Option<&'static str> {
    if !lat_deg.is_finite() || !lon_deg.is_finite() || !(-90.0..=90.0).contains(&lat_deg) {
        return None;
    }
    Some(if lat_deg >= 0.0 { "UPS N" } else { "UPS S" })
}

/// Format a WGS84 coordinate as compact MGRS (10 m precision / 4 digits per
/// axis, or `"UPS N"` / `"UPS S"` in polar zones) for the single-line GIS
/// status bar.
pub(crate) fn format_mgrs_compact(lat_deg: f64, lon_deg: f64) -> String {
    veldmap_core::geo::format_wgs84_to_mgrs(lat_deg, lon_deg, 4).unwrap_or_else(|| {
        polar_mgrs_fallback(lat_deg, lon_deg)
            .unwrap_or("—")
            .to_owned()
    })
}

/// Format a WGS84 coordinate as 1 m precision MGRS (5 digits per axis) for
/// clipboard copy, with clean polar fallback including DD coordinates.
pub(crate) fn format_mgrs_copy(lat_deg: f64, lon_deg: f64) -> String {
    veldmap_core::geo::format_wgs84_to_mgrs(lat_deg, lon_deg, 5).unwrap_or_else(|| {
        match polar_mgrs_fallback(lat_deg, lon_deg) {
            Some(zone) => format!("{zone} ({})", format_lat_lon(lat_deg, lon_deg)),
            None => "—".to_owned(),
        }
    })
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

use crate::layers::{BandChannelSample, CursorReadout, CursorReadoutValue};

/// Format a zoom-invariant [`CursorReadout`] into `(status_text, tooltip_text)`
/// for the single-line GIS status bar.
pub(crate) fn format_cursor_readout(readout: &CursorReadout) -> (String, String) {
    let mut hover_extra: Option<String> = None;
    let val_text = match &readout.value {
        CursorReadoutValue::Rgba([r, g, b, a]) => {
            if *a == 255 {
                format!(
                    "RGB({r},{g},{b}) @ ({},{})",
                    readout.pixel.0, readout.pixel.1
                )
            } else {
                format!(
                    "RGBA({r},{g},{b},{a}) @ ({},{})",
                    readout.pixel.0, readout.pixel.1
                )
            }
        }
        CursorReadoutValue::Scalar { raw, physical } => {
            let unit = readout
                .units
                .as_deref()
                .map(|u| format!(" {u}"))
                .unwrap_or_default();
            let db_part = if readout.log10_db && *physical > 0.0 {
                format!(" ({:.2} dB)", 10.0 * physical.log10())
            } else {
                String::new()
            };
            let raw_part = if readout.has_affine_scaling {
                format!(" [raw {}]", format_scalar(*raw))
            } else {
                String::new()
            };
            format!(
                "Val: {}{unit}{db_part}{raw_part} @ ({},{})",
                format_scalar(*physical),
                readout.pixel.0,
                readout.pixel.1
            )
        }
        CursorReadoutValue::Categorical {
            class_id,
            class_name,
        } => match class_name {
            Some(name) => format!(
                "SCL {class_id}: {name} @ ({},{})",
                readout.pixel.0, readout.pixel.1
            ),
            None => format!(
                "Class {class_id} @ ({},{})",
                readout.pixel.0, readout.pixel.1
            ),
        },
        CursorReadoutValue::BandMath(bm) => {
            let fmt_sample = |sample: BandChannelSample| match sample {
                BandChannelSample::Value(v) => format_scalar(v),
                BandChannelSample::Nodata => "nodata".to_owned(),
                BandChannelSample::Unavailable => "n/a".to_owned(),
            };
            let detailed_channels = bm
                .channels
                .iter()
                .map(|(lbl, sample)| format!("{lbl}: {}", fmt_sample(*sample)))
                .collect::<Vec<_>>()
                .join(", ");
            let detailed_scl = match bm.scl_class {
                Some(cls) => match bm.scl_class_name {
                    Some(name) => format!("\nSCL {cls}: {name}"),
                    None => format!("\nSCL {cls}"),
                },
                None => String::new(),
            };
            hover_extra = Some(format!(
                "{detailed_channels}{detailed_scl} @ ({},{})",
                readout.pixel.0, readout.pixel.1
            ));

            let masked_tag = if bm.scl_masked { " [masked]" } else { "" };
            let short_scl = bm
                .scl_class
                .map(|cls| format!(" · SCL {cls}"))
                .unwrap_or_default();
            let main_str = if bm.is_scalar {
                match bm.computed {
                    Some(v) => {
                        let unit = readout
                            .units
                            .as_deref()
                            .map(|u| format!(" {u}"))
                            .unwrap_or_default();
                        format!(
                            "{}: {}{unit}{masked_tag}",
                            bm.formula_label,
                            format_scalar(v)
                        )
                    }
                    None => {
                        let missing = if bm
                            .channels
                            .iter()
                            .take(2)
                            .any(|(_, s)| matches!(s, BandChannelSample::Unavailable))
                        {
                            "n/a"
                        } else {
                            "nodata"
                        };
                        format!("{}: {missing}{masked_tag}", bm.formula_label)
                    }
                }
            } else {
                let compact_vals = bm
                    .channels
                    .iter()
                    .map(|(_, sample)| fmt_sample(*sample))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("{}: ({compact_vals}){masked_tag}", bm.formula_label)
            };
            format!("{main_str}{short_scl}")
        }
        CursorReadoutValue::Pending => format!("… @ ({},{})", readout.pixel.0, readout.pixel.1),
        CursorReadoutValue::Nodata => format!("nodata @ ({},{})", readout.pixel.0, readout.pixel.1),
        CursorReadoutValue::Unavailable => {
            format!("n/a @ ({},{})", readout.pixel.0, readout.pixel.1)
        }
    };
    let tip = match hover_extra {
        Some(details) => format!(
            "Native base-resolution readout: {}\n{details}",
            readout.layer_name
        ),
        None => format!("Native base-resolution readout: {}", readout.layer_name),
    };
    (val_text, tip)
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
        assert_eq!(format_signed_dd(47.2357, 39.7015), "47.235700, 39.701500");
        assert_eq!(
            format_dms(47.5, -122.25),
            "47°30'00.00\" N, 122°15'00.00\" W"
        );
        // 59.9999 seconds rollover must carry cleanly into the next degree/minute, never `60.00"`.
        assert_eq!(
            format_dms(47.999_999_9, -0.000_000_1),
            "48°00'00.00\" N, 0°00'00.00\" W"
        );
        assert_eq!(format_dms(f64::NAN, 0.0), "—");
        assert_eq!(format_mgrs_compact(0.0, 0.0), "31N AA 6602 0000");
        assert_eq!(format_mgrs_compact(85.0, 10.0), "UPS N");
        assert_eq!(format_mgrs_compact(95.0, 10.0), "—");
        assert_eq!(format_mgrs_compact(f64::NAN, 0.0), "—");
        assert!(format_mgrs_copy(-85.0, 10.0).starts_with("UPS S ("));
        assert_eq!(format_mgrs_copy(f64::NAN, 0.0), "—");
    }
}
