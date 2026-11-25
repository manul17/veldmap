//! Exact WGS84 ellipsoidal geodesic calculations via Karney's algorithm
//! (`geographiclib-rs`, MIT).
//!
//! Used for interactive distance/area measurement tools, polygon perimeter and
//! area computation (including holes), and geodesic arc densification for
//! vector rendering on the smooth WGS84 ellipsoid.

use geographiclib_rs::{
    Geodesic, GeodesicLine, InverseGeodesic, PolygonArea, Winding, geodesic_capability as caps,
};

use super::footprint::{GeoPoint, dedup_geo_ring};

/// Result of Karney's inverse geodesic problem between two WGS84 points.
#[derive(Clone, Copy, PartialEq, Debug)]
struct GeodesicInverse {
    /// Shortest ellipsoidal distance along WGS84 in metres.
    distance_m: f64,
    /// Forward azimuth at the starting point in degrees (`[-180, 180]`, clockwise from North).
    azimuth1_deg: f64,
}

/// Result of geodesic polygon measurement.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct GeodesicPolygonMeasurement {
    /// Exterior (or total ring) geodesic perimeter in metres.
    pub perimeter_m: f64,
    /// Unsigned ellipsoidal area on WGS84 in square metres.
    pub area_m2: f64,
}

/// Shortest ellipsoidal distance between two WGS84 points in metres.
fn geodesic_distance_m(lat1_deg: f64, lon1_deg: f64, lat2_deg: f64, lon2_deg: f64) -> f64 {
    if !(lat1_deg.is_finite()
        && lon1_deg.is_finite()
        && lat2_deg.is_finite()
        && lon2_deg.is_finite())
    {
        return 0.0;
    }
    let g = Geodesic::wgs84();
    let d: f64 = g.inverse(lat1_deg, lon1_deg, lat2_deg, lon2_deg);
    if d.is_finite() { d.max(0.0) } else { 0.0 }
}

/// Solve Karney's inverse geodesic problem on WGS84.
fn geodesic_inverse(lat1_deg: f64, lon1_deg: f64, lat2_deg: f64, lon2_deg: f64) -> GeodesicInverse {
    let g = Geodesic::wgs84();
    let (distance_m, azimuth1_deg, _azimuth2_deg, _arc_deg): (f64, f64, f64, f64) =
        g.inverse(lat1_deg, lon1_deg, lat2_deg, lon2_deg);
    GeodesicInverse {
        distance_m: if distance_m.is_finite() {
            distance_m.max(0.0)
        } else {
            0.0
        },
        azimuth1_deg: if azimuth1_deg.is_finite() {
            azimuth1_deg
        } else {
            0.0
        },
    }
}

/// Total WGS84 geodesic length in metres of an open polyline.
pub fn polyline_length_m(points: &[GeoPoint]) -> f64 {
    points
        .windows(2)
        .map(|w| geodesic_distance_m(w[0].lat, w[0].lon, w[1].lat, w[1].lon))
        .sum()
}

fn compute_ring_polygon_area(pts: &[GeoPoint]) -> (f64, f64) {
    let g = Geodesic::wgs84();
    let mut pa = PolygonArea::new(&g, Winding::CounterClockwise);
    for p in pts {
        pa.add_point(p.lat, p.lon);
    }
    let (perimeter_m, signed_area_m2, _) = pa.compute(true);
    (perimeter_m, signed_area_m2)
}

/// Exact WGS84 ellipsoidal perimeter (m) and unsigned area (m²) of a closed
/// ring (automatically tolerating either clockwise or counter-clockwise winding
/// and optional duplicate closing vertex).
pub fn ring_perimeter_and_area(ring: &[GeoPoint]) -> GeodesicPolygonMeasurement {
    let pts = dedup_geo_ring(ring);
    if pts.len() < 3 {
        return GeodesicPolygonMeasurement {
            perimeter_m: polyline_length_m(&pts),
            area_m2: 0.0,
        };
    }
    let (perimeter_m, signed_area_m2) = compute_ring_polygon_area(&pts);
    GeodesicPolygonMeasurement {
        perimeter_m: if perimeter_m.is_finite() {
            perimeter_m.max(0.0)
        } else {
            0.0
        },
        area_m2: if signed_area_m2.is_finite() {
            signed_area_m2.abs()
        } else {
            0.0
        },
    }
}

/// Signed ellipsoidal area (m²) of a ring with counter-clockwise = positive
/// (folded into `(-A_earth/2, +A_earth/2]` by `PolygonArea::compute(true)`).
pub(super) fn ring_signed_area_m2(ring: &[GeoPoint]) -> f64 {
    let pts = dedup_geo_ring(ring);
    if pts.len() < 3 {
        return 0.0;
    }
    let (_, signed_area_m2) = compute_ring_polygon_area(&pts);
    if signed_area_m2.is_finite() {
        signed_area_m2
    } else {
        0.0
    }
}

/// Sample points along the WGS84 geodesic connecting `from` to `to` so that
/// consecutive samples are separated by at most `max_chord_deg` degrees of arc
/// (clamped to `[0.05, 10.0]` degrees and capped at `256` segments).
pub(super) fn sample_geodesic_arc(
    from: GeoPoint,
    to: GeoPoint,
    max_chord_deg: f64,
) -> Vec<GeoPoint> {
    if !(from.lat.is_finite() && from.lon.is_finite() && to.lat.is_finite() && to.lon.is_finite()) {
        return Vec::new();
    }
    let inv = geodesic_inverse(from.lat, from.lon, to.lat, to.lon);
    if inv.distance_m <= 1e-6 {
        return vec![from, to];
    }
    let step_deg = if max_chord_deg.is_finite() {
        max_chord_deg.clamp(0.05, 10.0)
    } else {
        2.0
    };
    let arc_deg = (inv.distance_m / super::geodesy::SEMI_MAJOR_M).to_degrees();
    let segments = ((arc_deg / step_deg).ceil() as usize).clamp(1, 256);
    if segments <= 1 {
        return vec![from, to];
    }
    let g = Geodesic::wgs84();
    let line = GeodesicLine::new(&g, from.lat, from.lon, inv.azimuth1_deg, None, None, None);
    let mut out = Vec::with_capacity(segments + 1);
    out.push(from);
    for i in 1..segments {
        let dist = inv.distance_m * (i as f64 / segments as f64);
        let (_a12, lat, lon, ..) =
            line._gen_position(false, dist, caps::LATITUDE | caps::LONGITUDE);
        if lat.is_finite() && lon.is_finite() {
            out.push(GeoPoint::new(lat, lon));
        }
    }
    out.push(to);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn karney_inverse_matches_reference_geodesic_distance() {
        // GeographicLib README example: LAX (34.095925, -118.2884237) to Tallinn (59.4323439, 24.7341649)
        let d = geodesic_distance_m(34.095_925, -118.288_423_7, 59.432_343_9, 24.734_164_9);
        assert!(
            (d - 9_094_718.727_5).abs() < 1e-2,
            "expected ~9,094,718.73 m, got {d}"
        );
        let inv = geodesic_inverse(34.095_925, -118.288_423_7, 59.432_343_9, 24.734_164_9);
        assert!((inv.distance_m - d).abs() < 1e-6);
    }

    #[test]
    fn polygon_area_matches_unit_degree_square() {
        let outer = vec![
            GeoPoint::new(0.0, 0.0),
            GeoPoint::new(0.0, 1.0),
            GeoPoint::new(1.0, 1.0),
            GeoPoint::new(1.0, 0.0),
        ];
        let m = ring_perimeter_and_area(&outer);
        assert!((m.area_m2 - 12_308_778_361.47).abs() < 1.0);
        assert!((m.perimeter_m - 443_770.917).abs() < 0.05);

        // Reversed winding yields identical unsigned area.
        let mut rev = outer.clone();
        rev.reverse();
        let m_rev = ring_perimeter_and_area(&rev);
        assert!((m_rev.area_m2 - m.area_m2).abs() < 1e-3);
    }

    #[test]
    fn sample_geodesic_arc_densifies_long_segments() {
        let pts = sample_geodesic_arc(GeoPoint::new(0.0, 0.0), GeoPoint::new(0.0, 20.0), 2.0);
        assert!(pts.len() >= 11);
        assert_eq!(pts.first().copied(), Some(GeoPoint::new(0.0, 0.0)));
        assert_eq!(pts.last().copied(), Some(GeoPoint::new(0.0, 20.0)));
    }
}
