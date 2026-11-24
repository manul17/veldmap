//! Exact WGS84 ellipsoidal geodesic calculations via Karney's algorithm
//! (`geographiclib-rs`, MIT).
//!
//! Used for interactive distance/area measurement tools, polygon perimeter and
//! area computation (including holes), and geodesic arc densification for
//! vector rendering on the smooth WGS84 ellipsoid.

use geographiclib_rs::{
    DirectGeodesic, Geodesic, GeodesicLine, InverseGeodesic, PolygonArea, Winding,
    geodesic_capability as caps,
};

use super::footprint::{GeoPoint, dedup_geo_ring};

/// Result of Karney's inverse geodesic problem between two WGS84 points.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct GeodesicInverse {
    /// Shortest ellipsoidal distance along WGS84 in metres.
    pub distance_m: f64,
    /// Forward azimuth at the starting point in degrees (`[-180, 180]`, clockwise from North).
    pub azimuth1_deg: f64,
    /// Forward azimuth at the arrival point in degrees (`[-180, 180]`, clockwise from North).
    pub azimuth2_deg: f64,
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
pub fn geodesic_distance_m(lat1_deg: f64, lon1_deg: f64, lat2_deg: f64, lon2_deg: f64) -> f64 {
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
pub fn geodesic_inverse(
    lat1_deg: f64,
    lon1_deg: f64,
    lat2_deg: f64,
    lon2_deg: f64,
) -> GeodesicInverse {
    let g = Geodesic::wgs84();
    let (distance_m, azimuth1_deg, azimuth2_deg, _arc_deg): (f64, f64, f64, f64) =
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
        azimuth2_deg: if azimuth2_deg.is_finite() {
            azimuth2_deg
        } else {
            0.0
        },
    }
}

/// Solve Karney's direct geodesic problem on WGS84, returning
/// `(lat2_deg, lon2_deg, azimuth2_deg)`.
pub fn geodesic_direct(
    lat1_deg: f64,
    lon1_deg: f64,
    azimuth1_deg: f64,
    distance_m: f64,
) -> (f64, f64, f64) {
    if !(lat1_deg.is_finite()
        && lon1_deg.is_finite()
        && azimuth1_deg.is_finite()
        && distance_m.is_finite())
    {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let g = Geodesic::wgs84();
    g.direct(lat1_deg, lon1_deg, azimuth1_deg, distance_m)
}

/// Total WGS84 geodesic length in metres of an open polyline.
pub fn polyline_length_m(points: &[GeoPoint]) -> f64 {
    points
        .windows(2)
        .map(|w| geodesic_distance_m(w[0].lat, w[0].lon, w[1].lat, w[1].lon))
        .sum()
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
    let g = Geodesic::wgs84();
    let mut pa = PolygonArea::new(&g, Winding::CounterClockwise);
    for p in &pts {
        pa.add_point(p.lat, p.lon);
    }
    let (perimeter_m, signed_area_m2, _) = pa.compute(true);
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
pub fn ring_signed_area_m2(ring: &[GeoPoint]) -> f64 {
    let pts = dedup_geo_ring(ring);
    if pts.len() < 3 {
        return 0.0;
    }
    let g = Geodesic::wgs84();
    let mut pa = PolygonArea::new(&g, Winding::CounterClockwise);
    for p in &pts {
        pa.add_point(p.lat, p.lon);
    }
    let (_, signed_area_m2, _) = pa.compute(true);
    if signed_area_m2.is_finite() {
        signed_area_m2
    } else {
        0.0
    }
}

/// Exact WGS84 ellipsoidal area (m²) and total boundary perimeter (m) of a
/// polygon with optional interior holes.
pub fn polygon_with_holes_measurement(
    exterior: &[GeoPoint],
    holes: &[Vec<GeoPoint>],
) -> GeodesicPolygonMeasurement {
    let outer = ring_perimeter_and_area(exterior);
    let mut area = outer.area_m2;
    let mut perimeter = outer.perimeter_m;
    for hole in holes {
        let h = ring_perimeter_and_area(hole);
        area -= h.area_m2;
        perimeter += h.perimeter_m;
    }
    GeodesicPolygonMeasurement {
        perimeter_m: perimeter,
        area_m2: area.max(0.0),
    }
}

/// Sample points along the WGS84 geodesic connecting `from` to `to` so that
/// consecutive samples are separated by at most `max_chord_deg` degrees of arc
/// (clamped to `[0.05, 10.0]` degrees and capped at `256` segments).
pub fn sample_geodesic_arc(from: GeoPoint, to: GeoPoint, max_chord_deg: f64) -> Vec<GeoPoint> {
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
    }

    #[test]
    fn karney_direct_and_inverse_round_trip() {
        let (lat2, lon2, _) = geodesic_direct(40.64, -73.78, 45.0, 1_000_000.0);
        let inv = geodesic_inverse(40.64, -73.78, lat2, lon2);
        assert!((inv.distance_m - 1_000_000.0).abs() < 1e-4);
        assert!((inv.azimuth1_deg - 45.0).abs() < 1e-6);
    }

    #[test]
    fn polygon_area_matches_unit_degree_square_and_subtracts_holes() {
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

        // Hole removes interior area cleanly.
        let hole = vec![
            GeoPoint::new(0.25, 0.25),
            GeoPoint::new(0.75, 0.25),
            GeoPoint::new(0.75, 0.75),
            GeoPoint::new(0.25, 0.75),
        ];
        let hole_m = ring_perimeter_and_area(&hole);
        let with_hole = polygon_with_holes_measurement(&outer, &[hole]);
        assert!((with_hole.area_m2 - (m.area_m2 - hole_m.area_m2)).abs() < 1.0);
    }

    #[test]
    fn sample_geodesic_arc_densifies_long_segments() {
        let pts = sample_geodesic_arc(GeoPoint::new(0.0, 0.0), GeoPoint::new(0.0, 20.0), 2.0);
        assert!(pts.len() >= 11);
        assert_eq!(pts.first().copied(), Some(GeoPoint::new(0.0, 0.0)));
        assert_eq!(pts.last().copied(), Some(GeoPoint::new(0.0, 20.0)));
    }
}
