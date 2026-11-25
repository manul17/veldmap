//! Geographic vector primitives (`GeoPoint`, `Ring`, `Polygon`) and ring
//! vertex deduplication helpers.

/// A latitude/longitude coordinate pair in WGS84 degrees.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeoPoint {
    pub lat: f64,
    pub lon: f64,
}

impl GeoPoint {
    pub const fn new(lat: f64, lon: f64) -> Self {
        Self { lat, lon }
    }
}

/// A closed ring of geographic vertices.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Ring {
    pub points: Vec<GeoPoint>,
}

/// An explicit geographic polygon with an exterior boundary and zero or more
/// interior holes.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Polygon {
    pub exterior: Ring,
    pub holes: Vec<Ring>,
}

impl Polygon {
    /// Construct a polygon without holes.
    pub fn new(exterior: Ring) -> Self {
        Self {
            exterior,
            holes: Vec::new(),
        }
    }

    /// Construct a polygon with explicit interior holes.
    pub fn with_holes(exterior: Ring, holes: Vec<Ring>) -> Self {
        Self { exterior, holes }
    }
}

/// Tolerance in degrees for collapsing coincident consecutive ring vertices.
const RING_DEDUP_EPS_DEG: f64 = 1e-12;

/// Append `point` to a ring accumulator if it is finite and distinct from the
/// previous vertex.
pub(crate) fn push_dedup_geo_point(out: &mut Vec<GeoPoint>, point: GeoPoint) {
    if !(point.lat.is_finite() && point.lon.is_finite()) {
        return;
    }
    if out.last().is_none_or(|prev| {
        (prev.lat - point.lat).abs() > RING_DEDUP_EPS_DEG
            || (prev.lon - point.lon).abs() > RING_DEDUP_EPS_DEG
    }) {
        out.push(point);
    }
}

/// Drop a trailing closing vertex when it duplicates the first vertex of a
/// closed ring.
pub(crate) fn trim_closing_geo_point(out: &mut Vec<GeoPoint>) {
    if out.len() > 1
        && let (Some(first), Some(last)) = (out.first(), out.last())
        && (first.lat - last.lat).abs() <= RING_DEDUP_EPS_DEG
        && (first.lon - last.lon).abs() <= RING_DEDUP_EPS_DEG
    {
        out.pop();
    }
}

/// Remove consecutive duplicate points (including wrap-around `last == first`)
/// from a closed [`GeoPoint`] ring, dropping non-finite coordinates.
pub(crate) fn dedup_geo_ring(points: &[GeoPoint]) -> Vec<GeoPoint> {
    let mut out: Vec<GeoPoint> = Vec::with_capacity(points.len());
    for &p in points {
        push_dedup_geo_point(&mut out, p);
    }
    trim_closing_geo_point(&mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dedup_geo_ring_removes_consecutive_wrap_around_and_non_finite_points() {
        let pts = [
            GeoPoint::new(0.0, 0.0),
            GeoPoint::new(0.0, 0.0),
            GeoPoint::new(f64::NAN, 1.0),
            GeoPoint::new(1.0, 0.0),
            GeoPoint::new(1.0, 1.0),
            GeoPoint::new(1.0, 1.0),
            GeoPoint::new(0.0, 0.0),
        ];
        assert_eq!(
            dedup_geo_ring(&pts),
            vec![
                GeoPoint::new(0.0, 0.0),
                GeoPoint::new(1.0, 0.0),
                GeoPoint::new(1.0, 1.0),
            ]
        );
        assert_eq!(
            dedup_geo_ring(&[GeoPoint::new(1.0, 2.0), GeoPoint::new(1.0, 2.0)]),
            vec![GeoPoint::new(1.0, 2.0)]
        );
        assert!(dedup_geo_ring(&[]).is_empty());
    }
}
