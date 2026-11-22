//! Scene footprint framing and spherical point-in-polygon hit testing.
//!
//! Uses unit-vector azimuth winding along the exact edge tracing rules in
//! [`super::geodesy`] so clicking on a footprint agrees with how it is drawn,
//! including across the antimeridian, around the poles, and for global bands.

use glam::DVec3;

use super::{geodesy, sag};

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

/// Spherical bounding circle around a set of footprint rings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frame {
    pub lat: f64,
    pub lon: f64,
    /// Angular radius in degrees of arc (`1 deg ~ 111 km`).
    pub radius_deg: f64,
}

/// Compute the spherical framing circle enclosing `rings`, or `None` if empty.
pub fn frame(rings: &[Ring]) -> Option<Frame> {
    let first = rings.iter().flat_map(|ring| &ring.points).next()?;
    let (lat, lon) = middle(points(rings)).unwrap_or((first.lat, first.lon));
    let radius_deg = points(rings)
        .map(|point| geodesy::separation((lat, lon), point))
        .fold(0.0, f64::max);

    Some(Frame {
        lat,
        lon: wrap(lon),
        radius_deg,
    })
}

/// Return whether any ring in `rings` covers `(lat, lon)`.
pub fn covers(rings: &[Ring], lat: f64, lon: f64) -> bool {
    rings.iter().any(|ring| inside(&angles(ring), lat, lon))
}

fn points(rings: &[Ring]) -> impl Iterator<Item = (f64, f64)> + '_ {
    rings
        .iter()
        .flat_map(|ring| &ring.points)
        .map(|point| (point.lat, point.lon))
}

fn angles(ring: &Ring) -> Vec<(f64, f64)> {
    ring.points.iter().map(|p| (p.lat, p.lon)).collect()
}

fn traced(ring: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let max_edge_deg = sag::outline_max_edge_deg();

    let mut out = Vec::with_capacity(ring.len());
    for (at, &from) in ring.iter().enumerate() {
        let to = ring[(at + 1) % ring.len()];
        geodesy::trace_edge(from, to, max_edge_deg, |p| out.push(p));
    }
    out
}

fn inside(ring: &[(f64, f64)], lat: f64, lon: f64) -> bool {
    let ring = geodesy::dedup_ring(ring);
    if ring.len() < 3 {
        return false;
    }
    let traced = traced(&ring);
    let turned = turn(&traced, lat, lon);
    if turned.abs() > 180.0 {
        let Some((centre_lat, centre_lon)) = middle(ring.iter().copied()) else {
            return true;
        };
        let centre_turn = turn(&traced, centre_lat, centre_lon);
        if centre_turn.abs() > 180.0 {
            return turned.signum() == centre_turn.signum();
        }
        // Concave ring whose spherical centroid falls outside: reject the
        // antipodal image by requiring the same hemisphere as the centroid.
        return geodesy::unit(lat, lon).dot(geodesy::unit(centre_lat, centre_lon)) > 0.0;
    }

    // Whole-Earth parallel bands cancel winding to ~0 between their parallels;
    // require genuine closed-band seam cuts so tilted polar rings never match.
    !geodesy::band_seam_cuts(&ring).is_empty() && {
        let (low, high) = ring.iter().fold((f64::MAX, f64::MIN), |(low, high), p| {
            (low.min(p.0), high.max(p.0))
        });
        (low..=high).contains(&lat)
    }
}

fn turn(ring: &[(f64, f64)], lat: f64, lon: f64) -> f64 {
    let mut turned = 0.0;
    for (index, from) in ring.iter().enumerate() {
        let to = ring[(index + 1) % ring.len()];
        turned += wrap(bearing(lat, lon, to.0, to.1) - bearing(lat, lon, from.0, from.1));
    }
    turned
}

fn middle(points: impl Iterator<Item = (f64, f64)>) -> Option<(f64, f64)> {
    let sum: DVec3 = points.map(|(lat, lon)| geodesy::unit(lat, lon)).sum();
    (sum.length() > 1e-9).then(|| geodesy::angles(sum.normalize()))
}

fn bearing(from_lat: f64, from_lon: f64, to_lat: f64, to_lon: f64) -> f64 {
    let (from, to) = (from_lat.to_radians(), to_lat.to_radians());
    let delta = (to_lon - from_lon).to_radians();
    let east = delta.sin() * to.cos();
    let north = from.cos() * to.sin() - from.sin() * to.cos() * delta.cos();
    east.atan2(north).to_degrees()
}

fn wrap(degrees: f64) -> f64 {
    geodesy::unwind(0.0, degrees)
}

#[cfg(test)]
#[path = "footprint_tests.rs"]
mod tests;
