//! Scene footprint framing and spherical point-in-polygon hit testing.
//!
//! Uses unit-vector azimuth winding along parallel, meridian, and great-circle
//! ring edges so hit testing stays consistent across the antimeridian, around
//! the poles, and for global parallel bands.

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

/// Spherical bounding circle around a set of footprint rings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FootprintBounds {
    pub lat: f64,
    pub lon: f64,
    /// Angular radius in degrees of arc (`1 deg ~ 111 km`).
    pub radius_deg: f64,
}

/// Compute the spherical framing circle enclosing an iterator of `(lat, lon)`
/// degree coordinates, or `None` if empty.
pub(crate) fn frame_coords(
    coords: impl Iterator<Item = (f64, f64)> + Clone,
) -> Option<FootprintBounds> {
    let first = coords.clone().next()?;
    let (lat, lon) = geodesy::spherical_centroid(coords.clone()).unwrap_or(first);
    let radius_deg = coords
        .map(|point| geodesy::separation((lat, lon), point))
        .fold(0.0, f64::max);

    Some(FootprintBounds {
        lat,
        lon: wrap(lon),
        radius_deg,
    })
}

/// Remove consecutive duplicate points (including wrap-around `last == first`)
/// from a closed [`GeoPoint`] ring, dropping non-finite coordinates.
pub(crate) fn dedup_geo_ring(points: &[GeoPoint]) -> Vec<GeoPoint> {
    let mut out: Vec<GeoPoint> = Vec::with_capacity(points.len());
    for &p in points {
        if !(p.lat.is_finite() && p.lon.is_finite()) {
            continue;
        }
        if out
            .last()
            .is_none_or(|prev| (prev.lat - p.lat).abs() > 1e-12 || (prev.lon - p.lon).abs() > 1e-12)
        {
            out.push(p);
        }
    }
    if out.len() > 1
        && let (Some(first), Some(last)) = (out.first(), out.last())
        && (first.lat - last.lat).abs() <= 1e-12
        && (first.lon - last.lon).abs() <= 1e-12
    {
        out.pop();
    }
    out
}

/// Compute the spherical framing circle enclosing `rings`, or `None` if empty.
pub fn frame(rings: &[Ring]) -> Option<FootprintBounds> {
    frame_coords(points(rings))
}

/// Compute the spherical framing circle enclosing `polygons` (their exterior
/// boundaries), or `None` if empty.
pub fn frame_polygons(polygons: &[Polygon]) -> Option<FootprintBounds> {
    frame_coords(
        polygons
            .iter()
            .flat_map(|p| &p.exterior.points)
            .map(|p| (p.lat, p.lon)),
    )
}

/// Return whether any ring in `rings` covers `(lat, lon)`.
pub fn covers(rings: &[Ring], lat: f64, lon: f64) -> bool {
    rings.iter().any(|ring| inside(&angles(ring), lat, lon))
}

/// Return whether `polygon` covers `(lat, lon)` (inside its exterior ring and
/// outside all of its interior holes).
pub fn covers_polygon(polygon: &Polygon, lat: f64, lon: f64) -> bool {
    if !inside(&angles(&polygon.exterior), lat, lon) {
        return false;
    }
    !polygon
        .holes
        .iter()
        .any(|hole| inside(&angles(hole), lat, lon))
}

/// Return whether any polygon in `polygons` covers `(lat, lon)`.
pub fn covers_polygons(polygons: &[Polygon], lat: f64, lon: f64) -> bool {
    polygons.iter().any(|poly| covers_polygon(poly, lat, lon))
}

fn points(rings: &[Ring]) -> impl Iterator<Item = (f64, f64)> + Clone + '_ {
    rings
        .iter()
        .flat_map(|ring| &ring.points)
        .map(|point| (point.lat, point.lon))
}

fn angles(ring: &Ring) -> Vec<(f64, f64)> {
    ring.points.iter().map(|p| (p.lat, p.lon)).collect()
}

/// Default chord sagitta tolerance used for geodesic ring densification.
const FOOTPRINT_SAG_TOLERANCE_M: f64 = 70.0;

fn traced(ring: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let max_edge_deg = sag::max_chord_deg(FOOTPRINT_SAG_TOLERANCE_M) * 0.5;

    let mut out = Vec::with_capacity(ring.len());
    for (at, &from) in ring.iter().enumerate() {
        let to = ring[(at + 1) % ring.len()];
        trace_edge(from, to, max_edge_deg, |p| out.push(p));
    }
    out
}

fn inside(ring: &[(f64, f64)], lat: f64, lon: f64) -> bool {
    let ring = dedup_ring(ring);
    if ring.len() < 3 {
        return false;
    }
    let traced = traced(&ring);
    let turned = turn(&traced, lat, lon);
    if turned.abs() > 180.0 {
        let Some((centre_lat, centre_lon)) = geodesy::spherical_centroid(ring.iter().copied())
        else {
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
    !band_seam_cuts(&ring).is_empty() && {
        let (low, high) = ring.iter().fold((f64::MAX, f64::MIN), |(low, high), p| {
            (low.min(p.0), high.max(p.0))
        });
        (low..=high).contains(&lat)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Along {
    Parallel,
    Meridian,
    Arc,
}

fn along(from: (f64, f64), to: (f64, f64)) -> Along {
    match (from.0 == to.0, from.1 == to.1) {
        (true, _) => Along::Parallel,
        (_, true) => Along::Meridian,
        _ => Along::Arc,
    }
}

/// Signed longitude sweep along an edge, preserving full-circle bands while
/// taking the shorter antimeridian branch for all other edges.
fn sweep(from: (f64, f64), to: (f64, f64)) -> f64 {
    if (to.1 - from.1).abs() >= geodesy::FULL_CIRCLE_DEG {
        to.1 - from.1
    } else {
        geodesy::unwind(from.1, to.1) - from.1
    }
}

/// Edge length in degrees of surface arc, matching how the edge is traced.
fn edge_span(from: (f64, f64), to: (f64, f64)) -> f64 {
    match along(from, to) {
        Along::Parallel => sweep(from, to).abs() * from.0.to_radians().cos(),
        Along::Meridian => (to.0 - from.0).abs(),
        Along::Arc => geodesy::separation(from, to),
    }
}

/// Point along an edge at fraction `t`, following a parallel, meridian, or
/// great-circle arc according to the endpoint geometry.
fn edge_point(from: (f64, f64), to: (f64, f64), t: f64) -> (f64, f64) {
    match along(from, to) {
        Along::Parallel => (from.0, from.1 + sweep(from, to) * t),
        Along::Meridian => (from.0 + (to.0 - from.0) * t, from.1),
        Along::Arc => geodesy::between(from, to, t),
    }
}

/// Emit densified `(lat_deg, lon_deg)` points along an edge from `from`
/// (inclusive) toward `to` (exclusive) so no segment exceeds `max_edge_deg`.
fn trace_edge(
    from: (f64, f64),
    to: (f64, f64),
    max_edge_deg: f64,
    mut emit: impl FnMut((f64, f64)),
) {
    let steps = if max_edge_deg.is_finite() && max_edge_deg > 0.0 {
        let most = (geodesy::TURN_DEG / max_edge_deg).ceil().max(1.0);
        let raw = (edge_span(from, to) / max_edge_deg).ceil();
        if raw.is_finite() {
            raw.clamp(1.0, most) as u32
        } else {
            1
        }
    } else {
        1
    };
    for step in 0..steps {
        emit(edge_point(from, to, f64::from(step) / f64::from(steps)));
    }
}

/// Remove consecutive duplicate points (including wrap-around `last == first`)
/// from a closed `(lat_deg, lon_deg)` ring.
fn dedup_ring(points: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let mut out: Vec<(f64, f64)> = Vec::with_capacity(points.len());
    for &p in points {
        if out.last() != Some(&p) {
            out.push(p);
        }
    }
    if out.len() > 1 && out.first() == out.last() {
        out.pop();
    }
    out
}

/// Identify meridian seam cuts of a closed whole-Earth band (constant-latitude
/// parallels wrapping `>= FULL_CIRCLE_DEG` connected by zero-sweep meridians).
/// Returns an empty vector if `places` is not a closed whole-Earth band.
fn band_seam_cuts(places: &[(f64, f64)]) -> Vec<usize> {
    if places.len() < 3 {
        return Vec::new();
    }
    let sweeps: Vec<f64> = (0..places.len())
        .map(|i| sweep(places[i], places[(i + 1) % places.len()]))
        .collect();
    let cuts: Vec<usize> = (0..sweeps.len())
        .filter(|&index| sweeps[index] == 0.0)
        .collect();
    if cuts.is_empty() {
        return Vec::new();
    }

    let mut carried = false;
    for pair in 0..cuts.len() {
        let (from, to) = (cuts[pair], cuts[(pair + 1) % cuts.len()]);
        let run: Vec<usize> = (1..sweeps.len())
            .map(|step| (from + step) % sweeps.len())
            .take_while(|index| *index != to)
            .collect();
        let Some((&first, &last)) = run.first().zip(run.last()) else {
            continue;
        };
        let (opens, closes) = (places[first], places[(last + 1) % places.len()]);
        let lon_gap = geodesy::unwind(opens.1, closes.1) - opens.1;
        if opens.0 != closes.0 || lon_gap.abs() > (geodesy::TURN_DEG - geodesy::FULL_CIRCLE_DEG) {
            return Vec::new();
        }
        let span: f64 = run.iter().map(|index| sweeps[*index]).sum();
        if span.abs() >= geodesy::FULL_CIRCLE_DEG {
            carried = true;
        } else if span == 0.0 {
            continue;
        } else {
            return Vec::new();
        }
    }
    if carried { cuts } else { Vec::new() }
}

/// Pole-safe 3D spherical dihedral winding sum around `p = unit(lat, lon)`.
fn turn(ring: &[(f64, f64)], lat: f64, lon: f64) -> f64 {
    let p = geodesy::unit(lat, lon);
    let mut turned = 0.0;
    for (index, from) in ring.iter().enumerate() {
        let to = ring[(index + 1) % ring.len()];
        let na = p.cross(geodesy::unit(from.0, from.1));
        let nb = p.cross(geodesy::unit(to.0, to.1));
        if na.length_squared() <= 1e-24 || nb.length_squared() <= 1e-24 {
            continue;
        }
        let sin_term = p.dot(na.cross(nb));
        let cos_term = na.dot(nb);
        turned += sin_term.atan2(cos_term).to_degrees();
    }
    turned
}

fn wrap(degrees: f64) -> f64 {
    geodesy::unwind(0.0, degrees)
}

#[cfg(test)]
#[path = "footprint_tests.rs"]
mod tests;
