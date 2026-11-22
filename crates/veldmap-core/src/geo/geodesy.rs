//! WGS84 ellipsoid and the single conversion from geodetic to ECEF Cartesian
//! coordinates.
//!
//! Everything with a place on Earth goes through this module: the globe body,
//! graticule, camera, outlines, and future tiled overlays.
//!
//! ## Axes and Units
//!
//! Standard ECEF (EPSG:4978): `X` toward the equator at Greenwich, `Y` toward
//! 90 degrees east, `Z` toward the north pole. World coordinates are expressed
//! in fractions of [`SEMI_MAJOR_M`] (`1.0` at the equator) with uniform scale
//! across all three axes so ellipsoidal flattening is preserved exactly.

use glam::{DVec3, Vec3};

/// WGS84 semi-major axis in metres.
pub const SEMI_MAJOR_M: f64 = 6_378_137.0;

/// WGS84 flattening (`1 / 298.257223563`).
pub const FLATTENING: f64 = 1.0 / 298.257_223_563;

/// Square of the first eccentricity, `e^2 = 2f - f^2`.
const ECCENTRICITY_SQ: f64 = FLATTENING * (2.0 - FLATTENING);

/// A world-space 3D coordinate in `f32` layout for GPU vertex buffers.
pub type World = [f32; 3];

/// Geodetic coordinates on the WGS84 ellipsoid.
///
/// `lat_deg` is geodetic latitude in degrees, `lon_deg` is longitude in
/// degrees, and `height_m` is ellipsoidal height in metres.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Geodetic {
    pub lat_deg: f64,
    pub lon_deg: f64,
    pub height_m: f64,
}

impl Geodetic {
    /// A point directly on the WGS84 ellipsoid surface (`height_m = 0.0`).
    pub fn surface(lat_deg: f64, lon_deg: f64) -> Self {
        Self {
            lat_deg,
            lon_deg,
            height_m: 0.0,
        }
    }
}

/// Convert geodetic coordinates to `f64` ECEF world coordinates (in units of
/// [`SEMI_MAJOR_M`]).
pub fn world(point: Geodetic) -> DVec3 {
    let (sin_lon, cos_lon) = point.lon_deg.to_radians().sin_cos();
    parallel(point.lat_deg, point.height_m).at(sin_lon, cos_lon)
}

/// Precomputed latitude-and-height terms for evaluating multiple points along
/// the same parallel without repeating trigonometry or square roots.
#[derive(Clone, Copy, Debug)]
pub struct Parallel {
    sin_lat: f64,
    cos_lat: f64,
    equatorial: f64,
    polar: f64,
}

/// Precompute ECEF conversion factors for a single latitude and ellipsoidal
/// height.
pub fn parallel(lat_deg: f64, height_m: f64) -> Parallel {
    let (sin_lat, cos_lat) = lat_deg.to_radians().sin_cos();
    let curvature = SEMI_MAJOR_M / (1.0 - ECCENTRICITY_SQ * sin_lat * sin_lat).sqrt();

    Parallel {
        sin_lat,
        cos_lat,
        equatorial: (curvature + height_m) / SEMI_MAJOR_M,
        polar: (curvature * (1.0 - ECCENTRICITY_SQ) + height_m) / SEMI_MAJOR_M,
    }
}

impl Parallel {
    /// Evaluate a point on this parallel given `(sin(lon), cos(lon))`.
    pub fn at(&self, sin_lon: f64, cos_lon: f64) -> DVec3 {
        DVec3::new(self.cos_lat * cos_lon, self.cos_lat * sin_lon, self.sin_lat)
            * DVec3::new(self.equatorial, self.equatorial, self.polar)
    }
}

/// Split an `f64` world point into `(high, low)` `f32` halves.
///
/// By Sterbenz's lemma, subtracting two nearby `f32` high halves is exact, and
/// the low halves preserve sub-millimetre precision relative to the eye or
/// local origin even at centimetre-scale zoom.
pub fn parts(point: DVec3) -> (World, World) {
    let high = point.as_vec3();
    (
        high.to_array(),
        (point - high.as_dvec3()).as_vec3().to_array(),
    )
}

/// Compute the `f32` vector offset `head - tail` from two high/low split pairs.
pub(crate) fn offset(head: (World, World), tail: (World, World)) -> World {
    let high = Vec3::from(head.0) - Vec3::from(tail.0);
    let low = Vec3::from(head.1) - Vec3::from(tail.1);
    (high + low).to_array()
}

/// Intersect a ray `(origin, direction)` with the WGS84 ellipsoid raised by
/// `height_m` metres. Exact on the WGS84 ellipsoid (`height_m == 0.0`); for
/// non-zero heights uses a proportionally scaled ellipsoid (`b * (1 + h / a)`).
/// Returns `None` if `direction` is zero or the ray misses / hits behind `origin`.
pub fn intersect_at(origin: DVec3, direction: DVec3, height_m: f64) -> Option<DVec3> {
    let stretch = DVec3::new(1.0, 1.0, 1.0 / (1.0 - FLATTENING));
    let start = origin * stretch;
    let along = direction * stretch;
    let radius = 1.0 + height_m / SEMI_MAJOR_M;

    let a = along.dot(along);
    if !a.is_finite() || a <= 0.0 {
        return None;
    }
    let b = 2.0 * start.dot(along);
    let c = start.dot(start) - radius * radius;
    let discriminant = b * b - 4.0 * a * c;
    if discriminant < 0.0 {
        return None;
    }

    let t = (-b - discriminant.sqrt()) / (2.0 * a);
    (t > 0.0).then(|| origin + direction * t)
}

/// Convert a surface point in ECEF world coordinates back to geodetic
/// `(lat_deg, lon_deg)`.
pub fn surface_at(point: DVec3) -> (f64, f64) {
    let lat = point
        .z
        .atan2((1.0 - ECCENTRICITY_SQ) * point.x.hypot(point.y));
    (lat.to_degrees(), point.y.atan2(point.x).to_degrees())
}

/// Angular separation between two `(lat_deg, lon_deg)` points in degrees of
/// arc (`1 deg ~ 111 km`).
pub fn separation(from: (f64, f64), to: (f64, f64)) -> f64 {
    let (a, b) = (unit(from.0, from.1), unit(to.0, to.1));
    a.cross(b).length().atan2(a.dot(b)).to_degrees()
}

/// Interpolate along the great-circle arc between two `(lat_deg, lon_deg)`
/// points at fraction `t`, keeping longitude continuous with `from.1`.
pub fn between(from: (f64, f64), to: (f64, f64), t: f64) -> (f64, f64) {
    if t <= 0.0 {
        return from;
    }
    if t >= 1.0 {
        return (to.0, unwind(from.1, to.1));
    }

    let (a, b) = (unit(from.0, from.1), unit(to.0, to.1));
    let angle = a.cross(b).length().atan2(a.dot(b));
    let mid = if angle < 1e-6 {
        let v = a * (1.0 - t) + b * t;
        if v.length_squared() <= 0.0 {
            return from;
        }
        v.normalize()
    } else {
        let sin = angle.sin();
        if sin.abs() < 1e-12 {
            return from;
        }
        let (head, tail) = (((1.0 - t) * angle).sin() / sin, (t * angle).sin() / sin);
        (a * head + b * tail).normalize()
    };
    (
        mid.z.asin().to_degrees(),
        unwind(from.1, mid.y.atan2(mid.x).to_degrees()),
    )
}

/// Minimum longitude span treated as a full circle around the globe.
///
/// Pixel-centered global grids may fall half a pixel short of 360 degrees
/// (for example `-179.9875 -> 179.9875`).
pub(crate) const FULL_CIRCLE_DEG: f64 = 359.0;

/// One full turn of longitude in degrees.
pub(crate) const TURN_DEG: f64 = 360.0;

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
pub(crate) fn sweep(from: (f64, f64), to: (f64, f64)) -> f64 {
    if (to.1 - from.1).abs() >= FULL_CIRCLE_DEG {
        to.1 - from.1
    } else {
        unwind(from.1, to.1) - from.1
    }
}

/// Edge length in degrees of surface arc, matching how the edge is traced.
pub(crate) fn edge_span(from: (f64, f64), to: (f64, f64)) -> f64 {
    match along(from, to) {
        Along::Parallel => sweep(from, to).abs() * from.0.to_radians().cos(),
        Along::Meridian => (to.0 - from.0).abs(),
        Along::Arc => separation(from, to),
    }
}

/// Point along an edge at fraction `t`, following a parallel, meridian, or
/// great-circle arc according to the endpoint geometry.
pub(crate) fn edge_point(from: (f64, f64), to: (f64, f64), t: f64) -> (f64, f64) {
    match along(from, to) {
        Along::Parallel => (from.0, from.1 + sweep(from, to) * t),
        Along::Meridian => (from.0 + (to.0 - from.0) * t, from.1),
        Along::Arc => between(from, to, t),
    }
}

/// Emit densified `(lat_deg, lon_deg)` points along an edge from `from`
/// (inclusive) toward `to` (exclusive) so no segment exceeds `max_edge_deg`.
pub(crate) fn trace_edge(
    from: (f64, f64),
    to: (f64, f64),
    max_edge_deg: f64,
    mut emit: impl FnMut((f64, f64)),
) {
    let steps = if max_edge_deg.is_finite() && max_edge_deg > 0.0 {
        let most = (TURN_DEG / max_edge_deg).ceil().max(1.0);
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
pub(crate) fn dedup_ring(points: &[(f64, f64)]) -> Vec<(f64, f64)> {
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
pub(crate) fn band_seam_cuts(places: &[(f64, f64)]) -> Vec<usize> {
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
        let lon_gap = unwind(opens.1, closes.1) - opens.1;
        if opens.0 != closes.0 || lon_gap.abs() > (TURN_DEG - FULL_CIRCLE_DEG) {
            return Vec::new();
        }
        let span: f64 = run.iter().map(|index| sweeps[*index]).sum();
        if span.abs() >= FULL_CIRCLE_DEG {
            carried = true;
        } else if span == 0.0 {
            continue;
        } else {
            return Vec::new();
        }
    }
    if carried { cuts } else { Vec::new() }
}

/// Return whether a densified `(lat_deg, lon_deg)` ring wraps completely around
/// the Earth in longitude (no empty longitude gap >= 180 degrees).
#[cfg(test)]
pub(crate) fn encircles(ring: &[(f64, f64)]) -> bool {
    if ring.len() < 2 || ring.iter().any(|point| !point.1.is_finite()) {
        return false;
    }
    let mut lons: Vec<f64> = ring.iter().map(|point| point.1.rem_euclid(360.0)).collect();
    lons.sort_by(f64::total_cmp);

    let wrap = lons[0] + 360.0 - lons[lons.len() - 1];
    let widest = lons
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .fold(wrap, f64::max);
    widest < 180.0
}

/// Shift `lon_deg` by an integer multiple of 360 degrees so it lies within
/// `[-180, +180]` of `base`.
pub fn unwind(base: f64, lon_deg: f64) -> f64 {
    let apart = lon_deg - base;
    let turns = if apart > 180.0 {
        ((apart - 180.0) / TURN_DEG).ceil()
    } else if apart < -180.0 {
        -((-apart - 180.0) / TURN_DEG).ceil()
    } else {
        0.0
    };
    lon_deg - turns * TURN_DEG
}

/// Outward unit normal to the WGS84 ellipsoid in `f32` vertex layout.
pub fn normal(lat_deg: f64, lon_deg: f64) -> World {
    unit(lat_deg, lon_deg).as_vec3().to_array()
}

/// Convert a unit ECEF direction vector back into `(lat_deg, lon_deg)`.
pub fn angles(unit: DVec3) -> (f64, f64) {
    (
        unit.z.clamp(-1.0, 1.0).asin().to_degrees(),
        unit.y.atan2(unit.x).to_degrees(),
    )
}

/// Unit ECEF vector for a `(lat_deg, lon_deg)` direction (also the exact
/// outward normal to the WGS84 ellipsoid when `lat_deg` is geodetic).
pub fn unit(lat_deg: f64, lon_deg: f64) -> DVec3 {
    let (sin_lat, cos_lat) = lat_deg.to_radians().sin_cos();
    let (sin_lon, cos_lon) = lon_deg.to_radians().sin_cos();
    DVec3::new(cos_lat * cos_lon, cos_lat * sin_lon, sin_lat)
}

#[cfg(test)]
#[path = "geodesy_tests.rs"]
mod tests;
