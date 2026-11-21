//! WGS84 ellipsoid and the single conversion from geodetic to ECEF Cartesian
//! coordinates.
//!
//! Everything with a place on Earth goes through this module: the globe body,
//! graticule, camera, outlines, and raster warp meshes.
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

/// Convert geodetic coordinates to single-precision GPU world coordinates.
pub fn position(point: Geodetic) -> World {
    world(point).as_vec3().to_array()
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
pub fn offset(head: (World, World), tail: (World, World)) -> World {
    let high = Vec3::from(head.0) - Vec3::from(tail.0);
    let low = Vec3::from(head.1) - Vec3::from(tail.1);
    (high + low).to_array()
}

/// Intersect a ray `(origin, direction)` with the WGS84 ellipsoid raised by
/// `height_m` metres. Returns `None` if the ray misses or hits behind the
/// origin.
pub fn intersect_at(origin: DVec3, direction: DVec3, height_m: f64) -> Option<DVec3> {
    let stretch = DVec3::new(1.0, 1.0, 1.0 / (1.0 - FLATTENING));
    let start = origin * stretch;
    let along = direction * stretch;
    let radius = 1.0 + height_m / SEMI_MAJOR_M;

    let a = along.dot(along);
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
    let lat = (point.z / ((1.0 - ECCENTRICITY_SQ) * point.x.hypot(point.y))).atan();
    (lat.to_degrees(), point.y.atan2(point.x).to_degrees())
}

/// Angular separation between two `(lat_deg, lon_deg)` points in degrees of
/// arc (`1 deg ~ 111 km`).
pub fn separation(from: (f64, f64), to: (f64, f64)) -> f64 {
    unit(from.0, from.1)
        .dot(unit(to.0, to.1))
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees()
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
    let angle = a.dot(b).clamp(-1.0, 1.0).acos();
    let sin = angle.sin();
    if sin.abs() < 1e-12 {
        return from;
    }

    let (head, tail) = (((1.0 - t) * angle).sin() / sin, (t * angle).sin() / sin);
    let mid = (a * head + b * tail).normalize();
    (
        mid.z.asin().to_degrees(),
        unwind(from.1, mid.y.atan2(mid.x).to_degrees()),
    )
}

/// Minimum longitude span treated as a full circle around the globe.
///
/// Global grids may be written half a pixel short of 360 degrees
/// (for example `-180.0125 -> 179.9875`).
pub const FULL_CIRCLE_DEG: f64 = 359.0;

/// One full turn of longitude in degrees.
pub const TURN_DEG: f64 = 360.0;

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
pub fn sweep(from: (f64, f64), to: (f64, f64)) -> f64 {
    if (to.1 - from.1).abs() >= FULL_CIRCLE_DEG {
        to.1 - from.1
    } else {
        unwind(from.1, to.1) - from.1
    }
}

/// Edge length in degrees of surface arc, matching how the edge is traced.
pub fn edge_span(from: (f64, f64), to: (f64, f64)) -> f64 {
    match along(from, to) {
        Along::Parallel => sweep(from, to).abs() * from.0.to_radians().cos(),
        Along::Meridian => (to.0 - from.0).abs(),
        Along::Arc => separation(from, to),
    }
}

/// Point along an edge at fraction `t`, following a parallel, meridian, or
/// great-circle arc according to the endpoint geometry.
pub fn edge_point(from: (f64, f64), to: (f64, f64), t: f64) -> (f64, f64) {
    match along(from, to) {
        Along::Parallel => (from.0, from.1 + sweep(from, to) * t),
        Along::Meridian => (from.0 + (to.0 - from.0) * t, from.1),
        Along::Arc => between(from, to, t),
    }
}

/// Return whether a densified `(lat_deg, lon_deg)` ring wraps completely around
/// the Earth in longitude (no empty longitude gap >= 180 degrees).
pub fn encircles(ring: &[(f64, f64)]) -> bool {
    if ring.len() < 2 {
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

/// Convert metres into world units (`value / SEMI_MAJOR_M`) as `f32`.
pub fn metres(value: f64) -> f32 {
    (value / SEMI_MAJOR_M) as f32
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
mod tests {
    use super::*;

    #[test]
    fn a_half_turn_stays_where_it_is() {
        assert_eq!(unwind(0.0, 180.0), 180.0);
        assert_eq!(unwind(0.0, -180.0), -180.0);
        assert_eq!(unwind(180.0, -180.0), 180.0);
        assert_eq!(unwind(-180.0, 180.0), -180.0);
    }

    #[test]
    fn a_longitude_lands_within_a_half_turn_of_its_base() {
        for base in [-179.0, 0.0, 37.0, 179.0, 900.0] {
            let dense = [336.457_855_398_182_46, -157.459_371_577_220_8, 0.5];
            for lon in [
                -1081.0, -359.0, -181.0, -180.0, -1.0, 0.0, 180.0, 181.0, 359.0, 1081.0,
            ]
            .into_iter()
            .chain(dense)
            {
                let near = unwind(base, lon);
                assert!(
                    (near - base).abs() <= 180.0,
                    "base {base} and lon {lon} produced {near}"
                );
                let turns = (near - lon) / TURN_DEG;
                assert!(
                    (turns - turns.round()).abs() < 1.0e-9,
                    "shift {} is not a multiple of a full turn",
                    near - lon
                );
            }
        }
    }

    #[test]
    fn an_endless_longitude_does_not_hang_the_unwinding() {
        assert!(unwind(0.0, f64::INFINITY).is_nan());
        assert!(unwind(0.0, f64::NEG_INFINITY).is_nan());
        assert!(unwind(0.0, f64::NAN).is_nan());
        assert!(unwind(f64::INFINITY, 0.0).is_infinite());
        assert_eq!(unwind(0.0, 1.0e300), 0.0);
        assert_eq!(unwind(0.0, f64::MAX), 0.0);
    }

    #[test]
    fn the_halves_add_back_up_to_the_point() {
        let places = [
            world(Geodetic::surface(45.0, 60.0)),
            world(Geodetic::surface(-78.0, -179.0)),
            world(Geodetic {
                lat_deg: 0.0,
                lon_deg: 0.0,
                height_m: 80_000_000.0,
            }),
            DVec3::ZERO,
        ];
        for point in places {
            let (high, low) = parts(point);
            let back = Vec3::from(high).as_dvec3() + Vec3::from(low).as_dvec3();
            let off_m = (back - point).length() * SEMI_MAJOR_M;
            assert!(off_m < 1e-6, "{point:?}: diverged by {off_m} m");
        }
    }

    #[test]
    fn two_halves_tell_apart_what_one_half_cannot() {
        let step_m = 0.01;
        let here = world(Geodetic::surface(45.0, 60.0));
        let there = here + DVec3::Z * (step_m / SEMI_MAJOR_M);

        let (near, far) = (parts(here), parts(there));
        assert_eq!(near.0, far.0, "high half alone cannot resolve 1 cm");

        let moved = f64::from(Vec3::from(offset(far, near)).length()) * SEMI_MAJOR_M;
        assert!(
            (moved - step_m).abs() < 1e-6,
            "{moved} m instead of {step_m}"
        );
    }

    #[test]
    fn the_offset_keeps_what_one_half_would_drop() {
        let here = world(Geodetic::surface(45.0, 60.0));
        for step_m in [0.01, 0.1, 1.0, 100.0, 10_000.0] {
            let there = here + DVec3::Z * (step_m / SEMI_MAJOR_M);
            let step = Vec3::from(offset(parts(there), parts(here)));
            let moved = f64::from(step.length()) * SEMI_MAJOR_M;
            assert!(
                (moved - step_m).abs() < 1e-4,
                "{moved} m instead of {step_m}"
            );
            assert!(step.z > 0.0, "offset points the wrong way: {step:?}");
        }
    }

    #[test]
    fn surface_roundtrip() {
        for &(lat, lon) in &[(0.0, 0.0), (45.0, 60.0), (-33.9, 151.2), (78.0, -179.5)] {
            let (back_lat, back_lon) = surface_at(world(Geodetic::surface(lat, lon)));
            assert!((back_lat - lat).abs() < 1e-9, "{lat} -> {back_lat}");
            assert!((back_lon - lon).abs() < 1e-9, "{lon} -> {back_lon}");
        }
    }

    #[test]
    fn flattening_is_real() {
        let point = world(Geodetic::surface(45.0, 0.0));
        assert!(
            point.z < 45.0_f64.to_radians().sin() - 1e-4,
            "z = {}",
            point.z
        );
        assert!((world(Geodetic::surface(0.0, 0.0)).x - 1.0).abs() < 1e-12);
        let polar = world(Geodetic::surface(90.0, 0.0)).z;
        assert!(
            ((1.0 - polar) * SEMI_MAJOR_M - 21_384.7).abs() < 1.0,
            "polar compression ~21.4 km: {polar}"
        );
    }

    #[test]
    fn between_follows_the_arc_not_the_degrees() {
        let (ne, nw) = ((73.395_172, 2.707_061), (74.523_872, -10.408_985));
        let arc = between(ne, nw, 0.5);
        let flat = ((ne.0 + nw.0) / 2.0, (ne.1 + nw.1) / 2.0);
        let apart = separation(arc, flat).to_radians() * SEMI_MAJOR_M;
        assert!(
            (12_000.0..14_000.0).contains(&apart),
            "diverged by {apart:.0} m"
        );
        assert!(arc.0 > flat.0, "{} vs {}", arc.0, flat.0);

        assert_eq!(between(ne, nw, 0.0), ne);
        assert_eq!(between(ne, nw, 1.0), nw);
    }

    #[test]
    fn between_crosses_the_antimeridian_the_short_way() {
        let (lat, lon) = between((10.0, 179.0), (10.0, -179.0), 0.5);
        assert!((lon - 180.0).abs() < 1e-9, "{lon}");
        assert!(lat > 10.0, "northern hemisphere arc bows poleward: {lat}");
        assert!((separation((10.0, 179.0), (10.0, -179.0)) - 1.9696).abs() < 1e-3);
    }

    #[test]
    fn an_edge_goes_the_way_it_is_named() {
        let (west, east) = ((60.0, 10.0), (60.0, 30.0));
        assert_eq!(edge_point(west, east, 0.5), (60.0, 20.0));
        assert!(between(west, east, 0.5).0 > 60.0);

        let (south, north) = ((50.0, 10.0), (70.0, 10.0));
        assert_eq!(edge_point(south, north, 0.5), (60.0, 10.0));

        let (from, to) = ((73.395_172, 2.707_061), (74.523_872, -10.408_985));
        assert_eq!(edge_point(from, to, 0.5), between(from, to, 0.5));
    }

    #[test]
    fn every_kind_of_edge_lands_on_its_ends() {
        let edges = [
            ((60.0, 10.0), (60.0, 30.0)),
            ((50.0, 10.0), (70.0, 10.0)),
            ((73.4, 2.7), (74.5, -10.4)),
            ((10.0, 179.0), (10.0, -179.0)),
            ((-78.676_191, 12.0), (1.323_561, 12.0)),
            ((43.0, -78.676_191), (43.0, 1.323_561)),
        ];
        for (from, to) in edges {
            assert_eq!(edge_point(from, to, 0.0), from);
            let end = edge_point(from, to, 1.0);
            let apart = (end.0 - to.0).abs() + (unwind(to.1, end.1) - to.1).abs();
            assert!(
                apart < 1e-9,
                "edge end {from:?} -> {to:?} off by {apart} deg"
            );
        }

        let (west, east) = ((80.0, -180.0125), (80.0, 179.9875));
        assert_eq!(edge_point(west, east, 1.0).1, 179.9875);
    }

    #[test]
    fn a_full_circle_stays_a_circle_and_a_seam_takes_the_short_way() {
        let (west, east) = ((80.0, -180.0125), (80.0, 179.9875));
        assert!((sweep(west, east) - 360.0).abs() < 0.05);
        assert!(edge_point(west, east, 0.5).1.abs() < 0.05);

        let (from, to) = ((10.0, 170.0), (10.0, -170.0));
        assert_eq!(sweep(from, to), 20.0);
        assert_eq!(edge_point(from, to, 0.5), (10.0, 180.0));
    }

    #[test]
    fn an_edge_is_measured_by_the_way_it_goes() {
        assert!((edge_span((70.0, 10.0), (70.0, 30.0)) - 6.8404).abs() < 1e-4);
        assert_eq!(edge_span((50.0, 10.0), (70.0, 10.0)), 20.0);
        assert!(
            (edge_span((73.395_172, 2.707_061), (74.523_872, -10.408_985)) - 3.7868).abs() < 1e-4
        );
    }

    #[test]
    fn ray_hits_and_misses() {
        let eye = DVec3::new(2.0, 0.0, 0.0);
        let hit = intersect_at(eye, DVec3::NEG_X, 0.0).expect("ray toward center");
        let (lat, lon) = surface_at(hit);
        assert!(lat.abs() < 1e-9 && lon.abs() < 1e-9);

        let raised = intersect_at(eye, DVec3::NEG_X, 1_000.0).expect("ray toward raised surface");
        let lifted = (raised.x - hit.x) * SEMI_MAJOR_M;
        assert!((lifted - 1_000.0).abs() < 1.0, "lift measured {lifted} m");

        assert!(intersect_at(eye, DVec3::Y, 0.0).is_none());
        assert!(intersect_at(eye, DVec3::X, 0.0).is_none());
    }
}
