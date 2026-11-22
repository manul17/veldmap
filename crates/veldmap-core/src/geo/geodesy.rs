//! WGS84 (and general reference) ellipsoid geodesy and `f64` ECEF Cartesian
//! coordinates in metres.
//!
//! Everything with a place on Earth goes through this module: the planetary
//! quadtree, polar cap, camera, frustum culling, projections, footprints, and
//! tiled overlays.
//!
//! ## Axes and Units
//!
//! Standard ECEF (EPSG:4978) in **metres**: `X` toward the equator at
//! Greenwich, `Y` toward 90 degrees east, `Z` toward the north pole.

use glam::DVec3;

/// WGS84 semi-major axis in metres.
pub const SEMI_MAJOR_M: f64 = 6_378_137.0;

/// WGS84 flattening (`1 / 298.257223563`).
pub const FLATTENING: f64 = 1.0 / 298.257_223_563;

/// Square of the WGS84 first eccentricity, `e^2 = 2f - f^2`.
const ECCENTRICITY_SQ: f64 = FLATTENING * (2.0 - FLATTENING);

/// Reference ellipsoid of revolution.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Ellipsoid {
    pub semi_major_m: f64,
    pub flattening: f64,
}

/// WGS84 reference ellipsoid.
pub const WGS84: Ellipsoid = Ellipsoid {
    semi_major_m: SEMI_MAJOR_M,
    flattening: FLATTENING,
};

impl Ellipsoid {
    /// Semi-minor (polar) axis in metres, `b = a * (1 - f)`.
    pub const fn semi_minor_m(&self) -> f64 {
        self.semi_major_m * (1.0 - self.flattening)
    }

    /// Square of the first eccentricity, `e^2 = 2f - f^2`.
    pub const fn eccentricity_sq(&self) -> f64 {
        self.flattening * (2.0 - self.flattening)
    }

    /// Third flattening, `n = f / (2 - f)`.
    pub const fn third_flattening(&self) -> f64 {
        self.flattening / (2.0 - self.flattening)
    }

    /// Convert geodetic `(lat_deg, lon_deg, height_m)` to `f64` ECEF Cartesian
    /// coordinates in **metres**.
    pub fn to_ecef(&self, point: Geodetic) -> DVec3 {
        let (sin_lat, cos_lat) = point.lat_deg.to_radians().sin_cos();
        let (sin_lon, cos_lon) = point.lon_deg.to_radians().sin_cos();
        let e2 = self.eccentricity_sq();
        let curvature = self.semi_major_m / (1.0 - e2 * sin_lat * sin_lat).sqrt();
        let equatorial = curvature + point.height_m;
        let polar = curvature * (1.0 - e2) + point.height_m;
        DVec3::new(
            equatorial * cos_lat * cos_lon,
            equatorial * cos_lat * sin_lon,
            polar * sin_lat,
        )
    }

    /// Convert `f64` ECEF Cartesian coordinates in **metres** back to 3D
    /// geodetic `(lat_deg, lon_deg, height_m)` using Fukushima's (2006)
    /// Halley-accelerated method.
    ///
    /// Accurate to sub-nanometre / `< 1e-12 deg` precision from surface to deep
    /// orbit (`80,000 km`) and exact poles without polar or equatorial
    /// singularities.
    pub fn from_ecef(&self, point: DVec3) -> Geodetic {
        let a = self.semi_major_m;
        let ec = 1.0 - self.flattening;
        let e2 = self.eccentricity_sq();
        let p = point.x.hypot(point.y);
        let z = point.z;

        if p == 0.0 {
            let lat_deg = if z == 0.0 { 0.0 } else { 90.0_f64.copysign(z) };
            return Geodetic {
                lat_deg,
                lon_deg: 0.0,
                height_m: z.abs() - a * ec,
            };
        }

        let lon_deg = point.y.atan2(point.x).to_degrees();
        let p_norm = p / a;
        let z_abs = z.abs();
        let z_norm = ec * (z_abs / a);

        // Initial reduced-latitude direction `(S_0, C_0)` normalized to unit length.
        let mut s = z_abs / a;
        let mut c = ec * p_norm;
        let norm0 = s.hypot(c);
        s /= norm0;
        c /= norm0;

        // Two Halley iterations reach full f64 machine precision from deep
        // underground up through 80,000 km orbital altitudes.
        for _ in 0..2 {
            let f_num = (p_norm * s - z_norm * c) - e2 * s * c;
            let d = z_norm + e2 * s * s * s;
            let f_p = p_norm - e2 * c * c * c;
            let b = 1.5 * e2 * s * c * c * f_num;

            let s_next = d * f_p - b * s;
            let c_next = f_p * f_p - b * c;
            let norm = s_next.hypot(c_next);
            if norm <= 0.0 || !norm.is_finite() {
                break;
            }
            s = s_next / norm;
            c = c_next / norm;
        }

        let cc = ec * c;
        let lat_rad = s.atan2(cc).copysign(z);
        let height_m = (p * cc + z_abs * s - a * ec) / s.hypot(cc);
        Geodetic {
            lat_deg: lat_rad.to_degrees(),
            lon_deg,
            height_m,
        }
    }
}

/// Geodetic coordinates on a reference ellipsoid (WGS84 by default).
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
    /// A point directly on the ellipsoid surface (`height_m = 0.0`).
    pub fn surface(lat_deg: f64, lon_deg: f64) -> Self {
        Self {
            lat_deg,
            lon_deg,
            height_m: 0.0,
        }
    }
}

/// Convert WGS84 geodetic coordinates to `f64` ECEF Cartesian coordinates in
/// **metres**.
pub fn world(point: Geodetic) -> DVec3 {
    WGS84.to_ecef(point)
}

/// Precomputed latitude-and-height terms for evaluating multiple points along
/// the same WGS84 parallel in ECEF metres without repeating trigonometry or
/// square roots.
#[derive(Clone, Copy, Debug)]
pub struct Parallel {
    sin_lat: f64,
    cos_lat: f64,
    equatorial: f64,
    polar: f64,
}

/// Precompute ECEF conversion factors (in metres) for a single WGS84 latitude
/// and ellipsoidal height.
pub fn parallel(lat_deg: f64, height_m: f64) -> Parallel {
    let (sin_lat, cos_lat) = lat_deg.to_radians().sin_cos();
    let curvature = SEMI_MAJOR_M / (1.0 - ECCENTRICITY_SQ * sin_lat * sin_lat).sqrt();

    Parallel {
        sin_lat,
        cos_lat,
        equatorial: curvature + height_m,
        polar: curvature * (1.0 - ECCENTRICITY_SQ) + height_m,
    }
}

impl Parallel {
    /// Evaluate a point on this parallel in ECEF metres given
    /// `(sin(lon), cos(lon))`.
    pub fn at(&self, sin_lon: f64, cos_lon: f64) -> DVec3 {
        DVec3::new(self.cos_lat * cos_lon, self.cos_lat * sin_lon, self.sin_lat)
            * DVec3::new(self.equatorial, self.equatorial, self.polar)
    }
}

/// Intersect a ray `(origin, direction)` (with `origin` in ECEF metres) with
/// the WGS84 ellipsoid raised by `height_m` metres. Exact on the WGS84
/// ellipsoid (`height_m == 0.0`); for non-zero heights uses a proportionally
/// scaled ellipsoid (`b * (1 + h / a)`). Returns `None` if `direction` is zero
/// or the ray misses / hits behind `origin`.
pub fn intersect_at(origin: DVec3, direction: DVec3, height_m: f64) -> Option<DVec3> {
    let stretch = DVec3::new(1.0, 1.0, 1.0 / (1.0 - FLATTENING));
    let start = origin * stretch;
    let along = direction * stretch;
    let radius = SEMI_MAJOR_M + height_m;

    let a = along.dot(along);
    if !a.is_finite() || a <= 0.0 {
        return None;
    }
    let b = 2.0 * start.dot(along);
    let start_len = start.length();
    let c = (start_len - radius) * (start_len + radius);
    let discriminant = b * b - 4.0 * a * c;
    if discriminant < 0.0 {
        return None;
    }

    let t = (-b - discriminant.sqrt()) / (2.0 * a);
    (t > 0.0).then(|| origin + direction * t)
}

/// Convert a WGS84 ECEF point in **metres** back to 3D [`Geodetic`]
/// `(lat_deg, lon_deg, height_m)` using Fukushima (2006).
pub fn geodetic_at(point: DVec3) -> Geodetic {
    WGS84.from_ecef(point)
}

/// Convert a WGS84 ECEF point in **metres** back to geodetic
/// `(lat_deg, lon_deg)`.
pub fn surface_at(point: DVec3) -> (f64, f64) {
    let geo = WGS84.from_ecef(point);
    (geo.lat_deg, geo.lon_deg)
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
pub fn normal(lat_deg: f64, lon_deg: f64) -> [f32; 3] {
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
