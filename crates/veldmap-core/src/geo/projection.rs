//! Projected coordinate systems in metres: transverse cylindrical projections
//! (Krueger series to 4th order in third flattening) with 7-parameter Helmert
//! datum shifts, plus Web Mercator (EPSG:3857).
//!
//! Only WGS84 geodetic coordinates cross the boundary of this module.

use glam::DVec3;

use super::geodesy::{Ellipsoid, Geodetic, SEMI_MAJOR_M, WGS84, unwind};

/// Krasovsky 1940 ellipsoid (SK-42, Gauss-Kruger zones, MSK local systems).
pub const KRASOVSKY: Ellipsoid = Ellipsoid {
    semi_major_m: 6_378_245.0,
    flattening: 1.0 / 298.3,
};

fn rectifying_radius(ellipsoid: Ellipsoid) -> f64 {
    let n = ellipsoid.third_flattening();
    ellipsoid.semi_major_m / (1.0 + n) * (1.0 + n * n / 4.0 + n * n * n * n / 64.0)
}

/// Forward Krueger series coefficients up to `n^4`.
fn krueger_alpha(ellipsoid: Ellipsoid) -> [f64; 4] {
    let n = ellipsoid.third_flattening();
    [
        n / 2.0 - 2.0 * n * n / 3.0 + 5.0 * n * n * n / 16.0 + 41.0 * n * n * n * n / 180.0,
        13.0 * n * n / 48.0 - 3.0 * n * n * n / 5.0 + 557.0 * n * n * n * n / 1440.0,
        61.0 * n * n * n / 240.0 - 103.0 * n * n * n * n / 140.0,
        49_561.0 * n * n * n * n / 161_280.0,
    ]
}

/// Inverse Krueger series coefficients up to `n^4`.
fn krueger_beta(ellipsoid: Ellipsoid) -> [f64; 4] {
    let n = ellipsoid.third_flattening();
    [
        n / 2.0 - 2.0 * n * n / 3.0 + 37.0 * n * n * n / 96.0 - n * n * n * n / 360.0,
        n * n / 48.0 + n * n * n / 15.0 - 437.0 * n * n * n * n / 1440.0,
        17.0 * n * n * n / 480.0 - 37.0 * n * n * n * n / 840.0,
        4_397.0 * n * n * n * n / 161_280.0,
    ]
}

fn geodetic_from_conformal(e: f64, tau_c: f64, rounds: u32) -> f64 {
    let psi = tau_c.asinh();
    let mut phi = tau_c.atan();
    for _ in 0..rounds {
        phi = (psi + e * (e * phi.sin()).atanh()).sinh().atan();
    }
    phi
}

const CONFORMAL_ROUNDS: u32 = 8;

/// Geodetic datum: reference ellipsoid plus 7-parameter Helmert shift to WGS84.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Datum {
    pub ellipsoid: Ellipsoid,
    /// Origin shift in metres.
    pub shift_m: [f64; 3],
    /// Axis rotations in arc-seconds.
    pub rotate_sec: [f64; 3],
    /// Scale correction in parts per million.
    pub scale_ppm: f64,
}

/// Identity WGS84 datum.
pub const WGS84_DATUM: Datum = Datum {
    ellipsoid: WGS84,
    shift_m: [0.0; 3],
    rotate_sec: [0.0; 3],
    scale_ppm: 0.0,
};

/// Pulkovo 1942 (SK-42) to WGS84 parameters per GOST R 51794-2008.
pub const PULKOVO_1942: Datum = Datum {
    ellipsoid: KRASOVSKY,
    shift_m: [23.57, -140.95, -79.8],
    rotate_sec: [0.0, -0.35, -0.79],
    scale_ppm: -0.22,
};

impl Datum {
    fn turn(&self) -> ([f64; 3], f64) {
        let second = std::f64::consts::PI / (180.0 * 3600.0);
        (
            self.rotate_sec.map(|value| value * second),
            1.0 + self.scale_ppm * 1e-6,
        )
    }

    fn helmert(&self, point: DVec3) -> DVec3 {
        let ([wx, wy, wz], scale) = self.turn();
        let [dx, dy, dz] = self.shift_m;
        DVec3::new(
            dx + scale * (point.x + wz * point.y - wy * point.z),
            dy + scale * (-wz * point.x + point.y + wx * point.z),
            dz + scale * (wy * point.x - wx * point.y + point.z),
        )
    }

    /// Exact closed-form inverse of `I + W`: `(I - W + w w^T) / (1 + |w|^2)`.
    fn helmert_back(&self, point: DVec3) -> DVec3 {
        let ([wx, wy, wz], scale) = self.turn();
        let x = (point.x - self.shift_m[0]) / scale;
        let y = (point.y - self.shift_m[1]) / scale;
        let z = (point.z - self.shift_m[2]) / scale;

        let square = wx * wx + wy * wy + wz * wz;
        let along = wx * x + wy * y + wz * z;
        DVec3::new(
            (x - (wz * y - wy * z) + wx * along) / (1.0 + square),
            (y - (-wz * x + wx * z) + wy * along) / (1.0 + square),
            (z - (wy * x - wx * y) + wz * along) / (1.0 + square),
        )
    }

    /// Convert 2D `(lat_deg, lon_deg)` from this datum into WGS84 (discarding
    /// the phantom Helmert ellipsoidal height `Delta h` on 2D output).
    pub fn to_wgs84(&self, lat_deg: f64, lon_deg: f64) -> (f64, f64) {
        if *self == WGS84_DATUM {
            return (lat_deg, lon_deg);
        }
        let point = self.ellipsoid.to_ecef(Geodetic::surface(lat_deg, lon_deg));
        let shifted = WGS84.from_ecef(self.helmert(point));
        (shifted.lat_deg, shifted.lon_deg)
    }

    /// Convert 2D WGS84 `(lat_deg, lon_deg)` into this datum (discarding
    /// phantom Helmert `Delta h` on 2D output).
    pub fn from_wgs84(&self, lat_deg: f64, lon_deg: f64) -> (f64, f64) {
        if *self == WGS84_DATUM {
            return (lat_deg, lon_deg);
        }
        let point = WGS84.to_ecef(Geodetic::surface(lat_deg, lon_deg));
        let shifted = self.ellipsoid.from_ecef(self.helmert_back(point));
        (shifted.lat_deg, shifted.lon_deg)
    }
}

/// Supported projected coordinate systems.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Projection {
    /// Transverse Mercator family (UTM, Gauss-Kruger, local TM systems).
    Transverse(System),
    /// Web Mercator (EPSG:3857).
    WebMercator,
}

impl From<System> for Projection {
    fn from(system: System) -> Self {
        Self::Transverse(system)
    }
}

const WEB_MERCATOR_RADIUS_M: f64 = SEMI_MAJOR_M;

/// Maximum latitude of the square Web Mercator domain (`atan(sinh(pi))`).
pub const WEB_MERCATOR_LIMIT_DEG: f64 = 85.051_128_779_806_59;

impl Projection {
    /// Look up a projection by EPSG (or regional MSK-61) code.
    pub fn from_epsg(code: u32) -> Option<Self> {
        match code {
            3857 => Some(Self::WebMercator),
            _ => System::from_epsg(code).map(Self::Transverse),
        }
    }

    /// Convert WGS84 `(lat_deg, lon_deg)` to projected `(easting_m, northing_m)`.
    pub fn from_geodetic(&self, lat_deg: f64, lon_deg: f64) -> (f64, f64) {
        match self {
            Self::Transverse(system) => system.from_geodetic(lat_deg, lon_deg),
            Self::WebMercator => {
                let lat = lat_deg.clamp(-WEB_MERCATOR_LIMIT_DEG, WEB_MERCATOR_LIMIT_DEG);
                let phi = lat.to_radians();
                (
                    WEB_MERCATOR_RADIUS_M * lon_deg.to_radians(),
                    WEB_MERCATOR_RADIUS_M * (std::f64::consts::FRAC_PI_4 + phi / 2.0).tan().ln(),
                )
            }
        }
    }

    /// Convert projected `(easting_m, northing_m)` to WGS84 `(lat_deg, lon_deg)`.
    pub fn to_geodetic(&self, easting: f64, northing: f64) -> (f64, f64) {
        match self {
            Self::Transverse(system) => system.to_geodetic(easting, northing),
            Self::WebMercator => {
                let lon = (easting / WEB_MERCATOR_RADIUS_M).to_degrees();
                let phi = 2.0 * (northing / WEB_MERCATOR_RADIUS_M).exp().atan()
                    - std::f64::consts::FRAC_PI_2;
                (phi.to_degrees(), lon)
            }
        }
    }
}

/// Transverse cylindrical projection parameters.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct System {
    pub datum: Datum,
    pub central_meridian_deg: f64,
    pub scale: f64,
    pub false_easting_m: f64,
    pub false_northing_m: f64,
}

const UTM_FALSE_EASTING: f64 = 500_000.0;
const UTM_FALSE_NORTHING_SOUTH: f64 = 10_000_000.0;

impl System {
    /// WGS84 UTM zone (`number` in `1..=60`).
    pub fn utm(number: u32, south: bool) -> Self {
        Self {
            datum: WGS84_DATUM,
            central_meridian_deg: f64::from(number) * 6.0 - 183.0,
            scale: 0.9996,
            false_easting_m: UTM_FALSE_EASTING,
            false_northing_m: if south { UTM_FALSE_NORTHING_SOUTH } else { 0.0 },
        }
    }

    /// Six-degree Gauss-Kruger zone on Pulkovo 1942 (`zone` in `1..=32`).
    pub fn gauss_kruger(zone: u32) -> Self {
        Self {
            datum: PULKOVO_1942,
            central_meridian_deg: f64::from(zone) * 6.0 - 3.0,
            scale: 1.0,
            false_easting_m: f64::from(zone) * 1_000_000.0 + 500_000.0,
            false_northing_m: 0.0,
        }
    }

    /// Local MSK-61 (Rostov oblast) 3-degree zone (`1..=3`) on Pulkovo 1942.
    pub fn msk61(zone: u32) -> Option<Self> {
        let central_meridian_deg = match zone {
            1 => 37.983_333_333_33,
            2 => 40.983_333_333_33,
            3 => 43.983_333_333_33,
            _ => return None,
        };
        Some(Self {
            datum: PULKOVO_1942,
            central_meridian_deg,
            scale: 1.0,
            false_easting_m: f64::from(zone) * 1_000_000.0 + 300_000.0,
            false_northing_m: -4_811_057.628,
        })
    }

    /// Look up a transverse projection system by EPSG (or MSK-61 `633610x`) code.
    pub fn from_epsg(code: u32) -> Option<Self> {
        match code {
            32601..=32660 => Some(Self::utm(code - 32600, false)),
            32701..=32760 => Some(Self::utm(code - 32700, true)),
            28401..=28432 => Some(Self::gauss_kruger(code - 28400)),
            6_336_101..=6_336_103 => Self::msk61(code - 6_336_100),
            _ => None,
        }
    }

    /// Convert WGS84 `(lat_deg, lon_deg)` to projected `(easting_m, northing_m)`.
    pub fn from_geodetic(&self, lat_deg: f64, lon_deg: f64) -> (f64, f64) {
        let (lat_deg, lon_deg) = self.datum.from_wgs84(lat_deg, lon_deg);
        let ellipsoid = self.datum.ellipsoid;
        let phi = lat_deg.to_radians();
        let lam = (lon_deg - self.central_meridian_deg).to_radians();

        let e = ellipsoid.eccentricity_sq().sqrt();
        let tau = phi.tan();
        let sigma = (e * (e * tau / tau.hypot(1.0)).atanh()).sinh();
        let tau_c = tau * sigma.hypot(1.0) - sigma * tau.hypot(1.0);

        let xi = tau_c.atan2(lam.cos());
        let eta = (lam.sin() / tau_c.hypot(lam.cos())).asinh();

        let (mut x, mut y) = (xi, eta);
        for (j, a) in krueger_alpha(ellipsoid).iter().enumerate() {
            let k = 2.0 * (j as f64 + 1.0);
            x += a * (k * xi).sin() * (k * eta).cosh();
            y += a * (k * xi).cos() * (k * eta).sinh();
        }

        let radius = self.scale * rectifying_radius(ellipsoid);
        (
            self.false_easting_m + radius * y,
            self.false_northing_m + radius * x,
        )
    }

    /// Convert projected `(easting_m, northing_m)` to WGS84 `(lat_deg, lon_deg)`,
    /// unwound onto the continuous longitude branch around `central_meridian_deg`.
    pub fn to_geodetic(&self, easting: f64, northing: f64) -> (f64, f64) {
        let ellipsoid = self.datum.ellipsoid;
        let radius = self.scale * rectifying_radius(ellipsoid);
        let xi = (northing - self.false_northing_m) / radius;
        let eta = (easting - self.false_easting_m) / radius;

        let (mut x, mut y) = (xi, eta);
        for (j, b) in krueger_beta(ellipsoid).iter().enumerate() {
            let k = 2.0 * (j as f64 + 1.0);
            x -= b * (k * xi).sin() * (k * eta).cosh();
            y -= b * (k * xi).cos() * (k * eta).sinh();
        }

        let slope = y.sinh();
        let tau_c = x.sin() / slope.hypot(x.cos());
        let lam = slope.atan2(x.cos());

        let e = ellipsoid.eccentricity_sq().sqrt();
        let phi = geodetic_from_conformal(e, tau_c, CONFORMAL_ROUNDS);

        let (lat_deg, lon_deg) = self.datum.to_wgs84(
            phi.to_degrees(),
            self.central_meridian_deg + lam.to_degrees(),
        );
        (lat_deg, unwind(self.central_meridian_deg, lon_deg))
    }
}

#[cfg(test)]
#[path = "projection_tests.rs"]
mod tests;
