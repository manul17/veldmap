//! Projected coordinate systems in metres: transverse cylindrical projections
//! (Krueger series to 4th order in third flattening) with 7-parameter Helmert
//! datum shifts, plus Web Mercator (EPSG:3857).
//!
//! Only WGS84 geodetic coordinates cross the boundary of this module.

use super::geodesy::{FLATTENING, SEMI_MAJOR_M, unwind};

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

/// Krasovsky 1940 ellipsoid (SK-42, Gauss-Kruger zones, MSK local systems).
pub const KRASOVSKY: Ellipsoid = Ellipsoid {
    semi_major_m: 6_378_245.0,
    flattening: 1.0 / 298.3,
};

impl Ellipsoid {
    fn third_flattening(&self) -> f64 {
        self.flattening / (2.0 - self.flattening)
    }

    fn eccentricity_sq(&self) -> f64 {
        self.flattening * (2.0 - self.flattening)
    }

    fn rectifying_radius(&self) -> f64 {
        let n = self.third_flattening();
        self.semi_major_m / (1.0 + n) * (1.0 + n * n / 4.0 + n * n * n * n / 64.0)
    }

    /// Forward Krueger series coefficients up to `n^4`.
    fn alpha(&self) -> [f64; 4] {
        let n = self.third_flattening();
        [
            n / 2.0 - 2.0 * n * n / 3.0 + 5.0 * n * n * n / 16.0 + 41.0 * n * n * n * n / 180.0,
            13.0 * n * n / 48.0 - 3.0 * n * n * n / 5.0 + 557.0 * n * n * n * n / 1440.0,
            61.0 * n * n * n / 240.0 - 103.0 * n * n * n * n / 140.0,
            49_561.0 * n * n * n * n / 161_280.0,
        ]
    }

    /// Inverse Krueger series coefficients up to `n^4`.
    fn beta(&self) -> [f64; 4] {
        let n = self.third_flattening();
        [
            n / 2.0 - 2.0 * n * n / 3.0 + 37.0 * n * n * n / 96.0 - n * n * n * n / 360.0,
            n * n / 48.0 + n * n * n / 15.0 - 437.0 * n * n * n * n / 1440.0,
            17.0 * n * n * n / 480.0 - 37.0 * n * n * n * n / 840.0,
            4_397.0 * n * n * n * n / 161_280.0,
        ]
    }

    fn to_earth_centred(self, lat_deg: f64, lon_deg: f64) -> [f64; 3] {
        let (lat, lon) = (lat_deg.to_radians(), lon_deg.to_radians());
        let e2 = self.eccentricity_sq();
        let curvature = self.semi_major_m / (1.0 - e2 * lat.sin() * lat.sin()).sqrt();
        [
            curvature * lat.cos() * lon.cos(),
            curvature * lat.cos() * lon.sin(),
            curvature * (1.0 - e2) * lat.sin(),
        ]
    }

    fn earth_centred_to_geodetic(self, point: [f64; 3]) -> (f64, f64) {
        let [x, y, z] = point;
        let flat = x.hypot(y);
        if flat == 0.0 {
            return (90.0_f64.copysign(z), 0.0);
        }
        let lat = self.latitude_at(flat, z, CENTRED_ROUNDS);
        (lat.to_degrees(), y.atan2(x).to_degrees())
    }

    fn latitude_at(&self, flat: f64, z: f64, rounds: u32) -> f64 {
        let e2 = self.eccentricity_sq();
        let mut lat = z.atan2(flat * (1.0 - e2));
        for _ in 0..rounds {
            let sin = lat.sin();
            let curvature = self.semi_major_m / (1.0 - e2 * sin * sin).sqrt();
            let height = flat / lat.cos() - curvature;
            lat = z.atan2(flat * (1.0 - e2 * curvature / (curvature + height)));
        }
        lat
    }
}

const CENTRED_ROUNDS: u32 = 6;

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

    fn helmert(&self, point: [f64; 3]) -> [f64; 3] {
        let ([wx, wy, wz], scale) = self.turn();
        let [x, y, z] = point;
        let [dx, dy, dz] = self.shift_m;
        [
            dx + scale * (x + wz * y - wy * z),
            dy + scale * (-wz * x + y + wx * z),
            dz + scale * (wy * x - wx * y + z),
        ]
    }

    /// Exact closed-form inverse of `I + W`: `(I - W + w w^T) / (1 + |w|^2)`.
    fn helmert_back(&self, point: [f64; 3]) -> [f64; 3] {
        let ([wx, wy, wz], scale) = self.turn();
        let [x, y, z] = [0, 1, 2].map(|axis| (point[axis] - self.shift_m[axis]) / scale);

        let square = wx * wx + wy * wy + wz * wz;
        let along = wx * x + wy * y + wz * z;
        [
            (x - (wz * y - wy * z) + wx * along) / (1.0 + square),
            (y - (-wz * x + wx * z) + wy * along) / (1.0 + square),
            (z - (wy * x - wx * y) + wz * along) / (1.0 + square),
        ]
    }

    /// Convert `(lat_deg, lon_deg)` from this datum into WGS84.
    pub fn to_wgs84(&self, lat_deg: f64, lon_deg: f64) -> (f64, f64) {
        if *self == WGS84_DATUM {
            return (lat_deg, lon_deg);
        }
        let point = self.ellipsoid.to_earth_centred(lat_deg, lon_deg);
        WGS84.earth_centred_to_geodetic(self.helmert(point))
    }

    /// Convert WGS84 `(lat_deg, lon_deg)` into this datum.
    pub fn from_wgs84(&self, lat_deg: f64, lon_deg: f64) -> (f64, f64) {
        if *self == WGS84_DATUM {
            return (lat_deg, lon_deg);
        }
        let point = WGS84.to_earth_centred(lat_deg, lon_deg);
        self.ellipsoid
            .earth_centred_to_geodetic(self.helmert_back(point))
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
        for (j, a) in ellipsoid.alpha().iter().enumerate() {
            let k = 2.0 * (j as f64 + 1.0);
            x += a * (k * xi).sin() * (k * eta).cosh();
            y += a * (k * xi).cos() * (k * eta).sinh();
        }

        let radius = self.scale * ellipsoid.rectifying_radius();
        (
            self.false_easting_m + radius * y,
            self.false_northing_m + radius * x,
        )
    }

    /// Convert projected `(easting_m, northing_m)` to WGS84 `(lat_deg, lon_deg)`,
    /// unwound onto the continuous longitude branch around `central_meridian_deg`.
    pub fn to_geodetic(&self, easting: f64, northing: f64) -> (f64, f64) {
        let ellipsoid = self.datum.ellipsoid;
        let radius = self.scale * ellipsoid.rectifying_radius();
        let xi = (northing - self.false_northing_m) / radius;
        let eta = (easting - self.false_easting_m) / radius;

        let (mut x, mut y) = (xi, eta);
        for (j, b) in ellipsoid.beta().iter().enumerate() {
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
mod tests {
    use super::*;

    #[test]
    fn web_mercator_world_is_square() {
        let map = Projection::WebMercator;
        let side = 20_037_508.34;

        let (x, y) = map.from_geodetic(0.0, 0.0);
        assert!(x.abs() < 1e-6 && y.abs() < 1e-6);

        let (x, _) = map.from_geodetic(0.0, 180.0);
        assert!((x - side).abs() < 0.5, "longitude edge: {x}");

        let (_, y) = map.from_geodetic(WEB_MERCATOR_LIMIT_DEG, 0.0);
        assert!((y - side).abs() < 0.5, "latitude edge: {y}");
    }

    #[test]
    fn web_mercator_reference_places_match_tables() {
        let map = Projection::WebMercator;
        for (lat, lon, ex, ey) in [
            (47.2357, 39.7015, 4_419_550.76, 5_980_631.53),
            (-33.8688, 151.2093, 16_832_542.28, -4_011_198.65),
        ] {
            let (x, y) = map.from_geodetic(lat, lon);
            assert!(
                (x - ex).abs() < 0.5 && (y - ey).abs() < 0.5,
                "{lat} {lon} -> {x} {y}"
            );
        }
    }

    #[test]
    fn web_mercator_roundtrip() {
        let map = Projection::WebMercator;
        for lat in [-84.0, -45.0, -0.5, 0.0, 12.25, 60.0, 84.9] {
            for lon in [-179.5, -90.0, 0.0, 33.3, 179.5] {
                let (x, y) = map.from_geodetic(lat, lon);
                let (back_lat, back_lon) = map.to_geodetic(x, y);
                assert!(
                    (back_lat - lat).abs() < 1e-9 && (back_lon - lon).abs() < 1e-9,
                    "{lat} {lon} -> {x} {y} -> {back_lat} {back_lon}"
                );
            }
        }
    }

    #[test]
    fn web_mercator_clamps_beyond_domain() {
        let map = Projection::WebMercator;
        let (_, edge) = map.from_geodetic(WEB_MERCATOR_LIMIT_DEG, 0.0);
        let (_, beyond) = map.from_geodetic(89.9, 0.0);
        assert_eq!(edge, beyond);
        assert!(beyond.is_finite());
    }

    #[test]
    fn web_mercator_recognized_by_epsg() {
        assert_eq!(Projection::from_epsg(3857), Some(Projection::WebMercator));
        assert!(matches!(
            Projection::from_epsg(32637),
            Some(Projection::Transverse(_))
        ));
        assert_eq!(Projection::from_epsg(3035), None);
    }

    fn ground_m(first: (f64, f64), second: (f64, f64)) -> f64 {
        let lat = ((first.0 + second.0) / 2.0).to_radians();
        let per_lat = 111_132.92 - 559.82 * (2.0 * lat).cos();
        let per_lon = 111_412.84 * lat.cos() - 93.5 * (3.0 * lat).cos();
        ((first.0 - second.0) * per_lat).hypot((first.1 - second.1) * per_lon)
    }

    #[test]
    fn the_local_grid_matches_an_outside_converter() {
        let checks = [
            (1, 500_000.0, 1_330_000.0, 47.932_102_711, 38.383_287_711),
            (2, 500_000.0, 2_330_000.0, 47.932_149_061, 41.383_334_064),
            (3, 500_000.0, 3_330_000.0, 47.932_194_230, 44.383_385_063),
            (2, 500_000.0, 2_350_000.0, 47.930_905_649, 41.650_973_905),
        ];
        for (zone, north, east, lat, lon) in checks {
            let system = System::msk61(zone).expect("MSK-61 zone");
            let got = system.to_geodetic(east, north);
            let off = ground_m(got, (lat, lon));
            assert!(
                off < 0.002,
                "zone {zone}: {got:?} vs ({lat}, {lon}) — {off} m"
            );
        }
    }

    #[test]
    fn the_datum_shift_is_worth_a_hundred_metres() {
        let rostov = (47.23, 39.72);
        let moved = PULKOVO_1942.to_wgs84(rostov.0, rostov.1);
        let shift = ground_m(rostov, moved);
        assert!((100.0..130.0).contains(&shift), "{shift} m");
        assert_eq!(WGS84_DATUM.to_wgs84(rostov.0, rostov.1), rostov);
    }

    #[test]
    fn the_helmert_and_its_inverse_are_exact() {
        let points = [
            [3_000_000.0, 2_000_000.0, 5_200_000.0],
            [-1_500_000.0, 5_800_000.0, 2_000_000.0],
            [4_000_000.0, -3_000_000.0, -3_500_000.0],
        ];
        let turned = Datum {
            rotate_sec: [-1.5, -0.35, -0.79],
            ..PULKOVO_1942
        };
        for datum in [PULKOVO_1942, turned] {
            for point in points {
                let back = datum.helmert_back(datum.helmert(point));
                for axis in 0..3 {
                    let off = (back[axis] - point[axis]).abs();
                    assert!(off < 1e-6, "{:?}, axis {axis}: {off} m", datum.rotate_sec);
                }
            }
        }
    }

    #[test]
    fn the_datum_shift_and_its_reverse_converge_where_it_is_fitted() {
        for (lat, lon) in [
            (47.23, 39.72),
            (55.75, 37.62),
            (66.53, 66.60),
            (43.1, 131.9),
        ] {
            let (there_lat, there_lon) = PULKOVO_1942.to_wgs84(lat, lon);
            let back = PULKOVO_1942.from_wgs84(there_lat, there_lon);
            assert!(ground_m(back, (lat, lon)) < 1e-3);
        }
    }

    #[test]
    fn epsg_names_a_system_only_inside_its_bands() {
        assert_eq!(System::from_epsg(32601), Some(System::utm(1, false)));
        assert_eq!(System::from_epsg(32660), Some(System::utm(60, false)));
        assert_eq!(System::from_epsg(32701), Some(System::utm(1, true)));
        assert_eq!(System::from_epsg(32760), Some(System::utm(60, true)));
        assert_eq!(System::from_epsg(28401), Some(System::gauss_kruger(1)));
        assert_eq!(System::from_epsg(28408), Some(System::gauss_kruger(8)));
        assert_eq!(System::from_epsg(28432), Some(System::gauss_kruger(32)));
        assert_eq!(System::from_epsg(6_336_101), System::msk61(1));
        assert_eq!(System::from_epsg(6_336_102), System::msk61(2));
        assert_eq!(System::from_epsg(6_336_103), System::msk61(3));

        assert_eq!(System::from_epsg(32600), None);
        assert_eq!(System::from_epsg(32700), None);
        assert_eq!(System::from_epsg(32661), None);
        assert_eq!(System::from_epsg(32761), None);
        assert_eq!(System::from_epsg(28400), None);
        assert_eq!(System::from_epsg(28433), None);
        assert_eq!(System::from_epsg(6_336_100), None);
        assert_eq!(System::from_epsg(6_336_104), None);
        assert_eq!(System::from_epsg(4326), None);
        assert_eq!(System::from_epsg(3031), None);
    }

    #[test]
    fn each_family_keeps_its_own_numbers() {
        let utm = System::from_epsg(32638).expect("UTM zone 38N");
        assert!((utm.central_meridian_deg - 45.0).abs() < 1e-12);
        assert_eq!(utm.scale, 0.9996);
        assert_eq!(utm.datum, WGS84_DATUM);

        let gk = System::gauss_kruger(8);
        assert!((gk.central_meridian_deg - 45.0).abs() < 1e-12);
        assert_eq!(gk.scale, 1.0);
        assert_eq!(gk.false_easting_m, 8_500_000.0);
        assert_eq!(gk.datum.ellipsoid, KRASOVSKY);

        let msk = System::msk61(2).expect("MSK-61 zone 2");
        assert!((msk.central_meridian_deg - (40.0 + 59.0 / 60.0)).abs() < 1e-10);
        assert_eq!(msk.false_easting_m, 2_300_000.0);
        assert_eq!(msk.false_northing_m, -4_811_057.628);
    }

    #[test]
    fn roundtrip_converges_across_zone() {
        let cases = [
            (System::utm(40, false), [-79.0, -8.0, 0.0, 45.0, 83.5], 1e-4),
            (System::utm(40, true), [-79.0, -8.0, 0.0, 45.0, 83.5], 1e-4),
            (
                System::gauss_kruger(8),
                [42.0, 50.0, 55.0, 65.0, 75.0],
                1e-3,
            ),
            (
                System::msk61(2).expect("MSK-61 zone 2"),
                [46.0, 47.0, 48.0, 49.0, 50.0],
                1e-3,
            ),
        ];
        for (system, latitudes, tolerance_m) in cases {
            let middle = system.central_meridian_deg;
            for lat in latitudes {
                for offset in [-2.5, 0.0, 1.7, 2.9] {
                    let lon = middle + offset;
                    let (e, n) = system.from_geodetic(lat, lon);
                    let back = system.to_geodetic(e, n);
                    let off = ground_m(back, (lat, lon));
                    assert!(off < tolerance_m, "{system:?}: {back:?} — {off} m");
                }
            }
        }
    }

    #[test]
    fn a_longitude_across_the_seam_lands_in_its_zone() {
        let zone = System::utm(1, false);
        assert!((zone.central_meridian_deg + 177.0).abs() < 1e-12);

        let (east, north) = zone.from_geodetic(10.0, 179.0);
        assert!((0.0..500_000.0).contains(&east));
        assert!((1_000_000.0..1_200_000.0).contains(&north));

        let (lat, lon) = zone.to_geodetic(east, north);
        assert!((lat - 10.0).abs() < 1e-9);
        assert!((lon + 181.0).abs() < 1e-9);
    }

    #[test]
    fn a_system_beyond_the_seam_keeps_its_branch() {
        let far = System::gauss_kruger(32);
        assert!((far.central_meridian_deg - 189.0).abs() < 1e-12);

        let (lat, lon) = far.to_geodetic(far.false_easting_m, 5_000_000.0);
        assert!(lon > 180.0);
        assert!((lon - 189.0).abs() < 0.01);
        assert!((44.0..46.0).contains(&lat));
    }

    #[test]
    fn a_shifted_datum_on_the_same_ellipsoid_still_moves() {
        let nudged = Datum {
            shift_m: [100.0, 0.0, 0.0],
            ..WGS84_DATUM
        };
        let here = (0.0, 90.0);
        let moved = nudged.to_wgs84(here.0, here.1);
        assert!((ground_m(moved, here) - 100.0).abs() < 0.5);

        let back = nudged.from_wgs84(moved.0, moved.1);
        assert!(ground_m(back, here) < 1e-3);
    }

    #[test]
    fn the_axis_gives_a_pole_and_not_a_nan() {
        let polar = KRASOVSKY.semi_major_m * (1.0 - KRASOVSKY.flattening);
        let (north, lon) = KRASOVSKY.earth_centred_to_geodetic([0.0, 0.0, polar]);
        assert!((north - 90.0).abs() < 1e-9);
        assert_eq!(lon, 0.0);

        let (south, _) = KRASOVSKY.earth_centred_to_geodetic([0.0, 0.0, -polar]);
        assert!((south + 90.0).abs() < 1e-9);
    }

    #[test]
    fn axis_anchors_are_exact() {
        let north = System::utm(31, false);
        let (e, n) = north.from_geodetic(0.0, 3.0);
        assert!((e - 500_000.0).abs() < 1e-6 && n.abs() < 1e-6);

        let south = System::utm(31, true);
        let (_, n) = south.from_geodetic(0.0, 3.0);
        assert!((n - 10_000_000.0).abs() < 1e-6);
    }

    #[test]
    fn hemispheres_mirror() {
        let zone = System::utm(40, false);
        let (e1, n1) = zone.from_geodetic(30.0, 58.0);
        let (e2, n2) = zone.from_geodetic(-30.0, 58.0);
        assert!((e1 - e2).abs() < 1e-6);
        assert!((n1 + n2).abs() < 1e-6);
    }

    #[test]
    fn the_conformal_turn_has_nothing_left_to_improve() {
        let ellipsoid = WGS84;
        let e = ellipsoid.eccentricity_sq().sqrt();
        let radius = ellipsoid.semi_major_m;

        let (mut settled, mut short) = (0.0_f64, 0.0_f64);
        for step in 0..=1680 {
            let lat = f64::from(step) * 0.05;
            let tau_c = {
                let phi = lat.to_radians();
                (phi.tan().asinh() - e * (e * phi.sin()).atanh()).sinh()
            };
            let whole = geodetic_from_conformal(e, tau_c, CONFORMAL_ROUNDS + 4);
            settled = settled
                .max((geodetic_from_conformal(e, tau_c, CONFORMAL_ROUNDS) - whole).abs() * radius);
            short = short.max((geodetic_from_conformal(e, tau_c, 4) - whole).abs() * radius);
        }
        assert!(settled < 1e-8, "insufficient rounds: {settled} m");
        assert!(short > 1e-6, "4 rounds already settled to {short} m");
    }

    #[test]
    fn the_centred_turn_works_only_above_the_ellipsoid() {
        let ellipsoid = WGS84;
        let lifted = |lat_deg: f64, height_m: f64| {
            let lat = lat_deg.to_radians();
            let e2 = ellipsoid.eccentricity_sq();
            let curvature = ellipsoid.semi_major_m / (1.0 - e2 * lat.sin() * lat.sin()).sqrt();
            (
                (curvature + height_m) * lat.cos(),
                (curvature * (1.0 - e2) + height_m) * lat.sin(),
            )
        };

        for step in 0..=90 {
            let (flat, z) = lifted(f64::from(step), 0.0);
            assert_eq!(
                ellipsoid.latitude_at(flat, z, 0),
                ellipsoid.latitude_at(flat, z, CENTRED_ROUNDS),
            );
        }

        let (mut settled, mut short) = (0.0_f64, 0.0_f64);
        for step in 0..=1800 {
            let (flat, z) = lifted(f64::from(step) * 0.05, 271.0);
            let whole = ellipsoid.latitude_at(flat, z, CENTRED_ROUNDS + 6);
            let radius = ellipsoid.semi_major_m;
            settled = settled
                .max((ellipsoid.latitude_at(flat, z, CENTRED_ROUNDS) - whole).abs() * radius);
            short = short.max((ellipsoid.latitude_at(flat, z, 2) - whole).abs() * radius);
        }
        assert!(settled < 1e-8, "insufficient rounds: {settled} m");
        assert!(short > 1e-7, "2 rounds already settled to {short} m");
    }

    #[test]
    fn sentinel_tile_corner_lands_where_the_tile_is() {
        let zone = System::utm(40, false);
        let (lat, lon) = zone.to_geodetic(600_000.0, 7_800_000.0);
        assert!((69.9..70.5).contains(&lat));
        assert!((59.0..60.3).contains(&lon));

        for step in 0..9 {
            let northing = 500_000.0 + f64::from(step) * 1_000_000.0;
            let (lat, lon) = zone.to_geodetic(600_000.0, northing);
            let (e, n) = zone.from_geodetic(lat, lon);
            assert!((e - 600_000.0).abs() < 1e-6);
            assert!((n - northing).abs() < 1e-6);
        }
    }
}
