//! Projected coordinate systems: fast exact in-house transverse cylindrical
//! projections (Krueger series to 4th order in third flattening with explicit
//! 7-parameter Helmert conventions, arbitrary origin latitude, and linear
//! units), Web Mercator (EPSG:3857), and general pure-Rust `proj4rs` /
//! `proj4wkt` / `crs-definitions` projections.
//!
//! Only WGS84 geodetic coordinates cross the boundary of this module.

use std::sync::Arc;

use super::super::geodesy::{Ellipsoid, SEMI_MAJOR_M, unwind};
use super::datum::{Datum, DatumWarning, PULKOVO_1942, WGS84_DATUM};
use super::proj4::Proj4Projection;

/// Base EPSG code for WGS 84 / UTM North zones (`UTM_NORTH_EPSG_BASE + zone`).
pub(super) const UTM_NORTH_EPSG_BASE: u32 = 32_600;

/// Base EPSG code for WGS 84 / UTM South zones (`UTM_SOUTH_EPSG_BASE + zone`).
pub(super) const UTM_SOUTH_EPSG_BASE: u32 = 32_700;

/// Valid UTM zone numbers (`1..=60`).
pub(super) const UTM_ZONES: std::ops::RangeInclusive<u32> = 1..=60;

/// Base EPSG code for Pulkovo 1942 / Gauss-Kruger zones (`GK_PULKOVO_EPSG_BASE + zone`).
pub(super) const GK_PULKOVO_EPSG_BASE: u32 = 28_400;

/// Valid Pulkovo 1942 / Gauss-Kruger EPSG zone numbers (`2..=32`).
pub(super) const GK_PULKOVO_ZONES: std::ops::RangeInclusive<u32> = 2..=32;

/// Base regional pseudo-EPSG code for MSK-61 zones (`MSK61_EPSG_BASE + zone`).
pub(super) const MSK61_EPSG_BASE: u32 = 6_336_100;

/// Valid MSK-61 3-degree zone numbers (`1..=3`).
pub(super) const MSK61_ZONES: std::ops::RangeInclusive<u32> = 1..=3;

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

/// Dimensionless rectifying meridian arc (`M(lat) / rectifying_radius`) from
/// the equator to `lat_deg` along the central meridian.
fn meridian_arc_rad(ellipsoid: Ellipsoid, lat_deg: f64) -> f64 {
    if lat_deg == 0.0 {
        return 0.0;
    }
    let phi = lat_deg.to_radians();
    let e = ellipsoid.eccentricity_sq().sqrt();
    let tau = phi.tan();
    let sigma = (e * (e * tau / tau.hypot(1.0)).atanh()).sinh();
    let tau_c = tau * sigma.hypot(1.0) - sigma * tau.hypot(1.0);
    let xi0 = tau_c.atan();
    let mut arc = xi0;
    for (j, a) in krueger_alpha(ellipsoid).iter().enumerate() {
        let k = 2.0 * (j as f64 + 1.0);
        arc += a * (k * xi0).sin();
    }
    arc
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

/// Supported projected coordinate systems.
#[derive(Clone, PartialEq, Debug)]
pub enum Projection {
    /// Fast exact in-house Transverse Mercator family (UTM, Gauss-Kruger,
    /// MSK-xx, and custom user-defined TM systems) with an optional display name.
    Transverse {
        system: System,
        name: Option<Arc<str>>,
    },
    /// Web Mercator (EPSG:3857).
    WebMercator,
    /// General pure-Rust `proj4rs` projection (EPSG catalog via
    /// `crs-definitions`, `.prj` WKT1/WKT2 via `proj4wkt`, or custom PROJ.4).
    Proj4(Arc<Proj4Projection>),
}

impl From<System> for Projection {
    fn from(system: System) -> Self {
        Self::Transverse { system, name: None }
    }
}

const WEB_MERCATOR_RADIUS_M: f64 = SEMI_MAJOR_M;

/// Maximum latitude of the square Web Mercator domain (`atan(sinh(pi))`).
pub const WEB_MERCATOR_LIMIT_DEG: f64 = 85.051_128_779_806_59;

/// Convert normalized Web Mercator Y radians (`[-pi, +pi]`) to geodetic
/// latitude in degrees.
#[inline]
pub fn web_mercator_lat_deg(merc_rad: f64) -> f64 {
    merc_rad.sinh().atan().to_degrees()
}

impl Projection {
    /// Construct a named in-house Transverse Mercator projection.
    pub fn transverse_named(system: System, name: impl Into<Arc<str>>) -> Self {
        let name_arc: Arc<str> = name.into();
        let trimmed_len = name_arc.trim().len();
        let name = if trimmed_len == 0 {
            None
        } else if trimmed_len == name_arc.len() {
            Some(name_arc)
        } else {
            Some(Arc::<str>::from(name_arc.trim()))
        };
        Self::Transverse { system, name }
    }

    /// Resolve an EPSG (or MSK-61 `633610x`) code via fast in-house formulas
    /// only, without falling back to `proj4rs`.
    pub(super) fn from_epsg_in_house(code: u32) -> Option<Self> {
        if code == 3857 {
            return Some(Self::WebMercator);
        }
        System::from_epsg(code).map(|system| Self::Transverse { system, name: None })
    }

    /// Look up a projected CRS by EPSG (or regional MSK-61 `633610x`) code.
    ///
    /// Retains fast exact in-house implementations for EPSG:3857, WGS84 UTM
    /// (`32601..=32660`, `32701..=32760`), Pulkovo 1942 Gauss-Kruger
    /// (`28402..=28432`), and MSK-61 (`6336101..=6336103`), falling back to
    /// pure-Rust `proj4rs` + `crs-definitions` for all other projected EPSG
    /// codes.
    pub fn from_epsg(code: u32) -> Option<Self> {
        if let Some(in_house) = Self::from_epsg_in_house(code) {
            return Some(in_house);
        }
        let proj4 = Proj4Projection::from_epsg(code).ok()?;
        (!proj4.is_geographic()).then(|| Self::Proj4(Arc::new(proj4)))
    }

    /// Convert WGS84 `(lat_deg, lon_deg)` to projected `(easting, northing)`
    /// in the projection's native linear units.
    pub fn from_geodetic(&self, lat_deg: f64, lon_deg: f64) -> (f64, f64) {
        match self {
            Self::Transverse { system, .. } => system.from_geodetic(lat_deg, lon_deg),
            Self::WebMercator => web_mercator_xy_unchecked(
                lat_deg.clamp(-WEB_MERCATOR_LIMIT_DEG, WEB_MERCATOR_LIMIT_DEG),
                lon_deg,
            ),
            Self::Proj4(proj) => proj.transform_from_wgs84_geodetic(lat_deg, lon_deg),
        }
    }

    /// Strict forward projection returning `None` when inputs or projected
    /// outputs are non-finite, or when a Web Mercator latitude hits `±90°`
    /// (never clamping polar latitudes onto `±WEB_MERCATOR_LIMIT_DEG`).
    pub(crate) fn project_geodetic_strict(&self, lat_deg: f64, lon_deg: f64) -> Option<(f64, f64)> {
        if !(lat_deg.is_finite() && lon_deg.is_finite()) {
            return None;
        }
        let (x, y) = match self {
            Self::WebMercator => web_mercator_from_geodetic_strict(lat_deg, lon_deg)?,
            Self::Transverse { system, .. } => system.from_geodetic(lat_deg, lon_deg),
            Self::Proj4(proj) => proj.transform_from_wgs84_geodetic(lat_deg, lon_deg),
        };
        (x.is_finite() && y.is_finite()).then_some((x, y))
    }

    /// Convert projected `(easting, northing)` in the projection's native
    /// linear units to WGS84 `(lat_deg, lon_deg)`.
    pub fn to_geodetic(&self, easting: f64, northing: f64) -> (f64, f64) {
        match self {
            Self::Transverse { system, .. } => system.to_geodetic(easting, northing),
            Self::WebMercator => {
                let lon = (easting / WEB_MERCATOR_RADIUS_M).to_degrees();
                let lat = web_mercator_lat_deg(northing / WEB_MERCATOR_RADIUS_M);
                (lat, lon)
            }
            Self::Proj4(proj) => proj.to_wgs84_geodetic(easting, northing),
        }
    }

    /// Scale factor from this projection's linear units to metres.
    pub fn unit_to_metres(&self) -> f64 {
        match self {
            Self::Transverse { system, .. } => system.safe_unit_to_metres(),
            Self::WebMercator => 1.0,
            Self::Proj4(proj) => proj.unit_to_metres(),
        }
    }

    /// Optional datum warning attached to this projection.
    pub fn datum_warning(&self) -> Option<&DatumWarning> {
        match self {
            Self::Proj4(proj) => proj.datum_warning(),
            Self::Transverse { .. } | Self::WebMercator => None,
        }
    }

    /// Concise human-readable CRS label.
    pub fn label(&self) -> String {
        match self {
            Self::WebMercator => "EPSG:3857 (Web Mercator)".to_string(),
            Self::Transverse {
                name: Some(name), ..
            } => name.to_string(),
            Self::Transverse { system, name: None } => system.label(),
            Self::Proj4(proj) => proj.label().to_string(),
        }
    }
}

#[inline]
fn web_mercator_xy_unchecked(lat_deg: f64, lon_deg: f64) -> (f64, f64) {
    let phi = lat_deg.to_radians();
    (
        WEB_MERCATOR_RADIUS_M * lon_deg.to_radians(),
        WEB_MERCATOR_RADIUS_M * (std::f64::consts::FRAC_PI_4 + phi / 2.0).tan().ln(),
    )
}

/// Unclamped EPSG:3857 forward projection for exact raster inverse hits.
///
/// Unlike [`Projection::from_geodetic`], this never clamps polar latitudes onto
/// `±WEB_MERCATOR_LIMIT_DEG`, preventing phantom cursor hits over polar caps
/// while remaining an exact inverse of [`Projection::to_geodetic`] across
/// `(-90°, +90°)`.
fn web_mercator_from_geodetic_strict(lat_deg: f64, lon_deg: f64) -> Option<(f64, f64)> {
    if !(lat_deg.is_finite() && lon_deg.is_finite()) || lat_deg.abs() >= 90.0 {
        return None;
    }
    let (x, y) = web_mercator_xy_unchecked(lat_deg, lon_deg);
    (x.is_finite() && y.is_finite()).then_some((x, y))
}

/// Transverse cylindrical projection parameters.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct System {
    pub datum: Datum,
    pub central_meridian_deg: f64,
    /// Latitude of natural origin in degrees (`ProjNatOriginLatGeoKey 3081`, default `0.0`).
    pub origin_lat_deg: f64,
    pub scale: f64,
    /// False easting in metres.
    pub false_easting_m: f64,
    /// False northing in metres.
    pub false_northing_m: f64,
    /// Metres per projected linear coordinate unit (`1.0` for metres, `0.3048`
    /// for International Feet, `1200/3937` for US Survey Feet).
    pub unit_to_metres: f64,
}

const UTM_FALSE_EASTING: f64 = 500_000.0;
const UTM_FALSE_NORTHING_SOUTH: f64 = 10_000_000.0;

impl System {
    /// WGS84 UTM zone (`number` in `1..=60`).
    pub fn utm(number: u32, south: bool) -> Self {
        Self {
            datum: WGS84_DATUM,
            central_meridian_deg: f64::from(number) * 6.0 - 183.0,
            origin_lat_deg: 0.0,
            scale: 0.9996,
            false_easting_m: UTM_FALSE_EASTING,
            false_northing_m: if south { UTM_FALSE_NORTHING_SOUTH } else { 0.0 },
            unit_to_metres: 1.0,
        }
    }

    /// Six-degree Gauss-Kruger zone on Pulkovo 1942 (`zone` in `1..=32`).
    pub fn gauss_kruger(zone: u32) -> Self {
        Self {
            datum: PULKOVO_1942,
            central_meridian_deg: f64::from(zone) * 6.0 - 3.0,
            origin_lat_deg: 0.0,
            scale: 1.0,
            false_easting_m: f64::from(zone) * 1_000_000.0 + 500_000.0,
            false_northing_m: 0.0,
            unit_to_metres: 1.0,
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
            origin_lat_deg: 0.0,
            scale: 1.0,
            false_easting_m: f64::from(zone) * 1_000_000.0 + 300_000.0,
            false_northing_m: -4_811_057.628,
            unit_to_metres: 1.0,
        })
    }

    /// Look up a transverse projection system by EPSG (or MSK-61 `633610x`) code.
    pub fn from_epsg(code: u32) -> Option<Self> {
        if let Some(zone) = code
            .checked_sub(UTM_NORTH_EPSG_BASE)
            .filter(|z| UTM_ZONES.contains(z))
        {
            return Some(Self::utm(zone, false));
        }
        if let Some(zone) = code
            .checked_sub(UTM_SOUTH_EPSG_BASE)
            .filter(|z| UTM_ZONES.contains(z))
        {
            return Some(Self::utm(zone, true));
        }
        if let Some(zone) = code
            .checked_sub(GK_PULKOVO_EPSG_BASE)
            .filter(|z| GK_PULKOVO_ZONES.contains(z))
        {
            return Some(Self::gauss_kruger(zone));
        }
        if let Some(zone) = code
            .checked_sub(MSK61_EPSG_BASE)
            .filter(|z| MSK61_ZONES.contains(z))
        {
            return Self::msk61(zone);
        }
        None
    }

    fn safe_unit_to_metres(&self) -> f64 {
        if self.unit_to_metres.is_finite() && self.unit_to_metres > 0.0 {
            self.unit_to_metres
        } else {
            1.0
        }
    }

    /// Human-readable description of this Transverse Mercator system.
    pub fn label(&self) -> String {
        if self.datum.is_wgs84()
            && self.origin_lat_deg == 0.0
            && (self.scale - 0.9996).abs() < 1e-12
            && (self.false_easting_m - UTM_FALSE_EASTING).abs() < 1e-6
            && (self.safe_unit_to_metres() - 1.0).abs() < 1e-12
        {
            let zone_f = (self.central_meridian_deg + 183.0) / 6.0;
            let zone = zone_f.round() as u32;
            if UTM_ZONES.contains(&zone) && (zone_f - f64::from(zone)).abs() < 1e-9 {
                if self.false_northing_m == 0.0 {
                    return format!("EPSG:{} (UTM {zone}N)", UTM_NORTH_EPSG_BASE + zone);
                }
                if (self.false_northing_m - UTM_FALSE_NORTHING_SOUTH).abs() < 1e-6 {
                    return format!("EPSG:{} (UTM {zone}S)", UTM_SOUTH_EPSG_BASE + zone);
                }
            }
        }
        if self.datum == PULKOVO_1942
            && self.origin_lat_deg == 0.0
            && (self.scale - 1.0).abs() < 1e-12
            && (self.safe_unit_to_metres() - 1.0).abs() < 1e-12
        {
            let zone_f = (self.central_meridian_deg + 3.0) / 6.0;
            let zone = zone_f.round() as u32;
            if GK_PULKOVO_ZONES.contains(&zone)
                && (zone_f - f64::from(zone)).abs() < 1e-9
                && (self.false_easting_m - (f64::from(zone) * 1_000_000.0 + 500_000.0)).abs() < 1e-3
                && self.false_northing_m.abs() < 1e-3
            {
                return format!(
                    "EPSG:{} (Pulkovo 1942 / GK zone {zone})",
                    GK_PULKOVO_EPSG_BASE + zone
                );
            }
            for zone in MSK61_ZONES {
                if let Some(msk) = Self::msk61(zone)
                    && (self.central_meridian_deg - msk.central_meridian_deg).abs() < 1e-8
                    && (self.false_easting_m - msk.false_easting_m).abs() < 1e-3
                    && (self.false_northing_m - msk.false_northing_m).abs() < 1e-3
                {
                    return format!("MSK-61 / Pulkovo 1942 zone {zone}");
                }
            }
        }
        format!(
            "Transverse Mercator (lon0={:.3} deg, lat0={:.3} deg, k={:.4})",
            self.central_meridian_deg, self.origin_lat_deg, self.scale
        )
    }

    /// Convert WGS84 `(lat_deg, lon_deg)` to projected `(easting, northing)`
    /// in this system's linear units (`unit_to_metres`).
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

        let m0 = meridian_arc_rad(ellipsoid, self.origin_lat_deg);
        let radius = self.scale * rectifying_radius(ellipsoid);
        let unit = self.safe_unit_to_metres();
        (
            (self.false_easting_m + radius * y) / unit,
            (self.false_northing_m + radius * (x - m0)) / unit,
        )
    }

    /// Convert projected `(easting, northing)` in this system's linear units
    /// to WGS84 `(lat_deg, lon_deg)`, unwound onto the continuous longitude
    /// branch around `central_meridian_deg`.
    pub fn to_geodetic(&self, easting: f64, northing: f64) -> (f64, f64) {
        let ellipsoid = self.datum.ellipsoid;
        let unit = self.safe_unit_to_metres();
        let easting_m = easting * unit;
        let northing_m = northing * unit;
        let radius = self.scale * rectifying_radius(ellipsoid);
        let m0 = meridian_arc_rad(ellipsoid, self.origin_lat_deg);
        let xi = (northing_m - self.false_northing_m) / radius + m0;
        let eta = (easting_m - self.false_easting_m) / radius;

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
