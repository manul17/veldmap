//! Geodetic datums, ellipsoids, linear units, and 7-parameter Helmert
//! transformations targeting WGS84 at the module boundary.
//!
//! Pure leaf module above [`super::geodesy`] with no projection or PROJ
//! dependencies so [`super::proj4`], [`super::projection`], and [`super::crs`]
//! form a strictly acyclic dependency graph.

use glam::DVec3;

use super::geodesy::{Ellipsoid, Geodetic, WGS84};

/// International foot in metres (EPSG:9002).
pub(crate) const INTERNATIONAL_FOOT_M: f64 = 0.3048;

/// US Survey foot in metres (EPSG:9003, `1200 / 3937`).
pub const US_SURVEY_FOOT_M: f64 = 1200.0 / 3937.0;

/// Krasovsky 1940 ellipsoid (SK-42, Gauss-Kruger zones, MSK local systems).
pub const KRASOVSKY: Ellipsoid = Ellipsoid {
    semi_major_m: 6_378_245.0,
    flattening: 1.0 / 298.3,
};

/// Airy 1830 ellipsoid (OSGB36 / EPSG:7001).
const AIRY_1830: Ellipsoid = Ellipsoid {
    semi_major_m: 6_377_563.396,
    flattening: 1.0 / 299.324_964_6,
};

/// Rotation sign convention for 7-parameter Helmert datum transformations.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum HelmertConvention {
    /// Coordinate Frame Rotation (EPSG:9607; used by GOST R 51794-2008 /
    /// Pulkovo 1942 and US/Australian surveying standards).
    ///
    /// Forward matrix: `[[1, +wz, -wy], [-wz, 1, +wx], [+wy, -wx, 1]]`.
    #[default]
    CoordinateFrame,
    /// Position Vector Transformation (EPSG:9606; used by PROJ `+towgs84` and
    /// WKT `TOWGS84` clauses).
    ///
    /// Forward matrix: `[[1, -rz, +ry], [+rz, 1, -rx], [-ry, +rx, 1]]`
    /// (negated rotation parameters relative to [`Self::CoordinateFrame`]).
    PositionVector,
}

/// Geodetic datum: reference ellipsoid plus 7-parameter Helmert shift to WGS84
/// with an explicit rotation sign [`HelmertConvention`].
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Datum {
    pub ellipsoid: Ellipsoid,
    /// Origin shift in metres.
    pub shift_m: [f64; 3],
    /// Axis rotations in arc-seconds under [`Self::convention`].
    pub rotate_sec: [f64; 3],
    /// Scale correction in parts per million.
    pub scale_ppm: f64,
    /// Sign convention for [`Self::rotate_sec`] (EPSG:9607 vs. EPSG:9606).
    pub convention: HelmertConvention,
}

/// Identity WGS84 datum.
pub const WGS84_DATUM: Datum = Datum {
    ellipsoid: WGS84,
    shift_m: [0.0; 3],
    rotate_sec: [0.0; 3],
    scale_ppm: 0.0,
    convention: HelmertConvention::CoordinateFrame,
};

/// Pulkovo 1942 (SK-42) to WGS84 parameters per GOST R 51794-2008
/// (Coordinate Frame Rotation, EPSG:9607; equivalent to PROJ Position Vector
/// `+towgs84=23.57,-140.95,-79.8,0,0.35,0.79,-0.22`).
pub const PULKOVO_1942: Datum = Datum {
    ellipsoid: KRASOVSKY,
    shift_m: [23.57, -140.95, -79.8],
    rotate_sec: [0.0, -0.35, -0.79],
    scale_ppm: -0.22,
    convention: HelmertConvention::CoordinateFrame,
};

impl Datum {
    /// Construct a datum from WKT / PROJ `+towgs84` parameters
    /// (`PositionVector`, EPSG:9606).
    pub const fn from_towgs84(ellipsoid: Ellipsoid, towgs84: [f64; 7]) -> Self {
        Self {
            ellipsoid,
            shift_m: [towgs84[0], towgs84[1], towgs84[2]],
            rotate_sec: [towgs84[3], towgs84[4], towgs84[5]],
            scale_ppm: towgs84[6],
            convention: HelmertConvention::PositionVector,
        }
    }

    /// Export this datum's 7-parameter Helmert transform in PROJ `+towgs84` /
    /// WKT `TOWGS84` convention (`PositionVector`, EPSG:9606).
    pub const fn to_towgs84(&self) -> [f64; 7] {
        let sign = match self.convention {
            HelmertConvention::CoordinateFrame => -1.0,
            HelmertConvention::PositionVector => 1.0,
        };
        [
            self.shift_m[0],
            self.shift_m[1],
            self.shift_m[2],
            self.rotate_sec[0] * sign + 0.0,
            self.rotate_sec[1] * sign + 0.0,
            self.rotate_sec[2] * sign + 0.0,
            self.scale_ppm,
        ]
    }

    /// Return whether this datum is mathematically identical to WGS84.
    pub fn is_wgs84(&self) -> bool {
        self.ellipsoid == WGS84
            && self.shift_m == [0.0; 3]
            && self.rotate_sec == [0.0; 3]
            && self.scale_ppm == 0.0
    }

    /// Return Coordinate-Frame rotation angles `(wx, wy, wz)` in radians and
    /// linear scale factor.
    fn turn(&self) -> ([f64; 3], f64) {
        let second = std::f64::consts::PI / (180.0 * 3600.0);
        let sign = match self.convention {
            HelmertConvention::CoordinateFrame => 1.0,
            HelmertConvention::PositionVector => -1.0,
        };
        (
            self.rotate_sec.map(|value| value * sign * second),
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
        if self.is_wgs84() {
            return (lat_deg, lon_deg);
        }
        let point = self.ellipsoid.to_ecef(Geodetic::surface(lat_deg, lon_deg));
        let shifted = WGS84.from_ecef(self.helmert(point));
        (shifted.lat_deg, shifted.lon_deg)
    }

    /// Convert 2D WGS84 `(lat_deg, lon_deg)` into this datum (discarding
    /// phantom Helmert `Delta h` on 2D output).
    pub fn from_wgs84(&self, lat_deg: f64, lon_deg: f64) -> (f64, f64) {
        if self.is_wgs84() {
            return (lat_deg, lon_deg);
        }
        let point = WGS84.to_ecef(Geodetic::surface(lat_deg, lon_deg));
        let shifted = self.ellipsoid.from_ecef(self.helmert_back(point));
        (shifted.lat_deg, shifted.lon_deg)
    }
}

/// Diagnostic warning attached to a CRS or raster binding when horizontal
/// datum conversion to WGS84 is approximate or requires an external shift grid.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DatumWarning {
    /// Foreign datum has no 7-parameter Helmert shift; coordinates are treated
    /// on WGS84 without a datum shift and may be offset by tens to hundreds of
    /// metres.
    UnshiftedForeignDatum { label: String },
    /// Datum is normally aligned to WGS84/ITRF via a shift grid (e.g. NADCON5,
    /// NTv2, OSTN15) that is not bundled; static Helmert or null shift may
    /// retain decimetre-to-metre residual offsets.
    GridShiftRecommended { label: String },
}

impl DatumWarning {
    /// Concise user-facing warning message for UI banners and metadata panels.
    pub fn message(&self) -> String {
        match self {
            Self::UnshiftedForeignDatum { label } => format!(
                "Foreign datum ({label}) has no bundled shift to WGS84; placement may be offset."
            ),
            Self::GridShiftRecommended { label } => format!(
                "Datum ({label}) uses a grid shift (e.g. NADCON/NTv2/OSTN15) that is not bundled; decimetre-to-metre offset possible."
            ),
        }
    }
}

/// Supported linear coordinate units for projected coordinate systems.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum LinearUnit {
    #[default]
    Metre,
    InternationalFoot,
    UsSurveyFoot,
}

impl LinearUnit {
    /// Conversion factor from this unit to metres.
    pub const fn to_metres(self) -> f64 {
        match self {
            Self::Metre => 1.0,
            Self::InternationalFoot => INTERNATIONAL_FOOT_M,
            Self::UsSurveyFoot => US_SURVEY_FOOT_M,
        }
    }

    /// Resolve a GeoTIFF `ProjLinearUnitsGeoKey` (3076) EPSG unit code.
    pub const fn from_geotiff_code(code: u16) -> Option<Self> {
        match code {
            9001 => Some(Self::Metre),
            9002 => Some(Self::InternationalFoot),
            9003 => Some(Self::UsSurveyFoot),
            _ => None,
        }
    }

    /// Human-readable unit label.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Metre => "m",
            Self::InternationalFoot => "ft (intl)",
            Self::UsSurveyFoot => "ft (US survey)",
        }
    }
}

struct KnownDatumEntry {
    datum: Datum,
    name: &'static str,
    is_geographic_crs: bool,
    grid_shift_recommended: bool,
}

fn lookup_known_datum(code: u32) -> Option<KnownDatumEntry> {
    const OSGB36: Datum = Datum::from_towgs84(
        AIRY_1830,
        [446.448, -125.157, 542.060, 0.1502, 0.2470, 0.8421, -20.4894],
    );
    let (datum, name, grid_shift_recommended) = match code {
        4326 | 4979 | 6326 => (WGS84_DATUM, "WGS 84", false),
        4284 | 6284 => (PULKOVO_1942, "Pulkovo 1942", false),
        4269 | 6269 => (WGS84_DATUM, "NAD83", true),
        4258 | 6258 => (WGS84_DATUM, "ETRS89", true),
        4277 | 6277 => (OSGB36, "OSGB36", true),
        _ => return None,
    };
    Some(KnownDatumEntry {
        datum,
        name,
        is_geographic_crs: code < 6000,
        grid_shift_recommended,
    })
}

impl KnownDatumEntry {
    fn warning(&self, code: u32) -> Option<DatumWarning> {
        self.grid_shift_recommended
            .then(|| DatumWarning::GridShiftRecommended {
                label: format!("{} (EPSG:{code})", self.name),
            })
    }
}

/// Resolve a known static [`Datum`] from a geographic CRS EPSG code (`2048`) or
/// geodetic datum EPSG code (`2050`).
pub fn known_datum_from_code(code: u32) -> Option<(Datum, Option<DatumWarning>)> {
    let entry = lookup_known_datum(code)?;
    Some((entry.datum, entry.warning(code)))
}

/// Resolve a known static geographic CRS (4xxx EPSG code only, keeping 6xxx
/// datum codes strictly distinct) into its `(Datum, label, Option<DatumWarning>)`.
pub(crate) fn known_geographic_epsg(code: u32) -> Option<(Datum, String, Option<DatumWarning>)> {
    let entry = lookup_known_datum(code)?;
    if !entry.is_geographic_crs {
        return None;
    }
    Some((
        entry.datum,
        format!("EPSG:{code} ({})", entry.name),
        entry.warning(code),
    ))
}

/// Resolve an ellipsoid from common GeoTIFF `GeogEllipsoidGeoKey` (2056) codes.
pub const fn ellipsoid_from_epsg(code: u16) -> Option<Ellipsoid> {
    match code {
        7030 => Some(WGS84),
        7019 => Some(Ellipsoid {
            semi_major_m: 6_378_137.0,
            flattening: 1.0 / 298.257_222_101,
        }),
        7024 => Some(KRASOVSKY),
        7001 => Some(AIRY_1830),
        7004 => Some(Ellipsoid {
            semi_major_m: 6_377_397.155,
            flattening: 1.0 / 299.152_812_8,
        }),
        _ => None,
    }
}

#[cfg(test)]
#[path = "datum_tests.rs"]
mod tests;
