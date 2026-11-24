//! Pure-Rust CRS catalog, WKT1/WKT2 parsing (`proj4wkt`), and general PROJ
//! coordinate transforms (`proj4rs` + `crs-definitions`).
//!
//! Fast in-house projections (WGS84 UTM, Pulkovo 1942 Gauss-Kruger, MSK-61,
//! Web Mercator, and custom Transverse Mercator) remain in
//! [`super::projection`]; this module extends coverage to arbitrary EPSG codes,
//! `.prj` WKT1/WKT2 sidecars, user-defined CRS definitions, and searchable UI
//! presets with explicit foreign-datum / grid-shift diagnostics.

use std::sync::{Arc, OnceLock};

use proj4rs::proj::Proj;
use thiserror::Error;

use super::geodesy::{Ellipsoid, WGS84};
use super::projection::{
    Datum, INTERNATIONAL_FOOT_M, KRASOVSKY, PULKOVO_1942, Projection, System, US_SURVEY_FOOT_M,
    WGS84_DATUM,
};

/// Errors produced when parsing or instantiating a CRS definition.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum CrsError {
    #[error("unsupported or unknown EPSG code: EPSG:{0}")]
    UnsupportedEpsg(u32),
    #[error("failed to parse WKT CRS definition: {0}")]
    InvalidWkt(String),
    #[error("failed to parse PROJ CRS string: {0}")]
    InvalidProj(String),
    #[error("CRS definition is missing inverse or forward transform support")]
    NonInvertible,
    #[error("invalid custom Transverse Mercator parameters: {0}")]
    InvalidCustomTm(String),
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

/// Pure-Rust `proj4rs` coordinate system wrapper targeting WGS84 geodetic
/// coordinates at the module boundary.
#[derive(Debug)]
pub struct Proj4Projection {
    proj: Proj,
    label: String,
    proj_string: String,
    warning: Option<DatumWarning>,
}

impl PartialEq for Proj4Projection {
    fn eq(&self, other: &Self) -> bool {
        self.proj_string == other.proj_string && self.label == other.label
    }
}

fn wgs84_proj() -> &'static Proj {
    static WGS84_PROJ: OnceLock<Proj> = OnceLock::new();
    WGS84_PROJ.get_or_init(|| {
        // Constant valid PROJ.4 string for WGS84 geographic coordinates.
        #[expect(
            clippy::expect_used,
            reason = "static WGS84 PROJ definition is compile-time valid"
        )]
        Proj::from_proj_string("+proj=longlat +datum=WGS84 +no_defs")
            .expect("static WGS84 PROJ definition must parse")
    })
}

impl Proj4Projection {
    /// Construct from a PROJ.4 parameter string.
    pub fn from_proj_string(proj_string: &str, label: Option<String>) -> Result<Self, CrsError> {
        let trimmed = proj_string.trim();
        let (proj, fallback_grid_shift) = match Proj::from_proj_string(trimmed) {
            Ok(proj) => (proj, false),
            Err(proj4rs::errors::Error::NadGridNotAvailable) => {
                let rewritten = rewrite_missing_nadgrid_proj(trimmed);
                let proj = Proj::from_proj_string(&rewritten)
                    .map_err(|e| CrsError::InvalidProj(e.to_string()))?;
                (proj, true)
            }
            Err(e) => return Err(CrsError::InvalidProj(e.to_string())),
        };
        if proj.is_geocent() {
            return Err(CrsError::InvalidProj(
                "3D geocentric CRS (+proj=geocent) is not supported as a 2D map CRS".to_string(),
            ));
        }
        if !proj.is_latlong() && (!proj.has_inverse() || !proj.has_forward()) {
            return Err(CrsError::NonInvertible);
        }
        let display_label = label.unwrap_or_else(|| trimmed.to_string());
        let warning = if fallback_grid_shift {
            Some(DatumWarning::GridShiftRecommended {
                label: display_label.clone(),
            })
        } else {
            detect_proj_datum_warning(&proj, trimmed, &display_label)
        };
        Ok(Self {
            proj,
            label: display_label,
            proj_string: trimmed.to_string(),
            warning,
        })
    }

    /// Construct from a WKT1 or WKT2 CRS string (such as a `.prj` sidecar).
    pub fn from_wkt(wkt: &str) -> Result<Self, CrsError> {
        let trimmed = wkt.trim().trim_start_matches('\u{FEFF}').trim();
        let proj_string = proj4wkt::wkt_to_projstring(trimmed)
            .map_err(|e| CrsError::InvalidWkt(e.to_string()))?;
        let label = extract_wkt_name(trimmed).unwrap_or_else(|| proj_string.clone());
        let mut out = Self::from_proj_string(&proj_string, Some(label.clone()))?;
        if out.warning.is_none() {
            out.warning = detect_wkt_datum_warning(trimmed, &proj_string, &label);
        }
        Ok(out)
    }

    /// Construct from an EPSG code backed by `crs-definitions`.
    pub fn from_epsg(code: u32) -> Result<Self, CrsError> {
        let code16 = u16::try_from(code).map_err(|_| CrsError::UnsupportedEpsg(code))?;
        let def = crs_definitions::from_code(code16).ok_or(CrsError::UnsupportedEpsg(code))?;
        let label = format!("EPSG:{code}");
        let mut out = Self::from_proj_string(def.proj4, Some(label.clone()))?;
        if out.warning.is_none() && is_plate_fixed_grid_epsg(code) {
            out.warning = Some(DatumWarning::GridShiftRecommended { label });
        }
        Ok(out)
    }

    /// Whether this PROJ definition is geographic (longitude/latitude) rather
    /// than projected.
    pub fn is_geographic(&self) -> bool {
        self.proj.is_latlong()
    }

    /// Human-readable CRS label.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Underlying normalized PROJ.4 string.
    pub fn proj_string(&self) -> &str {
        &self.proj_string
    }

    /// Scale factor from this projection's linear units to metres (`1.0` for
    /// metres or geographic).
    pub fn unit_to_metres(&self) -> f64 {
        let m = self.proj.to_meter();
        if m.is_finite() && m > 0.0 { m } else { 1.0 }
    }

    /// Optional datum warning (unshifted foreign datum or missing shift grid).
    pub fn datum_warning(&self) -> Option<&DatumWarning> {
        self.warning.as_ref()
    }

    /// Convert projected `(x, y)` (or source-datum `(lon_deg, lat_deg)` when
    /// geographic) into WGS84 `(lat_deg, lon_deg)`.
    pub fn to_wgs84_geodetic(&self, x: f64, y: f64) -> (f64, f64) {
        if !(x.is_finite() && y.is_finite()) {
            return (f64::NAN, f64::NAN);
        }
        let mut pt = if self.proj.is_latlong() {
            (x.to_radians(), y.to_radians())
        } else {
            (x, y)
        };
        if proj4rs::transform::transform(&self.proj, wgs84_proj(), &mut pt).is_err() {
            return (f64::NAN, f64::NAN);
        }
        let lon_deg = pt.0.to_degrees();
        let lat_deg = pt.1.to_degrees();
        if lat_deg.is_finite() && lon_deg.is_finite() {
            (lat_deg, lon_deg)
        } else {
            (f64::NAN, f64::NAN)
        }
    }

    /// Convert WGS84 `(lat_deg, lon_deg)` into projected `(x, y)` (or
    /// source-datum `(lon_deg, lat_deg)` when geographic).
    pub fn from_wgs84_geodetic(&self, lat_deg: f64, lon_deg: f64) -> (f64, f64) {
        if !(lat_deg.is_finite() && lon_deg.is_finite()) {
            return (f64::NAN, f64::NAN);
        }
        let mut pt = (lon_deg.to_radians(), lat_deg.to_radians());
        if proj4rs::transform::transform(wgs84_proj(), &self.proj, &mut pt).is_err() {
            return (f64::NAN, f64::NAN);
        }
        if self.proj.is_latlong() {
            (pt.0.to_degrees(), pt.1.to_degrees())
        } else {
            pt
        }
    }
}

/// Parsed CRS specification covering both projected and geographic coordinate
/// reference systems (including foreign-datum geographic bindings).
#[derive(Clone, Debug, PartialEq)]
pub enum CrsSpec {
    /// Projected coordinate system mapping `(x, y)` in CRS linear units to/from
    /// WGS84 `(lat_deg, lon_deg)`.
    Projected(Projection),
    /// Geographic coordinate system on a known static 7-parameter [`Datum`].
    Geographic {
        datum: Datum,
        label: String,
        warning: Option<DatumWarning>,
    },
    /// General geographic coordinate system backed by `proj4rs` (e.g., from a
    /// `.prj` WKT file with custom `TOWGS84` or ellipsoid).
    GeographicProj(Arc<Proj4Projection>),
}

impl CrsSpec {
    /// Resolve any supported EPSG (or MSK-61 `633610x`) code into a [`CrsSpec`].
    pub fn from_epsg(code: u32) -> Result<Self, CrsError> {
        match code {
            4326 | 4979 => Ok(Self::Geographic {
                datum: WGS84_DATUM,
                label: format!("EPSG:{code} (WGS 84)"),
                warning: None,
            }),
            4284 => Ok(Self::Geographic {
                datum: PULKOVO_1942,
                label: "EPSG:4284 (Pulkovo 1942)".to_string(),
                warning: None,
            }),
            4269 => Ok(Self::Geographic {
                datum: WGS84_DATUM,
                label: "EPSG:4269 (NAD83)".to_string(),
                warning: Some(DatumWarning::GridShiftRecommended {
                    label: "NAD83 / EPSG:4269".to_string(),
                }),
            }),
            4258 => Ok(Self::Geographic {
                datum: WGS84_DATUM,
                label: "EPSG:4258 (ETRS89)".to_string(),
                warning: Some(DatumWarning::GridShiftRecommended {
                    label: "ETRS89 / EPSG:4258".to_string(),
                }),
            }),
            4277 => {
                let (datum, warning) =
                    known_datum_from_code(4277).ok_or(CrsError::UnsupportedEpsg(4277))?;
                Ok(Self::Geographic {
                    datum,
                    label: "EPSG:4277 (OSGB36)".to_string(),
                    warning,
                })
            }
            _ => {
                if let Some(proj) = Projection::from_epsg(code) {
                    return Ok(Self::Projected(proj));
                }
                let p4 = Proj4Projection::from_epsg(code)?;
                if p4.is_geographic() {
                    Ok(Self::GeographicProj(Arc::new(p4)))
                } else {
                    Ok(Self::Projected(Projection::Proj4(Arc::new(p4))))
                }
            }
        }
    }

    /// Parse a WKT1 or WKT2 CRS string (such as a `.prj` sidecar).
    pub fn from_wkt(wkt: &str) -> Result<Self, CrsError> {
        let p4 = Proj4Projection::from_wkt(wkt)?;
        if p4.is_geographic() {
            Ok(Self::GeographicProj(Arc::new(p4)))
        } else {
            Ok(Self::Projected(Projection::Proj4(Arc::new(p4))))
        }
    }

    /// Parse a PROJ.4 string (or `EPSG:<code>` shorthand).
    pub fn from_user_string(input: &str) -> Result<Self, CrsError> {
        let trimmed = input.trim();
        if let Some(rest) = trimmed
            .strip_prefix("EPSG:")
            .or_else(|| trimmed.strip_prefix("epsg:"))
            && let Ok(code) = rest.trim().parse::<u32>()
        {
            return Self::from_epsg(code);
        }
        if let Ok(code) = trimmed.parse::<u32>() {
            return Self::from_epsg(code);
        }
        let p4 = Proj4Projection::from_proj_string(trimmed, None)?;
        if p4.is_geographic() {
            Ok(Self::GeographicProj(Arc::new(p4)))
        } else {
            Ok(Self::Projected(Projection::Proj4(Arc::new(p4))))
        }
    }

    /// Human-readable CRS label.
    pub fn label(&self) -> String {
        match self {
            Self::Projected(p) => p.label(),
            Self::Geographic { label, .. } => label.clone(),
            Self::GeographicProj(p) => p.label().to_string(),
        }
    }

    /// Optional datum shift warning.
    pub fn datum_warning(&self) -> Option<&DatumWarning> {
        match self {
            Self::Projected(p) => p.datum_warning(),
            Self::Geographic { warning, .. } => warning.as_ref(),
            Self::GeographicProj(p) => p.datum_warning(),
        }
    }
}

/// Datum preset choice for custom Transverse Mercator definitions (e.g., MSK-xx).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum DatumPreset {
    #[default]
    Wgs84,
    Pulkovo1942,
}

impl DatumPreset {
    pub const fn to_datum(self) -> Datum {
        match self {
            Self::Wgs84 => WGS84_DATUM,
            Self::Pulkovo1942 => PULKOVO_1942,
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::Wgs84 => "WGS 84",
            Self::Pulkovo1942 => "Pulkovo 1942 (SK-42, GOST R 51794-2008)",
        }
    }
}

/// User-configurable Transverse Mercator projection parameters (unlocking all
/// regional MSK-xx zones, custom survey grids, and GeoTIFF user-defined TM).
#[derive(Clone, PartialEq, Debug)]
pub struct CustomTmParams {
    pub name: String,
    pub datum: DatumPreset,
    pub central_meridian_deg: f64,
    pub origin_lat_deg: f64,
    pub scale: f64,
    pub false_easting_m: f64,
    pub false_northing_m: f64,
    pub unit: LinearUnit,
}

impl Default for CustomTmParams {
    fn default() -> Self {
        Self {
            name: "Custom Transverse Mercator".to_string(),
            datum: DatumPreset::Pulkovo1942,
            central_meridian_deg: 37.983_333_333_33,
            origin_lat_deg: 0.0,
            scale: 1.0,
            false_easting_m: 1_300_000.0,
            false_northing_m: -4_811_057.628,
            unit: LinearUnit::Metre,
        }
    }
}

impl CustomTmParams {
    /// Validate and convert into an exact in-house [`System`].
    pub fn to_system(&self) -> Result<System, CrsError> {
        if !self.central_meridian_deg.is_finite()
            || !(-180.0..=180.0).contains(&self.central_meridian_deg)
        {
            return Err(CrsError::InvalidCustomTm(
                "central meridian must be within [-180, 180] deg".to_string(),
            ));
        }
        if !self.origin_lat_deg.is_finite() || !(-90.0..=90.0).contains(&self.origin_lat_deg) {
            return Err(CrsError::InvalidCustomTm(
                "origin latitude must be within [-90, 90] deg".to_string(),
            ));
        }
        if !self.scale.is_finite() || self.scale <= 0.0 || self.scale > 2.0 {
            return Err(CrsError::InvalidCustomTm(
                "scale factor must be finite and in (0, 2]".to_string(),
            ));
        }
        if !(self.false_easting_m.is_finite() && self.false_northing_m.is_finite()) {
            return Err(CrsError::InvalidCustomTm(
                "false easting/northing must be finite".to_string(),
            ));
        }
        Ok(System {
            datum: self.datum.to_datum(),
            central_meridian_deg: self.central_meridian_deg,
            origin_lat_deg: self.origin_lat_deg,
            scale: self.scale,
            false_easting_m: self.false_easting_m,
            false_northing_m: self.false_northing_m,
            unit_to_metres: self.unit.to_metres(),
        })
    }

    /// Validate and convert into a named [`Projection::Transverse`], preserving
    /// [`Self::name`] in UI labels and metadata inspectors.
    pub fn to_projection(&self) -> Result<Projection, CrsError> {
        let sys = self.to_system()?;
        Ok(Projection::transverse_named(sys, self.name.as_str()))
    }
}

/// Searchable CRS preset entry for the UI CRS picker.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CrsPreset {
    pub epsg: u32,
    pub name: String,
    pub category: &'static str,
}

impl CrsPreset {
    /// Display code label (`EPSG:<code>` for standard registry codes, or
    /// `MSK-61:z<n>` for non-standard regional pseudo-codes).
    pub fn code_label(&self) -> String {
        if (6_336_101..=6_336_103).contains(&self.epsg) {
            format!("MSK-61:z{}", self.epsg - 6_336_100)
        } else {
            format!("EPSG:{}", self.epsg)
        }
    }
}

/// Curated searchable CRS preset catalog (WGS84, Web Mercator, UTM 1–60 N/S,
/// Pulkovo 1942 GK 1–32, MSK-61 1–3, and major international projected CRSs).
pub fn builtin_crs_presets() -> &'static [CrsPreset] {
    static PRESETS: OnceLock<Vec<CrsPreset>> = OnceLock::new();
    PRESETS.get_or_init(|| {
        let mut out = Vec::with_capacity(170);
        out.push(CrsPreset {
            epsg: 4326,
            name: "WGS 84 (Geographic lon/lat)".to_string(),
            category: "Global",
        });
        out.push(CrsPreset {
            epsg: 3857,
            name: "WGS 84 / Pseudo-Mercator (Web Mercator)".to_string(),
            category: "Global",
        });
        out.push(CrsPreset {
            epsg: 4284,
            name: "Pulkovo 1942 (Geographic lon/lat)".to_string(),
            category: "Regional",
        });
        out.push(CrsPreset {
            epsg: 3035,
            name: "ETRS89-extended / LAEA Europe".to_string(),
            category: "Regional",
        });
        out.push(CrsPreset {
            epsg: 27700,
            name: "OSGB36 / British National Grid".to_string(),
            category: "Regional",
        });
        out.push(CrsPreset {
            epsg: 2154,
            name: "RGF93 / Lambert-93 (France)".to_string(),
            category: "Regional",
        });
        out.push(CrsPreset {
            epsg: 25832,
            name: "ETRS89 / UTM zone 32N".to_string(),
            category: "Regional",
        });
        out.push(CrsPreset {
            epsg: 25833,
            name: "ETRS89 / UTM zone 33N".to_string(),
            category: "Regional",
        });
        out.push(CrsPreset {
            epsg: 3413,
            name: "WGS 84 / NSIDC Sea Ice Polar Stereographic North".to_string(),
            category: "Polar",
        });
        out.push(CrsPreset {
            epsg: 3031,
            name: "WGS 84 / Antarctic Polar Stereographic".to_string(),
            category: "Polar",
        });
        out.push(CrsPreset {
            epsg: 26910,
            name: "NAD83 / UTM zone 10N".to_string(),
            category: "North America",
        });
        out.push(CrsPreset {
            epsg: 26918,
            name: "NAD83 / UTM zone 18N".to_string(),
            category: "North America",
        });
        out.push(CrsPreset {
            epsg: 2263,
            name: "NAD83 / New York Long Island (ftUS)".to_string(),
            category: "North America",
        });
        for zone in 1..=3 {
            out.push(CrsPreset {
                epsg: 6_336_100 + zone,
                name: format!("MSK-61 / Pulkovo 1942 zone {zone}"),
                category: "MSK / Pulkovo",
            });
        }
        for zone in 2..=32 {
            out.push(CrsPreset {
                epsg: 28_400 + zone,
                name: format!("Pulkovo 1942 / Gauss-Kruger zone {zone}"),
                category: "MSK / Pulkovo",
            });
        }
        for zone in 1..=60 {
            out.push(CrsPreset {
                epsg: 32_600 + zone,
                name: format!("WGS 84 / UTM zone {zone}N"),
                category: "UTM North",
            });
        }
        for zone in 1..=60 {
            out.push(CrsPreset {
                epsg: 32_700 + zone,
                name: format!("WGS 84 / UTM zone {zone}S"),
                category: "UTM South",
            });
        }
        out
    })
}

/// Search curated presets (plus direct numeric EPSG lookup in `crs-definitions`)
/// matching `query` case-insensitively.
pub fn search_crs_presets(query: &str, limit: usize) -> Vec<CrsPreset> {
    let q = query.trim().to_ascii_lowercase();
    let mut results: Vec<CrsPreset> = builtin_crs_presets()
        .iter()
        .filter(|preset| {
            if q.is_empty() {
                return true;
            }
            let code_str = preset.epsg.to_string();
            let epsg_tag = format!("epsg:{code_str}");
            code_str.contains(&q)
                || epsg_tag.contains(&q)
                || preset.name.to_ascii_lowercase().contains(&q)
                || preset.category.to_ascii_lowercase().contains(&q)
        })
        .take(limit)
        .cloned()
        .collect();

    // Allow direct lookup of any EPSG code in `crs-definitions` even if not in the curated list.
    let digits = q.strip_prefix("epsg:").unwrap_or(&q).trim();
    if let Ok(code) = digits.parse::<u32>()
        && !results.iter().any(|p| p.epsg == code)
        && CrsSpec::from_epsg(code).is_ok()
    {
        results.insert(
            0,
            CrsPreset {
                epsg: code,
                name: format!("EPSG:{code} (crs-definitions)"),
                category: "EPSG Catalog",
            },
        );
        results.truncate(limit);
    }
    results
}

/// Resolve a known static [`Datum`] from a geographic CRS EPSG code (`2048`) or
/// geodetic datum EPSG code (`2050`).
pub fn known_datum_from_code(code: u32) -> Option<(Datum, Option<DatumWarning>)> {
    match code {
        4326 | 4979 | 6326 => Some((WGS84_DATUM, None)),
        4284 | 6284 => Some((PULKOVO_1942, None)),
        4269 | 6269 => Some((
            WGS84_DATUM,
            Some(DatumWarning::GridShiftRecommended {
                label: format!("NAD83 (EPSG:{code})"),
            }),
        )),
        4258 | 6258 => Some((
            WGS84_DATUM,
            Some(DatumWarning::GridShiftRecommended {
                label: format!("ETRS89 (EPSG:{code})"),
            }),
        )),
        4277 | 6277 => Some((
            Datum::from_towgs84(
                Ellipsoid {
                    semi_major_m: 6_377_563.396,
                    flattening: 1.0 / 299.324_964_6,
                },
                [446.448, -125.157, 542.060, 0.1502, 0.2470, 0.8421, -20.4894],
            ),
            Some(DatumWarning::GridShiftRecommended {
                label: format!("OSGB36 (EPSG:{code})"),
            }),
        )),
        _ => None,
    }
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
        7001 => Some(Ellipsoid {
            semi_major_m: 6_377_563.396,
            flattening: 1.0 / 299.324_964_6,
        }),
        7004 => Some(Ellipsoid {
            semi_major_m: 6_377_397.155,
            flattening: 1.0 / 299.152_812_8,
        }),
        _ => None,
    }
}

/// Rewrite a PROJ string whose mandatory `+nadgrids=` or `+datum=NAD27` grid
/// files are not bundled, substituting a North-America mean 3-parameter Helmert
/// shift for NAD27 (`+ellps=clrk66 +towgs84=-8,160,176`) and stripping
/// unavailable `+nadgrids=` tokens so the CRS still opens with a visible
/// [`DatumWarning::GridShiftRecommended`] banner.
fn rewrite_missing_nadgrid_proj(proj_str: &str) -> String {
    let has_towgs84 = proj_str
        .split_whitespace()
        .any(|tok| tok.to_ascii_lowercase().starts_with("+towgs84="));
    let mut out = Vec::new();
    for tok in proj_str.split_whitespace() {
        let lower = tok.to_ascii_lowercase();
        if lower == "+datum=nad27" {
            out.push("+ellps=clrk66".to_string());
            if !has_towgs84 {
                out.push("+towgs84=-8,160,176".to_string());
            }
        } else if lower.starts_with("+nadgrids=") {
            if lower == "+nadgrids=@null" {
                out.push(tok.to_string());
            }
            // Drop unavailable shift-grid filenames; callers emit GridShiftRecommended.
        } else {
            out.push(tok.to_string());
        }
    }
    out.join(" ")
}

fn is_plate_fixed_grid_epsg(code: u32) -> bool {
    matches!(
        code,
        2154 | 3035 | 4171 | 4258 | 25_828..=25_838 | 3038..=3051
    )
}

fn extract_wkt_name(wkt: &str) -> Option<String> {
    let start = wkt.find('"')? + 1;
    let rest = wkt.get(start..)?;
    let end = rest.find('"')?;
    let name = rest.get(..end)?.trim();
    (!name.is_empty()).then(|| name.to_string())
}

fn has_nonzero_towgs84(proj_lower: &str) -> bool {
    proj_lower.split_whitespace().any(|tok| {
        tok.strip_prefix("+towgs84=").is_some_and(|vals| {
            vals.split(',')
                .filter_map(|s| s.parse::<f64>().ok())
                .any(|v| v.is_finite() && v.abs() > 1e-12)
        })
    })
}

fn detect_proj_datum_warning(proj: &Proj, proj_str: &str, label: &str) -> Option<DatumWarning> {
    let lower = proj_str.to_ascii_lowercase();
    if lower.contains("+nadgrids=") && !lower.contains("+nadgrids=@null") {
        return Some(DatumWarning::GridShiftRecommended {
            label: label.to_string(),
        });
    }
    if lower.contains("+datum=nad27")
        || lower.contains("+datum=nad83")
        || lower.contains("+datum=osgb36")
    {
        return Some(DatumWarning::GridShiftRecommended {
            label: label.to_string(),
        });
    }
    let has_towgs84 = lower.contains("+towgs84=");
    let has_datum = lower.contains("+datum=");
    let has_null_nadgrid = lower.contains("+nadgrids=@null");
    if !has_towgs84 && !has_datum && !has_null_nadgrid {
        let (a, b) = proj.ellipse_parameters();
        let is_sphere = (a - b).abs() <= 1.0;
        let differs_from_wgs84 =
            (a - WGS84.semi_major_m).abs() > 1.0 || (b - WGS84.semi_minor_m()).abs() > 1.0;
        if !is_sphere && differs_from_wgs84 {
            return Some(DatumWarning::UnshiftedForeignDatum {
                label: label.to_string(),
            });
        }
    }
    None
}

fn is_itrf_aligned_wkt(wkt_upper: &str) -> bool {
    const ITRF_TOKENS: &[&str] = &[
        "WGS_1984",
        "WGS 84",
        "WORLD GEODETIC SYSTEM 1984",
        "ITRF",
        "GDA2020",
        "GDA_2020",
        "GEOCENTRIC_DATUM_OF_AUSTRALIA_2020",
        "SIRGAS",
        "NZGD2000",
        "NZGD_2000",
        "JGD2000",
        "JGD2011",
        "CGCS2000",
        "GSK-2011",
        "GSK_2011",
        "PZ-90.11",
    ];
    ITRF_TOKENS.iter().any(|tok| wkt_upper.contains(tok))
}

fn detect_wkt_datum_warning(wkt: &str, proj_str: &str, label: &str) -> Option<DatumWarning> {
    let wkt_upper = wkt.to_ascii_uppercase();
    let proj_lower = proj_str.to_ascii_lowercase();
    if is_itrf_aligned_wkt(&wkt_upper) || proj_lower.contains("+datum=wgs84") {
        return None;
    }
    if wkt_upper.contains("NAD83")
        || wkt_upper.contains("NORTH_AMERICAN_DATUM_1983")
        || wkt_upper.contains("NAD27")
        || wkt_upper.contains("OSGB_1936")
        || wkt_upper.contains("OSGB36")
        || wkt_upper.contains("ETRS89")
        || wkt_upper.contains("RGF93")
        || wkt_upper.contains("RESEAU_GEODESIQUE_FRANCAIS")
        || wkt_upper.contains("GDA94")
    {
        return Some(DatumWarning::GridShiftRecommended {
            label: label.to_string(),
        });
    }
    if !has_nonzero_towgs84(&proj_lower) {
        return Some(DatumWarning::UnshiftedForeignDatum {
            label: label.to_string(),
        });
    }
    None
}
