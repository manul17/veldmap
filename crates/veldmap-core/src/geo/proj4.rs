//! General pure-Rust PROJ.4 / WKT1 / WKT2 coordinate transforms
//! (`proj4rs` + `proj4wkt` + `crs-definitions`) targeting WGS84 geodetic
//! coordinates at the module boundary.
//!
//! Leaf module above [`super::datum`] and [`super::geodesy`] with no dependency
//! on [`super::projection`] or [`super::crs`].

use std::sync::OnceLock;

use proj4rs::proj::Proj;
use thiserror::Error;

use super::datum::DatumWarning;
use super::geodesy::WGS84;

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

/// Return whether `text` begins with a WKT1/WKT2 CRS root keyword (ignoring a
/// leading UTF-8 BOM and ASCII whitespace).
pub(super) fn looks_like_wkt(text: &str) -> bool {
    const WKT_PREFIXES: &[&str] = &[
        "PROJCS[",
        "GEOGCS[",
        "PROJCRS[",
        "GEOGCRS[",
        "BOUNDCRS[",
        "COMPOUNDCRS[",
        "COMPD_CS[",
        "LOCAL_CS[",
    ];
    let trimmed = text.trim_start_matches('\u{FEFF}').trim_start();
    WKT_PREFIXES.iter().any(|prefix| {
        trimmed
            .get(..prefix.len())
            .is_some_and(|head| head.eq_ignore_ascii_case(prefix))
    })
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
