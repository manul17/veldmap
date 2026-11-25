//! Pure-Rust CRS catalog, unified CRS specification ([`CrsSpec`]), custom
//! Transverse Mercator builder, and searchable UI CRS presets.
//!
//! Fast in-house projections remain in [`super::projection`], datums/units in
//! [`super::datum`], and general `proj4rs` transforms in [`super::proj4`].

mod presets;

use std::sync::Arc;

use super::datum::{
    Datum, DatumWarning, LinearUnit, PULKOVO_1942, WGS84_DATUM, known_geographic_epsg,
};
use super::proj4::{CrsError, Proj4Projection};
use super::projection::{Projection, System};

pub use presets::{CrsPreset, builtin_crs_presets, search_crs_presets};

/// Return whether `text` begins with a WKT1/WKT2 CRS root keyword (ignoring a
/// leading UTF-8 BOM and ASCII whitespace).
fn looks_like_wkt(text: &str) -> bool {
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

/// Syntax kind detected by [`CrsSpec::parse_text`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CrsTextKind {
    /// WKT1 or WKT2 CRS definition.
    Wkt,
    /// PROJ.4 string or `EPSG:<code>` shorthand.
    UserString,
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
        if let Some((datum, label, warning)) = known_geographic_epsg(code) {
            return Ok(Self::Geographic {
                datum,
                label,
                warning,
            });
        }
        if let Some(proj) = Projection::from_epsg_in_house(code) {
            return Ok(Self::Projected(proj));
        }
        Ok(Self::from_proj4(Proj4Projection::from_epsg(code)?))
    }

    /// Parse a WKT1 or WKT2 CRS string (such as a `.prj` sidecar).
    pub fn from_wkt(wkt: &str) -> Result<Self, CrsError> {
        Ok(Self::from_proj4(Proj4Projection::from_wkt(wkt)?))
    }

    /// Parse a PROJ.4 parameter string with an optional display label.
    pub fn from_proj_string_labeled(
        proj_string: &str,
        label: Option<String>,
    ) -> Result<Self, CrsError> {
        Ok(Self::from_proj4(Proj4Projection::from_proj_string(
            proj_string,
            label,
        )?))
    }

    /// Parse a PROJ.4 string (or `EPSG:<code>` shorthand).
    pub fn from_user_string(input: &str) -> Result<Self, CrsError> {
        let trimmed = input.trim().trim_start_matches('\u{FEFF}').trim();
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
        Self::from_proj_string_labeled(trimmed, None)
    }

    /// Parse a CRS string that may be WKT1/WKT2, a PROJ.4 string, or an
    /// `EPSG:<code>` shorthand, returning both the resolved [`CrsSpec`] and
    /// which syntax matched ([`CrsTextKind`]).
    ///
    /// Inputs starting with a WKT root keyword are parsed strictly as WKT so
    /// malformed WKT definitions report [`CrsError::InvalidWkt`] rather than a
    /// misleading PROJ error.
    pub fn parse_text(input: &str) -> Result<(Self, CrsTextKind), CrsError> {
        let trimmed = input.trim().trim_start_matches('\u{FEFF}').trim();
        if looks_like_wkt(trimmed) {
            return Self::from_wkt(trimmed).map(|crs| (crs, CrsTextKind::Wkt));
        }
        if let Ok(crs) = Self::from_wkt(trimmed) {
            return Ok((crs, CrsTextKind::Wkt));
        }
        Self::from_user_string(trimmed).map(|crs| (crs, CrsTextKind::UserString))
    }

    fn from_proj4(p4: Proj4Projection) -> Self {
        if p4.is_geographic() {
            Self::GeographicProj(Arc::new(p4))
        } else {
            Self::Projected(Projection::Proj4(Arc::new(p4)))
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
    const fn to_datum(self) -> Datum {
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
    fn to_system(&self) -> Result<System, CrsError> {
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

#[cfg(test)]
#[path = "crs_tests.rs"]
mod tests;
