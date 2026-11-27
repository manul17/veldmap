//! Pure-Rust CRS catalog, datums/units, fast in-house map projections, general
//! `proj4rs` transforms, unified CRS specification ([`CrsSpec`]), custom
//! Transverse Mercator builder, and searchable UI CRS presets.

mod datum;
mod error;
mod presets;
mod proj4;
mod projection;
mod spec;

pub use self::datum::{
    Datum, DatumWarning, HelmertConvention, KRASOVSKY, LinearUnit, PULKOVO_1942, US_SURVEY_FOOT_M,
    WGS84_DATUM, ellipsoid_from_epsg, known_datum_from_code,
};
pub use self::error::CrsError;
pub use self::presets::{CrsPreset, builtin_crs_presets, search_crs_presets};
pub use self::proj4::{Proj4Projection, trim_crs_text};
pub use self::projection::{Projection, System, WEB_MERCATOR_LIMIT_DEG, web_mercator_lat_deg};
pub use self::spec::{CrsSpec, CrsTextKind, CustomTmParams, DatumPreset};

#[cfg(test)]
use self::spec::looks_like_wkt;

#[cfg(test)]
#[path = "crs_tests.rs"]
mod tests;
