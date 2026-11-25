//! Unified raster band metadata (`BandInfo`), sample types, and georeference
//! models (`GeoTie`, `PlacementCrs`, `GeoPlacement`, `RasterDescription`).
//!
//! Raw GeoTIFF binary tag parsing (`GeoKeyDirectoryTag` / `34735`–`34737`)
//! lives in `crate::tiff::georef`.

use veldmap_core::geo::{CrsSpec, DatumWarning};
use veldmap_core::tile::{BandKey, PixelFormat};

use super::plan::LevelPlan;

/// Raw numeric sample storage type of a decoded raster band.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SampleType {
    U8,
    U16,
    I16,
    F32,
}

/// Metadata for one display composite or scientific/categorical band in a
/// raster source.
#[derive(Clone, Debug, PartialEq)]
pub struct BandInfo {
    pub band_key: BandKey,
    pub index: u16,
    pub name: Option<String>,
    pub units: Option<String>,
    pub pixel_format: PixelFormat,
    pub sample_type: SampleType,
    pub scale: f64,
    pub offset: f64,
    pub nodata: Option<f64>,
    pub channels: usize,
    pub has_alpha: bool,
}

impl BandInfo {
    /// Construct a standard display-composite band entry (`BandKey::Display`).
    pub(crate) fn display(
        sample_type: SampleType,
        channels: usize,
        has_alpha: bool,
        nodata: Option<f64>,
    ) -> Self {
        Self {
            band_key: BandKey::Display,
            index: 0,
            name: None,
            units: None,
            pixel_format: PixelFormat::Rgba8UnormSrgb,
            sample_type,
            scale: 1.0,
            offset: 0.0,
            nodata,
            channels,
            has_alpha,
        }
    }
}

/// Single georeference tie point mapping raster pixel `(px, py)` to geographic
/// `(lat, lon)` degrees.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeoTie {
    pub px: f64,
    pub py: f64,
    pub lat: f64,
    pub lon: f64,
}

/// Validate that `(lat, lon)` falls within geographic bounds (`|lat| <= 90`,
/// `|lon| <= 360`, both finite) so out-of-range or NaN/Inf tie tags never reach
/// globe geometry.
pub(crate) fn is_valid_lat_lon(lat: f64, lon: f64) -> bool {
    lat.is_finite()
        && lon.is_finite()
        && (-90.0..=90.0).contains(&lat)
        && (-360.0..=360.0).contains(&lon)
}

/// Coordinate reference system associated with a [`GeoPlacement`].
#[derive(Clone, Debug, PartialEq)]
pub enum PlacementCrs {
    /// Standard EPSG (or MSK-61 regional pseudo-EPSG) code resolved via
    /// [`CrsSpec::from_epsg`].
    Epsg(u32),
    /// Pre-resolved CRS specification (from user-defined GeoTIFF GeoKeys,
    /// `.prj` WKT1/WKT2 sidecars, or an explicit user override) paired with an
    /// optional datum diagnostic warning.
    Resolved {
        crs: CrsSpec,
        warning: Option<DatumWarning>,
    },
}

/// Projected or affine-georeferenced raster binding: optional CRS and
/// pixel-to-CRS affine transform.
#[derive(Clone, Debug, PartialEq)]
pub struct GeoPlacement {
    /// Optional CRS associated with [`Self::affine`]. `None` when a world file
    /// or embedded pixel affine lacks a CRS specification.
    pub crs: Option<PlacementCrs>,
    /// `x = a[0] * i + a[1] * j + a[2]`, `y = a[3] * i + a[4] * j + a[5]`,
    /// where `(i, j)` are pixel top-left corner coordinates (`j` increasing
    /// southward, with `PixelIsPoint` half-pixel shift already removed).
    pub affine: [f64; 6],
}

/// Header-derived description of an opened raster source.
#[derive(Clone, Debug)]
pub struct RasterDescription {
    pub width: u32,
    pub height: u32,
    pub bands: Vec<BandInfo>,
    pub ties: Vec<GeoTie>,
    pub placement: Option<GeoPlacement>,
    /// Resolved geographic CRS for tie-point grid bindings (`None` implies
    /// default WGS 84 / EPSG:4326).
    pub geographic_crs: Option<CrsSpec>,
    pub binding_trouble: Option<String>,
    pub levels: Vec<LevelPlan>,
}
