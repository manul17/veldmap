//! Unified raster band metadata (`BandInfo`), sample types, and georeference
//! models (`GeoTie`, `GeoPlacement`, `GeoKeyParams`, `RasterDescription`).
//!
//! Raw GeoTIFF binary tag parsing (`GeoKeyDirectoryTag` / `34735`–`34737`)
//! lives in `crate::tiff::georef`.

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

/// Single WGS84 georeference tie point mapping raster pixel `(px, py)` to
/// geographic `(lat, lon)` degrees.
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

/// Pixel coordinate reference convention (`GTRasterTypeGeoKey = 1025`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum GeoRasterType {
    /// Pixel coordinates denote the top-left corner of the pixel cell (`1`).
    #[default]
    PixelIsArea,
    /// Pixel coordinates denote the center of the pixel cell (`2`).
    PixelIsPoint,
}

impl GeoRasterType {
    /// Half-pixel shift added when converting `(px, py)` into top-left cell
    /// corner coordinates (`0.0` for `PixelIsArea`, `0.5` for `PixelIsPoint`).
    pub(crate) const fn half_pixel_shift(self) -> f64 {
        match self {
            Self::PixelIsArea => 0.0,
            Self::PixelIsPoint => 0.5,
        }
    }
}

/// Parsed raw GeoTIFF key directory entry value.
#[derive(Clone, Debug, PartialEq)]
pub enum GeoKeyValue {
    Short(u16),
    Double(Vec<f64>),
    Ascii(String),
}

/// Parsed GeoTIFF key directory preserving resolved double/ASCII parameters
/// for downstream CRS resolution.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GeoKeyParams {
    pub model_type: Option<u16>,
    pub raster_type: GeoRasterType,
    pub geographic_crs: Option<u16>,
    pub projected_crs: Option<u16>,
    pub keys: Vec<(u16, GeoKeyValue)>,
}

impl GeoKeyParams {
    /// Look up a short (`u16`) GeoKey value by numeric ID.
    pub fn get_u16(&self, id: u16) -> Option<u16> {
        self.keys.iter().find_map(|(k, v)| match (k, v) {
            (key, GeoKeyValue::Short(val)) if *key == id => Some(*val),
            _ => None,
        })
    }

    /// Look up a `f64` scalar GeoKey value.
    pub fn get_f64(&self, id: u16) -> Option<f64> {
        self.keys.iter().find_map(|(k, v)| match (k, v) {
            (key, GeoKeyValue::Double(vals)) if *key == id => vals.first().copied(),
            _ => None,
        })
    }

    /// Look up an ASCII GeoKey string.
    pub fn get_ascii(&self, id: u16) -> Option<&str> {
        self.keys.iter().find_map(|(k, v)| match (k, v) {
            (key, GeoKeyValue::Ascii(s)) if *key == id => Some(s.as_str()),
            _ => None,
        })
    }
}

/// Projected raster binding: optional EPSG code, pixel-to-CRS affine transform,
/// and optional parsed GeoKey parameters (present for GeoTIFF sources).
#[derive(Clone, Debug, PartialEq)]
pub struct GeoPlacement {
    pub epsg: Option<u32>,
    /// `x = a[0] * i + a[1] * j + a[2]`, `y = a[3] * i + a[4] * j + a[5]`,
    /// where `(i, j)` are pixel top-left corner coordinates (`j` increasing
    /// southward, with `PixelIsPoint` half-pixel shift already removed).
    pub affine: [f64; 6],
    pub geo_params: Option<GeoKeyParams>,
}

/// Header-derived description of an opened raster source.
#[derive(Clone, Debug)]
pub struct RasterDescription {
    pub width: u32,
    pub height: u32,
    pub bands: Vec<BandInfo>,
    pub ties: Vec<GeoTie>,
    pub placement: Option<GeoPlacement>,
    /// Non-WGS84 geographic datum EPSG code if declared on either a projected
    /// or geographic binding (e.g. Pulkovo-42 `4284`).
    pub foreign_datum_epsg: Option<u16>,
    pub binding_trouble: Option<String>,
    pub levels: Vec<LevelPlan>,
}
