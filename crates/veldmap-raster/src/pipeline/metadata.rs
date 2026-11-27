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
    I8,
    U16,
    I16,
    U32,
    I32,
    F16,
    F32,
    F64,
}

impl SampleType {
    /// Storage width of one sample in bytes.
    pub const fn byte_size(self) -> usize {
        match self {
            Self::U8 | Self::I8 => 1,
            Self::U16 | Self::I16 | Self::F16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::F64 => 8,
        }
    }

    /// Return whether this sample type maps losslessly into an unsigned 8- or
    /// 16-bit categorical GPU track (`R8Uint` / `R16Uint`).
    pub const fn categorical_pixel_format(self) -> Option<PixelFormat> {
        match self {
            Self::U8 => Some(PixelFormat::R8Uint),
            Self::U16 => Some(PixelFormat::R16Uint),
            _ => None,
        }
    }
}

/// Zoom-invariant native-resolution CPU sample readout for a raster band.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BandSample {
    /// Display composite `RGBA8` pixel.
    Rgba([u8; 4]),
    /// Scientific scalar pixel with raw source value and physical `raw * scale + offset`.
    Scalar { raw: f64, physical: f64 },
    /// Categorical discrete class or QA bitmask value.
    Categorical(u16),
    /// Pixel lies inside raster bounds but carries declared nodata / NaN / alpha = 0.
    Nodata,
    /// Pixel lies outside raster bounds or source cannot sample pointwise.
    Unavailable,
}

/// Number of histogram bins exposed in [`BandStats`].
pub const HISTOGRAM_BINS: usize = 64;

/// Sampled band statistics and histograms in physical value space (`val * scale + offset`).
#[derive(Clone, Debug, PartialEq)]
pub struct BandStats {
    pub min: f64,
    pub max: f64,
    pub p2: f64,
    pub p98: f64,
    pub db_min: Option<f64>,
    pub db_max: Option<f64>,
    pub db_p2: Option<f64>,
    pub db_p98: Option<f64>,
    pub sample_count: u64,
    pub histogram: [u32; HISTOGRAM_BINS],
    pub db_histogram: [u32; HISTOGRAM_BINS],
}

/// External per-band radiometric metadata (e.g., from STAC `raster:bands`)
/// applied to bands of a GeoTIFF/COG source whose embedded metadata did not
/// declare the corresponding fields.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ExternalBandMeta {
    pub scale: Option<f64>,
    pub offset: Option<f64>,
    pub nodata: Option<f64>,
    pub unit: Option<String>,
}

impl ExternalBandMeta {
    /// Return whether at least one external metadata field is populated.
    pub fn has_any(&self) -> bool {
        self.scale.is_some()
            || self.offset.is_some()
            || self.nodata.is_some()
            || self.unit.as_deref().is_some_and(|u| !u.trim().is_empty())
    }
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
    /// True when the raster container explicitly declared `SCALE` or `OFFSET`
    /// metadata for this band (even if equal to identity `1.0` / `0.0`), so
    /// external STAC `raster:bands` overrides never clobber embedded metadata.
    pub has_embedded_scale_offset: bool,
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
            has_embedded_scale_offset: false,
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

impl RasterDescription {
    /// Return whether this raster carries enough pixel-to-coordinate geometry
    /// (an affine placement or tie-point grid) for a CRS override to bind it.
    pub fn can_override_crs(&self) -> bool {
        self.placement.is_some() || !self.ties.is_empty()
    }

    /// Look up band metadata by [`BandKey`].
    pub fn band_info(&self, band: BandKey) -> Option<&BandInfo> {
        self.bands.iter().find(|b| b.band_key == band)
    }

    /// Iterate over individual indexed bands (`BandKey::Index(_)`).
    pub fn indexed_bands(&self) -> impl Iterator<Item = &BandInfo> {
        self.bands
            .iter()
            .filter(|b| matches!(b.band_key, BandKey::Index(_)))
    }
}
