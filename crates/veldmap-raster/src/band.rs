//! Unified raster band metadata (`BandInfo`), sample types, and GeoTIFF-ready
//! georeference models (`GeoTie`, `GeoPlacement`, `GeoKeyParams`,
//! `RasterDescription`).

use veldmap_core::tile::{BandKey, LevelPlan, PixelFormat};

/// Raw numeric sample storage type of a decoded raster band.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SampleType {
    U8,
    U16,
    I16,
    F32,
}

impl SampleType {
    /// Byte width of one raw scalar sample.
    pub const fn byte_size(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::U16 | Self::I16 => 2,
            Self::F32 => 4,
        }
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
    pub nodata: Option<f64>,
    pub channels: usize,
    pub has_alpha: bool,
}

impl BandInfo {
    /// Construct a standard display-composite band entry (`BandKey::Display`).
    pub fn display(
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
pub fn is_valid_lat_lon(lat: f64, lon: f64) -> bool {
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
    pub const fn half_pixel_shift(self) -> f64 {
        match self {
            Self::PixelIsArea => 0.0,
            Self::PixelIsPoint => 0.5,
        }
    }
}

/// Parsed raw GeoTIFF key directory entry resolved against `GeoDoubleParamsTag`
/// (`34736`) and `GeoAsciiParamsTag` (`34737`).
#[derive(Clone, Debug, PartialEq)]
pub enum GeoKeyValue {
    Short(u16),
    Double(Vec<f64>),
    Ascii(String),
}

/// Parsed GeoTIFF key directory preserving resolved `GeoDoubleParamsTag` and
/// `GeoAsciiParamsTag` parameters for downstream CRS resolution.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GeoKeyParams {
    pub model_type: Option<u16>,
    pub raster_type: GeoRasterType,
    pub geographic_crs: Option<u16>,
    pub projected_crs: Option<u16>,
    pub keys: Vec<(u16, GeoKeyValue)>,
}

impl GeoKeyParams {
    /// Parse raw `GeoKeyDirectoryTag` (`34735`), `GeoDoubleParamsTag` (`34736`),
    /// and `GeoAsciiParamsTag` (`34737`) payloads.
    pub fn parse(raw_keys: &[u16], double_params: &[f64], ascii_params: &str) -> Self {
        let mut out = Self::default();

        let Some(&num_keys_raw) = raw_keys.get(3) else {
            return out;
        };
        let Some(entries) = raw_keys.get(4..) else {
            return out;
        };

        let num_keys = usize::from(num_keys_raw);
        let ascii_bytes = ascii_params.as_bytes();

        for entry in entries.chunks_exact(4).take(num_keys) {
            let id = entry[0];
            let location = entry[1];
            let count = usize::from(entry[2]);
            let offset_or_val = entry[3];

            let value = match location {
                0 => Some(GeoKeyValue::Short(offset_or_val)),
                34736 => {
                    let start = usize::from(offset_or_val);
                    let end = start.saturating_add(count);
                    double_params
                        .get(start..end)
                        .filter(|vals| vals.iter().all(|v| v.is_finite()))
                        .map(|vals| GeoKeyValue::Double(vals.to_vec()))
                }
                34737 => {
                    let start = usize::from(offset_or_val);
                    let end = start.saturating_add(count);
                    // `tiff::Decoder::get_tag_ascii_string` truncates at the first NUL byte,
                    // so the final key's `count` may extend 1 byte past `ascii_bytes.len()`.
                    let slice = ascii_bytes.get(start..end).or_else(|| {
                        (start <= ascii_bytes.len() && end == ascii_bytes.len() + 1)
                            .then(|| &ascii_bytes[start..])
                    });
                    slice
                        .and_then(|bytes| std::str::from_utf8(bytes).ok())
                        .map(|s| {
                            GeoKeyValue::Ascii(s.trim_end_matches(['|', '\0']).trim().to_string())
                        })
                }
                _ => None,
            };

            if let Some(val) = value {
                match (id, &val) {
                    (1024, GeoKeyValue::Short(v)) => out.model_type = Some(*v),
                    (1025, GeoKeyValue::Short(2)) => out.raster_type = GeoRasterType::PixelIsPoint,
                    (1025, GeoKeyValue::Short(1)) => out.raster_type = GeoRasterType::PixelIsArea,
                    (2048, GeoKeyValue::Short(v)) => out.geographic_crs = Some(*v),
                    (3072, GeoKeyValue::Short(v)) => out.projected_crs = Some(*v),
                    _ => {}
                }
                out.keys.push((id, val));
            }
        }

        out
    }

    /// Look up a short (`u16`) GeoKey value by numeric ID.
    pub fn get_u16(&self, id: u16) -> Option<u16> {
        self.keys.iter().find_map(|(k, v)| match (k, v) {
            (key, GeoKeyValue::Short(val)) if *key == id => Some(*val),
            _ => None,
        })
    }

    /// Look up a `f64` scalar GeoKey value resolved from `GeoDoubleParamsTag`.
    pub fn get_f64(&self, id: u16) -> Option<f64> {
        self.keys.iter().find_map(|(k, v)| match (k, v) {
            (key, GeoKeyValue::Double(vals)) if *key == id => vals.first().copied(),
            _ => None,
        })
    }

    /// Look up an ASCII GeoKey string resolved from `GeoAsciiParamsTag`.
    pub fn get_ascii(&self, id: u16) -> Option<&str> {
        self.keys.iter().find_map(|(k, v)| match (k, v) {
            (key, GeoKeyValue::Ascii(s)) if *key == id => Some(s.as_str()),
            _ => None,
        })
    }
}

/// Projected raster binding: optional EPSG code, pixel-to-CRS affine transform,
/// and parsed GeoKey parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct GeoPlacement {
    pub epsg: Option<u32>,
    /// `x = a[0] * i + a[1] * j + a[2]`, `y = a[3] * i + a[4] * j + a[5]`,
    /// where `(i, j)` are pixel top-left corner coordinates (`j` increasing
    /// southward, with `PixelIsPoint` half-pixel shift already removed).
    pub affine: [f64; 6],
    /// Non-WGS84 geographic datum EPSG code if declared (e.g. Pulkovo-42 `4284`).
    pub foreign_datum_epsg: Option<u16>,
    pub geo_params: GeoKeyParams,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geokey_parse_respects_count_and_nul_stripped_ascii_tail() {
        // NumberOfKeys = 3, followed by trailing zero padding that must be ignored.
        let raw_keys = [
            1, 1, 0, 3, // header
            1024, 0, 1, 1, // ModelType = Projected
            3073, 34737, 8,
            4, // PCSCitation starting at 4, count 8 (includes stripped trailing NUL)
            3080, 34736, 1, 0, // ProjFalseOriginLong
            0, 0, 0, 0, // trailing padding ignored
        ];
        let doubles = [39.0_f64, f64::NAN];
        let ascii = "Foo|Zone 37"; // length 11; slice 4..12 extends 1 past len because NUL was stripped
        let parsed = GeoKeyParams::parse(&raw_keys, &doubles, ascii);
        assert_eq!(parsed.model_type, Some(1));
        assert_eq!(parsed.get_ascii(3073), Some("Zone 37"));
        assert_eq!(parsed.get_f64(3080), Some(39.0));
        assert_eq!(parsed.keys.len(), 3);
    }
}
