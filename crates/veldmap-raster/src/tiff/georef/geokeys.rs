//! GeoTIFF GeoKey types, constants, and binary `GeoKeyDirectoryTag`
//! (`34735`–`34737`) decoder.

const TAG_GEO_DOUBLE_PARAMS: u16 = 34736;
const TAG_GEO_ASCII_PARAMS: u16 = 34737;

const GEOKEY_LOCATION_SHORT: u16 = 0;

const GEOKEY_MODEL_TYPE: u16 = 1024;
const GEOKEY_RASTER_TYPE: u16 = 1025;
const GEOKEY_GEOGRAPHIC_TYPE: u16 = 2048;
pub(super) const GEOKEY_GEOG_GEODETIC_DATUM: u16 = 2050;
pub(super) const GEOKEY_GEOG_ELLIPSOID: u16 = 2056;
pub(super) const GEOKEY_GEOG_SEMI_MAJOR_AXIS: u16 = 2057;
pub(super) const GEOKEY_GEOG_SEMI_MINOR_AXIS: u16 = 2058;
pub(super) const GEOKEY_GEOG_INV_FLATTENING: u16 = 2059;
pub(super) const GEOKEY_GEOG_TOWGS84: u16 = 2062;
const GEOKEY_PROJECTED_CS_TYPE: u16 = 3072;
pub(super) const GEOKEY_PCS_CITATION: u16 = 3073;
pub(super) const GEOKEY_PROJ_COORD_TRANS: u16 = 3075;
pub(super) const GEOKEY_PROJ_LINEAR_UNITS: u16 = 3076;
pub(super) const GEOKEY_PROJ_LINEAR_UNIT_SIZE: u16 = 3077;
pub(super) const GEOKEY_PROJ_STD_PARALLEL1: u16 = 3078;
pub(super) const GEOKEY_PROJ_STD_PARALLEL2: u16 = 3079;
pub(super) const GEOKEY_PROJ_NAT_ORIGIN_LONG: u16 = 3080;
pub(super) const GEOKEY_PROJ_NAT_ORIGIN_LAT: u16 = 3081;
pub(super) const GEOKEY_PROJ_FALSE_EASTING: u16 = 3082;
pub(super) const GEOKEY_PROJ_FALSE_NORTHING: u16 = 3083;
pub(super) const GEOKEY_PROJ_FALSE_ORIGIN_LONG: u16 = 3084;
pub(super) const GEOKEY_PROJ_FALSE_ORIGIN_LAT: u16 = 3085;
pub(super) const GEOKEY_PROJ_FALSE_ORIGIN_EASTING: u16 = 3086;
pub(super) const GEOKEY_PROJ_FALSE_ORIGIN_NORTHING: u16 = 3087;
pub(super) const GEOKEY_PROJ_CENTER_LONG: u16 = 3088;
pub(super) const GEOKEY_PROJ_CENTER_LAT: u16 = 3089;
pub(super) const GEOKEY_PROJ_CENTER_EASTING: u16 = 3090;
pub(super) const GEOKEY_PROJ_CENTER_NORTHING: u16 = 3091;
pub(super) const GEOKEY_PROJ_SCALE_AT_NAT_ORIGIN: u16 = 3092;
pub(super) const GEOKEY_PROJ_SCALE_AT_CENTER: u16 = 3093;
pub(super) const GEOKEY_PROJ_STRAIGHT_VERT_POLE_LONG: u16 = 3095;

pub(super) const CT_TRANSVERSE_MERCATOR: u16 = 1;
pub(super) const CT_MERCATOR: u16 = 7;
pub(super) const CT_LAMBERT_CONF_CONIC_2SP: u16 = 8;
pub(super) const CT_LAMBERT_CONF_CONIC_1SP: u16 = 9;
pub(super) const CT_LAMBERT_AZIMUTHAL_EQUAL_AREA: u16 = 10;
pub(super) const CT_ALBERS_EQUAL_AREA: u16 = 11;
pub(super) const CT_POLAR_STEREOGRAPHIC: u16 = 15;

pub(super) const MODEL_TYPE_PROJECTED: u16 = 1;
pub(super) const MODEL_TYPE_GEOGRAPHIC: u16 = 2;

const RASTER_TYPE_PIXEL_IS_AREA: u16 = 1;
const RASTER_TYPE_PIXEL_IS_POINT: u16 = 2;

pub(super) const GEOKEY_UNDEFINED: u16 = 0;
pub(super) const GEOKEY_USER_DEFINED: u16 = 32767;
pub(super) const UNSAID_GEOKEY_CODES: [u16; 2] = [GEOKEY_UNDEFINED, GEOKEY_USER_DEFINED];

pub(super) const WGS84_GEOGRAPHIC_CODES: [u16; 2] = [4326, 4979];
pub(super) const EPSG_UNIT_KILOMETRE: u16 = 9036;
pub(super) const EPSG_WGS84_DATUM: u16 = 6326;

/// Pixel coordinate reference convention (`GTRasterTypeGeoKey = 1025`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(super) enum GeoRasterType {
    /// Pixel coordinates denote the top-left corner of the pixel cell (`1`).
    #[default]
    PixelIsArea,
    /// Pixel coordinates denote the center of the pixel cell (`2`).
    PixelIsPoint,
}

impl GeoRasterType {
    /// Half-pixel shift added when converting `(px, py)` into top-left cell
    /// corner coordinates (`0.0` for `PixelIsArea`, `0.5` for `PixelIsPoint`).
    pub(super) const fn half_pixel_shift(self) -> f64 {
        match self {
            Self::PixelIsArea => 0.0,
            Self::PixelIsPoint => 0.5,
        }
    }
}

/// Parsed raw GeoTIFF key directory entry value.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum GeoKeyValue {
    Short(u16),
    Double(Vec<f64>),
    Ascii(String),
}

/// Parsed GeoTIFF key directory preserving resolved double/ASCII parameters
/// for downstream CRS resolution.
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct GeoKeyParams {
    pub(super) model_type: Option<u16>,
    pub(super) raster_type: GeoRasterType,
    pub(super) geographic_crs: Option<u16>,
    pub(super) projected_crs: Option<u16>,
    pub(super) keys: Vec<(u16, GeoKeyValue)>,
}

impl GeoKeyParams {
    /// Look up a short (`u16`) GeoKey value by numeric ID.
    pub(super) fn get_u16(&self, id: u16) -> Option<u16> {
        self.keys.iter().find_map(|(k, v)| match (k, v) {
            (key, GeoKeyValue::Short(val)) if *key == id => Some(*val),
            _ => None,
        })
    }

    /// Look up a `f64` scalar GeoKey value.
    pub(super) fn get_f64(&self, id: u16) -> Option<f64> {
        self.get_f64_slice(id)
            .and_then(|vals| vals.first().copied())
    }

    /// Look up a slice of `f64` GeoKey values (e.g. `GeogTOWGS84GeoKey`).
    pub(super) fn get_f64_slice(&self, id: u16) -> Option<&[f64]> {
        self.keys.iter().find_map(|(k, v)| match (k, v) {
            (key, GeoKeyValue::Double(vals)) if *key == id => Some(vals.as_slice()),
            _ => None,
        })
    }

    /// Look up an ASCII GeoKey string.
    pub(super) fn get_ascii(&self, id: u16) -> Option<&str> {
        self.keys.iter().find_map(|(k, v)| match (k, v) {
            (key, GeoKeyValue::Ascii(s)) if *key == id => Some(s.as_str()),
            _ => None,
        })
    }
}

/// Parse raw `GeoKeyDirectoryTag` (`34735`), `GeoDoubleParamsTag` (`34736`),
/// and `GeoAsciiParamsTag` (`34737`) raw byte payloads.
pub(super) fn parse_geokeys(
    raw_keys: &[u16],
    double_params: &[f64],
    ascii_bytes: &[u8],
) -> GeoKeyParams {
    let mut out = GeoKeyParams::default();

    let Some(&num_keys_raw) = raw_keys.get(3) else {
        return out;
    };
    let Some(entries) = raw_keys.get(4..) else {
        return out;
    };

    let num_keys = usize::from(num_keys_raw);

    for entry in entries.chunks_exact(4).take(num_keys) {
        let id = entry[0];
        let location = entry[1];
        let count = usize::from(entry[2]);
        let offset_or_val = entry[3];

        let value = match location {
            GEOKEY_LOCATION_SHORT => Some(GeoKeyValue::Short(offset_or_val)),
            TAG_GEO_DOUBLE_PARAMS => {
                let start = usize::from(offset_or_val);
                let end = start.saturating_add(count);
                double_params
                    .get(start..end)
                    .filter(|vals| vals.iter().all(|v| v.is_finite()))
                    .map(|vals| GeoKeyValue::Double(vals.to_vec()))
            }
            TAG_GEO_ASCII_PARAMS => {
                let start = usize::from(offset_or_val);
                let end = start.saturating_add(count);
                // `read_ascii_bytes_opt` truncates at the first NUL byte, so the
                // final key's `count` may extend 1 byte past `ascii_bytes.len()`.
                // Slice raw byte offsets BEFORE lossy UTF-8 conversion so a
                // single Latin-1 byte in an earlier key never shifts later keys.
                let slice = ascii_bytes.get(start..end).or_else(|| {
                    (start <= ascii_bytes.len() && end == ascii_bytes.len() + 1)
                        .then(|| &ascii_bytes[start..])
                });
                slice.map(|bytes| {
                    let s = String::from_utf8_lossy(bytes);
                    GeoKeyValue::Ascii(s.trim_end_matches(['|', '\0']).trim().to_string())
                })
            }
            _ => None,
        };

        if let Some(val) = value {
            match (id, &val) {
                (GEOKEY_MODEL_TYPE, GeoKeyValue::Short(v)) => out.model_type = Some(*v),
                (GEOKEY_RASTER_TYPE, GeoKeyValue::Short(RASTER_TYPE_PIXEL_IS_POINT)) => {
                    out.raster_type = GeoRasterType::PixelIsPoint;
                }
                (GEOKEY_RASTER_TYPE, GeoKeyValue::Short(RASTER_TYPE_PIXEL_IS_AREA)) => {
                    out.raster_type = GeoRasterType::PixelIsArea;
                }
                (GEOKEY_GEOGRAPHIC_TYPE, GeoKeyValue::Short(v)) => out.geographic_crs = Some(*v),
                (GEOKEY_PROJECTED_CS_TYPE, GeoKeyValue::Short(v)) => out.projected_crs = Some(*v),
                _ => {}
            }
            out.keys.push((id, val));
        }
    }

    out
}
