//! GeoTIFF georeference and `GeoKeyDirectoryTag` (`34735`–`34737`) parsing.

use std::sync::Arc;

use tiff::decoder::Decoder;
use tiff::tags::Tag;
use veldmap_core::geo::{
    CrsSpec, Datum, DatumWarning, Ellipsoid, LinearUnit, Proj4Projection, Projection, System,
    WGS84, WGS84_DATUM, datum_from_towgs84, ellipsoid_from_epsg, known_datum_from_code,
};
use veldmap_core::io::BufferedByteReader;
use veldmap_core::sync::CancelToken;

use crate::RasterError;
use crate::pipeline::metadata::{
    GeoKeyParams, GeoKeyValue, GeoPlacement, GeoRasterType, GeoTie, is_valid_lat_lon,
};

use super::tags::{read_tag_ascii_opt, read_tag_f64_vec_opt, read_tag_u16_vec_opt};

const GEOKEY_LOCATION_SHORT: u16 = 0;
const TAG_GEO_DOUBLE_PARAMS: u16 = 34736;
const TAG_GEO_ASCII_PARAMS: u16 = 34737;

const GEOKEY_MODEL_TYPE: u16 = 1024;
const GEOKEY_RASTER_TYPE: u16 = 1025;
const GEOKEY_GEOGRAPHIC_TYPE: u16 = 2048;
const GEOKEY_GEOG_GEODETIC_DATUM: u16 = 2050;
const GEOKEY_GEOG_ELLIPSOID: u16 = 2056;
const GEOKEY_GEOG_SEMI_MAJOR_AXIS: u16 = 2057;
const GEOKEY_GEOG_SEMI_MINOR_AXIS: u16 = 2058;
const GEOKEY_GEOG_INV_FLATTENING: u16 = 2059;
const GEOKEY_GEOG_TOWGS84: u16 = 2062;
const GEOKEY_PROJECTED_CS_TYPE: u16 = 3072;
const GEOKEY_PCS_CITATION: u16 = 3073;
const GEOKEY_PROJ_COORD_TRANS: u16 = 3075;
const GEOKEY_PROJ_LINEAR_UNITS: u16 = 3076;
const GEOKEY_PROJ_LINEAR_UNIT_SIZE: u16 = 3077;
const GEOKEY_PROJ_STD_PARALLEL1: u16 = 3078;
const GEOKEY_PROJ_STD_PARALLEL2: u16 = 3079;
const GEOKEY_PROJ_NAT_ORIGIN_LONG: u16 = 3080;
const GEOKEY_PROJ_NAT_ORIGIN_LAT: u16 = 3081;
const GEOKEY_PROJ_FALSE_EASTING: u16 = 3082;
const GEOKEY_PROJ_FALSE_NORTHING: u16 = 3083;
const GEOKEY_PROJ_CENTER_LONG: u16 = 3088;
const GEOKEY_PROJ_CENTER_LAT: u16 = 3089;
const GEOKEY_PROJ_SCALE_AT_NAT_ORIGIN: u16 = 3092;
const GEOKEY_PROJ_SCALE_AT_CENTER: u16 = 3093;
const GEOKEY_PROJ_STRAIGHT_VERT_POLE_LONG: u16 = 3095;

const CT_TRANSVERSE_MERCATOR: u16 = 1;
const CT_MERCATOR: u16 = 7;
const CT_LAMBERT_CONF_CONIC_2SP: u16 = 8;
const CT_LAMBERT_CONF_CONIC_1SP: u16 = 9;
const CT_LAMBERT_AZIMUTHAL_EQUAL_AREA: u16 = 10;
const CT_ALBERS_EQUAL_AREA: u16 = 11;
const CT_POLAR_STEREOGRAPHIC: u16 = 15;

const MODEL_TYPE_PROJECTED: u16 = 1;
const MODEL_TYPE_GEOGRAPHIC: u16 = 2;

const RASTER_TYPE_PIXEL_IS_AREA: u16 = 1;
const RASTER_TYPE_PIXEL_IS_POINT: u16 = 2;

const GEOKEY_UNDEFINED: u16 = 0;
const GEOKEY_USER_DEFINED: u16 = 32767;
const UNSAID_GEOKEY_CODES: [u16; 2] = [GEOKEY_UNDEFINED, GEOKEY_USER_DEFINED];

const WGS84_GEOGRAPHIC_CODES: [u16; 3] = [4326, 4979, 4055];
const EPSG_WGS84_DATUM: u16 = 6326;
const EPSG_PULKOVO_1942: u16 = 4284;
const PULKOVO_42_GK_START: u16 = 28400;
const PULKOVO_42_GK_END: u16 = 28499;

/// Parse raw `GeoKeyDirectoryTag` (`34735`), `GeoDoubleParamsTag` (`34736`),
/// and `GeoAsciiParamsTag` (`34737`) payloads.
fn parse_geokeys(raw_keys: &[u16], double_params: &[f64], ascii_params: &str) -> GeoKeyParams {
    let mut out = GeoKeyParams::default();

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
                // `tiff::Decoder::get_tag_ascii_string` truncates at the first NUL byte,
                // so the final key's `count` may extend 1 byte past `ascii_bytes.len()`.
                let slice = ascii_bytes.get(start..end).or_else(|| {
                    (start <= ascii_bytes.len() && end == ascii_bytes.len() + 1)
                        .then(|| &ascii_bytes[start..])
                });
                slice
                    .and_then(|bytes| std::str::from_utf8(bytes).ok())
                    .map(|s| GeoKeyValue::Ascii(s.trim_end_matches(['|', '\0']).trim().to_string()))
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

pub(super) struct ParsedGeoref {
    pub(super) ties: Vec<GeoTie>,
    pub(super) placement: Option<GeoPlacement>,
    pub(super) foreign_datum_epsg: Option<u16>,
    pub(super) trouble: Option<String>,
}

pub(super) fn georef(
    decoder: &mut Decoder<BufferedByteReader>,
    width: u32,
    height: u32,
    cancel: &CancelToken,
) -> Result<ParsedGeoref, RasterError> {
    let raw_keys = read_tag_u16_vec_opt(decoder, Tag::GeoKeyDirectoryTag, cancel)?;
    let double_params = read_tag_f64_vec_opt(decoder, Tag::GeoDoubleParamsTag, cancel)?;
    let ascii_params = read_tag_ascii_opt(decoder, Tag::GeoAsciiParamsTag, cancel)?;
    let geo_params = parse_geokeys(&raw_keys, &double_params, &ascii_params);

    let points = read_tag_f64_vec_opt(decoder, Tag::ModelTiepointTag, cancel)?;
    let scale = read_tag_f64_vec_opt(decoder, Tag::ModelPixelScaleTag, cancel)?;
    let matrix = read_tag_f64_vec_opt(decoder, Tag::ModelTransformationTag, cancel)?;

    let foreign_datum_epsg = foreign_datum(&geo_params);
    let placement = geo_placement(&geo_params, &points, &scale, &matrix, width, height);
    let ties = geo_ties(&geo_params, &points, &scale, &matrix, width, height);

    let carries = !points.is_empty() || !scale.is_empty() || !matrix.is_empty();
    let taken = !ties.is_empty() || placement.is_some();
    let trouble = (carries && !taken).then(|| {
        let named = |opt: Option<u16>| opt.map_or_else(|| "unnamed".to_string(), |v| v.to_string());
        format!(
            "raster carries georeference tags, but could not be bound: model {}, projected {}, tie points {}",
            named(geo_params.model_type),
            named(geo_params.projected_crs),
            points.len() / 6
        )
    });

    Ok(ParsedGeoref {
        ties,
        placement,
        foreign_datum_epsg,
        trouble,
    })
}

fn foreign_datum(geo_params: &GeoKeyParams) -> Option<u16> {
    if let Some(code) = geo_params.geographic_crs.filter(|code| {
        !WGS84_GEOGRAPHIC_CODES.contains(code) && !UNSAID_GEOKEY_CODES.contains(code)
    }) {
        return Some(code);
    }
    if let Some(datum) = geo_params
        .get_u16(GEOKEY_GEOG_GEODETIC_DATUM)
        .filter(|code| *code != EPSG_WGS84_DATUM && !UNSAID_GEOKEY_CODES.contains(code))
    {
        return Some(datum);
    }
    // Pulkovo 1942 Gauss-Kruger zones imply Pulkovo-42 even when GeoKey 2048 is omitted.
    if matches!(
        geo_params.projected_crs,
        Some(PULKOVO_42_GK_START..=PULKOVO_42_GK_END)
    ) {
        return Some(EPSG_PULKOVO_1942);
    }
    None
}

fn usable_step(scale: &[f64]) -> bool {
    matches!(scale.get(..2), Some([x, y]) if x.is_finite() && y.is_finite() && *x != 0.0 && *y != 0.0)
}

fn affine_from_matrix(matrix: &[f64], half: f64) -> Option<[f64; 6]> {
    if matrix.len() != 16 {
        return None;
    }
    let a = matrix.get(..8)?;
    if !a.iter().all(|v| v.is_finite()) || a[0] * a[5] - a[1] * a[4] == 0.0 {
        return None;
    }
    Some([
        a[0],
        a[1],
        a[3] - (a[0] + a[1]) * half,
        a[4],
        a[5],
        a[7] - (a[4] + a[5]) * half,
    ])
}

fn affine_from_tags(points: &[f64], scale: &[f64], matrix: &[f64], half: f64) -> Option<[f64; 6]> {
    // Prefer `ModelTransformationTag` when it carries non-zero rotation/shear
    // (which `ModelTiepointTag + ModelPixelScaleTag` cannot represent).
    if let Some(mat_affine) = affine_from_matrix(matrix, half)
        && (mat_affine[1] != 0.0 || mat_affine[3] != 0.0)
    {
        return Some(mat_affine);
    }
    if let (Some(tie), true) = (points.get(..6), usable_step(scale)) {
        let affine = [
            scale[0],
            0.0,
            tie[3] - (tie[0] + half) * scale[0],
            0.0,
            -scale[1],
            tie[4] + (tie[1] + half) * scale[1],
        ];
        if affine.iter().all(|v| v.is_finite()) {
            return Some(affine);
        }
    }
    affine_from_matrix(matrix, half)
}

fn geo_placement(
    geo_params: &GeoKeyParams,
    points: &[f64],
    scale: &[f64],
    matrix: &[f64],
    width: u32,
    height: u32,
) -> Option<GeoPlacement> {
    if geo_params.model_type != Some(MODEL_TYPE_PROJECTED) || points.len() > 6 {
        return None;
    }
    let (epsg, custom_crs, datum_warning) = match geo_params.projected_crs {
        Some(code) if !UNSAID_GEOKEY_CODES.contains(&code) => (Some(u32::from(code)), None, None),
        Some(GEOKEY_USER_DEFINED) | Some(GEOKEY_UNDEFINED) | None => {
            let (custom, warning) = build_user_defined_projected_crs(geo_params)?;
            (None, Some(custom), warning)
        }
        _ => return None,
    };

    let half = geo_params.raster_type.half_pixel_shift();
    let affine = affine_from_tags(points, scale, matrix, half)?;

    let (right, bottom) = (f64::from(width), f64::from(height));
    let corner = [
        affine[0] * right + affine[1] * bottom + affine[2],
        affine[3] * right + affine[4] * bottom + affine[5],
    ];
    if !affine.iter().chain(corner.iter()).all(|v| v.is_finite()) {
        return None;
    }

    Some(GeoPlacement {
        epsg,
        custom_crs,
        datum_warning,
        affine,
        geo_params: Some(geo_params.clone()),
    })
}

fn resolve_user_datum(geo_params: &GeoKeyParams) -> (Datum, Option<DatumWarning>) {
    let base_ellipsoid = geo_params
        .get_u16(GEOKEY_GEOG_ELLIPSOID)
        .and_then(ellipsoid_from_epsg)
        .or_else(|| {
            let a = geo_params.get_f64(GEOKEY_GEOG_SEMI_MAJOR_AXIS)?;
            if !(a.is_finite() && a > 0.0) {
                return None;
            }
            if let Some(rf) = geo_params.get_f64(GEOKEY_GEOG_INV_FLATTENING)
                && rf.is_finite()
                && rf > 1.0
            {
                return Some(Ellipsoid {
                    semi_major_m: a,
                    flattening: 1.0 / rf,
                });
            }
            if let Some(b) = geo_params.get_f64(GEOKEY_GEOG_SEMI_MINOR_AXIS)
                && b.is_finite()
                && b > 0.0
                && b <= a
            {
                return Some(Ellipsoid {
                    semi_major_m: a,
                    flattening: (a - b) / a,
                });
            }
            None
        });

    let datum_code = geo_params
        .geographic_crs
        .or_else(|| geo_params.get_u16(GEOKEY_GEOG_GEODETIC_DATUM));
    let known = datum_code.and_then(|code| known_datum_from_code(u32::from(code)));

    let towgs84_vals = geo_params.keys.iter().find_map(|(k, v)| match (k, v) {
        (id, GeoKeyValue::Double(vals)) if *id == GEOKEY_GEOG_TOWGS84 => Some(vals.as_slice()),
        _ => None,
    });
    if let Some(vals) = towgs84_vals
        && (vals.len() == 3 || vals.len() == 7)
        && vals.iter().all(|v| v.is_finite())
    {
        // Preserve ellipsoid from explicit GeoKey 2056/2057-2059, or fall back
        // to the ellipsoid of the referenced datum/geographic CRS (2048/2050)
        // before defaulting to WGS84.
        let ell = base_ellipsoid
            .or_else(|| known.as_ref().map(|(d, _)| d.ellipsoid))
            .unwrap_or(WGS84);
        let p7 = [
            vals[0],
            vals[1],
            vals[2],
            vals.get(3).copied().unwrap_or(0.0),
            vals.get(4).copied().unwrap_or(0.0),
            vals.get(5).copied().unwrap_or(0.0),
            vals.get(6).copied().unwrap_or(0.0),
        ];
        return (datum_from_towgs84(ell, p7), None);
    }

    if let Some((datum, warning)) = known {
        return (datum, warning);
    }

    if let Some(ell) = base_ellipsoid {
        let non_wgs84 = (ell.semi_major_m - WGS84.semi_major_m).abs() > 1.0
            || (ell.flattening - WGS84.flattening).abs() > 1e-6;
        let warning = non_wgs84.then(|| DatumWarning::UnshiftedForeignDatum {
            label: "user-defined ellipsoid".to_string(),
        });
        return (
            Datum {
                ellipsoid: ell,
                ..WGS84_DATUM
            },
            warning,
        );
    }

    if let Some(code) = datum_code
        && !UNSAID_GEOKEY_CODES.contains(&code)
        && !WGS84_GEOGRAPHIC_CODES.contains(&code)
        && code != EPSG_WGS84_DATUM
    {
        return (
            WGS84_DATUM,
            Some(DatumWarning::UnshiftedForeignDatum {
                label: format!("EPSG:{code}"),
            }),
        );
    }

    (WGS84_DATUM, None)
}

fn resolve_linear_unit_m(geo_params: &GeoKeyParams) -> f64 {
    if let Some(code) = geo_params.get_u16(GEOKEY_PROJ_LINEAR_UNITS) {
        if let Some(unit) = LinearUnit::from_geotiff_code(code) {
            return unit.to_metres();
        }
        if code == GEOKEY_USER_DEFINED
            && let Some(size) = geo_params.get_f64(GEOKEY_PROJ_LINEAR_UNIT_SIZE)
            && size.is_finite()
            && size > 0.0
        {
            return size;
        }
    }
    1.0
}

fn build_user_defined_projected_crs(
    geo_params: &GeoKeyParams,
) -> Option<(CrsSpec, Option<DatumWarning>)> {
    let ct = geo_params.get_u16(GEOKEY_PROJ_COORD_TRANS)?;
    let unit_m = resolve_linear_unit_m(geo_params);
    let (datum, warning) = resolve_user_datum(geo_params);
    let lon_0 = geo_params
        .get_f64(GEOKEY_PROJ_NAT_ORIGIN_LONG)
        .or_else(|| geo_params.get_f64(GEOKEY_PROJ_CENTER_LONG))
        .unwrap_or(0.0);
    let lat_0 = geo_params
        .get_f64(GEOKEY_PROJ_NAT_ORIGIN_LAT)
        .or_else(|| geo_params.get_f64(GEOKEY_PROJ_CENTER_LAT))
        .unwrap_or(0.0);
    let scale = geo_params
        .get_f64(GEOKEY_PROJ_SCALE_AT_NAT_ORIGIN)
        .or_else(|| geo_params.get_f64(GEOKEY_PROJ_SCALE_AT_CENTER))
        .unwrap_or(1.0);
    // GeoTIFF false easting/northing are stored in the CRS linear unit.
    let false_easting_m = geo_params.get_f64(GEOKEY_PROJ_FALSE_EASTING).unwrap_or(0.0) * unit_m;
    let false_northing_m = geo_params
        .get_f64(GEOKEY_PROJ_FALSE_NORTHING)
        .unwrap_or(0.0)
        * unit_m;

    if ct == CT_TRANSVERSE_MERCATOR {
        if !(lon_0.is_finite()
            && lat_0.is_finite()
            && scale.is_finite()
            && scale > 0.0
            && false_easting_m.is_finite()
            && false_northing_m.is_finite())
        {
            return None;
        }
        let sys = System {
            datum,
            central_meridian_deg: lon_0,
            origin_lat_deg: lat_0,
            scale,
            false_easting_m,
            false_northing_m,
            unit_to_metres: unit_m,
        };
        return Some((CrsSpec::Projected(Projection::Transverse(sys)), warning));
    }

    // Build a PROJ.4 string for other user-defined GeoTIFF coordinate transforms.
    let proj_name = match ct {
        CT_MERCATOR => "merc",
        CT_LAMBERT_CONF_CONIC_2SP | CT_LAMBERT_CONF_CONIC_1SP => "lcc",
        CT_LAMBERT_AZIMUTHAL_EQUAL_AREA => "laea",
        CT_ALBERS_EQUAL_AREA => "aea",
        CT_POLAR_STEREOGRAPHIC => "stere",
        _ => return None,
    };
    let mut parts = vec![format!("+proj={proj_name}")];
    if ct == CT_POLAR_STEREOGRAPHIC {
        let pole_lon = geo_params
            .get_f64(GEOKEY_PROJ_STRAIGHT_VERT_POLE_LONG)
            .unwrap_or(lon_0);
        let lat_ts = geo_params
            .get_f64(GEOKEY_PROJ_STD_PARALLEL1)
            .or_else(|| geo_params.get_f64(GEOKEY_PROJ_NAT_ORIGIN_LAT))
            .unwrap_or(lat_0);
        let pole_lat = if lat_ts < 0.0 { -90.0 } else { 90.0 };
        parts.push(format!("+lat_0={pole_lat}"));
        parts.push(format!("+lon_0={pole_lon}"));
        if lat_ts.is_finite() && lat_ts.abs() < 90.0 {
            parts.push(format!("+lat_ts={lat_ts}"));
        } else {
            parts.push(format!("+k_0={scale}"));
        }
    } else {
        parts.push(format!("+lat_0={lat_0}"));
        parts.push(format!("+lon_0={lon_0}"));
        parts.push(format!("+k_0={scale}"));
    }
    parts.extend([
        format!("+x_0={false_easting_m}"),
        format!("+y_0={false_northing_m}"),
        format!("+to_meter={unit_m}"),
        format!("+a={}", datum.ellipsoid.semi_major_m),
        format!("+b={}", datum.ellipsoid.semi_minor_m()),
    ]);
    if ct != CT_POLAR_STEREOGRAPHIC {
        if let Some(lat1) = geo_params.get_f64(GEOKEY_PROJ_STD_PARALLEL1) {
            parts.push(format!("+lat_1={lat1}"));
        }
        if let Some(lat2) = geo_params.get_f64(GEOKEY_PROJ_STD_PARALLEL2) {
            parts.push(format!("+lat_2={lat2}"));
        }
    }
    if !datum.is_wgs84() {
        let sign = match datum.convention {
            veldmap_core::geo::HelmertConvention::CoordinateFrame => -1.0,
            veldmap_core::geo::HelmertConvention::PositionVector => 1.0,
        };
        parts.push(format!(
            "+towgs84={},{},{},{},{},{},{}",
            datum.shift_m[0],
            datum.shift_m[1],
            datum.shift_m[2],
            datum.rotate_sec[0] * sign,
            datum.rotate_sec[1] * sign,
            datum.rotate_sec[2] * sign,
            datum.scale_ppm
        ));
    }
    parts.push("+no_defs".to_string());
    let proj_str = parts.join(" ");
    let label = geo_params
        .get_ascii(GEOKEY_PCS_CITATION)
        .map(str::to_owned)
        .unwrap_or_else(|| format!("User-Defined ({proj_name})"));
    let p4 = Proj4Projection::from_proj_string(&proj_str, Some(label)).ok()?;
    let final_warning = warning.or_else(|| p4.datum_warning().cloned());
    Some((
        CrsSpec::Projected(Projection::Proj4(Arc::new(p4))),
        final_warning,
    ))
}

fn snap_deg_bound(v: f64, limit: f64) -> f64 {
    const EPS_DEG: f64 = 1e-6;
    if v > limit && v <= limit + EPS_DEG {
        limit
    } else if v < -limit && v >= -limit - EPS_DEG {
        -limit
    } else {
        v
    }
}

fn synthesize_affine_ties(a: [f64; 6], width: u32, height: u32, half: f64) -> Vec<GeoTie> {
    let (right, bottom) = (f64::from(width), f64::from(height));
    // For `PixelIsPoint`, sample ties at the outer pixel centers (matching
    // the explicit multi-tiepoint branch) so pole-touching global rasters
    // (`lat = ±90°`) and `[0°, 360°]` rasters stay within valid geodetic
    // bounds while `TieGrid` linearly extrapolates outer half-pixel edges.
    let (px0, px1) = if half > 0.0 && width > 1 {
        (half, right - half)
    } else {
        (0.0, right)
    };
    let (py0, py1) = if half > 0.0 && height > 1 {
        (half, bottom - half)
    } else {
        (0.0, bottom)
    };

    let dlon_x = (a[0] * (px1 - px0)).abs();
    let dlon_y = (a[1] * (py1 - py0)).abs();
    // Guard against corrupt TIFFs before subdividing; valid longitudes are
    // within `[-360°, +360°]` (`span <= 720°`).
    if !dlon_x.is_finite() || !dlon_y.is_finite() || dlon_x > 720.0 || dlon_y > 720.0 {
        return Vec::new();
    }
    // Subdivide any axis whose longitude span exceeds `170°` so adjacent
    // lattice samples stay `< 180°` apart and `TieGrid::unwind` never wraps
    // wide (`180° < Δlon < 359°`) geographic rasters backwards.
    let x_steps = if dlon_x > 170.0 {
        (dlon_x / 90.0).ceil().max(2.0) as usize
    } else {
        1
    };
    let y_steps = if dlon_y > 170.0 {
        (dlon_y / 90.0).ceil().max(2.0) as usize
    } else {
        1
    };

    let mut ties = Vec::with_capacity((x_steps + 1) * (y_steps + 1));
    for iy in 0..=y_steps {
        let py = if iy == y_steps {
            py1
        } else {
            py0 + (py1 - py0) * (iy as f64 / y_steps as f64)
        };
        for ix in 0..=x_steps {
            let px = if ix == x_steps {
                px1
            } else {
                px0 + (px1 - px0) * (ix as f64 / x_steps as f64)
            };
            // Snap tiny IEEE-754 rounding noise near ±90° / ±360° without masking
            // genuine out-of-range coordinates (e.g. UTM metres mis-tagged as geo).
            let lon = snap_deg_bound(a[0] * px + a[1] * py + a[2], 360.0);
            let lat = snap_deg_bound(a[3] * px + a[4] * py + a[5], 90.0);
            ties.push(GeoTie { px, py, lon, lat });
        }
    }
    ties
}

fn geo_ties(
    geo_params: &GeoKeyParams,
    points: &[f64],
    scale: &[f64],
    matrix: &[f64],
    width: u32,
    height: u32,
) -> Vec<GeoTie> {
    if geo_params.model_type != Some(MODEL_TYPE_GEOGRAPHIC) {
        return Vec::new();
    }
    let half = geo_params.raster_type.half_pixel_shift();
    let finite_ties = |ties: Vec<GeoTie>| -> Vec<GeoTie> {
        let valid = ties
            .iter()
            .all(|t| t.px.is_finite() && t.py.is_finite() && is_valid_lat_lon(t.lat, t.lon));
        if valid { ties } else { Vec::new() }
    };

    if points.len() > 6 {
        let ties: Vec<GeoTie> = points
            .chunks_exact(6)
            .map(|tie| GeoTie {
                px: tie[0] + half,
                py: tie[1] + half,
                lon: tie[3],
                lat: tie[4],
            })
            .collect();
        return finite_ties(ties);
    }

    if let Some(a) = affine_from_tags(points, scale, matrix, half) {
        return finite_ties(synthesize_affine_ties(a, width, height, half));
    }

    Vec::new()
}

#[cfg(test)]
#[path = "georef_tests.rs"]
mod tests;
