//! GeoTIFF georeference and `GeoKeyDirectoryTag` (`34735`–`34737`) parsing.

use tiff::decoder::Decoder;
use tiff::tags::Tag;
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
const GEOKEY_PROJECTED_CS_TYPE: u16 = 3072;

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
    let epsg = match geo_params.projected_crs {
        Some(code) if !UNSAID_GEOKEY_CODES.contains(&code) => Some(u32::from(code)),
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
        affine,
        geo_params: Some(geo_params.clone()),
    })
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
