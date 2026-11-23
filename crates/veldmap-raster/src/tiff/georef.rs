//! GeoTIFF / GDAL georeference, `GeoKeyDirectoryTag` (`34735`–`34737`), and
//! nodata tag parsing.

use tiff::decoder::Decoder;
use tiff::tags::Tag;
use veldmap_core::io::BufferedByteReader;
use veldmap_core::sync::CancelToken;

use crate::RasterError;
use crate::pipeline::band::{
    GeoKeyParams, GeoKeyValue, GeoPlacement, GeoRasterType, GeoTie, is_valid_lat_lon,
};

use super::tags::{
    is_missing_tag, map_tiff_error, read_tag_ascii_opt, read_tag_f64_vec_opt, read_tag_u16_vec_opt,
};

/// Parse raw `GeoKeyDirectoryTag` (`34735`), `GeoDoubleParamsTag` (`34736`),
/// and `GeoAsciiParamsTag` (`34737`) payloads.
pub(crate) fn parse_geokeys(
    raw_keys: &[u16],
    double_params: &[f64],
    ascii_params: &str,
) -> GeoKeyParams {
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
                    .map(|s| GeoKeyValue::Ascii(s.trim_end_matches(['|', '\0']).trim().to_string()))
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

pub(super) fn parse_gdal_nodata(
    decoder: &mut Decoder<BufferedByteReader>,
    cancel: &CancelToken,
) -> Result<Option<f64>, RasterError> {
    let raw = match decoder.get_tag_ascii_string(Tag::GdalNodata) {
        Ok(s) => s,
        Err(err) if is_missing_tag(&err) => return Ok(None),
        Err(_) if cancel.is_cancelled() => return Err(RasterError::Cancelled),
        Err(err) if matches!(err, tiff::TiffError::IoError(_)) => {
            return Err(map_tiff_error(err, cancel));
        }
        Err(_) => return Ok(None),
    };
    let trimmed = raw.trim_matches(|c: char| c == '\0' || c.is_whitespace());
    if trimmed.eq_ignore_ascii_case("nan") {
        return Ok(Some(f64::NAN));
    }
    Ok(trimmed.parse::<f64>().ok().filter(|v| v.is_finite()))
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

pub(super) fn foreign_datum(geo_params: &GeoKeyParams) -> Option<u16> {
    const WGS84: [u16; 3] = [4326, 4979, 4055];
    const UNSAID: [u16; 2] = [0, 32767];
    if let Some(code) = geo_params
        .geographic_crs
        .filter(|code| !WGS84.contains(code) && !UNSAID.contains(code))
    {
        return Some(code);
    }
    // Check GeogGeodeticDatumGeoKey (2050): 6326 = WGS 84.
    if let Some(datum) = geo_params
        .get_u16(2050)
        .filter(|code| *code != 6326 && !UNSAID.contains(code))
    {
        return Some(datum);
    }
    // Pulkovo 1942 Gauss-Kruger zones (EPSG:28401..=28499) imply Pulkovo-42 (4284)
    // even when GeoKey 2048 is omitted.
    if matches!(geo_params.projected_crs, Some(28400..=28499)) {
        return Some(4284);
    }
    None
}

fn usable_step(scale: &[f64]) -> bool {
    matches!(scale.get(..2), Some([x, y]) if x.is_finite() && y.is_finite() && *x != 0.0 && *y != 0.0)
}

pub(super) fn affine_from_matrix(matrix: &[f64], half: f64) -> Option<[f64; 6]> {
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

pub(super) fn geo_placement(
    geo_params: &GeoKeyParams,
    points: &[f64],
    scale: &[f64],
    matrix: &[f64],
    width: u32,
    height: u32,
) -> Option<GeoPlacement> {
    if geo_params.model_type != Some(1) || points.len() > 6 {
        return None;
    }
    let epsg = match geo_params.projected_crs {
        Some(code) if code != 0 && code != 32767 => Some(u32::from(code)),
        _ => return None,
    };

    let half = geo_params.raster_type.half_pixel_shift();
    let affine = match (points.get(..6), usable_step(scale)) {
        (Some(tie), true) => [
            scale[0],
            0.0,
            tie[3] - (tie[0] + half) * scale[0],
            0.0,
            -scale[1],
            tie[4] + (tie[1] + half) * scale[1],
        ],
        _ => affine_from_matrix(matrix, half)?,
    };

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
        geo_params: geo_params.clone(),
    })
}

pub(super) fn geo_ties(
    geo_params: &GeoKeyParams,
    points: &[f64],
    scale: &[f64],
    matrix: &[f64],
    width: u32,
    height: u32,
) -> Vec<GeoTie> {
    if geo_params.model_type != Some(2) {
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

    if let (Some(tie), true) = (points.get(..6), usable_step(scale)) {
        let x0 = tie[3] - (tie[0] + half) * scale[0];
        let y0 = tie[4] + (tie[1] + half) * scale[1];
        let (right, bottom) = (f64::from(width), f64::from(height));
        return finite_ties(vec![
            GeoTie {
                px: 0.0,
                py: 0.0,
                lon: x0,
                lat: y0,
            },
            GeoTie {
                px: right,
                py: 0.0,
                lon: x0 + right * scale[0],
                lat: y0,
            },
            GeoTie {
                px: 0.0,
                py: bottom,
                lon: x0,
                lat: y0 - bottom * scale[1],
            },
            GeoTie {
                px: right,
                py: bottom,
                lon: x0 + right * scale[0],
                lat: y0 - bottom * scale[1],
            },
        ]);
    }

    if let Some(a) = affine_from_matrix(matrix, half) {
        let (right, bottom) = (f64::from(width), f64::from(height));
        let place = |px: f64, py: f64| GeoTie {
            px,
            py,
            lon: a[0] * px + a[1] * py + a[2],
            lat: a[3] * px + a[4] * py + a[5],
        };
        return finite_ties(vec![
            place(0.0, 0.0),
            place(right, 0.0),
            place(0.0, bottom),
            place(right, bottom),
        ]);
    }

    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geokey_parse_respects_count_and_nul_stripped_ascii_tail() {
        let raw_keys = [
            1, 1, 0, 3, // header
            1024, 0, 1, 1, // ModelType = Projected
            3073, 34737, 8, 4, // PCSCitation starting at 4, count 8 (stripped NUL)
            3080, 34736, 1, 0, // ProjFalseOriginLong
            0, 0, 0, 0, // trailing padding ignored
        ];
        let doubles = [39.0_f64, f64::NAN];
        let ascii = "Foo|Zone 37";
        let parsed = parse_geokeys(&raw_keys, &doubles, ascii);
        assert_eq!(parsed.model_type, Some(1));
        assert_eq!(parsed.get_ascii(3073), Some("Zone 37"));
        assert_eq!(parsed.get_f64(3080), Some(39.0));
        assert_eq!(parsed.keys.len(), 3);
    }
}
