//! STAC `/search` JSON payload construction and Item/Asset/`raster:bands` parsing.

use serde_json::{Map, Value};
use veldmap_core::geo::{geojson_geometry_to_value, parse_geojson_geometry_value};

use super::types::{
    MAX_STAC_SEARCH_LIMIT, StacAsset, StacError, StacItem, StacRasterBandMeta, StacSearchQuery,
    StacSearchResult,
};

pub(super) fn build_search_payload(collection: &str, query: &StacSearchQuery) -> Value {
    let mut obj = Map::new();
    let col = collection.trim();
    if !col.is_empty() {
        obj.insert(
            "collections".to_owned(),
            Value::Array(vec![Value::String(col.to_owned())]),
        );
    }
    let limit = query.limit.clamp(1, MAX_STAC_SEARCH_LIMIT);
    obj.insert("limit".to_owned(), Value::from(limit as u64));

    if let Some(geom) = &query.intersects {
        obj.insert("intersects".to_owned(), geojson_geometry_to_value(geom));
    } else if let Some(bbox) = query.bbox
        && bbox.iter().all(|v| v.is_finite())
    {
        obj.insert(
            "bbox".to_owned(),
            serde_json::json!([bbox[0], bbox[1], bbox[2], bbox[3]]),
        );
    }

    if let Some(dt) = &query.datetime {
        let trimmed = dt.trim();
        if !trimmed.is_empty() {
            obj.insert("datetime".to_owned(), Value::String(trimmed.to_owned()));
        }
    }

    if let Some(max_cc) = query.max_cloud_cover
        && max_cc.is_finite()
    {
        let clamped = f64::from(max_cc.clamp(0.0, 100.0));
        obj.insert(
            "query".to_owned(),
            serde_json::json!({
                "eo:cloud_cover": { "lte": clamped }
            }),
        );
    }

    Value::Object(obj)
}

pub(super) fn parse_search_response(val: &Value) -> Result<StacSearchResult, StacError> {
    let obj = val.as_object().ok_or_else(|| {
        StacError::InvalidJson("STAC response root is not a JSON object".to_owned())
    })?;
    let matched = obj
        .get("numberMatched")
        .and_then(Value::as_u64)
        .or_else(|| {
            obj.get("context")
                .and_then(|c| c.get("matched"))
                .and_then(Value::as_u64)
        });
    let features = obj
        .get("features")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            StacError::InvalidJson("STAC response missing 'features' array".to_owned())
        })?;

    let mut items = Vec::with_capacity(features.len());
    for feat in features {
        if let Some(item) = parse_stac_item(feat) {
            items.push(item);
        }
    }
    Ok(StacSearchResult { items, matched })
}

/// Parse an RFC 7946 §5 2D (`[west, south, east, north]`) or 3D
/// (`[west, south, min_z, east, north, max_z]`) bounding box into finite 2D
/// WGS84 coordinates.
fn parse_item_bbox(obj: &Map<String, Value>) -> Option<[f64; 4]> {
    let arr = obj.get("bbox").and_then(Value::as_array)?;
    let coords = match arr.len() {
        4 => [
            arr[0].as_f64()?,
            arr[1].as_f64()?,
            arr[2].as_f64()?,
            arr[3].as_f64()?,
        ],
        6 => [
            arr[0].as_f64()?,
            arr[1].as_f64()?,
            arr[3].as_f64()?,
            arr[4].as_f64()?,
        ],
        _ => return None,
    };
    coords.iter().all(|v| v.is_finite()).then_some(coords)
}

fn parse_mgrs_tile(props: &Map<String, Value>) -> Option<String> {
    props
        .get("grid:code")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .or_else(|| {
            let zone = props.get("mgrs:utm_zone").and_then(Value::as_u64)?;
            let lat_band = props.get("mgrs:latitude_band").and_then(Value::as_str)?;
            let sq = props.get("mgrs:grid_square").and_then(Value::as_str)?;
            Some(format!("{zone:02}{lat_band}{sq}"))
        })
        .or_else(|| {
            props
                .get("s2:mgrs_tile")
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
}

fn infer_processing_level(
    props: Option<&Map<String, Value>>,
    collection: Option<&str>,
) -> Option<String> {
    props
        .and_then(|p| {
            ["s2:product_type", "processing:level", "s2:processing_level"]
                .into_iter()
                .find_map(|key| {
                    p.get(key)
                        .and_then(Value::as_str)
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                        .map(str::to_owned)
                })
        })
        .or_else(|| {
            collection.and_then(|c| {
                let lower = c.to_ascii_lowercase();
                if lower.contains("l2a") {
                    Some("L2A (BOA)".to_owned())
                } else if lower.contains("l1c") {
                    Some("L1C (TOA)".to_owned())
                } else {
                    None
                }
            })
        })
}

fn parse_stac_item(val: &Value) -> Option<StacItem> {
    let obj = val.as_object()?;
    let id = obj.get("id")?.as_str()?.to_owned();
    let collection = obj
        .get("collection")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let props = obj.get("properties").and_then(Value::as_object);

    let datetime = props
        .and_then(|p| {
            p.get("datetime")
                .and_then(Value::as_str)
                .or_else(|| p.get("start_datetime").and_then(Value::as_str))
        })
        .map(str::to_owned);
    let cloud_cover = props
        .and_then(|p| p.get("eo:cloud_cover"))
        .and_then(Value::as_f64)
        .map(|v| v as f32);
    let platform = props
        .and_then(|p| p.get("platform"))
        .and_then(Value::as_str)
        .map(str::to_owned);
    let mgrs_tile = props.and_then(parse_mgrs_tile);
    let processing_level = infer_processing_level(props, collection.as_deref());
    let bbox = parse_item_bbox(obj);
    let geometry = obj
        .get("geometry")
        .filter(|g| !g.is_null())
        .and_then(|g| parse_geojson_geometry_value(g).ok().flatten());

    let mut assets = Vec::new();
    if let Some(asset_map) = obj.get("assets").and_then(Value::as_object) {
        for (key, a_val) in asset_map {
            if let Some(asset) = parse_stac_asset(key, a_val) {
                assets.push(asset);
            }
        }
    }
    assets.sort_by_key(|a| (asset_sort_rank(&a.key), a.key.clone()));

    Some(StacItem {
        id,
        collection,
        datetime,
        cloud_cover,
        platform,
        mgrs_tile,
        processing_level,
        bbox,
        geometry,
        assets,
    })
}

fn asset_sort_rank(key: &str) -> u8 {
    match key.to_ascii_lowercase().as_str() {
        "visual" | "tci" => 0,
        "data" => 1,
        "red" | "b04" => 2,
        "green" | "b03" => 3,
        "blue" | "b02" => 4,
        "nir" | "b08" => 5,
        "nir08" | "b8a" => 6,
        "swir16" | "b11" => 7,
        "swir22" | "b12" => 8,
        "scl" => 9,
        _ => 20,
    }
}

fn parse_stac_asset(key: &str, val: &Value) -> Option<StacAsset> {
    let obj = val.as_object()?;
    let href = obj.get("href")?.as_str()?.to_owned();
    let title = obj.get("title").and_then(Value::as_str).map(str::to_owned);
    let media_type = obj.get("type").and_then(Value::as_str).map(str::to_owned);
    let roles = obj
        .get("roles")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(str::to_owned))
                .collect()
        })
        .unwrap_or_default();
    let eo_bands = obj
        .get("eo:bands")
        .or_else(|| obj.get("bands"))
        .and_then(Value::as_array)
        .map(|arr| {
            let mut out = Vec::new();
            for b in arr {
                for field in ["eo:common_name", "common_name", "name"] {
                    if let Some(s) = b.get(field).and_then(Value::as_str).map(str::trim)
                        && !s.is_empty()
                        && !out
                            .iter()
                            .any(|existing: &String| existing.eq_ignore_ascii_case(s))
                    {
                        out.push(s.to_owned());
                    }
                }
            }
            out
        })
        .unwrap_or_default();
    let gsd = obj.get("gsd").and_then(Value::as_f64);
    let raster_bands = obj
        .get("raster:bands")
        .and_then(Value::as_array)
        .filter(|arr| !arr.is_empty())
        .or_else(|| obj.get("bands").and_then(Value::as_array))
        .map(|arr| arr.iter().filter_map(parse_raster_band_meta).collect())
        .unwrap_or_default();

    Some(StacAsset {
        key: key.to_owned(),
        title,
        href,
        media_type,
        roles,
        eo_bands,
        gsd,
        raster_bands,
    })
}

fn parse_raster_band_meta(val: &Value) -> Option<StacRasterBandMeta> {
    let obj = val.as_object()?;
    let nodata = obj
        .get("nodata")
        .or_else(|| obj.get("raster:nodata"))
        .and_then(|v| {
            v.as_f64()
                .or_else(|| v.as_str().and_then(|s| s.trim().parse::<f64>().ok()))
        })
        .filter(|v| v.is_finite() || v.is_nan());
    Some(StacRasterBandMeta {
        scale: obj
            .get("scale")
            .or_else(|| obj.get("raster:scale"))
            .and_then(Value::as_f64),
        offset: obj
            .get("offset")
            .or_else(|| obj.get("raster:offset"))
            .and_then(Value::as_f64),
        nodata,
        data_type: obj
            .get("data_type")
            .or_else(|| obj.get("raster:data_type"))
            .and_then(Value::as_str)
            .map(str::to_owned),
        spatial_resolution: obj
            .get("spatial_resolution")
            .or_else(|| obj.get("raster:spatial_resolution"))
            .and_then(Value::as_f64),
        unit: obj
            .get("unit")
            .or_else(|| obj.get("raster:unit"))
            .and_then(Value::as_str)
            .map(str::to_owned),
    })
}

#[cfg(test)]
#[path = "parse_tests.rs"]
mod tests;
