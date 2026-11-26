//! STAC `/search` JSON payload construction and Item/Asset/`raster:bands` parsing.

use serde_json::{Map, Value};
use veldmap_core::geo::{export_geojson_geometry, parse_geojson};

use super::{
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
        let geom_str = export_geojson_geometry(geom);
        if let Ok(geom_val) = serde_json::from_str::<Value>(&geom_str) {
            obj.insert("intersects".to_owned(), geom_val);
        }
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
    let mgrs_tile = props.and_then(|p| {
        p.get("grid:code")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .or_else(|| {
                let zone = p.get("mgrs:utm_zone").and_then(Value::as_u64)?;
                let lat_band = p.get("mgrs:latitude_band").and_then(Value::as_str)?;
                let sq = p.get("mgrs:grid_square").and_then(Value::as_str)?;
                Some(format!("{zone:02}{lat_band}{sq}"))
            })
            .or_else(|| {
                p.get("s2:mgrs_tile")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            })
    });
    let processing_level = props
        .and_then(|p| {
            p.get("s2:product_type")
                .or_else(|| p.get("processing:level"))
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .or_else(|| {
            collection.as_deref().and_then(|c| {
                if c.contains("l2a") {
                    Some("L2A (BOA)".to_owned())
                } else if c.contains("l1c") {
                    Some("L1C (TOA)".to_owned())
                } else {
                    None
                }
            })
        });

    // RFC 7946 §5: 2D bbox is `[west, south, east, north]` (len 4);
    // 3D bbox is `[west, south, min_z, east, north, max_z]` (len 6).
    let bbox = obj
        .get("bbox")
        .and_then(Value::as_array)
        .and_then(|arr| match arr.len() {
            4 => {
                let coords = [
                    arr[0].as_f64()?,
                    arr[1].as_f64()?,
                    arr[2].as_f64()?,
                    arr[3].as_f64()?,
                ];
                coords.iter().all(|v| v.is_finite()).then_some(coords)
            }
            6 => {
                let coords = [
                    arr[0].as_f64()?,
                    arr[1].as_f64()?,
                    arr[3].as_f64()?,
                    arr[4].as_f64()?,
                ];
                coords.iter().all(|v| v.is_finite()).then_some(coords)
            }
            _ => None,
        });

    let geometry = obj.get("geometry").and_then(|g| {
        if g.is_null() {
            return None;
        }
        let wrapped = serde_json::json!({
            "type": "Feature",
            "properties": {},
            "geometry": g,
        });
        let text = serde_json::to_string(&wrapped).ok()?;
        let fc = parse_geojson(&text).ok()?;
        fc.features.into_iter().next().map(|f| f.geometry)
    });

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
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|b| {
                    b.get("common_name")
                        .or_else(|| b.get("name"))
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                })
                .collect()
        })
        .unwrap_or_default();
    let gsd = obj.get("gsd").and_then(Value::as_f64);
    let raster_bands = obj
        .get("raster:bands")
        .and_then(Value::as_array)
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
    let nodata = obj.get("nodata").and_then(|v| {
        v.as_f64()
            .or_else(|| v.as_str().and_then(|s| s.trim().parse::<f64>().ok()))
    });
    Some(StacRasterBandMeta {
        scale: obj.get("scale").and_then(Value::as_f64),
        offset: obj.get("offset").and_then(Value::as_f64),
        nodata,
        data_type: obj
            .get("data_type")
            .and_then(Value::as_str)
            .map(str::to_owned),
        spatial_resolution: obj.get("spatial_resolution").and_then(Value::as_f64),
        unit: obj.get("unit").and_then(Value::as_str).map(str::to_owned),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_3d_bbox_start_datetime_fallback_mgrs_and_nan_nodata() {
        let json: Value = serde_json::json!({
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "id": "ITEM_3D",
                "collection": "cop-dem-glo-30",
                "bbox": [7.5, 45.8, 120.0, 9.2, 47.1, 4500.0],
                "properties": {
                    "datetime": null,
                    "start_datetime": "2024-06-01T00:00:00Z",
                    "mgrs:utm_zone": 32,
                    "mgrs:latitude_band": "T",
                    "mgrs:grid_square": "MS"
                },
                "assets": {
                    "data": {
                        "href": "https://example.org/dem.tif",
                        "raster:bands": [{"nodata": "NaN", "scale": 1.0}]
                    }
                }
            }]
        });
        let parsed = parse_search_response(&json).expect("parse");
        assert_eq!(parsed.items.len(), 1);
        let item = &parsed.items[0];
        assert_eq!(item.bbox, Some([7.5, 45.8, 9.2, 47.1]));
        assert_eq!(item.datetime.as_deref(), Some("2024-06-01T00:00:00Z"));
        assert_eq!(item.mgrs_tile.as_deref(), Some("32TMS"));
        assert!(item.assets[0].raster_bands[0].nodata.unwrap().is_nan());
    }

    #[test]
    fn build_search_payload_omits_non_finite_bbox_and_defaults_cloud_cover_none() {
        let q = StacSearchQuery {
            bbox: Some([f64::NAN, 0.0, 1.0, 1.0]),
            ..StacSearchQuery::default()
        };
        let payload = build_search_payload("sentinel-1-grd", &q);
        assert!(payload.get("bbox").is_none());
        assert!(payload.get("query").is_none());
    }
}
