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
            "geometry": {
                "type": "Point",
                "coordinates": [8.5, 46.5]
            },
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
    let parsed = parse_search_response(&json, "https://stac.example.org/v1/search").expect("parse");
    assert_eq!(parsed.items.len(), 1);
    let item = &parsed.items[0];
    assert_eq!(item.bbox, Some([7.5, 45.8, 9.2, 47.1]));
    assert_eq!(item.datetime.as_deref(), Some("2024-06-01T00:00:00Z"));
    assert_eq!(item.mgrs_tile.as_deref(), Some("32TMS"));
    assert!(item.assets[0].raster_bands[0].nodata.unwrap().is_nan());
    let geom = item.geometry.clone().expect("parsed point geometry");
    let payload = build_search_payload(
        "cop-dem-glo-30",
        &StacSearchQuery {
            intersects: Some(geom),
            ..StacSearchQuery::default()
        },
    );
    assert_eq!(
        payload.get("intersects").and_then(|v| v.get("type")),
        Some(&Value::String("Point".to_owned()))
    );
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

#[test]
fn parses_stac11_bands_processing_level_scl_and_skips_visual_for_spectral() {
    let json: Value = serde_json::json!({
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "id": "S2A_32TMS_20240715_0_L2A",
            "collection": "custom-collection",
            "bbox": [7.5, 45.8, 9.2, 47.1],
            "properties": {
                "datetime": "2024-07-15T10:20:30Z",
                "platform": "sentinel-2a",
                "s2:product_type": null,
                "s2:processing_level": "level-2a",
                "eo:cloud_cover": 4.2
            },
            "assets": {
                "visual": {
                    "href": "https://example.org/TCI.tif",
                    "type": "image/tiff; application=geotiff; profile=cloud-optimized",
                    "roles": ["visual"],
                    "eo:bands": [
                        {"common_name": "red"},
                        {"common_name": "green"},
                        {"common_name": "blue"}
                    ]
                },
                "nir08": {
                    "href": "https://example.org/B8A.tif",
                    "type": "image/tiff; application=geotiff; profile=cloud-optimized",
                    "roles": ["data", "reflectance"],
                    "eo:bands": [{"name": "B8A", "common_name": "nir08"}]
                },
                "nir": {
                    "href": "https://example.org/B08.tif",
                    "type": "image/tiff; application=geotiff; profile=cloud-optimized",
                    "roles": ["data", "reflectance"],
                    "eo:bands": [{"name": "B08", "common_name": "nir"}],
                    "raster:bands": [{"scale": 0.0001, "offset": -0.1, "nodata": 0, "unit": "reflectance"}]
                },
                "B04_10m": {
                    "href": "https://example.org/B04.tif",
                    "type": "image/tiff; application=geotiff; profile=cloud-optimized",
                    "roles": ["data"],
                    "bands": [{
                        "name": "B04",
                        "eo:common_name": "red",
                        "raster:scale": 0.0001,
                        "raster:offset": -0.1,
                        "nodata": 0,
                        "unit": "reflectance"
                    }]
                },
                "SCL_20m": {
                    "href": "https://example.org/SCL.tif",
                    "type": "image/tiff; application=geotiff; profile=cloud-optimized",
                    "roles": ["data", "scl"]
                }
            }
        }, {
            "type": "Feature",
            "id": "S2_COLLECTION_FALLBACK",
            "collection": "sentinel-2-l2a",
            "properties": {"datetime": "2024-07-15T10:20:30Z"},
            "assets": {}
        }]
    });
    let parsed = parse_search_response(&json, "https://stac.example.org/v1/search").expect("parse");
    let item = &parsed.items[0];
    assert_eq!(item.processing_level.as_deref(), Some("level-2a"));
    assert_eq!(item.platform.as_deref(), Some("sentinel-2a"));
    // Collection fallback still works when explicit properties are absent.
    assert_eq!(
        parsed.items[1].processing_level.as_deref(),
        Some("L2A (BOA)")
    );
    // Prefer 10m nir/B08 over 20m nir08/B8A.
    let nir = item
        .find_band_asset(&["nir", "b08", "nir08", "b8a"])
        .expect("find nir");
    assert_eq!(nir.key, "nir");
    assert!(nir.eo_bands.iter().any(|b| b == "nir"));
    assert!(nir.eo_bands.iter().any(|b| b == "B08"));
    // STAC 1.1 `bands` with `eo:common_name` resolves `B04_10m` and skips `visual` TCI!
    let red = item.find_band_asset(&["red", "b04"]).expect("find red");
    assert_eq!(red.key, "B04_10m");
    assert!(red.eo_bands.iter().any(|b| b == "red"));
    assert_eq!(red.raster_bands.len(), 1);
    assert_eq!(red.raster_bands[0].scale, Some(0.0001));
    assert_eq!(red.raster_bands[0].offset, Some(-0.1));
    assert_eq!(red.raster_bands[0].nodata, Some(0.0));
    // `SCL_20m` + `roles: ["scl"]` detected cleanly.
    let scl = item.scl_asset().expect("find scl");
    assert_eq!(scl.key, "SCL_20m");
    assert!(scl.is_scl_asset());
}

#[test]
fn parse_stac_item_sanitizes_self_href_and_ignores_relative_or_non_self_links() {
    let json = serde_json::json!({
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "S2_WITH_SELF",
                "links": [
                    {"rel": "parent", "href": "https://stac.example.org/collections/s2"},
                    {"rel": "SELF", "href": "https://user:secret@stac.example.org/items/s2?sig=abc#frag"}
                ],
                "properties": {"datetime": "2026-10-01T10:00:00Z"},
                "assets": {}
            },
            {
                "type": "Feature",
                "id": "S2_RELATIVE_SELF",
                "links": [
                    {"rel": "self", "href": "/collections/s2/items/rel"}
                ],
                "properties": {"datetime": "2026-10-01T10:00:00Z"},
                "assets": {}
            }
        ]
    });
    let parsed = parse_search_response(&json, "https://stac.example.org/v1/search").expect("parse");
    assert_eq!(
        parsed.items[0].self_href.as_deref(),
        Some("https://stac.example.org/items/s2")
    );
    assert_eq!(parsed.items[1].self_href, None);
}

#[test]
fn build_search_payload_emits_sortby_only_when_requested_and_parses_next_links() {
    let base = build_search_payload("sentinel-2-l2a", &StacSearchQuery::default());
    assert!(base.get("sortby").is_none());

    let newest = build_search_payload(
        "sentinel-2-l2a",
        &StacSearchQuery {
            sort_by: Some(StacSortBy::Newest),
            ..StacSearchQuery::default()
        },
    );
    assert_eq!(
        newest["sortby"],
        serde_json::json!([{"field": "properties.datetime", "direction": "desc"}])
    );

    let cloud = build_search_payload(
        "sentinel-2-l2a",
        &StacSearchQuery {
            sort_by: Some(StacSortBy::LeastCloud),
            ..StacSearchQuery::default()
        },
    );
    assert_eq!(
        cloud["sortby"],
        serde_json::json!([{"field": "properties.eo:cloud_cover", "direction": "asc"}])
    );

    // Relative GET next link resolves against request_url.
    let get_resp = serde_json::json!({
        "type": "FeatureCollection",
        "features": [],
        "links": [
            {"rel": "next", "href": "/v1/search?token=abc"}
        ]
    });
    let parsed_get =
        parse_search_response(&get_resp, "https://stac.example.org/v1/search").expect("parse get");
    let next = parsed_get.next_link.expect("next link");
    assert_eq!(next.href, "https://stac.example.org/v1/search?token=abc");
    assert_eq!(next.method, StacNextMethod::Get);
    assert!(!next.merge);

    // POST next link with body + merge: true (AWS Earth Search style).
    let post_resp = serde_json::json!({
        "type": "FeatureCollection",
        "features": [],
        "links": [
            {
                "rel": "next",
                "href": "https://stac.example.org/v1/search",
                "method": "POST",
                "merge": true,
                "body": {"next": "cursor-2"}
            }
        ]
    });
    let parsed_post = parse_search_response(&post_resp, "https://stac.example.org/v1/search")
        .expect("parse post");
    let next_post = parsed_post.next_link.expect("post next link");
    assert_eq!(next_post.method, StacNextMethod::Post);
    assert!(next_post.merge);
    assert!(next_post.body.as_deref().unwrap().contains("cursor-2"));
}
