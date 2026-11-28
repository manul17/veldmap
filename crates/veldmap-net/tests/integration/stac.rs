//! Offline mock STAC API integration tests covering Sentinel-2 L2A & COP-DEM
//! search parsing, `raster:bands` `scale`/`offset` metadata, custom Bearer /
//! API-key / SAS auth headers, and secret redaction.

use std::sync::{Arc, Mutex};

use veldmap_core::sync::CancelToken;
use veldmap_net::{
    NetConfig, NetContext, StacAuth, StacCatalogConfig, StacClient, StacError, StacNextLink,
    StacNextMethod, StacSearchQuery, StacSortBy,
};

use crate::mock_http::{MockHttpServer, write_http_response};

struct RecordedStacReq {
    path_and_query: String,
    authorization: Option<String>,
    api_key: Option<String>,
    body: String,
}

struct MockStacServer {
    endpoint: String,
    recorded: Arc<Mutex<Vec<RecordedStacReq>>>,
    _server: MockHttpServer,
}

impl MockStacServer {
    fn start(response_json: String) -> Self {
        let recorded = Arc::new(Mutex::new(Vec::new()));
        let rec_clone = Arc::clone(&recorded);
        let body_shared = Arc::new(response_json);

        let server = MockHttpServer::start(move |req, mut stream, _stop| {
            let rec = Arc::clone(&rec_clone);
            let body_out = Arc::clone(&body_shared);
            async move {
                rec.lock().expect("lock").push(RecordedStacReq {
                    path_and_query: req.path_and_query.clone(),
                    authorization: req.header("authorization").map(str::to_owned),
                    api_key: req.header("x-api-key").map(str::to_owned),
                    body: req.body_string(),
                });
                write_http_response(
                    &mut stream,
                    200,
                    "OK",
                    "Content-Type: application/geo+json\r\n",
                    body_out.as_bytes(),
                )
                .await;
            }
        });

        let addr = server.addr();
        Self {
            endpoint: format!("http://{addr}/v1"),
            recorded,
            _server: server,
        }
    }
}

#[test]
fn stac_search_parses_sentinel2_assets_raster_bands_auth_and_redacts_secrets() {
    let fixture = r#"{
        "type": "FeatureCollection",
        "numberMatched": 42,
        "features": [
            {
                "type": "Feature",
                "id": "S2A_32TMS_20230820_0_L2A",
                "collection": "sentinel-2-l2a",
                "bbox": [7.6, 45.9, 9.1, 47.0],
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[7.6, 45.9], [9.1, 45.9], [9.1, 47.0], [7.6, 47.0], [7.6, 45.9]]]
                },
                "properties": {
                    "datetime": "2023-08-20T10:20:31Z",
                    "eo:cloud_cover": 4.2,
                    "platform": "sentinel-2a",
                    "grid:code": "MGRS-32TMS"
                },
                "assets": {
                    "red": {
                        "title": "Red (band 4) - 10m",
                        "href": "https://sentinel-cogs.example.org/B04.tif",
                        "type": "image/tiff; application=geotiff; profile=cloud-optimized",
                        "roles": ["data", "reflectance"],
                        "eo:bands": [{"name": "red", "common_name": "red"}],
                        "raster:bands": [{"nodata": 0, "data_type": "uint16", "scale": 0.0001, "offset": -0.1, "spatial_resolution": 10}]
                    },
                    "visual": {
                        "title": "True color image",
                        "href": "https://sentinel-cogs.example.org/TCI.tif",
                        "type": "image/tiff; application=geotiff; profile=cloud-optimized",
                        "roles": ["visual"]
                    },
                    "scl": {
                        "title": "Scene classification map (SCL)",
                        "href": "https://sentinel-cogs.example.org/SCL.tif",
                        "type": "image/tiff; application=geotiff; profile=cloud-optimized",
                        "roles": ["data", "metadata"]
                    }
                }
            }
        ]
    }"#;

    let server = MockStacServer::start(fixture.to_owned());
    let net = Arc::new(NetContext::new(NetConfig::default()).expect("net"));
    let client = StacClient::new(Arc::clone(&net));
    let cancel = CancelToken::new();

    // Test Bearer auth + bbox + cloud cover filter.
    let catalog = StacCatalogConfig {
        endpoint: server.endpoint.clone(),
        collection: "sentinel-2-l2a".to_owned(),
        auth: StacAuth::Bearer("super-secret-bearer".to_owned()),
    };
    assert!(
        !format!("{catalog:?}").contains("super-secret-bearer"),
        "Debug output must redact bearer token"
    );

    let query = StacSearchQuery {
        bbox: Some([7.5, 45.8, 9.2, 47.1]),
        intersects: None,
        datetime: Some("2023-08-01T00:00:00Z/2023-08-31T23:59:59Z".to_owned()),
        max_cloud_cover: Some(20.0),
        sort_by: Some(StacSortBy::Newest),
        limit: 10,
    };

    let result = net
        .handle()
        .block_on(client.search_async(&catalog, &query, &cancel))
        .expect("stac search");
    assert_eq!(result.matched, Some(42));
    assert_eq!(result.items.len(), 1);
    let item = &result.items[0];
    assert_eq!(item.id, "S2A_32TMS_20230820_0_L2A");
    assert_eq!(item.mgrs_tile.as_deref(), Some("MGRS-32TMS"));
    assert_eq!(item.processing_level.as_deref(), Some("L2A (BOA)"));
    assert!(item.geometry.is_some());
    assert_eq!(
        item.preferred_cog_asset().map(|a| a.key.as_str()),
        Some("visual")
    );
    let red = item
        .assets
        .iter()
        .find(|a| a.key == "red")
        .expect("red asset");
    assert_eq!(red.raster_bands[0].scale, Some(0.0001));
    assert_eq!(red.raster_bands[0].offset, Some(-0.1));
    assert_eq!(red.raster_bands[0].nodata, Some(0.0));

    let reqs = server.recorded.lock().expect("lock");
    assert_eq!(reqs.len(), 1);
    assert_eq!(reqs[0].path_and_query, "/v1/search");
    assert_eq!(
        reqs[0].authorization.as_deref(),
        Some("Bearer super-secret-bearer")
    );
    assert!(reqs[0].body.contains("\"eo:cloud_cover\""));
    drop(reqs);

    // Test API-key header and SAS query propagation to asset hrefs.
    let api_catalog = StacCatalogConfig {
        endpoint: server.endpoint.clone(),
        collection: "sentinel-2-l2a".to_owned(),
        auth: StacAuth::ApiKey {
            header_name: "x-api-key".to_owned(),
            value: "my-api-secret".to_owned(),
        },
    };
    let _ = net
        .handle()
        .block_on(client.search_async(&api_catalog, &query, &cancel))
        .expect("api key search");
    assert_eq!(
        server.recorded.lock().expect("lock")[1].api_key.as_deref(),
        Some("my-api-secret")
    );

    let sas_catalog = StacCatalogConfig {
        endpoint: server.endpoint.clone(),
        collection: "sentinel-2-l2a".to_owned(),
        auth: StacAuth::SasQuery("sv=2024&sig=sas_secret".to_owned()),
    };
    let sas_res = net
        .handle()
        .block_on(client.search_async(&sas_catalog, &query, &cancel))
        .expect("sas search");
    // Asset hrefs remain unsigned during parse so callers apply SAS auth once
    // at stream-open time without duplicating query parameters.
    let asset = &sas_res.items[0].assets[0];
    assert!(
        !asset.href.contains("sas_secret"),
        "parsed asset href should stay unsigned until stream-open"
    );
    let signed_href = sas_catalog.auth.apply_to_url(&asset.href);
    assert!(signed_href.ends_with("?sv=2024&sig=sas_secret"));

    // Manual Debug on StacAsset must redact presigned/SAS query parameters.
    let mut presigned = asset.clone();
    presigned.href = signed_href;
    let asset_dbg = format!("{presigned:?}");
    assert!(
        !asset_dbg.contains("sas_secret"),
        "StacAsset Debug leaked query secret: {asset_dbg}"
    );
    assert!(
        server.recorded.lock().expect("lock")[2]
            .path_and_query
            .contains("sig=sas_secret")
    );
}

#[test]
fn stac_rejects_invalid_endpoint_forbidden_header_and_redacts_error_snippets() {
    let net = Arc::new(NetContext::new(NetConfig::default()).expect("net"));
    let client = StacClient::new(Arc::clone(&net));
    let cancel = CancelToken::new();

    // Invalid endpoint with whitespace.
    let bad_ep = StacCatalogConfig {
        endpoint: "https://example.org/stac v1".to_owned(),
        collection: "sentinel-2-l2a".to_owned(),
        auth: StacAuth::None,
    };
    let err = net
        .handle()
        .block_on(client.search_async(&bad_ep, &StacSearchQuery::default(), &cancel))
        .expect_err("bad endpoint");
    assert!(matches!(err, veldmap_net::StacError::InvalidEndpoint(_)));

    // Forbidden header override.
    let bad_hdr = StacCatalogConfig {
        endpoint: "https://example.org/stac".to_owned(),
        collection: "sentinel-2-l2a".to_owned(),
        auth: StacAuth::ApiKey {
            header_name: "Content-Type".to_owned(),
            value: "text/plain".to_owned(),
        },
    };
    let err = net
        .handle()
        .block_on(client.search_async(&bad_hdr, &StacSearchQuery::default(), &cancel))
        .expect_err("forbidden header");
    assert!(matches!(
        err,
        veldmap_net::StacError::InvalidAuthHeader { .. }
    ));

    // Non-2xx response includes redacted diagnostic body snippet.
    let err_server = MockHttpServer::start(|_req, mut stream, _stop| async move {
        let body = br#"{"code":"BadRequest","description":"invalid datetime for secret-tok"}"#;
        write_http_response(
            &mut stream,
            400,
            "Bad Request",
            "Content-Type: application/json\r\n",
            body,
        )
        .await;
    });
    let cat = StacCatalogConfig {
        endpoint: format!("http://{}/v1", err_server.addr()),
        collection: "sentinel-2-l2a".to_owned(),
        auth: StacAuth::Bearer("secret-tok".to_owned()),
    };
    let err = net
        .handle()
        .block_on(client.search_async(&cat, &StacSearchQuery::default(), &cancel))
        .expect_err("400 status");
    let msg = err.to_string();
    assert!(
        msg.contains("invalid datetime"),
        "expected error snippet: {msg}"
    );
    assert!(
        !msg.contains("secret-tok"),
        "secret leaked in snippet: {msg}"
    );
}

#[test]
fn stac_pagination_get_post_merge_origin_guards_and_fetch_item() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let page_hits = Arc::new(AtomicUsize::new(0));
    let recorded = Arc::new(Mutex::new(
        Vec::<(String, String, Option<String>, String)>::new(),
    ));
    let hits_srv = Arc::clone(&page_hits);
    let rec_srv = Arc::clone(&recorded);

    let server = MockHttpServer::start(move |req, mut stream, _stop| {
        let hits = Arc::clone(&hits_srv);
        let rec = Arc::clone(&rec_srv);
        async move {
            let method = req.method.clone();
            let path = req.path_and_query.clone();
            let auth = req.header("authorization").map(str::to_owned);
            let body = req.body_string();
            rec.lock()
                .expect("lock")
                .push((method.clone(), path.clone(), auth, body.clone()));

            if path == "/v1/items/S2A_REFRESH" && method == "GET" {
                let item_json = r#"{
                    "type": "Feature",
                    "id": "S2A_REFRESH",
                    "collection": "sentinel-2-l2a",
                    "properties": {"datetime": "2025-05-01T10:00:00Z"},
                    "assets": {
                        "red": {
                            "href": "https://storage.example.org/b04.tif?sig=fresh_token_123",
                            "type": "image/tiff; application=geotiff; profile=cloud-optimized"
                        }
                    }
                }"#;
                write_http_response(
                    &mut stream,
                    200,
                    "OK",
                    "Content-Type: application/geo+json\r\n",
                    item_json.as_bytes(),
                )
                .await;
                return;
            }

            if path == "/v1/items/REDIRECT_CROSS" {
                write_http_response(
                    &mut stream,
                    302,
                    "Found",
                    "Location: http://evil.example.org/steal\r\n",
                    b"",
                )
                .await;
                return;
            }

            let n = hits.fetch_add(1, Ordering::SeqCst);
            let resp = match n {
                // Page 1: POST next with merge: true (AWS Earth Search pattern).
                0 => {
                    r#"{
                    "type": "FeatureCollection",
                    "numberMatched": 3,
                    "features": [{"type": "Feature", "id": "ITEM_1", "properties": {"datetime": "2025-05-03T10:00:00Z"}, "assets": {}}],
                    "links": [{
                        "rel": "next",
                        "href": "/v1/search",
                        "method": "POST",
                        "merge": true,
                        "body": {"next": "tok-page-2"}
                    }]
                }"#
                }
                // Page 2: relative GET next link.
                1 => {
                    r#"{
                    "type": "FeatureCollection",
                    "numberMatched": 3,
                    "features": [{"type": "Feature", "id": "ITEM_2", "properties": {"datetime": "2025-05-02T10:00:00Z"}, "assets": {}}],
                    "links": [{
                        "rel": "next",
                        "href": "/v1/search?cursor=page3"
                    }]
                }"#
                }
                // Page 3: final page without next link.
                _ => {
                    r#"{
                    "type": "FeatureCollection",
                    "numberMatched": 3,
                    "features": [{"type": "Feature", "id": "ITEM_3", "properties": {"datetime": "2025-05-01T10:00:00Z"}, "assets": {}}],
                    "links": []
                }"#
                }
            };
            write_http_response(
                &mut stream,
                200,
                "OK",
                "Content-Type: application/geo+json\r\n",
                resp.as_bytes(),
            )
            .await;
        }
    });

    let endpoint = format!("http://{}/v1", server.addr());
    let net = Arc::new(NetContext::new(NetConfig::default()).expect("net"));
    let client = StacClient::new(Arc::clone(&net));
    let cancel = CancelToken::new();
    let catalog = StacCatalogConfig {
        endpoint: endpoint.clone(),
        collection: "sentinel-2-l2a".to_owned(),
        auth: StacAuth::Bearer("page-secret".to_owned()),
    };
    let query = StacSearchQuery {
        bbox: Some([10.0, 50.0, 11.0, 51.0]),
        sort_by: Some(StacSortBy::LeastCloud),
        limit: 1,
        ..StacSearchQuery::default()
    };

    // Page 1 -> POST search.
    let p1 = net
        .handle()
        .block_on(client.search_async(&catalog, &query, &cancel))
        .expect("page 1");
    assert_eq!(p1.items[0].id, "ITEM_1");
    let next1 = p1.next_link.expect("next1");
    assert_eq!(next1.method, StacNextMethod::Post);
    assert!(next1.merge);

    // Page 2 -> POST merge preserves original collection + bbox + sortby + adds `next`.
    let p2 = net
        .handle()
        .block_on(client.search_next_async(&catalog, &query, &next1, &cancel))
        .expect("page 2");
    assert_eq!(p2.items[0].id, "ITEM_2");
    let next2 = p2.next_link.expect("next2");
    assert_eq!(next2.method, StacNextMethod::Get);

    // Page 3 -> GET cursor URL.
    let p3 = net
        .handle()
        .block_on(client.search_next_async(&catalog, &query, &next2, &cancel))
        .expect("page 3");
    assert_eq!(p3.items[0].id, "ITEM_3");
    assert!(p3.next_link.is_none());

    {
        let reqs = recorded.lock().expect("lock");
        assert_eq!(reqs.len(), 3);
        assert_eq!(reqs[1].0, "POST");
        assert!(reqs[1].3.contains("\"sentinel-2-l2a\""));
        assert!(reqs[1].3.contains("\"tok-page-2\""));
        assert!(reqs[1].3.contains("\"properties.eo:cloud_cover\""));
        assert_eq!(reqs[2].0, "GET");
        assert_eq!(reqs[2].1, "/v1/search?cursor=page3");
        assert_eq!(reqs[2].2.as_deref(), Some("Bearer page-secret"));
    }

    // Security guard 1: HTTPS catalog refuses HTTP next-link downgrade.
    let https_catalog = StacCatalogConfig {
        endpoint: "https://stac.secure.example/v1".to_owned(),
        collection: "s2".to_owned(),
        auth: StacAuth::None,
    };
    let downgrade_link = StacNextLink {
        href: "http://stac.secure.example/v1/search?page=2".to_owned(),
        method: StacNextMethod::Get,
        body: None,
        merge: false,
    };
    let err = net
        .handle()
        .block_on(client.search_next_async(&https_catalog, &query, &downgrade_link, &cancel))
        .expect_err("downgrade rejected");
    assert!(matches!(err, StacError::LinkRejected(_)));

    // Security guard 2: Authenticated catalog refuses off-origin next link.
    let off_origin = StacNextLink {
        href: "http://attacker.example.org/v1/search?page=2".to_owned(),
        method: StacNextMethod::Get,
        body: None,
        merge: false,
    };
    let err = net
        .handle()
        .block_on(client.search_next_async(&catalog, &query, &off_origin, &cancel))
        .expect_err("off-origin auth rejected");
    assert!(matches!(err, StacError::LinkRejected(_)));

    // fetch_item_async succeeds with auth and returns refreshed presigned asset href.
    let item = net
        .handle()
        .block_on(client.fetch_item_async(
            &format!("{endpoint}/items/S2A_REFRESH"),
            &catalog.auth,
            &cancel,
        ))
        .expect("fetch item");
    assert_eq!(item.id, "S2A_REFRESH");
    assert_eq!(
        item.assets[0].href,
        "https://storage.example.org/b04.tif?sig=fresh_token_123"
    );

    // fetch_item_async with custom auth blocks cross-origin redirect.
    let redir_err = net
        .handle()
        .block_on(client.fetch_item_async(
            &format!("{endpoint}/items/REDIRECT_CROSS"),
            &catalog.auth,
            &cancel,
        ))
        .expect_err("cross-origin redirect blocked");
    assert!(matches!(redir_err, StacError::Network(_)));
    assert!(!redir_err.to_string().contains("page-secret"));
}

#[test]
fn stac_item_flight_cache_does_not_poison_errors_respects_ttl_and_isolates_auth() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    let hits = Arc::new(AtomicUsize::new(0));
    let hits_srv = Arc::clone(&hits);

    let server = MockHttpServer::start(move |req, mut stream, _stop| {
        let hits = Arc::clone(&hits_srv);
        async move {
            let path = req.path_and_query.clone();
            if path == "/items/RETRYABLE" {
                let n = hits.fetch_add(1, Ordering::SeqCst);
                if n == 0 {
                    write_http_response(
                        &mut stream,
                        503,
                        "Service Unavailable",
                        "Content-Type: text/plain\r\n",
                        b"transient outage",
                    )
                    .await;
                    return;
                }
                let body = format!(
                    r#"{{"type":"Feature","id":"RETRYABLE","properties":{{"datetime":"2025-05-01T10:00:00Z"}},"assets":{{"red":{{"href":"https://storage.example.org/b04.tif?sig=tok_{n}"}}}}}}"#
                );
                write_http_response(
                    &mut stream,
                    200,
                    "OK",
                    "Content-Type: application/geo+json\r\n",
                    body.as_bytes(),
                )
                .await;
                return;
            }
            if path == "/items/AUTH_ISOLATED" {
                let has_bearer = req
                    .header("authorization")
                    .is_some_and(|v| v == "Bearer secret-a");
                if !has_bearer {
                    write_http_response(
                        &mut stream,
                        401,
                        "Unauthorized",
                        "Content-Type: text/plain\r\n",
                        b"missing bearer",
                    )
                    .await;
                    return;
                }
                let body = r#"{"type":"Feature","id":"AUTH_ISOLATED","properties":{"datetime":"2025-05-01T10:00:00Z"},"assets":{"red":{"href":"https://storage.example.org/b04.tif?sig=authed"}}}"#;
                write_http_response(
                    &mut stream,
                    200,
                    "OK",
                    "Content-Type: application/geo+json\r\n",
                    body.as_bytes(),
                )
                .await;
                return;
            }
            write_http_response(
                &mut stream,
                404,
                "Not Found",
                "Content-Type: text/plain\r\n",
                b"not found",
            )
            .await;
        }
    });

    let net = Arc::new(NetContext::new(NetConfig::default()).expect("net context"));
    let client = StacClient::new(Arc::clone(&net));
    let cancel = CancelToken::new();
    let cache = veldmap_net::StacItemFlightCache::with_ttl(Duration::from_millis(60));
    let retry_url = format!("http://{}/items/RETRYABLE", server.addr());
    let auth_url = format!("http://{}/items/AUTH_ISOLATED", server.addr());

    // 1. Transient 503 failure must NOT poison a subsequent retry in the same epoch.
    let err = net
        .handle()
        .block_on(cache.fetch_item_coalesced_async(
            &client,
            1,
            &retry_url,
            &StacAuth::None,
            &cancel,
        ))
        .expect_err("first 503 fails");
    assert!(matches!(*err, StacError::HttpStatus { code: 503, .. }));

    let ok1 = net
        .handle()
        .block_on(cache.fetch_item_coalesced_async(
            &client,
            1,
            &retry_url,
            &StacAuth::None,
            &cancel,
        ))
        .expect("retry after 503 succeeds");
    assert_eq!(
        ok1.assets[0].href,
        "https://storage.example.org/b04.tif?sig=tok_1"
    );

    // 2. Within TTL, second call coalesces to the cached item (no extra HTTP call).
    let ok_cached = net
        .handle()
        .block_on(cache.fetch_item_coalesced_async(
            &client,
            1,
            &retry_url,
            &StacAuth::None,
            &cancel,
        ))
        .expect("cached hit");
    assert_eq!(ok_cached.assets[0].href, ok1.assets[0].href);
    assert_eq!(hits.load(Ordering::SeqCst), 2);

    // 3. After TTL expires, next call re-fetches a fresh presigned token.
    std::thread::sleep(Duration::from_millis(80));
    let ok2 = net
        .handle()
        .block_on(cache.fetch_item_coalesced_async(
            &client,
            1,
            &retry_url,
            &StacAuth::None,
            &cancel,
        ))
        .expect("post-TTL re-fetch succeeds");
    assert_eq!(
        ok2.assets[0].href,
        "https://storage.example.org/b04.tif?sig=tok_2"
    );

    // 4. Auth-isolated cache keys: anonymous 401 never poisons authenticated call,
    //    and authenticated Ok never leaks to anonymous caller.
    let anon_err = net
        .handle()
        .block_on(cache.fetch_item_coalesced_async(&client, 1, &auth_url, &StacAuth::None, &cancel))
        .expect_err("anonymous 401");
    assert!(matches!(*anon_err, StacError::HttpStatus { code: 401, .. }));

    let authed = net
        .handle()
        .block_on(cache.fetch_item_coalesced_async(
            &client,
            1,
            &auth_url,
            &StacAuth::Bearer("secret-a".to_owned()),
            &cancel,
        ))
        .expect("authenticated call succeeds");
    assert_eq!(
        authed.assets[0].href,
        "https://storage.example.org/b04.tif?sig=authed"
    );

    let anon_after = net
        .handle()
        .block_on(cache.fetch_item_coalesced_async(&client, 1, &auth_url, &StacAuth::None, &cancel))
        .expect_err("anonymous call must not receive cached authenticated item");
    assert!(matches!(
        *anon_after,
        StacError::HttpStatus { code: 401, .. }
    ));
}
