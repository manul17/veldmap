//! Offline mock STAC API integration tests covering Sentinel-2 L2A & COP-DEM
//! search parsing, `raster:bands` `scale`/`offset` metadata, custom Bearer /
//! API-key / SAS auth headers, secret redaction, and built-in sample scenes.

use std::sync::{Arc, Mutex};
use std::thread;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use veldmap_core::sync::CancelToken;
use veldmap_net::{
    NetConfig, NetContext, StacAuth, StacCatalogConfig, StacClient, StacSearchQuery,
    builtin_sample_scenes,
};

struct RecordedStacReq {
    path_and_query: String,
    authorization: Option<String>,
    api_key: Option<String>,
    body: String,
}

struct MockStacServer {
    endpoint: String,
    recorded: Arc<Mutex<Vec<RecordedStacReq>>>,
    shutdown: CancelToken,
    handle: Option<thread::JoinHandle<()>>,
}

impl MockStacServer {
    fn start(response_json: String) -> Self {
        let recorded = Arc::new(Mutex::new(Vec::new()));
        let rec_clone = Arc::clone(&recorded);
        let shutdown = CancelToken::new();
        let stop = shutdown.clone();
        let (addr_tx, addr_rx) = std::sync::mpsc::channel();

        let handle = thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("stac mock rt");
            rt.block_on(async move {
                let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
                let addr = listener.local_addr().expect("addr");
                let _ = addr_tx.send(addr);

                loop {
                    tokio::select! {
                        biased;
                        _ = stop.cancelled() => break,
                        res = listener.accept() => {
                            let Ok((mut stream, _)) = res else { break; };
                            let rec = Arc::clone(&rec_clone);
                            let body_out = response_json.clone();
                            let stop_c = stop.clone();
                            tokio::spawn(async move {
                                let mut buf = vec![0u8; 16384];
                                let mut filled = 0usize;
                                let mut header_end = None;
                                let mut content_len = 0usize;
                                loop {
                                    let n = tokio::select! {
                                        _ = stop_c.cancelled() => return,
                                        r = stream.read(&mut buf[filled..]) => match r {
                                            Ok(0) | Err(_) => break,
                                            Ok(n) => n,
                                        }
                                    };
                                    filled += n;
                                    if header_end.is_none()
                                        && let Some(pos) = buf[..filled].windows(4).position(|w| w == b"\r\n\r\n")
                                    {
                                        let h_str = String::from_utf8_lossy(&buf[..pos]);
                                        for line in h_str.lines().skip(1) {
                                            if let Some((k, v)) = line.split_once(':')
                                                && k.trim().eq_ignore_ascii_case("content-length")
                                            {
                                                content_len = v.trim().parse().unwrap_or(0);
                                            }
                                        }
                                        header_end = Some(pos + 4);
                                    }
                                    if let Some(he) = header_end
                                        && filled >= he + content_len
                                    {
                                        break;
                                    }
                                    if filled == buf.len() {
                                        break;
                                    }
                                }
                                let he = header_end.unwrap_or(filled);
                                let head_str = String::from_utf8_lossy(&buf[..he]);
                                let first_line = head_str.lines().next().unwrap_or("");
                                let path_and_query = first_line
                                    .split_whitespace()
                                    .nth(1)
                                    .unwrap_or("")
                                    .to_owned();
                                let mut authorization = None;
                                let mut api_key = None;
                                for line in head_str.lines().skip(1) {
                                    if let Some((k, v)) = line.split_once(':') {
                                        if k.trim().eq_ignore_ascii_case("authorization") {
                                            authorization = Some(v.trim().to_owned());
                                        } else if k.trim().eq_ignore_ascii_case("x-api-key") {
                                            api_key = Some(v.trim().to_owned());
                                        }
                                    }
                                }
                                let body = String::from_utf8_lossy(&buf[he..filled.min(he + content_len)]).to_string();
                                rec.lock().expect("lock").push(RecordedStacReq {
                                    path_and_query,
                                    authorization,
                                    api_key,
                                    body,
                                });
                                let resp = format!(
                                    "HTTP/1.1 200 OK\r\nContent-Type: application/geo+json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                                    body_out.len(),
                                    body_out
                                );
                                let _ = stream.write_all(resp.as_bytes()).await;
                            });
                        }
                    }
                }
            });
        });

        let addr = addr_rx.recv().expect("stac addr");
        Self {
            endpoint: format!("http://{addr}/v1"),
            recorded,
            shutdown,
            handle: Some(handle),
        }
    }
}

impl Drop for MockStacServer {
    fn drop(&mut self) {
        self.shutdown.cancel();
        if let Some(h) = self.handle.take() {
            let _ = h.join();
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
    assert_eq!(red.primary_scale_offset(), Some((0.0001, -0.1)));
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

    // Built-in sample scenes are non-empty and use public HTTPS COG URLs.
    let samples = builtin_sample_scenes();
    assert!(samples.len() >= 3);
    for s in samples {
        assert!(s.cog_url.starts_with("https://"));
        assert!(s.cog_url.ends_with(".tif"));
    }
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
    let shutdown = CancelToken::new();
    let stop = shutdown.clone();
    let (addr_tx, addr_rx) = std::sync::mpsc::channel();
    let handle = thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("rt");
        rt.block_on(async move {
            let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
            let addr = listener.local_addr().expect("addr");
            let _ = addr_tx.send(addr);
            if let Ok((mut stream, _)) = tokio::select! {
                _ = stop.cancelled() => Err(std::io::Error::other("stop")),
                res = listener.accept() => res,
            } {
                let mut buf = [0u8; 4096];
                let _ = stream.read(&mut buf).await;
                let body = r#"{"code":"BadRequest","description":"invalid datetime for secret-tok"}"#;
                let resp = format!(
                    "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = stream.write_all(resp.as_bytes()).await;
            }
        });
    });
    let addr = addr_rx.recv().expect("addr");
    let cat = StacCatalogConfig {
        endpoint: format!("http://{addr}/v1"),
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
    shutdown.cancel();
    let _ = handle.join();
}
