use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant};

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use veldmap_core::sync::{CancelToken, Wakeup, lock_unpoisoned};
use veldmap_core::tile::{DecodedTile, PixelFormat, PlanetaryScheme, QuadTileId, TileKey};
use veldmap_net::{
    BlobStoreError, CustomBasemapConfig, DecodeFailure, DecodeRequest, FetcherConfig,
    LiveBasemapSpec, LiveTilePipeline, TileBlobStore, TileDecodeExecutor,
};

type BlobMapKey = (String, PlanetaryScheme, u8, u32, u32);

const TEST_TILE_MAGIC: &[u8] = b"VELDMAP_TEST_TILE_8X8";

#[derive(Default)]
struct MemBlobStore {
    map: Mutex<HashMap<BlobMapKey, Vec<u8>>>,
}

impl TileBlobStore for MemBlobStore {
    fn get(&self, source_key: &str, id: QuadTileId) -> Result<Option<Vec<u8>>, BlobStoreError> {
        let map = lock_unpoisoned(&self.map);
        Ok(map
            .get(&(source_key.to_owned(), id.scheme, id.level, id.x, id.y))
            .cloned())
    }

    fn put(&self, source_key: &str, id: QuadTileId, payload: &[u8]) -> Result<(), BlobStoreError> {
        let mut map = lock_unpoisoned(&self.map);
        map.insert(
            (source_key.to_owned(), id.scheme, id.level, id.x, id.y),
            payload.to_vec(),
        );
        Ok(())
    }

    fn delete(&self, source_key: &str, id: QuadTileId) -> Result<(), BlobStoreError> {
        let mut map = lock_unpoisoned(&self.map);
        map.remove(&(source_key.to_owned(), id.scheme, id.level, id.x, id.y));
        Ok(())
    }
}

/// Lightweight test [`TileDecodeExecutor`] keeping `veldmap-net` integration
/// tests completely independent of `veldmap-raster` and `image`.
struct TestDecodeExecutor {
    done_tx: mpsc::Sender<(TileKey, DecodedTile)>,
    wakeup: Wakeup,
}

impl TileDecodeExecutor for TestDecodeExecutor {
    fn submit(&self, req: DecodeRequest) -> bool {
        if req.cancel.is_cancelled() {
            return false;
        }
        if req.bytes.starts_with(TEST_TILE_MAGIC) {
            let tile = DecodedTile::try_with_mips(
                8,
                8,
                PixelFormat::Rgba8UnormSrgb,
                vec![128u8; 8 * 8 * 4],
                vec![
                    vec![128u8; 4 * 4 * 4],
                    vec![128u8; 2 * 2 * 4],
                    vec![128u8; 4],
                ],
            )
            .expect("synthetic tile");
            if self.done_tx.send((req.key, tile)).is_ok() {
                req.on_decoded.complete(Ok(()));
                (self.wakeup)();
            }
        } else {
            req.on_decoded.complete(Err(DecodeFailure::InvalidPayload(
                "corrupt test tile payload".to_owned(),
            )));
        }
        true
    }
}

fn make_valid_tile_bytes(tag: u8) -> Vec<u8> {
    let mut out = TEST_TILE_MAGIC.to_vec();
    out.push(tag);
    out
}

struct MockServer {
    addr: std::net::SocketAddr,
    hits: Arc<AtomicUsize>,
    max_concurrent: Arc<AtomicUsize>,
    handle: std::thread::JoinHandle<()>,
    stop: CancelToken,
}

#[derive(Clone)]
struct MockResponse {
    status: u16,
    headers: String,
    body: Vec<u8>,
    delay: Duration,
}

impl MockResponse {
    fn ok_tile(body: Vec<u8>, delay: Duration) -> Self {
        Self {
            status: 200,
            headers: "Content-Type: image/png\r\n".to_owned(),
            body,
            delay,
        }
    }
}

fn spawn_mock_server<F>(handler: F) -> MockServer
where
    F: Fn(&str, usize) -> MockResponse + Send + Sync + 'static,
{
    let hits = Arc::new(AtomicUsize::new(0));
    let active = Arc::new(AtomicUsize::new(0));
    let max_concurrent = Arc::new(AtomicUsize::new(0));
    let stop = CancelToken::new();

    let (addr_tx, addr_rx) = mpsc::channel();
    let hits_c = Arc::clone(&hits);
    let active_c = Arc::clone(&active);
    let max_c = Arc::clone(&max_concurrent);
    let stop_c = stop.clone();
    let handler = Arc::new(handler);

    let handle = std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("mock server rt");
        rt.block_on(async move {
            let listener = TcpListener::bind("127.0.0.1:0")
                .await
                .expect("bind loopback");
            let addr = listener.local_addr().expect("local addr");
            let _ = addr_tx.send(addr);

            loop {
                let accepted = tokio::select! {
                    _ = stop_c.cancelled() => break,
                    res = listener.accept() => res,
                };
                let Ok((mut socket, _)) = accepted else {
                    break;
                };
                let hits_t = Arc::clone(&hits_c);
                let active_t = Arc::clone(&active_c);
                let max_t = Arc::clone(&max_c);
                let stop_t = stop_c.clone();
                let handler_t = Arc::clone(&handler);
                tokio::spawn(async move {
                    let mut req_buf = [0u8; 2048];
                    let n = socket.read(&mut req_buf).await.unwrap_or(0);
                    let req_str = String::from_utf8_lossy(&req_buf[..n]);
                    let path = req_str
                        .lines()
                        .next()
                        .and_then(|l| l.split_whitespace().nth(1))
                        .unwrap_or("/");

                    let hit_no = hits_t.fetch_add(1, Ordering::AcqRel) + 1;
                    let cur = active_t.fetch_add(1, Ordering::AcqRel) + 1;
                    max_t.fetch_max(cur, Ordering::AcqRel);

                    let resp = handler_t(path, hit_no);
                    if !resp.delay.is_zero() {
                        tokio::select! {
                            _ = stop_t.cancelled() => {
                                active_t.fetch_sub(1, Ordering::AcqRel);
                                return;
                            }
                            _ = tokio::time::sleep(resp.delay) => {}
                        }
                    }

                    let reason = match resp.status {
                        200 => "OK",
                        204 => "No Content",
                        401 => "Unauthorized",
                        404 => "Not Found",
                        429 => "Too Many Requests",
                        500 => "Internal Server Error",
                        _ => "Status",
                    };
                    let header = format!(
                        "HTTP/1.1 {} {reason}\r\n{}Content-Length: {}\r\nConnection: close\r\n\r\n",
                        resp.status,
                        resp.headers,
                        resp.body.len()
                    );
                    let _ = socket.write_all(header.as_bytes()).await;
                    if !resp.body.is_empty() {
                        let _ = socket.write_all(&resp.body).await;
                    }
                    let _ = socket.shutdown().await;
                    active_t.fetch_sub(1, Ordering::AcqRel);
                });
            }
        });
    });

    let addr = addr_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("mock server addr");
    MockServer {
        addr,
        hits,
        max_concurrent,
        handle,
        stop,
    }
}

impl MockServer {
    fn shutdown(self) {
        self.stop.cancel();
        let _ = self.handle.join();
    }
}

fn custom_spec(addr: std::net::SocketAddr, max_level: u8) -> LiveBasemapSpec {
    LiveBasemapSpec::custom(CustomBasemapConfig {
        name: "Local Test",
        url_template: &format!("http://{addr}/{{z}}/{{x}}/{{y}}.png"),
        api_key: "",
        tms_flip_y: false,
        max_level,
        attribution: "Test",
    })
    .expect("custom spec")
}

#[test]
fn streams_mipmapped_tiles_enforces_host_limit_max_level_polar_and_disk_cache() {
    let payload = make_valid_tile_bytes(1);
    let payload_c = payload.clone();
    let server = spawn_mock_server(move |_, _| {
        MockResponse::ok_tile(payload_c.clone(), Duration::from_millis(25))
    });

    let disk_cache: Arc<dyn TileBlobStore> = Arc::new(MemBlobStore::default());
    let (done_tx, done_rx) = mpsc::channel();
    let wakes = Arc::new(AtomicUsize::new(0));
    let wakes_c = Arc::clone(&wakes);
    let wakeup: Wakeup = Arc::new(move || {
        wakes_c.fetch_add(1, Ordering::AcqRel);
    });
    let decoder: Arc<dyn TileDecodeExecutor> = Arc::new(TestDecodeExecutor {
        done_tx,
        wakeup: Arc::clone(&wakeup),
    });

    let cfg = FetcherConfig {
        max_concurrent_per_host: 2,
        ..FetcherConfig::default()
    };
    let pipeline = LiveTilePipeline::spawn(cfg, Some(Arc::clone(&disk_cache)), decoder, wakeup)
        .expect("spawn pipeline");

    let spec = custom_spec(server.addr, 2);
    pipeline.set_active_spec(Some(spec.clone()));

    let valid_keys: Vec<TileKey> = (0..4u32)
        .map(|x| {
            spec.tile_key(QuadTileId {
                scheme: PlanetaryScheme::WebMercator,
                level: 2,
                x,
                y: 0,
            })
        })
        .collect();
    let forbidden_polar = spec.tile_key(QuadTileId {
        scheme: PlanetaryScheme::PolarCap { north: true },
        level: 0,
        x: 0,
        y: 0,
    });
    let forbidden_above_max = spec.tile_key(QuadTileId {
        scheme: PlanetaryScheme::WebMercator,
        level: 3,
        x: 0,
        y: 0,
    });

    let mut wanted: HashSet<TileKey> = valid_keys.iter().copied().collect();
    wanted.insert(forbidden_polar);
    wanted.insert(forbidden_above_max);
    pipeline.reconcile_wanted(&wanted, |_| false);

    let mut received = HashMap::new();
    let deadline = Instant::now() + Duration::from_secs(5);
    while received.len() < 4 && Instant::now() < deadline {
        if let Ok((k, tile)) = done_rx.recv_timeout(Duration::from_millis(100)) {
            received.insert(k, tile);
        }
    }
    assert_eq!(received.len(), 4, "expected 4 valid live tiles");
    assert!(!received.contains_key(&forbidden_polar));
    assert!(!received.contains_key(&forbidden_above_max));

    for tile in received.values() {
        assert_eq!(tile.mip_level_count(), 4);
    }
    assert_eq!(server.hits.load(Ordering::Acquire), 4);
    assert!(
        server.max_concurrent.load(Ordering::Acquire) <= 2,
        "exceeded per-host limit: {}",
        server.max_concurrent.load(Ordering::Acquire)
    );
    let settle_deadline = Instant::now() + Duration::from_secs(2);
    while Instant::now() < settle_deadline {
        let all_cached = valid_keys.iter().all(|k| {
            disk_cache
                .get(spec.cache_key(), k.quad_id().expect("quad"))
                .ok()
                .flatten()
                .is_some()
        });
        if wakes.load(Ordering::Acquire) >= 4
            && pipeline.status().completed_from_net == 4
            && all_cached
        {
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(wakes.load(Ordering::Acquire) >= 4);
    assert_eq!(pipeline.status().completed_from_net, 4);

    let hits_before_cache = server.hits.load(Ordering::Acquire);
    let single_wanted = HashSet::from([valid_keys[0]]);
    pipeline.reconcile_wanted(&single_wanted, |_| false);
    let (cached_key, cached_tile) = done_rx
        .recv_timeout(Duration::from_secs(2))
        .expect("receive disk-cached tile");
    assert_eq!(cached_key, valid_keys[0]);
    assert_eq!(cached_tile.mip_level_count(), 4);
    assert_eq!(
        server.hits.load(Ordering::Acquire),
        hits_before_cache,
        "disk cache hit must trigger 0 HTTP requests"
    );
    let disk_deadline = Instant::now() + Duration::from_secs(2);
    while pipeline.status().completed_from_disk < 1 && Instant::now() < disk_deadline {
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(pipeline.status().completed_from_disk >= 1);

    pipeline.shutdown();
    server.shutdown();
}

#[test]
fn corrupt_disk_blob_self_heals_from_http_and_cached_tiles_do_not_starve_behind_slow_http() {
    let payload = make_valid_tile_bytes(2);
    let payload_c = payload.clone();
    let server = spawn_mock_server(move |path, _| {
        if path.contains("/2/1/0") {
            MockResponse::ok_tile(payload_c.clone(), Duration::from_secs(5))
        } else {
            MockResponse::ok_tile(payload_c.clone(), Duration::ZERO)
        }
    });

    let store = Arc::new(MemBlobStore::default());
    let spec = custom_spec(server.addr, 2);

    let k_corrupt = spec.tile_key(QuadTileId {
        scheme: PlanetaryScheme::WebMercator,
        level: 2,
        x: 0,
        y: 0,
    });
    let k_slow = spec.tile_key(QuadTileId {
        scheme: PlanetaryScheme::WebMercator,
        level: 2,
        x: 1,
        y: 0,
    });
    let k_cached = spec.tile_key(QuadTileId {
        scheme: PlanetaryScheme::WebMercator,
        level: 2,
        x: 2,
        y: 0,
    });

    store
        .put(
            spec.cache_key(),
            k_corrupt.quad_id().expect("quad"),
            b"not-a-valid-tile",
        )
        .expect("seed corrupt");
    store
        .put(
            spec.cache_key(),
            k_cached.quad_id().expect("quad"),
            &payload,
        )
        .expect("seed valid");

    let (done_tx, done_rx) = mpsc::channel();
    let wakeup: Wakeup = Arc::new(|| {});
    let decoder: Arc<dyn TileDecodeExecutor> = Arc::new(TestDecodeExecutor {
        done_tx,
        wakeup: Arc::clone(&wakeup),
    });
    let cfg = FetcherConfig {
        max_concurrent_per_host: 1,
        ..FetcherConfig::default()
    };
    let pipeline = LiveTilePipeline::spawn(
        cfg,
        Some(Arc::clone(&store) as Arc<dyn TileBlobStore>),
        decoder,
        wakeup,
    )
    .expect("spawn pipeline");
    pipeline.set_active_spec(Some(spec));

    pipeline.reconcile_wanted(&HashSet::from([k_corrupt]), |_| false);
    let (healed_key, _) = done_rx
        .recv_timeout(Duration::from_secs(3))
        .expect("corrupt disk tile should self-heal from HTTP");
    assert_eq!(healed_key, k_corrupt);

    pipeline.reconcile_wanted(&HashSet::from([k_slow]), |_| false);
    let wait_slow = Instant::now() + Duration::from_secs(2);
    while server.hits.load(Ordering::Acquire) < 2 && Instant::now() < wait_slow {
        std::thread::sleep(Duration::from_millis(5));
    }

    let start = Instant::now();
    pipeline.reconcile_wanted(&HashSet::from([k_slow, k_cached]), |_| false);
    let (fast_key, _) = done_rx
        .recv_timeout(Duration::from_secs(2))
        .expect("disk hit must not starve behind slow HTTP");
    assert_eq!(fast_key, k_cached);
    assert!(start.elapsed() < Duration::from_secs(1));

    pipeline.shutdown();
    server.shutdown();
}

#[test]
fn handles_http_204_404_corrupt_200_idle_retry_and_resets_failed_count() {
    let payload = make_valid_tile_bytes(3);
    let payload_c = payload.clone();
    let flaky_hits = Arc::new(AtomicUsize::new(0));
    let flaky_c = Arc::clone(&flaky_hits);
    let server = spawn_mock_server(move |path, _| {
        if path.contains("/1/0/0") {
            MockResponse {
                status: 204,
                headers: String::new(),
                body: Vec::new(),
                delay: Duration::ZERO,
            }
        } else if path.contains("/1/1/0") {
            MockResponse::ok_tile(b"<html>captive portal</html>".to_vec(), Duration::ZERO)
        } else if path.contains("/1/0/1") {
            let n = flaky_c.fetch_add(1, Ordering::AcqRel);
            if n == 0 {
                MockResponse {
                    status: 500,
                    headers: String::new(),
                    body: Vec::new(),
                    delay: Duration::ZERO,
                }
            } else {
                MockResponse::ok_tile(payload_c.clone(), Duration::ZERO)
            }
        } else {
            MockResponse {
                status: 404,
                headers: String::new(),
                body: Vec::new(),
                delay: Duration::ZERO,
            }
        }
    });

    let store = Arc::new(MemBlobStore::default());
    let (done_tx, done_rx) = mpsc::channel();
    let wakeup: Wakeup = Arc::new(|| {});
    let decoder: Arc<dyn TileDecodeExecutor> = Arc::new(TestDecodeExecutor {
        done_tx,
        wakeup: Arc::clone(&wakeup),
    });
    let cfg = FetcherConfig {
        max_concurrent_per_host: 4,
        transient_retry_backoff: Duration::from_millis(80),
        ..FetcherConfig::default()
    };
    let pipeline = LiveTilePipeline::spawn(
        cfg,
        Some(Arc::clone(&store) as Arc<dyn TileBlobStore>),
        decoder,
        wakeup,
    )
    .expect("spawn pipeline");
    let spec = custom_spec(server.addr, 2);
    pipeline.set_active_spec(Some(spec.clone()));

    let k_204 = spec.tile_key(QuadTileId {
        scheme: PlanetaryScheme::WebMercator,
        level: 1,
        x: 0,
        y: 0,
    });
    let k_corrupt = spec.tile_key(QuadTileId {
        scheme: PlanetaryScheme::WebMercator,
        level: 1,
        x: 1,
        y: 0,
    });
    let k_flaky = spec.tile_key(QuadTileId {
        scheme: PlanetaryScheme::WebMercator,
        level: 1,
        x: 0,
        y: 1,
    });

    pipeline.reconcile_wanted(&HashSet::from([k_204]), |_| false);
    let d1 = Instant::now() + Duration::from_secs(2);
    while {
        let st = pipeline.status();
        (st.in_flight + st.queued > 0 || server.hits.load(Ordering::Acquire) < 1)
            && Instant::now() < d1
    } {
        std::thread::sleep(Duration::from_millis(10));
    }
    assert_eq!(pipeline.status().failed_count, 0);
    assert!(pipeline.status().last_error.is_none());
    let hits_after_204 = server.hits.load(Ordering::Acquire);
    pipeline.reconcile_wanted(&HashSet::from([k_204]), |_| false);
    std::thread::sleep(Duration::from_millis(40));
    assert_eq!(server.hits.load(Ordering::Acquire), hits_after_204);

    pipeline.reconcile_wanted(&HashSet::from([k_corrupt]), |_| false);
    let d2 = Instant::now() + Duration::from_secs(2);
    while pipeline.status().failed_count == 0 && Instant::now() < d2 {
        std::thread::sleep(Duration::from_millis(10));
    }
    assert_eq!(pipeline.status().failed_count, 1);
    assert!(
        store
            .get(spec.cache_key(), k_corrupt.quad_id().expect("quad"))
            .expect("get")
            .is_none()
    );

    pipeline.reconcile_wanted(&HashSet::from([k_flaky]), |_| false);
    let (retried_key, _) = done_rx
        .recv_timeout(Duration::from_secs(3))
        .expect("flaky tile must succeed on automatic idle backoff retry");
    assert_eq!(retried_key, k_flaky);
    assert_eq!(flaky_hits.load(Ordering::Acquire), 2);

    pipeline.retry_failed();
    assert_eq!(pipeline.status().failed_count, 0);
    assert!(pipeline.status().last_error.is_none());

    pipeline.shutdown();
    server.shutdown();
}

#[test]
fn cancel_token_aborts_slow_inflight_request_promptly_on_shutdown() {
    let payload = make_valid_tile_bytes(4);
    let server = spawn_mock_server(move |_, _| {
        MockResponse::ok_tile(payload.clone(), Duration::from_secs(10))
    });

    let (done_tx, _done_rx) = mpsc::channel();
    let wakeup: Wakeup = Arc::new(|| {});
    let decoder: Arc<dyn TileDecodeExecutor> = Arc::new(TestDecodeExecutor {
        done_tx,
        wakeup: Arc::clone(&wakeup),
    });
    let pipeline = LiveTilePipeline::spawn(FetcherConfig::default(), None, decoder, wakeup)
        .expect("spawn pipeline");
    let spec = custom_spec(server.addr, 1);
    pipeline.set_active_spec(Some(spec.clone()));

    let k0 = spec.tile_key(QuadTileId {
        scheme: PlanetaryScheme::WebMercator,
        level: 0,
        x: 0,
        y: 0,
    });
    pipeline.reconcile_wanted(&HashSet::from([k0]), |_| false);

    let wait_hit = Instant::now() + Duration::from_secs(2);
    while server.hits.load(Ordering::Acquire) == 0 && Instant::now() < wait_hit {
        std::thread::sleep(Duration::from_millis(10));
    }
    assert_eq!(server.hits.load(Ordering::Acquire), 1);

    let start = Instant::now();
    pipeline.shutdown();
    let elapsed = start.elapsed();
    assert!(
        elapsed < Duration::from_millis(800),
        "shutdown should abort slow HTTP promptly, took {elapsed:?}"
    );

    server.shutdown();
}

struct DropFirstDecodeExecutor {
    dropped_once: AtomicUsize,
    inner: TestDecodeExecutor,
}

impl TileDecodeExecutor for DropFirstDecodeExecutor {
    fn submit(&self, req: DecodeRequest) -> bool {
        if self.dropped_once.fetch_add(1, Ordering::AcqRel) == 0 {
            // Simulate post-acceptance queue eviction or panic dropping `DecodeRequest`.
            drop(req);
            return true;
        }
        self.inner.submit(req)
    }
}

#[test]
fn dropped_or_evicted_decode_request_requeues_instead_of_orphaning() {
    let payload = make_valid_tile_bytes(5);
    let server =
        spawn_mock_server(move |_, _| MockResponse::ok_tile(payload.clone(), Duration::ZERO));

    let (done_tx, done_rx) = mpsc::channel();
    let wakeup: Wakeup = Arc::new(|| {});
    let decoder: Arc<dyn TileDecodeExecutor> = Arc::new(DropFirstDecodeExecutor {
        dropped_once: AtomicUsize::new(0),
        inner: TestDecodeExecutor {
            done_tx,
            wakeup: Arc::clone(&wakeup),
        },
    });
    let pipeline = LiveTilePipeline::spawn(FetcherConfig::default(), None, decoder, wakeup)
        .expect("spawn pipeline");
    let spec = custom_spec(server.addr, 1);
    pipeline.set_active_spec(Some(spec.clone()));

    let k0 = spec.tile_key(QuadTileId {
        scheme: PlanetaryScheme::WebMercator,
        level: 0,
        x: 0,
        y: 0,
    });
    pipeline.reconcile_wanted(&HashSet::from([k0]), |_| false);

    let (got_key, _) = done_rx
        .recv_timeout(Duration::from_secs(2))
        .expect("evicted decode must re-queue via backoff and complete");
    assert_eq!(got_key, k0);
    assert_eq!(pipeline.status().failed_count, 0);

    pipeline.shutdown();
    server.shutdown();
}
