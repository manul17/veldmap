use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant};

use veldmap_core::sync::{CancelToken, Wakeup, lock_unpoisoned};
use veldmap_core::tile::{DecodedTile, PixelFormat, PlanetaryScheme, QuadTileId, TileKey};
use veldmap_net::{
    BlobLease, BlobStoreError, CustomBasemapConfig, DecodeFailure, DecodeRequest, FetcherConfig,
    LiveBasemapSpec, LiveTilePipeline, NetConfig, NetContext, TileBlobStore, TileDecodeExecutor,
};

use crate::mock_http::{MockHttpServer, write_http_response};

type BlobMapKey = (String, PlanetaryScheme, u8, u32, u32);

const TEST_TILE_MAGIC: &[u8] = b"VELDMAP_TEST_TILE_8X8";

#[derive(Default)]
struct MemBlobStore {
    map: Mutex<HashMap<BlobMapKey, (u64, Vec<u8>)>>,
    next_rev: AtomicUsize,
}

impl MemBlobStore {
    fn has_entry(&self, source_key: &str, id: QuadTileId) -> bool {
        lock_unpoisoned(&self.map).contains_key(&(
            source_key.to_owned(),
            id.scheme,
            id.level,
            id.x,
            id.y,
        ))
    }

    fn seed(&self, source_key: &str, id: QuadTileId, payload: &[u8]) {
        self.put(source_key, id, payload, &CancelToken::new())
            .expect("seed blob");
    }
}

impl TileBlobStore for MemBlobStore {
    fn get(
        &self,
        source_key: &str,
        id: QuadTileId,
    ) -> Result<Option<(Vec<u8>, BlobLease)>, BlobStoreError> {
        let key = (source_key.to_owned(), id.scheme, id.level, id.x, id.y);
        let map = lock_unpoisoned(&self.map);
        Ok(map
            .get(&key)
            .map(|(rev, bytes)| (bytes.clone(), BlobLease::new((key, *rev)))))
    }

    fn touch(&self, _lease: BlobLease) {}

    fn put(
        &self,
        source_key: &str,
        id: QuadTileId,
        payload: &[u8],
        cancel: &CancelToken,
    ) -> Result<(), BlobStoreError> {
        if cancel.is_cancelled() {
            return Ok(());
        }
        let rev = self.next_rev.fetch_add(1, Ordering::AcqRel) as u64 + 1;
        let mut map = lock_unpoisoned(&self.map);
        map.insert(
            (source_key.to_owned(), id.scheme, id.level, id.x, id.y),
            (rev, payload.to_vec()),
        );
        Ok(())
    }

    fn delete(&self, lease: BlobLease) -> Result<(), BlobStoreError> {
        let Some((key, expected_rev)) = lease.downcast::<(BlobMapKey, u64)>() else {
            return Ok(());
        };
        let mut map = lock_unpoisoned(&self.map);
        if map.get(&key).is_some_and(|(rev, _)| *rev == expected_rev) {
            map.remove(&key);
        }
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
    hits: Arc<AtomicUsize>,
    max_concurrent: Arc<AtomicUsize>,
    server: MockHttpServer,
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

    let hits_c = Arc::clone(&hits);
    let active_c = Arc::clone(&active);
    let max_c = Arc::clone(&max_concurrent);
    let handler = Arc::new(handler);

    let server = MockHttpServer::start(move |req, mut socket, stop_t| {
        let hits_t = Arc::clone(&hits_c);
        let active_t = Arc::clone(&active_c);
        let max_t = Arc::clone(&max_c);
        let handler_t = Arc::clone(&handler);
        async move {
            let hit_no = hits_t.fetch_add(1, Ordering::AcqRel) + 1;
            let cur = active_t.fetch_add(1, Ordering::AcqRel) + 1;
            max_t.fetch_max(cur, Ordering::AcqRel);

            let resp = handler_t(&req.path_and_query, hit_no);
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
            write_http_response(&mut socket, resp.status, reason, &resp.headers, &resp.body).await;
            active_t.fetch_sub(1, Ordering::AcqRel);
        }
    });

    MockServer {
        hits,
        max_concurrent,
        server,
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

/// RAII harness owning a [`LiveTilePipeline`], its shared [`NetContext`], and
/// its loopback [`MockServer`], guaranteeing `pipeline` shuts down before
/// `net` and `net` shuts down before `server` even on panic.
struct TestPipelineHarness {
    pipeline: Option<LiveTilePipeline>,
    net: Option<Arc<NetContext>>,
    server: Option<MockServer>,
    spec: LiveBasemapSpec,
    store: Option<Arc<MemBlobStore>>,
    done_rx: mpsc::Receiver<(TileKey, DecodedTile)>,
    wakes: Arc<AtomicUsize>,
}

impl TestPipelineHarness {
    fn spawn<F>(
        max_concurrent_per_host: usize,
        cfg: FetcherConfig,
        with_disk_cache: bool,
        max_level: u8,
        handler: F,
    ) -> Self
    where
        F: Fn(&str, usize) -> MockResponse + Send + Sync + 'static,
    {
        Self::spawn_with_decoder(
            max_concurrent_per_host,
            cfg,
            with_disk_cache,
            max_level,
            handler,
            |done_tx, wakeup| Arc::new(TestDecodeExecutor { done_tx, wakeup }),
        )
    }

    fn spawn_with_decoder<F, D>(
        max_concurrent_per_host: usize,
        cfg: FetcherConfig,
        with_disk_cache: bool,
        max_level: u8,
        handler: F,
        make_decoder: D,
    ) -> Self
    where
        F: Fn(&str, usize) -> MockResponse + Send + Sync + 'static,
        D: FnOnce(mpsc::Sender<(TileKey, DecodedTile)>, Wakeup) -> Arc<dyn TileDecodeExecutor>,
    {
        let server = spawn_mock_server(handler);
        let spec = custom_spec(server.server.addr(), max_level);
        let store = with_disk_cache.then(|| Arc::new(MemBlobStore::default()));
        let disk_cache = store
            .as_ref()
            .map(|s| Arc::clone(s) as Arc<dyn TileBlobStore>);
        let (done_tx, done_rx) = mpsc::channel();
        let wakes = Arc::new(AtomicUsize::new(0));
        let wakes_c = Arc::clone(&wakes);
        let wakeup: Wakeup = Arc::new(move || {
            wakes_c.fetch_add(1, Ordering::AcqRel);
        });
        let decoder = make_decoder(done_tx, Arc::clone(&wakeup));
        let net = Arc::new(
            NetContext::new(NetConfig {
                max_concurrent_per_host,
                ..NetConfig::default()
            })
            .expect("net context"),
        );
        let pipeline = LiveTilePipeline::spawn_with_context(
            Arc::clone(&net),
            cfg,
            disk_cache,
            decoder,
            wakeup,
        );
        pipeline.set_active_spec(Some(spec.clone()));

        Self {
            pipeline: Some(pipeline),
            net: Some(net),
            server: Some(server),
            spec,
            store,
            done_rx,
            wakes,
        }
    }

    fn pipeline(&self) -> &LiveTilePipeline {
        self.pipeline.as_ref().expect("pipeline active")
    }

    fn server(&self) -> &MockServer {
        self.server.as_ref().expect("server active")
    }

    fn store(&self) -> &Arc<MemBlobStore> {
        self.store.as_ref().expect("disk cache enabled")
    }

    fn wm_key(&self, level: u8, x: u32, y: u32) -> TileKey {
        self.spec.tile_key(QuadTileId {
            scheme: PlanetaryScheme::WebMercator,
            level,
            x,
            y,
        })
    }

    fn shutdown_pipeline(&mut self) {
        if let Some(pipeline) = self.pipeline.take() {
            pipeline.shutdown();
        }
        if let Some(net) = self.net.take() {
            net.shutdown();
        }
    }
}

impl Drop for TestPipelineHarness {
    fn drop(&mut self) {
        self.shutdown_pipeline();
        drop(self.server.take());
    }
}

#[test]
fn streams_mipmapped_tiles_enforces_host_limit_max_level_polar_and_disk_cache() {
    let payload = make_valid_tile_bytes(1);
    let payload_c = payload.clone();
    let h = TestPipelineHarness::spawn(2, FetcherConfig::default(), true, 2, move |_, _| {
        MockResponse::ok_tile(payload_c.clone(), Duration::from_millis(25))
    });

    let valid_keys: Vec<TileKey> = (0..4u32).map(|x| h.wm_key(2, x, 0)).collect();
    let forbidden_polar = h.spec.tile_key(QuadTileId {
        scheme: PlanetaryScheme::PolarCap { north: true },
        level: 0,
        x: 0,
        y: 0,
    });
    let forbidden_above_max = h.wm_key(3, 0, 0);

    let mut wanted: HashSet<TileKey> = valid_keys.iter().copied().collect();
    wanted.insert(forbidden_polar);
    wanted.insert(forbidden_above_max);
    h.pipeline().reconcile_wanted(&wanted, |_| false);

    let mut received = HashMap::new();
    let deadline = Instant::now() + Duration::from_secs(5);
    while received.len() < 4 && Instant::now() < deadline {
        if let Ok((k, tile)) = h.done_rx.recv_timeout(Duration::from_millis(100)) {
            received.insert(k, tile);
        }
    }
    assert_eq!(received.len(), 4, "expected 4 valid live tiles");
    assert!(!received.contains_key(&forbidden_polar));
    assert!(!received.contains_key(&forbidden_above_max));

    for tile in received.values() {
        assert_eq!(tile.mip_level_count(), 4);
    }
    assert_eq!(h.server().hits.load(Ordering::Acquire), 4);
    assert!(
        h.server().max_concurrent.load(Ordering::Acquire) <= 2,
        "exceeded per-host limit: {}",
        h.server().max_concurrent.load(Ordering::Acquire)
    );
    let settle_deadline = Instant::now() + Duration::from_secs(2);
    while Instant::now() < settle_deadline {
        let all_cached = valid_keys.iter().all(|k| {
            h.store()
                .has_entry(h.spec.cache_key(), k.quad_id().expect("quad"))
        });
        if h.wakes.load(Ordering::Acquire) >= 4
            && h.pipeline().status().completed_from_net == 4
            && all_cached
        {
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(h.wakes.load(Ordering::Acquire) >= 4);
    assert_eq!(h.pipeline().status().completed_from_net, 4);

    let hits_before_cache = h.server().hits.load(Ordering::Acquire);
    let single_wanted = HashSet::from([valid_keys[0]]);
    h.pipeline().reconcile_wanted(&single_wanted, |_| false);
    let (cached_key, cached_tile) = h
        .done_rx
        .recv_timeout(Duration::from_secs(2))
        .expect("receive disk-cached tile");
    assert_eq!(cached_key, valid_keys[0]);
    assert_eq!(cached_tile.mip_level_count(), 4);
    assert_eq!(
        h.server().hits.load(Ordering::Acquire),
        hits_before_cache,
        "disk cache hit must trigger 0 HTTP requests"
    );
    let disk_deadline = Instant::now() + Duration::from_secs(2);
    while h.pipeline().status().completed_from_disk < 1 && Instant::now() < disk_deadline {
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(h.pipeline().status().completed_from_disk >= 1);
}

#[test]
fn corrupt_disk_blob_self_heals_from_http_and_cached_tiles_do_not_starve_behind_slow_http() {
    let payload = make_valid_tile_bytes(2);
    let payload_c = payload.clone();
    let h = TestPipelineHarness::spawn(1, FetcherConfig::default(), true, 2, move |path, _| {
        if path.contains("/2/1/0") {
            MockResponse::ok_tile(payload_c.clone(), Duration::from_secs(5))
        } else {
            MockResponse::ok_tile(payload_c.clone(), Duration::ZERO)
        }
    });

    let k_corrupt = h.wm_key(2, 0, 0);
    let k_slow = h.wm_key(2, 1, 0);
    let k_cached = h.wm_key(2, 2, 0);

    h.store().seed(
        h.spec.cache_key(),
        k_corrupt.quad_id().expect("quad"),
        b"not-a-valid-tile",
    );
    h.store().seed(
        h.spec.cache_key(),
        k_cached.quad_id().expect("quad"),
        &payload,
    );

    h.pipeline()
        .reconcile_wanted(&HashSet::from([k_corrupt]), |_| false);
    let (healed_key, _) = h
        .done_rx
        .recv_timeout(Duration::from_secs(3))
        .expect("corrupt disk tile should self-heal from HTTP");
    assert_eq!(healed_key, k_corrupt);

    h.pipeline()
        .reconcile_wanted(&HashSet::from([k_slow]), |_| false);
    let wait_slow = Instant::now() + Duration::from_secs(2);
    while h.server().hits.load(Ordering::Acquire) < 2 && Instant::now() < wait_slow {
        std::thread::sleep(Duration::from_millis(5));
    }

    let start = Instant::now();
    h.pipeline()
        .reconcile_wanted(&HashSet::from([k_slow, k_cached]), |_| false);
    let (fast_key, _) = h
        .done_rx
        .recv_timeout(Duration::from_secs(2))
        .expect("disk hit must not starve behind slow HTTP");
    assert_eq!(fast_key, k_cached);
    assert!(start.elapsed() < Duration::from_secs(1));
}

#[test]
fn handles_http_204_404_corrupt_200_idle_retry_and_resets_failed_count() {
    let payload = make_valid_tile_bytes(3);
    let payload_c = payload.clone();
    let flaky_hits = Arc::new(AtomicUsize::new(0));
    let flaky_c = Arc::clone(&flaky_hits);
    let cfg = FetcherConfig {
        transient_retry_backoff: Duration::from_millis(80),
        ..FetcherConfig::default()
    };
    let h = TestPipelineHarness::spawn(4, cfg, true, 2, move |path, _| {
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

    let k_204 = h.wm_key(1, 0, 0);
    let k_corrupt = h.wm_key(1, 1, 0);
    let k_flaky = h.wm_key(1, 0, 1);

    h.pipeline()
        .reconcile_wanted(&HashSet::from([k_204]), |_| false);
    let d1 = Instant::now() + Duration::from_secs(2);
    while {
        let st = h.pipeline().status();
        (st.in_flight + st.queued > 0 || h.server().hits.load(Ordering::Acquire) < 1)
            && Instant::now() < d1
    } {
        std::thread::sleep(Duration::from_millis(10));
    }
    assert_eq!(h.pipeline().status().failed_count, 0);
    assert!(h.pipeline().status().last_error.is_none());
    let hits_after_204 = h.server().hits.load(Ordering::Acquire);
    h.pipeline()
        .reconcile_wanted(&HashSet::from([k_204]), |_| false);
    std::thread::sleep(Duration::from_millis(40));
    assert_eq!(h.server().hits.load(Ordering::Acquire), hits_after_204);

    h.pipeline()
        .reconcile_wanted(&HashSet::from([k_corrupt]), |_| false);
    let d2 = Instant::now() + Duration::from_secs(2);
    while h.pipeline().status().failed_count == 0 && Instant::now() < d2 {
        std::thread::sleep(Duration::from_millis(10));
    }
    assert_eq!(h.pipeline().status().failed_count, 1);
    assert!(
        !h.store()
            .has_entry(h.spec.cache_key(), k_corrupt.quad_id().expect("quad"))
    );

    h.pipeline()
        .reconcile_wanted(&HashSet::from([k_flaky]), |_| false);
    let (retried_key, _) = h
        .done_rx
        .recv_timeout(Duration::from_secs(3))
        .expect("flaky tile must succeed on automatic idle backoff retry");
    assert_eq!(retried_key, k_flaky);
    assert_eq!(flaky_hits.load(Ordering::Acquire), 2);

    h.pipeline().retry_failed();
    assert_eq!(h.pipeline().status().failed_count, 0);
    assert!(h.pipeline().status().last_error.is_none());
}

#[test]
fn cancel_token_aborts_slow_inflight_request_promptly_on_shutdown() {
    let payload = make_valid_tile_bytes(4);
    let mut h = TestPipelineHarness::spawn(
        NetConfig::default().max_concurrent_per_host,
        FetcherConfig::default(),
        false,
        1,
        move |_, _| MockResponse::ok_tile(payload.clone(), Duration::from_secs(10)),
    );

    let k0 = h.wm_key(0, 0, 0);
    h.pipeline()
        .reconcile_wanted(&HashSet::from([k0]), |_| false);

    let wait_hit = Instant::now() + Duration::from_secs(2);
    while h.server().hits.load(Ordering::Acquire) == 0 && Instant::now() < wait_hit {
        std::thread::sleep(Duration::from_millis(10));
    }
    assert_eq!(h.server().hits.load(Ordering::Acquire), 1);

    let start = Instant::now();
    h.shutdown_pipeline();
    let elapsed = start.elapsed();
    assert!(
        elapsed < Duration::from_millis(800),
        "shutdown should abort slow HTTP promptly, took {elapsed:?}"
    );
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
    let h = TestPipelineHarness::spawn_with_decoder(
        NetConfig::default().max_concurrent_per_host,
        FetcherConfig::default(),
        false,
        1,
        move |_, _| MockResponse::ok_tile(payload.clone(), Duration::ZERO),
        |done_tx, wakeup| {
            Arc::new(DropFirstDecodeExecutor {
                dropped_once: AtomicUsize::new(0),
                inner: TestDecodeExecutor { done_tx, wakeup },
            })
        },
    );

    let k0 = h.wm_key(0, 0, 0);
    h.pipeline()
        .reconcile_wanted(&HashSet::from([k0]), |_| false);

    let (got_key, _) = h
        .done_rx
        .recv_timeout(Duration::from_secs(2))
        .expect("evicted decode must re-queue via backoff and complete");
    assert_eq!(got_key, k0);
    assert_eq!(h.pipeline().status().failed_count, 0);
}
