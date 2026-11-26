//! Offline local mock HTTP Range integration harness verifying RFC 9110
//! `206` / `200` rejection / `416` 0-byte / strong vs. weak `ETag` & `If-Match` /
//! `412` non-retryability / `Content-Encoding: identity` enforcement / N-way
//! singleflight deduplication + refcounted follower cancellation / pinned
//! coalesced prefetch / Tokio-thread guard / request & delivered-byte ceilings.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::Notify;
use veldmap_core::io::{ByteSource, IoError};
use veldmap_core::sync::CancelToken;
use veldmap_net::{
    HttpRangeConfig, HttpRangeSource, NetConfig, NetContext, RANGE_BLOCK_SIZE, RangeOpenError,
};

#[derive(Clone, Debug)]
struct RecordedReq {
    if_match: Option<String>,
    accept_encoding: Option<String>,
}

struct MockRangeState {
    data: Mutex<Vec<u8>>,
    etag: Mutex<Option<String>>,
    force_200_on_range: AtomicBool,
    force_gzip_header: AtomicBool,
    enforce_if_match: AtomicBool,
    delay_gate: Mutex<Option<Arc<Notify>>>,
    requests: Mutex<Vec<RecordedReq>>,
    range_gets_started: AtomicUsize,
}

impl MockRangeState {
    fn new(data: Vec<u8>, etag: Option<&str>) -> Arc<Self> {
        Arc::new(Self {
            data: Mutex::new(data),
            etag: Mutex::new(etag.map(str::to_owned)),
            force_200_on_range: AtomicBool::new(false),
            force_gzip_header: AtomicBool::new(false),
            enforce_if_match: AtomicBool::new(true),
            delay_gate: Mutex::new(None),
            requests: Mutex::new(Vec::new()),
            range_gets_started: AtomicUsize::new(0),
        })
    }

    fn request_count(&self) -> usize {
        self.requests.lock().expect("lock").len()
    }

    fn recorded(&self) -> Vec<RecordedReq> {
        self.requests.lock().expect("lock").clone()
    }
}

struct MockRangeServer {
    url: String,
    shutdown: CancelToken,
    handle: Option<thread::JoinHandle<()>>,
}

impl MockRangeServer {
    fn start(state: Arc<MockRangeState>) -> Self {
        let shutdown = CancelToken::new();
        let stop = shutdown.clone();
        let st = Arc::clone(&state);
        let (addr_tx, addr_rx) = std::sync::mpsc::channel();

        let handle = thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("mock server runtime");
            rt.block_on(async move {
                let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
                let addr = listener.local_addr().expect("local_addr");
                let _ = addr_tx.send(addr);

                loop {
                    tokio::select! {
                        biased;
                        _ = stop.cancelled() => break,
                        res = listener.accept() => {
                            let Ok((mut stream, _)) = res else { break; };
                            let st_conn = Arc::clone(&st);
                            let stop_conn = stop.clone();
                            tokio::spawn(async move {
                                let mut buf = vec![0u8; 8192];
                                let mut filled = 0usize;
                                loop {
                                    let n = tokio::select! {
                                        _ = stop_conn.cancelled() => return,
                                        r = stream.read(&mut buf[filled..]) => match r {
                                            Ok(0) | Err(_) => return,
                                            Ok(n) => n,
                                        }
                                    };
                                    filled += n;
                                    if buf[..filled].windows(4).any(|w| w == b"\r\n\r\n") || filled == buf.len() {
                                        break;
                                    }
                                }
                                let req_str = String::from_utf8_lossy(&buf[..filled]);
                                let mut range = None;
                                let mut if_match = None;
                                let mut accept_encoding = None;
                                for line in req_str.lines().skip(1) {
                                    if let Some((k, v)) = line.split_once(':') {
                                        let key = k.trim().to_ascii_lowercase();
                                        let val = v.trim().to_owned();
                                        match key.as_str() {
                                            "range" => range = Some(val),
                                            "if-match" => if_match = Some(val),
                                            "accept-encoding" => accept_encoding = Some(val),
                                            _ => {}
                                        }
                                    }
                                }

                                st_conn.requests.lock().expect("lock").push(RecordedReq {
                                    if_match: if_match.clone(),
                                    accept_encoding,
                                });
                                st_conn.range_gets_started.fetch_add(1, Ordering::SeqCst);

                                let gate = st_conn.delay_gate.lock().expect("lock").clone();
                                if let Some(notify) = gate {
                                    tokio::select! {
                                        _ = stop_conn.cancelled() => return,
                                        _ = notify.notified() => {}
                                    }
                                }

                                let data = st_conn.data.lock().expect("lock").clone();
                                let etag = st_conn.etag.lock().expect("lock").clone();

                                if st_conn.enforce_if_match.load(Ordering::SeqCst)
                                    && let Some(ref req_etag) = if_match
                                    && etag.as_deref() != Some(req_etag.as_str())
                                {
                                    let resp = "HTTP/1.1 412 Precondition Failed\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
                                    let _ = stream.write_all(resp.as_bytes()).await;
                                    return;
                                }

                                if st_conn.force_200_on_range.load(Ordering::SeqCst) {
                                    let header = format!(
                                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                                        data.len()
                                    );
                                    let _ = stream.write_all(header.as_bytes()).await;
                                    let _ = stream.write_all(&data).await;
                                    return;
                                }

                                let Some(range_val) = range else {
                                    let resp = "HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
                                    let _ = stream.write_all(resp.as_bytes()).await;
                                    return;
                                };

                                let total = data.len() as u64;
                                if total == 0 {
                                    let mut hdr = String::from(
                                        "HTTP/1.1 416 Range Not Satisfiable\r\nContent-Range: bytes */0\r\nContent-Length: 0\r\n",
                                    );
                                    if let Some(ref e) = etag {
                                        hdr.push_str(&format!("ETag: {e}\r\n"));
                                    }
                                    hdr.push_str("Connection: close\r\n\r\n");
                                    let _ = stream.write_all(hdr.as_bytes()).await;
                                    return;
                                }

                                let (start, req_end) = range_val
                                    .strip_prefix("bytes=")
                                    .and_then(|s| s.split_once('-'))
                                    .and_then(|(a, b)| Some((a.parse::<u64>().ok()?, b.parse::<u64>().ok()?)))
                                    .expect("valid Range header");
                                if start >= total {
                                    let hdr = format!(
                                        "HTTP/1.1 416 Range Not Satisfiable\r\nContent-Range: bytes */{total}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                                    );
                                    let _ = stream.write_all(hdr.as_bytes()).await;
                                    return;
                                }
                                let end = req_end.min(total - 1);
                                let slice = &data[start as usize..=end as usize];
                                let mut hdr = format!(
                                    "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {start}-{end}/{total}\r\nContent-Length: {}\r\n",
                                    slice.len()
                                );
                                if let Some(ref e) = etag {
                                    hdr.push_str(&format!("ETag: {e}\r\n"));
                                }
                                if st_conn.force_gzip_header.load(Ordering::SeqCst) {
                                    hdr.push_str("Content-Encoding: gzip\r\n");
                                } else {
                                    hdr.push_str("Content-Encoding: identity\r\n");
                                }
                                hdr.push_str("Connection: close\r\n\r\n");
                                let _ = stream.write_all(hdr.as_bytes()).await;
                                let _ = stream.write_all(slice).await;
                            });
                        }
                    }
                }
            });
        });

        let addr = addr_rx.recv().expect("server addr");
        Self {
            url: format!("http://{addr}/scene/B04.tif?sig=secret_token"),
            shutdown,
            handle: Some(handle),
        }
    }
}

impl Drop for MockRangeServer {
    fn drop(&mut self) {
        self.shutdown.cancel();
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn pattern_bytes(len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| ((i.wrapping_mul(31) ^ (i >> 8)) & 0xFF) as u8)
        .collect()
}

fn test_net(pool_bytes: usize) -> Arc<NetContext> {
    Arc::new(
        NetContext::new(NetConfig {
            range_block_pool_bytes: pool_bytes,
            remote_prefetch_bytes: 8 * 1024 * 1024,
            ..NetConfig::default()
        })
        .expect("net context"),
    )
}

#[test]
fn range_open_reads_head_in_one_rtt_enforces_identity_and_handles_zero_byte_416() {
    let payload = pattern_bytes(512 * 1024 + 123);
    let state = MockRangeState::new(payload.clone(), Some("\"etag-v1\""));
    let server = MockRangeServer::start(Arc::clone(&state));
    let net = test_net(4 * 1024 * 1024);
    let cancel = CancelToken::new();

    let src = HttpRangeSource::open(
        Arc::clone(&net),
        &server.url,
        HttpRangeConfig::default(),
        &cancel,
    )
    .expect("open 206");
    assert_eq!(src.len(), payload.len() as u64);
    assert!(src.is_remote());
    assert_eq!(src.http_requests(), 1, "open must take exactly 1 RTT");
    assert_eq!(src.strong_etag(), Some("\"etag-v1\""));
    assert!(
        !src.canonical_url().contains("secret_token"),
        "canonical URL must strip query secrets"
    );

    // Reading inside the pinned 256 KiB head must perform 0 additional HTTP requests.
    let mut head_buf = vec![0u8; 4096];
    src.read_exact_at(1024, &mut head_buf, &cancel)
        .expect("head read");
    assert_eq!(head_buf, payload[1024..5120]);
    assert_eq!(src.http_requests(), 1);

    // Random read past head fetches only the needed block (no speculative readahead on jump)
    // and sends `Accept-Encoding: identity` + `If-Match: "etag-v1"`.
    let tail_off = (512 * 1024) as u64;
    let mut tail_buf = vec![0u8; 123];
    src.read_exact_at(tail_off, &mut tail_buf, &cancel)
        .expect("tail read");
    assert_eq!(tail_buf, payload[512 * 1024..]);
    assert_eq!(src.http_requests(), 2);
    assert!(
        src.bytes_delivered() <= (256 * 1024 + 123) as u64,
        "delivered bytes must stay tightly bounded: {}",
        src.bytes_delivered()
    );
    let reqs = state.recorded();
    assert_eq!(reqs[0].accept_encoding.as_deref(), Some("identity"));
    assert_eq!(reqs[1].accept_encoding.as_deref(), Some("identity"));
    assert_eq!(reqs[1].if_match.as_deref(), Some("\"etag-v1\""));

    // 416 with `Content-Range: bytes */0` opens cleanly as a 0-byte source.
    let empty_server = MockRangeServer::start(MockRangeState::new(Vec::new(), Some("\"empty\"")));
    let empty_src = HttpRangeSource::open(
        Arc::clone(&net),
        &empty_server.url,
        HttpRangeConfig::default(),
        &cancel,
    )
    .expect("0-byte 416 open");
    assert_eq!(empty_src.len(), 0);
    assert!(empty_src.is_empty());
    let mut tmp = [0u8; 16];
    assert_eq!(empty_src.read_at(0, &mut tmp, &cancel).expect("eof"), 0);
}

#[test]
fn rejects_200_ok_non_identity_encoding_and_non_retryable_412() {
    let payload = pattern_bytes(384 * 1024);
    let cancel = CancelToken::new();
    let net = test_net(2 * 1024 * 1024);

    // 1. 200 OK rejected on open.
    let s200 = MockRangeState::new(payload.clone(), Some("\"v1\""));
    s200.force_200_on_range.store(true, Ordering::SeqCst);
    let srv200 = MockRangeServer::start(Arc::clone(&s200));
    let err = HttpRangeSource::open(
        Arc::clone(&net),
        &srv200.url,
        HttpRangeConfig::default(),
        &cancel,
    )
    .expect_err("200 OK must be rejected");
    assert!(matches!(err, RangeOpenError::RangeNotSupported(_)));

    // 2. Non-identity Content-Encoding rejected on open.
    let sgzip = MockRangeState::new(payload.clone(), Some("\"v1\""));
    sgzip.force_gzip_header.store(true, Ordering::SeqCst);
    let srvgzip = MockRangeServer::start(Arc::clone(&sgzip));
    let err = HttpRangeSource::open(
        Arc::clone(&net),
        &srvgzip.url,
        HttpRangeConfig::default(),
        &cancel,
    )
    .expect_err("gzip encoding must be rejected");
    assert!(matches!(err, RangeOpenError::InvalidResponse(_)));

    // 3. Strong ETag changes -> 412 Precondition Failed is non-retryable (1 request, 0 retries).
    let s412 = MockRangeState::new(payload, Some("\"v1\""));
    let srv412 = MockRangeServer::start(Arc::clone(&s412));
    let src = HttpRangeSource::open(
        Arc::clone(&net),
        &srv412.url,
        HttpRangeConfig::default(),
        &cancel,
    )
    .expect("open v1");
    *s412.etag.lock().expect("lock") = Some("\"v2\"".to_owned());
    let mut buf = [0u8; 64];
    let err = src
        .read_exact_at(300 * 1024, &mut buf, &cancel)
        .expect_err("412 must fail");
    assert!(err.to_string().contains("412"));
    assert_eq!(
        s412.request_count(),
        2,
        "412 must never be retried (1 open + 1 failed range GET)"
    );
}

#[test]
fn weak_etag_omits_if_match_and_isolates_pool_entries_across_opens() {
    let mut data_v1 = pattern_bytes(384 * 1024);
    data_v1[300 * 1024] = 0x11;
    let state = MockRangeState::new(data_v1, Some("W/\"weak-1\""));
    let server = MockRangeServer::start(Arc::clone(&state));
    let net = test_net(4 * 1024 * 1024);
    let cancel = CancelToken::new();

    let src1 = HttpRangeSource::open(
        Arc::clone(&net),
        &server.url,
        HttpRangeConfig::default(),
        &cancel,
    )
    .expect("open 1");
    assert_eq!(src1.strong_etag(), None);
    let mut b1 = [0u8; 1];
    src1.read_exact_at(300 * 1024, &mut b1, &cancel)
        .expect("read 1");
    assert_eq!(b1[0], 0x11);
    assert_eq!(
        state.recorded()[1].if_match,
        None,
        "weak ETag must NEVER be sent in If-Match"
    );

    // Mutate bytes behind the weak ETag and open a fresh instance on the same NetContext.
    state.data.lock().expect("lock")[300 * 1024] = 0x99;
    let src2 = HttpRangeSource::open(
        Arc::clone(&net),
        &server.url,
        HttpRangeConfig::default(),
        &cancel,
    )
    .expect("open 2");
    let mut b2 = [0u8; 1];
    src2.read_exact_at(300 * 1024, &mut b2, &cancel)
        .expect("read 2");
    assert_eq!(
        b2[0], 0x99,
        "weak-ETag sources must have isolated pool entries"
    );
}

#[test]
fn singleflight_deduplicates_n_way_and_follower_cancel_preserves_other_waiters() {
    let payload = pattern_bytes(512 * 1024);
    let state = MockRangeState::new(payload.clone(), Some("\"sf-v1\""));
    let server = MockRangeServer::start(Arc::clone(&state));
    let net = test_net(4 * 1024 * 1024);
    let cancel = CancelToken::new();

    let src = HttpRangeSource::open(
        Arc::clone(&net),
        &server.url,
        HttpRangeConfig {
            max_readahead_blocks: 0,
            ..HttpRangeConfig::default()
        },
        &cancel,
    )
    .expect("open");
    assert_eq!(state.request_count(), 1);

    // Pause the next range GET so 8 concurrent threads pile onto the same block.
    let gate = Arc::new(Notify::new());
    *state.delay_gate.lock().expect("lock") = Some(Arc::clone(&gate));

    let cancel_a = CancelToken::new();
    let cancel_b = CancelToken::new();
    let offset = (320 * 1024) as u64; // block 5

    let mut handles = Vec::new();
    // Follower A will cancel mid-flight; Followers 1..8 stay alive.
    {
        let s = src.clone();
        let c = cancel_a.clone();
        handles.push(thread::spawn(move || {
            let mut buf = vec![0u8; 4096];
            s.read_exact_at(offset, &mut buf, &c).map(|()| buf)
        }));
    }
    for _ in 0..7 {
        let s = src.clone();
        let c = cancel_b.clone();
        handles.push(thread::spawn(move || {
            let mut buf = vec![0u8; 4096];
            s.read_exact_at(offset, &mut buf, &c).map(|()| buf)
        }));
    }

    // Wait until the single deduplicated HTTP GET has reached the server.
    while state.range_gets_started.load(Ordering::SeqCst) < 2 {
        thread::sleep(Duration::from_millis(5));
    }
    // Give remaining threads a moment to join singleflight, then cancel Follower A only.
    thread::sleep(Duration::from_millis(30));
    cancel_a.cancel();

    let res_a = handles.remove(0).join().expect("join A");
    assert!(
        matches!(res_a, Err(IoError::Cancelled)),
        "cancelled follower must unwind immediately"
    );

    // Release the mock server response: all 7 remaining followers must succeed from the 1 GET.
    gate.notify_waiters();
    for h in handles {
        let buf = h.join().expect("join follower").expect("follower ok");
        assert_eq!(buf, payload[offset as usize..offset as usize + 4096]);
    }
    assert_eq!(
        state.request_count(),
        2,
        "8-way concurrent read must produce 0 duplicate HTTP fetches (1 open + 1 block GET)"
    );
}

#[test]
fn pinned_coalesced_prefetch_survives_lru_thrashing_and_tokio_thread_guard_holds() {
    let payload = pattern_bytes(1024 * 1024);
    let state = MockRangeState::new(payload.clone(), Some("\"pin-v1\""));
    let server = MockRangeServer::start(Arc::clone(&state));
    // Tiny 2-block LRU pool so unpinned reads immediately evict earlier blocks.
    let net = test_net(2 * RANGE_BLOCK_SIZE);
    let cancel = CancelToken::new();

    let src = HttpRangeSource::open(
        Arc::clone(&net),
        &server.url,
        HttpRangeConfig {
            max_readahead_blocks: 0,
            ..HttpRangeConfig::default()
        },
        &cancel,
    )
    .expect("open");

    // Prefetch two nearby ranges separated by < 1 block gap -> coalesced into 1 HTTP GET!
    let pin = src
        .prefetch_ranges(
            &[((300 * 1024) as u64, 4096), ((370 * 1024) as u64, 4096)],
            &cancel,
        )
        .expect("prefetch");
    assert_eq!(
        state.request_count(),
        2,
        "nearby prefetch ranges must coalesce into 1 HTTP GET"
    );
    assert!(pin.block_count() >= 2);

    // Thrash the tiny 2-block LRU pool with unrelated distant reads.
    let mut thrash = [0u8; 64];
    src.read_exact_at(700 * 1024, &mut thrash, &cancel)
        .expect("thrash 1");
    src.read_exact_at(800 * 1024, &mut thrash, &cancel)
        .expect("thrash 2");
    src.read_exact_at(900 * 1024, &mut thrash, &cancel)
        .expect("thrash 3");
    let reqs_before_pinned_read = state.request_count();

    // Reading the pinned ranges must still hit RAM with 0 extra HTTP requests!
    let mut b1 = vec![0u8; 4096];
    let mut b2 = vec![0u8; 4096];
    src.read_exact_at(300 * 1024, &mut b1, &cancel)
        .expect("pinned read 1");
    src.read_exact_at(370 * 1024, &mut b2, &cancel)
        .expect("pinned read 2");
    assert_eq!(b1, payload[300 * 1024..300 * 1024 + 4096]);
    assert_eq!(b2, payload[370 * 1024..370 * 1024 + 4096]);
    assert_eq!(
        state.request_count(),
        reqs_before_pinned_read,
        "pinned blocks must not trigger HTTP GETs even after LRU thrashing"
    );
    drop(pin);

    // Calling blocking `read_at` for an uncached block inside Tokio must return InvalidInput.
    let src_rt = src.clone();
    let res = net.handle().block_on(async move {
        let mut out = [0u8; 16];
        src_rt.read_at(550 * 1024, &mut out, &CancelToken::new())
    });
    assert!(
        matches!(res, Err(IoError::InvalidInput(_))),
        "blocking read inside Tokio worker must be rejected by guard"
    );
}
