//! Asynchronous HTTP tile fetcher with per-host concurrency limits, compliant
//! `User-Agent`, replaceable visible-first queue, validated SQLite disk cache
//! read/write-through, end-to-end `CancelToken` abort through CPU decode, and
//! deterministic shutdown (`CancelToken` -> join decode pool -> drop
//! `tokio::Runtime`).

use std::collections::{HashMap, HashSet};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use tokio::sync::{Notify, Semaphore};
use veldmap_core::io::CancelToken;
use veldmap_core::tile::{DecodedTile, QuadTileId, SourceId, TileKey, TileScheme};
use veldmap_storage::TileDiskCache;

use crate::NetError;
use crate::basemap::LiveBasemapSpec;
use crate::decode_pool::{DecodeJob, DecodePool, DecodePoolHandle, Wakeup};

/// Default maximum concurrent HTTP requests per remote host.
pub const DEFAULT_MAX_CONCURRENT_PER_HOST: usize = 6;

/// Default maximum encoded response body accepted for one live tile (`16 MiB`).
pub const DEFAULT_MAX_TILE_RESPONSE_BYTES: usize = 16 * 1024 * 1024;

/// Backoff duration after transient HTTP / network errors before retrying a tile.
const TRANSIENT_RETRY_BACKOFF: Duration = Duration::from_secs(15);

/// Compliant HTTP `User-Agent` identifying VeldMap.
pub fn default_user_agent() -> String {
    format!(
        "VeldMap/{} (+https://github.com/veldmap/veldmap)",
        env!("CARGO_PKG_VERSION")
    )
}

/// Configuration for [`LiveTilePipeline`].
#[derive(Clone, Debug)]
pub struct FetcherConfig {
    pub user_agent: String,
    pub max_concurrent_per_host: usize,
    pub connect_timeout: Duration,
    pub request_timeout: Duration,
    pub max_tile_bytes: usize,
    pub decode_threads: usize,
}

impl Default for FetcherConfig {
    fn default() -> Self {
        let decode_threads = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(2)
            .clamp(2, 4);
        Self {
            user_agent: default_user_agent(),
            max_concurrent_per_host: DEFAULT_MAX_CONCURRENT_PER_HOST,
            connect_timeout: Duration::from_secs(10),
            request_timeout: Duration::from_secs(20),
            max_tile_bytes: DEFAULT_MAX_TILE_RESPONSE_BYTES,
            decode_threads,
        }
    }
}

/// Observable status snapshot of the live tile pipeline for the GIS HUD / UI.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NetworkStatus {
    pub in_flight: usize,
    pub queued: usize,
    pub completed_from_disk: u64,
    pub completed_from_net: u64,
    pub failed_count: u64,
    pub offline: bool,
    pub last_error: Option<String>,
}

#[derive(Clone, Copy, Debug)]
enum FailureState {
    Terminal,
    BackoffUntil(Instant),
}

#[derive(Default)]
struct PipelineState {
    active_spec: Option<Arc<LiveBasemapSpec>>,
    pending: Vec<TileKey>,
    /// Tracks active `(generation, CancelToken)` from async dispatch all the
    /// way through `DecodePool` completion so stale guards never clobber a
    /// newer spawn and pan/switch cancels queued decode jobs immediately.
    in_flight: HashMap<TileKey, (u64, CancelToken)>,
    next_gen: u64,
    failures: HashMap<TileKey, FailureState>,
    completed_from_disk: u64,
    completed_from_net: u64,
    failed_count: u64,
    offline: bool,
    last_error: Option<String>,
}

fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

fn ensure_rustls_ring_provider() {
    static INIT: OnceLock<()> = OnceLock::new();
    INIT.get_or_init(|| {
        // Explicit `ring` crypto provider for `rustls` (installed once per
        // process; ignores Err if another subsystem already installed one).
        let _ = rustls::crypto::ring::default_provider().install_default();
    });
}

fn build_http_client(cfg: &FetcherConfig) -> Result<reqwest::Client, NetError> {
    ensure_rustls_ring_provider();
    reqwest::Client::builder()
        .user_agent(&cfg.user_agent)
        .connect_timeout(cfg.connect_timeout)
        .timeout(cfg.request_timeout)
        .pool_max_idle_per_host(cfg.max_concurrent_per_host.max(1))
        .build()
        .map_err(|err| NetError::HttpClient(err.to_string()))
}

struct HostLimiters {
    max_per_host: usize,
    map: Mutex<HashMap<String, Arc<Semaphore>>>,
}

impl HostLimiters {
    fn new(max_per_host: usize) -> Self {
        Self {
            max_per_host: max_per_host.max(1),
            map: Mutex::new(HashMap::new()),
        }
    }

    fn for_url(&self, url: &str) -> Arc<Semaphore> {
        let host = extract_host(url);
        let mut map = lock_unpoisoned(&self.map);
        Arc::clone(
            map.entry(host)
                .or_insert_with(|| Arc::new(Semaphore::new(self.max_per_host))),
        )
    }
}

fn extract_host(url: &str) -> String {
    let without_scheme = url.split_once("://").map_or(url, |(_, rest)| rest);
    let authority = without_scheme
        .split(['/', '?', '#'])
        .next()
        .unwrap_or("unknown");
    let host_port = authority.rsplit_once('@').map_or(authority, |(_, hp)| hp);
    host_port.to_ascii_lowercase()
}

/// Generation-checked RAII guard keeping `key` registered in `state.in_flight`
/// across both the async HTTP fetch AND CPU decode in `DecodePool`.
struct InFlightGuard {
    state: Arc<Mutex<PipelineState>>,
    key: TileKey,
    generation: u64,
    notify: Arc<Notify>,
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        {
            let mut st = lock_unpoisoned(&self.state);
            if st
                .in_flight
                .get(&self.key)
                .is_some_and(|(gen_id, _)| *gen_id == self.generation)
            {
                st.in_flight.remove(&self.key);
            }
        }
        self.notify.notify_one();
    }
}

/// Complete live HTTP tile streaming pipeline owning:
/// 1. `CancelToken` (`shutdown`)
/// 2. Bounded `std::thread` [`DecodePool`]
/// 3. Dedicated `tokio::runtime::Runtime` running HTTP fetches
///
/// Shuts down deterministically via [`Self::signal_shutdown`] / `Drop`:
/// `CancelToken` -> close/join decode pool -> shutdown `tokio::Runtime`.
pub struct LiveTilePipeline {
    shutdown: CancelToken,
    state: Arc<Mutex<PipelineState>>,
    notify: Arc<Notify>,
    decode_pool: Option<DecodePool>,
    runtime: Option<tokio::runtime::Runtime>,
}

impl LiveTilePipeline {
    /// Spawn a live tile pipeline pushing decoded, mipmapped tiles to `done_tx`
    /// and invoking `wakeup` on tile arrival or offline status change.
    pub fn spawn(
        config: FetcherConfig,
        disk_cache: Option<Arc<TileDiskCache>>,
        done_tx: Sender<(TileKey, DecodedTile)>,
        wakeup: Wakeup,
    ) -> Result<Self, NetError> {
        let client = build_http_client(&config)?;
        let shutdown = CancelToken::new();
        let decode_pool = DecodePool::spawn(
            config.decode_threads,
            done_tx,
            Arc::clone(&wakeup),
            shutdown.clone(),
        );
        let decode_handle = decode_pool.handle();

        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .thread_name("veldmap-net")
            .enable_all()
            .build()?;

        let state = Arc::new(Mutex::new(PipelineState::default()));
        let notify = Arc::new(Notify::new());
        let host_limiters = Arc::new(HostLimiters::new(config.max_concurrent_per_host));
        let global_concurrency = config
            .max_concurrent_per_host
            .clamp(1, 16)
            .saturating_mul(2);

        let disp_state = Arc::clone(&state);
        let disp_notify = Arc::clone(&notify);
        let disp_shutdown = shutdown.clone();
        let disp_max_bytes = config.max_tile_bytes;

        runtime.spawn(async move {
            dispatcher_loop(
                disp_state,
                disp_notify,
                disp_shutdown,
                client,
                host_limiters,
                disk_cache,
                decode_handle,
                wakeup,
                global_concurrency,
                disp_max_bytes,
            )
            .await;
        });

        Ok(Self {
            shutdown,
            state,
            notify,
            decode_pool: Some(decode_pool),
            runtime: Some(runtime),
        })
    }

    /// Switch the active live basemap specification (or `None` when viewing
    /// offline / ellipsoid-only), immediately cancelling all in-flight HTTP
    /// and queued CPU decode jobs from any previous source.
    pub fn set_active_spec(&self, spec: Option<LiveBasemapSpec>) {
        let to_cancel: Vec<CancelToken> = {
            let mut st = lock_unpoisoned(&self.state);
            let same = match (&st.active_spec, &spec) {
                (Some(a), Some(b)) => a.as_ref() == b,
                (None, None) => true,
                _ => false,
            };
            if same {
                return;
            }
            st.active_spec = spec.map(Arc::new);
            st.pending.clear();
            st.failures.clear();
            st.offline = false;
            st.last_error = None;
            st.in_flight.drain().map(|(_, (_, tok))| tok).collect()
        };

        for tok in to_cancel {
            tok.cancel();
        }
        self.notify.notify_one();
    }

    /// Replace the visible-first fetch queue with `wanted`, immediately
    /// cancelling any in-flight HTTP or decode request no longer wanted
    /// (retaining active-source coarsest roots `key.parent().is_none()`).
    ///
    /// Strictly filters out `PolarCap` keys and any `level > spec.max_level`
    /// before queueing so zero forbidden HTTP requests are ever issued.
    pub fn reconcile_wanted<F>(&self, wanted: &HashSet<TileKey>, is_ready: F)
    where
        F: Fn(&TileKey) -> bool,
    {
        let now = Instant::now();
        let to_cancel: Vec<CancelToken> = {
            let mut st = lock_unpoisoned(&self.state);
            let Some(spec) = st.active_spec.clone() else {
                st.pending.clear();
                return;
            };
            let active_source = spec.source_id;

            let mut cancelled = Vec::new();
            st.in_flight.retain(|key, (_, tok)| {
                let keep = key.source == active_source
                    && !is_ready(key)
                    && (wanted.contains(key) || is_active_root(key, active_source));
                if !keep {
                    cancelled.push(tok.clone());
                }
                keep
            });

            // Prune expired backoff markers so transient errors retry cleanly.
            st.failures.retain(|_, f| match f {
                FailureState::Terminal => true,
                FailureState::BackoffUntil(until) => *until > now,
            });

            let mut next_pending = Vec::new();
            for &key in wanted {
                if key.source != active_source || is_ready(&key) {
                    continue;
                }
                if st.in_flight.contains_key(&key) || st.failures.contains_key(&key) {
                    continue;
                }
                let id = QuadTileId {
                    scheme: key.scheme,
                    level: key.level,
                    x: key.x,
                    y: key.y,
                };
                // Enforce 0 requests for PolarCap and 0 requests above max_level.
                if spec.validate_tile(id).is_err() {
                    continue;
                }
                next_pending.push(key);
            }
            next_pending.sort_by_key(TileKey::coarsest_first_key);
            st.pending = next_pending;
            cancelled
        };

        for tok in to_cancel {
            tok.cancel();
        }
        self.notify.notify_one();
    }

    /// Clear transient failure / offline markers so the user can retry
    /// immediately (e.g. after reconnecting network or updating settings).
    pub fn retry_failed(&self) {
        {
            let mut st = lock_unpoisoned(&self.state);
            st.failures.clear();
            st.offline = false;
            st.last_error = None;
        }
        self.notify.notify_one();
    }

    /// Return current number of in-flight + queued tiles for appetite
    /// budgeting.
    pub fn active_work_count(&self) -> usize {
        let st = lock_unpoisoned(&self.state);
        st.in_flight.len().saturating_add(st.pending.len())
    }

    /// Snapshot observable status counters for the GIS UI.
    pub fn status(&self) -> NetworkStatus {
        let st = lock_unpoisoned(&self.state);
        NetworkStatus {
            in_flight: st.in_flight.len(),
            queued: st.pending.len(),
            completed_from_disk: st.completed_from_disk,
            completed_from_net: st.completed_from_net,
            failed_count: st.failed_count,
            offline: st.offline,
            last_error: st.last_error.clone(),
        }
    }

    /// Trigger global and per-tile `CancelToken`s immediately (Phase 1 of
    /// deterministic shutdown, called BEFORE dropping the completion rx).
    pub fn signal_shutdown(&self) {
        self.shutdown.cancel();
        let active_tokens: Vec<CancelToken> = {
            let mut st = lock_unpoisoned(&self.state);
            st.pending.clear();
            st.in_flight.drain().map(|(_, (_, t))| t).collect()
        };
        for tok in active_tokens {
            tok.cancel();
        }
        self.notify.notify_waiters();
    }

    /// Deterministically shut down the pipeline in exact order:
    /// 1. Cancel `CancelToken` (aborts all in-flight `tokio::select!` fetches)
    /// 2. Close & join the bounded CPU `DecodePool`
    /// 3. Shut down `tokio::Runtime`
    pub fn shutdown(mut self) {
        self.shutdown_inner();
    }

    fn shutdown_inner(&mut self) {
        self.signal_shutdown();

        if let Some(pool) = self.decode_pool.take() {
            pool.shutdown();
        }

        if let Some(rt) = self.runtime.take() {
            rt.shutdown_timeout(Duration::from_secs(2));
        }
    }
}

impl Drop for LiveTilePipeline {
    fn drop(&mut self) {
        self.shutdown_inner();
    }
}

fn is_active_root(key: &TileKey, active_source: SourceId) -> bool {
    key.source == active_source && key.parent().is_none()
}

#[expect(
    clippy::too_many_arguments,
    reason = "dispatcher coordinates runtime handles, limiters, cache, and decode pool"
)]
async fn dispatcher_loop(
    state: Arc<Mutex<PipelineState>>,
    notify: Arc<Notify>,
    shutdown: CancelToken,
    client: reqwest::Client,
    host_limiters: Arc<HostLimiters>,
    disk_cache: Option<Arc<TileDiskCache>>,
    decode_handle: DecodePoolHandle,
    wakeup: Wakeup,
    global_concurrency: usize,
    max_tile_bytes: usize,
) {
    loop {
        if shutdown.is_cancelled() {
            break;
        }

        // Pop as many coarsest pending jobs as global concurrency permits.
        let mut to_spawn: Vec<(TileKey, u64, Arc<LiveBasemapSpec>, CancelToken)> = Vec::new();
        {
            let mut st = lock_unpoisoned(&state);
            if let Some(spec) = st.active_spec.clone() {
                while st.in_flight.len() < global_concurrency && !st.pending.is_empty() {
                    let key = st.pending.remove(0);
                    if key.source != spec.source_id
                        || st.in_flight.contains_key(&key)
                        || st.failures.contains_key(&key)
                    {
                        continue;
                    }
                    st.next_gen = st.next_gen.wrapping_add(1);
                    let gen_id = st.next_gen;
                    let tile_cancel = shutdown.child_token();
                    st.in_flight.insert(key, (gen_id, tile_cancel.clone()));
                    to_spawn.push((key, gen_id, Arc::clone(&spec), tile_cancel));
                }
            } else {
                st.pending.clear();
            }
        }

        if to_spawn.is_empty() {
            tokio::select! {
                _ = shutdown.cancelled() => break,
                _ = notify.notified() => continue,
            }
        }

        for (key, generation, spec, tile_cancel) in to_spawn {
            let task_state = Arc::clone(&state);
            let task_notify = Arc::clone(&notify);
            let task_client = client.clone();
            let task_limiters = Arc::clone(&host_limiters);
            let task_cache = disk_cache.clone();
            let task_decode = decode_handle.clone();
            let task_wake = Arc::clone(&wakeup);

            tokio::spawn(async move {
                let guard = InFlightGuard {
                    state: Arc::clone(&task_state),
                    key,
                    generation,
                    notify: task_notify,
                };
                fetch_one_tile(
                    key,
                    spec,
                    tile_cancel,
                    guard,
                    task_state,
                    task_client,
                    task_limiters,
                    task_cache,
                    task_decode,
                    task_wake,
                    max_tile_bytes,
                )
                .await;
            });
        }
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "single-tile async worker receives shared client, cache, decode handle, guard, and status state"
)]
async fn fetch_one_tile(
    key: TileKey,
    spec: Arc<LiveBasemapSpec>,
    cancel: CancelToken,
    guard: InFlightGuard,
    state: Arc<Mutex<PipelineState>>,
    client: reqwest::Client,
    host_limiters: Arc<HostLimiters>,
    disk_cache: Option<Arc<TileDiskCache>>,
    decode_handle: DecodePoolHandle,
    wakeup: Wakeup,
    max_tile_bytes: usize,
) {
    if cancel.is_cancelled() {
        return;
    }

    let id = QuadTileId {
        scheme: key.scheme,
        level: key.level,
        x: key.x,
        y: key.y,
    };
    // Final defense-in-depth contract check: never touch PolarCap or > max_level.
    if matches!(id.scheme, TileScheme::PolarCap { .. }) {
        return;
    }
    let Ok(url) = spec.resolve_url(id) else {
        record_failure(
            &state,
            &cancel,
            key,
            FailureState::Terminal,
            false,
            None,
            &wakeup,
        );
        return;
    };
    let disk_key = spec.disk_key(id).ok();

    // 1. Disk cache read-through (zero HTTP requests on hit).
    if let (Some(cache), Some(dk)) = (&disk_cache, &disk_key) {
        let cache_clone = Arc::clone(cache);
        let dk_clone = dk.clone();
        let disk_res = tokio::select! {
            _ = cancel.cancelled() => return,
            res = tokio::task::spawn_blocking(move || cache_clone.get(&dk_clone)) => res,
        };
        if let Ok(Ok(Some(bytes))) = disk_res {
            if cancel.is_cancelled() {
                return;
            }
            let cb_state = Arc::clone(&state);
            let cb_cancel = cancel.clone();
            let cb_cache = Arc::clone(cache);
            let cb_dk = dk.clone();
            let cb_wake = Arc::clone(&wakeup);
            let job = DecodeJob {
                key,
                bytes,
                cancel,
                on_decoded: Box::new(move |outcome| {
                    let _guard = guard;
                    match outcome {
                        Ok(_) => {
                            let mut st = lock_unpoisoned(&cb_state);
                            if !cb_cancel.is_cancelled()
                                && st
                                    .active_spec
                                    .as_ref()
                                    .is_some_and(|s| s.source_id == key.source)
                            {
                                st.completed_from_disk = st.completed_from_disk.saturating_add(1);
                            }
                        }
                        Err(err) => {
                            // Evict corrupted disk payload and mark terminal to avoid loops.
                            let _ = cb_cache.delete(&cb_dk);
                            record_failure(
                                &cb_state,
                                &cb_cancel,
                                key,
                                FailureState::Terminal,
                                false,
                                Some(err),
                                &cb_wake,
                            );
                        }
                    }
                }),
            };
            decode_handle.submit(job);
            return;
        }
    }

    if cancel.is_cancelled() {
        return;
    }

    // 2. Per-host concurrency gate (cancel-aware).
    let sem = host_limiters.for_url(&url);
    let _permit = tokio::select! {
        _ = cancel.cancelled() => return,
        res = sem.acquire_owned() => match res {
            Ok(p) => p,
            Err(_) => return,
        },
    };

    if cancel.is_cancelled() {
        return;
    }

    // 3. HTTP GET with immediate `tokio::select!` cancellation and size cap.
    let fetch_res = tokio::select! {
        _ = cancel.cancelled() => return,
        res = download_capped(&client, &url, max_tile_bytes) => res,
    };

    if cancel.is_cancelled() {
        return;
    }

    match fetch_res {
        Ok(bytes) => {
            // Hand off ownership of `guard` to `DecodeJob` so `in_flight`
            // remains active through decode, SQLite persists ONLY decode-
            // validated image bytes, and corrupt HTTP 200 responses are marked
            // terminal (preventing infinite retry loops).
            let cb_state = Arc::clone(&state);
            let cb_cancel = cancel.clone();
            let cb_spec = Arc::clone(&spec);
            let cb_wake = Arc::clone(&wakeup);
            let job = DecodeJob {
                key,
                bytes,
                cancel,
                on_decoded: Box::new(move |outcome| {
                    let _guard = guard;
                    match outcome {
                        Ok(valid_bytes) => {
                            if let (Some(cache), Some(dk)) = (disk_cache, disk_key)
                                && let Err(err) = cache.put(&dk, valid_bytes)
                            {
                                tracing::debug!(
                                    error = %err,
                                    "Failed to persist validated live tile to disk cache"
                                );
                            }
                            let mut st = lock_unpoisoned(&cb_state);
                            if !cb_cancel.is_cancelled()
                                && st
                                    .active_spec
                                    .as_ref()
                                    .is_some_and(|s| s.source_id == key.source)
                            {
                                st.completed_from_net = st.completed_from_net.saturating_add(1);
                                st.offline = false;
                                st.last_error = None;
                            }
                        }
                        Err(err) => {
                            let safe_msg = cb_spec.redact_secrets(&err);
                            record_failure(
                                &cb_state,
                                &cb_cancel,
                                key,
                                FailureState::Terminal,
                                false,
                                Some(safe_msg),
                                &cb_wake,
                            );
                        }
                    }
                }),
            };
            decode_handle.submit(job);
        }
        Err(FetchFailure::Terminal(msg)) => {
            let safe_msg = spec.redact_secrets(&msg);
            tracing::debug!(?key, error = %safe_msg, "Terminal HTTP tile failure");
            record_failure(
                &state,
                &cancel,
                key,
                FailureState::Terminal,
                false,
                Some(safe_msg),
                &wakeup,
            );
        }
        Err(FetchFailure::Transient { msg, offline }) => {
            let safe_msg = spec.redact_secrets(&msg);
            tracing::debug!(?key, offline, error = %safe_msg, "Transient HTTP tile failure");
            record_failure(
                &state,
                &cancel,
                key,
                FailureState::BackoffUntil(Instant::now() + TRANSIENT_RETRY_BACKOFF),
                offline,
                Some(safe_msg),
                &wakeup,
            );
        }
    }
}

fn record_failure(
    state: &Arc<Mutex<PipelineState>>,
    cancel: &CancelToken,
    key: TileKey,
    failure: FailureState,
    offline: bool,
    error_msg: Option<String>,
    wakeup: &Wakeup,
) {
    let mut should_wake = false;
    {
        let mut st = lock_unpoisoned(state);
        // Re-check cancellation and active source under lock so a late error
        // from a previous basemap never contaminates the newly active source.
        if !cancel.is_cancelled()
            && st
                .active_spec
                .as_ref()
                .is_some_and(|s| s.source_id == key.source)
        {
            st.failures.insert(key, failure);
            st.failed_count = st.failed_count.saturating_add(1);
            if offline {
                st.offline = true;
            }
            if let Some(msg) = error_msg {
                st.last_error = Some(msg);
            }
            should_wake = true;
        }
    }
    if should_wake {
        (wakeup)();
    }
}

enum FetchFailure {
    Terminal(String),
    Transient { msg: String, offline: bool },
}

async fn download_capped(
    client: &reqwest::Client,
    url: &str,
    max_bytes: usize,
) -> Result<Vec<u8>, FetchFailure> {
    let mut resp = client.get(url).send().await.map_err(|err| {
        let offline = err.is_connect() || err.is_timeout();
        FetchFailure::Transient {
            msg: err.to_string(),
            offline,
        }
    })?;

    let status = resp.status();
    if !status.is_success() {
        let code = status.as_u16();
        let msg = format!("HTTP {code}");
        if matches!(code, 400 | 401 | 403 | 404 | 410 | 451) {
            return Err(FetchFailure::Terminal(msg));
        }
        return Err(FetchFailure::Transient {
            msg,
            offline: false,
        });
    }

    if let Some(len) = resp.content_length()
        && len > max_bytes as u64
    {
        return Err(FetchFailure::Terminal(format!(
            "tile payload {len} B exceeds limit {max_bytes} B"
        )));
    }

    let mut buf = Vec::with_capacity(
        resp.content_length()
            .map_or(16 * 1024, |n| (n as usize).min(max_bytes)),
    );
    while let Some(chunk) = resp.chunk().await.map_err(|err| FetchFailure::Transient {
        msg: err.to_string(),
        offline: err.is_connect() || err.is_timeout(),
    })? {
        if buf.len().saturating_add(chunk.len()) > max_bytes {
            return Err(FetchFailure::Terminal(format!(
                "tile payload exceeded limit {max_bytes} B"
            )));
        }
        buf.extend_from_slice(&chunk);
    }

    if buf.is_empty() {
        return Err(FetchFailure::Terminal("empty tile response".to_owned()));
    }
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc;

    use image::{ImageBuffer, ImageFormat, Rgba};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    use veldmap_storage::CustomXyzSettings;

    use super::*;

    fn make_png_bytes(color: [u8; 4]) -> Vec<u8> {
        let img: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::from_pixel(8, 8, Rgba(color));
        let mut out = Vec::new();
        img.write_to(&mut Cursor::new(&mut out), ImageFormat::Png)
            .expect("encode test png");
        out
    }

    fn spawn_mock_tile_server(
        payload: Vec<u8>,
        content_type: &'static str,
        delay: Duration,
    ) -> (
        String,
        Arc<AtomicUsize>,
        Arc<AtomicUsize>,
        Arc<AtomicUsize>,
        CancelToken,
        std::thread::JoinHandle<()>,
    ) {
        let hits = Arc::new(AtomicUsize::new(0));
        let active = Arc::new(AtomicUsize::new(0));
        let peak_active = Arc::new(AtomicUsize::new(0));
        let stop = CancelToken::new();

        let (addr_tx, addr_rx) = mpsc::channel();
        let hits_c = Arc::clone(&hits);
        let active_c = Arc::clone(&active);
        let peak_c = Arc::clone(&peak_active);
        let stop_c = stop.clone();

        let thread = std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("mock server rt");
            rt.block_on(async move {
                let listener = TcpListener::bind("127.0.0.1:0")
                    .await
                    .expect("bind mock server");
                let addr = listener.local_addr().expect("local addr");
                let _ = addr_tx.send(format!("http://{addr}"));

                loop {
                    let accept = tokio::select! {
                        _ = stop_c.cancelled() => break,
                        res = listener.accept() => res,
                    };
                    let Ok((mut stream, _)) = accept else {
                        break;
                    };
                    let payload_c = payload.clone();
                    let h = Arc::clone(&hits_c);
                    let a = Arc::clone(&active_c);
                    let p = Arc::clone(&peak_c);
                    let s = stop_c.clone();

                    tokio::spawn(async move {
                        let cur = a.fetch_add(1, Ordering::AcqRel) + 1;
                        p.fetch_max(cur, Ordering::AcqRel);
                        h.fetch_add(1, Ordering::AcqRel);

                        let mut req_buf = [0u8; 1024];
                        let _ = stream.read(&mut req_buf).await;
                        if !delay.is_zero() {
                            tokio::select! {
                                _ = s.cancelled() => {
                                    a.fetch_sub(1, Ordering::AcqRel);
                                    return;
                                }
                                _ = tokio::time::sleep(delay) => {}
                            }
                        }
                        let header = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                            payload_c.len()
                        );
                        let _ = stream.write_all(header.as_bytes()).await;
                        let _ = stream.write_all(&payload_c).await;
                        let _ = stream.shutdown().await;
                        a.fetch_sub(1, Ordering::AcqRel);
                    });
                }
            });
        });

        let base_url = addr_rx.recv().expect("receive mock server address");
        (base_url, hits, active, peak_active, stop, thread)
    }

    #[test]
    fn streams_mipmapped_tiles_enforces_host_limit_max_level_polar_and_disk_cache() {
        let png = make_png_bytes([40, 180, 90, 255]);
        let (base_url, hits, _active, peak_active, server_stop, server_thread) =
            spawn_mock_tile_server(png, "image/png", Duration::from_millis(25));

        let spec = LiveBasemapSpec::from_custom(&CustomXyzSettings {
            name: "Mock".to_owned(),
            url_template: format!("{base_url}/{{z}}/{{x}}/{{y}}.png"),
            api_key: String::new(),
            tms_flip_y: false,
            max_level: 2,
            attribution: "Mock".to_owned(),
        })
        .expect("build mock spec");

        let disk_cache = Arc::new(TileDiskCache::open_in_memory(1024 * 1024).expect("mem cache"));
        let (done_tx, done_rx) = mpsc::channel();
        let wake_hits = Arc::new(AtomicUsize::new(0));
        let wake_c = Arc::clone(&wake_hits);
        let wakeup: Wakeup = Arc::new(move || {
            wake_c.fetch_add(1, Ordering::AcqRel);
        });

        let pipeline = LiveTilePipeline::spawn(
            FetcherConfig {
                max_concurrent_per_host: 2,
                decode_threads: 2,
                ..FetcherConfig::default()
            },
            Some(Arc::clone(&disk_cache)),
            done_tx.clone(),
            Arc::clone(&wakeup),
        )
        .expect("spawn pipeline");
        pipeline.set_active_spec(Some(spec.clone()));

        // Build wanted set containing:
        // - 4 valid WebMercator z=1 tiles
        // - 1 forbidden PolarCap tile (MUST emit 0 HTTP requests)
        // - 1 forbidden z=3 tile above max_level=2 (MUST emit 0 HTTP requests)
        let mut wanted = HashSet::new();
        for y in 0..2 {
            for x in 0..2 {
                wanted.insert(spec.tile_key(QuadTileId {
                    scheme: TileScheme::WebMercator,
                    level: 1,
                    x,
                    y,
                }));
            }
        }
        wanted.insert(spec.tile_key(QuadTileId {
            scheme: TileScheme::PolarCap { north: true },
            level: 0,
            x: 0,
            y: 0,
        }));
        wanted.insert(spec.tile_key(QuadTileId {
            scheme: TileScheme::WebMercator,
            level: 3,
            x: 0,
            y: 0,
        }));

        pipeline.reconcile_wanted(&wanted, |_| false);

        for _ in 0..4 {
            let (_k, tile) = done_rx
                .recv_timeout(Duration::from_secs(5))
                .expect("receive decoded live tile");
            assert!(tile.validate());
            assert_eq!(tile.width, 8);
            assert_eq!(tile.mip_level_count(), 4, "8x8 must generate 4 mip levels");
            assert_eq!(&tile.data[0..4], &[40, 180, 90, 255]);
        }
        // Wait briefly for on_decoded SQLite persistence to finish on workers.
        let wait_disk = Instant::now();
        while disk_cache.stats().expect("stats").tile_count < 4
            && wait_disk.elapsed() < Duration::from_secs(2)
        {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(disk_cache.stats().expect("stats").tile_count, 4);

        // Ensure no extra tile arrives for PolarCap or z=3.
        assert!(done_rx.recv_timeout(Duration::from_millis(80)).is_err());

        assert_eq!(
            hits.load(Ordering::Acquire),
            4,
            "exactly 4 HTTP requests (0 for PolarCap, 0 for z > max_level)"
        );
        assert!(
            peak_active.load(Ordering::Acquire) <= 2,
            "per-host concurrency limit (2) exceeded: {}",
            peak_active.load(Ordering::Acquire)
        );
        assert!(wake_hits.load(Ordering::Acquire) >= 4);

        // Shut down cleanly and create a second pipeline backed by the SAME disk cache:
        // all 4 tiles must serve from disk with ZERO additional HTTP requests!
        pipeline.shutdown();

        let (done_tx2, done_rx2) = mpsc::channel();
        let pipeline2 = LiveTilePipeline::spawn(
            FetcherConfig {
                max_concurrent_per_host: 2,
                decode_threads: 2,
                ..FetcherConfig::default()
            },
            Some(Arc::clone(&disk_cache)),
            done_tx2,
            wakeup,
        )
        .expect("spawn pipeline2");
        pipeline2.set_active_spec(Some(spec));
        pipeline2.reconcile_wanted(&wanted, |_| false);

        for _ in 0..4 {
            let (_k, tile) = done_rx2
                .recv_timeout(Duration::from_secs(5))
                .expect("receive cached tile from disk");
            assert_eq!(tile.mip_level_count(), 4);
        }
        let wait_stat = Instant::now();
        while pipeline2.status().completed_from_disk < 4
            && wait_stat.elapsed() < Duration::from_secs(2)
        {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(
            hits.load(Ordering::Acquire),
            4,
            "disk cache hit must issue 0 additional HTTP requests"
        );
        assert_eq!(pipeline2.status().completed_from_disk, 4);
        pipeline2.shutdown();

        server_stop.cancel();
        let _ = server_thread.join();
    }

    #[test]
    fn corrupt_http_200_is_not_cached_and_marked_terminal_without_retry_loop() {
        let (base_url, hits, _active, _peak, server_stop, server_thread) = spawn_mock_tile_server(
            b"<html>captive portal</html>".to_vec(),
            "text/html",
            Duration::ZERO,
        );

        let spec = LiveBasemapSpec::from_custom(&CustomXyzSettings {
            name: "CorruptMock".to_owned(),
            url_template: format!("{base_url}/{{z}}/{{x}}/{{y}}.png"),
            api_key: String::new(),
            tms_flip_y: false,
            max_level: 4,
            attribution: "Mock".to_owned(),
        })
        .expect("spec");

        let disk_cache = Arc::new(TileDiskCache::open_in_memory(1024 * 1024).expect("mem cache"));
        let (done_tx, done_rx) = mpsc::channel();
        let wakeup: Wakeup = Arc::new(|| {});
        let pipeline = LiveTilePipeline::spawn(
            FetcherConfig::default(),
            Some(Arc::clone(&disk_cache)),
            done_tx,
            wakeup,
        )
        .expect("spawn");
        pipeline.set_active_spec(Some(spec.clone()));

        let key = spec.tile_key(QuadTileId {
            scheme: TileScheme::WebMercator,
            level: 1,
            x: 0,
            y: 0,
        });
        let wanted = HashSet::from([key]);
        pipeline.reconcile_wanted(&wanted, |_| false);

        let wait = Instant::now();
        while pipeline.status().failed_count == 0 && wait.elapsed() < Duration::from_secs(2) {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(pipeline.status().failed_count, 1);
        assert_eq!(
            disk_cache.stats().expect("stats").tile_count,
            0,
            "corrupt HTTP 200 payload must never be persisted to SQLite"
        );
        assert!(done_rx.try_recv().is_err());

        // Reconciling again must NOT re-fetch the terminally failed corrupt tile.
        pipeline.reconcile_wanted(&wanted, |_| false);
        std::thread::sleep(Duration::from_millis(60));
        assert_eq!(hits.load(Ordering::Acquire), 1);

        pipeline.shutdown();
        server_stop.cancel();
        let _ = server_thread.join();
    }

    #[test]
    fn cancel_token_aborts_slow_inflight_request_promptly_on_shutdown() {
        let png = make_png_bytes([10, 20, 30, 255]);
        // Server sleeps 10 seconds before replying; shutdown must abort immediately.
        let (base_url, hits, _active, _peak, server_stop, server_thread) =
            spawn_mock_tile_server(png, "image/png", Duration::from_secs(10));

        let spec = LiveBasemapSpec::from_custom(&CustomXyzSettings {
            name: "SlowMock".to_owned(),
            url_template: format!("{base_url}/{{z}}/{{x}}/{{y}}.png"),
            api_key: String::new(),
            tms_flip_y: false,
            max_level: 4,
            attribution: String::new(),
        })
        .expect("spec");

        let (done_tx, _done_rx) = mpsc::channel();
        let wakeup: Wakeup = Arc::new(|| {});
        let pipeline = LiveTilePipeline::spawn(FetcherConfig::default(), None, done_tx, wakeup)
            .expect("spawn");
        pipeline.set_active_spec(Some(spec.clone()));

        let key = spec.tile_key(QuadTileId {
            scheme: TileScheme::WebMercator,
            level: 1,
            x: 0,
            y: 0,
        });
        pipeline.reconcile_wanted(&HashSet::from([key]), |_| false);

        // Wait briefly until the mock server observes the in-flight connection.
        let wait_start = Instant::now();
        while hits.load(Ordering::Acquire) == 0 && wait_start.elapsed() < Duration::from_secs(2) {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(hits.load(Ordering::Acquire), 1);

        let shutdown_start = Instant::now();
        pipeline.shutdown();
        assert!(
            shutdown_start.elapsed() < Duration::from_secs(1),
            "CancelToken shutdown stalled waiting for 10 s server response: {:?}",
            shutdown_start.elapsed()
        );

        server_stop.cancel();
        let _ = server_thread.join();
    }

    #[test]
    fn stale_inflight_guard_never_removes_newer_generation() {
        let state = Arc::new(Mutex::new(PipelineState::default()));
        let notify = Arc::new(Notify::new());
        let key = LiveBasemapSpec::eox_sentinel2_cloudless().tile_key(QuadTileId {
            scheme: TileScheme::WebMercator,
            level: 1,
            x: 0,
            y: 0,
        });

        let tok2 = CancelToken::new();
        {
            let mut st = lock_unpoisoned(&state);
            st.in_flight.insert(key, (2, tok2.clone()));
        }

        // Dropping stale generation-1 guard must NOT remove generation-2 entry.
        let stale = InFlightGuard {
            state: Arc::clone(&state),
            key,
            generation: 1,
            notify: Arc::clone(&notify),
        };
        drop(stale);
        assert!(lock_unpoisoned(&state).in_flight.contains_key(&key));

        // Dropping generation-2 guard removes it cleanly.
        let current = InFlightGuard {
            state: Arc::clone(&state),
            key,
            generation: 2,
            notify,
        };
        drop(current);
        assert!(!lock_unpoisoned(&state).in_flight.contains_key(&key));
    }
}
