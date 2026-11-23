//! Asynchronous HTTP tile fetcher with two-stage disk/HTTP scheduling,
//! per-host concurrency & rate-limit cooldowns, replaceable visible-first
//! queues, end-to-end `CancelToken` abort through an injected
//! [`TileDecodeExecutor`], and deterministic shutdown.

mod dispatch;
mod guards;
mod state;
mod workers;

use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::sync::{Notify, Semaphore};
use veldmap_core::io::{CancelToken, Wakeup, lock_unpoisoned};
use veldmap_core::tile::TileKey;

use crate::basemap::LiveBasemapSpec;
use crate::http::{
    DEFAULT_MAX_CONCURRENT_PER_HOST, DEFAULT_MAX_TILE_RESPONSE_BYTES, HostLimiters,
    HttpClientConfig, build_http_client, default_user_agent,
};
use crate::{NetError, TileBlobStore, TileDecodeExecutor};

use self::dispatch::dispatcher_loop;
use self::state::{FailureState, PipelineState};

/// Default backoff duration after transient HTTP / network errors before
/// retrying a tile.
const DEFAULT_TRANSIENT_RETRY_BACKOFF: Duration = Duration::from_secs(15);

/// Short silent cooldown before re-attempting a tile whose decode submission
/// was rejected under CPU decode-pool backpressure.
const DECODE_BACKPRESSURE_BACKOFF: Duration = Duration::from_millis(50);

/// Maximum concurrent blocking disk-cache lookup tasks so cached tiles stay
/// responsive and never starve behind slow HTTP hosts.
const MAX_CONCURRENT_DISK_LOOKUPS: usize = 4;

/// Maximum concurrent blocking disk-cache write tasks (`cache.put()`) so burst
/// completions never pile up unbounded Tokio blocking threads on SQLite's
/// writer mutex.
const MAX_CONCURRENT_DISK_WRITES: usize = 4;

/// Maximum total pending + active post-decode disk-cache write tasks so SQLite
/// stalls never accumulate unbounded queued tasks and buffered tile payloads.
const MAX_QUEUED_DISK_WRITES: usize = 32;

/// Configuration for [`LiveTilePipeline`].
#[derive(Clone, Debug)]
pub struct FetcherConfig {
    pub user_agent: String,
    pub max_concurrent_per_host: usize,
    pub connect_timeout: Duration,
    pub request_timeout: Duration,
    pub max_tile_bytes: usize,
    pub transient_retry_backoff: Duration,
}

impl Default for FetcherConfig {
    fn default() -> Self {
        Self {
            user_agent: default_user_agent(),
            max_concurrent_per_host: DEFAULT_MAX_CONCURRENT_PER_HOST,
            connect_timeout: Duration::from_secs(10),
            request_timeout: Duration::from_secs(20),
            max_tile_bytes: DEFAULT_MAX_TILE_RESPONSE_BYTES,
            transient_retry_backoff: DEFAULT_TRANSIENT_RETRY_BACKOFF,
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

/// Shared immutable context passed across the dispatcher and per-tile tasks.
struct PipelineContext {
    state: Arc<Mutex<PipelineState>>,
    notify: Arc<Notify>,
    shutdown: CancelToken,
    client: reqwest::Client,
    host_limiters: Arc<HostLimiters>,
    disk_cache: Option<Arc<dyn TileBlobStore>>,
    disk_put_queue_sem: Arc<Semaphore>,
    disk_put_sem: Arc<Semaphore>,
    decoder: Arc<dyn TileDecodeExecutor>,
    wakeup: Wakeup,
    rt_handle: tokio::runtime::Handle,
    max_tile_bytes: usize,
    transient_retry_backoff: Duration,
}

/// Complete live HTTP tile streaming pipeline owning:
/// 1. `CancelToken` (`shutdown`)
/// 2. Dedicated `tokio::runtime::Runtime` running HTTP fetches
///
/// Decodes encoded tile payloads via the caller-injected [`TileDecodeExecutor`],
/// keeping `veldmap-net` free of `veldmap-raster` and image codecs.
/// Shuts down deterministically via [`Self::signal_shutdown`] / [`Self::shutdown`].
pub struct LiveTilePipeline {
    shutdown: CancelToken,
    state: Arc<Mutex<PipelineState>>,
    host_limiters: Arc<HostLimiters>,
    notify: Arc<Notify>,
    runtime: Option<tokio::runtime::Runtime>,
}

impl LiveTilePipeline {
    /// Spawn a live tile pipeline forwarding encoded tile payloads to `decoder`
    /// and invoking `wakeup` on status changes.
    pub fn spawn(
        config: FetcherConfig,
        disk_cache: Option<Arc<dyn TileBlobStore>>,
        decoder: Arc<dyn TileDecodeExecutor>,
        wakeup: Wakeup,
    ) -> Result<Self, NetError> {
        let client = build_http_client(&HttpClientConfig {
            user_agent: &config.user_agent,
            connect_timeout: config.connect_timeout,
            request_timeout: config.request_timeout,
            max_concurrent_per_host: config.max_concurrent_per_host,
        })?;
        let shutdown = CancelToken::new();

        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .thread_name("veldmap-net")
            .enable_all()
            .build()?;

        let state = Arc::new(Mutex::new(PipelineState::default()));
        let notify = Arc::new(Notify::new());
        let host_limiters = Arc::new(HostLimiters::new(config.max_concurrent_per_host));

        let ctx = Arc::new(PipelineContext {
            state: Arc::clone(&state),
            notify: Arc::clone(&notify),
            shutdown: shutdown.clone(),
            client,
            host_limiters: Arc::clone(&host_limiters),
            disk_cache,
            disk_put_queue_sem: Arc::new(Semaphore::new(MAX_QUEUED_DISK_WRITES)),
            disk_put_sem: Arc::new(Semaphore::new(MAX_CONCURRENT_DISK_WRITES)),
            decoder,
            wakeup,
            rt_handle: runtime.handle().clone(),
            max_tile_bytes: config.max_tile_bytes,
            transient_retry_backoff: config.transient_retry_backoff,
        });

        runtime.spawn(async move {
            dispatcher_loop(ctx).await;
        });

        Ok(Self {
            shutdown,
            state,
            host_limiters,
            notify,
            runtime: Some(runtime),
        })
    }

    /// Switch the active live basemap specification (or `None` when viewing
    /// offline / ellipsoid-only), immediately cancelling all in-flight HTTP
    /// and queued CPU decode jobs from any previous source.
    pub fn set_active_spec(&self, spec: Option<LiveBasemapSpec>) {
        let to_cancel: Vec<CancelToken> = {
            let mut st = lock_unpoisoned(&self.state);
            if st.active_spec.as_deref() == spec.as_ref() {
                return;
            }
            st.active_spec = spec.map(Arc::new);
            st.last_wanted.clear();
            st.pending_disk.clear();
            st.pending_http.clear();
            st.failures.clear();
            st.failed_count = 0;
            st.offline = false;
            st.last_error = None;
            self.host_limiters.clear_cooldowns();
            st.in_flight.drain().map(|(_, (_, tok))| tok).collect()
        };

        for tok in to_cancel {
            tok.cancel();
        }
        self.notify.notify_one();
    }

    /// Replace the visible-first fetch queue with `wanted`, immediately
    /// cancelling any in-flight HTTP or decode request no longer wanted
    /// (retaining active-source coarsest roots `key.is_root_of(active_source)`).
    ///
    /// Preserves already-checked tiles waiting in `pending_http` so they never
    /// bounce back to SQLite disk lookups on every frame/upload. Strictly
    /// filters out `PolarCap` keys and any `level > spec.max_level()`.
    pub fn reconcile_wanted<F>(&self, wanted: &HashSet<TileKey>, is_ready: F)
    where
        F: Fn(&TileKey) -> bool,
    {
        let now = Instant::now();
        let to_cancel: Vec<CancelToken> = {
            let mut st = lock_unpoisoned(&self.state);
            let Some(spec) = st.active_spec.clone() else {
                st.last_wanted.clear();
                st.pending_disk.clear();
                st.pending_http.clear();
                return;
            };
            let active_source = spec.source_id();

            let mut cancelled = Vec::new();
            st.in_flight.retain(|key, (_, tok)| {
                let keep = key.source == active_source
                    && !is_ready(key)
                    && (wanted.contains(key) || key.is_root_of(active_source));
                if !keep {
                    cancelled.push(tok.clone());
                }
                keep
            });

            // Prune expired backoff markers before rebuilding queues.
            st.failures.retain(|_, f| match f {
                FailureState::Terminal | FailureState::Empty => true,
                FailureState::BackoffUntil { until, .. } => *until > now,
            });

            // Track strictly unready, active-source, contract-valid wanted keys.
            st.last_wanted.clear();
            for &key in wanted {
                if key.source == active_source
                    && !is_ready(&key)
                    && key
                        .quad_id()
                        .is_some_and(|id| spec.validate_tile(id).is_ok())
                {
                    st.last_wanted.insert(key);
                }
            }

            // Retain still-wanted / active-root entries already in `pending_http`
            // so disk misses never bounce back to `pending_disk`.
            let mut already_in_http = HashSet::new();
            {
                let PipelineState {
                    last_wanted,
                    failures,
                    in_flight,
                    pending_http,
                    ..
                } = &mut *st;
                pending_http.retain(|key| {
                    let keep = key.source == active_source
                        && !is_ready(key)
                        && !in_flight.contains_key(key)
                        && !failures.contains_key(key)
                        && (last_wanted.contains(key) || key.is_root_of(active_source));
                    if keep {
                        already_in_http.insert(*key);
                    }
                    keep
                });
            }

            let mut next_disk = Vec::new();
            for &key in &st.last_wanted {
                if st.in_flight.contains_key(&key)
                    || st.failures.contains_key(&key)
                    || already_in_http.contains(&key)
                {
                    continue;
                }
                next_disk.push(key);
            }
            PipelineState::sort_queue_coarsest_at_end(&mut next_disk);
            PipelineState::sort_queue_coarsest_at_end(&mut st.pending_http);
            st.pending_disk = next_disk;
            cancelled
        };

        for tok in to_cancel {
            tok.cancel();
        }
        self.notify.notify_one();
    }

    /// Clear failure / offline markers and per-host cooldowns, immediately
    /// re-queueing any previously failed keys still in `last_wanted`.
    pub fn retry_failed(&self) {
        {
            let mut st = lock_unpoisoned(&self.state);
            let cleared: Vec<TileKey> = st.failures.drain().map(|(k, _)| k).collect();
            st.failed_count = 0;
            st.offline = false;
            st.last_error = None;
            self.host_limiters.clear_cooldowns();

            if let Some(spec) = st.active_spec.as_ref() {
                let active_source = spec.source_id();
                let mut queued_disk: HashSet<TileKey> = st.pending_disk.iter().copied().collect();
                let queued_http: HashSet<TileKey> = st.pending_http.iter().copied().collect();
                let mut added = false;
                for key in cleared {
                    if key.source == active_source
                        && st.last_wanted.contains(&key)
                        && !st.in_flight.contains_key(&key)
                        && !queued_http.contains(&key)
                        && queued_disk.insert(key)
                    {
                        st.pending_disk.push(key);
                        added = true;
                    }
                }
                if added {
                    PipelineState::sort_queue_coarsest_at_end(&mut st.pending_disk);
                }
            }
        }
        self.notify.notify_one();
    }

    /// Snapshot observable status counters for the GIS UI.
    pub fn status(&self) -> NetworkStatus {
        let st = lock_unpoisoned(&self.state);
        NetworkStatus {
            in_flight: st.in_flight.len(),
            queued: st.pending_disk.len().saturating_add(st.pending_http.len()),
            completed_from_disk: st.completed_from_disk,
            completed_from_net: st.completed_from_net,
            failed_count: st.failed_count,
            offline: st.offline,
            last_error: st.last_error.clone(),
        }
    }

    /// Trigger global and per-tile `CancelToken`s immediately (Phase 1 of
    /// deterministic shutdown, called BEFORE closing the upload channel or
    /// joining the caller-owned decode pool).
    pub fn signal_shutdown(&self) {
        self.shutdown.cancel();
        let active_tokens: Vec<CancelToken> = {
            let mut st = lock_unpoisoned(&self.state);
            st.last_wanted.clear();
            st.pending_disk.clear();
            st.pending_http.clear();
            st.in_flight.drain().map(|(_, (_, t))| t).collect()
        };
        for tok in active_tokens {
            tok.cancel();
        }
        self.notify.notify_waiters();
    }

    /// Deterministically shut down the async runtime after signalling
    /// cancellation (`CancelToken` -> `runtime.shutdown_timeout`).
    pub fn shutdown(mut self) {
        self.shutdown_inner();
    }

    fn shutdown_inner(&mut self) {
        self.signal_shutdown();
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
