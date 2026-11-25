//! Asynchronous HTTP tile fetcher with two-stage disk/HTTP scheduling,
//! per-host concurrency & rate-limit cooldowns, replaceable visible-first
//! queues, end-to-end `CancelToken` abort through an injected
//! [`TileDecodeExecutor`], and deterministic shutdown.

mod context;
mod disk;
mod dispatch;
mod guards;
mod state;
mod workers;

use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::sync::Notify;
use veldmap_core::sync::{CancelToken, Wakeup, lock_unpoisoned};
use veldmap_core::tile::TileKey;

use crate::basemap::LiveBasemapSpec;
use crate::http::{HostLimiters, HttpClientConfig, build_http_client, default_user_agent};
use crate::{NetError, TileBlobStore, TileDecodeExecutor};

use self::context::{PipelineContext, PipelineShared};
use self::disk::DiskCacheGate;
use self::dispatch::dispatcher_loop;
pub use self::state::NetworkStatus;
use self::state::PipelineState;

/// Default maximum concurrent HTTP requests per remote host.
const DEFAULT_MAX_CONCURRENT_PER_HOST: usize = 6;

/// Default maximum encoded response body accepted for one live tile (`16 MiB`).
const DEFAULT_MAX_TILE_RESPONSE_BYTES: usize = 16 * 1024 * 1024;

/// Default backoff duration after transient HTTP / network errors before
/// retrying a tile.
const DEFAULT_TRANSIENT_RETRY_BACKOFF: Duration = Duration::from_secs(15);

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

/// Complete live HTTP tile streaming pipeline owning:
/// 1. Shared control state (`CancelToken`, queue/in-flight state, host limiters)
/// 2. Dedicated `tokio::runtime::Runtime` running HTTP fetches
///
/// Decodes encoded tile payloads via the caller-injected [`TileDecodeExecutor`],
/// keeping `veldmap-net` free of `veldmap-raster` and image codecs.
/// Shuts down deterministically via [`Self::signal_shutdown`] / [`Self::shutdown`].
pub struct LiveTilePipeline {
    shared: Arc<PipelineShared>,
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

        let shared = Arc::new(PipelineShared {
            shutdown,
            state: Mutex::new(PipelineState::new(disk_cache.is_some())),
            host_limiters: HostLimiters::new(config.max_concurrent_per_host),
            notify: Notify::new(),
        });

        let ctx = Arc::new(PipelineContext {
            shared: Arc::clone(&shared),
            client,
            disk_cache: disk_cache.map(DiskCacheGate::new),
            decoder,
            wakeup,
            max_tile_bytes: config.max_tile_bytes,
            transient_retry_backoff: config.transient_retry_backoff,
        });

        runtime.spawn(async move {
            dispatcher_loop(ctx).await;
        });

        Ok(Self {
            shared,
            runtime: Some(runtime),
        })
    }

    /// Switch the active live basemap specification (or `None` when viewing
    /// offline / ellipsoid-only), immediately cancelling all in-flight HTTP
    /// and queued CPU decode jobs from any previous source.
    pub fn set_active_spec(&self, spec: Option<LiveBasemapSpec>) {
        let to_cancel = {
            let mut st = lock_unpoisoned(&self.shared.state);
            let Some(tokens) = st.set_active_spec(spec) else {
                return;
            };
            self.shared.host_limiters.clear_cooldowns();
            tokens
        };

        for tok in to_cancel {
            tok.cancel();
        }
        self.shared.notify.notify_one();
    }

    /// Replace the visible-first fetch queue with `wanted`, immediately
    /// cancelling any in-flight HTTP or decode request no longer wanted
    /// (retaining active-source coarsest roots `key.is_root_of(active_source)`).
    ///
    /// Preserves already-checked tiles waiting in `pending_http` so they never
    /// bounce back to disk-cache lookups on every frame/upload. Strictly
    /// filters out `PolarCap` keys and any `level > spec.max_level()`.
    pub fn reconcile_wanted<F>(&self, wanted: &HashSet<TileKey>, is_ready: F)
    where
        F: Fn(&TileKey) -> bool,
    {
        let now = Instant::now();
        let to_cancel = {
            let mut st = lock_unpoisoned(&self.shared.state);
            let Some(tokens) = st.reconcile_wanted(wanted, &is_ready, now) else {
                return;
            };
            tokens
        };

        for tok in to_cancel {
            tok.cancel();
        }
        self.shared.notify.notify_one();
    }

    /// Clear failure / offline markers and per-host cooldowns, immediately
    /// re-queueing any previously failed keys still in `last_wanted`.
    pub fn retry_failed(&self) {
        {
            let mut st = lock_unpoisoned(&self.shared.state);
            self.shared.host_limiters.clear_cooldowns();
            st.retry_failed();
        }
        self.shared.notify.notify_one();
    }

    /// Snapshot observable status counters for the GIS UI.
    pub fn status(&self) -> NetworkStatus {
        lock_unpoisoned(&self.shared.state).status()
    }

    /// Trigger global and per-tile `CancelToken`s immediately (Phase 1 of
    /// deterministic shutdown, called BEFORE closing the upload channel or
    /// joining the caller-owned decode pool).
    pub fn signal_shutdown(&self) {
        self.shared.shutdown.cancel();
        let active_tokens = lock_unpoisoned(&self.shared.state).signal_shutdown();
        for tok in active_tokens {
            tok.cancel();
        }
        self.shared.notify.notify_waiters();
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
