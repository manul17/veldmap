//! Asynchronous HTTP tile fetcher with two-stage disk/HTTP scheduling,
//! per-host concurrency & rate-limit cooldowns, replaceable visible-first
//! queues, end-to-end `CancelToken` abort through an injected
//! [`TileDecodeExecutor`], and deterministic shutdown.

mod disk;
mod dispatch;
mod download;
mod guards;
mod ports;
mod state;
mod workers;

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::sync::Notify;
use veldmap_core::sync::{CancelToken, Wakeup, lock_unpoisoned};
use veldmap_core::tile::TileKey;

use crate::basemap::LiveBasemapSpec;
use crate::context::NetContext;
use crate::http::HostLimiters;

use self::disk::DiskCacheGate;
use self::dispatch::dispatcher_loop;
#[cfg(any(test, feature = "test-support"))]
pub use self::ports::test_support;
pub use self::ports::{
    BlobLease, BlobStoreError, DecodeCompletion, DecodeFailure, DecodeRequest, TileBlobStore,
    TileDecodeExecutor,
};
pub use self::state::NetworkStatus;
use self::state::PipelineState;

/// Default maximum encoded response body accepted for one live tile (`16 MiB`).
const DEFAULT_MAX_TILE_RESPONSE_BYTES: usize = 16 * 1024 * 1024;

/// Default backoff duration after transient HTTP / network errors before
/// retrying a tile.
const DEFAULT_TRANSIENT_RETRY_BACKOFF: Duration = Duration::from_secs(15);

/// Shared control-plane handles held by both [`LiveTilePipeline`] and
/// [`PipelineContext`].
///
/// Kept distinct from [`PipelineContext`] so the frontend handle never extends
/// the lifetimes of `reqwest::Client`, `DiskCacheGate`, `TileDecodeExecutor`,
/// or `Wakeup` past `dispatcher_loop` exit.
///
/// Lock ordering: always acquire `state` BEFORE `host_limiters` (never
/// reverse; `HostLimiters` is a leaf lock).
struct PipelineShared {
    shutdown: CancelToken,
    shutdown_signaled: AtomicBool,
    state: Mutex<PipelineState>,
    host_limiters: Arc<HostLimiters>,
    notify: Notify,
}

/// Shared immutable context passed across the dispatcher and per-tile tasks.
struct PipelineContext {
    shared: Arc<PipelineShared>,
    client: reqwest::Client,
    disk_cache: Option<DiskCacheGate>,
    decoder: Arc<dyn TileDecodeExecutor>,
    wakeup: Wakeup,
    max_tile_bytes: usize,
    transient_retry_backoff: Duration,
}

/// Per-pipeline configuration for [`LiveTilePipeline`].
///
/// Transport-level settings (`user_agent`, `max_concurrent_per_host`,
/// `connect_timeout`, and `read_timeout`) are governed by the shared
/// [`NetContext`].
#[derive(Clone, Debug)]
pub struct FetcherConfig {
    pub max_tile_bytes: usize,
    pub transient_retry_backoff: Duration,
}

impl Default for FetcherConfig {
    fn default() -> Self {
        Self {
            max_tile_bytes: DEFAULT_MAX_TILE_RESPONSE_BYTES,
            transient_retry_backoff: DEFAULT_TRANSIENT_RETRY_BACKOFF,
        }
    }
}

/// Complete live HTTP tile streaming pipeline sharing a [`NetContext`].
///
/// Decodes encoded tile payloads via the caller-injected [`TileDecodeExecutor`],
/// keeping `veldmap-net` free of `veldmap-raster` and image codecs.
/// Shuts down deterministically via [`Self::signal_shutdown`] / [`Self::shutdown`].
pub struct LiveTilePipeline {
    shared: Arc<PipelineShared>,
}

impl LiveTilePipeline {
    /// Spawn a live tile pipeline sharing an existing [`NetContext`]'s Tokio
    /// runtime, HTTP client pool, and per-host limiters.
    pub fn spawn_with_context(
        net: Arc<NetContext>,
        config: FetcherConfig,
        disk_cache: Option<Arc<dyn TileBlobStore>>,
        decoder: Arc<dyn TileDecodeExecutor>,
        wakeup: Wakeup,
    ) -> Self {
        let shutdown = net.shutdown_token().child_token();

        let shared = Arc::new(PipelineShared {
            shutdown,
            shutdown_signaled: AtomicBool::new(false),
            state: Mutex::new(PipelineState::new(disk_cache.is_some())),
            host_limiters: net.host_limiters(),
            notify: Notify::new(),
        });

        let ctx = Arc::new(PipelineContext {
            shared: Arc::clone(&shared),
            client: net.range_client().clone(),
            disk_cache: disk_cache.map(DiskCacheGate::new),
            decoder,
            wakeup,
            max_tile_bytes: config.max_tile_bytes,
            transient_retry_backoff: config.transient_retry_backoff,
        });

        net.handle().spawn(async move {
            dispatcher_loop(ctx).await;
        });

        Self { shared }
    }

    /// Switch the active live basemap specification (or `None` when viewing
    /// offline / ellipsoid-only), immediately cancelling all in-flight HTTP
    /// and queued CPU decode jobs from any previous source.
    pub fn set_active_spec(&self, spec: Option<LiveBasemapSpec>) {
        let to_cancel = {
            let mut st = lock_unpoisoned(&self.shared.state);
            let Some((tokens, hosts)) = st.set_active_spec(spec) else {
                return;
            };
            self.shared
                .host_limiters
                .clear_cooldowns_for(hosts.iter().map(String::as_str));
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

    /// Clear failure / offline markers and per-host cooldowns strictly for the
    /// active basemap's hosts, immediately re-queueing any previously failed
    /// keys still in `last_wanted`.
    pub fn retry_failed(&self) {
        {
            let mut st = lock_unpoisoned(&self.shared.state);
            let hosts = st.retry_failed();
            self.shared
                .host_limiters
                .clear_cooldowns_for(hosts.iter().map(String::as_str));
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
        if self.shared.shutdown_signaled.swap(true, Ordering::AcqRel) {
            return;
        }
        self.shared.shutdown.cancel();
        let active_tokens = lock_unpoisoned(&self.shared.state).signal_shutdown();
        for tok in active_tokens {
            tok.cancel();
        }
        self.shared.notify.notify_waiters();
    }

    /// Deterministically shut down the pipeline by dropping it (invoking
    /// [`Self::signal_shutdown`] via `Drop`).
    pub fn shutdown(self) {}
}

impl Drop for LiveTilePipeline {
    fn drop(&mut self) {
        self.signal_shutdown();
    }
}
