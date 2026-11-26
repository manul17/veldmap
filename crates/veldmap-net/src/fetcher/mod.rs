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
use veldmap_core::sync::{Wakeup, lock_unpoisoned};
use veldmap_core::tile::TileKey;

use crate::basemap::LiveBasemapSpec;
use crate::context::{NetConfig, NetContext};
use crate::http::default_user_agent;
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

/// Complete live HTTP tile streaming pipeline sharing a [`NetContext`] (or
/// owning a private one when constructed via [`Self::spawn`]).
///
/// Decodes encoded tile payloads via the caller-injected [`TileDecodeExecutor`],
/// keeping `veldmap-net` free of `veldmap-raster` and image codecs.
/// Shuts down deterministically via [`Self::signal_shutdown`] / [`Self::shutdown`].
pub struct LiveTilePipeline {
    shared: Arc<PipelineShared>,
    owned_net: Option<Arc<NetContext>>,
}

impl LiveTilePipeline {
    /// Spawn a live tile pipeline backed by a dedicated [`NetContext`].
    pub fn spawn(
        config: FetcherConfig,
        disk_cache: Option<Arc<dyn TileBlobStore>>,
        decoder: Arc<dyn TileDecodeExecutor>,
        wakeup: Wakeup,
    ) -> Result<Self, NetError> {
        let net_cfg = NetConfig {
            user_agent: config.user_agent.clone(),
            max_concurrent_per_host: config.max_concurrent_per_host,
            connect_timeout: config.connect_timeout,
            read_timeout: config.request_timeout,
            ..NetConfig::default()
        };
        let net = Arc::new(NetContext::new(net_cfg)?);
        let mut pipeline =
            Self::spawn_with_context(Arc::clone(&net), config, disk_cache, decoder, wakeup);
        pipeline.owned_net = Some(net);
        Ok(pipeline)
    }

    /// Spawn a live tile pipeline sharing an existing [`NetContext`]'s Tokio
    /// runtime, HTTP client pool, and per-host limiters.
    ///
    /// Note: transport-level settings (`user_agent`, `max_concurrent_per_host`,
    /// `connect_timeout`, and `request_timeout`) are governed by the shared
    /// [`NetContext`]; `config.max_tile_bytes` and `config.transient_retry_backoff`
    /// still apply per pipeline.
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

        Self {
            shared,
            owned_net: None,
        }
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
        self.shared.shutdown.cancel();
        let active_tokens = lock_unpoisoned(&self.shared.state).signal_shutdown();
        for tok in active_tokens {
            tok.cancel();
        }
        self.shared.notify.notify_waiters();
    }

    /// Deterministically shut down the pipeline (and its private async runtime
    /// when constructed via [`Self::spawn`]).
    pub fn shutdown(mut self) {
        self.shutdown_inner();
    }

    fn shutdown_inner(&mut self) {
        self.signal_shutdown();
        if let Some(net) = self.owned_net.take() {
            net.shutdown();
        }
    }
}

impl Drop for LiveTilePipeline {
    fn drop(&mut self) {
        self.shutdown_inner();
    }
}
