//! Shared asynchronous networking runtime (`NetContext`) owning the multi-thread
//! Tokio runtime, split identity/gzip HTTP clients, host-scoped concurrency and
//! cooldown limiters, the bounded 64 KiB HTTP Range LRU block pool, and the
//! upfront pinned-prefetch memory permit tracker.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use veldmap_core::budget::MemoryBudget;
use veldmap_core::sync::{CancelToken, lock_unpoisoned};

use crate::NetError;
use crate::http::{HostLimiters, HttpClientConfig, build_http_client, default_user_agent};
use crate::range::{PrefetchBudget, RangeBlockPool};

/// Default maximum concurrent HTTP requests per remote host.
const DEFAULT_MAX_CONCURRENT_PER_HOST: usize = 6;

/// Default Tokio worker thread count for shared network I/O (minimum 2 so
/// concurrent basemap, COG range, and STAC streams never serialize on 1 core).
const DEFAULT_RUNTIME_WORKER_THREADS: usize = 2;

/// Configuration for [`NetContext`].
#[derive(Clone, Debug)]
pub struct NetConfig {
    pub user_agent: String,
    pub worker_threads: usize,
    pub max_concurrent_per_host: usize,
    pub connect_timeout: Duration,
    pub read_timeout: Duration,
    pub range_block_pool_bytes: usize,
    pub remote_prefetch_bytes: usize,
}

impl NetConfig {
    /// Construct a [`NetConfig`] scaled to the host's [`MemoryBudget`].
    pub fn from_budget(budget: &MemoryBudget) -> Self {
        Self {
            range_block_pool_bytes: budget.range_block_pool_bytes,
            remote_prefetch_bytes: budget.remote_prefetch_bytes,
            ..Self::default()
        }
    }
}

impl Default for NetConfig {
    fn default() -> Self {
        let budget = MemoryBudget::default();
        Self {
            user_agent: default_user_agent(),
            worker_threads: DEFAULT_RUNTIME_WORKER_THREADS,
            max_concurrent_per_host: DEFAULT_MAX_CONCURRENT_PER_HOST,
            connect_timeout: Duration::from_secs(10),
            read_timeout: Duration::from_secs(20),
            range_block_pool_bytes: budget.range_block_pool_bytes,
            remote_prefetch_bytes: budget.remote_prefetch_bytes,
        }
    }
}

/// Shared networking context owned by the application and injected into
/// [`crate::LiveTilePipeline`], [`crate::HttpRangeSource`], and STAC clients.
pub struct NetContext {
    runtime: Mutex<Option<tokio::runtime::Runtime>>,
    handle: tokio::runtime::Handle,
    shutdown: CancelToken,
    host_limiters: Arc<HostLimiters>,
    range_client: reqwest::Client,
    range_auth_client: reqwest::Client,
    api_client: reqwest::Client,
    api_auth_client: reqwest::Client,
    range_pool: Arc<RangeBlockPool>,
    prefetch_budget: Arc<PrefetchBudget>,
    next_instance_id: AtomicU64,
}

impl NetContext {
    /// Initialize a shared [`NetContext`] with a multi-thread Tokio runtime,
    /// non-decompressing range/tile clients, gzip-capable JSON API clients
    /// (with same-origin redirect guards when custom auth headers are sent),
    /// shared per-host limiters, and bounded range block / prefetch pools.
    pub fn new(config: NetConfig) -> Result<Self, NetError> {
        let build_client = |enable_gzip, same_origin_redirects| {
            build_http_client(&HttpClientConfig {
                user_agent: &config.user_agent,
                connect_timeout: config.connect_timeout,
                read_timeout: config.read_timeout,
                max_concurrent_per_host: config.max_concurrent_per_host,
                enable_gzip,
                same_origin_redirects,
            })
        };
        let range_client = build_client(false, false)?;
        let range_auth_client = build_client(false, true)?;
        let api_client = build_client(true, false)?;
        let api_auth_client = build_client(true, true)?;

        let workers = config.worker_threads.max(DEFAULT_RUNTIME_WORKER_THREADS);
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(workers)
            .thread_name("veldmap-net")
            .enable_all()
            .build()?;
        let handle = runtime.handle().clone();

        Ok(Self {
            runtime: Mutex::new(Some(runtime)),
            handle,
            shutdown: CancelToken::new(),
            host_limiters: Arc::new(HostLimiters::new(config.max_concurrent_per_host)),
            range_client,
            range_auth_client,
            api_client,
            api_auth_client,
            range_pool: Arc::new(RangeBlockPool::new(config.range_block_pool_bytes)),
            prefetch_budget: Arc::new(PrefetchBudget::new(config.remote_prefetch_bytes)),
            next_instance_id: AtomicU64::new(1),
        })
    }

    /// Borrow the Tokio runtime [`tokio::runtime::Handle`] for spawning async tasks.
    pub fn handle(&self) -> &tokio::runtime::Handle {
        &self.handle
    }

    /// Borrow the root shutdown [`CancelToken`] for all network subsystems.
    pub fn shutdown_token(&self) -> &CancelToken {
        &self.shutdown
    }

    /// Borrow the non-decompressing HTTP client used by HTTP Range reads and
    /// pre-compressed map tiles.
    pub(crate) fn range_client(&self) -> &reqwest::Client {
        &self.range_client
    }

    /// Borrow the non-decompressing HTTP client appropriate for requests with or
    /// without custom auth headers (custom-header requests forbid cross-origin
    /// redirects so secrets never leak to third-party origins).
    pub(crate) fn range_client_for(&self, has_custom_headers: bool) -> &reqwest::Client {
        if has_custom_headers {
            &self.range_auth_client
        } else {
            &self.range_client
        }
    }

    /// Borrow the gzip-capable HTTP client used by JSON metadata / STAC APIs.
    pub(crate) fn api_client(&self) -> &reqwest::Client {
        &self.api_client
    }

    /// Borrow the same-origin-redirect-guarded gzip HTTP client used by STAC
    /// requests carrying custom auth headers.
    pub(crate) fn api_auth_client(&self) -> &reqwest::Client {
        &self.api_auth_client
    }

    /// Clone the shared per-host concurrency and cooldown limiters.
    pub(crate) fn host_limiters(&self) -> Arc<HostLimiters> {
        Arc::clone(&self.host_limiters)
    }

    /// Clone the shared 64 KiB O(1) LRU HTTP Range block pool.
    pub(crate) fn range_pool(&self) -> Arc<RangeBlockPool> {
        Arc::clone(&self.range_pool)
    }

    /// Clone the shared upfront pinned-prefetch memory permit tracker.
    pub(crate) fn prefetch_budget(&self) -> Arc<PrefetchBudget> {
        Arc::clone(&self.prefetch_budget)
    }

    /// Allocate a process-unique instance ID for isolating weak/missing-ETag
    /// sources inside the shared [`RangeBlockPool`].
    pub(crate) fn alloc_source_instance_id(&self) -> u64 {
        self.next_instance_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Clear rate-limit cooldowns strictly for `hosts` (leaving all other hosts
    /// untouched).
    pub fn clear_cooldowns_for<'a>(&self, hosts: impl IntoIterator<Item = &'a str>) {
        self.host_limiters.clear_cooldowns_for(hosts);
    }

    /// Signal global cancellation across all live pipelines, range readers,
    /// and STAC requests sharing this context (Phase 1 of shutdown).
    pub fn signal_shutdown(&self) {
        self.shutdown.cancel();
    }

    /// Cancel all work and shut down the Tokio runtime cleanly with a bounded
    /// timeout (called after worker pools depending on `NetContext` have been
    /// joined).
    pub fn shutdown(&self) {
        self.signal_shutdown();
        let rt = lock_unpoisoned(&self.runtime).take();
        if let Some(rt) = rt {
            if tokio::runtime::Handle::try_current().is_ok() {
                rt.shutdown_background();
            } else {
                rt.shutdown_timeout(Duration::from_secs(2));
            }
        }
    }
}

impl Drop for NetContext {
    fn drop(&mut self) {
        self.shutdown();
    }
}
