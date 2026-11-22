//! Asynchronous HTTP tile fetcher with two-stage disk/HTTP scheduling,
//! per-host concurrency & rate-limit cooldowns, replaceable visible-first
//! queues, end-to-end `CancelToken` abort through an injected
//! [`TileDecodeExecutor`], and deterministic shutdown.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::sync::{Notify, OwnedSemaphorePermit, Semaphore};
use veldmap_core::io::{CancelToken, Wakeup, lock_unpoisoned};
use veldmap_core::tile::{PlanetaryScheme, TileKey};

use crate::basemap::LiveBasemapSpec;
use crate::http::{
    DEFAULT_MAX_CONCURRENT_PER_HOST, DEFAULT_MAX_TILE_RESPONSE_BYTES, FetchFailure, FetchOutcome,
    HostLimiters, HttpClientConfig, build_http_client, default_user_agent, download_capped,
    extract_host,
};
use crate::{DecodeRequest, NetError, TileBlobStore, TileDecodeExecutor};

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

/// Upper bound on retained per-tile failure/empty markers to prevent unbounded
/// memory growth on sparse 404 layers over long sessions.
const MAX_FAILURE_ENTRIES: usize = 4096;

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

#[derive(Clone, Copy, Debug)]
enum FailureState {
    Terminal,
    /// HTTP `204 No Content`: suppresses infinite refetch without flagging an error.
    Empty,
    BackoffUntil {
        until: Instant,
        retry_disk: bool,
    },
}

#[derive(Default)]
struct PipelineState {
    active_spec: Option<Arc<LiveBasemapSpec>>,
    /// Unready, active-source, contract-valid keys from the most recent
    /// `reconcile_wanted`, used by idle backoff retries and `retry_failed`.
    last_wanted: HashSet<TileKey>,
    /// Coarsest-first queue awaiting fast local disk-cache lookup (stored
    /// reverse-sorted so `pop()` yields the coarsest key in O(1)).
    pending_disk: Vec<TileKey>,
    /// Coarsest-first queue of disk misses awaiting an HTTP per-host slot.
    pending_http: Vec<TileKey>,
    /// Active blocking disk-lookup tasks (bounded by `MAX_CONCURRENT_DISK_LOOKUPS`).
    disk_in_flight: usize,
    /// Tracks active `(generation, CancelToken)` from dispatch all the way
    /// through CPU decode completion so stale guards never clobber a newer
    /// spawn and pan/switch cancels queued decode jobs immediately.
    in_flight: HashMap<TileKey, (u64, CancelToken)>,
    next_gen: u64,
    failures: HashMap<TileKey, FailureState>,
    completed_from_disk: u64,
    completed_from_net: u64,
    failed_count: u64,
    offline: bool,
    last_error: Option<String>,
}

impl PipelineState {
    fn sort_queue_coarsest_at_end(queue: &mut [TileKey]) {
        queue.sort_by_key(|k| std::cmp::Reverse(k.coarsest_first_key()));
    }

    fn insert_pending_http(&mut self, key: TileKey) {
        if !self.pending_http.contains(&key) {
            self.pending_http.push(key);
            Self::sort_queue_coarsest_at_end(&mut self.pending_http);
        }
    }

    fn insert_failure(&mut self, key: TileKey, failure: FailureState, now: Instant) {
        if self.failures.len() >= MAX_FAILURE_ENTRIES && !self.failures.contains_key(&key) {
            self.failures.retain(|_, f| match f {
                FailureState::BackoffUntil { until, .. } => *until > now,
                FailureState::Terminal | FailureState::Empty => true,
            });
            if self.failures.len() >= MAX_FAILURE_ENTRIES {
                let wanted = &self.last_wanted;
                self.failures.retain(|k, _| wanted.contains(k));
            }
            if self.failures.len() >= MAX_FAILURE_ENTRIES
                && let Some(&evict_key) = self.failures.keys().next()
            {
                self.failures.remove(&evict_key);
            }
        }
        self.failures.insert(key, failure);
    }

    /// Prune all expired `BackoffUntil <= now` markers and re-enqueue any that
    /// are still in `last_wanted` into `pending_disk` or `pending_http`.
    /// Returns whether new work was queued.
    fn prune_expired_backoffs(&mut self, now: Instant) -> bool {
        let Some(spec) = self.active_spec.as_ref() else {
            self.failures.clear();
            return false;
        };
        let active_source = spec.source_id();
        let mut requeued_disk = false;
        let mut requeued_http = false;
        let mut expired = Vec::new();
        self.failures.retain(|&key, f| match f {
            FailureState::BackoffUntil { until, retry_disk } if *until <= now => {
                expired.push((key, *retry_disk));
                false
            }
            _ => true,
        });
        for (key, retry_disk) in expired {
            if key.source == active_source
                && self.last_wanted.contains(&key)
                && !self.in_flight.contains_key(&key)
                && !self.pending_disk.contains(&key)
                && !self.pending_http.contains(&key)
            {
                if retry_disk {
                    self.pending_disk.push(key);
                    requeued_disk = true;
                } else {
                    self.pending_http.push(key);
                    requeued_http = true;
                }
            }
        }
        if requeued_disk {
            Self::sort_queue_coarsest_at_end(&mut self.pending_disk);
        }
        if requeued_http {
            Self::sort_queue_coarsest_at_end(&mut self.pending_http);
        }
        requeued_disk || requeued_http
    }

    /// Earliest `BackoffUntil` timestamp among currently wanted tiles (used by
    /// `dispatcher_loop` to self-wake for idle retries without busy-spinning).
    fn earliest_wanted_backoff(&self, now: Instant) -> Option<Instant> {
        self.failures
            .iter()
            .filter_map(|(k, f)| match f {
                FailureState::BackoffUntil { until, .. }
                    if *until > now && self.last_wanted.contains(k) =>
                {
                    Some(*until)
                }
                _ => None,
            })
            .min()
    }
}

/// Shared immutable context passed across the dispatcher and per-tile tasks.
struct PipelineContext {
    state: Arc<Mutex<PipelineState>>,
    notify: Arc<Notify>,
    shutdown: CancelToken,
    client: reqwest::Client,
    host_limiters: Arc<HostLimiters>,
    disk_cache: Option<Arc<dyn TileBlobStore>>,
    disk_put_sem: Arc<Semaphore>,
    decoder: Arc<dyn TileDecodeExecutor>,
    wakeup: Wakeup,
    rt_handle: tokio::runtime::Handle,
    max_tile_bytes: usize,
    transient_retry_backoff: Duration,
}

/// Generation-checked RAII guard keeping `key` registered in `state.in_flight`
/// across async fetch, disk lookup/delete, and CPU decode submission.
struct InFlightGuard {
    state: Arc<Mutex<PipelineState>>,
    key: TileKey,
    generation: u64,
    notify: Arc<Notify>,
    armed: bool,
}

impl InFlightGuard {
    fn new(
        state: Arc<Mutex<PipelineState>>,
        key: TileKey,
        generation: u64,
        notify: Arc<Notify>,
    ) -> Self {
        Self {
            state,
            key,
            generation,
            notify,
            armed: true,
        }
    }

    /// Disarm this guard after manually removing `key` from `in_flight` under
    /// `PipelineState`'s lock.
    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
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

/// RAII guard for the bounded disk-lookup lane: decrements `disk_in_flight`
/// under lock BEFORE notifying `dispatcher_loop`.
struct DiskSlotGuard {
    state: Arc<Mutex<PipelineState>>,
    notify: Arc<Notify>,
}

impl Drop for DiskSlotGuard {
    fn drop(&mut self) {
        {
            let mut st = lock_unpoisoned(&self.state);
            st.disk_in_flight = st.disk_in_flight.saturating_sub(1);
        }
        self.notify.notify_one();
    }
}

/// RAII guard for a per-host HTTP semaphore permit.
///
/// Explicitly releases `permit` BEFORE calling `notify.notify_one()` so
/// `dispatcher_loop`'s `try_acquire_owned()` never races ahead of permit drop.
struct PermitGuard {
    permit: Option<OwnedSemaphorePermit>,
    notify: Arc<Notify>,
}

impl Drop for PermitGuard {
    fn drop(&mut self) {
        drop(self.permit.take());
        self.notify.notify_one();
    }
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
                let mut added = false;
                for key in cleared {
                    if key.source == active_source
                        && st.last_wanted.contains(&key)
                        && !st.in_flight.contains_key(&key)
                        && !st.pending_disk.contains(&key)
                        && !st.pending_http.contains(&key)
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

struct HttpTask {
    key: TileKey,
    generation: u64,
    spec: Arc<LiveBasemapSpec>,
    url: String,
    host: String,
    permit: OwnedSemaphorePermit,
    cancel: CancelToken,
}

async fn dispatcher_loop(ctx: Arc<PipelineContext>) {
    loop {
        if ctx.shutdown.is_cancelled() {
            break;
        }

        let now = Instant::now();
        let mut disk_tasks: Vec<(TileKey, u64, Arc<LiveBasemapSpec>, CancelToken)> = Vec::new();
        let mut http_tasks: Vec<HttpTask> = Vec::new();
        let mut earliest_wakeup: Option<Instant> = None;

        {
            let mut st = lock_unpoisoned(&ctx.state);
            st.prune_expired_backoffs(now);

            if let Some(spec) = st.active_spec.clone() {
                let active_source = spec.source_id();
                // Stage 1: Pop from `pending_disk` (or batch-move straight to
                // `pending_http` when no disk cache is configured).
                if ctx.disk_cache.is_none() {
                    if !st.pending_disk.is_empty() {
                        let drained: Vec<TileKey> = st.pending_disk.drain(..).collect();
                        let mut existing: HashSet<TileKey> =
                            st.pending_http.iter().copied().collect();
                        let mut added = false;
                        for key in drained {
                            if key.source == active_source
                                && !st.in_flight.contains_key(&key)
                                && !st.failures.contains_key(&key)
                                && existing.insert(key)
                            {
                                st.pending_http.push(key);
                                added = true;
                            }
                        }
                        if added {
                            PipelineState::sort_queue_coarsest_at_end(&mut st.pending_http);
                        }
                    }
                } else {
                    while st.disk_in_flight < MAX_CONCURRENT_DISK_LOOKUPS {
                        let Some(key) = st.pending_disk.pop() else {
                            break;
                        };
                        if key.source != active_source
                            || st.in_flight.contains_key(&key)
                            || st.failures.contains_key(&key)
                        {
                            continue;
                        }
                        st.next_gen = st.next_gen.wrapping_add(1);
                        let gen_id = st.next_gen;
                        let tile_cancel = ctx.shutdown.child_token();
                        st.disk_in_flight += 1;
                        st.in_flight.insert(key, (gen_id, tile_cancel.clone()));
                        disk_tasks.push((key, gen_id, Arc::clone(&spec), tile_cancel));
                    }
                }

                // Stage 2: Pop from `pending_http` ONLY when a per-host permit
                // is immediately available and the host is not cooling down.
                // Short-circuits immediately once a single-host spec (no `{s}`)
                // is saturated or cooling down so remaining queued tiles are
                // preserved without redundant URL resolution or re-sorting.
                let multi_host = spec.uses_subdomains();
                let mut deferred_http = Vec::new();
                while let Some(key) = st.pending_http.pop() {
                    if key.source != active_source
                        || st.in_flight.contains_key(&key)
                        || st.failures.contains_key(&key)
                    {
                        continue;
                    }
                    let Some(id) = key.quad_id() else {
                        continue;
                    };
                    if matches!(id.scheme, PlanetaryScheme::PolarCap { .. }) {
                        continue;
                    }
                    let Ok(url) = spec.resolve_url(id) else {
                        st.insert_failure(key, FailureState::Terminal, now);
                        continue;
                    };
                    let host = extract_host(&url);
                    if let Some(until) = ctx.host_limiters.cooldown_until(&host, now) {
                        earliest_wakeup = Some(match earliest_wakeup {
                            Some(cur) if cur < until => cur,
                            _ => until,
                        });
                        deferred_http.push(key);
                        if !multi_host {
                            break;
                        }
                        continue;
                    }
                    let sem = ctx.host_limiters.semaphore_for_host(&host);
                    match sem.try_acquire_owned() {
                        Ok(permit) => {
                            st.next_gen = st.next_gen.wrapping_add(1);
                            let gen_id = st.next_gen;
                            let tile_cancel = ctx.shutdown.child_token();
                            st.in_flight.insert(key, (gen_id, tile_cancel.clone()));
                            http_tasks.push(HttpTask {
                                key,
                                generation: gen_id,
                                spec: Arc::clone(&spec),
                                url,
                                host,
                                permit,
                                cancel: tile_cancel,
                            });
                        }
                        Err(_) => {
                            deferred_http.push(key);
                            if !multi_host {
                                break;
                            }
                        }
                    }
                }
                if !deferred_http.is_empty() {
                    // `deferred_http` was popped coarsest-first from the back of
                    // `st.pending_http`; reversing restores ascending-priority
                    // order (coarsest at the end) cleanly without a full sort
                    // for single-host short-circuits, and sorting when mixed.
                    if multi_host {
                        st.pending_http.extend(deferred_http);
                        PipelineState::sort_queue_coarsest_at_end(&mut st.pending_http);
                    } else {
                        st.pending_http.extend(deferred_http.into_iter().rev());
                    }
                }
            } else {
                st.pending_disk.clear();
                st.pending_http.clear();
            }

            if let Some(backoff_at) = st.earliest_wanted_backoff(now) {
                earliest_wakeup = Some(match earliest_wakeup {
                    Some(cur) if cur < backoff_at => cur,
                    _ => backoff_at,
                });
            }
        }

        if disk_tasks.is_empty() && http_tasks.is_empty() {
            if let Some(wake_at) = earliest_wakeup {
                let deadline = tokio::time::Instant::from_std(wake_at);
                tokio::select! {
                    _ = ctx.shutdown.cancelled() => break,
                    _ = ctx.notify.notified() => continue,
                    _ = tokio::time::sleep_until(deadline) => continue,
                }
            } else {
                tokio::select! {
                    _ = ctx.shutdown.cancelled() => break,
                    _ = ctx.notify.notified() => continue,
                }
            }
        }

        for (key, generation, spec, tile_cancel) in disk_tasks {
            let task_ctx = Arc::clone(&ctx);
            tokio::spawn(async move {
                let guard = InFlightGuard::new(
                    Arc::clone(&task_ctx.state),
                    key,
                    generation,
                    Arc::clone(&task_ctx.notify),
                );
                let disk_slot = DiskSlotGuard {
                    state: Arc::clone(&task_ctx.state),
                    notify: Arc::clone(&task_ctx.notify),
                };
                run_disk_lookup(task_ctx, key, spec, tile_cancel, guard, disk_slot).await;
            });
        }

        for task in http_tasks {
            let task_ctx = Arc::clone(&ctx);
            tokio::spawn(async move {
                let guard = InFlightGuard::new(
                    Arc::clone(&task_ctx.state),
                    task.key,
                    task.generation,
                    Arc::clone(&task_ctx.notify),
                );
                let permit_guard = PermitGuard {
                    permit: Some(task.permit),
                    notify: Arc::clone(&task_ctx.notify),
                };
                run_http_fetch(
                    task_ctx,
                    task.key,
                    task.spec,
                    task.url,
                    task.host,
                    task.cancel,
                    guard,
                    permit_guard,
                )
                .await;
            });
        }
    }
}

/// Submit `request` to the injected [`TileDecodeExecutor`]; if rejected under
/// backpressure while still active and uncancelled, schedule a short silent
/// backoff (`DECODE_BACKPRESSURE_BACKOFF`) routing back to `pending_disk`
/// (when `retry_disk` is `true`, avoiding unnecessary HTTP traffic for cached
/// tiles) or `pending_http` so rejected tiles neither vanish nor spin in a
/// tight retry loop while the decode pool drains.
fn submit_or_requeue(ctx: &PipelineContext, request: DecodeRequest, retry_disk: bool) {
    let key = request.key;
    let cancel = request.cancel.clone();
    if ctx.decoder.submit(request) {
        return;
    }
    if cancel.is_cancelled() || ctx.shutdown.is_cancelled() {
        return;
    }
    let mut should_notify = false;
    {
        let now = Instant::now();
        let mut st = lock_unpoisoned(&ctx.state);
        if st
            .active_spec
            .as_ref()
            .is_some_and(|s| s.source_id() == key.source)
            && st.last_wanted.contains(&key)
            && !st.in_flight.contains_key(&key)
            && !st.failures.contains_key(&key)
        {
            st.insert_failure(
                key,
                FailureState::BackoffUntil {
                    until: now + DECODE_BACKPRESSURE_BACKOFF,
                    retry_disk,
                },
                now,
            );
            should_notify = true;
        }
    }
    if should_notify {
        ctx.notify.notify_one();
    }
}

async fn run_disk_lookup(
    ctx: Arc<PipelineContext>,
    key: TileKey,
    spec: Arc<LiveBasemapSpec>,
    cancel: CancelToken,
    mut guard: InFlightGuard,
    disk_slot: DiskSlotGuard,
) {
    if cancel.is_cancelled() {
        return;
    }
    let Some(cache) = ctx.disk_cache.clone() else {
        promote_to_http(&ctx, key, &cancel, &mut guard);
        return;
    };
    let Some(id) = key.quad_id() else {
        return;
    };
    let source_key = spec.cache_key().to_owned();
    let cache_for_read = Arc::clone(&cache);
    let disk_res = tokio::select! {
        _ = cancel.cancelled() => return,
        res = tokio::task::spawn_blocking(move || cache_for_read.get(&source_key, id)) => res,
    };
    // Release the disk-lookup concurrency slot immediately before decode/HTTP.
    drop(disk_slot);

    if cancel.is_cancelled() {
        return;
    }

    match disk_res {
        Ok(Ok(Some(bytes))) => {
            let cb_ctx = Arc::clone(&ctx);
            let cb_cancel = cancel.clone();
            let cb_spec = Arc::clone(&spec);
            let req = DecodeRequest {
                key,
                bytes,
                cancel,
                on_decoded: Box::new(move |outcome| match outcome {
                    Ok(_) => {
                        let _guard = guard;
                        let mut st = lock_unpoisoned(&cb_ctx.state);
                        if st
                            .active_spec
                            .as_ref()
                            .is_some_and(|s| s.source_id() == key.source)
                        {
                            st.last_wanted.remove(&key);
                            st.completed_from_disk = st.completed_from_disk.saturating_add(1);
                        }
                    }
                    Err(err) => {
                        tracing::debug!(
                            ?key,
                            error = %err,
                            "Cached disk tile failed decode; deleting corrupt entry and falling back to HTTP"
                        );
                        // Keep `guard` alive inside the blocking delete task so
                        // `delete` strictly happens-before re-queueing into
                        // `pending_http` (preventing delete-vs-put and
                        // `reconcile_wanted` races).
                        let del_ctx = Arc::clone(&cb_ctx);
                        let del_cache = Arc::clone(&cache);
                        let del_source_key = cb_spec.cache_key().to_owned();
                        cb_ctx.rt_handle.spawn_blocking(move || {
                            let _ = del_cache.delete(&del_source_key, id);
                            let mut guard = guard;
                            promote_to_http(&del_ctx, key, &cb_cancel, &mut guard);
                        });
                    }
                }),
            };
            submit_or_requeue(&ctx, req, true);
        }
        _ => {
            // Disk miss (or transient read error): move to HTTP stage cleanly.
            promote_to_http(&ctx, key, &cancel, &mut guard);
        }
    }
}

fn promote_to_http(
    ctx: &PipelineContext,
    key: TileKey,
    cancel: &CancelToken,
    guard: &mut InFlightGuard,
) {
    let mut should_notify = false;
    {
        let mut st = lock_unpoisoned(&ctx.state);
        if st
            .in_flight
            .get(&key)
            .is_some_and(|(gen_id, _)| *gen_id == guard.generation)
        {
            st.in_flight.remove(&key);
            guard.disarm();
            should_notify = true;
            if !cancel.is_cancelled()
                && !ctx.shutdown.is_cancelled()
                && st
                    .active_spec
                    .as_ref()
                    .is_some_and(|s| s.source_id() == key.source)
                && !st.failures.contains_key(&key)
            {
                st.insert_pending_http(key);
            }
        }
    }
    if should_notify {
        ctx.notify.notify_one();
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "HTTP worker receives shared PipelineContext plus per-request key, spec, url, host, cancel, and RAII guards"
)]
async fn run_http_fetch(
    ctx: Arc<PipelineContext>,
    key: TileKey,
    spec: Arc<LiveBasemapSpec>,
    url: String,
    host: String,
    cancel: CancelToken,
    guard: InFlightGuard,
    permit_guard: PermitGuard,
) {
    if cancel.is_cancelled() {
        return;
    }

    let Some(id) = key.quad_id() else {
        return;
    };
    // Final defense-in-depth contract check: never touch PolarCap.
    if matches!(id.scheme, PlanetaryScheme::PolarCap { .. }) {
        return;
    }

    let fetch_res = tokio::select! {
        _ = cancel.cancelled() => return,
        res = download_capped(&ctx.client, &url, ctx.max_tile_bytes) => res,
    };

    // Release the per-host HTTP permit BEFORE submitting CPU decode / disk put.
    drop(permit_guard);

    if cancel.is_cancelled() {
        return;
    }

    match fetch_res {
        Ok(FetchOutcome::Payload(bytes)) => {
            let cb_ctx = Arc::clone(&ctx);
            let cb_cancel = cancel.clone();
            let cb_spec = Arc::clone(&spec);
            let req = DecodeRequest {
                key,
                bytes,
                cancel,
                on_decoded: Box::new(move |outcome| {
                    let _guard = guard;
                    match outcome {
                        Ok(valid_bytes) => {
                            // Update stats synchronously (nanoseconds, race-free
                            // even if UI `reconcile_wanted` already cancelled
                            // `cb_cancel` after receiving the tile), and persist
                            // validated bytes off the decode thread under a
                            // bounded write semaphore.
                            {
                                let mut st = lock_unpoisoned(&cb_ctx.state);
                                if st
                                    .active_spec
                                    .as_ref()
                                    .is_some_and(|s| s.source_id() == key.source)
                                {
                                    st.last_wanted.remove(&key);
                                    st.completed_from_net = st.completed_from_net.saturating_add(1);
                                    st.offline = false;
                                    st.last_error = None;
                                }
                            }
                            if let Some(cache) = cb_ctx.disk_cache.clone() {
                                let source_key = cb_spec.cache_key().to_owned();
                                let put_sem = Arc::clone(&cb_ctx.disk_put_sem);
                                let rt = cb_ctx.rt_handle.clone();
                                rt.spawn(async move {
                                    let Ok(_permit) = put_sem.acquire_owned().await else {
                                        return;
                                    };
                                    let _ = tokio::task::spawn_blocking(move || {
                                        if let Err(err) = cache.put(&source_key, id, &valid_bytes) {
                                            tracing::debug!(
                                                error = %err,
                                                "Failed to persist validated live tile to disk cache"
                                            );
                                        }
                                    })
                                    .await;
                                });
                            }
                        }
                        Err(err) => {
                            let safe_msg = cb_spec.redact_secrets(&err);
                            record_failure(
                                &cb_ctx,
                                &cb_cancel,
                                key,
                                FailureState::Terminal,
                                false,
                                Some(safe_msg),
                            );
                        }
                    }
                }),
            };
            submit_or_requeue(&ctx, req, false);
        }
        Ok(FetchOutcome::Empty204) => {
            // Valid empty/nodata tile: suppress infinite refetch without
            // bumping `failed_count` or displaying an error badge.
            let now = Instant::now();
            let mut st = lock_unpoisoned(&ctx.state);
            if !cancel.is_cancelled()
                && st
                    .active_spec
                    .as_ref()
                    .is_some_and(|s| s.source_id() == key.source)
            {
                st.last_wanted.remove(&key);
                st.insert_failure(key, FailureState::Empty, now);
                st.offline = false;
            }
        }
        Err(FetchFailure::Terminal(msg)) => {
            let safe_msg = spec.redact_secrets(&msg);
            tracing::debug!(?key, error = %safe_msg, "Terminal HTTP tile failure");
            record_failure(
                &ctx,
                &cancel,
                key,
                FailureState::Terminal,
                false,
                Some(safe_msg),
            );
        }
        Err(FetchFailure::Transient {
            msg,
            offline,
            retry_after,
        }) => {
            let safe_msg = spec.redact_secrets(&msg);
            let delay = retry_after.unwrap_or(ctx.transient_retry_backoff);
            let until = Instant::now() + delay;
            if retry_after.is_some() {
                ctx.host_limiters.record_cooldown(&host, until);
            }
            tracing::debug!(
                ?key,
                offline,
                delay_s = delay.as_secs(),
                error = %safe_msg,
                "Transient HTTP tile failure"
            );
            record_failure(
                &ctx,
                &cancel,
                key,
                FailureState::BackoffUntil {
                    until,
                    retry_disk: false,
                },
                offline,
                Some(safe_msg),
            );
        }
    }
}

fn record_failure(
    ctx: &PipelineContext,
    cancel: &CancelToken,
    key: TileKey,
    failure: FailureState,
    offline: bool,
    error_msg: Option<String>,
) {
    let now = Instant::now();
    let mut should_wake = false;
    {
        let mut st = lock_unpoisoned(&ctx.state);
        // Re-check cancellation and active source under lock so a late error
        // from a previous basemap never contaminates the newly active source.
        if !cancel.is_cancelled()
            && st
                .active_spec
                .as_ref()
                .is_some_and(|s| s.source_id() == key.source)
        {
            st.insert_failure(key, failure, now);
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
        (ctx.wakeup)();
    }
}

#[cfg(test)]
mod tests {
    use veldmap_core::tile::QuadTileId;

    use super::*;

    fn wm_key(spec: &LiveBasemapSpec, level: u8, x: u32, y: u32) -> TileKey {
        spec.tile_key(QuadTileId {
            scheme: PlanetaryScheme::WebMercator,
            level,
            x,
            y,
        })
    }

    #[test]
    fn stale_inflight_guard_never_removes_newer_generation() {
        let state = Arc::new(Mutex::new(PipelineState::default()));
        let notify = Arc::new(Notify::new());
        let spec = LiveBasemapSpec::eox_sentinel2_cloudless();
        let key = wm_key(&spec, 0, 0, 0);

        let tok1 = CancelToken::new();
        let tok2 = CancelToken::new();
        {
            let mut st = lock_unpoisoned(&state);
            st.in_flight.insert(key, (1, tok1));
        }
        let guard1 = InFlightGuard::new(Arc::clone(&state), key, 1, Arc::clone(&notify));
        {
            let mut st = lock_unpoisoned(&state);
            st.in_flight.insert(key, (2, tok2));
        }
        drop(guard1);

        let st = lock_unpoisoned(&state);
        assert_eq!(
            st.in_flight.get(&key).map(|(g, _)| *g),
            Some(2),
            "stale generation 1 guard must not remove generation 2 entry"
        );
    }

    #[test]
    fn pipeline_state_enforces_failure_cap_and_requeues_expired_backoffs() {
        let spec = Arc::new(LiveBasemapSpec::eox_sentinel2_cloudless());
        let mut st = PipelineState {
            active_spec: Some(Arc::clone(&spec)),
            ..PipelineState::default()
        };
        let now = Instant::now();
        let http_key = wm_key(&spec, 4, 1, 2);
        let disk_key = wm_key(&spec, 4, 2, 2);
        st.last_wanted.insert(http_key);
        st.last_wanted.insert(disk_key);
        st.insert_failure(
            http_key,
            FailureState::BackoffUntil {
                until: now - Duration::from_secs(1),
                retry_disk: false,
            },
            now,
        );
        st.insert_failure(
            disk_key,
            FailureState::BackoffUntil {
                until: now - Duration::from_secs(1),
                retry_disk: true,
            },
            now,
        );
        assert!(st.prune_expired_backoffs(now));
        assert!(st.pending_http.contains(&http_key));
        assert!(!st.pending_disk.contains(&http_key));
        assert!(st.pending_disk.contains(&disk_key));
        assert!(!st.pending_http.contains(&disk_key));

        // Filling beyond MAX_FAILURE_ENTRIES prunes non-wanted entries without exceeding cap.
        for i in 0..(MAX_FAILURE_ENTRIES as u32 + 16) {
            let k = wm_key(&spec, 12, i, 0);
            st.insert_failure(k, FailureState::Terminal, now);
        }
        assert!(st.failures.len() <= MAX_FAILURE_ENTRIES);
    }
}
