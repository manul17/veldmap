//! Asynchronous HTTP tile fetcher with two-stage disk/HTTP scheduling,
//! per-host concurrency & rate-limit cooldowns, replaceable visible-first
//! queues, end-to-end `CancelToken` abort through CPU decode, and deterministic
//! shutdown (`CancelToken` -> join decode pool -> drop `tokio::Runtime`).

use std::collections::{HashMap, HashSet};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::sync::{Notify, OwnedSemaphorePermit};
use veldmap_core::io::{CancelToken, lock_unpoisoned};
use veldmap_core::tile::{DecodedTile, TileKey, TileScheme};

use crate::basemap::LiveBasemapSpec;
use crate::decode_pool::{DecodeJob, DecodePool, DecodePoolHandle, Wakeup};
use crate::http::{
    DEFAULT_MAX_CONCURRENT_PER_HOST, DEFAULT_MAX_TILE_RESPONSE_BYTES, FetchFailure, FetchOutcome,
    HostLimiters, build_http_client, default_user_agent, download_capped, extract_host,
};
use crate::{NetError, TileBlobStore};

/// Default backoff duration after transient HTTP / network errors before
/// retrying a tile.
const DEFAULT_TRANSIENT_RETRY_BACKOFF: Duration = Duration::from_secs(15);

/// Maximum concurrent blocking disk-cache lookup tasks so cached tiles stay
/// responsive and never starve behind slow HTTP hosts.
const MAX_CONCURRENT_DISK_LOOKUPS: usize = 4;

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
    pub decode_threads: usize,
    pub transient_retry_backoff: Duration,
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
    BackoffUntil(Instant),
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
    /// through `DecodePool` completion so stale guards never clobber a newer
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
                FailureState::BackoffUntil(until) => *until > now,
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
    /// are still in `last_wanted` into `pending_http`. Returns whether new work
    /// was queued.
    fn prune_expired_backoffs(&mut self, now: Instant) -> bool {
        let Some(spec) = self.active_spec.as_ref() else {
            self.failures.clear();
            return false;
        };
        let active_source = spec.source_id;
        let mut requeued = false;
        let mut expired = Vec::new();
        self.failures.retain(|&key, f| match f {
            FailureState::BackoffUntil(until) if *until <= now => {
                expired.push(key);
                false
            }
            _ => true,
        });
        for key in expired {
            if key.source == active_source
                && self.last_wanted.contains(&key)
                && !self.in_flight.contains_key(&key)
                && !self.pending_disk.contains(&key)
                && !self.pending_http.contains(&key)
            {
                self.pending_http.push(key);
                requeued = true;
            }
        }
        if requeued {
            Self::sort_queue_coarsest_at_end(&mut self.pending_http);
        }
        requeued
    }

    /// Earliest `BackoffUntil` timestamp among currently wanted tiles (used by
    /// `dispatcher_loop` to self-wake for idle retries without busy-spinning).
    fn earliest_wanted_backoff(&self, now: Instant) -> Option<Instant> {
        self.failures
            .iter()
            .filter_map(|(k, f)| match f {
                FailureState::BackoffUntil(until)
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
    decode_handle: DecodePoolHandle,
    wakeup: Wakeup,
    rt_handle: tokio::runtime::Handle,
    max_tile_bytes: usize,
    transient_retry_backoff: Duration,
}

/// Generation-checked RAII guard keeping `key` registered in `state.in_flight`
/// across async fetch, disk lookup/delete, and CPU decode in `DecodePool`.
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
/// 2. Bounded `std::thread` [`DecodePool`]
/// 3. Dedicated `tokio::runtime::Runtime` running HTTP fetches
///
/// Shuts down deterministically via [`Self::signal_shutdown`] / `Drop`:
/// `CancelToken` -> close/join decode pool -> shutdown `tokio::Runtime`.
pub struct LiveTilePipeline {
    shutdown: CancelToken,
    state: Arc<Mutex<PipelineState>>,
    host_limiters: Arc<HostLimiters>,
    notify: Arc<Notify>,
    decode_pool: Option<DecodePool>,
    runtime: Option<tokio::runtime::Runtime>,
}

impl LiveTilePipeline {
    /// Spawn a live tile pipeline pushing decoded, mipmapped tiles to `done_tx`
    /// and invoking `wakeup` on tile arrival or status changes.
    pub fn spawn(
        config: FetcherConfig,
        disk_cache: Option<Arc<dyn TileBlobStore>>,
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
            decode_handle,
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
    /// filters out `PolarCap` keys and any `level > spec.max_level`.
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
            let active_source = spec.source_id;

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
                FailureState::BackoffUntil(until) => *until > now,
            });

            // Track strictly unready, active-source, contract-valid wanted keys.
            st.last_wanted.clear();
            for &key in wanted {
                if key.source == active_source
                    && !is_ready(&key)
                    && spec.validate_tile(key.quad_id()).is_ok()
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
                let active_source = spec.source_id;
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

    /// Return current number of in-flight + queued tiles for appetite
    /// budgeting.
    pub fn active_work_count(&self) -> usize {
        let st = lock_unpoisoned(&self.state);
        st.in_flight
            .len()
            .saturating_add(st.pending_disk.len())
            .saturating_add(st.pending_http.len())
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
    /// deterministic shutdown, called BEFORE dropping the completion rx).
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
                // Stage 1: Pop from `pending_disk` (or move straight to `pending_http`
                // when no disk cache is configured).
                if ctx.disk_cache.is_none() {
                    let drained: Vec<TileKey> = st.pending_disk.drain(..).collect();
                    for key in drained {
                        if key.source == spec.source_id
                            && !st.in_flight.contains_key(&key)
                            && !st.failures.contains_key(&key)
                        {
                            st.insert_pending_http(key);
                        }
                    }
                } else {
                    while st.disk_in_flight < MAX_CONCURRENT_DISK_LOOKUPS {
                        let Some(key) = st.pending_disk.pop() else {
                            break;
                        };
                        if key.source != spec.source_id
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
                // This keeps unstarted HTTP tiles sorted coarsest-first instead
                // of parked FIFO on a semaphore, and never starves Stage 1.
                let mut deferred_http = Vec::new();
                while let Some(key) = st.pending_http.pop() {
                    if key.source != spec.source_id
                        || st.in_flight.contains_key(&key)
                        || st.failures.contains_key(&key)
                    {
                        continue;
                    }
                    let id = key.quad_id();
                    if matches!(id.scheme, TileScheme::PolarCap { .. }) {
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
                            // If single-host (typical) and saturated, stop scanning
                            // only after checking other potential `{s}` subdomains.
                        }
                    }
                }
                if !deferred_http.is_empty() {
                    st.pending_http = deferred_http;
                    PipelineState::sort_queue_coarsest_at_end(&mut st.pending_http);
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
    let id = key.quad_id();
    let source_key = spec.cache_key.clone();
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
            let job = DecodeJob {
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
                            .is_some_and(|s| s.source_id == key.source)
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
                        let del_source_key = cb_spec.cache_key.clone();
                        cb_ctx.rt_handle.spawn_blocking(move || {
                            let _ = del_cache.delete(&del_source_key, id);
                            let mut guard = guard;
                            promote_to_http(&del_ctx, key, &cb_cancel, &mut guard);
                        });
                    }
                }),
            };
            cb_ctx_submit(&ctx, job);
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
                    .is_some_and(|s| s.source_id == key.source)
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

fn cb_ctx_submit(ctx: &PipelineContext, job: DecodeJob) {
    let _ = ctx.decode_handle.submit(job);
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

    let id = key.quad_id();
    // Final defense-in-depth contract check: never touch PolarCap.
    if matches!(id.scheme, TileScheme::PolarCap { .. }) {
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
            let job = DecodeJob {
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
                            // validated bytes off the decode thread.
                            {
                                let mut st = lock_unpoisoned(&cb_ctx.state);
                                if st
                                    .active_spec
                                    .as_ref()
                                    .is_some_and(|s| s.source_id == key.source)
                                {
                                    st.last_wanted.remove(&key);
                                    st.completed_from_net = st.completed_from_net.saturating_add(1);
                                    st.offline = false;
                                    st.last_error = None;
                                }
                            }
                            if let Some(cache) = cb_ctx.disk_cache.clone() {
                                let source_key = cb_spec.cache_key.clone();
                                cb_ctx.rt_handle.spawn_blocking(move || {
                                    if let Err(err) = cache.put(&source_key, id, &valid_bytes) {
                                        tracing::debug!(
                                            error = %err,
                                            "Failed to persist validated live tile to disk cache"
                                        );
                                    }
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
            cb_ctx_submit(&ctx, job);
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
                    .is_some_and(|s| s.source_id == key.source)
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
                FailureState::BackoffUntil(until),
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
                .is_some_and(|s| s.source_id == key.source)
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
    use std::io::Cursor;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc;

    use image::{ImageBuffer, ImageFormat, Rgba};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    use veldmap_core::tile::QuadTileId;

    use super::*;
    use crate::CustomBasemapConfig;

    type BlobMapKey = (String, TileScheme, u8, u32, u32);

    #[derive(Default)]
    struct MemBlobStore {
        map: Mutex<HashMap<BlobMapKey, Vec<u8>>>,
    }

    impl TileBlobStore for MemBlobStore {
        fn get(&self, source_key: &str, id: QuadTileId) -> Result<Option<Vec<u8>>, String> {
            let map = lock_unpoisoned(&self.map);
            Ok(map
                .get(&(source_key.to_owned(), id.scheme, id.level, id.x, id.y))
                .cloned())
        }

        fn put(&self, source_key: &str, id: QuadTileId, payload: &[u8]) -> Result<(), String> {
            let mut map = lock_unpoisoned(&self.map);
            map.insert(
                (source_key.to_owned(), id.scheme, id.level, id.x, id.y),
                payload.to_vec(),
            );
            Ok(())
        }

        fn delete(&self, source_key: &str, id: QuadTileId) -> Result<(), String> {
            let mut map = lock_unpoisoned(&self.map);
            map.remove(&(source_key.to_owned(), id.scheme, id.level, id.x, id.y));
            Ok(())
        }
    }

    fn make_png_bytes(color: [u8; 4]) -> Vec<u8> {
        let img: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::from_pixel(8, 8, Rgba(color));
        let mut out = Vec::new();
        img.write_to(&mut Cursor::new(&mut out), ImageFormat::Png)
            .expect("encode png");
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
        fn ok_png(body: Vec<u8>, delay: Duration) -> Self {
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
        let png = make_png_bytes([30, 180, 90, 255]);
        let png_c = png.clone();
        let server = spawn_mock_server(move |_, _| {
            MockResponse::ok_png(png_c.clone(), Duration::from_millis(25))
        });

        let disk_cache: Arc<dyn TileBlobStore> = Arc::new(MemBlobStore::default());
        let (done_tx, done_rx) = mpsc::channel();
        let wakes = Arc::new(AtomicUsize::new(0));
        let wakes_c = Arc::clone(&wakes);
        let wakeup: Wakeup = Arc::new(move || {
            wakes_c.fetch_add(1, Ordering::AcqRel);
        });

        let cfg = FetcherConfig {
            max_concurrent_per_host: 2,
            decode_threads: 2,
            ..FetcherConfig::default()
        };
        let pipeline = LiveTilePipeline::spawn(cfg, Some(Arc::clone(&disk_cache)), done_tx, wakeup)
            .expect("spawn pipeline");

        let spec = custom_spec(server.addr, 2);
        pipeline.set_active_spec(Some(spec.clone()));

        let valid_keys: Vec<TileKey> = (0..4u32)
            .map(|x| {
                spec.tile_key(QuadTileId {
                    scheme: TileScheme::WebMercator,
                    level: 2,
                    x,
                    y: 0,
                })
            })
            .collect();
        let forbidden_polar = spec.tile_key(QuadTileId {
            scheme: TileScheme::PolarCap { north: true },
            level: 0,
            x: 0,
            y: 0,
        });
        let forbidden_above_max = spec.tile_key(QuadTileId {
            scheme: TileScheme::WebMercator,
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
            assert!(tile.validate());
            assert_eq!(tile.mip_level_count(), 4); // 8 -> 4 -> 2 -> 1
        }
        assert_eq!(server.hits.load(Ordering::Acquire), 4);
        assert!(
            server.max_concurrent.load(Ordering::Acquire) <= 2,
            "exceeded per-host limit: {}",
            server.max_concurrent.load(Ordering::Acquire)
        );
        // Wait for worker post-send callbacks (`on_decoded` + `wake` + async disk put)
        // to complete cleanly before checking wakeup & disk cache counts.
        let settle_deadline = Instant::now() + Duration::from_secs(2);
        while Instant::now() < settle_deadline {
            let all_cached = valid_keys.iter().all(|k| {
                disk_cache
                    .get(&spec.cache_key, k.quad_id())
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
        assert!(pipeline.status().completed_from_disk >= 1);

        pipeline.shutdown();
        server.shutdown();
    }

    #[test]
    fn corrupt_disk_blob_self_heals_from_http_and_cached_tiles_do_not_starve_behind_slow_http() {
        let png = make_png_bytes([20, 120, 240, 255]);
        let png_c = png.clone();
        let server = spawn_mock_server(move |path, _| {
            if path.contains("/2/1/0") {
                // Slow HTTP tile occupying the single HTTP slot.
                MockResponse::ok_png(png_c.clone(), Duration::from_secs(5))
            } else {
                // Fast valid tile for self-healing corrupt disk blob.
                MockResponse::ok_png(png_c.clone(), Duration::ZERO)
            }
        });

        let store = Arc::new(MemBlobStore::default());
        let spec = custom_spec(server.addr, 2);

        let k_corrupt = spec.tile_key(QuadTileId {
            scheme: TileScheme::WebMercator,
            level: 2,
            x: 0,
            y: 0,
        });
        let k_slow = spec.tile_key(QuadTileId {
            scheme: TileScheme::WebMercator,
            level: 2,
            x: 1,
            y: 0,
        });
        let k_cached = spec.tile_key(QuadTileId {
            scheme: TileScheme::WebMercator,
            level: 2,
            x: 2,
            y: 0,
        });

        // Pre-seed one corrupt blob and one valid cached blob.
        store
            .put(&spec.cache_key, k_corrupt.quad_id(), b"not-a-valid-png")
            .expect("seed corrupt");
        store
            .put(&spec.cache_key, k_cached.quad_id(), &png)
            .expect("seed valid");

        let (done_tx, done_rx) = mpsc::channel();
        let cfg = FetcherConfig {
            max_concurrent_per_host: 1,
            decode_threads: 2,
            ..FetcherConfig::default()
        };
        let pipeline = LiveTilePipeline::spawn(
            cfg,
            Some(Arc::clone(&store) as Arc<dyn TileBlobStore>),
            done_tx,
            Arc::new(|| {}),
        )
        .expect("spawn pipeline");
        pipeline.set_active_spec(Some(spec));

        // 1. Corrupt disk blob deletes itself and heals via HTTP!
        pipeline.reconcile_wanted(&HashSet::from([k_corrupt]), |_| false);
        let (healed_key, _) = done_rx
            .recv_timeout(Duration::from_secs(3))
            .expect("corrupt disk tile should self-heal from HTTP");
        assert_eq!(healed_key, k_corrupt);

        // 2. Saturate the 1-slot HTTP lane with `k_slow`, then request `k_cached`:
        //    `k_cached` MUST finish promptly from Stage-1 disk lookup without
        //    waiting for `k_slow`!
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
        let png = make_png_bytes([90, 210, 40, 255]);
        let png_c = png.clone();
        let flaky_hits = Arc::new(AtomicUsize::new(0));
        let flaky_c = Arc::clone(&flaky_hits);
        let server = spawn_mock_server(move |path, _| {
            if path.contains("/1/0/0") {
                // HTTP 204 No Content (valid empty tile).
                MockResponse {
                    status: 204,
                    headers: String::new(),
                    body: Vec::new(),
                    delay: Duration::ZERO,
                }
            } else if path.contains("/1/1/0") {
                // Corrupt HTTP 200 (must not cache, marked terminal).
                MockResponse::ok_png(b"<html>captive portal</html>".to_vec(), Duration::ZERO)
            } else if path.contains("/1/0/1") {
                // First attempt returns 500 (transient), second attempt returns PNG.
                let n = flaky_c.fetch_add(1, Ordering::AcqRel);
                if n == 0 {
                    MockResponse {
                        status: 500,
                        headers: String::new(),
                        body: Vec::new(),
                        delay: Duration::ZERO,
                    }
                } else {
                    MockResponse::ok_png(png_c.clone(), Duration::ZERO)
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
        let cfg = FetcherConfig {
            max_concurrent_per_host: 4,
            decode_threads: 2,
            transient_retry_backoff: Duration::from_millis(80),
            ..FetcherConfig::default()
        };
        let pipeline = LiveTilePipeline::spawn(
            cfg,
            Some(Arc::clone(&store) as Arc<dyn TileBlobStore>),
            done_tx,
            Arc::new(|| {}),
        )
        .expect("spawn pipeline");
        let spec = custom_spec(server.addr, 2);
        pipeline.set_active_spec(Some(spec.clone()));

        let k_204 = spec.tile_key(QuadTileId {
            scheme: TileScheme::WebMercator,
            level: 1,
            x: 0,
            y: 0,
        });
        let k_corrupt = spec.tile_key(QuadTileId {
            scheme: TileScheme::WebMercator,
            level: 1,
            x: 1,
            y: 0,
        });
        let k_flaky = spec.tile_key(QuadTileId {
            scheme: TileScheme::WebMercator,
            level: 1,
            x: 0,
            y: 1,
        });

        // 1. HTTP 204 suppresses refetch without incrementing failed_count or last_error.
        pipeline.reconcile_wanted(&HashSet::from([k_204]), |_| false);
        let d1 = Instant::now() + Duration::from_secs(2);
        while (pipeline.active_work_count() > 0 || server.hits.load(Ordering::Acquire) < 1)
            && Instant::now() < d1
        {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(pipeline.status().failed_count, 0);
        assert!(pipeline.status().last_error.is_none());
        // Re-reconciling must NOT refetch 204.
        let hits_after_204 = server.hits.load(Ordering::Acquire);
        pipeline.reconcile_wanted(&HashSet::from([k_204]), |_| false);
        std::thread::sleep(Duration::from_millis(40));
        assert_eq!(server.hits.load(Ordering::Acquire), hits_after_204);

        // 2. Corrupt HTTP 200 is not cached and marked terminal.
        pipeline.reconcile_wanted(&HashSet::from([k_corrupt]), |_| false);
        let d2 = Instant::now() + Duration::from_secs(2);
        while pipeline.status().failed_count == 0 && Instant::now() < d2 {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(pipeline.status().failed_count, 1);
        assert!(
            store
                .get(&spec.cache_key, k_corrupt.quad_id())
                .expect("get")
                .is_none()
        );

        // 3. Transient 500 automatically retries after backoff WITHOUT another reconcile!
        pipeline.reconcile_wanted(&HashSet::from([k_flaky]), |_| false);
        let (retried_key, _) = done_rx
            .recv_timeout(Duration::from_secs(3))
            .expect("flaky tile must succeed on automatic idle backoff retry");
        assert_eq!(retried_key, k_flaky);
        assert_eq!(flaky_hits.load(Ordering::Acquire), 2);

        // 4. `retry_failed` resets `failed_count` and `last_error`.
        pipeline.retry_failed();
        assert_eq!(pipeline.status().failed_count, 0);
        assert!(pipeline.status().last_error.is_none());

        pipeline.shutdown();
        server.shutdown();
    }

    #[test]
    fn cancel_token_aborts_slow_inflight_request_promptly_on_shutdown() {
        let png = make_png_bytes([10, 20, 30, 255]);
        let server = spawn_mock_server(move |_, _| {
            MockResponse::ok_png(png.clone(), Duration::from_secs(10))
        });

        let (done_tx, _done_rx) = mpsc::channel();
        let pipeline =
            LiveTilePipeline::spawn(FetcherConfig::default(), None, done_tx, Arc::new(|| {}))
                .expect("spawn pipeline");
        let spec = custom_spec(server.addr, 1);
        pipeline.set_active_spec(Some(spec.clone()));

        let k0 = spec.tile_key(QuadTileId {
            scheme: TileScheme::WebMercator,
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

    #[test]
    fn stale_inflight_guard_never_removes_newer_generation() {
        let state = Arc::new(Mutex::new(PipelineState::default()));
        let notify = Arc::new(Notify::new());
        let key = LiveBasemapSpec::eox_sentinel2_cloudless().tile_key(QuadTileId {
            scheme: TileScheme::WebMercator,
            level: 0,
            x: 0,
            y: 0,
        });

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
}
