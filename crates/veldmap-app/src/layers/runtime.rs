//! Shared overlay runtime handles, scheduler reconciliation helpers, and
//! [`DecodeContext`] construction.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use veldmap_core::budget::{DecodeBudgetTracker, MemoryBudget};
use veldmap_core::sync::{CancelToken, Wakeup, lock_unpoisoned};
use veldmap_core::tile::{SourceId, StreamKey, TileKey};
use veldmap_net::NetContext;
use veldmap_raster::{
    AdmittedPass, AdmittedTile, DecodeContext, DecodeJob, DecodePoolHandle, PyramidMeta,
    SharedChunkCache, SubmitError, TileFailureOutcome, TileScheduler,
};
use veldmap_storage::{LocalFingerprintCache, TileDiskCache};

use crate::gpu_tiles::{GpuTileCache, PendingUploads, UploadSender};

use super::disk_cache::TileWriteHandle;

/// Backoff window applied when a decode lane queue is saturated (`SubmitError::Full`)
/// so the UI retries cleanly without spinning at 100% CPU.
const SUBMIT_BACKOFF_DELAY: Duration = Duration::from_millis(50);

/// Maximum concurrent asynchronous remote COG pointwise prefetch tasks spawned
/// onto [`NetContext`] so zooming into remote scenes applies bounded
/// backpressure instead of spawning unbounded Tokio / SQLite blocking tasks.
const MAX_REMOTE_PREFETCH_IN_FLIGHT: usize = 32;

/// RAII permit bounding concurrent remote COG pointwise prefetch tasks.
pub(super) struct RemotePrefetchPermit {
    runtime: Arc<OverlayRuntime>,
}

impl Drop for RemotePrefetchPermit {
    fn drop(&mut self) {
        self.runtime
            .remote_prefetch_in_flight
            .fetch_sub(1, Ordering::AcqRel);
        if self
            .runtime
            .remote_prefetch_saturated
            .swap(false, Ordering::AcqRel)
        {
            self.runtime.request_reschedule();
        }
    }
}

/// Shared runtime dependencies for overlay scheduling, decoding, caching, and
/// wakeups. Held behind `Arc` so worker closures and helpers stay concise.
pub(super) struct OverlayRuntime {
    pub(super) budget: MemoryBudget,
    pub(super) chunk_cache: Arc<SharedChunkCache>,
    pub(super) budget_tracker: Arc<DecodeBudgetTracker>,
    pub(super) fingerprint_cache: Arc<LocalFingerprintCache>,
    scheduler: Mutex<TileScheduler>,
    source_write_tokens: Mutex<HashMap<SourceId, (u64, CancelToken)>>,
    next_write_gen: AtomicU64,
    pub(super) disk_cache: Option<Arc<TileDiskCache>>,
    pub(super) tile_writer: Option<TileWriteHandle>,
    pub(super) net: Option<Arc<NetContext>>,
    pub(super) fast_pool: DecodePoolHandle,
    pub(super) bulk_pool: DecodePoolHandle,
    pub(super) remote_pool: DecodePoolHandle,
    pub(super) done_tx: UploadSender,
    pub(super) wakeup: Wakeup,
    pub(super) shutdown: CancelToken,
    reschedule_needed: AtomicBool,
    submit_backoff_until: Mutex<Option<Instant>>,
    remote_prefetch_in_flight: AtomicUsize,
    remote_prefetch_saturated: AtomicBool,
}

impl OverlayRuntime {
    #[expect(
        clippy::too_many_arguments,
        reason = "internal runtime wiring injects all explicit shared subsystems"
    )]
    pub(super) fn new(
        budget: &MemoryBudget,
        disk_cache: Option<Arc<TileDiskCache>>,
        tile_writer: Option<TileWriteHandle>,
        net: Option<Arc<NetContext>>,
        fast_pool: DecodePoolHandle,
        bulk_pool: DecodePoolHandle,
        remote_pool: DecodePoolHandle,
        done_tx: UploadSender,
        wakeup: Wakeup,
        shutdown: CancelToken,
    ) -> Self {
        Self {
            budget: *budget,
            chunk_cache: Arc::new(SharedChunkCache::new(budget.decoded_chunk_cache_bytes)),
            budget_tracker: Arc::new(DecodeBudgetTracker::new(budget.decode_working_bytes)),
            fingerprint_cache: Arc::new(LocalFingerprintCache::new()),
            scheduler: Mutex::new(TileScheduler::default()),
            source_write_tokens: Mutex::new(HashMap::new()),
            next_write_gen: AtomicU64::new(1),
            disk_cache,
            tile_writer,
            net,
            fast_pool,
            bulk_pool,
            remote_pool,
            done_tx,
            wakeup,
            shutdown,
            reschedule_needed: AtomicBool::new(false),
            submit_backoff_until: Mutex::new(None),
            remote_prefetch_in_flight: AtomicUsize::new(0),
            remote_prefetch_saturated: AtomicBool::new(false),
        }
    }

    /// Return the current source-scoped `(write_gen, cancel_token)` pair for
    /// `TileWriteWorker` commits, allocating a fresh child of `self.shutdown`
    /// on first use or after a source purge/reload.
    pub(super) fn write_token_for_source(&self, source_id: SourceId) -> (u64, CancelToken) {
        let mut map = lock_unpoisoned(&self.source_write_tokens);
        if let Some((write_gen, tok)) = map.get(&source_id) {
            return (*write_gen, tok.clone());
        }
        let write_gen = self.next_write_gen.fetch_add(1, Ordering::Relaxed);
        let tok = self.shutdown.child_token();
        map.insert(source_id, (write_gen, tok.clone()));
        (write_gen, tok)
    }

    /// Attempt to acquire a bounded remote pointwise prefetch slot.
    pub(super) fn try_acquire_remote_prefetch(self: &Arc<Self>) -> Option<RemotePrefetchPermit> {
        let mut cur = self.remote_prefetch_in_flight.load(Ordering::Acquire);
        loop {
            if cur >= MAX_REMOTE_PREFETCH_IN_FLIGHT {
                self.remote_prefetch_saturated
                    .store(true, Ordering::Release);
                return None;
            }
            match self.remote_prefetch_in_flight.compare_exchange_weak(
                cur,
                cur + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    return Some(RemotePrefetchPermit {
                        runtime: Arc::clone(self),
                    });
                }
                Err(actual) => cur = actual,
            }
        }
    }

    /// Build a [`DecodeContext`] borrowing this runtime's shared chunk cache,
    /// decode working-memory tracker, budget, and the caller's `cancel` token.
    pub(super) fn decode_context<'a>(&'a self, cancel: &'a CancelToken) -> DecodeContext<'a> {
        DecodeContext {
            chunk_cache: &self.chunk_cache,
            budget_tracker: &self.budget_tracker,
            budget: &self.budget,
            cancel,
        }
    }

    pub(super) fn request_reschedule(&self) {
        self.reschedule_needed.store(true, Ordering::Release);
        (self.wakeup)();
    }

    pub(super) fn note_submit_backoff(&self) {
        let now = Instant::now();
        let should_wake = {
            let mut until = lock_unpoisoned(&self.submit_backoff_until);
            let was_idle = until.is_none_or(|prev| prev <= now);
            let next = now + SUBMIT_BACKOFF_DELAY;
            *until = Some(until.map_or(next, |prev| prev.min(next)));
            was_idle
        };
        // Wake an idle UI loop once per backoff window so background pool
        // rejections (e.g. remote pointwise handoff to `fast_pool`) schedule a
        // timed repaint via `retry_delay()`.
        if should_wake {
            (self.wakeup)();
        }
    }

    /// Submit `job` to `pool`, guaranteeing that on rejection `on_reject` runs
    /// strictly **before** the rejected [`DecodeJob`] (and any RAII flight
    /// guard inside its closure) is dropped so `Drop` cannot trigger a 100% CPU
    /// repaint spin loop.
    ///
    /// When `backoff_on_full` is `true` and `pool` rejects with
    /// [`SubmitError::Full`], records a short retry backoff window.
    pub(super) fn try_submit_guarded(
        &self,
        pool: &DecodePoolHandle,
        job: DecodeJob,
        backoff_on_full: bool,
        on_reject: impl FnOnce(),
    ) -> bool {
        match pool.try_submit(job) {
            Ok(()) => true,
            Err(err) => {
                let is_full = matches!(err, SubmitError::Full(_));
                on_reject();
                drop(err);
                if backoff_on_full && is_full {
                    self.note_submit_backoff();
                }
                false
            }
        }
    }

    /// Return `true` if a background overlay event requires re-running
    /// scheduling (e.g. queue eviction, transient/terminal failure, pass
    /// completion, or an elapsed queue-full backoff window).
    pub(super) fn take_reschedule_needed(&self) -> bool {
        let mut needed = self.reschedule_needed.swap(false, Ordering::AcqRel);
        let mut backoff = lock_unpoisoned(&self.submit_backoff_until);
        if let Some(deadline) = *backoff
            && Instant::now() >= deadline
        {
            *backoff = None;
            needed = true;
        }
        needed
    }

    /// Remaining backoff delay if a decode queue was recently full, or `None`
    /// when idle (preserving 0 idle re-renders).
    pub(super) fn retry_delay(&self) -> Option<Duration> {
        let deadline = (*lock_unpoisoned(&self.submit_backoff_until))?;
        Some(deadline.saturating_duration_since(Instant::now()))
    }

    /// Snapshot `(in_flight_count, pass_active)` for each active [`StreamKey`]
    /// under a single scheduler lock acquisition.
    pub(super) fn layer_flight_stats(
        &self,
        streams: impl IntoIterator<Item = StreamKey>,
    ) -> Vec<(usize, bool)> {
        let sched = lock_unpoisoned(&self.scheduler);
        streams
            .into_iter()
            .map(|stream_key| {
                (
                    sched.in_flight_count_for_stream(stream_key),
                    sched.has_active_pass(stream_key),
                )
            })
            .collect()
    }

    /// Admit pointwise decode jobs for `rung` and report whether any candidate
    /// on this rung is still unfinished (neither cached in VRAM nor terminally
    /// failed).
    pub(super) fn admit_pointwise_rung(
        &self,
        meta: &PyramidMeta,
        rung: u8,
        candidates: &[TileKey],
        gpu_cache: &GpuTileCache,
        pending: &PendingUploads,
    ) -> (Vec<AdmittedTile>, bool) {
        let mut sched = lock_unpoisoned(&self.scheduler);
        let admitted = sched.admit_pointwise(
            meta,
            rung,
            candidates,
            |key| gpu_cache.contains(key) || pending.contains(key),
            &self.shutdown,
        );
        let unfinished = candidates
            .iter()
            .any(|k| !gpu_cache.contains(k) && !sched.is_terminal_failed(k));
        (admitted, unfinished)
    }

    /// Update the active pass target for `meta.stream_key()`, admit a sequential
    /// pass for `rung` if needed, and cancel any superseded pass token strictly
    /// outside the scheduler lock.
    pub(super) fn admit_pass_rung(
        &self,
        meta: &PyramidMeta,
        rung: u8,
        from: u8,
        candidates: &[TileKey],
        gpu_cache: &GpuTileCache,
        pending: &PendingUploads,
    ) -> (Option<AdmittedPass>, bool) {
        let stream_key = meta.stream_key();
        let (cancelled_old, admitted, active) = {
            let mut sched = lock_unpoisoned(&self.scheduler);
            let cancelled_old = sched.update_target(stream_key, rung, Some(from));
            let admitted = sched.admit_pass(
                meta,
                rung,
                candidates,
                |key| gpu_cache.contains(key) || pending.contains(key),
                &self.shutdown,
            );
            let active = admitted.is_some() || sched.has_active_pass(stream_key);
            (cancelled_old, admitted, active)
        };
        if let Some(tok) = cancelled_old {
            tok.cancel();
        }
        (admitted, active)
    }

    pub(super) fn on_pointwise_arrived(&self, key: &TileKey, job_id: u64) -> bool {
        lock_unpoisoned(&self.scheduler).on_pointwise_arrived(key, job_id)
    }

    pub(super) fn on_pointwise_failed(
        &self,
        key: &TileKey,
        job_id: u64,
        terminal: bool,
    ) -> Option<TileFailureOutcome> {
        lock_unpoisoned(&self.scheduler).on_pointwise_failed(key, job_id, terminal)
    }

    pub(super) fn cancel_pointwise_in_flight(&self, key: &TileKey, job_id: u64) -> bool {
        lock_unpoisoned(&self.scheduler).cancel_in_flight(key, job_id)
    }

    pub(super) fn on_pass_tile_arrived(&self, stream_key: StreamKey, pass_id: u64, key: &TileKey) {
        lock_unpoisoned(&self.scheduler).on_pass_tile_arrived(stream_key, pass_id, key);
    }

    pub(super) fn finish_pass(&self, stream_key: StreamKey, pass_id: u64, succeeded: bool) -> bool {
        lock_unpoisoned(&self.scheduler).finish_pass(stream_key, pass_id, succeeded)
    }

    /// Abort an active sequential pass in the scheduler and cancel its token
    /// strictly outside the scheduler lock. Returns `true` if the pass was
    /// still active when aborted.
    pub(super) fn abort_pass(&self, stream_key: StreamKey, pass_id: u64) -> bool {
        let token = lock_unpoisoned(&self.scheduler).abort_pass(stream_key, pass_id);
        if let Some(tok) = token {
            tok.cancel();
            true
        } else {
            false
        }
    }

    /// Reconcile in-flight pointwise jobs across `streams` under a single
    /// scheduler lock acquisition, cancelling stale tokens strictly outside
    /// the lock.
    pub(super) fn reconcile_pointwise_streams<'a>(
        &self,
        streams: impl IntoIterator<Item = (StreamKey, &'a HashSet<TileKey>)>,
        gpu_cache: &GpuTileCache,
        pending: &PendingUploads,
    ) {
        let mut to_cancel = Vec::new();
        {
            let mut sched = lock_unpoisoned(&self.scheduler);
            for (stream_key, wanted) in streams {
                to_cancel.extend(sched.reconcile_pointwise(stream_key, wanted, |k| {
                    gpu_cache.contains(k) || pending.contains(k)
                }));
            }
        }
        for tok in to_cancel {
            tok.cancel();
        }
    }

    /// Cancel all pointwise and pass jobs for a single [`StreamKey`] under one
    /// scheduler lock acquisition, cancelling tokens strictly outside the lock.
    pub(super) fn cancel_stream(&self, stream_key: StreamKey) {
        let tokens = lock_unpoisoned(&self.scheduler).reset_stream(stream_key);
        for tok in tokens {
            tok.cancel();
        }
    }

    /// Cancel all pointwise, pass, and deferred disk-write jobs for every source
    /// in `sources`, cancelling tokens strictly outside internal locks.
    pub(super) fn cancel_sources(&self, sources: impl IntoIterator<Item = SourceId>) {
        let sources: Vec<SourceId> = sources.into_iter().collect();
        let mut tokens = Vec::new();
        {
            let mut sched = lock_unpoisoned(&self.scheduler);
            for &source_id in &sources {
                tokens.extend(sched.reset_source(source_id));
            }
        }
        {
            let mut write_tokens = lock_unpoisoned(&self.source_write_tokens);
            for &source_id in &sources {
                if let Some((_, tok)) = write_tokens.remove(&source_id) {
                    tokens.push(tok);
                }
            }
        }
        if let Some(writer) = &self.tile_writer {
            for &source_id in &sources {
                writer.purge_source(source_id);
            }
        }
        for tok in tokens {
            tok.cancel();
        }
    }

    /// Cancel all pointwise and pass jobs for `source_id` (cancelling tokens
    /// strictly outside the scheduler lock) and invalidate decoded chunks.
    pub(super) fn purge_source_runtime(&self, source_id: SourceId) {
        self.cancel_sources([source_id]);
        self.chunk_cache.invalidate_source(source_id);
    }
}
