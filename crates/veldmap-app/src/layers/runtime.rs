//! Shared overlay runtime handles, scheduler reconciliation helpers, and
//! [`DecodeContext`] construction.

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use veldmap_core::budget::{DecodeBudgetTracker, MemoryBudget};
use veldmap_core::sync::{CancelToken, Wakeup, lock_unpoisoned};
use veldmap_core::tile::{SourceId, StreamKey, TileKey};
use veldmap_raster::{
    AdmittedPass, AdmittedTile, DecodeContext, DecodeJob, DecodePoolHandle, PyramidMeta,
    SharedChunkCache, SubmitError, TileFailureOutcome, TileScheduler,
};
use veldmap_storage::{LocalFingerprintCache, TileDiskCache};

use crate::gpu_tiles::{GpuTileCache, PendingUploads, UploadSender};

/// Backoff window applied when a decode lane queue is saturated (`SubmitError::Full`)
/// so the UI retries cleanly without spinning at 100% CPU.
const SUBMIT_BACKOFF_DELAY: Duration = Duration::from_millis(50);

/// Shared runtime dependencies for overlay scheduling, decoding, caching, and
/// wakeups. Held behind `Arc` so worker closures and helpers stay concise.
pub(super) struct OverlayRuntime {
    pub(super) budget: MemoryBudget,
    pub(super) chunk_cache: Arc<SharedChunkCache>,
    pub(super) budget_tracker: Arc<DecodeBudgetTracker>,
    pub(super) fingerprint_cache: Arc<LocalFingerprintCache>,
    scheduler: Mutex<TileScheduler>,
    pub(super) disk_cache: Option<Arc<TileDiskCache>>,
    pub(super) fast_pool: DecodePoolHandle,
    pub(super) bulk_pool: DecodePoolHandle,
    pub(super) done_tx: UploadSender,
    pub(super) wakeup: Wakeup,
    pub(super) shutdown: CancelToken,
    reschedule_needed: AtomicBool,
    submit_backoff_until: Mutex<Option<Instant>>,
}

impl OverlayRuntime {
    pub(super) fn new(
        budget: &MemoryBudget,
        disk_cache: Option<Arc<TileDiskCache>>,
        fast_pool: DecodePoolHandle,
        bulk_pool: DecodePoolHandle,
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
            disk_cache,
            fast_pool,
            bulk_pool,
            done_tx,
            wakeup,
            shutdown,
            reschedule_needed: AtomicBool::new(false),
            submit_backoff_until: Mutex::new(None),
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

    fn note_submit_backoff(&self) {
        let mut until = lock_unpoisoned(&self.submit_backoff_until);
        let next = Instant::now() + SUBMIT_BACKOFF_DELAY;
        *until = Some(until.map_or(next, |prev| prev.min(next)));
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

    /// Snapshot `(in_flight_count, pass_active)` for each `(source_id, stream_key)`
    /// pair under a single scheduler lock acquisition.
    pub(super) fn layer_flight_stats(
        &self,
        streams: impl IntoIterator<Item = (SourceId, StreamKey)>,
    ) -> Vec<(usize, bool)> {
        let sched = lock_unpoisoned(&self.scheduler);
        streams
            .into_iter()
            .map(|(source_id, stream_key)| {
                (
                    sched.in_flight_count_for_source(source_id),
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

    /// Cancel all pointwise and pass jobs for every source in `sources` under a
    /// single scheduler lock acquisition, cancelling tokens strictly outside
    /// the lock.
    pub(super) fn cancel_sources(&self, sources: impl IntoIterator<Item = SourceId>) {
        let mut tokens = Vec::new();
        {
            let mut sched = lock_unpoisoned(&self.scheduler);
            for source_id in sources {
                tokens.extend(sched.reset_source(source_id));
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
