//! Shared overlay runtime handles, scheduler reconciliation helpers, and
//! [`DecodeContext`] construction.

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use veldmap_core::budget::{DecodeBudgetTracker, MemoryBudget};
use veldmap_core::sync::{CancelToken, Wakeup, lock_unpoisoned};
use veldmap_core::tile::{SourceId, TileKey};
use veldmap_raster::{DecodeContext, DecodePoolHandle, SharedChunkCache, TileScheduler};
use veldmap_storage::{LocalFingerprintCache, TileDiskCache};

use crate::gpu_tiles::{GpuTileCache, PendingUploads, UploadSender};

use super::stack::OverlayLayer;

/// Backoff window applied when a decode lane queue is saturated (`SubmitError::Full`)
/// so the UI retries cleanly without spinning at 100% CPU.
pub(super) const SUBMIT_BACKOFF_DELAY: Duration = Duration::from_millis(50);

/// Shared runtime dependencies for overlay scheduling, decoding, caching, and
/// wakeups. Held behind `Arc` so worker closures and helpers stay concise.
pub(super) struct OverlayRuntime {
    pub(super) budget: MemoryBudget,
    pub(super) chunk_cache: Arc<SharedChunkCache>,
    pub(super) budget_tracker: Arc<DecodeBudgetTracker>,
    pub(super) fingerprint_cache: Arc<LocalFingerprintCache>,
    pub(super) scheduler: Arc<Mutex<TileScheduler>>,
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
            scheduler: Arc::new(Mutex::new(TileScheduler::default())),
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

    pub(super) fn note_submit_backoff(&self) {
        let mut until = lock_unpoisoned(&self.submit_backoff_until);
        let next = Instant::now() + SUBMIT_BACKOFF_DELAY;
        *until = Some(until.map_or(next, |prev| prev.min(next)));
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

    /// Reconcile in-flight pointwise jobs for a single `source` against
    /// `wanted` and current GPU/pending cache state, cancelling stale tokens
    /// strictly outside the scheduler lock.
    pub(super) fn reconcile_pointwise_source(
        &self,
        source: SourceId,
        wanted: &HashSet<TileKey>,
        gpu_cache: &GpuTileCache,
        pending: &PendingUploads,
    ) {
        let stale_tokens =
            lock_unpoisoned(&self.scheduler).reconcile_pointwise(source, wanted, |k| {
                gpu_cache.contains(k) || pending.contains(k)
            });
        for tok in stale_tokens {
            tok.cancel();
        }
    }

    /// Reconcile in-flight pointwise jobs across all active `layers` under a
    /// single scheduler lock acquisition, cancelling stale tokens strictly
    /// outside the lock.
    pub(super) fn reconcile_pointwise_layers(
        &self,
        layers: &[OverlayLayer],
        gpu_cache: &GpuTileCache,
        pending: &PendingUploads,
    ) {
        let mut to_cancel = Vec::new();
        {
            let mut sched = lock_unpoisoned(&self.scheduler);
            for layer in layers {
                if !layer.is_active() {
                    continue;
                }
                to_cancel.extend(sched.reconcile_pointwise(
                    layer.source_id(),
                    &layer.wanted,
                    |k| gpu_cache.contains(k) || pending.contains(k),
                ));
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
