//! Shared overlay runtime handles, RAII flight guards, and the dedicated
//! off-UI-thread [`OverlayIoWorker`] for non-blocking local file open/reload.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use veldmap_core::budget::{DecodeBudgetTracker, MemoryBudget};
use veldmap_core::geo::Frame;
use veldmap_core::sync::{CancelToken, Wakeup, lock_unpoisoned};
use veldmap_core::tile::{SourceId, TileKey};
use veldmap_raster::{
    DecodePoolHandle, PassKey, RasterSource, SharedChunkCache, TileFailureOutcome, TileScheduler,
};
use veldmap_storage::{LocalFingerprintCache, TileDiskCache};

use crate::gpu_tiles::UploadSender;

use super::LayerId;
use super::opener::open_local_raster;

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

    /// Cancel all pointwise and pass jobs for `source_id` (cancelling tokens
    /// strictly outside the scheduler lock) and invalidate decoded chunks.
    pub(super) fn purge_source_runtime(&self, source_id: SourceId) {
        let tokens = lock_unpoisoned(&self.scheduler).reset_source(source_id);
        for tok in tokens {
            tok.cancel();
        }
        self.chunk_cache.invalidate_source(source_id);
    }
}

/// RAII guard guaranteeing a pointwise tile never remains stuck in-flight in
/// [`TileScheduler`] if its `DecodeJob` is evicted from `DecodeQueue`,
/// cancelled, or panics inside a worker thread.
pub(super) struct PointwiseFlightGuard {
    runtime: Arc<OverlayRuntime>,
    key: TileKey,
    job_id: u64,
    completed: bool,
}

impl PointwiseFlightGuard {
    pub(super) fn new(runtime: Arc<OverlayRuntime>, key: TileKey, job_id: u64) -> Self {
        Self {
            runtime,
            key,
            job_id,
            completed: false,
        }
    }

    pub(super) fn mark_arrived(mut self) {
        self.completed = true;
        if lock_unpoisoned(&self.runtime.scheduler).on_pointwise_arrived(&self.key, self.job_id) {
            (self.runtime.wakeup)();
        }
    }

    pub(super) fn mark_failed(mut self, terminal: bool) {
        self.completed = true;
        let outcome = lock_unpoisoned(&self.runtime.scheduler).on_pointwise_failed(
            &self.key,
            self.job_id,
            terminal,
        );
        // Both RetryEligible and TerminalFailed require re-evaluating scheduling
        // (either to retry or to unblock finer ladder rungs behind a terminal failure).
        if matches!(
            outcome,
            Some(TileFailureOutcome::RetryEligible | TileFailureOutcome::TerminalFailed)
        ) {
            self.runtime.request_reschedule();
        }
    }

    pub(super) fn mark_cancelled(mut self) {
        self.completed = true;
        self.cleanup_cancelled();
    }

    fn cleanup_cancelled(&self) {
        // Only wake/reschedule if `(key, job_id)` was still registered in the
        // scheduler (e.g. evicted from DecodeQueue or panicked). Intentional
        // cancellations via `reconcile_pointwise`, `reset_source`, or `try_submit`
        // rejection already removed the entry and skip spurious wakeups.
        if lock_unpoisoned(&self.runtime.scheduler).cancel_in_flight(&self.key, self.job_id) {
            self.runtime.request_reschedule();
        }
    }
}

impl Drop for PointwiseFlightGuard {
    fn drop(&mut self) {
        if !self.completed {
            self.cleanup_cancelled();
        }
    }
}

/// RAII guard guaranteeing an active sequential pass in [`TileScheduler`] is
/// always closed even if evicted from `DecodeQueue` or unwound by a worker panic.
pub(super) struct PassFlightGuard {
    runtime: Arc<OverlayRuntime>,
    pass_key: PassKey,
    pass_id: u64,
    finished: bool,
}

impl PassFlightGuard {
    pub(super) fn new(runtime: Arc<OverlayRuntime>, pass_key: PassKey, pass_id: u64) -> Self {
        Self {
            runtime,
            pass_key,
            pass_id,
            finished: false,
        }
    }

    pub(super) fn finish(mut self, succeeded: bool) {
        self.finished = true;
        if lock_unpoisoned(&self.runtime.scheduler).finish_pass(
            self.pass_key,
            self.pass_id,
            succeeded,
        ) {
            self.runtime.request_reschedule();
        }
    }

    pub(super) fn abort(mut self) {
        self.finished = true;
        self.cleanup_aborted();
    }

    fn cleanup_aborted(&self) {
        let token =
            lock_unpoisoned(&self.runtime.scheduler).abort_pass(self.pass_key, self.pass_id);
        if let Some(tok) = token {
            tok.cancel();
            self.runtime.request_reschedule();
        }
    }
}

impl Drop for PassFlightGuard {
    fn drop(&mut self) {
        if !self.finished {
            self.cleanup_aborted();
        }
    }
}

/// Commands processed off the UI thread by [`OverlayIoWorker`].
pub(super) enum OverlayIoCmd {
    Open {
        path: PathBuf,
        fly_to: bool,
        cancel: CancelToken,
    },
    Reload {
        id: LayerId,
        reload_seq: u64,
        path: PathBuf,
        old_fingerprint_hex: String,
        cancel: CancelToken,
    },
}

/// Results posted back from [`OverlayIoWorker`] to `OverlayManager`.
pub(super) enum OverlayIoResult {
    Opened {
        path: PathBuf,
        fly_to: bool,
        outcome: Result<(Arc<dyn RasterSource>, Frame), String>,
    },
    Reloaded {
        id: LayerId,
        reload_seq: u64,
        path: PathBuf,
        outcome: Result<(Arc<dyn RasterSource>, Frame), String>,
    },
}

/// Dedicated single-thread background worker handling local GeoTIFF/COG
/// fingerprinting, IFD opening, and reload SQLite cache clearing strictly off
/// the UI thread (joined deterministically on shutdown).
pub(super) struct OverlayIoWorker {
    tx: Option<Sender<OverlayIoCmd>>,
    rx: Receiver<OverlayIoResult>,
    handle: Option<JoinHandle<()>>,
}

impl OverlayIoWorker {
    pub(super) fn spawn(runtime: Arc<OverlayRuntime>) -> Self {
        let (cmd_tx, cmd_rx) = mpsc::channel::<OverlayIoCmd>();
        let (res_tx, res_rx) = mpsc::channel::<OverlayIoResult>();
        let handle = match thread::Builder::new()
            .name("veldmap-overlay-io".to_owned())
            .spawn(move || {
                while let Ok(cmd) = cmd_rx.recv() {
                    if runtime.shutdown.is_cancelled() {
                        break;
                    }
                    match cmd {
                        OverlayIoCmd::Open {
                            path,
                            fly_to,
                            cancel,
                        } => {
                            if cancel.is_cancelled() {
                                continue;
                            }
                            let outcome = open_local_raster(
                                &path,
                                &runtime.fingerprint_cache,
                                &runtime.budget,
                                false,
                                &cancel,
                            );
                            if cancel.is_cancelled() {
                                continue;
                            }
                            let _ = res_tx.send(OverlayIoResult::Opened {
                                path,
                                fly_to,
                                outcome,
                            });
                            (runtime.wakeup)();
                        }
                        OverlayIoCmd::Reload {
                            id,
                            reload_seq,
                            path,
                            old_fingerprint_hex,
                            cancel,
                        } => {
                            if cancel.is_cancelled() {
                                continue;
                            }
                            runtime.fingerprint_cache.invalidate(&path);
                            let outcome = open_local_raster(
                                &path,
                                &runtime.fingerprint_cache,
                                &runtime.budget,
                                true,
                                &cancel,
                            );
                            if cancel.is_cancelled() {
                                continue;
                            }
                            // Only clear disk cache entries AFTER reopen succeeds so
                            // a failed reload preserves existing state, and run
                            // `clear_source` strictly off the UI thread.
                            if let Ok((ref source, _)) = outcome
                                && let Some(ref disk) = runtime.disk_cache
                            {
                                let _ = disk.clear_source(&old_fingerprint_hex);
                                let new_fp = source.fingerprint().to_hex();
                                if new_fp != old_fingerprint_hex {
                                    let _ = disk.clear_source(&new_fp);
                                }
                            }
                            let _ = res_tx.send(OverlayIoResult::Reloaded {
                                id,
                                reload_seq,
                                path,
                                outcome,
                            });
                            (runtime.wakeup)();
                        }
                    }
                }
            }) {
            Ok(h) => Some(h),
            Err(err) => {
                tracing::error!(error = %err, "Failed to spawn overlay I/O worker");
                None
            }
        };

        Self {
            tx: handle.is_some().then_some(cmd_tx),
            rx: res_rx,
            handle,
        }
    }

    pub(super) fn send(&self, cmd: OverlayIoCmd) -> bool {
        self.tx.as_ref().is_some_and(|tx| tx.send(cmd).is_ok())
    }

    pub(super) fn try_recv(&self) -> Option<OverlayIoResult> {
        self.rx.try_recv().ok()
    }

    pub(super) fn close_channel(&mut self) {
        self.tx = None;
    }

    pub(super) fn join(&mut self) {
        self.tx = None;
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for OverlayIoWorker {
    fn drop(&mut self) {
        self.join();
    }
}
