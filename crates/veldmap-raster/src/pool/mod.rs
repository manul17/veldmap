//! Generic bounded `std::thread` CPU decode pool with panic isolation,
//! configurable job-count + byte queue capping, coarsest-first priority
//! scheduling, deduplication by [`DecodeJobKey`] (`Probe`, `BandStats`, `Tile`, or `Pass`),
//! immediate cancellation checks, outside-mutex drop hygiene, and deterministic
//! shutdown joining.

mod queue;

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

use veldmap_core::budget::MemoryBudget;
use veldmap_core::sync::{CancelGuard, CancelToken, lock_unpoisoned, wait_unpoisoned};

use queue::DecodeQueue;
pub use queue::{DecodeJob, DecodeJobKey, SubmitError};

/// Default maximum queued decode jobs awaiting a CPU worker.
const DEFAULT_MAX_PENDING_DECODE_JOBS: usize = 128;

/// Default maximum cumulative queued bytes awaiting a CPU worker (`64 MiB`).
const DEFAULT_MAX_PENDING_DECODE_BYTES: usize = 64 * 1024 * 1024;

type SharedQueue = Arc<(Mutex<DecodeQueue>, Condvar)>;

fn close_queue_shared(shared: &SharedQueue) {
    let drained = {
        let (lock, cvar) = &**shared;
        let mut queue = lock_unpoisoned(lock);
        queue.closed = true;
        let drained = queue.drain_all();
        cvar.notify_all();
        drained
    };
    drop(drained);
}

/// Configuration for [`DecodePool::spawn`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodePoolConfig {
    /// Number of worker threads (clamped to `1..=8`).
    pub worker_count: usize,
    /// Maximum number of queued jobs before finer jobs are evicted or rejected.
    pub max_pending_jobs: usize,
    /// Maximum cumulative `queued_bytes` across queued jobs.
    pub max_pending_bytes: usize,
}

impl Default for DecodePoolConfig {
    fn default() -> Self {
        Self {
            worker_count: 4,
            max_pending_jobs: DEFAULT_MAX_PENDING_DECODE_JOBS,
            max_pending_bytes: DEFAULT_MAX_PENDING_DECODE_BYTES,
        }
    }
}

/// Cloneable producer handle for submitting work to a [`DecodePool`].
#[derive(Clone)]
pub struct DecodePoolHandle {
    shared: SharedQueue,
}

impl DecodePoolHandle {
    /// Enqueue a CPU decode job, returning `Err(SubmitError)` with ownership of
    /// the rejected job so callers can clean up scheduler/flight state before
    /// dropping it.
    ///
    /// Always drops evicted/cancelled queued jobs OUTSIDE `DecodeQueue`'s mutex
    /// to avoid lock-order deadlocks with caller drop guards.
    pub fn try_submit(&self, job: DecodeJob) -> Result<(), SubmitError> {
        if job.cancel.is_cancelled() {
            return Err(SubmitError::Cancelled(job));
        }
        let mut to_drop = Vec::new();
        let result = {
            let (lock, cvar) = &*self.shared;
            let mut queue = lock_unpoisoned(lock);
            let res = queue.admit_job(job, &mut to_drop);
            if res.is_ok() {
                cvar.notify_one();
            }
            res
        };
        // Drop evicted/cancelled jobs AFTER releasing `DecodeQueue`'s lock.
        drop(to_drop);
        result
    }

    /// Enqueue a CPU decode job. Convenience wrapper over [`Self::try_submit`]
    /// returning `true` if admitted and dropping rejected jobs outside the lock.
    pub fn submit(&self, job: DecodeJob) -> bool {
        self.try_submit(job).is_ok()
    }
}

/// Owned bounded `std::thread` CPU decode pool.
pub struct DecodePool {
    shared: SharedQueue,
    _shutdown_guard: CancelGuard,
    workers: Vec<JoinHandle<()>>,
}

impl DecodePool {
    /// Spawn a bounded CPU decode pool configured by `config`.
    pub fn spawn(config: DecodePoolConfig, shutdown: CancelToken) -> Self {
        Self::spawn_named("veldmap-tile-decode", config, shutdown)
    }

    /// Spawn a bounded CPU decode pool with explicit worker thread name prefix
    /// (`"{thread_prefix}-{idx}"`).
    fn spawn_named(thread_prefix: &str, config: DecodePoolConfig, shutdown: CancelToken) -> Self {
        let count = config.worker_count.clamp(1, 8);
        let shared = Arc::new((
            Mutex::new(DecodeQueue::new(
                config.max_pending_jobs,
                config.max_pending_bytes,
            )),
            Condvar::new(),
        ));

        // Wake sleeping decode threads immediately when `shutdown` cancels,
        // dropping cleared jobs strictly outside `DecodeQueue`'s lock.
        let wake_shared = Arc::clone(&shared);
        let shutdown_guard = shutdown.on_cancel_guarded(move || {
            close_queue_shared(&wake_shared);
        });

        let mut workers = Vec::with_capacity(count);
        for idx in 0..count {
            let worker_shared = Arc::clone(&shared);
            let stop = shutdown.clone();

            match thread::Builder::new()
                .name(format!("{thread_prefix}-{idx}"))
                .spawn(move || {
                    loop {
                        let (job, cancelled_batch) = {
                            let (lock, cvar) = &*worker_shared;
                            let mut queue = lock_unpoisoned(lock);
                            while queue.is_empty() && !queue.closed && !stop.is_cancelled() {
                                queue = wait_unpoisoned(cvar, queue);
                            }
                            if queue.closed || stop.is_cancelled() {
                                break;
                            }
                            queue.pop_coarsest_or_drain_cancelled()
                        };
                        drop(cancelled_batch);

                        let Some(job) = job else {
                            continue;
                        };
                        if job.cancel.is_cancelled() || stop.is_cancelled() {
                            continue;
                        }

                        // Panic isolation around decoders per workspace safety invariants.
                        let key = job.key;
                        if catch_unwind(AssertUnwindSafe(job.work)).is_err() {
                            tracing::error!(
                                ?key,
                                "Tile decode job panicked; isolated worker continues"
                            );
                        }
                    }
                }) {
                Ok(handle) => workers.push(handle),
                Err(err) => {
                    tracing::error!(idx, error = %err, "Failed to spawn tile decode worker");
                }
            }
        }

        if workers.is_empty() {
            let (lock, _) = &*shared;
            lock_unpoisoned(lock).closed = true;
        }

        Self {
            shared,
            _shutdown_guard: shutdown_guard,
            workers,
        }
    }

    /// Return a cloneable handle for submitting decode jobs.
    pub fn handle(&self) -> DecodePoolHandle {
        DecodePoolHandle {
            shared: Arc::clone(&self.shared),
        }
    }

    /// Close the queue and deterministically join all CPU decode threads.
    pub fn shutdown(mut self) {
        self.shutdown_inner();
    }

    fn close_queue(&self) {
        close_queue_shared(&self.shared);
    }

    fn shutdown_inner(&mut self) {
        self.close_queue();
        for handle in self.workers.drain(..) {
            let _ = handle.join();
        }
    }
}

impl Drop for DecodePool {
    fn drop(&mut self) {
        self.shutdown_inner();
    }
}

/// Configuration for [`TwoLaneDecodePool::spawn`], nesting per-lane
/// [`DecodePoolConfig`] settings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TwoLaneDecodePoolConfig {
    /// Fast pointwise tile lane (worker count clamped to `1..=8` by
    /// [`DecodePool::spawn`]).
    pub fast: DecodePoolConfig,
    /// Bulk sequential-pass lane (worker count clamped to `1..=4` by
    /// [`TwoLaneDecodePool::spawn`] so long-running passes do not saturate
    /// host CPU cores).
    pub bulk: DecodePoolConfig,
}

impl Default for TwoLaneDecodePoolConfig {
    fn default() -> Self {
        Self {
            fast: DecodePoolConfig {
                worker_count: 3,
                max_pending_jobs: DEFAULT_MAX_PENDING_DECODE_JOBS,
                max_pending_bytes: DEFAULT_MAX_PENDING_DECODE_BYTES,
            },
            bulk: DecodePoolConfig {
                worker_count: 2,
                max_pending_jobs: 64,
                max_pending_bytes: DEFAULT_MAX_PENDING_DECODE_BYTES,
            },
        }
    }
}

impl TwoLaneDecodePoolConfig {
    /// Derive lane pending byte caps from startup [`MemoryBudget`].
    pub fn from_budget(budget: &MemoryBudget) -> Self {
        let mut cfg = Self::default();
        cfg.fast.max_pending_bytes = budget.fast_lane_pending_bytes;
        cfg.bulk.max_pending_bytes = budget.bulk_lane_pending_bytes;
        cfg
    }
}

/// Owned two-lane bounded CPU decode pool preventing bulk passes and
/// full-image decodes from head-of-line blocking fast pointwise tile reads.
pub struct TwoLaneDecodePool {
    fast: DecodePool,
    bulk: DecodePool,
}

impl TwoLaneDecodePool {
    /// Spawn independent fast (`1..=8` workers) and bulk (`1..=4` workers)
    /// decode lanes governed by `shutdown`.
    pub fn spawn(config: TwoLaneDecodePoolConfig, shutdown: CancelToken) -> Self {
        let fast = DecodePool::spawn_named("veldmap-decode-fast", config.fast, shutdown.clone());
        let bulk = DecodePool::spawn_named(
            "veldmap-decode-bulk",
            DecodePoolConfig {
                worker_count: config.bulk.worker_count.clamp(1, 4),
                ..config.bulk
            },
            shutdown,
        );
        Self { fast, bulk }
    }

    /// Return a cloneable handle targeting the fast/pointwise decode lane.
    pub fn fast_handle(&self) -> DecodePoolHandle {
        self.fast.handle()
    }

    /// Return a cloneable handle targeting the bulk/sequential-pass decode lane.
    pub fn bulk_handle(&self) -> DecodePoolHandle {
        self.bulk.handle()
    }

    /// Close both queues before deterministically joining all worker threads.
    pub fn shutdown(mut self) {
        self.shutdown_inner();
    }

    fn shutdown_inner(&mut self) {
        self.fast.close_queue();
        self.bulk.close_queue();
        self.fast.shutdown_inner();
        self.bulk.shutdown_inner();
    }
}

impl Drop for TwoLaneDecodePool {
    fn drop(&mut self) {
        self.shutdown_inner();
    }
}

#[cfg(test)]
#[path = "pool_tests.rs"]
mod tests;
