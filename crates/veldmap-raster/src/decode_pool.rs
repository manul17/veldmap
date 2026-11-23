//! Generic bounded `std::thread` CPU decode pool with panic isolation,
//! configurable job-count + byte queue capping, coarsest-first priority
//! scheduling, deduplication by `TileKey`, immediate cancellation checks,
//! outside-mutex drop hygiene, and deterministic shutdown joining.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

use veldmap_core::budget::MemoryBudget;
use veldmap_core::io::{CancelGuard, CancelToken, lock_unpoisoned};
use veldmap_core::tile::TileKey;

/// Default maximum queued decode jobs awaiting a CPU worker.
pub const DEFAULT_MAX_PENDING_DECODE_JOBS: usize = 128;

/// Default maximum cumulative queued bytes awaiting a CPU worker (`64 MiB`).
pub const DEFAULT_MAX_PENDING_DECODE_BYTES: usize = 64 * 1024 * 1024;

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

/// Single prioritized, byte-accounted CPU decode job submitted to
/// [`DecodePoolHandle`].
pub struct DecodeJob {
    pub key: TileKey,
    /// Accounted payload/working bytes charged against `max_pending_bytes`.
    pub queued_bytes: usize,
    pub cancel: CancelToken,
    /// Decode closure executed on a worker thread inside `catch_unwind`.
    pub work: Box<dyn FnOnce() + Send + 'static>,
}

struct DecodeQueue {
    jobs: Vec<DecodeJob>,
    queued_bytes: usize,
    max_pending_jobs: usize,
    max_pending_bytes: usize,
    closed: bool,
}

impl DecodeQueue {
    fn new(max_pending_jobs: usize, max_pending_bytes: usize) -> Self {
        Self {
            jobs: Vec::new(),
            queued_bytes: 0,
            max_pending_jobs: max_pending_jobs.max(1),
            max_pending_bytes,
            closed: false,
        }
    }

    fn push_job(&mut self, job: DecodeJob) {
        self.queued_bytes = self.queued_bytes.saturating_add(job.queued_bytes);
        self.jobs.push(job);
    }

    fn swap_remove_job(&mut self, idx: usize) -> DecodeJob {
        let job = self.jobs.swap_remove(idx);
        self.queued_bytes = self.queued_bytes.saturating_sub(job.queued_bytes);
        job
    }

    fn drain_all(&mut self) -> Vec<DecodeJob> {
        self.queued_bytes = 0;
        std::mem::take(&mut self.jobs)
    }

    fn prune_cancelled(&mut self, to_drop: &mut Vec<DecodeJob>) {
        let mut i = 0;
        while i < self.jobs.len() {
            if self.jobs[i].cancel.is_cancelled() {
                let removed = self.swap_remove_job(i);
                to_drop.push(removed);
            } else {
                i += 1;
            }
        }
    }

    /// Plan eviction victims transactionally before mutating the queue so
    /// rejecting an incoming job never destroys valid finer queued jobs
    /// (always permitting a single job into an empty queue).
    fn plan_evictions(&self, key: &TileKey, incoming_bytes: usize) -> Option<Vec<usize>> {
        let needs_eviction = !self.jobs.is_empty()
            && (self.jobs.len() >= self.max_pending_jobs
                || self.queued_bytes.saturating_add(incoming_bytes) > self.max_pending_bytes);
        if !needs_eviction {
            return Some(Vec::new());
        }

        let incoming_coarse = key.scheme.coarseness_rank(key.level);
        let mut candidates: Vec<(usize, _, usize)> = self
            .jobs
            .iter()
            .enumerate()
            .filter_map(|(idx, j)| {
                let victim_coarse = j.key.scheme.coarseness_rank(j.key.level);
                (incoming_coarse < victim_coarse).then_some((
                    idx,
                    j.key.coarsest_first_key(),
                    j.queued_bytes,
                ))
            })
            .collect();
        // Finest-first victim order (highest `coarsest_first_key` first).
        candidates.sort_by_key(|c| std::cmp::Reverse(c.1));

        let mut remaining_jobs = self.jobs.len();
        let mut remaining_bytes = self.queued_bytes;
        let mut victims = Vec::new();
        for (idx, _, bytes) in candidates {
            if remaining_jobs == 0
                || (remaining_jobs < self.max_pending_jobs
                    && remaining_bytes.saturating_add(incoming_bytes) <= self.max_pending_bytes)
            {
                break;
            }
            remaining_jobs -= 1;
            remaining_bytes = remaining_bytes.saturating_sub(bytes);
            victims.push(idx);
        }

        let fits = remaining_jobs == 0
            || (remaining_jobs < self.max_pending_jobs
                && remaining_bytes.saturating_add(incoming_bytes) <= self.max_pending_bytes);
        fits.then_some(victims)
    }

    fn admit_job(&mut self, job: DecodeJob, to_drop: &mut Vec<DecodeJob>) -> bool {
        if self.closed {
            to_drop.push(job);
            return false;
        }
        self.prune_cancelled(to_drop);
        if self.jobs.iter().any(|j| j.key == job.key) {
            to_drop.push(job);
            return false;
        }
        let Some(mut victims) = self.plan_evictions(&job.key, job.queued_bytes) else {
            to_drop.push(job);
            return false;
        };
        // Remove in descending index order so `swap_remove` never invalidates
        // earlier planned indices.
        victims.sort_unstable_by_key(|&idx| std::cmp::Reverse(idx));
        for idx in victims {
            to_drop.push(self.swap_remove_job(idx));
        }
        self.push_job(job);
        true
    }
}

/// Cloneable producer handle for submitting work to a [`DecodePool`].
#[derive(Clone)]
pub struct DecodePoolHandle {
    shared: Arc<(Mutex<DecodeQueue>, Condvar)>,
}

impl DecodePoolHandle {
    /// Enqueue a CPU decode job. Prunes cancelled jobs, deduplicates by
    /// `TileKey`, and enforces both `max_pending_jobs` and `max_pending_bytes`
    /// caps while always dropping evicted or rejected `DecodeJob`s OUTSIDE
    /// `DecodeQueue`'s mutex to avoid lock-order deadlocks with caller drop
    /// guards.
    pub fn submit(&self, job: DecodeJob) -> bool {
        if job.cancel.is_cancelled() {
            return false;
        }
        let mut to_drop = Vec::new();
        let accepted = {
            let (lock, cvar) = &*self.shared;
            let mut queue = lock_unpoisoned(lock);
            let admitted = queue.admit_job(job, &mut to_drop);
            if admitted {
                cvar.notify_one();
            }
            admitted
        };
        // Drop evicted/cancelled/rejected jobs AFTER releasing `DecodeQueue`'s lock.
        drop(to_drop);
        accepted
    }
}

/// Owned bounded `std::thread` CPU decode pool.
pub struct DecodePool {
    shared: Arc<(Mutex<DecodeQueue>, Condvar)>,
    _shutdown_guard: CancelGuard,
    workers: Vec<JoinHandle<()>>,
}

impl DecodePool {
    /// Spawn a bounded CPU decode pool configured by `config`.
    pub fn spawn(config: DecodePoolConfig, shutdown: CancelToken) -> Self {
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
            let drained = {
                let (lock, cvar) = &*wake_shared;
                let mut queue = lock_unpoisoned(lock);
                queue.closed = true;
                let drained = queue.drain_all();
                cvar.notify_all();
                drained
            };
            drop(drained);
        });

        let mut workers = Vec::with_capacity(count);
        for idx in 0..count {
            let worker_shared = Arc::clone(&shared);
            let stop = shutdown.clone();

            match thread::Builder::new()
                .name(format!("veldmap-tile-decode-{idx}"))
                .spawn(move || {
                    loop {
                        let (job, cancelled_batch) = {
                            let (lock, cvar) = &*worker_shared;
                            let mut queue = lock_unpoisoned(lock);
                            while queue.jobs.is_empty() && !queue.closed && !stop.is_cancelled() {
                                queue = match cvar.wait(queue) {
                                    Ok(guard) => guard,
                                    Err(poisoned) => poisoned.into_inner(),
                                };
                            }
                            if queue.closed || stop.is_cancelled() {
                                break;
                            }
                            // Pop coarsest uncancelled job first so roots land first.
                            let best_idx = queue
                                .jobs
                                .iter()
                                .enumerate()
                                .filter(|(_, j)| !j.cancel.is_cancelled())
                                .min_by_key(|(_, j)| j.key.coarsest_first_key())
                                .map(|(i, _)| i);
                            match best_idx {
                                Some(i) => (Some(queue.swap_remove_job(i)), Vec::new()),
                                None => (None, queue.drain_all()),
                            }
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
        let drained = {
            let (lock, cvar) = &*self.shared;
            let mut queue = lock_unpoisoned(lock);
            queue.closed = true;
            let drained = queue.drain_all();
            cvar.notify_all();
            drained
        };
        drop(drained);
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

/// Execution lane for [`TwoLaneDecodePool`], isolating small pointwise tiled
/// reads from slow sequential passes and full-image decodes so neither starves
/// or head-of-line blocks the other.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DecodeLane {
    /// Pointwise COG/basemap tile decodes (fast, small bounded chunks).
    Fast,
    /// Sequential raster passes and full-image decodes (bulk, long-running).
    Bulk,
}

/// Configuration for [`TwoLaneDecodePool::spawn`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TwoLaneDecodePoolConfig {
    pub fast_workers: usize,
    pub bulk_workers: usize,
    pub max_pending_fast_jobs: usize,
    pub max_pending_fast_bytes: usize,
    pub max_pending_bulk_jobs: usize,
    pub max_pending_bulk_bytes: usize,
}

impl Default for TwoLaneDecodePoolConfig {
    fn default() -> Self {
        Self {
            fast_workers: 3,
            bulk_workers: 2,
            max_pending_fast_jobs: DEFAULT_MAX_PENDING_DECODE_JOBS,
            max_pending_fast_bytes: DEFAULT_MAX_PENDING_DECODE_BYTES,
            max_pending_bulk_jobs: 64,
            max_pending_bulk_bytes: DEFAULT_MAX_PENDING_DECODE_BYTES,
        }
    }
}

impl TwoLaneDecodePoolConfig {
    /// Derive lane pending byte caps from startup [`MemoryBudget`].
    pub fn from_budget(budget: &MemoryBudget) -> Self {
        Self {
            max_pending_fast_bytes: budget.fast_lane_pending_bytes,
            max_pending_bulk_bytes: budget.bulk_lane_pending_bytes,
            ..Self::default()
        }
    }
}

/// Cloneable producer handle for submitting work to either [`DecodeLane`] of a
/// [`TwoLaneDecodePool`].
#[derive(Clone)]
pub struct TwoLaneDecodePoolHandle {
    fast: DecodePoolHandle,
    bulk: DecodePoolHandle,
}

impl TwoLaneDecodePoolHandle {
    /// Submit `job` onto `lane`.
    pub fn submit(&self, lane: DecodeLane, job: DecodeJob) -> bool {
        match lane {
            DecodeLane::Fast => self.fast.submit(job),
            DecodeLane::Bulk => self.bulk.submit(job),
        }
    }

    /// Borrow the fast/pointwise lane handle directly.
    pub fn fast(&self) -> &DecodePoolHandle {
        &self.fast
    }

    /// Borrow the bulk/pass/full-image lane handle directly.
    pub fn bulk(&self) -> &DecodePoolHandle {
        &self.bulk
    }
}

/// Owned two-lane bounded CPU decode pool preventing bulk passes and
/// full-image decodes from head-of-line blocking fast pointwise tile reads.
pub struct TwoLaneDecodePool {
    fast: DecodePool,
    bulk: DecodePool,
}

impl TwoLaneDecodePool {
    /// Spawn independent fast and bulk decode lanes governed by `shutdown`.
    pub fn spawn(config: TwoLaneDecodePoolConfig, shutdown: CancelToken) -> Self {
        let fast = DecodePool::spawn(
            DecodePoolConfig {
                worker_count: config.fast_workers.clamp(1, 8),
                max_pending_jobs: config.max_pending_fast_jobs,
                max_pending_bytes: config.max_pending_fast_bytes,
            },
            shutdown.clone(),
        );
        let bulk = DecodePool::spawn(
            DecodePoolConfig {
                worker_count: config.bulk_workers.clamp(1, 4),
                max_pending_jobs: config.max_pending_bulk_jobs,
                max_pending_bytes: config.max_pending_bulk_bytes,
            },
            shutdown,
        );
        Self { fast, bulk }
    }

    /// Return a cloneable handle targeting both lanes.
    pub fn handle(&self) -> TwoLaneDecodePoolHandle {
        TwoLaneDecodePoolHandle {
            fast: self.fast.handle(),
            bulk: self.bulk.handle(),
        }
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
#[path = "decode_pool_tests.rs"]
mod tests;
