//! Bounded `std::thread` CPU decode pool generating linear-light mipmaps on
//! decoded `Rgba8UnormSrgb` tiles with panic isolation, deduplication by
//! `TileKey`, immediate cancellation checks, and deterministic shutdown
//! joining.

use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

use veldmap_core::io::CancelToken;
use veldmap_core::tile::{DecodedTile, TileKey};
use veldmap_raster::decode_map_tile_with_mipmaps;

/// Maximum encoded tile payloads queued awaiting a CPU decode worker.
const MAX_PENDING_DECODE_JOBS: usize = 128;

/// UI repaint wakeup callback invoked when a decoded tile lands on the upload
/// channel.
pub type Wakeup = Arc<dyn Fn() + Send + Sync>;

/// Completion callback executed after decode finishes (`Ok(&raw_bytes)` after
/// the decoded tile is sent to the upload channel, or `Err(reason)` if decode
/// failed or panicked). Owning the tile's `InFlightGuard` inside this callback
/// keeps `in_flight` and cancellation active all the way through CPU decode.
pub type DecodeCompletion = Box<dyn FnOnce(Result<&[u8], String>) + Send + 'static>;

/// Single encoded tile decode request submitted to [`DecodePoolHandle`].
pub struct DecodeJob {
    pub key: TileKey,
    pub bytes: Vec<u8>,
    pub cancel: CancelToken,
    pub on_decoded: DecodeCompletion,
}

#[derive(Default)]
struct DecodeQueue {
    jobs: VecDeque<DecodeJob>,
    closed: bool,
}

fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// Cloneable producer handle for submitting raw encoded tile bytes to the
/// bounded CPU decode pool.
#[derive(Clone)]
pub struct DecodePoolHandle {
    shared: Arc<(Mutex<DecodeQueue>, Condvar)>,
}

impl DecodePoolHandle {
    /// Enqueue raw PNG/JPEG bytes for CPU decode + linear-light mipmap
    /// generation. Prunes cancelled jobs, deduplicates by `TileKey`, and
    /// enforces the bounded queue cap so decode backlog RAM stays bounded.
    pub fn submit(&self, job: DecodeJob) -> bool {
        if job.cancel.is_cancelled() {
            return false;
        }
        let (lock, cvar) = &*self.shared;
        let mut queue = lock_unpoisoned(lock);
        if queue.closed {
            return false;
        }
        queue.jobs.retain(|j| !j.cancel.is_cancelled());
        if queue.jobs.iter().any(|j| j.key == job.key) {
            return false;
        }
        if queue.jobs.len() >= MAX_PENDING_DECODE_JOBS {
            // Keep coarser tiles by evicting the finest unstarted job when saturated.
            if let Some((worst_idx, _)) = queue
                .jobs
                .iter()
                .enumerate()
                .max_by_key(|(_, j)| j.key.coarsest_first_key())
            {
                queue.jobs.remove(worst_idx);
            }
        }
        queue.jobs.push_back(job);
        cvar.notify_one();
        true
    }
}

/// Owned bounded `std::thread` CPU decode pool.
pub struct DecodePool {
    shared: Arc<(Mutex<DecodeQueue>, Condvar)>,
    workers: Vec<JoinHandle<()>>,
}

impl DecodePool {
    /// Spawn `worker_count` (clamped to `1..=8`) CPU decode threads pushing
    /// mipmapped `DecodedTile`s into `done_tx` and triggering `wakeup`.
    pub fn spawn(
        worker_count: usize,
        done_tx: Sender<(TileKey, DecodedTile)>,
        wakeup: Wakeup,
        shutdown: CancelToken,
    ) -> Self {
        let count = worker_count.clamp(1, 8);
        let shared = Arc::new((Mutex::new(DecodeQueue::default()), Condvar::new()));

        // Wake sleeping decode threads immediately when `shutdown` cancels.
        let wake_shared = Arc::clone(&shared);
        shutdown.on_cancel(move || {
            let (lock, cvar) = &*wake_shared;
            let mut queue = lock_unpoisoned(lock);
            queue.closed = true;
            queue.jobs.clear();
            cvar.notify_all();
        });

        let mut workers = Vec::with_capacity(count);
        for idx in 0..count {
            let worker_shared = Arc::clone(&shared);
            let tx = done_tx.clone();
            let wake = Arc::clone(&wakeup);
            let stop = shutdown.clone();

            match thread::Builder::new()
                .name(format!("veldmap-tile-decode-{idx}"))
                .spawn(move || {
                    loop {
                        let job = {
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
                                Some(i) => match queue.jobs.remove(i) {
                                    Some(job) => job,
                                    None => continue,
                                },
                                None => {
                                    queue.jobs.clear();
                                    continue;
                                }
                            }
                        };

                        if job.cancel.is_cancelled() || stop.is_cancelled() {
                            continue;
                        }

                        // Panic isolation around image codecs per workspace safety invariants.
                        let decoded = catch_unwind(AssertUnwindSafe(|| {
                            decode_map_tile_with_mipmaps(&job.bytes)
                        }));

                        match decoded {
                            Ok(Ok(tile)) => {
                                if job.cancel.is_cancelled() || stop.is_cancelled() {
                                    continue;
                                }
                                if tx.send((job.key, tile)).is_err() {
                                    break;
                                }
                                (job.on_decoded)(Ok(&job.bytes));
                                (wake)();
                            }
                            Ok(Err(err)) => {
                                tracing::debug!(
                                    key = ?job.key,
                                    error = %err,
                                    "Failed to decode map tile payload"
                                );
                                (job.on_decoded)(Err(err.to_string()));
                            }
                            Err(_) => {
                                tracing::error!(
                                    key = ?job.key,
                                    "Tile decoder panicked; isolated worker continues"
                                );
                                (job.on_decoded)(Err("tile decoder panicked".to_owned()));
                            }
                        }
                    }
                }) {
                Ok(handle) => workers.push(handle),
                Err(err) => {
                    tracing::error!(idx, error = %err, "Failed to spawn tile decode worker");
                }
            }
        }

        Self { shared, workers }
    }

    /// Return a cloneable handle for submitting encoded tile jobs.
    pub fn handle(&self) -> DecodePoolHandle {
        DecodePoolHandle {
            shared: Arc::clone(&self.shared),
        }
    }

    /// Close the queue and deterministically join all CPU decode threads.
    pub fn shutdown(mut self) {
        self.shutdown_inner();
    }

    fn shutdown_inner(&mut self) {
        {
            let (lock, cvar) = &*self.shared;
            let mut queue = lock_unpoisoned(lock);
            queue.closed = true;
            queue.jobs.clear();
            cvar.notify_all();
        }
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
