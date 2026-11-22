//! Bounded `std::thread` CPU decode pool generating linear-light mipmaps on
//! decoded `Rgba8UnormSrgb` tiles with panic isolation, deduplication by
//! `TileKey`, immediate cancellation checks, and deterministic shutdown
//! joining.

use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

use veldmap_core::io::{CancelGuard, CancelToken, lock_unpoisoned};
use veldmap_core::tile::{DecodedTile, TileKey};
use veldmap_raster::decode_map_tile_with_mipmaps;

/// Maximum encoded tile payloads queued awaiting a CPU decode worker.
const MAX_PENDING_DECODE_JOBS: usize = 128;

/// UI repaint wakeup callback invoked when a decoded tile lands on the upload
/// channel.
pub type Wakeup = Arc<dyn Fn() + Send + Sync>;

/// Completion callback executed after decode finishes (`Ok(raw_bytes)` after
/// the decoded tile is sent to the upload channel, or `Err(reason)` if decode
/// failed or panicked). Owning the tile's `InFlightGuard` inside this callback
/// keeps `in_flight` and cancellation active all the way through CPU decode.
pub type DecodeCompletion = Box<dyn FnOnce(Result<Vec<u8>, String>) + Send + 'static>;

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

/// Cloneable producer handle for submitting raw encoded tile bytes to the
/// bounded CPU decode pool.
#[derive(Clone)]
pub struct DecodePoolHandle {
    shared: Arc<(Mutex<DecodeQueue>, Condvar)>,
}

impl DecodePoolHandle {
    /// Enqueue raw PNG/JPEG bytes for CPU decode + linear-light mipmap
    /// generation. Prunes cancelled jobs, deduplicates by `TileKey`, and
    /// enforces the bounded queue cap while always dropping evicted/rejected
    /// `DecodeJob`s OUTSIDE `DecodeQueue`'s mutex to avoid lock-order
    /// deadlocks with `InFlightGuard::drop`.
    pub fn submit(&self, job: DecodeJob) -> bool {
        if job.cancel.is_cancelled() {
            return false;
        }
        let mut to_drop = Vec::new();
        let accepted = {
            let (lock, cvar) = &*self.shared;
            let mut queue = lock_unpoisoned(lock);
            if queue.closed {
                to_drop.push(job);
                false
            } else {
                // Partition out cancelled jobs without dropping them under lock.
                let mut kept = VecDeque::with_capacity(queue.jobs.len());
                while let Some(existing) = queue.jobs.pop_front() {
                    if existing.cancel.is_cancelled() {
                        to_drop.push(existing);
                    } else {
                        kept.push_back(existing);
                    }
                }
                queue.jobs = kept;

                if queue.jobs.iter().any(|j| j.key == job.key) {
                    to_drop.push(job);
                    false
                } else if queue.jobs.len() >= MAX_PENDING_DECODE_JOBS {
                    // Only evict the finest queued job if the incoming job is
                    // strictly coarser; otherwise reject the finer incoming job.
                    let incoming_rank = job.key.coarsest_first_key();
                    let worst = queue
                        .jobs
                        .iter()
                        .enumerate()
                        .max_by_key(|(_, j)| j.key.coarsest_first_key())
                        .map(|(idx, j)| (idx, j.key.coarsest_first_key()));
                    match worst {
                        Some((worst_idx, worst_rank)) if incoming_rank < worst_rank => {
                            if let Some(evicted) = queue.jobs.remove(worst_idx) {
                                to_drop.push(evicted);
                            }
                            queue.jobs.push_back(job);
                            cvar.notify_one();
                            true
                        }
                        _ => {
                            to_drop.push(job);
                            false
                        }
                    }
                } else {
                    queue.jobs.push_back(job);
                    cvar.notify_one();
                    true
                }
            }
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

        // Wake sleeping decode threads immediately when `shutdown` cancels,
        // dropping cleared jobs strictly outside `DecodeQueue`'s lock.
        let wake_shared = Arc::clone(&shared);
        let shutdown_guard = shutdown.on_cancel_guarded(move || {
            let drained: Vec<DecodeJob> = {
                let (lock, cvar) = &*wake_shared;
                let mut queue = lock_unpoisoned(lock);
                queue.closed = true;
                let drained = queue.jobs.drain(..).collect();
                cvar.notify_all();
                drained
            };
            drop(drained);
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
                                Some(i) => (queue.jobs.remove(i), Vec::new()),
                                None => (None, queue.jobs.drain(..).collect()),
                            }
                        };
                        drop(cancelled_batch);

                        let Some(job) = job else {
                            continue;
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
                                (job.on_decoded)(Ok(job.bytes));
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
        let drained: Vec<DecodeJob> = {
            let (lock, cvar) = &*self.shared;
            let mut queue = lock_unpoisoned(lock);
            queue.closed = true;
            let drained = queue.jobs.drain(..).collect();
            cvar.notify_all();
            drained
        };
        drop(drained);
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

#[cfg(test)]
mod tests {
    use std::io::Cursor;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc;
    use std::time::Duration;

    use image::{ImageBuffer, ImageFormat, Rgba};
    use veldmap_core::tile::{QuadTileId, SourceId, TileScheme};

    use super::*;

    fn make_png(color: [u8; 4]) -> Vec<u8> {
        let img: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::from_pixel(4, 4, Rgba(color));
        let mut out = Vec::new();
        img.write_to(&mut Cursor::new(&mut out), ImageFormat::Png)
            .expect("encode png");
        out
    }

    fn tile_key(level: u8, x: u32, y: u32) -> TileKey {
        TileKey::display_quad(
            SourceId::from_stable_key("decode-test"),
            QuadTileId {
                scheme: TileScheme::WebMercator,
                level,
                x,
                y,
            },
        )
    }

    #[test]
    fn queue_cap_evicts_finer_for_coarser_and_rejects_finer_incoming() {
        let shared = Arc::new((Mutex::new(DecodeQueue::default()), Condvar::new()));
        let handle = DecodePoolHandle {
            shared: Arc::clone(&shared),
        };
        let tok = CancelToken::new();

        // Fill queue with MAX_PENDING_DECODE_JOBS at level 5.
        for x in 0..MAX_PENDING_DECODE_JOBS as u32 {
            assert!(handle.submit(DecodeJob {
                key: tile_key(5, x, 0),
                bytes: Vec::new(),
                cancel: tok.clone(),
                on_decoded: Box::new(|_| {}),
            }));
        }
        // Duplicate key is rejected.
        assert!(!handle.submit(DecodeJob {
            key: tile_key(5, 0, 0),
            bytes: Vec::new(),
            cancel: tok.clone(),
            on_decoded: Box::new(|_| {}),
        }));
        // Finer incoming job (level 6) must be rejected without evicting level 5.
        assert!(!handle.submit(DecodeJob {
            key: tile_key(6, 0, 0),
            bytes: Vec::new(),
            cancel: tok.clone(),
            on_decoded: Box::new(|_| {}),
        }));
        assert_eq!(
            lock_unpoisoned(&shared.0).jobs.len(),
            MAX_PENDING_DECODE_JOBS
        );
        assert!(
            lock_unpoisoned(&shared.0)
                .jobs
                .iter()
                .all(|j| j.key.level == 5)
        );

        // Coarser incoming job (level 0) evicts one level 5 job and is admitted.
        assert!(handle.submit(DecodeJob {
            key: tile_key(0, 0, 0),
            bytes: Vec::new(),
            cancel: tok.clone(),
            on_decoded: Box::new(|_| {}),
        }));
        let q = lock_unpoisoned(&shared.0);
        assert_eq!(q.jobs.len(), MAX_PENDING_DECODE_JOBS);
        assert!(q.jobs.iter().any(|j| j.key == tile_key(0, 0, 0)));
    }

    #[test]
    fn decodes_valid_tiles_handles_corrupt_payloads_and_drops_cancelled_outside_lock() {
        let (tx, rx) = mpsc::channel();
        let wakes = Arc::new(AtomicUsize::new(0));
        let w_clone = Arc::clone(&wakes);
        let wakeup: Wakeup = Arc::new(move || {
            w_clone.fetch_add(1, Ordering::AcqRel);
        });
        let shutdown = CancelToken::new();
        let pool = DecodePool::spawn(1, tx, wakeup, shutdown.clone());
        let handle = pool.handle();

        // Submit a cancelled job that re-enters `handle.submit` on Drop; if Drop
        // ran while holding `DecodeQueue`'s lock, this would deadlock immediately.
        let cancelled = CancelToken::new();
        let reenter_handle = handle.clone();
        let reenter_tok = CancelToken::new();
        struct ReenterOnDrop(DecodePoolHandle, CancelToken);
        impl Drop for ReenterOnDrop {
            fn drop(&mut self) {
                let _ = self.0.submit(DecodeJob {
                    key: tile_key(9, 0, 0),
                    bytes: Vec::new(),
                    cancel: self.1.clone(),
                    on_decoded: Box::new(|_| {}),
                });
            }
        }
        cancelled.cancel();
        let guard_probe = ReenterOnDrop(reenter_handle, reenter_tok);
        assert!(!handle.submit(DecodeJob {
            key: tile_key(8, 0, 0),
            bytes: Vec::new(),
            cancel: cancelled,
            on_decoded: Box::new(move |_| {
                let _ = guard_probe;
            }),
        }));

        // Valid PNG decodes with full mip chain and invokes on_decoded(Ok) + wakeup.
        let ok_hits = Arc::new(AtomicUsize::new(0));
        let ok_clone = Arc::clone(&ok_hits);
        let png = make_png([10, 200, 30, 255]);
        assert!(handle.submit(DecodeJob {
            key: tile_key(1, 0, 0),
            bytes: png,
            cancel: shutdown.child_token(),
            on_decoded: Box::new(move |res| {
                assert!(res.is_ok());
                ok_clone.fetch_add(1, Ordering::AcqRel);
            }),
        }));

        // Corrupt bytes report Err to on_decoded without crashing the worker.
        let err_hits = Arc::new(AtomicUsize::new(0));
        let err_clone = Arc::clone(&err_hits);
        assert!(handle.submit(DecodeJob {
            key: tile_key(2, 0, 0),
            bytes: b"not a valid image".to_vec(),
            cancel: shutdown.child_token(),
            on_decoded: Box::new(move |res| {
                assert!(res.is_err());
                err_clone.fetch_add(1, Ordering::AcqRel);
            }),
        }));

        let (k, decoded) = rx
            .recv_timeout(Duration::from_secs(2))
            .expect("receive decoded tile");
        assert_eq!(k, tile_key(1, 0, 0));
        assert_eq!(decoded.mip_level_count(), 3); // 4x4 -> 2x2 -> 1x1

        // Wait briefly for corrupt job to finish and verify worker stays alive.
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        while err_hits.load(Ordering::Acquire) == 0 && std::time::Instant::now() < deadline {
            thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(ok_hits.load(Ordering::Acquire), 1);
        assert_eq!(err_hits.load(Ordering::Acquire), 1);
        assert!(wakes.load(Ordering::Acquire) >= 1);

        pool.shutdown();
    }
}
