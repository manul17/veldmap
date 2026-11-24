//! Bounded multi-producer GPU upload queue (`PendingUploads` / `UploadSender`)
//! enforcing `max_queued_bytes` on enqueue so minimized or throttled UI loops
//! never accumulate unbounded decoded tiles in CPU RAM.

use std::cmp::Reverse;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use veldmap_core::sync::{CancelToken, lock_unpoisoned};
use veldmap_core::tile::{DecodedTile, SourceId, TileKey};

use super::gpu_cache::{GpuTileCache, UploadInsertOutcome};

/// Default per-frame upload byte budget (`4 MiB`).
pub(crate) const DEFAULT_UPLOAD_BYTE_BUDGET: usize = 4 * 1024 * 1024;

/// Default per-frame upload wall-clock budget (`2 ms`).
pub(crate) const DEFAULT_UPLOAD_TIME_BUDGET: Duration = Duration::from_millis(2);

/// Default CPU byte cap on decoded tiles buffered in [`PendingUploads`]
/// awaiting GPU upload (`64 MiB`).
const DEFAULT_MAX_QUEUED_UPLOAD_BYTES: usize = 64 * 1024 * 1024;

/// Error returned by [`UploadSender::send`] after [`PendingUploads::close`]
/// shuts down the upload queue.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UploadChannelClosed;

impl fmt::Display for UploadChannelClosed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("GPU tile upload queue is closed")
    }
}

impl std::error::Error for UploadChannelClosed {}

/// Parameters governing one frame's [`PendingUploads::drain_and_upload`] pass.
///
/// Pinned root sources are configured once via
/// [`PendingUploads::set_pinned_root_sources`] so enqueue-time overflow
/// shedding and drain-time pruning always share a single source of truth.
#[derive(Clone, Copy, Debug)]
pub struct UploadPassParams<'a> {
    /// Currently desired tile key sets across all active layers (an empty
    /// slice skips `wanted` pruning; otherwise a non-pinned tile is kept if it
    /// appears in any set).
    pub wanted: &'a [&'a HashSet<TileKey>],
    /// Active-frame carrier keys protected from LRU eviction during upload.
    pub protected: Option<&'a HashSet<TileKey>>,
    pub byte_budget: usize,
    pub time_budget: Duration,
}

/// Summary of one frame's [`PendingUploads::drain_and_upload`] pass.
#[derive(Clone, Debug, Default)]
pub struct UploadBatchReport {
    pub uploaded: usize,
    pub evicted: usize,
    /// Total tiles still queued (`remaining_ready + deferred_protected`).
    pub remaining: usize,
    /// Tiles deferred solely by per-frame byte/time budgets (should request an
    /// immediate follow-up repaint).
    pub remaining_ready: usize,
    /// Tiles kept queued because active-frame protection blocked eviction (must
    /// NOT trigger an infinite idle repaint loop on its own).
    pub deferred_protected: usize,
    /// Largest single uploaded tile byte size observed per source in this pass.
    pub max_tile_bytes_by_source: Vec<(SourceId, usize)>,
}

impl UploadBatchReport {
    /// Return the largest single tile byte size uploaded for `source` during
    /// this pass (`0` if no tile of `source` was uploaded).
    pub fn max_tile_bytes_for(&self, source: SourceId) -> usize {
        self.max_tile_bytes_by_source
            .iter()
            .find_map(|&(src, bytes)| (src == source).then_some(bytes))
            .unwrap_or(0)
    }

    fn record_source_tile_bytes(&mut self, source: SourceId, byte_size: usize) {
        if let Some((_, max_bytes)) = self
            .max_tile_bytes_by_source
            .iter_mut()
            .find(|(src, _)| *src == source)
        {
            *max_bytes = (*max_bytes).max(byte_size);
        } else {
            self.max_tile_bytes_by_source.push((source, byte_size));
        }
    }
}

fn is_pinned_root(key: &TileKey, pinned_root_sources: &[SourceId]) -> bool {
    pinned_root_sources.iter().any(|&src| key.is_root_of(src))
}

fn is_in_any_wanted(key: &TileKey, wanted: &[&HashSet<TileKey>]) -> bool {
    wanted.iter().any(|set| set.contains(key))
}

struct SharedUploadQueue {
    closed: bool,
    queued: HashMap<TileKey, DecodedTile>,
    queued_bytes: usize,
    max_queued_bytes: usize,
    pinned_root_sources: Vec<SourceId>,
}

impl SharedUploadQueue {
    fn new(max_queued_bytes: usize) -> Self {
        Self {
            closed: false,
            queued: HashMap::new(),
            queued_bytes: 0,
            max_queued_bytes,
            pinned_root_sources: vec![SourceId::BUNDLED_BASEMAP],
        }
    }

    fn insert_and_bound(
        &mut self,
        key: TileKey,
        decoded: DecodedTile,
        to_drop: &mut Vec<DecodedTile>,
    ) {
        let pinned = is_pinned_root(&key, &self.pinned_root_sources);
        let added = decoded.byte_size();
        if !pinned && added > self.max_queued_bytes {
            to_drop.push(decoded);
            return;
        }
        if let Some(prev) = self.queued.insert(key, decoded) {
            self.queued_bytes = self.queued_bytes.saturating_sub(prev.byte_size());
            to_drop.push(prev);
        }
        self.queued_bytes = self.queued_bytes.saturating_add(added);
        self.enforce_cap(to_drop);
    }

    fn remove_key(&mut self, key: &TileKey) -> Option<DecodedTile> {
        let removed = self.queued.remove(key)?;
        self.queued_bytes = self.queued_bytes.saturating_sub(removed.byte_size());
        Some(removed)
    }

    fn retain_keys(
        &mut self,
        mut keep: impl FnMut(&TileKey) -> bool,
        to_drop: &mut Vec<DecodedTile>,
    ) {
        let keys_to_remove: Vec<TileKey> =
            self.queued.keys().copied().filter(|k| !keep(k)).collect();
        for key in keys_to_remove {
            if let Some(tile) = self.remove_key(&key) {
                to_drop.push(tile);
            }
        }
    }

    fn enforce_cap(&mut self, to_drop: &mut Vec<DecodedTile>) {
        if self.queued_bytes <= self.max_queued_bytes {
            return;
        }
        // Drop finest non-pinned tiles first so coarsest carriers and pinned
        // roots always survive bursts from fast SSD/COG decode workers.
        let mut candidates: Vec<TileKey> = self
            .queued
            .keys()
            .copied()
            .filter(|k| !is_pinned_root(k, &self.pinned_root_sources))
            .collect();
        candidates.sort_by_key(|k| Reverse(k.coarsest_first_key()));
        for key in candidates {
            if self.queued_bytes <= self.max_queued_bytes {
                break;
            }
            if let Some(tile) = self.remove_key(&key) {
                to_drop.push(tile);
            }
        }
    }
}

/// Cloneable producer handle that enqueues decoded tiles directly into the
/// byte-bounded upload queue, shedding finest non-pinned tiles immediately on
/// overflow so stalled or minimized UI loops cannot leak unbounded RAM.
#[derive(Clone)]
pub struct UploadSender {
    shared: Arc<Mutex<SharedUploadQueue>>,
}

impl UploadSender {
    /// Submit a decoded tile for GPU upload only if `cancel` has not been
    /// triggered (checked both before and under the queue lock so concurrent
    /// `cancel + clear_source` reloads cannot be raced by stale workers).
    ///
    /// Returns `Ok(true)` if admitted, `Ok(false)` if cancelled, or
    /// `Err(UploadChannelClosed)` once [`PendingUploads::close`] has run.
    pub fn send_if_uncancelled(
        &self,
        item: (TileKey, DecodedTile),
        cancel: &CancelToken,
    ) -> Result<bool, UploadChannelClosed> {
        if cancel.is_cancelled() {
            return Ok(false);
        }
        let (key, decoded) = item;
        let mut to_drop = Vec::new();
        let outcome = {
            let mut state = lock_unpoisoned(&self.shared);
            if state.closed {
                to_drop.push(decoded);
                Err(UploadChannelClosed)
            } else if cancel.is_cancelled() {
                to_drop.push(decoded);
                Ok(false)
            } else {
                state.insert_and_bound(key, decoded, &mut to_drop);
                Ok(true)
            }
        };
        drop(to_drop);
        outcome
    }

    /// Submit a decoded tile for GPU upload.
    ///
    /// Returns `Err(UploadChannelClosed)` once [`PendingUploads::close`] has
    /// been called; otherwise admits the tile under `max_queued_bytes`
    /// (evicting finest non-pinned tiles outside the lock if needed).
    pub fn send(&self, item: (TileKey, DecodedTile)) -> Result<(), UploadChannelClosed> {
        let (key, decoded) = item;
        let mut to_drop = Vec::new();
        let closed = {
            let mut state = lock_unpoisoned(&self.shared);
            if state.closed {
                to_drop.push(decoded);
                true
            } else {
                state.insert_and_bound(key, decoded, &mut to_drop);
                false
            }
        };
        drop(to_drop);
        if closed {
            Err(UploadChannelClosed)
        } else {
            Ok(())
        }
    }
}

/// Byte-capped, coarsest-first budgeted GPU tile upload queue shared with
/// background decode producers via [`UploadSender`].
pub struct PendingUploads {
    shared: Arc<Mutex<SharedUploadQueue>>,
}

impl Default for PendingUploads {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for PendingUploads {
    fn drop(&mut self) {
        self.close();
    }
}

impl PendingUploads {
    /// Create a new upload queue bounded by `DEFAULT_MAX_QUEUED_UPLOAD_BYTES`
    /// (64 MiB).
    pub fn new() -> Self {
        Self::with_max_queued_bytes(DEFAULT_MAX_QUEUED_UPLOAD_BYTES)
    }

    /// Create a new upload queue with an explicit CPU queued byte ceiling
    /// enforced immediately on producer enqueue.
    pub fn with_max_queued_bytes(max_queued_bytes: usize) -> Self {
        Self {
            shared: Arc::new(Mutex::new(SharedUploadQueue::new(max_queued_bytes))),
        }
    }

    /// Clone a producer sender for a background tile worker or decode pool.
    /// Returns `None` once [`Self::close`] has been called.
    pub fn sender(&self) -> Option<UploadSender> {
        let state = lock_unpoisoned(&self.shared);
        (!state.closed).then(|| UploadSender {
            shared: Arc::clone(&self.shared),
        })
    }

    /// Update the set of sources whose coarsest root tiles stay pinned against
    /// enqueue-time overflow shedding.
    pub fn set_pinned_root_sources(&self, sources: &[SourceId]) {
        let mut to_drop = Vec::new();
        {
            let mut state = lock_unpoisoned(&self.shared);
            if !state.closed {
                state.pinned_root_sources.clear();
                state.pinned_root_sources.extend_from_slice(sources);
                state.enforce_cap(&mut to_drop);
            }
        }
        drop(to_drop);
    }

    /// Close the upload queue and release all queued payloads during
    /// deterministic shutdown so background producers exit promptly on send.
    pub fn close(&mut self) {
        let drained: Vec<DecodedTile> = {
            let mut state = lock_unpoisoned(&self.shared);
            state.closed = true;
            state.queued_bytes = 0;
            state.queued.drain().map(|(_, tile)| tile).collect()
        };
        drop(drained);
    }

    /// Total CPU bytes currently buffered across queued decoded tiles.
    pub fn queued_bytes(&self) -> usize {
        lock_unpoisoned(&self.shared).queued_bytes
    }

    /// Return whether `key` is currently decoded and waiting for GPU upload.
    pub fn contains(&self, key: &TileKey) -> bool {
        lock_unpoisoned(&self.shared).queued.contains_key(key)
    }

    /// Count queued decoded tiles belonging to `source`.
    pub fn count_for_source(&self, source: SourceId) -> usize {
        lock_unpoisoned(&self.shared)
            .queued
            .keys()
            .filter(|k| k.source == source)
            .count()
    }

    /// Discard any queued decoded tiles belonging to `source`.
    pub fn clear_source(&mut self, source: SourceId) {
        self.clear_where(|key| key.source == source);
    }

    /// Discard any queued decoded tiles matching `predicate`.
    pub fn clear_where<F>(&mut self, mut predicate: F)
    where
        F: FnMut(&TileKey) -> bool,
    {
        let mut to_drop = Vec::new();
        {
            let mut state = lock_unpoisoned(&self.shared);
            state.retain_keys(|k| !predicate(k), &mut to_drop);
        }
        drop(to_drop);
    }

    /// Prune cached/unwanted tiles (retaining roots configured via
    /// [`Self::set_pinned_root_sources`]), enforce `max_queued_bytes`, and
    /// upload coarsest-first within `(byte_budget, time_budget)` while
    /// protecting active-frame carriers and guaranteeing `>= 1` tile per frame
    /// whenever an admissible tile is queued.
    pub fn drain_and_upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        cache: &mut GpuTileCache,
        params: UploadPassParams<'_>,
    ) -> UploadBatchReport {
        let mut to_drop = Vec::new();
        let order: Vec<TileKey> = {
            let mut state = lock_unpoisoned(&self.shared);
            if state.closed {
                return UploadBatchReport::default();
            }

            let pinned = state.pinned_root_sources.clone();
            state.retain_keys(
                |key| {
                    if cache.contains(key) {
                        return false;
                    }
                    if params.wanted.is_empty() {
                        true
                    } else {
                        is_in_any_wanted(key, params.wanted) || is_pinned_root(key, &pinned)
                    }
                },
                &mut to_drop,
            );
            state.enforce_cap(&mut to_drop);

            let mut keys: Vec<TileKey> = state.queued.keys().copied().collect();
            keys.sort_by_key(TileKey::coarsest_first_key);
            keys
        };
        drop(to_drop);

        if order.is_empty() {
            return UploadBatchReport::default();
        }

        self.upload_ordered_tiles(device, queue, cache, &params, order)
    }

    fn upload_ordered_tiles(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        cache: &mut GpuTileCache,
        params: &UploadPassParams<'_>,
        order: Vec<TileKey>,
    ) -> UploadBatchReport {
        let start = Instant::now();
        let mut report = UploadBatchReport::default();
        let mut bytes_uploaded = 0usize;

        for key in order {
            // Anti-starvation: always upload at least 1 admissible tile per frame.
            if report.uploaded >= 1
                && (bytes_uploaded >= params.byte_budget || start.elapsed() >= params.time_budget)
            {
                break;
            }
            let decoded = {
                let mut state = lock_unpoisoned(&self.shared);
                state.remove_key(&key)
            };
            let Some(decoded) = decoded else {
                continue;
            };
            if cache.contains(&key) {
                continue;
            }

            let is_protected = |k: &TileKey| params.protected.is_some_and(|set| set.contains(k));
            match cache.upload_and_insert_protecting(device, queue, key, &decoded, is_protected) {
                UploadInsertOutcome::Inserted {
                    tile,
                    evicted: ev_count,
                } => {
                    bytes_uploaded = bytes_uploaded.saturating_add(tile.byte_size);
                    report.evicted = report.evicted.saturating_add(ev_count);
                    report.uploaded += 1;
                    report.record_source_tile_bytes(key.source, tile.byte_size);
                }
                UploadInsertOutcome::RefusedByProtection => {
                    let mut to_drop = Vec::new();
                    let retained = {
                        let mut state = lock_unpoisoned(&self.shared);
                        if !state.closed && !state.queued.contains_key(&key) {
                            state.insert_and_bound(key, decoded, &mut to_drop);
                            state.queued.contains_key(&key)
                        } else {
                            to_drop.push(decoded);
                            false
                        }
                    };
                    drop(to_drop);
                    if retained {
                        report.deferred_protected += 1;
                    }
                }
                UploadInsertOutcome::Rejected => {}
            }
        }

        report.remaining = lock_unpoisoned(&self.shared).queued.len();
        report.remaining_ready = report.remaining.saturating_sub(report.deferred_protected);
        report
    }
}

#[cfg(test)]
#[path = "upload_queue_tests.rs"]
mod tests;
