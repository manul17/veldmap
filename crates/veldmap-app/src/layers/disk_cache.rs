//! Format-dispatched lossless disk-cache helpers (`TileBlobEncoding::Qoi` for
//! `Rgba8UnormSrgb` display tiles; byte-shuffled `TileBlobEncoding::Lz4Raw` for
//! `R32Float` / `R8Uint` / `R16Uint` scientific & categorical tiles) and the
//! offloaded [`TileWriteWorker`] / [`TileWriteHandle`] persistence pipeline.

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use veldmap_core::sync::{CancelToken, lock_unpoisoned, wait_unpoisoned};
use veldmap_core::tile::{DecodedTile, PixelFormat, SourceId};
use veldmap_raster::{
    MAX_LZ4_RAW_DECODED_BYTES, MAX_QOI_DECODED_BYTES, decode_lz4_raw_tile, decode_qoi_rgba8,
    encode_lz4_raw_tile, encode_qoi_rgba8, generate_tile_mipmaps,
};
use veldmap_storage::{DiskTileKey, TileBlobEncoding, TileDiskCache};

/// Maximum tiles buffered before committing a single SQLite batch write during
/// `ServeMode::Pass` or offloaded pointwise tile persistence.
const PASS_BATCH_MAX_TILES: usize = 16;

/// Maximum encoded bytes buffered before flushing a batch write (`4 MiB`).
const PASS_BATCH_MAX_BYTES: usize = 4 * 1024 * 1024;

/// Maximum pending mip-0 tiles queued for [`TileWriteWorker`] before
/// non-blocking enqueue drops excess persistence writes (never stalling
/// `fast_pool`).
const TILE_WRITE_QUEUE_CAPACITY: usize = 64;

/// Maximum coalescing window waited by [`TileWriteWorker`] after receiving the
/// first tile of a batch before committing to SQLite (`50 ms`).
const TILE_WRITE_COALESCE_WINDOW: Duration = Duration::from_millis(50);

fn encoding_for_format(format: PixelFormat) -> TileBlobEncoding {
    match format {
        PixelFormat::Rgba8UnormSrgb => TileBlobEncoding::Qoi,
        PixelFormat::R32Float | PixelFormat::R8Uint | PixelFormat::R16Uint => {
            TileBlobEncoding::Lz4Raw
        }
    }
}

fn encode_tile_blob(tile: &DecodedTile) -> Option<(TileBlobEncoding, Vec<u8>)> {
    let res = match tile.format() {
        PixelFormat::Rgba8UnormSrgb => encode_qoi_rgba8(
            tile.width(),
            tile.height(),
            tile.data(),
            MAX_QOI_DECODED_BYTES,
        )
        .map(|bytes| (TileBlobEncoding::Qoi, bytes)),
        PixelFormat::R32Float | PixelFormat::R8Uint | PixelFormat::R16Uint => {
            encode_lz4_raw_tile(tile, MAX_LZ4_RAW_DECODED_BYTES)
                .map(|bytes| (TileBlobEncoding::Lz4Raw, bytes))
        }
    };
    match res {
        Ok(encoded) => Some(encoded),
        Err(err) => {
            tracing::warn!(
                format = ?tile.format(),
                width = tile.width(),
                height = tile.height(),
                error = %err,
                "Failed to encode overlay tile for disk cache"
            );
            None
        }
    }
}

/// Look up and decode a cached overlay tile (`Qoi` or `Lz4Raw`) from `disk`,
/// recording its debounced LRU touch ONLY after validation succeeds and
/// regenerating CPU mip chains (`Rgba8UnormSrgb` / `R32Float`).
pub(super) fn load_cached_tile(
    disk: &TileDiskCache,
    disk_key: &DiskTileKey,
) -> Option<DecodedTile> {
    let expected_encoding = encoding_for_format(disk_key.format);
    let (entry, _, payload) = disk
        .get_encoded_entry(disk_key, Some(expected_encoding))
        .ok()??;

    let decoded = match disk_key.format {
        PixelFormat::Rgba8UnormSrgb => {
            decode_qoi_rgba8(&payload, MAX_QOI_DECODED_BYTES).and_then(|(w, h, rgba)| {
                DecodedTile::try_new(w, h, PixelFormat::Rgba8UnormSrgb, rgba)
                    .map_err(|e| veldmap_raster::RasterError::Decode(e.to_string()))
            })
        }
        PixelFormat::R32Float | PixelFormat::R8Uint | PixelFormat::R16Uint => {
            decode_lz4_raw_tile(&payload, disk_key.format, MAX_LZ4_RAW_DECODED_BYTES)
        }
    };

    match decoded {
        Ok(mut tile) => {
            disk.touch_entry(entry);
            generate_tile_mipmaps(&mut tile);
            Some(tile)
        }
        Err(err) => {
            tracing::warn!(
                source_key = %disk_key.source_key,
                format = ?disk_key.format,
                level = disk_key.level,
                x = disk_key.x,
                y = disk_key.y,
                error = %err,
                "Purging corrupt cached overlay tile entry"
            );
            let _ = disk.delete_entry(entry);
            None
        }
    }
}

/// Bounded write buffer amortizing SQLite WAL commits across sequential-pass
/// tile emissions (`Qoi` or `Lz4Raw`).
pub(super) struct TileWriteBatch {
    entries: Vec<(DiskTileKey, TileBlobEncoding, Vec<u8>)>,
    bytes: usize,
}

impl TileWriteBatch {
    pub(super) fn new() -> Self {
        Self {
            entries: Vec::with_capacity(PASS_BATCH_MAX_TILES),
            bytes: 0,
        }
    }

    pub(super) fn push_and_maybe_flush(
        &mut self,
        disk: &TileDiskCache,
        disk_key: DiskTileKey,
        tile: &DecodedTile,
        cancel: &CancelToken,
    ) {
        if disk.max_bytes() == 0 || cancel.is_cancelled() {
            return;
        }
        let Some((encoding, encoded)) = encode_tile_blob(tile) else {
            return;
        };
        self.bytes = self.bytes.saturating_add(encoded.len());
        self.entries.push((disk_key, encoding, encoded));
        if self.entries.len() >= PASS_BATCH_MAX_TILES || self.bytes >= PASS_BATCH_MAX_BYTES {
            self.flush(disk, cancel);
        }
    }

    pub(super) fn flush(&mut self, disk: &TileDiskCache, cancel: &CancelToken) {
        if self.entries.is_empty() {
            return;
        }
        if !cancel.is_cancelled() && disk.max_bytes() > 0 {
            let refs: Vec<(&DiskTileKey, TileBlobEncoding, &[u8])> = self
                .entries
                .iter()
                .map(|(k, enc, bytes)| (k, *enc, bytes.as_slice()))
                .collect();
            let _ = disk.put_encoded_batch_if_uncancelled(&refs, cancel);
        }
        self.clear();
    }

    pub(super) fn clear(&mut self) {
        self.entries.clear();
        self.bytes = 0;
    }
}

struct PendingTileWrite {
    source_id: SourceId,
    disk_key: DiskTileKey,
    tile: DecodedTile,
    write_gen: u64,
    source_cancel: CancelToken,
}

struct WriteQueueInner {
    items: VecDeque<PendingTileWrite>,
    closed: bool,
}

type SharedWriteQueue = Arc<(Mutex<WriteQueueInner>, Condvar)>;

/// Non-blocking handle for offloading pointwise tile QOI/LZ4 encoding and
/// batched SQLite disk-cache persistence off `fast_pool`.
#[derive(Clone)]
pub(super) struct TileWriteHandle {
    shared: SharedWriteQueue,
}

impl TileWriteHandle {
    /// Enqueue a pre-cloned mip-0 [`DecodedTile`] for off-thread encoding and
    /// SQLite persistence. Strictly non-blocking: drops the write if the
    /// bounded queue (`64` tiles) is saturated, closed, or `source_cancel` fired.
    pub(super) fn enqueue_mip0_tile(
        &self,
        source_id: SourceId,
        disk_key: DiskTileKey,
        mip0: DecodedTile,
        write_gen: u64,
        source_cancel: CancelToken,
    ) {
        if source_cancel.is_cancelled() {
            return;
        }
        let (lock, cvar) = &*self.shared;
        let mut queue = lock_unpoisoned(lock);
        if queue.closed || queue.items.len() >= TILE_WRITE_QUEUE_CAPACITY {
            return;
        }
        queue.items.push_back(PendingTileWrite {
            source_id,
            disk_key,
            tile: mip0,
            write_gen,
            source_cancel,
        });
        cvar.notify_one();
    }

    /// Immediately discard any queued but unencoded writes for `source_id`
    /// (paired with cancelling the source's write token before `clear_source`
    /// so pre-reload tiles can never race into SQLite).
    pub(super) fn purge_source(&self, source_id: SourceId) {
        let (lock, _) = &*self.shared;
        let mut queue = lock_unpoisoned(lock);
        queue.items.retain(|item| item.source_id != source_id);
    }
}

/// Dedicated background worker (`"veldmap-tile-write"`) that batches pointwise
/// overlay tile QOI/LZ4 encoding and SQLite commits off `fast_pool`, with zero
/// idle wakeups and deterministic shutdown joining.
pub(super) struct TileWriteWorker {
    shared: SharedWriteQueue,
    handle: Option<JoinHandle<()>>,
}

impl TileWriteWorker {
    pub(super) fn spawn(
        disk: Arc<TileDiskCache>,
        shutdown: CancelToken,
    ) -> (Self, TileWriteHandle) {
        let shared = Arc::new((
            Mutex::new(WriteQueueInner {
                items: VecDeque::with_capacity(PASS_BATCH_MAX_TILES),
                closed: false,
            }),
            Condvar::new(),
        ));
        let worker_shared = Arc::clone(&shared);
        let builder = thread::Builder::new().name("veldmap-tile-write".to_owned());
        let handle = match builder.spawn(move || {
            run_tile_write_worker(worker_shared, disk, shutdown);
        }) {
            Ok(h) => Some(h),
            Err(err) => {
                tracing::warn!(
                    error = %err,
                    "Failed to spawn veldmap-tile-write worker; pointwise disk caching disabled"
                );
                let (lock, _) = &*shared;
                lock_unpoisoned(lock).closed = true;
                None
            }
        };
        let producer = TileWriteHandle {
            shared: Arc::clone(&shared),
        };
        (Self { shared, handle }, producer)
    }

    pub(super) fn close(&self) {
        let (lock, cvar) = &*self.shared;
        let mut queue = lock_unpoisoned(lock);
        queue.closed = true;
        cvar.notify_all();
    }

    pub(super) fn join(&mut self) {
        self.close();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for TileWriteWorker {
    fn drop(&mut self) {
        self.join();
    }
}

fn run_tile_write_worker(
    shared: SharedWriteQueue,
    disk: Arc<TileDiskCache>,
    shutdown: CancelToken,
) {
    let mut batch: Vec<PendingTileWrite> = Vec::with_capacity(PASS_BATCH_MAX_TILES);
    let mut encoded: Vec<(DiskTileKey, TileBlobEncoding, Vec<u8>, u64, CancelToken)> =
        Vec::with_capacity(PASS_BATCH_MAX_TILES);

    loop {
        batch.clear();
        {
            let (lock, cvar) = &*shared;
            let mut queue = lock_unpoisoned(lock);
            while queue.items.is_empty() && !queue.closed && !shutdown.is_cancelled() {
                queue = wait_unpoisoned(cvar, queue);
            }
            if shutdown.is_cancelled() || (queue.closed && queue.items.is_empty()) {
                return;
            }
            if let Some(first) = queue.items.pop_front() {
                batch.push(first);
            }
            let deadline = Instant::now() + TILE_WRITE_COALESCE_WINDOW;
            while batch.len() < PASS_BATCH_MAX_TILES && !shutdown.is_cancelled() {
                while batch.len() < PASS_BATCH_MAX_TILES {
                    if let Some(next) = queue.items.pop_front() {
                        batch.push(next);
                    } else {
                        break;
                    }
                }
                if batch.len() >= PASS_BATCH_MAX_TILES || queue.closed {
                    break;
                }
                let Some(rem) = deadline.checked_duration_since(Instant::now()) else {
                    break;
                };
                if rem.is_zero() {
                    break;
                }
                let (next_guard, wait_res) = match cvar.wait_timeout(queue, rem) {
                    Ok(pair) => pair,
                    Err(poisoned) => poisoned.into_inner(),
                };
                queue = next_guard;
                if wait_res.timed_out() {
                    while batch.len() < PASS_BATCH_MAX_TILES {
                        if let Some(next) = queue.items.pop_front() {
                            batch.push(next);
                        } else {
                            break;
                        }
                    }
                    break;
                }
            }
        }

        if shutdown.is_cancelled() || disk.max_bytes() == 0 {
            continue;
        }

        encoded.clear();
        let mut encoded_bytes = 0usize;
        for item in batch.drain(..) {
            if shutdown.is_cancelled() || item.source_cancel.is_cancelled() {
                continue;
            }
            if let Some((enc, bytes)) = encode_tile_blob(&item.tile) {
                encoded_bytes = encoded_bytes.saturating_add(bytes.len());
                encoded.push((
                    item.disk_key,
                    enc,
                    bytes,
                    item.write_gen,
                    item.source_cancel,
                ));
                if encoded_bytes >= PASS_BATCH_MAX_BYTES {
                    flush_encoded_batch(&disk, &mut encoded, &shutdown);
                    encoded_bytes = 0;
                }
            }
        }
        flush_encoded_batch(&disk, &mut encoded, &shutdown);
    }
}

fn flush_encoded_batch(
    disk: &TileDiskCache,
    encoded: &mut Vec<(DiskTileKey, TileBlobEncoding, Vec<u8>, u64, CancelToken)>,
    shutdown: &CancelToken,
) {
    if encoded.is_empty() || shutdown.is_cancelled() || disk.max_bytes() == 0 {
        encoded.clear();
        return;
    }
    // Group by `(write_gen, source_cancel)` so each source generation passes
    // its own child-of-shutdown token into `put_encoded_batch_if_uncancelled`,
    // enforcing reload/purge cancellation strictly under SQLite's writer lock.
    encoded.sort_unstable_by_key(|(_, _, _, write_gen, _)| *write_gen);
    let mut start = 0usize;
    while start < encoded.len() {
        let write_gen = encoded[start].3;
        let mut end = start + 1;
        while end < encoded.len() && encoded[end].3 == write_gen {
            end += 1;
        }
        let source_cancel = &encoded[start].4;
        if !source_cancel.is_cancelled() {
            let refs: Vec<(&DiskTileKey, TileBlobEncoding, &[u8])> = encoded[start..end]
                .iter()
                .map(|(k, enc, bytes, _, _)| (k, *enc, bytes.as_slice()))
                .collect();
            let _ = disk.put_encoded_batch_if_uncancelled(&refs, source_cancel);
        }
        start = end;
    }
    encoded.clear();
}
