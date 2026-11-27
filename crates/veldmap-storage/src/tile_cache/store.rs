//! File-backed SQLite WAL [`TileDiskCache`] store, concurrent WAL reader pool,
//! and debounced LRU touch flushing.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, TryLockError};

use rusqlite::{Connection, Transaction, TransactionBehavior};
use veldmap_core::sync::{CancelToken, lock_unpoisoned};

use crate::StorageError;
use crate::paths::append_path_suffix;

use super::evict::{EvictOptions, evict_in_tx, reclaim_space};
use super::key::{DiskTileKey, EncodedKey, TileBlobEncoding, valid_source_key};
use super::queries::{
    TileUpserter, clear_all_in_tx, clear_source_in_tx, delete_entry_in_tx, flush_touches_in_tx,
    read_cache_meta_totals, select_tile_blob,
};
use super::readers::ReaderPool;
use super::schema::{is_corrupt_sqlite_error, open_and_init_file_db, recreate_file_db};
use super::touches::{MonotonicClock, TouchBuffer};

/// Maximum encoded payload size admitted for a single cached tile (`16 MiB`).
const MAX_CACHED_TILE_BYTES: usize = 16 * 1024 * 1024;

/// Observable disk cache utilization metrics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DiskCacheStats {
    pub tile_count: u64,
    pub payload_bytes: u64,
    pub file_bytes: u64,
    pub max_bytes: u64,
}

/// Opaque handle returned by [`TileDiskCache::get_encoded_entry`] allowing
/// callers to validate a payload before recording an LRU touch
/// ([`TileDiskCache::touch_entry`]) or safely purging a corrupt row
/// ([`TileDiskCache::delete_entry`]) without `ROWID`-reuse or concurrent
/// overwrite ABA races.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DiskTileEntry {
    tile_id: i64,
    row_last_used_ms: i64,
    read_ts: i64,
}

struct ValidatedPut<'a> {
    key: &'a DiskTileKey,
    enc: EncodedKey<'a>,
    encoding: TileBlobEncoding,
    payload: &'a [u8],
}

fn log_skipped_put(key: &DiskTileKey, new_bytes: u64, max_bytes: u64) {
    tracing::debug!(
        source_key = %key.source_key,
        level = key.level,
        x = key.x,
        y = key.y,
        payload_bytes = new_bytes,
        max_bytes,
        "Skipping disk cache put outside size bounds"
    );
}

fn validate_put_batch<'a>(
    entries: &'a [(&'a DiskTileKey, TileBlobEncoding, &'a [u8])],
    max_bytes: u64,
) -> Result<Vec<ValidatedPut<'a>>, StorageError> {
    let mut valid = Vec::with_capacity(entries.len());
    for &(key, encoding, payload) in entries {
        let enc = EncodedKey::from_key(key)?;
        let new_bytes = payload.len() as u64;
        if payload.is_empty()
            || payload.len() > MAX_CACHED_TILE_BYTES
            || max_bytes == 0
            || new_bytes > max_bytes
        {
            log_skipped_put(key, new_bytes, max_bytes);
            continue;
        }
        valid.push(ValidatedPut {
            key,
            enc,
            encoding,
            payload,
        });
    }
    Ok(valid)
}

/// Thread-safe, crash-safe SQLite WAL tile disk cache bounded by `max_bytes`.
///
/// Uses a serialized writer connection paired with a small pool of
/// `SQLITE_OPEN_READ_ONLY` WAL connections so concurrent `get()` and `stats()`
/// calls never stall behind background evictions or vacuums.
///
/// # Lock ordering
///
/// - `writer -> pending_touches` (inside `TouchBuffer`): whenever both locks
///   are held at the same time, `writer` is always acquired first. Touch
///   recording acquires `pending_touches` briefly as a standalone leaf lock and
///   releases it before calling `writer.try_lock()`.
/// - `reader_pool` is an independent leaf lock never acquired while holding
///   `writer` or `pending_touches` (and never acquiring them while held).
pub struct TileDiskCache {
    writer: Mutex<Connection>,
    reader_pool: ReaderPool,
    max_bytes: AtomicU64,
    clock: MonotonicClock,
    write_generation: AtomicU64,
    touches: TouchBuffer,
}

impl Drop for TileDiskCache {
    fn drop(&mut self) {
        self.reader_pool.clear();
        if self.touches.is_empty() {
            return;
        }
        let mut conn = lock_unpoisoned(&self.writer);
        self.flush_touches_best_effort(&mut conn);
    }
}

impl std::fmt::Debug for TileDiskCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TileDiskCache")
            .field("db_path", &self.reader_pool.db_path())
            .field("max_bytes", &self.max_bytes.load(Ordering::Acquire))
            .finish()
    }
}

impl TileDiskCache {
    /// Open or create a file-backed SQLite WAL tile cache at `path` bounded to
    /// `max_bytes` payload bytes. Automatically deletes and recreates a
    /// corrupted or older-schema ephemeral cache database.
    pub fn open(path: impl AsRef<Path>, max_bytes: u64) -> Result<Self, StorageError> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }

        let (mut conn, max_ts) = match open_and_init_file_db(&path) {
            Ok(pair) => pair,
            Err(err) if is_corrupt_sqlite_error(&err) => {
                tracing::warn!(
                    path = %path.display(),
                    error = %err,
                    "Corrupt ephemeral tile cache detected; deleting and recreating clean SQLite WAL cache"
                );
                recreate_file_db(&path)?
            }
            Err(err) => return Err(err.into()),
        };

        let (total_bytes, _) = read_cache_meta_totals(&conn)?;
        if total_bytes > max_bytes {
            let evicted_ids =
                Self::with_write_tx(&mut conn, |tx| evict_to_cap_in_tx(tx, max_bytes))?;
            if !evicted_ids.is_empty() {
                reclaim_space(&conn, false);
            }
        }

        Ok(Self {
            writer: Mutex::new(conn),
            reader_pool: ReaderPool::new(path),
            max_bytes: AtomicU64::new(max_bytes),
            clock: MonotonicClock::new(max_ts),
            write_generation: AtomicU64::new(0),
            touches: TouchBuffer::new(),
        })
    }

    /// Current configured disk cache byte budget (`0` means caching is disabled).
    pub fn max_bytes(&self) -> u64 {
        self.max_bytes.load(Ordering::Acquire)
    }

    /// Monotonic counter incremented on every cache mutation (`put`, `delete`,
    /// `clear_source`, `clear`, `set_max_bytes`), allowing UI stats polling to
    /// react to any subsystem's disk writes without coupling to networking.
    pub fn write_generation(&self) -> u64 {
        self.write_generation.load(Ordering::Acquire)
    }

    fn flush_touches_best_effort(&self, conn: &mut Connection) {
        if self.touches.is_empty() {
            return;
        }
        let _ = self.with_write_tx_flushing_touches(conn, |_| Ok(()));
    }

    /// Execute an `IMMEDIATE` write transaction without flushing pending LRU
    /// touches beforehand (used by non-evicting deletes such as
    /// [`Self::delete_entry`] and [`Self::clear_source`] so buffered touches
    /// neither mutate `last_used_ms` ahead of an ABA delete guard nor perform
    /// wasted `UPDATE`s on rows about to be deleted).
    fn with_write_tx<T>(
        conn: &mut Connection,
        f: impl FnOnce(&rusqlite::Transaction<'_>) -> Result<T, rusqlite::Error>,
    ) -> Result<T, StorageError> {
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let out = f(&tx)?;
        tx.commit()?;
        Ok(out)
    }

    /// Flush buffered LRU touches before executing an evicting write
    /// transaction so recently read tiles are protected from LRU eviction.
    /// Restores drained touches if the transaction fails or rolls back.
    fn with_write_tx_flushing_touches<T>(
        &self,
        conn: &mut Connection,
        f: impl FnOnce(&rusqlite::Transaction<'_>) -> Result<T, rusqlite::Error>,
    ) -> Result<T, StorageError> {
        let touches = self.touches.take_all();
        let res = Self::with_write_tx(conn, |tx| {
            flush_touches_in_tx(tx, &touches)?;
            f(tx)
        });
        if res.is_err() {
            self.touches.restore(touches);
        }
        res
    }

    /// Look up cached tile bytes without recording an LRU touch yet, returning
    /// an opaque [`DiskTileEntry`] handle so callers can validate the decoded
    /// payload before calling [`Self::touch_entry`] or safely purging a corrupt
    /// entry via [`Self::delete_entry`].
    pub fn get_encoded_entry(
        &self,
        key: &DiskTileKey,
        expected_encoding: Option<TileBlobEncoding>,
    ) -> Result<Option<(DiskTileEntry, TileBlobEncoding, Vec<u8>)>, StorageError> {
        let enc = EncodedKey::from_key(key)?;
        let Some(row) = self
            .reader_pool
            .with_reader(|conn| select_tile_blob(conn, &enc))?
        else {
            return Ok(None);
        };
        let Some(encoding) = TileBlobEncoding::from_i64(row.encoding_raw) else {
            return Ok(None);
        };
        if let Some(expected) = expected_encoding
            && encoding != expected
        {
            return Ok(None);
        }
        let read_ts = self.clock.next_ms();
        Ok(Some((
            DiskTileEntry {
                tile_id: row.tile_id,
                row_last_used_ms: row.last_used_ms,
                read_ts,
            },
            encoding,
            row.payload,
        )))
    }

    /// Record a debounced LRU touch for a previously validated [`DiskTileEntry`].
    pub fn touch_entry(&self, entry: DiskTileEntry) {
        if self.touches.record(entry.tile_id, entry.read_ts) {
            // Non-blocking opportunistic flush: if the writer is busy (e.g.
            // vacuum or put), the writer will pick up pending touches on its
            // next pass so concurrent WAL readers never stall.
            let maybe_conn = match self.writer.try_lock() {
                Ok(guard) => Some(guard),
                Err(TryLockError::Poisoned(poisoned)) => Some(poisoned.into_inner()),
                Err(TryLockError::WouldBlock) => None,
            };
            if let Some(mut conn) = maybe_conn {
                self.flush_touches_best_effort(&mut conn);
            }
        }
    }

    /// Delete the exact row version represented by `entry` (matching both
    /// `tile_id` and its `last_used_ms` at read time) so a stale reader can
    /// never delete a newer valid overwrite or reused SQLite `ROWID`.
    pub fn delete_entry(&self, entry: DiskTileEntry) -> Result<bool, StorageError> {
        let mut conn = lock_unpoisoned(&self.writer);
        let deleted = Self::with_write_tx(&mut conn, |tx| {
            delete_entry_in_tx(tx, entry.tile_id, entry.row_last_used_ms)
        })?;
        if deleted {
            self.touches.prune(&[entry.tile_id]);
            self.write_generation.fetch_add(1, Ordering::AcqRel);
        }
        Ok(deleted)
    }

    /// Insert or replace tile bytes only if `cancel` has not been triggered.
    ///
    /// Checks `cancel` both before and after acquiring the serialized SQLite
    /// writer lock so a concurrent `cancel + clear_source` reload sequence
    /// cannot be raced by an in-flight writer re-inserting stale tiles.
    pub fn put_encoded_if_uncancelled(
        &self,
        key: &DiskTileKey,
        encoding: TileBlobEncoding,
        payload: &[u8],
        cancel: &CancelToken,
    ) -> Result<(), StorageError> {
        self.put_encoded_batch_inner(&[(key, encoding, payload)], cancel)
    }

    /// Insert a batch of encoded tiles in a single SQLite `IMMEDIATE`
    /// transaction unless `cancel` has been triggered before or under the
    /// writer lock.
    ///
    /// Pre-batch tiles evict down to the 90% headroom target, while entries
    /// from the current batch only evict if the batch itself pushes totals
    /// above `max_bytes` (always protecting the newest entry).
    pub fn put_encoded_batch_if_uncancelled(
        &self,
        entries: &[(&DiskTileKey, TileBlobEncoding, &[u8])],
        cancel: &CancelToken,
    ) -> Result<(), StorageError> {
        self.put_encoded_batch_inner(entries, cancel)
    }

    fn put_encoded_batch_inner(
        &self,
        entries: &[(&DiskTileKey, TileBlobEncoding, &[u8])],
        cancel: &CancelToken,
    ) -> Result<(), StorageError> {
        if entries.is_empty() || cancel.is_cancelled() {
            return Ok(());
        }

        // Validate keys and filter empty/oversized payloads before locking `writer`.
        let pre_max_bytes = self.max_bytes.load(Ordering::Acquire);
        let valid = validate_put_batch(entries, pre_max_bytes)?;
        if valid.is_empty() {
            return Ok(());
        }

        let mut conn = lock_unpoisoned(&self.writer);
        if cancel.is_cancelled() {
            return Ok(());
        }
        // Re-check under `self.writer` so a concurrent `set_max_bytes()` cannot
        // lower the cap between the lock-free pre-check and transaction commit.
        let max_bytes = self.max_bytes.load(Ordering::Acquire);
        self.commit_put_batch_locked(&mut conn, &valid, max_bytes, valid.len() > 1)
    }

    fn commit_put_batch_locked(
        &self,
        conn: &mut Connection,
        valid: &[ValidatedPut<'_>],
        max_bytes: u64,
        is_multi_batch: bool,
    ) -> Result<(), StorageError> {
        if max_bytes == 0 {
            return Ok(());
        }

        let (evicted_ids, wrote_any) = self.with_write_tx_flushing_touches(conn, |tx| {
            let (mut total_bytes, mut tile_count) = read_cache_meta_totals(tx)?;
            let mut batch_start_ts: Option<i64> = None;
            let mut last_id: Option<i64> = None;

            {
                let mut upserter = TileUpserter::new(tx)?;
                for item in valid {
                    let new_bytes = item.payload.len() as u64;
                    if new_bytes > max_bytes {
                        log_skipped_put(item.key, new_bytes, max_bytes);
                        continue;
                    }
                    let ts_ms = self.clock.next_ms();
                    if is_multi_batch && batch_start_ts.is_none() {
                        batch_start_ts = Some(ts_ms);
                    }
                    let (inserted_id, old_bytes, count_delta) =
                        upserter.upsert(&item.enc, item.encoding, item.payload, ts_ms)?;
                    total_bytes = total_bytes
                        .saturating_sub(old_bytes)
                        .saturating_add(new_bytes);
                    tile_count = tile_count.saturating_add(count_delta);
                    last_id = Some(inserted_id);
                }
            }

            let Some(last_inserted) = last_id else {
                return Ok((Vec::new(), false));
            };

            let evicted_ids = evict_in_tx(
                tx,
                total_bytes,
                tile_count,
                max_bytes,
                EvictOptions {
                    protect_id: Some(last_inserted),
                    batch_start_ts,
                    always_write_meta: true,
                },
            )?;
            Ok((evicted_ids, true))
        })?;
        self.touches.prune(&evicted_ids);

        if wrote_any {
            self.write_generation.fetch_add(1, Ordering::AcqRel);
            if !evicted_ids.is_empty() {
                reclaim_space(conn, false);
            }
        }
        Ok(())
    }

    /// Update the configured byte budget, immediately evicting oldest tiles,
    /// reclaiming freed pages, and truncating WAL when utilization exceeds the
    /// new cap.
    pub fn set_max_bytes(&self, max_bytes: u64) -> Result<(), StorageError> {
        let mut conn = lock_unpoisoned(&self.writer);
        self.max_bytes.store(max_bytes, Ordering::Release);
        let evicted_ids =
            self.with_write_tx_flushing_touches(&mut conn, |tx| evict_to_cap_in_tx(tx, max_bytes))?;
        self.touches.prune(&evicted_ids);
        if !evicted_ids.is_empty() {
            reclaim_space(&conn, true);
        }
        self.write_generation.fetch_add(1, Ordering::AcqRel);
        Ok(())
    }

    /// Remove all cached tiles for a specific `source_key`, returning the
    /// number of deleted rows.
    pub fn clear_source(&self, source_key: &str) -> Result<usize, StorageError> {
        if !valid_source_key(source_key) {
            return Err(StorageError::InvalidSourceKey(source_key.to_owned()));
        }
        let mut conn = lock_unpoisoned(&self.writer);
        let deleted_ids = Self::with_write_tx(&mut conn, |tx| clear_source_in_tx(tx, source_key))?;
        self.touches.prune(&deleted_ids);
        let deleted = deleted_ids.len();
        if deleted > 0 {
            reclaim_space(&conn, true);
            self.write_generation.fetch_add(1, Ordering::AcqRel);
        }
        Ok(deleted)
    }

    /// Delete every cached tile and reclaim both main-DB pages and WAL space.
    pub fn clear(&self) -> Result<(), StorageError> {
        let mut conn = lock_unpoisoned(&self.writer);
        Self::with_write_tx(&mut conn, clear_all_in_tx)?;
        self.touches.clear();
        reclaim_space(&conn, true);
        self.write_generation.fetch_add(1, Ordering::AcqRel);
        Ok(())
    }

    /// Return current tile count, payload byte total (both O(1) from
    /// `cache_meta`), file size (including `-wal` sidecar), and cap.
    pub fn stats(&self) -> Result<DiskCacheStats, StorageError> {
        let (payload_bytes, tile_count) = self.reader_pool.with_reader(read_cache_meta_totals)?;
        let db_path = self.reader_pool.db_path();
        let main = std::fs::metadata(db_path).map_or(0, |m| m.len());
        let wal = std::fs::metadata(append_path_suffix(db_path, "-wal")).map_or(0, |m| m.len());
        Ok(DiskCacheStats {
            tile_count,
            payload_bytes,
            file_bytes: main.saturating_add(wal),
            max_bytes: self.max_bytes.load(Ordering::Acquire),
        })
    }
}

fn evict_to_cap_in_tx(tx: &Transaction<'_>, max_bytes: u64) -> Result<Vec<i64>, rusqlite::Error> {
    let (current_bytes, current_count) = read_cache_meta_totals(tx)?;
    evict_in_tx(
        tx,
        current_bytes,
        current_count,
        max_bytes,
        EvictOptions::default(),
    )
}

#[cfg(test)]
#[path = "tile_cache_tests.rs"]
mod tests;
