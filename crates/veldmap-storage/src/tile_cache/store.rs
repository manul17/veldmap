//! File-backed SQLite WAL [`TileDiskCache`] store, concurrent WAL reader pool,
//! and debounced LRU touch flushing.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::{Mutex, TryLockError};
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{Connection, TransactionBehavior};
use veldmap_core::sync::{CancelToken, lock_unpoisoned};

use crate::StorageError;

use super::evict::{evict_in_tx, reclaim_space};
use super::key::{DiskTileKey, EncodedKey, TileBlobEncoding, valid_source_key};
use super::queries::{
    clear_all_in_tx, clear_source_in_tx, delete_entry_in_tx, delete_tile_in_tx,
    flush_touches_in_tx, read_cache_meta_totals, select_tile_blob, set_cache_meta_totals,
    upsert_tile_in_tx, upsert_tile_raw_in_tx,
};
use super::schema::{
    is_corrupt_sqlite_error, open_and_init_file_db, open_read_only_connection, recreate_file_db,
    sidecar_path,
};

/// Maximum encoded payload size admitted for a single cached tile (`16 MiB`).
pub const MAX_CACHED_TILE_BYTES: usize = 16 * 1024 * 1024;

/// Number of buffered `get()` LRU hits before attempting a best-effort batched
/// flush (avoids a WAL write transaction on every cache read).
const TOUCH_FLUSH_THRESHOLD: usize = 64;

/// Maximum pooled read-only WAL connections retained for concurrent lookups.
const MAX_READ_POOL_CONNS: usize = 4;

/// Summary returned by [`TileDiskCache::put`] and [`TileDiskCache::set_max_bytes`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EvictionReport {
    pub evicted_tiles: usize,
    pub evicted_bytes: u64,
    pub total_bytes_after: u64,
}

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

fn flush_touches_best_effort(conn: &mut Connection, touches: &HashMap<i64, i64>) {
    if touches.is_empty() {
        return;
    }
    if let Ok(tx) = conn.transaction_with_behavior(TransactionBehavior::Immediate)
        && flush_touches_in_tx(&tx, touches).is_ok()
    {
        let _ = tx.commit();
    }
}

struct ReaderPool {
    db_path: PathBuf,
    conns: Mutex<Vec<Connection>>,
}

impl ReaderPool {
    fn new(db_path: PathBuf) -> Self {
        Self {
            db_path,
            conns: Mutex::new(Vec::with_capacity(MAX_READ_POOL_CONNS)),
        }
    }

    fn clear(&self) {
        lock_unpoisoned(&self.conns).clear();
    }

    fn with_reader<T>(
        &self,
        f: impl FnOnce(&Connection) -> Result<T, rusqlite::Error>,
    ) -> Result<T, StorageError> {
        let conn = {
            let mut guard = lock_unpoisoned(&self.conns);
            guard.pop()
        };
        let conn = match conn {
            Some(c) => c,
            None => open_read_only_connection(&self.db_path)?,
        };
        let result = f(&conn);
        if result.is_ok() {
            let mut guard = lock_unpoisoned(&self.conns);
            if guard.len() < MAX_READ_POOL_CONNS {
                guard.push(conn);
            }
        }
        Ok(result?)
    }
}

fn skipped_put_report(
    key: &DiskTileKey,
    new_bytes: u64,
    max_bytes: u64,
    total_bytes_after: u64,
) -> EvictionReport {
    tracing::debug!(
        source_key = %key.source_key,
        level = key.level,
        x = key.x,
        y = key.y,
        payload_bytes = new_bytes,
        max_bytes,
        "Skipping disk cache put outside size bounds"
    );
    EvictionReport {
        evicted_tiles: 0,
        evicted_bytes: 0,
        total_bytes_after,
    }
}

/// Thread-safe, crash-safe SQLite WAL tile disk cache bounded by `max_bytes`.
///
/// Uses a serialized writer connection paired with a small pool of
/// `SQLITE_OPEN_READ_ONLY` WAL connections so concurrent `get()` and `stats()`
/// calls never stall behind background evictions or vacuums.
pub struct TileDiskCache {
    writer: Mutex<Connection>,
    reader_pool: ReaderPool,
    db_path: PathBuf,
    max_bytes: AtomicU64,
    last_ts_ms: AtomicI64,
    write_generation: AtomicU64,
    pending_touches: Mutex<HashMap<i64, i64>>,
}

impl Drop for TileDiskCache {
    fn drop(&mut self) {
        self.reader_pool.clear();
        let touches = self.take_pending_touches();
        if touches.is_empty() {
            return;
        }
        let mut conn = lock_unpoisoned(&self.writer);
        flush_touches_best_effort(&mut conn, &touches);
    }
}

impl std::fmt::Debug for TileDiskCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TileDiskCache")
            .field("db_path", &self.db_path)
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

        let (conn, max_ts) = match open_and_init_file_db(&path) {
            Ok(pair) => pair,
            Err(err) if is_corrupt_sqlite_error(&err) => {
                tracing::warn!(
                    path = %path.display(),
                    error = %err,
                    "Corrupt ephemeral tile cache detected; deleting and recreating clean SQLite WAL cache"
                );
                recreate_file_db(&path)?
            }
            Err(err) => return Err(StorageError::Sqlite(err)),
        };

        let (total_bytes, _) = read_cache_meta_totals(&conn)?;
        let cache = Self {
            writer: Mutex::new(conn),
            reader_pool: ReaderPool::new(path.clone()),
            db_path: path,
            max_bytes: AtomicU64::new(max_bytes),
            last_ts_ms: AtomicI64::new(max_ts),
            write_generation: AtomicU64::new(0),
            pending_touches: Mutex::new(HashMap::new()),
        };
        if total_bytes > max_bytes {
            let _ = cache.evict_if_needed()?;
        }
        Ok(cache)
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

    fn next_monotonic_ts(&self) -> i64 {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_millis().min(i64::MAX as u128) as i64);
        let mut prev = self.last_ts_ms.load(Ordering::Relaxed);
        loop {
            let next = if now > prev {
                now
            } else {
                prev.saturating_add(1)
            };
            match self.last_ts_ms.compare_exchange_weak(
                prev,
                next,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => return next,
                Err(actual) => prev = actual,
            }
        }
    }

    fn record_touch_at(&self, tile_id: i64, ts: i64) {
        let should_try_flush = {
            let mut touches = lock_unpoisoned(&self.pending_touches);
            touches
                .entry(tile_id)
                .and_modify(|prev| *prev = (*prev).max(ts))
                .or_insert(ts);
            touches.len() >= TOUCH_FLUSH_THRESHOLD
        };
        // Non-blocking opportunistic flush: if the writer is busy (e.g. vacuum
        // or put), the writer will pick up `pending_touches` on its next pass
        // so concurrent WAL readers never stall.
        if should_try_flush {
            let maybe_conn = match self.writer.try_lock() {
                Ok(guard) => Some(guard),
                Err(TryLockError::Poisoned(poisoned)) => Some(poisoned.into_inner()),
                Err(TryLockError::WouldBlock) => None,
            };
            if let Some(mut conn) = maybe_conn {
                let touches = self.take_pending_touches();
                flush_touches_best_effort(&mut conn, &touches);
            }
        }
    }

    fn record_touch(&self, tile_id: i64) {
        self.record_touch_at(tile_id, self.next_monotonic_ts());
    }

    fn take_pending_touches(&self) -> HashMap<i64, i64> {
        std::mem::take(&mut *lock_unpoisoned(&self.pending_touches))
    }

    fn with_write_tx<T>(
        &self,
        conn: &mut Connection,
        f: impl FnOnce(&rusqlite::Transaction<'_>) -> Result<T, rusqlite::Error>,
    ) -> Result<T, StorageError> {
        let touches = self.take_pending_touches();
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        flush_touches_in_tx(&tx, &touches)?;
        let out = f(&tx)?;
        tx.commit()?;
        Ok(out)
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
        let read_ts = self.next_monotonic_ts();
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
        self.record_touch_at(entry.tile_id, entry.read_ts);
    }

    /// Delete the exact row version represented by `entry` (matching both
    /// `tile_id` and its `last_used_ms` at read time) so a stale reader can
    /// never delete a newer valid overwrite or reused SQLite `ROWID`.
    pub fn delete_entry(&self, entry: DiskTileEntry) -> Result<bool, StorageError> {
        let mut conn = lock_unpoisoned(&self.writer);
        let deleted = self.with_write_tx(&mut conn, |tx| {
            delete_entry_in_tx(tx, entry.tile_id, entry.row_last_used_ms)
        })?;
        if deleted {
            lock_unpoisoned(&self.pending_touches).remove(&entry.tile_id);
            self.write_generation.fetch_add(1, Ordering::AcqRel);
        }
        Ok(deleted)
    }

    /// Look up cached encoded image bytes (`TileBlobEncoding::EncodedImage`)
    /// for `key`, recording a debounced LRU touch on hit. Returns `Ok(None)` if
    /// the stored entry uses a different encoding (e.g. `Qoi` or `Lz4Raw`).
    pub fn get(&self, key: &DiskTileKey) -> Result<Option<Vec<u8>>, StorageError> {
        let Some((entry, _, payload)) =
            self.get_encoded_entry(key, Some(TileBlobEncoding::EncodedImage))?
        else {
            return Ok(None);
        };
        self.record_touch(entry.tile_id);
        Ok(Some(payload))
    }

    /// Look up cached tile bytes and their [`TileBlobEncoding`] tag for `key`,
    /// recording a debounced LRU touch on hit.
    pub fn get_encoded(
        &self,
        key: &DiskTileKey,
    ) -> Result<Option<(TileBlobEncoding, Vec<u8>)>, StorageError> {
        let Some((entry, encoding, payload)) = self.get_encoded_entry(key, None)? else {
            return Ok(None);
        };
        self.record_touch(entry.tile_id);
        Ok(Some((encoding, payload)))
    }

    /// Insert or replace encoded image tile bytes (`TileBlobEncoding::EncodedImage`)
    /// for `key`, enforcing `payload_bytes <= max_bytes` via O(log N) per-tile
    /// LRU eviction and `PRAGMA incremental_vacuum`.
    pub fn put(&self, key: &DiskTileKey, payload: &[u8]) -> Result<EvictionReport, StorageError> {
        self.put_encoded(key, TileBlobEncoding::EncodedImage, payload)
    }

    /// Insert or replace tile bytes with an explicit [`TileBlobEncoding`] tag.
    pub fn put_encoded(
        &self,
        key: &DiskTileKey,
        encoding: TileBlobEncoding,
        payload: &[u8],
    ) -> Result<EvictionReport, StorageError> {
        self.put_encoded_inner(key, encoding, payload, None)
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
    ) -> Result<EvictionReport, StorageError> {
        self.put_encoded_inner(key, encoding, payload, Some(cancel))
    }

    fn put_encoded_inner(
        &self,
        key: &DiskTileKey,
        encoding: TileBlobEncoding,
        payload: &[u8],
        cancel: Option<&CancelToken>,
    ) -> Result<EvictionReport, StorageError> {
        if cancel.is_some_and(CancelToken::is_cancelled) {
            return Ok(EvictionReport::default());
        }
        let enc = EncodedKey::from_key(key)?;
        let pre_max_bytes = self.max_bytes.load(Ordering::Acquire);
        let new_bytes = payload.len() as u64;
        if payload.is_empty()
            || payload.len() > MAX_CACHED_TILE_BYTES
            || pre_max_bytes == 0
            || new_bytes > pre_max_bytes
        {
            let (total_bytes_after, _) = self.reader_pool.with_reader(read_cache_meta_totals)?;
            return Ok(skipped_put_report(
                key,
                new_bytes,
                pre_max_bytes,
                total_bytes_after,
            ));
        }

        let mut conn = lock_unpoisoned(&self.writer);
        if cancel.is_some_and(CancelToken::is_cancelled) {
            return Ok(EvictionReport::default());
        }
        // Re-check under `self.writer` so a concurrent `set_max_bytes()` cannot
        // lower the cap between the lock-free pre-check and transaction commit.
        let max_bytes = self.max_bytes.load(Ordering::Acquire);
        if max_bytes == 0 || new_bytes > max_bytes {
            let (total_bytes_after, _) = read_cache_meta_totals(&conn)?;
            return Ok(skipped_put_report(
                key,
                new_bytes,
                max_bytes,
                total_bytes_after,
            ));
        }

        let ts_ms = self.next_monotonic_ts();
        let outcome = self.with_write_tx(&mut conn, |tx| {
            let (inserted_id, updated_total, updated_count) =
                upsert_tile_in_tx(tx, &enc, encoding, payload, ts_ms)?;
            evict_in_tx(
                tx,
                updated_total,
                updated_count,
                max_bytes,
                Some(inserted_id),
                None,
            )
        })?;
        self.prune_pending_touches(&outcome.evicted_ids);

        self.write_generation.fetch_add(1, Ordering::AcqRel);
        if outcome.report.evicted_tiles > 0 {
            reclaim_space(&conn, false);
        }
        Ok(outcome.report)
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
    ) -> Result<EvictionReport, StorageError> {
        if entries.is_empty() || cancel.is_cancelled() {
            return Ok(EvictionReport::default());
        }
        let pre_max_bytes = self.max_bytes.load(Ordering::Acquire);
        if pre_max_bytes == 0 {
            return Ok(EvictionReport::default());
        }

        // Validate keys and filter oversized/empty payloads before locking `writer`.
        let mut valid = Vec::with_capacity(entries.len());
        for &(key, encoding, payload) in entries {
            let enc = EncodedKey::from_key(key)?;
            let sz = payload.len() as u64;
            if !payload.is_empty() && payload.len() <= MAX_CACHED_TILE_BYTES && sz <= pre_max_bytes
            {
                valid.push((enc, encoding, payload));
            }
        }
        if valid.is_empty() {
            return Ok(EvictionReport::default());
        }

        let mut conn = lock_unpoisoned(&self.writer);
        if cancel.is_cancelled() {
            return Ok(EvictionReport::default());
        }
        let max_bytes = self.max_bytes.load(Ordering::Acquire);
        if max_bytes == 0 {
            return Ok(EvictionReport::default());
        }

        let outcome = self.with_write_tx(&mut conn, |tx| {
            let (mut total_bytes, mut tile_count) = read_cache_meta_totals(tx)?;
            let mut batch_start_ts: Option<i64> = None;
            let mut last_id: Option<i64> = None;

            for (enc, encoding, payload) in &valid {
                let new_bytes = payload.len() as u64;
                if new_bytes > max_bytes {
                    continue;
                }
                let ts_ms = self.next_monotonic_ts();
                if batch_start_ts.is_none() {
                    batch_start_ts = Some(ts_ms);
                }
                let (inserted_id, old_bytes, count_delta) =
                    upsert_tile_raw_in_tx(tx, enc, *encoding, payload, ts_ms)?;
                total_bytes = total_bytes
                    .saturating_sub(old_bytes)
                    .saturating_add(new_bytes);
                tile_count = tile_count.saturating_add(count_delta);
                last_id = Some(inserted_id);
            }

            let Some(last_inserted) = last_id else {
                return Ok(super::evict::EvictionOutcome {
                    report: EvictionReport {
                        evicted_tiles: 0,
                        evicted_bytes: 0,
                        total_bytes_after: total_bytes,
                    },
                    evicted_ids: Vec::new(),
                });
            };

            set_cache_meta_totals(tx, total_bytes, tile_count)?;
            evict_in_tx(
                tx,
                total_bytes,
                tile_count,
                max_bytes,
                Some(last_inserted),
                batch_start_ts,
            )
        })?;
        self.prune_pending_touches(&outcome.evicted_ids);

        self.write_generation.fetch_add(1, Ordering::AcqRel);
        if outcome.report.evicted_tiles > 0 {
            reclaim_space(&conn, false);
        }
        Ok(outcome.report)
    }

    fn prune_pending_touches(&self, tile_ids: &[i64]) {
        if tile_ids.is_empty() {
            return;
        }
        let mut touches = lock_unpoisoned(&self.pending_touches);
        for id in tile_ids {
            touches.remove(id);
        }
    }

    /// Delete one tile entry (e.g. if a cached payload fails decode validation).
    pub fn delete(&self, key: &DiskTileKey) -> Result<bool, StorageError> {
        let enc = EncodedKey::from_key(key)?;
        let mut conn = lock_unpoisoned(&self.writer);
        let deleted_id = self.with_write_tx(&mut conn, |tx| delete_tile_in_tx(tx, &enc))?;
        let Some(tile_id) = deleted_id else {
            return Ok(false);
        };
        lock_unpoisoned(&self.pending_touches).remove(&tile_id);
        self.write_generation.fetch_add(1, Ordering::AcqRel);
        Ok(true)
    }

    /// Update the configured byte budget, immediately evicting oldest tiles,
    /// reclaiming freed pages, and truncating WAL when utilization exceeds the
    /// new cap.
    pub fn set_max_bytes(&self, max_bytes: u64) -> Result<EvictionReport, StorageError> {
        let mut conn = lock_unpoisoned(&self.writer);
        self.max_bytes.store(max_bytes, Ordering::Release);
        let report = self.evict_with_vacuum_locked(&mut conn, max_bytes, true)?;
        self.write_generation.fetch_add(1, Ordering::AcqRel);
        Ok(report)
    }

    /// Enforce `total_bytes <= max_bytes` on the current database state.
    fn evict_if_needed(&self) -> Result<EvictionReport, StorageError> {
        let mut conn = lock_unpoisoned(&self.writer);
        let max_bytes = self.max_bytes.load(Ordering::Acquire);
        let report = self.evict_with_vacuum_locked(&mut conn, max_bytes, false)?;
        if report.evicted_tiles > 0 {
            self.write_generation.fetch_add(1, Ordering::AcqRel);
        }
        Ok(report)
    }

    fn evict_with_vacuum_locked(
        &self,
        conn: &mut Connection,
        max_bytes: u64,
        full_vacuum: bool,
    ) -> Result<EvictionReport, StorageError> {
        let outcome = self.with_write_tx(conn, |tx| {
            let (current_bytes, current_count) = read_cache_meta_totals(tx)?;
            evict_in_tx(tx, current_bytes, current_count, max_bytes, None, None)
        })?;
        self.prune_pending_touches(&outcome.evicted_ids);
        if outcome.report.evicted_tiles > 0 {
            reclaim_space(conn, full_vacuum);
        }
        Ok(outcome.report)
    }

    /// Remove all cached tiles for a specific `source_key`, returning the
    /// number of deleted rows.
    pub fn clear_source(&self, source_key: &str) -> Result<usize, StorageError> {
        if !valid_source_key(source_key) {
            return Err(StorageError::InvalidSourceKey(source_key.to_owned()));
        }
        let mut conn = lock_unpoisoned(&self.writer);
        let deleted_ids = self.with_write_tx(&mut conn, |tx| clear_source_in_tx(tx, source_key))?;
        self.prune_pending_touches(&deleted_ids);
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
        lock_unpoisoned(&self.pending_touches).clear();
        {
            let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
            clear_all_in_tx(&tx)?;
            tx.commit()?;
        }
        reclaim_space(&conn, true);
        self.write_generation.fetch_add(1, Ordering::AcqRel);
        Ok(())
    }

    /// Return current tile count, payload byte total (both O(1) from
    /// `cache_meta`), file size (including `-wal` sidecar), and cap.
    pub fn stats(&self) -> Result<DiskCacheStats, StorageError> {
        let (payload_bytes, tile_count) = self.reader_pool.with_reader(read_cache_meta_totals)?;
        let main = std::fs::metadata(&self.db_path).map_or(0, |m| m.len());
        let wal = std::fs::metadata(sidecar_path(&self.db_path, "-wal")).map_or(0, |m| m.len());
        Ok(DiskCacheStats {
            tile_count,
            payload_bytes,
            file_bytes: main.saturating_add(wal),
            max_bytes: self.max_bytes.load(Ordering::Acquire),
        })
    }
}

#[cfg(test)]
#[path = "tile_cache_tests.rs"]
mod tests;
