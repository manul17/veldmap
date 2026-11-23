//! Crash-safe, multi-process-safe SQLite WAL tile disk cache with O(log N)
//! per-tile LRU eviction, vertically partitioned `tile_meta`/`tile_blobs`
//! tables, concurrent WAL read connections, debounced LRU touch flushing, and
//! `incremental_vacuum`.
//!
//! Storing tiles in a single WAL-mode SQLite database avoids both whole-source
//! eviction thrashing and loose-file `relatime`/Windows Defender storms.
//! Separating `tile_meta` (`last_used_ms`, dimensions/keys) from `tile_blobs`
//! (`payload BLOB`) ensures batched `get()` LRU touch updates never trigger
//! SQLite `OP_MakeRecord` overflow-page rewrites of 256 KiB–1 MiB scientific
//! tile blobs. Because tile caches are ephemeral, corrupted or older-schema
//! cache files are recreated cleanly on startup.

mod evict;
mod key;
mod schema;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{Connection, OptionalExtension, TransactionBehavior, params};
use veldmap_core::io::lock_unpoisoned;

use crate::StorageError;

use self::evict::{evict_in_tx, reclaim_space};
use self::key::EncodedKey;
use self::schema::{
    SELECT_META_ID_BYTES_SQL, SELECT_TILE_BLOB_SQL, UPSERT_META_SQL, init_connection,
    is_corrupt_sqlite_error, open_and_init_file_db, open_read_only_connection, remove_sqlite_files,
    sidecar_path,
};

pub use self::key::{DiskTileKey, TileBlobEncoding, valid_source_key};

/// Default disk cache cap in mebibytes (`512 MiB`).
pub const DEFAULT_DISK_CACHE_MAX_MB: u64 = 512;

/// Minimum configurable disk cache cap in mebibytes (`16 MiB`).
pub const MIN_DISK_CACHE_MAX_MB: u64 = 16;

/// Maximum configurable disk cache cap in mebibytes (`16 GiB`).
pub const MAX_DISK_CACHE_MAX_MB: u64 = 16 * 1024;

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

fn flush_touches_in_tx(
    tx: &rusqlite::Transaction<'_>,
    touches: &HashMap<i64, i64>,
) -> Result<(), rusqlite::Error> {
    if touches.is_empty() {
        return Ok(());
    }
    // Updates `tile_meta` only (never `tile_blobs`), so read-hit LRU touches
    // never rewrite BLOB overflow pages. Guarded with `last_used_ms < ?1` so a
    // buffered touch can never roll back a newer timestamp on a reused tile_id.
    let mut stmt = tx.prepare(
        "UPDATE tile_meta SET last_used_ms = ?1 WHERE tile_id = ?2 AND last_used_ms < ?1",
    )?;
    for (&tile_id, &ts) in touches {
        stmt.execute(params![ts, tile_id])?;
    }
    Ok(())
}

fn current_total_bytes(conn: &Connection) -> Result<u64, rusqlite::Error> {
    let total: i64 =
        conn.query_row("SELECT total_bytes FROM cache_meta WHERE id = 1", [], |r| {
            r.get(0)
        })?;
    Ok(total.max(0) as u64)
}

/// Thread-safe, crash-safe SQLite WAL tile disk cache bounded by `max_bytes`.
///
/// File-backed caches use a serialized writer connection paired with a small
/// pool of `SQLITE_OPEN_READ_ONLY` WAL connections so concurrent `get()` and
/// `stats()` calls never stall behind background evictions or vacuums.
pub struct TileDiskCache {
    writer: Mutex<Connection>,
    read_pool: Option<Mutex<Vec<Connection>>>,
    db_path: Option<PathBuf>,
    max_bytes: AtomicU64,
    last_ts_ms: AtomicI64,
    write_generation: AtomicU64,
    pending_touches: Mutex<HashMap<i64, i64>>,
}

impl Drop for TileDiskCache {
    fn drop(&mut self) {
        if let Some(pool) = &self.read_pool {
            lock_unpoisoned(pool).clear();
        }
        let touches = std::mem::take(&mut *lock_unpoisoned(&self.pending_touches));
        if touches.is_empty() {
            return;
        }
        let mut conn = lock_unpoisoned(&self.writer);
        if let Ok(tx) = conn.transaction_with_behavior(TransactionBehavior::Immediate)
            && flush_touches_in_tx(&tx, &touches).is_ok()
        {
            let _ = tx.commit();
        }
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
                remove_sqlite_files(&path);
                let fresh = Connection::open(&path)?;
                let max_ts = init_connection(&fresh, true, true)?;
                (fresh, max_ts)
            }
            Err(err) => return Err(StorageError::Sqlite(err)),
        };

        let cache = Self {
            writer: Mutex::new(conn),
            read_pool: Some(Mutex::new(Vec::with_capacity(MAX_READ_POOL_CONNS))),
            db_path: Some(path),
            max_bytes: AtomicU64::new(max_bytes),
            last_ts_ms: AtomicI64::new(max_ts),
            write_generation: AtomicU64::new(0),
            pending_touches: Mutex::new(HashMap::new()),
        };
        let _ = cache.evict_if_needed()?;
        Ok(cache)
    }

    /// Open an isolated in-memory SQLite tile cache for unit tests.
    #[cfg(test)]
    pub(crate) fn open_in_memory(max_bytes: u64) -> Result<Self, StorageError> {
        let conn = Connection::open_in_memory()?;
        let max_ts = init_connection(&conn, false, true)?;
        Ok(Self {
            writer: Mutex::new(conn),
            read_pool: None,
            db_path: None,
            max_bytes: AtomicU64::new(max_bytes),
            last_ts_ms: AtomicI64::new(max_ts),
            write_generation: AtomicU64::new(0),
            pending_touches: Mutex::new(HashMap::new()),
        })
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

    fn with_reader<T>(
        &self,
        f: impl FnOnce(&Connection) -> Result<T, rusqlite::Error>,
    ) -> Result<T, StorageError> {
        if let (Some(pool), Some(db_path)) = (&self.read_pool, &self.db_path) {
            let conn = {
                let mut guard = lock_unpoisoned(pool);
                guard.pop()
            };
            let conn = match conn {
                Some(c) => c,
                None => open_read_only_connection(db_path)?,
            };
            let result = f(&conn);
            if result.is_ok() {
                let mut guard = lock_unpoisoned(pool);
                if guard.len() < MAX_READ_POOL_CONNS {
                    guard.push(conn);
                }
            }
            return Ok(result?);
        }
        let conn = lock_unpoisoned(&self.writer);
        Ok(f(&conn)?)
    }

    fn record_touch(&self, tile_id: i64) {
        let ts = self.next_monotonic_ts();
        let should_try_flush = {
            let mut touches = lock_unpoisoned(&self.pending_touches);
            touches.insert(tile_id, ts);
            touches.len() >= TOUCH_FLUSH_THRESHOLD
        };
        // Non-blocking opportunistic flush: if the writer is busy (e.g. vacuum
        // or put), the writer will pick up `pending_touches` on its next pass
        // so concurrent WAL readers never stall.
        if should_try_flush && let Ok(mut conn) = self.writer.try_lock() {
            let touches = std::mem::take(&mut *lock_unpoisoned(&self.pending_touches));
            if !touches.is_empty()
                && let Ok(tx) = conn.transaction_with_behavior(TransactionBehavior::Immediate)
                && flush_touches_in_tx(&tx, &touches).is_ok()
            {
                let _ = tx.commit();
            }
        }
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

    fn get_encoded_filtered(
        &self,
        key: &DiskTileKey,
        expected_encoding: Option<TileBlobEncoding>,
    ) -> Result<Option<(TileBlobEncoding, Vec<u8>)>, StorageError> {
        let enc = EncodedKey::from_key(key)?;
        let row: Option<(i64, i64, Vec<u8>)> = self.with_reader(|conn| {
            let mut stmt = conn.prepare(SELECT_TILE_BLOB_SQL)?;
            stmt.query_row(enc.key_params(), |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
                .optional()
        })?;

        if let Some((tile_id, encoding_raw, payload)) = row {
            let Some(encoding) = TileBlobEncoding::from_i64(encoding_raw) else {
                return Ok(None);
            };
            if let Some(expected) = expected_encoding
                && encoding != expected
            {
                return Ok(None);
            }
            self.record_touch(tile_id);
            Ok(Some((encoding, payload)))
        } else {
            Ok(None)
        }
    }

    /// Look up cached encoded image bytes (`TileBlobEncoding::EncodedImage`)
    /// for `key`, recording a debounced LRU touch on hit. Returns `Ok(None)` if
    /// the stored entry uses a different encoding (e.g. `Qoi` or `Lz4Raw`).
    pub fn get(&self, key: &DiskTileKey) -> Result<Option<Vec<u8>>, StorageError> {
        Ok(self
            .get_encoded_filtered(key, Some(TileBlobEncoding::EncodedImage))?
            .map(|(_, payload)| payload))
    }

    /// Look up cached tile bytes and their [`TileBlobEncoding`] tag for `key`.
    pub fn get_encoded(
        &self,
        key: &DiskTileKey,
    ) -> Result<Option<(TileBlobEncoding, Vec<u8>)>, StorageError> {
        self.get_encoded_filtered(key, None)
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
        let enc = EncodedKey::from_key(key)?;
        let mut conn = lock_unpoisoned(&self.writer);
        let max_bytes = self.max_bytes.load(Ordering::Acquire);
        let new_bytes = payload.len() as u64;
        if payload.is_empty()
            || payload.len() > MAX_CACHED_TILE_BYTES
            || max_bytes == 0
            || new_bytes > max_bytes
        {
            return Ok(EvictionReport {
                evicted_tiles: 0,
                evicted_bytes: 0,
                total_bytes_after: current_total_bytes(&conn)?,
            });
        }

        let ts_ms = self.next_monotonic_ts();
        let report = self.with_write_tx(&mut conn, |tx| {
            let prev_bytes: Option<i64> = {
                let mut stmt = tx.prepare(SELECT_META_ID_BYTES_SQL)?;
                stmt.query_row(enc.key_params(), |r| r.get(1)).optional()?
            };

            let encoding_i64 = i64::from(encoding as u8);
            let new_bytes_i64 = new_bytes as i64;
            let inserted_id: i64 = {
                let mut stmt = tx.prepare(UPSERT_META_SQL)?;
                stmt.query_row(
                    params![
                        enc.source_key,
                        enc.band_kind,
                        enc.band_id,
                        enc.scheme,
                        enc.format,
                        enc.gutter,
                        enc.decode_rev,
                        enc.level,
                        enc.x,
                        enc.y,
                        encoding_i64,
                        new_bytes_i64,
                        ts_ms,
                    ],
                    |r| r.get(0),
                )?
            };

            tx.prepare(
                "INSERT INTO tile_blobs (tile_id, payload)
                 VALUES (?1, ?2)
                 ON CONFLICT(tile_id) DO UPDATE SET payload = excluded.payload",
            )?
            .execute(params![inserted_id, payload])?;

            let old = prev_bytes.unwrap_or(0).max(0) as u64;
            let count_delta: i64 = if prev_bytes.is_none() { 1 } else { 0 };
            let (current_total, current_count): (i64, i64) = tx.query_row(
                "SELECT total_bytes, tile_count FROM cache_meta WHERE id = 1",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )?;
            let updated_total = (current_total.max(0) as u64)
                .saturating_sub(old)
                .saturating_add(new_bytes);
            let updated_count = current_count.max(0).saturating_add(count_delta);
            tx.execute(
                "UPDATE cache_meta SET total_bytes = ?1, tile_count = ?2 WHERE id = 1",
                params![updated_total as i64, updated_count],
            )?;

            evict_in_tx(
                tx,
                updated_total,
                updated_count as u64,
                max_bytes,
                Some(inserted_id),
            )
        })?;

        self.write_generation.fetch_add(1, Ordering::AcqRel);
        if report.evicted_tiles > 0 {
            reclaim_space(&conn, self.db_path.is_some(), false);
        }
        Ok(report)
    }

    /// Delete one tile entry (e.g. if a cached payload fails decode validation).
    pub fn delete(&self, key: &DiskTileKey) -> Result<bool, StorageError> {
        let enc = EncodedKey::from_key(key)?;
        let mut conn = lock_unpoisoned(&self.writer);
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let prev: Option<(i64, i64)> = {
            let mut stmt = tx.prepare(SELECT_META_ID_BYTES_SQL)?;
            stmt.query_row(enc.key_params(), |r| Ok((r.get(0)?, r.get(1)?)))
                .optional()?
        };
        let Some((tile_id, sz)) = prev else {
            return Ok(false);
        };
        tx.execute("DELETE FROM tile_meta WHERE tile_id = ?1", params![tile_id])?;
        tx.execute(
            "UPDATE cache_meta
             SET total_bytes = MAX(0, total_bytes - ?1),
                 tile_count  = MAX(0, tile_count - 1)
             WHERE id = 1",
            params![sz],
        )?;
        tx.commit()?;
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
    pub(crate) fn evict_if_needed(&self) -> Result<EvictionReport, StorageError> {
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
        let report = self.with_write_tx(conn, |tx| {
            let (current_bytes, current_count): (i64, i64) = tx.query_row(
                "SELECT total_bytes, tile_count FROM cache_meta WHERE id = 1",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )?;
            evict_in_tx(
                tx,
                current_bytes.max(0) as u64,
                current_count.max(0) as u64,
                max_bytes,
                None,
            )
        })?;
        if report.evicted_tiles > 0 {
            reclaim_space(conn, self.db_path.is_some(), full_vacuum);
        }
        Ok(report)
    }

    /// Remove all cached tiles for a specific `source_key`, returning the
    /// number of deleted rows.
    pub fn clear_source(&self, source_key: &str) -> Result<usize, StorageError> {
        if !valid_source_key(source_key) {
            return Err(StorageError::InvalidSourceKey(source_key.to_owned()));
        }
        let mut conn = lock_unpoisoned(&self.writer);
        let deleted = self.with_write_tx(&mut conn, |tx| {
            let removed_bytes: i64 = tx.query_row(
                "SELECT COALESCE(SUM(byte_size), 0) FROM tile_meta WHERE source_key = ?1",
                params![source_key],
                |r| r.get(0),
            )?;
            let deleted = tx.execute(
                "DELETE FROM tile_meta WHERE source_key = ?1",
                params![source_key],
            )?;
            tx.execute(
                "UPDATE cache_meta
                 SET total_bytes = MAX(0, total_bytes - ?1),
                     tile_count  = MAX(0, tile_count - ?2)
                 WHERE id = 1",
                params![removed_bytes, deleted as i64],
            )?;
            Ok(deleted)
        })?;
        if deleted > 0 {
            reclaim_space(&conn, self.db_path.is_some(), true);
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
            // Delete blobs first to avoid per-row cascade work and allow SQLite's
            // table-truncate optimization.
            tx.execute("DELETE FROM tile_blobs", [])?;
            tx.execute("DELETE FROM tile_meta", [])?;
            tx.execute(
                "UPDATE cache_meta SET total_bytes = 0, tile_count = 0 WHERE id = 1",
                [],
            )?;
            tx.commit()?;
        }
        reclaim_space(&conn, self.db_path.is_some(), true);
        self.write_generation.fetch_add(1, Ordering::AcqRel);
        Ok(())
    }

    /// Return current tile count, payload byte total (both O(1) from
    /// `cache_meta`), file size (including `-wal` sidecar), and cap.
    pub fn stats(&self) -> Result<DiskCacheStats, StorageError> {
        let (tile_count, payload_bytes): (i64, i64) = self.with_reader(|conn| {
            conn.query_row(
                "SELECT tile_count, total_bytes FROM cache_meta WHERE id = 1",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
        })?;
        let file_bytes = self.db_path.as_ref().map_or(0, |p| {
            let main = std::fs::metadata(p).map_or(0, |m| m.len());
            let wal = std::fs::metadata(sidecar_path(p, "-wal")).map_or(0, |m| m.len());
            main.saturating_add(wal)
        });
        Ok(DiskCacheStats {
            tile_count: tile_count.max(0) as u64,
            payload_bytes: payload_bytes.max(0) as u64,
            file_bytes,
            max_bytes: self.max_bytes.load(Ordering::Acquire),
        })
    }
}

#[cfg(test)]
#[path = "tile_cache_tests.rs"]
mod tests;
