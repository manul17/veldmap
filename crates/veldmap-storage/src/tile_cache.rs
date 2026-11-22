//! Crash-safe, multi-process-safe SQLite WAL tile disk cache with O(log N)
//! per-tile LRU eviction, debounced LRU touch flushing, and
//! `incremental_vacuum`.
//!
//! Storing tiles in a single WAL-mode SQLite database avoids both whole-source
//! eviction thrashing and loose-file `relatime`/Windows Defender storms.
//! Because tile caches are ephemeral, corrupted cache files (`DatabaseCorrupt` /
//! `NotADatabase`) are deleted and recreated cleanly on startup.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{Connection, ErrorCode, OptionalExtension, TransactionBehavior, params};
use veldmap_core::io::lock_unpoisoned;
use veldmap_core::tile::TileScheme;

use crate::StorageError;
use crate::settings::DEFAULT_DISK_CACHE_MAX_MB;

/// Default on-disk tile cache ceiling (`512 MiB`).
pub const DEFAULT_DISK_CACHE_BYTES: u64 = DEFAULT_DISK_CACHE_MAX_MB * 1024 * 1024;

/// Maximum encoded payload size admitted for a single cached tile (`16 MiB`).
pub const MAX_CACHED_TILE_BYTES: usize = 16 * 1024 * 1024;

/// Eviction headroom divisor (`10` => evict down to `90%` of budget so
/// subsequent inserts do not trigger eviction on every single tile).
const EVICT_HEADROOM_DIVISOR: u64 = 10;

/// Maximum SQLite pages reclaimed per eviction pass via `incremental_vacuum`.
const INCREMENTAL_VACUUM_PAGES: i64 = 128;

/// Maximum batch size pulled per SQLite LRU eviction iteration.
const EVICT_BATCH_ROWS: i64 = 128;

/// Number of buffered `get()` LRU hits before flushing touches in a single
/// batched transaction (avoids a WAL write transaction on every cache read).
const TOUCH_FLUSH_THRESHOLD: usize = 64;

/// Validate a disk cache source key: non-empty, `<= 128` bytes, not `.` or `..`,
/// closed safe ASCII alphabet (`a-z`, `A-Z`, `0-9`, `-`, `_`, `.`).
pub fn valid_source_key(key: &str) -> bool {
    !key.is_empty()
        && key.len() <= 128
        && key != "."
        && key != ".."
        && key
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
}

fn encode_scheme(scheme: TileScheme) -> i64 {
    match scheme {
        TileScheme::WebMercator => 1,
        TileScheme::Geodetic => 2,
        TileScheme::PolarCap { north: true } => 3,
        TileScheme::PolarCap { north: false } => 4,
        TileScheme::RasterPyramid { max_level } => 100 + i64::from(max_level),
    }
}

/// Composite lookup key for one encoded tile stored in [`TileDiskCache`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DiskTileKey {
    pub source_key: String,
    pub scheme: TileScheme,
    pub level: u8,
    pub x: u32,
    pub y: u32,
}

impl DiskTileKey {
    pub fn new(
        source_key: impl Into<String>,
        scheme: TileScheme,
        level: u8,
        x: u32,
        y: u32,
    ) -> Self {
        Self {
            source_key: source_key.into(),
            scheme,
            level,
            x,
            y,
        }
    }
}

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

struct CacheInner {
    conn: Connection,
    db_path: Option<PathBuf>,
    max_bytes: u64,
    last_ts_ms: i64,
    pending_touches: HashMap<i64, i64>,
}

impl CacheInner {
    fn next_monotonic_ts(&mut self) -> i64 {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_millis().min(i64::MAX as u128) as i64);
        let ts = if now > self.last_ts_ms {
            now
        } else {
            self.last_ts_ms.saturating_add(1)
        };
        self.last_ts_ms = ts;
        ts
    }

    /// Best-effort flush of buffered `get()` touches in one transaction.
    /// Drains `pending_touches` before executing so a transient `SQLITE_BUSY`
    /// never triggers a repeated flush storm on subsequent reads.
    fn flush_pending_touches_best_effort(&mut self) {
        if self.pending_touches.is_empty() {
            return;
        }
        let touches = std::mem::take(&mut self.pending_touches);
        let Ok(tx) = self
            .conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
        else {
            return;
        };
        if flush_touches_in_tx(&tx, &touches).is_ok() {
            let _ = tx.commit();
        }
    }
}

fn flush_touches_in_tx(
    tx: &rusqlite::Transaction<'_>,
    touches: &HashMap<i64, i64>,
) -> Result<(), rusqlite::Error> {
    if touches.is_empty() {
        return Ok(());
    }
    // Guard with `last_used_ms < ?1` so a buffered touch can never roll back a
    // newer timestamp written to a replaced/reused rowid.
    let mut stmt =
        tx.prepare("UPDATE tiles SET last_used_ms = ?1 WHERE rowid = ?2 AND last_used_ms < ?1")?;
    for (&rowid, &ts) in touches {
        stmt.execute(params![ts, rowid])?;
    }
    Ok(())
}

fn is_corrupt_sqlite_error(err: &rusqlite::Error) -> bool {
    matches!(
        err.sqlite_error_code(),
        Some(ErrorCode::DatabaseCorrupt | ErrorCode::NotADatabase)
    )
}

fn remove_sqlite_files(path: &Path) {
    let _ = std::fs::remove_file(path);
    let mut wal = path.as_os_str().to_os_string();
    wal.push("-wal");
    let _ = std::fs::remove_file(PathBuf::from(wal));
    let mut shm = path.as_os_str().to_os_string();
    shm.push("-shm");
    let _ = std::fs::remove_file(PathBuf::from(shm));
}

fn init_connection(conn: &Connection, is_file_backed: bool) -> Result<i64, rusqlite::Error> {
    conn.pragma_update(None, "busy_timeout", 5000i64)?;
    conn.pragma_update(None, "auto_vacuum", "INCREMENTAL")?;
    if is_file_backed {
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
    }
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS tiles (
            source_key   TEXT    NOT NULL,
            scheme       INTEGER NOT NULL,
            level        INTEGER NOT NULL,
            x            INTEGER NOT NULL,
            y            INTEGER NOT NULL,
            payload      BLOB    NOT NULL,
            byte_size    INTEGER NOT NULL,
            last_used_ms INTEGER NOT NULL,
            PRIMARY KEY (source_key, scheme, level, x, y)
        );
        DROP INDEX IF EXISTS idx_tiles_lru;
        CREATE INDEX IF NOT EXISTS idx_tiles_lru_v2 ON tiles (last_used_ms ASC, byte_size);
        CREATE TABLE IF NOT EXISTS cache_meta (
            id          INTEGER PRIMARY KEY CHECK (id = 1),
            total_bytes INTEGER NOT NULL
        );",
    )?;

    // Only reconcile `cache_meta` when missing so warm startups stay O(1)
    // instead of scanning multi-GB tile tables.
    let has_meta: Option<i64> = conn
        .query_row("SELECT 1 FROM cache_meta WHERE id = 1", [], |r| r.get(0))
        .optional()?;
    if has_meta.is_none() {
        conn.execute(
            "INSERT OR IGNORE INTO cache_meta (id, total_bytes)
             VALUES (1, (SELECT COALESCE(SUM(byte_size), 0) FROM tiles))",
            [],
        )?;
    }

    // Seed monotonic timestamp from the rightmost leaf of `idx_tiles_lru_v2`
    // (O(log N)) so backward system-clock jumps never invert LRU order.
    let max_ts: i64 = conn.query_row(
        "SELECT COALESCE(MAX(last_used_ms), 0) FROM tiles",
        [],
        |r| r.get(0),
    )?;
    Ok(max_ts.max(0))
}

fn current_total_bytes(conn: &Connection) -> Result<u64, rusqlite::Error> {
    let total: i64 =
        conn.query_row("SELECT total_bytes FROM cache_meta WHERE id = 1", [], |r| {
            r.get(0)
        })?;
    Ok(total.max(0) as u64)
}

/// Thread-safe, crash-safe SQLite WAL tile disk cache bounded by `max_bytes`.
pub struct TileDiskCache {
    inner: Mutex<CacheInner>,
}

impl Drop for TileDiskCache {
    fn drop(&mut self) {
        let mut inner = lock_unpoisoned(&self.inner);
        inner.flush_pending_touches_best_effort();
    }
}

impl std::fmt::Debug for TileDiskCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = lock_unpoisoned(&self.inner);
        f.debug_struct("TileDiskCache")
            .field("db_path", &inner.db_path)
            .field("max_bytes", &inner.max_bytes)
            .finish()
    }
}

impl TileDiskCache {
    /// Open or create a file-backed SQLite WAL tile cache at `path` bounded to
    /// `max_bytes` payload bytes. Automatically deletes and recreates a
    /// corrupted ephemeral cache database (`DatabaseCorrupt` / `NotADatabase`).
    pub fn open(path: impl AsRef<Path>, max_bytes: u64) -> Result<Self, StorageError> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }

        let (conn, max_ts) = match Connection::open(&path).and_then(|c| {
            let max_ts = init_connection(&c, true)?;
            Ok((c, max_ts))
        }) {
            Ok(pair) => pair,
            Err(err) if is_corrupt_sqlite_error(&err) => {
                tracing::warn!(
                    path = %path.display(),
                    error = %err,
                    "Corrupt ephemeral tile cache detected; deleting and recreating clean SQLite WAL cache"
                );
                remove_sqlite_files(&path);
                let fresh = Connection::open(&path)?;
                let max_ts = init_connection(&fresh, true)?;
                (fresh, max_ts)
            }
            Err(err) => return Err(StorageError::Sqlite(err)),
        };

        let cache = Self {
            inner: Mutex::new(CacheInner {
                conn,
                db_path: Some(path),
                max_bytes,
                last_ts_ms: max_ts,
                pending_touches: HashMap::new(),
            }),
        };
        let _ = cache.evict_if_needed()?;
        Ok(cache)
    }

    /// Open an isolated in-memory SQLite tile cache (primarily for unit tests).
    pub fn open_in_memory(max_bytes: u64) -> Result<Self, StorageError> {
        let conn = Connection::open_in_memory()?;
        let max_ts = init_connection(&conn, false)?;
        Ok(Self {
            inner: Mutex::new(CacheInner {
                conn,
                db_path: None,
                max_bytes,
                last_ts_ms: max_ts,
                pending_touches: HashMap::new(),
            }),
        })
    }

    /// Look up cached encoded tile bytes for `key`, recording a debounced LRU
    /// touch on hit (flushed in batches, before eviction, and on drop).
    pub fn get(&self, key: &DiskTileKey) -> Result<Option<Vec<u8>>, StorageError> {
        if !valid_source_key(&key.source_key) {
            return Err(StorageError::InvalidSourceKey(key.source_key.clone()));
        }
        let mut inner = lock_unpoisoned(&self.inner);
        let scheme = encode_scheme(key.scheme);
        let row: Option<(i64, Vec<u8>)> = inner
            .conn
            .query_row(
                "SELECT rowid, payload FROM tiles
                 WHERE source_key = ?1 AND scheme = ?2 AND level = ?3 AND x = ?4 AND y = ?5",
                params![
                    &key.source_key,
                    scheme,
                    i64::from(key.level),
                    i64::from(key.x),
                    i64::from(key.y)
                ],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?;

        if let Some((rowid, payload)) = row {
            let ts = inner.next_monotonic_ts();
            inner.pending_touches.insert(rowid, ts);
            if inner.pending_touches.len() >= TOUCH_FLUSH_THRESHOLD {
                inner.flush_pending_touches_best_effort();
            }
            Ok(Some(payload))
        } else {
            Ok(None)
        }
    }

    /// Insert or replace encoded tile bytes for `key`, enforcing
    /// `payload_bytes <= max_bytes` via O(log N) per-tile LRU eviction and
    /// `PRAGMA incremental_vacuum`.
    pub fn put(&self, key: &DiskTileKey, payload: &[u8]) -> Result<EvictionReport, StorageError> {
        let mut inner = lock_unpoisoned(&self.inner);
        let ts = inner.next_monotonic_ts();
        Self::put_locked(&mut inner, key, payload, ts)
    }

    fn put_locked(
        inner: &mut CacheInner,
        key: &DiskTileKey,
        payload: &[u8],
        ts_ms: i64,
    ) -> Result<EvictionReport, StorageError> {
        if !valid_source_key(&key.source_key) {
            return Err(StorageError::InvalidSourceKey(key.source_key.clone()));
        }
        let new_bytes = payload.len() as u64;
        if payload.is_empty()
            || payload.len() > MAX_CACHED_TILE_BYTES
            || inner.max_bytes == 0
            || new_bytes > inner.max_bytes
        {
            return Ok(EvictionReport {
                evicted_tiles: 0,
                evicted_bytes: 0,
                total_bytes_after: current_total_bytes(&inner.conn)?,
            });
        }

        let scheme = encode_scheme(key.scheme);
        let max_bytes = inner.max_bytes;
        let touches = std::mem::take(&mut inner.pending_touches);
        let report = {
            let tx = inner
                .conn
                .transaction_with_behavior(TransactionBehavior::Immediate)?;
            flush_touches_in_tx(&tx, &touches)?;

            let prev_bytes: Option<i64> = tx
                .query_row(
                    "SELECT byte_size FROM tiles
                     WHERE source_key = ?1 AND scheme = ?2 AND level = ?3 AND x = ?4 AND y = ?5",
                    params![
                        &key.source_key,
                        scheme,
                        i64::from(key.level),
                        i64::from(key.x),
                        i64::from(key.y)
                    ],
                    |r| r.get(0),
                )
                .optional()?;

            let inserted_rowid: i64 = tx.query_row(
                "INSERT INTO tiles (source_key, scheme, level, x, y, payload, byte_size, last_used_ms)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
                 ON CONFLICT(source_key, scheme, level, x, y) DO UPDATE SET
                   payload = excluded.payload,
                   byte_size = excluded.byte_size,
                   last_used_ms = excluded.last_used_ms
                 RETURNING rowid",
                params![
                    &key.source_key,
                    scheme,
                    i64::from(key.level),
                    i64::from(key.x),
                    i64::from(key.y),
                    payload,
                    new_bytes as i64,
                    ts_ms,
                ],
                |r| r.get(0),
            )?;

            let old = prev_bytes.unwrap_or(0).max(0) as u64;
            let current_total: i64 =
                tx.query_row("SELECT total_bytes FROM cache_meta WHERE id = 1", [], |r| {
                    r.get(0)
                })?;
            let updated_total = (current_total.max(0) as u64)
                .saturating_sub(old)
                .saturating_add(new_bytes);
            tx.execute(
                "UPDATE cache_meta SET total_bytes = ?1 WHERE id = 1",
                params![updated_total as i64],
            )?;

            let rep = evict_in_tx(&tx, updated_total, max_bytes, Some(inserted_rowid))?;
            tx.commit()?;
            rep
        };

        if report.evicted_tiles > 0 {
            let _ = inner.conn.execute_batch(&format!(
                "PRAGMA incremental_vacuum({INCREMENTAL_VACUUM_PAGES});"
            ));
        }
        Ok(report)
    }

    /// Delete one tile entry (e.g. if a cached payload fails decode validation).
    pub fn delete(&self, key: &DiskTileKey) -> Result<bool, StorageError> {
        if !valid_source_key(&key.source_key) {
            return Err(StorageError::InvalidSourceKey(key.source_key.clone()));
        }
        let mut inner = lock_unpoisoned(&self.inner);
        let scheme = encode_scheme(key.scheme);
        let tx = inner
            .conn
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let prev: Option<(i64, i64)> = tx
            .query_row(
                "SELECT rowid, byte_size FROM tiles
                 WHERE source_key = ?1 AND scheme = ?2 AND level = ?3 AND x = ?4 AND y = ?5",
                params![
                    &key.source_key,
                    scheme,
                    i64::from(key.level),
                    i64::from(key.x),
                    i64::from(key.y)
                ],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?;
        let Some((rowid, sz)) = prev else {
            return Ok(false);
        };
        tx.execute("DELETE FROM tiles WHERE rowid = ?1", params![rowid])?;
        tx.execute(
            "UPDATE cache_meta SET total_bytes = MAX(0, total_bytes - ?1) WHERE id = 1",
            params![sz],
        )?;
        tx.commit()?;
        inner.pending_touches.remove(&rowid);
        Ok(true)
    }

    /// Update the configured byte budget, immediately evicting oldest tiles,
    /// reclaiming freed pages, and truncating WAL when utilization exceeds the
    /// new cap.
    pub fn set_max_bytes(&self, max_bytes: u64) -> Result<EvictionReport, StorageError> {
        let mut inner = lock_unpoisoned(&self.inner);
        inner.max_bytes = max_bytes;
        Self::evict_with_vacuum_locked(&mut inner, true)
    }

    /// Enforce `total_bytes <= max_bytes` on the current database state.
    pub fn evict_if_needed(&self) -> Result<EvictionReport, StorageError> {
        let mut inner = lock_unpoisoned(&self.inner);
        Self::evict_with_vacuum_locked(&mut inner, false)
    }

    fn evict_with_vacuum_locked(
        inner: &mut CacheInner,
        full_vacuum: bool,
    ) -> Result<EvictionReport, StorageError> {
        let max_bytes = inner.max_bytes;
        let touches = std::mem::take(&mut inner.pending_touches);
        let report = {
            let tx = inner
                .conn
                .transaction_with_behavior(TransactionBehavior::Immediate)?;
            flush_touches_in_tx(&tx, &touches)?;
            let current: i64 =
                tx.query_row("SELECT total_bytes FROM cache_meta WHERE id = 1", [], |r| {
                    r.get(0)
                })?;
            let rep = evict_in_tx(&tx, current.max(0) as u64, max_bytes, None)?;
            tx.commit()?;
            rep
        };
        if report.evicted_tiles > 0 {
            if full_vacuum {
                let _ = inner.conn.execute_batch("PRAGMA incremental_vacuum;");
                if inner.db_path.is_some() {
                    let _ = inner.conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);");
                }
            } else {
                let _ = inner.conn.execute_batch(&format!(
                    "PRAGMA incremental_vacuum({INCREMENTAL_VACUUM_PAGES});"
                ));
            }
        }
        Ok(report)
    }

    /// Remove all cached tiles for a specific `source_key`, returning the
    /// number of deleted rows.
    pub fn clear_source(&self, source_key: &str) -> Result<usize, StorageError> {
        if !valid_source_key(source_key) {
            return Err(StorageError::InvalidSourceKey(source_key.to_owned()));
        }
        let mut inner = lock_unpoisoned(&self.inner);
        let deleted = {
            let tx = inner
                .conn
                .transaction_with_behavior(TransactionBehavior::Immediate)?;
            let removed_bytes: i64 = tx.query_row(
                "SELECT COALESCE(SUM(byte_size), 0) FROM tiles WHERE source_key = ?1",
                params![source_key],
                |r| r.get(0),
            )?;
            let deleted = tx.execute(
                "DELETE FROM tiles WHERE source_key = ?1",
                params![source_key],
            )?;
            tx.execute(
                "UPDATE cache_meta SET total_bytes = MAX(0, total_bytes - ?1) WHERE id = 1",
                params![removed_bytes],
            )?;
            tx.commit()?;
            deleted
        };
        if deleted > 0 {
            let _ = inner.conn.execute_batch("PRAGMA incremental_vacuum;");
            if inner.db_path.is_some() {
                let _ = inner.conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);");
            }
        }
        Ok(deleted)
    }

    /// Delete every cached tile and reclaim both main-DB pages and WAL space.
    pub fn clear(&self) -> Result<(), StorageError> {
        let mut inner = lock_unpoisoned(&self.inner);
        inner.pending_touches.clear();
        {
            let tx = inner
                .conn
                .transaction_with_behavior(TransactionBehavior::Immediate)?;
            tx.execute("DELETE FROM tiles", [])?;
            tx.execute("UPDATE cache_meta SET total_bytes = 0 WHERE id = 1", [])?;
            tx.commit()?;
        }
        let _ = inner.conn.execute_batch("PRAGMA incremental_vacuum;");
        if inner.db_path.is_some() {
            let _ = inner.conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);");
        }
        Ok(())
    }

    /// Return current tile count, payload byte total, file size (including
    /// `-wal` sidecar), and cap inside a consistent read snapshot.
    pub fn stats(&self) -> Result<DiskCacheStats, StorageError> {
        let mut inner = lock_unpoisoned(&self.inner);
        let (tile_count, payload_bytes): (i64, i64) = {
            let tx = inner
                .conn
                .transaction_with_behavior(TransactionBehavior::Deferred)?;
            let count: i64 = tx.query_row("SELECT COUNT(*) FROM tiles", [], |r| r.get(0))?;
            let bytes: i64 =
                tx.query_row("SELECT total_bytes FROM cache_meta WHERE id = 1", [], |r| {
                    r.get(0)
                })?;
            (count, bytes)
        };
        let file_bytes = inner.db_path.as_ref().map_or(0, |p| {
            let main = std::fs::metadata(p).map_or(0, |m| m.len());
            let mut wal_os = p.as_os_str().to_os_string();
            wal_os.push("-wal");
            let wal = std::fs::metadata(PathBuf::from(wal_os)).map_or(0, |m| m.len());
            main.saturating_add(wal)
        });
        Ok(DiskCacheStats {
            tile_count: tile_count.max(0) as u64,
            payload_bytes: payload_bytes.max(0) as u64,
            file_bytes,
            max_bytes: inner.max_bytes,
        })
    }
}

fn evict_in_tx(
    tx: &rusqlite::Transaction<'_>,
    mut total_bytes: u64,
    max_bytes: u64,
    protect_rowid: Option<i64>,
) -> Result<EvictionReport, rusqlite::Error> {
    if total_bytes <= max_bytes {
        return Ok(EvictionReport {
            evicted_tiles: 0,
            evicted_bytes: 0,
            total_bytes_after: total_bytes,
        });
    }

    let target = max_bytes.saturating_sub(max_bytes / EVICT_HEADROOM_DIVISOR);
    let mut evicted_tiles = 0usize;
    let mut evicted_bytes = 0u64;

    // `ORDER BY last_used_ms ASC` (without `, rowid ASC`) lets SQLite satisfy
    // the query directly from covering index `idx_tiles_lru_v2 (last_used_ms,
    // byte_size)` with zero temp B-tree sorting and zero table lookups.
    let mut select_stmt = match protect_rowid {
        Some(_) => tx.prepare(
            "SELECT rowid, byte_size FROM tiles
             WHERE rowid != ?2
             ORDER BY last_used_ms ASC
             LIMIT ?1",
        )?,
        None => tx.prepare(
            "SELECT rowid, byte_size FROM tiles
             ORDER BY last_used_ms ASC
             LIMIT ?1",
        )?,
    };
    let mut delete_stmt = tx.prepare("DELETE FROM tiles WHERE rowid = ?1")?;

    loop {
        if total_bytes <= target {
            break;
        }
        let mut batch = Vec::with_capacity(EVICT_BATCH_ROWS as usize);
        match protect_rowid {
            Some(protected) => {
                let rows = select_stmt.query_map(params![EVICT_BATCH_ROWS, protected], |r| {
                    let rowid: i64 = r.get(0)?;
                    let sz: i64 = r.get(1)?;
                    Ok((rowid, sz.max(0) as u64))
                })?;
                for row in rows {
                    batch.push(row?);
                }
            }
            None => {
                let rows = select_stmt.query_map(params![EVICT_BATCH_ROWS], |r| {
                    let rowid: i64 = r.get(0)?;
                    let sz: i64 = r.get(1)?;
                    Ok((rowid, sz.max(0) as u64))
                })?;
                for row in rows {
                    batch.push(row?);
                }
            }
        }

        if batch.is_empty() {
            break;
        }

        for (rowid, sz) in batch {
            if total_bytes <= target {
                break;
            }
            delete_stmt.execute(params![rowid])?;
            total_bytes = total_bytes.saturating_sub(sz);
            evicted_bytes = evicted_bytes.saturating_add(sz);
            evicted_tiles += 1;
        }
    }

    tx.execute(
        "UPDATE cache_meta SET total_bytes = ?1 WHERE id = 1",
        params![total_bytes as i64],
    )?;

    Ok(EvictionReport {
        evicted_tiles,
        evicted_bytes,
        total_bytes_after: total_bytes,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::thread;

    use super::*;
    use crate::TestTempDir;

    fn key(source: &str, level: u8, x: u32, y: u32) -> DiskTileKey {
        DiskTileKey::new(source, TileScheme::WebMercator, level, x, y)
    }

    #[test]
    fn validates_source_keys_and_enforces_lru_eviction_with_headroom() {
        assert!(valid_source_key("nasa-gibs_blue_marble.v1"));
        assert!(valid_source_key(&"a".repeat(128)));
        assert!(!valid_source_key(&"a".repeat(129)));
        assert!(!valid_source_key(""));
        assert!(!valid_source_key("."));
        assert!(!valid_source_key(".."));
        assert!(!valid_source_key("../escape"));
        assert!(!valid_source_key("a/b"));

        // Budget = 100 B, 10% headroom target = 90 B.
        let cache = TileDiskCache::open_in_memory(100).expect("open mem cache");
        let k_a = key("eox", 12, 0, 0);
        let k_b = key("eox", 12, 1, 0);
        let k_c = key("eox", 12, 2, 0);
        let k_d = key("eox", 12, 3, 0);

        cache.put(&k_a, &[1u8; 40]).expect("put a");
        cache.put(&k_b, &[2u8; 20]).expect("put b");
        cache.put(&k_c, &[3u8; 40]).expect("put c");
        assert_eq!(cache.stats().expect("stats").payload_bytes, 100);

        // Touch `k_a` (debounced in memory) -> must flush before eviction so `k_b`
        // is evicted first, NOT `k_a`.
        assert_eq!(cache.get(&k_a).expect("get a"), Some(vec![1u8; 40]));

        // Insert `k_d` (30 B): total becomes 130 > 100, target = 90.
        // Evicting `k_b` (20) -> 110 > 90; evicting `k_c` (40) -> 70 <= 90.
        let report = cache.put(&k_d, &[4u8; 30]).expect("put d");
        assert_eq!(report.evicted_tiles, 2);
        assert_eq!(report.evicted_bytes, 60);
        assert_eq!(report.total_bytes_after, 70);
        assert!(report.total_bytes_after <= 100);

        assert!(cache.get(&k_a).expect("get a").is_some());
        assert!(cache.get(&k_b).expect("get b").is_none());
        assert!(cache.get(&k_c).expect("get c").is_none());
        assert!(cache.get(&k_d).expect("get d").is_some());

        let stats = cache.stats().expect("stats");
        assert_eq!(stats.tile_count, 2);
        assert_eq!(stats.payload_bytes, 70);
    }

    #[test]
    fn eviction_query_uses_covering_lru_index_without_temp_btree() {
        let cache = TileDiskCache::open_in_memory(4096).expect("open mem");
        let inner = lock_unpoisoned(&cache.inner);
        let mut stmt = inner
            .conn
            .prepare(
                "EXPLAIN QUERY PLAN
                 SELECT rowid, byte_size FROM tiles
                 WHERE rowid != ?2
                 ORDER BY last_used_ms ASC
                 LIMIT ?1",
            )
            .expect("explain");
        let details: Vec<String> = stmt
            .query_map(params![EVICT_BATCH_ROWS, 1i64], |r| r.get::<_, String>(3))
            .expect("query plan")
            .map(|r| r.expect("plan row"))
            .collect();
        let joined = details.join(" | ");
        assert!(
            joined.contains("idx_tiles_lru_v2") && joined.contains("COVERING"),
            "expected covering idx_tiles_lru_v2 scan, got: {joined}"
        );
        assert!(
            !joined.contains("TEMP B-TREE"),
            "unexpected temp B-tree sort in eviction plan: {joined}"
        );
    }

    #[test]
    fn recovers_cleanly_from_corrupt_sqlite_file_and_persists_across_reopen() {
        let dir = TestTempDir::new("sqlite-recovery");
        let db_path = dir.path().join("tiles.sqlite3");

        // Write garbage bytes simulating a corrupted cache file.
        std::fs::write(&db_path, b"not a valid sqlite database file!").expect("write corrupt");

        let cache = TileDiskCache::open(&db_path, 4096).expect("recover corrupt db");
        assert_eq!(cache.stats().expect("stats").tile_count, 0);

        let k1 = key("gibs", 3, 2, 1);
        cache.put(&k1, b"png-bytes").expect("put k1");
        let ts_before = lock_unpoisoned(&cache.inner).last_ts_ms;
        drop(cache);

        // Reopen and verify persistence + monotonic timestamp seed + shrinking + clear.
        let reopened = TileDiskCache::open(&db_path, 4096).expect("reopen");
        assert!(
            lock_unpoisoned(&reopened.inner).last_ts_ms >= ts_before,
            "reopen must seed last_ts_ms from MAX(last_used_ms)"
        );
        assert_eq!(
            reopened.get(&k1).expect("get k1"),
            Some(b"png-bytes".to_vec())
        );

        let ev = reopened.set_max_bytes(4).expect("shrink cap");
        assert_eq!(ev.evicted_tiles, 1);
        assert_eq!(reopened.stats().expect("stats").payload_bytes, 0);

        // Put a sizeable blob, then clear() and confirm WAL/file bytes shrink.
        reopened.set_max_bytes(256 * 1024).expect("grow cap");
        for i in 0..8u32 {
            reopened
                .put(&key("gibs", 4, i, 0), &vec![0xAB; 16 * 1024])
                .expect("put chunk");
        }
        let before_clear = reopened.stats().expect("stats before clear").file_bytes;
        reopened.clear().expect("clear");
        let after_clear = reopened.stats().expect("stats after clear");
        assert_eq!(after_clear.tile_count, 0);
        assert_eq!(after_clear.payload_bytes, 0);
        assert!(
            after_clear.file_bytes < before_clear,
            "clear() should reclaim space (before={before_clear}, after={})",
            after_clear.file_bytes
        );
    }

    #[test]
    fn overwrite_accounting_skipped_puts_and_multi_batch_eviction_stay_exact() {
        let cache = TileDiskCache::open_in_memory(10_000).expect("open mem");
        let k = key("eox", 5, 1, 1);

        // Overwrite with larger then smaller payloads; byte accounting stays exact.
        cache.put(&k, &[1u8; 100]).expect("put 100");
        assert_eq!(cache.stats().expect("s1").payload_bytes, 100);
        cache.put(&k, &[2u8; 350]).expect("put 350");
        assert_eq!(cache.stats().expect("s2").payload_bytes, 350);
        cache.put(&k, &[3u8; 50]).expect("put 50");
        assert_eq!(cache.stats().expect("s3").payload_bytes, 50);

        // Skipped puts (empty or > max_bytes) return honest current total_bytes_after.
        let empty_rep = cache.put(&k, &[]).expect("skip empty");
        assert_eq!(empty_rep.total_bytes_after, 50);
        let big_rep = cache.put(&k, &vec![0u8; 20_000]).expect("skip oversized");
        assert_eq!(big_rep.total_bytes_after, 50);

        // Insert 200 tiles (crossing EVICT_BATCH_ROWS = 128), then shrink cap to trigger
        // multi-batch eviction.
        cache.clear().expect("clear");
        for i in 0..200u32 {
            cache
                .put(&key("eox", 8, i, 0), &[i as u8; 40])
                .expect("put batch tile");
        }
        assert_eq!(cache.stats().expect("pre-evict").tile_count, 200);
        let ev = cache.set_max_bytes(1000).expect("multi-batch evict");
        assert!(ev.evicted_tiles > EVICT_BATCH_ROWS as usize);
        assert!(ev.total_bytes_after <= 900);
        assert_eq!(
            cache.stats().expect("post-evict").payload_bytes,
            ev.total_bytes_after
        );
    }

    #[test]
    fn clear_source_removes_only_matching_source() {
        let cache = TileDiskCache::open_in_memory(4096).expect("open mem");
        let k1 = key("gibs", 1, 0, 0);
        let k2 = key("eox", 1, 0, 0);
        cache.put(&k1, &[1u8; 100]).expect("put k1");
        cache.put(&k2, &[2u8; 200]).expect("put k2");

        let removed = cache.clear_source("gibs").expect("clear gibs");
        assert_eq!(removed, 1);
        assert!(cache.get(&k1).expect("get k1").is_none());
        assert!(cache.get(&k2).expect("get k2").is_some());
        assert_eq!(cache.stats().expect("stats").payload_bytes, 200);
    }

    #[test]
    fn large_tile_near_budget_never_self_evicts_and_delete_updates_total() {
        let cache = TileDiskCache::open_in_memory(100).expect("open mem");
        let old = key("eox", 1, 0, 0);
        let big = key("eox", 1, 1, 0);

        cache.put(&old, &[1u8; 20]).expect("put old");
        // 95 B exceeds the 90 B headroom target, yet fits within 100 B cap.
        // Eviction must drop `old` and MUST NOT self-evict `big`.
        let rep = cache.put(&big, &[2u8; 95]).expect("put big");
        assert_eq!(rep.evicted_tiles, 1);
        assert_eq!(rep.total_bytes_after, 95);
        assert!(cache.get(&old).expect("get old").is_none());
        assert_eq!(cache.get(&big).expect("get big"), Some(vec![2u8; 95]));

        assert!(cache.delete(&big).expect("delete big"));
        assert_eq!(cache.stats().expect("stats").payload_bytes, 0);
    }

    #[test]
    fn concurrent_reads_and_writes_keep_budget_and_accounting_consistent() {
        let cache = Arc::new(TileDiskCache::open_in_memory(4096).expect("open mem"));
        let mut threads = Vec::new();
        for tid in 0..4u32 {
            let c = Arc::clone(&cache);
            threads.push(thread::spawn(move || {
                for i in 0..40u32 {
                    let k = key("eox", 10, tid, i);
                    c.put(&k, &[tid as u8; 128]).expect("concurrent put");
                    let _ = c.get(&k).expect("concurrent get");
                }
            }));
        }
        for h in threads {
            h.join().expect("join worker");
        }
        let stats = cache.stats().expect("stats");
        assert!(stats.payload_bytes <= 4096);
        assert_eq!(stats.payload_bytes, stats.tile_count * 128);
    }
}
