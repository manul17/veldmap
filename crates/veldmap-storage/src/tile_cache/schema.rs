//! SQLite WAL schema initialization, pragmas, schema-version upgrades, sidecar
//! path utilities, and corrupt-file recovery for [`super::TileDiskCache`].

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use rusqlite::{Connection, ErrorCode, OpenFlags, OptionalExtension, Transaction, params};

use super::key::EncodedKey;

/// Current SQLite schema version (`PRAGMA user_version`).
///
/// Version 2 vertically partitions `tile_meta` and `tile_blobs`, sets
/// `page_size = 16384`, adds `(band, format, gutter, decode_rev, encoding)`
/// columns, and tracks O(1) `tile_count` in `cache_meta`.
pub(super) const SCHEMA_VERSION: i64 = 2;

/// SQLite page size (`16 KiB`) reducing overflow-chain fragmentation for
/// 256 KiB–1 MiB tile payloads.
pub(super) const SQLITE_PAGE_SIZE: i64 = 16_384;

/// SQLite busy timeout in milliseconds (`5 s`) applied immediately on open so
/// multi-process WAL startups never fail prematurely with `SQLITE_BUSY`.
pub(super) const SQLITE_BUSY_TIMEOUT_MS: i64 = 5000;

/// Truncate-friendly deletion of all cached blobs and metadata.
pub(super) const CLEAR_BLOBS_SQL: &str = "DELETE FROM tile_blobs";
pub(super) const CLEAR_META_SQL: &str = "DELETE FROM tile_meta";

const SELECT_TILE_BLOB_SQL: &str = "SELECT m.tile_id, m.encoding, b.payload
     FROM tile_meta m
     JOIN tile_blobs b ON b.tile_id = m.tile_id
     WHERE m.source_key = ?1
       AND m.band_kind = ?2
       AND m.band_id = ?3
       AND m.scheme = ?4
       AND m.format = ?5
       AND m.gutter = ?6
       AND m.decode_rev = ?7
       AND m.level = ?8
       AND m.x = ?9
       AND m.y = ?10";

const SELECT_META_ID_BYTES_SQL: &str = "SELECT tile_id, byte_size FROM tile_meta
     WHERE source_key = ?1
       AND band_kind = ?2
       AND band_id = ?3
       AND scheme = ?4
       AND format = ?5
       AND gutter = ?6
       AND decode_rev = ?7
       AND level = ?8
       AND x = ?9
       AND y = ?10";

const UPSERT_META_SQL: &str = "INSERT INTO tile_meta (
         source_key, band_kind, band_id, scheme, format, gutter,
         decode_rev, level, x, y, encoding, byte_size, last_used_ms
     )
     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)
     ON CONFLICT(
         source_key, band_kind, band_id, scheme, format, gutter,
         decode_rev, level, x, y
     ) DO UPDATE SET
       encoding = excluded.encoding,
       byte_size = excluded.byte_size,
       last_used_ms = excluded.last_used_ms
     RETURNING tile_id";

const UPSERT_BLOB_SQL: &str = "INSERT INTO tile_blobs (tile_id, payload)
     VALUES (?1, ?2)
     ON CONFLICT(tile_id) DO UPDATE SET payload = excluded.payload";

const TOUCH_META_SQL: &str =
    "UPDATE tile_meta SET last_used_ms = ?1 WHERE tile_id = ?2 AND last_used_ms < ?1";

const DELETE_META_BY_ID_SQL: &str = "DELETE FROM tile_meta WHERE tile_id = ?1";

const DELETE_SOURCE_META_SQL: &str =
    "DELETE FROM tile_meta WHERE source_key = ?1 RETURNING tile_id, byte_size";

const SELECT_EVICT_BATCH_SQL: &str = "SELECT tile_id, byte_size FROM tile_meta
     ORDER BY last_used_ms ASC
     LIMIT ?1";

const SELECT_EVICT_BATCH_PROTECTED_SQL: &str = "SELECT tile_id, byte_size FROM tile_meta
     WHERE tile_id != ?1
     ORDER BY last_used_ms ASC
     LIMIT ?2";

pub(super) fn select_tile_blob(
    conn: &Connection,
    enc: &EncodedKey<'_>,
) -> Result<Option<(i64, i64, Vec<u8>)>, rusqlite::Error> {
    let mut stmt = conn.prepare(SELECT_TILE_BLOB_SQL)?;
    stmt.query_row(enc.key_params(), |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
        .optional()
}

pub(super) fn select_meta_id_bytes(
    tx: &Transaction<'_>,
    enc: &EncodedKey<'_>,
) -> Result<Option<(i64, i64)>, rusqlite::Error> {
    let mut stmt = tx.prepare(SELECT_META_ID_BYTES_SQL)?;
    stmt.query_row(enc.key_params(), |r| Ok((r.get(0)?, r.get(1)?)))
        .optional()
}

pub(super) fn upsert_tile_meta(
    tx: &Transaction<'_>,
    enc: &EncodedKey<'_>,
    encoding_raw: i64,
    byte_size: i64,
    ts_ms: i64,
) -> Result<i64, rusqlite::Error> {
    let mut stmt = tx.prepare(UPSERT_META_SQL)?;
    stmt.query_row(enc.upsert_params(&encoding_raw, &byte_size, &ts_ms), |r| {
        r.get(0)
    })
}

pub(super) fn upsert_tile_blob(
    tx: &Transaction<'_>,
    tile_id: i64,
    payload: &[u8],
) -> Result<(), rusqlite::Error> {
    let mut stmt = tx.prepare(UPSERT_BLOB_SQL)?;
    stmt.execute(params![tile_id, payload])?;
    Ok(())
}

pub(super) fn flush_touches_in_tx(
    tx: &Transaction<'_>,
    touches: &HashMap<i64, i64>,
) -> Result<(), rusqlite::Error> {
    if touches.is_empty() {
        return Ok(());
    }
    // Updates `tile_meta` only (never `tile_blobs`), so read-hit LRU touches
    // never rewrite BLOB overflow pages. Guarded with `last_used_ms < ?1` so a
    // buffered touch can never roll back a newer timestamp on a reused tile_id.
    let mut stmt = tx.prepare(TOUCH_META_SQL)?;
    for (&tile_id, &ts) in touches {
        stmt.execute(params![ts, tile_id])?;
    }
    Ok(())
}

pub(super) fn delete_meta_by_id(tx: &Transaction<'_>, tile_id: i64) -> Result<(), rusqlite::Error> {
    let mut stmt = tx.prepare(DELETE_META_BY_ID_SQL)?;
    stmt.execute(params![tile_id])?;
    Ok(())
}

pub(super) fn delete_meta_ids(
    tx: &Transaction<'_>,
    tile_ids: &[i64],
) -> Result<(), rusqlite::Error> {
    if tile_ids.is_empty() {
        return Ok(());
    }
    let mut stmt = tx.prepare(DELETE_META_BY_ID_SQL)?;
    for &tile_id in tile_ids {
        stmt.execute(params![tile_id])?;
    }
    Ok(())
}

pub(super) fn delete_source_meta(
    tx: &Transaction<'_>,
    source_key: &str,
) -> Result<(Vec<i64>, i64), rusqlite::Error> {
    let mut stmt = tx.prepare(DELETE_SOURCE_META_SQL)?;
    let rows = stmt.query_map(params![source_key], |r| {
        Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?))
    })?;
    let mut ids = Vec::new();
    let mut removed_bytes = 0i64;
    for row in rows {
        let (tile_id, byte_size) = row?;
        ids.push(tile_id);
        removed_bytes = removed_bytes.saturating_add(byte_size.max(0));
    }
    Ok((ids, removed_bytes))
}

pub(super) fn select_evict_batch(
    tx: &Transaction<'_>,
    limit: i64,
    protect_id: Option<i64>,
) -> Result<Vec<(i64, u64)>, rusqlite::Error> {
    // `ORDER BY last_used_ms ASC` (without `, tile_id ASC`) lets SQLite satisfy
    // the query directly from covering index `idx_tile_meta_lru (last_used_ms,
    // byte_size)` with zero temp B-tree sorting and zero table lookups.
    let mut batch = Vec::with_capacity(limit.max(0) as usize);
    if let Some(protected) = protect_id {
        let mut stmt = tx.prepare(SELECT_EVICT_BATCH_PROTECTED_SQL)?;
        let rows = stmt.query_map(params![protected, limit], |r| {
            let tile_id: i64 = r.get(0)?;
            let sz: i64 = r.get(1)?;
            Ok((tile_id, sz.max(0) as u64))
        })?;
        for row in rows {
            batch.push(row?);
        }
    } else {
        let mut stmt = tx.prepare(SELECT_EVICT_BATCH_SQL)?;
        let rows = stmt.query_map(params![limit], |r| {
            let tile_id: i64 = r.get(0)?;
            let sz: i64 = r.get(1)?;
            Ok((tile_id, sz.max(0) as u64))
        })?;
        for row in rows {
            batch.push(row?);
        }
    }
    Ok(batch)
}

pub(super) fn read_cache_meta_totals(conn: &Connection) -> Result<(u64, u64), rusqlite::Error> {
    let (total_bytes, tile_count): (i64, i64) = conn.query_row(
        "SELECT total_bytes, tile_count FROM cache_meta WHERE id = 1",
        [],
        |r| Ok((r.get(0)?, r.get(1)?)),
    )?;
    Ok((total_bytes.max(0) as u64, tile_count.max(0) as u64))
}

pub(super) fn set_cache_meta_totals(
    tx: &Transaction<'_>,
    total_bytes: u64,
    tile_count: u64,
) -> Result<(), rusqlite::Error> {
    tx.execute(
        "UPDATE cache_meta SET total_bytes = ?1, tile_count = ?2 WHERE id = 1",
        params![total_bytes as i64, tile_count as i64],
    )?;
    Ok(())
}

pub(super) fn decrement_cache_meta(
    tx: &Transaction<'_>,
    removed_bytes: i64,
    removed_count: i64,
) -> Result<(), rusqlite::Error> {
    tx.execute(
        "UPDATE cache_meta
         SET total_bytes = MAX(0, total_bytes - ?1),
             tile_count  = MAX(0, tile_count - ?2)
         WHERE id = 1",
        params![removed_bytes, removed_count],
    )?;
    Ok(())
}

pub(super) fn reset_cache_meta(tx: &Transaction<'_>) -> Result<(), rusqlite::Error> {
    tx.execute(
        "UPDATE cache_meta SET total_bytes = 0, tile_count = 0 WHERE id = 1",
        [],
    )?;
    Ok(())
}

pub(super) fn sidecar_path(path: &Path, suffix: &str) -> PathBuf {
    let mut os = path.as_os_str().to_os_string();
    os.push(suffix);
    PathBuf::from(os)
}

pub(super) fn is_corrupt_sqlite_error(err: &rusqlite::Error) -> bool {
    matches!(
        err.sqlite_error_code(),
        Some(ErrorCode::DatabaseCorrupt | ErrorCode::NotADatabase)
    )
}

pub(super) fn remove_sqlite_files(path: &Path) {
    for candidate in [
        path.to_path_buf(),
        sidecar_path(path, "-wal"),
        sidecar_path(path, "-shm"),
    ] {
        if let Err(err) = std::fs::remove_file(&candidate)
            && err.kind() != std::io::ErrorKind::NotFound
        {
            tracing::warn!(
                path = %candidate.display(),
                error = %err,
                "Failed to remove ephemeral SQLite cache file"
            );
        }
    }
}

pub(super) fn open_read_only_connection(path: &Path) -> Result<Connection, rusqlite::Error> {
    let flags = OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX;
    let conn = Connection::open_with_flags(path, flags)?;
    conn.pragma_update(None, "busy_timeout", SQLITE_BUSY_TIMEOUT_MS)?;
    Ok(conn)
}

fn query_page_count_and_version(conn: &Connection) -> Result<(i64, i64), rusqlite::Error> {
    let page_count: i64 = conn.query_row("PRAGMA page_count;", [], |r| r.get(0))?;
    let user_version: i64 = conn.query_row("PRAGMA user_version;", [], |r| r.get(0))?;
    Ok((page_count, user_version))
}

pub(super) fn init_connection(
    conn: &Connection,
    is_file_backed: bool,
    is_empty: bool,
) -> Result<i64, rusqlite::Error> {
    conn.pragma_update(None, "busy_timeout", SQLITE_BUSY_TIMEOUT_MS)?;
    conn.pragma_update(None, "foreign_keys", "ON")?;

    if is_empty {
        // `page_size` and `auto_vacuum` must be configured before entering WAL
        // mode or creating any tables on a brand-new database.
        conn.pragma_update(None, "page_size", SQLITE_PAGE_SIZE)?;
        conn.pragma_update(None, "auto_vacuum", "INCREMENTAL")?;
    }

    if is_file_backed {
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
    }

    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS tile_meta (
            tile_id      INTEGER PRIMARY KEY,
            source_key   TEXT    NOT NULL,
            band_kind    INTEGER NOT NULL,
            band_id      INTEGER NOT NULL,
            scheme       INTEGER NOT NULL,
            format       INTEGER NOT NULL,
            gutter       INTEGER NOT NULL,
            decode_rev   INTEGER NOT NULL,
            level        INTEGER NOT NULL,
            x            INTEGER NOT NULL,
            y            INTEGER NOT NULL,
            encoding     INTEGER NOT NULL,
            byte_size    INTEGER NOT NULL,
            last_used_ms INTEGER NOT NULL,
            UNIQUE (source_key, band_kind, band_id, scheme, format, gutter, decode_rev, level, x, y)
        );
        CREATE TABLE IF NOT EXISTS tile_blobs (
            tile_id INTEGER PRIMARY KEY REFERENCES tile_meta(tile_id) ON DELETE CASCADE,
            payload BLOB    NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_tile_meta_lru ON tile_meta (last_used_ms ASC, byte_size);
        CREATE TABLE IF NOT EXISTS cache_meta (
            id          INTEGER PRIMARY KEY CHECK (id = 1),
            total_bytes INTEGER NOT NULL,
            tile_count  INTEGER NOT NULL
        );",
    )?;
    conn.pragma_update(None, "user_version", SCHEMA_VERSION)?;

    // Only reconcile `cache_meta` when missing so warm startups stay O(1)
    // instead of scanning multi-GB tile tables.
    let has_meta: Option<i64> = conn
        .query_row("SELECT 1 FROM cache_meta WHERE id = 1", [], |r| r.get(0))
        .optional()?;
    if has_meta.is_none() {
        conn.execute(
            "INSERT OR IGNORE INTO cache_meta (id, total_bytes, tile_count)
             VALUES (
                 1,
                 (SELECT COALESCE(SUM(byte_size), 0) FROM tile_meta),
                 (SELECT COUNT(*) FROM tile_meta)
             )",
            [],
        )?;
    }

    // Seed monotonic timestamp from the rightmost leaf of `idx_tile_meta_lru`
    // (O(log N)) so backward system-clock jumps never invert LRU order.
    let max_ts: i64 = conn.query_row(
        "SELECT COALESCE(MAX(last_used_ms), 0) FROM tile_meta",
        [],
        |r| r.get(0),
    )?;
    Ok(max_ts.max(0))
}

pub(super) fn recreate_file_db(path: &Path) -> Result<(Connection, i64), rusqlite::Error> {
    remove_sqlite_files(path);
    let fresh = Connection::open(path)?;
    let max_ts = init_connection(&fresh, true, true)?;
    Ok((fresh, max_ts))
}

pub(super) fn open_and_init_file_db(path: &Path) -> Result<(Connection, i64), rusqlite::Error> {
    let conn = Connection::open(path)?;
    // Configure `busy_timeout` BEFORE probing PRAGMAs so multi-process WAL
    // startup contention waits cleanly instead of failing immediately with
    // `SQLITE_BUSY`.
    conn.pragma_update(None, "busy_timeout", SQLITE_BUSY_TIMEOUT_MS)?;
    let (page_count, user_version) = query_page_count_and_version(&conn)?;
    if page_count > 0 && user_version != SCHEMA_VERSION {
        tracing::info!(
            path = %path.display(),
            old_version = user_version,
            new_version = SCHEMA_VERSION,
            "Upgrading ephemeral SQLite tile cache schema; recreating clean cache"
        );
        drop(conn);
        return recreate_file_db(path);
    }
    let max_ts = init_connection(&conn, true, page_count == 0)?;
    Ok((conn, max_ts))
}
