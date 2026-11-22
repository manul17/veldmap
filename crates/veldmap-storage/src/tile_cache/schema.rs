//! SQLite WAL schema initialization, pragmas, schema-version upgrades, sidecar
//! path utilities, and corrupt-file recovery for [`super::TileDiskCache`].

use std::path::{Path, PathBuf};

use rusqlite::{Connection, ErrorCode, OpenFlags, OptionalExtension};

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
        remove_sqlite_files(path);
        let fresh = Connection::open(path)?;
        let max_ts = init_connection(&fresh, true, true)?;
        return Ok((fresh, max_ts));
    }
    let max_ts = init_connection(&conn, true, page_count == 0)?;
    Ok((conn, max_ts))
}
