//! Crash-safe, multi-process-safe SQLite WAL tile disk cache with O(log N)
//! per-tile LRU eviction, vertically partitioned `tile_meta`/`tile_blobs`
//! tables, debounced LRU touch flushing, and `incremental_vacuum`.
//!
//! Storing tiles in a single WAL-mode SQLite database avoids both whole-source
//! eviction thrashing and loose-file `relatime`/Windows Defender storms.
//! Separating `tile_meta` (`last_used_ms`, dimensions/keys) from `tile_blobs`
//! (`payload BLOB`) ensures batched `get()` LRU touch updates never trigger
//! SQLite `OP_MakeRecord` overflow-page rewrites of 256 KiB–1 MiB scientific
//! tile blobs. Because tile caches are ephemeral, corrupted or older-schema
//! cache files are recreated cleanly on startup.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{Connection, ErrorCode, OptionalExtension, TransactionBehavior, params};
use veldmap_core::io::lock_unpoisoned;
use veldmap_core::tile::{BandKey, PixelFormat, TileKey, TileScheme};

use crate::StorageError;
use crate::settings::DEFAULT_DISK_CACHE_MAX_MB;

/// Default on-disk tile cache ceiling (`512 MiB`).
pub const DEFAULT_DISK_CACHE_BYTES: u64 = DEFAULT_DISK_CACHE_MAX_MB * 1024 * 1024;

/// Maximum encoded payload size admitted for a single cached tile (`16 MiB`).
pub const MAX_CACHED_TILE_BYTES: usize = 16 * 1024 * 1024;

/// Current SQLite schema version (`PRAGMA user_version`).
///
/// Version 2 vertically partitions `tile_meta` and `tile_blobs`, sets
/// `page_size = 16384`, adds `(band, format, gutter, decode_rev, encoding)`
/// columns, and tracks O(1) `tile_count` in `cache_meta`.
const SCHEMA_VERSION: i64 = 2;

/// SQLite page size (`16 KiB`) reducing overflow-chain fragmentation for
/// 256 KiB–1 MiB tile payloads.
const SQLITE_PAGE_SIZE: i64 = 16_384;

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

/// Wire/disk payload encoding tag stored alongside each cached tile blob.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum TileBlobEncoding {
    /// Raw upstream encoded image bytes (PNG/JPEG/WebP from live tile servers).
    #[default]
    EncodedImage = 0,
    /// Lossless QOI-encoded 8-bit display RGBA tiles (Slice 3A-1b).
    Qoi = 1,
    /// Byte-shuffled LZ4 scientific/categorical tile payloads (Slice 3B).
    Lz4Raw = 2,
}

impl TileBlobEncoding {
    fn from_i64(raw: i64) -> Option<Self> {
        match raw {
            0 => Some(Self::EncodedImage),
            1 => Some(Self::Qoi),
            2 => Some(Self::Lz4Raw),
            _ => None,
        }
    }
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

fn encode_band(band: BandKey) -> (i64, i64) {
    match band {
        BandKey::Display => (0, 0),
        BandKey::Index(idx) => (1, i64::from(idx)),
        BandKey::NamedHash(hash) => (2, hash as i64),
    }
}

fn encode_format(format: PixelFormat) -> i64 {
    match format {
        PixelFormat::Rgba8UnormSrgb => 0,
        PixelFormat::R32Float => 1,
        PixelFormat::R8Uint => 2,
        PixelFormat::R16Uint => 3,
    }
}

/// Composite lookup key for one encoded tile stored in [`TileDiskCache`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DiskTileKey {
    pub source_key: String,
    pub band: BandKey,
    pub scheme: TileScheme,
    pub format: PixelFormat,
    pub gutter: u8,
    pub decode_rev: u32,
    pub level: u8,
    pub x: u32,
    pub y: u32,
}

impl DiskTileKey {
    /// Construct a display-track (`BandKey::Display`, `Rgba8UnormSrgb`,
    /// `gutter = 0`, `decode_rev = 0`) disk cache key.
    pub fn new(
        source_key: impl Into<String>,
        scheme: TileScheme,
        level: u8,
        x: u32,
        y: u32,
    ) -> Self {
        Self {
            source_key: source_key.into(),
            band: BandKey::Display,
            scheme,
            format: PixelFormat::Rgba8UnormSrgb,
            gutter: 0,
            decode_rev: 0,
            level,
            x,
            y,
        }
    }

    /// Construct a full multi-band / multi-format disk cache key from a
    /// [`TileKey`] and its stable ASCII `source_key` / fingerprint string.
    pub fn from_tile_key(source_key: impl Into<String>, key: &TileKey) -> Self {
        Self {
            source_key: source_key.into(),
            band: key.band,
            scheme: key.scheme,
            format: key.format,
            gutter: key.gutter,
            decode_rev: key.decode_rev,
            level: key.level,
            x: key.x,
            y: key.y,
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

fn query_page_count_and_version(conn: &Connection) -> Result<(i64, i64), rusqlite::Error> {
    let page_count: i64 = conn.query_row("PRAGMA page_count;", [], |r| r.get(0))?;
    let user_version: i64 = conn.query_row("PRAGMA user_version;", [], |r| r.get(0))?;
    Ok((page_count, user_version))
}

fn reset_in_place_for_upgrade(conn: &Connection) -> Result<(), rusqlite::Error> {
    conn.execute_batch(
        "DROP TABLE IF EXISTS tile_blobs;
         DROP TABLE IF EXISTS tile_meta;
         DROP TABLE IF EXISTS tiles;
         DROP TABLE IF EXISTS cache_meta;
         PRAGMA journal_mode = DELETE;",
    )?;
    conn.pragma_update(None, "page_size", SQLITE_PAGE_SIZE)?;
    conn.pragma_update(None, "auto_vacuum", "INCREMENTAL")?;
    conn.execute_batch("VACUUM;")?;
    Ok(())
}

fn init_connection(conn: &Connection, is_file_backed: bool) -> Result<i64, rusqlite::Error> {
    conn.pragma_update(None, "busy_timeout", 5000i64)?;
    conn.pragma_update(None, "foreign_keys", "ON")?;

    let (page_count, user_version) = query_page_count_and_version(conn)?;
    if page_count == 0 {
        // `page_size` and `auto_vacuum` must be configured before entering WAL
        // mode or creating any tables on a brand-new database.
        conn.pragma_update(None, "page_size", SQLITE_PAGE_SIZE)?;
        conn.pragma_update(None, "auto_vacuum", "INCREMENTAL")?;
    } else if user_version != SCHEMA_VERSION {
        reset_in_place_for_upgrade(conn)?;
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

fn open_and_init_file_db(path: &Path) -> Result<(Connection, i64), rusqlite::Error> {
    let conn = Connection::open(path)?;
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
        let max_ts = init_connection(&fresh, true)?;
        return Ok((fresh, max_ts));
    }
    let max_ts = init_connection(&conn, true)?;
    Ok((conn, max_ts))
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
    /// touch on hit (flushed in batches to `tile_meta` only, before eviction,
    /// and on drop).
    pub fn get(&self, key: &DiskTileKey) -> Result<Option<Vec<u8>>, StorageError> {
        Ok(self.get_encoded(key)?.map(|(_, payload)| payload))
    }

    /// Look up cached tile bytes and their [`TileBlobEncoding`] tag for `key`.
    pub fn get_encoded(
        &self,
        key: &DiskTileKey,
    ) -> Result<Option<(TileBlobEncoding, Vec<u8>)>, StorageError> {
        if !valid_source_key(&key.source_key) {
            return Err(StorageError::InvalidSourceKey(key.source_key.clone()));
        }
        let mut inner = lock_unpoisoned(&self.inner);
        let (band_kind, band_id) = encode_band(key.band);
        let scheme = encode_scheme(key.scheme);
        let format = encode_format(key.format);
        let row: Option<(i64, i64, Vec<u8>)> = inner
            .conn
            .query_row(
                "SELECT m.tile_id, m.encoding, b.payload
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
                   AND m.y = ?10",
                params![
                    &key.source_key,
                    band_kind,
                    band_id,
                    scheme,
                    format,
                    i64::from(key.gutter),
                    i64::from(key.decode_rev),
                    i64::from(key.level),
                    i64::from(key.x),
                    i64::from(key.y)
                ],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .optional()?;

        if let Some((tile_id, encoding_raw, payload)) = row {
            let Some(encoding) = TileBlobEncoding::from_i64(encoding_raw) else {
                return Ok(None);
            };
            let ts = inner.next_monotonic_ts();
            inner.pending_touches.insert(tile_id, ts);
            if inner.pending_touches.len() >= TOUCH_FLUSH_THRESHOLD {
                inner.flush_pending_touches_best_effort();
            }
            Ok(Some((encoding, payload)))
        } else {
            Ok(None)
        }
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
        let mut inner = lock_unpoisoned(&self.inner);
        let ts = inner.next_monotonic_ts();
        Self::put_locked(&mut inner, key, encoding, payload, ts)
    }

    fn put_locked(
        inner: &mut CacheInner,
        key: &DiskTileKey,
        encoding: TileBlobEncoding,
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

        let (band_kind, band_id) = encode_band(key.band);
        let scheme = encode_scheme(key.scheme);
        let format = encode_format(key.format);
        let max_bytes = inner.max_bytes;
        let touches = std::mem::take(&mut inner.pending_touches);
        let report = {
            let tx = inner
                .conn
                .transaction_with_behavior(TransactionBehavior::Immediate)?;
            flush_touches_in_tx(&tx, &touches)?;

            let prev_bytes: Option<i64> = tx
                .query_row(
                    "SELECT byte_size FROM tile_meta
                     WHERE source_key = ?1
                       AND band_kind = ?2
                       AND band_id = ?3
                       AND scheme = ?4
                       AND format = ?5
                       AND gutter = ?6
                       AND decode_rev = ?7
                       AND level = ?8
                       AND x = ?9
                       AND y = ?10",
                    params![
                        &key.source_key,
                        band_kind,
                        band_id,
                        scheme,
                        format,
                        i64::from(key.gutter),
                        i64::from(key.decode_rev),
                        i64::from(key.level),
                        i64::from(key.x),
                        i64::from(key.y)
                    ],
                    |r| r.get(0),
                )
                .optional()?;

            let inserted_id: i64 = tx.query_row(
                "INSERT INTO tile_meta (
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
                 RETURNING tile_id",
                params![
                    &key.source_key,
                    band_kind,
                    band_id,
                    scheme,
                    format,
                    i64::from(key.gutter),
                    i64::from(key.decode_rev),
                    i64::from(key.level),
                    i64::from(key.x),
                    i64::from(key.y),
                    i64::from(encoding as u8),
                    new_bytes as i64,
                    ts_ms,
                ],
                |r| r.get(0),
            )?;

            tx.execute(
                "INSERT INTO tile_blobs (tile_id, payload)
                 VALUES (?1, ?2)
                 ON CONFLICT(tile_id) DO UPDATE SET payload = excluded.payload",
                params![inserted_id, payload],
            )?;

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

            let rep = evict_in_tx(
                &tx,
                updated_total,
                updated_count as u64,
                max_bytes,
                Some(inserted_id),
            )?;
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
        let (band_kind, band_id) = encode_band(key.band);
        let scheme = encode_scheme(key.scheme);
        let format = encode_format(key.format);
        let tx = inner
            .conn
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let prev: Option<(i64, i64)> = tx
            .query_row(
                "SELECT tile_id, byte_size FROM tile_meta
                 WHERE source_key = ?1
                   AND band_kind = ?2
                   AND band_id = ?3
                   AND scheme = ?4
                   AND format = ?5
                   AND gutter = ?6
                   AND decode_rev = ?7
                   AND level = ?8
                   AND x = ?9
                   AND y = ?10",
                params![
                    &key.source_key,
                    band_kind,
                    band_id,
                    scheme,
                    format,
                    i64::from(key.gutter),
                    i64::from(key.decode_rev),
                    i64::from(key.level),
                    i64::from(key.x),
                    i64::from(key.y)
                ],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?;
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
        inner.pending_touches.remove(&tile_id);
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
            let (current_bytes, current_count): (i64, i64) = tx.query_row(
                "SELECT total_bytes, tile_count FROM cache_meta WHERE id = 1",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )?;
            let rep = evict_in_tx(
                &tx,
                current_bytes.max(0) as u64,
                current_count.max(0) as u64,
                max_bytes,
                None,
            )?;
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
        let touches = std::mem::take(&mut inner.pending_touches);
        let deleted = {
            let tx = inner
                .conn
                .transaction_with_behavior(TransactionBehavior::Immediate)?;
            flush_touches_in_tx(&tx, &touches)?;
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
        let _ = inner.conn.execute_batch("PRAGMA incremental_vacuum;");
        if inner.db_path.is_some() {
            let _ = inner.conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);");
        }
        Ok(())
    }

    /// Return current tile count, payload byte total (both O(1) from
    /// `cache_meta`), file size (including `-wal` sidecar), and cap.
    pub fn stats(&self) -> Result<DiskCacheStats, StorageError> {
        let inner = lock_unpoisoned(&self.inner);
        let (tile_count, payload_bytes): (i64, i64) = inner.conn.query_row(
            "SELECT tile_count, total_bytes FROM cache_meta WHERE id = 1",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )?;
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
    mut tile_count: u64,
    max_bytes: u64,
    protect_id: Option<i64>,
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

    // `ORDER BY last_used_ms ASC` (without `, tile_id ASC`) lets SQLite satisfy
    // the query directly from covering index `idx_tile_meta_lru (last_used_ms,
    // byte_size)` with zero temp B-tree sorting and zero table lookups.
    let mut select_stmt = match protect_id {
        Some(_) => tx.prepare(
            "SELECT tile_id, byte_size FROM tile_meta
             WHERE tile_id != ?2
             ORDER BY last_used_ms ASC
             LIMIT ?1",
        )?,
        None => tx.prepare(
            "SELECT tile_id, byte_size FROM tile_meta
             ORDER BY last_used_ms ASC
             LIMIT ?1",
        )?,
    };
    let mut delete_stmt = tx.prepare("DELETE FROM tile_meta WHERE tile_id = ?1")?;

    loop {
        if total_bytes <= target {
            break;
        }
        let mut batch = Vec::with_capacity(EVICT_BATCH_ROWS as usize);
        match protect_id {
            Some(protected) => {
                let rows = select_stmt.query_map(params![EVICT_BATCH_ROWS, protected], |r| {
                    let tile_id: i64 = r.get(0)?;
                    let sz: i64 = r.get(1)?;
                    Ok((tile_id, sz.max(0) as u64))
                })?;
                for row in rows {
                    batch.push(row?);
                }
            }
            None => {
                let rows = select_stmt.query_map(params![EVICT_BATCH_ROWS], |r| {
                    let tile_id: i64 = r.get(0)?;
                    let sz: i64 = r.get(1)?;
                    Ok((tile_id, sz.max(0) as u64))
                })?;
                for row in rows {
                    batch.push(row?);
                }
            }
        }

        if batch.is_empty() {
            break;
        }

        for (tile_id, sz) in batch {
            if total_bytes <= target {
                break;
            }
            delete_stmt.execute(params![tile_id])?;
            total_bytes = total_bytes.saturating_sub(sz);
            tile_count = tile_count.saturating_sub(1);
            evicted_bytes = evicted_bytes.saturating_add(sz);
            evicted_tiles += 1;
        }
    }

    tx.execute(
        "UPDATE cache_meta SET total_bytes = ?1, tile_count = ?2 WHERE id = 1",
        params![total_bytes as i64, tile_count as i64],
    )?;

    Ok(EvictionReport {
        evicted_tiles,
        evicted_bytes,
        total_bytes_after: total_bytes,
    })
}

#[cfg(test)]
#[path = "tile_cache_tests.rs"]
mod tests;
