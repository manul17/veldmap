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
mod schema;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{Connection, OptionalExtension, TransactionBehavior, params};
use veldmap_core::io::lock_unpoisoned;
use veldmap_core::tile::{BandKey, PixelFormat, TileKey, TileScheme};

use crate::StorageError;

use self::evict::{evict_in_tx, reclaim_space};
use self::schema::{
    init_connection, is_corrupt_sqlite_error, open_and_init_file_db, open_read_only_connection,
    remove_sqlite_files, sidecar_path,
};

/// Default disk cache cap in mebibytes (`512 MiB`).
pub const DEFAULT_DISK_CACHE_MAX_MB: u64 = 512;

/// Minimum configurable disk cache cap in mebibytes (`16 MiB`).
pub const MIN_DISK_CACHE_MAX_MB: u64 = 16;

/// Maximum configurable disk cache cap in mebibytes (`16 GiB`).
pub const MAX_DISK_CACHE_MAX_MB: u64 = 16 * 1024;

/// Default on-disk tile cache ceiling (`512 MiB`).
pub const DEFAULT_DISK_CACHE_BYTES: u64 = DEFAULT_DISK_CACHE_MAX_MB * 1024 * 1024;

/// Maximum encoded payload size admitted for a single cached tile (`16 MiB`).
pub const MAX_CACHED_TILE_BYTES: usize = 16 * 1024 * 1024;

/// Number of buffered `get()` LRU hits before attempting a best-effort batched
/// flush (avoids a WAL write transaction on every cache read).
const TOUCH_FLUSH_THRESHOLD: usize = 64;

/// Maximum pooled read-only WAL connections retained for concurrent lookups.
const MAX_READ_POOL_CONNS: usize = 4;

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

/// Pre-encoded 10-column composite key shared across all `tile_meta` queries.
struct EncodedKey<'a> {
    source_key: &'a str,
    band_kind: i64,
    band_id: i64,
    scheme: i64,
    format: i64,
    gutter: i64,
    decode_rev: i64,
    level: i64,
    x: i64,
    y: i64,
}

impl<'a> EncodedKey<'a> {
    fn from_key(key: &'a DiskTileKey) -> Result<Self, StorageError> {
        if !valid_source_key(&key.source_key) {
            return Err(StorageError::InvalidSourceKey(key.source_key.clone()));
        }
        let (band_kind, band_id) = encode_band(key.band);
        Ok(Self {
            source_key: &key.source_key,
            band_kind,
            band_id,
            scheme: encode_scheme(key.scheme),
            format: encode_format(key.format),
            gutter: i64::from(key.gutter),
            decode_rev: i64::from(key.decode_rev),
            level: i64::from(key.level),
            x: i64::from(key.x),
            y: i64::from(key.y),
        })
    }

    fn key_params(&self) -> [&dyn rusqlite::ToSql; 10] {
        [
            &self.source_key,
            &self.band_kind,
            &self.band_id,
            &self.scheme,
            &self.format,
            &self.gutter,
            &self.decode_rev,
            &self.level,
            &self.x,
            &self.y,
        ]
    }
}

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

    /// Open an isolated in-memory SQLite tile cache (primarily for unit tests).
    pub fn open_in_memory(max_bytes: u64) -> Result<Self, StorageError> {
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
            self.record_touch(tile_id);
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
        let touches = self.take_pending_touches();
        let report = {
            let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
            flush_touches_in_tx(&tx, &touches)?;

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
    pub fn evict_if_needed(&self) -> Result<EvictionReport, StorageError> {
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
        let touches = self.take_pending_touches();
        let report = {
            let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
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
        let touches = self.take_pending_touches();
        let deleted = {
            let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
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
