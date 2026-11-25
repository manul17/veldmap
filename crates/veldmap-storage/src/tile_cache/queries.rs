//! Runtime DML statements for [`super::TileDiskCache`].

use std::collections::HashMap;

use rusqlite::{Connection, OptionalExtension, Params, Statement, Transaction, params};

use super::key::{EncodedKey, TileBlobEncoding};

macro_rules! key_columns_sql {
    () => {
        "source_key, band_kind, band_id, scheme, format, gutter, decode_rev, level, x, y"
    };
}

macro_rules! key_where_sql {
    () => {
        "source_key = ?1
       AND band_kind = ?2
       AND band_id = ?3
       AND scheme = ?4
       AND format = ?5
       AND gutter = ?6
       AND decode_rev = ?7
       AND level = ?8
       AND x = ?9
       AND y = ?10"
    };
}

/// Truncate-friendly deletion of all cached blobs and metadata.
const CLEAR_BLOBS_SQL: &str = "DELETE FROM tile_blobs";
const CLEAR_META_SQL: &str = "DELETE FROM tile_meta";

const SELECT_TILE_BLOB_SQL: &str = concat!(
    "SELECT m.tile_id, m.encoding, m.last_used_ms, b.payload
     FROM tile_meta m
     JOIN tile_blobs b ON b.tile_id = m.tile_id
     WHERE ",
    key_where_sql!()
);

const SELECT_META_BYTE_SIZE_SQL: &str = concat!(
    "SELECT byte_size FROM tile_meta
     WHERE ",
    key_where_sql!()
);

const UPSERT_META_SQL: &str = concat!(
    "INSERT INTO tile_meta (",
    key_columns_sql!(),
    ", encoding, byte_size, last_used_ms)
     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)
     ON CONFLICT(",
    key_columns_sql!(),
    ") DO UPDATE SET
       encoding = excluded.encoding,
       byte_size = excluded.byte_size,
       last_used_ms = excluded.last_used_ms
     RETURNING tile_id"
);

const UPSERT_BLOB_SQL: &str = "INSERT INTO tile_blobs (tile_id, payload)
     VALUES (?1, ?2)
     ON CONFLICT(tile_id) DO UPDATE SET payload = excluded.payload";

const TOUCH_META_SQL: &str =
    "UPDATE tile_meta SET last_used_ms = ?1 WHERE tile_id = ?2 AND last_used_ms < ?1";

const DELETE_META_BY_KEY_SQL: &str = concat!(
    "DELETE FROM tile_meta
     WHERE ",
    key_where_sql!(),
    "
     RETURNING tile_id, byte_size"
);

const DELETE_META_BY_ID_SQL: &str = "DELETE FROM tile_meta WHERE tile_id = ?1";

const DELETE_META_BY_ID_AND_TS_SQL: &str =
    "DELETE FROM tile_meta WHERE tile_id = ?1 AND last_used_ms = ?2 RETURNING byte_size";

const DELETE_SOURCE_META_SQL: &str =
    "DELETE FROM tile_meta WHERE source_key = ?1 RETURNING tile_id, byte_size";

const SELECT_EVICT_BATCH_SQL: &str = "SELECT tile_id, byte_size, last_used_ms FROM tile_meta
     ORDER BY last_used_ms ASC
     LIMIT ?1";

const SELECT_EVICT_BATCH_PROTECTED_SQL: &str =
    "SELECT tile_id, byte_size, last_used_ms FROM tile_meta
     WHERE tile_id != ?1
     ORDER BY last_used_ms ASC
     LIMIT ?2";

const READ_CACHE_META_SQL: &str = "SELECT total_bytes, tile_count FROM cache_meta WHERE id = 1";

const SET_CACHE_META_SQL: &str =
    "UPDATE cache_meta SET total_bytes = ?1, tile_count = ?2 WHERE id = 1";

const DECREMENT_CACHE_META_SQL: &str = "UPDATE cache_meta
     SET total_bytes = MAX(0, total_bytes - ?1),
         tile_count  = MAX(0, tile_count - ?2)
     WHERE id = 1";

const RESET_CACHE_META_SQL: &str =
    "UPDATE cache_meta SET total_bytes = 0, tile_count = 0 WHERE id = 1";

pub(super) struct SelectedTileBlob {
    pub(super) tile_id: i64,
    pub(super) encoding_raw: i64,
    pub(super) last_used_ms: i64,
    pub(super) payload: Vec<u8>,
}

pub(super) fn select_tile_blob(
    conn: &Connection,
    enc: &EncodedKey<'_>,
) -> Result<Option<SelectedTileBlob>, rusqlite::Error> {
    let mut stmt = conn.prepare(SELECT_TILE_BLOB_SQL)?;
    stmt.query_row(enc.key_params(), |r| {
        Ok(SelectedTileBlob {
            tile_id: r.get(0)?,
            encoding_raw: r.get(1)?,
            last_used_ms: r.get(2)?,
            payload: r.get(3)?,
        })
    })
    .optional()
}

/// Transaction-scoped prepared statements for upserting one or more tiles
/// without re-preparing SQL on each entry.
pub(super) struct TileUpserter<'tx> {
    select_byte_size: Statement<'tx>,
    upsert_meta: Statement<'tx>,
    upsert_blob: Statement<'tx>,
}

impl<'tx> TileUpserter<'tx> {
    pub(super) fn new(tx: &'tx Transaction<'_>) -> Result<Self, rusqlite::Error> {
        Ok(Self {
            select_byte_size: tx.prepare(SELECT_META_BYTE_SIZE_SQL)?,
            upsert_meta: tx.prepare(UPSERT_META_SQL)?,
            upsert_blob: tx.prepare(UPSERT_BLOB_SQL)?,
        })
    }

    pub(super) fn upsert(
        &mut self,
        enc: &EncodedKey<'_>,
        encoding: TileBlobEncoding,
        payload: &[u8],
        ts_ms: i64,
    ) -> Result<(i64, u64, u64), rusqlite::Error> {
        let new_bytes = payload.len() as u64;
        let prev_bytes: Option<i64> = self
            .select_byte_size
            .query_row(enc.key_params(), |r| r.get(0))
            .optional()?;
        let encoding_raw = i64::from(encoding as u8);
        let byte_size = new_bytes as i64;
        let inserted_id: i64 = self
            .upsert_meta
            .query_row(enc.upsert_params(&encoding_raw, &byte_size, &ts_ms), |r| {
                r.get(0)
            })?;
        self.upsert_blob.execute(params![inserted_id, payload])?;

        let old = prev_bytes.unwrap_or(0).max(0) as u64;
        let count_delta = u64::from(prev_bytes.is_none());
        Ok((inserted_id, old, count_delta))
    }
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

pub(super) fn delete_tile_in_tx(
    tx: &Transaction<'_>,
    enc: &EncodedKey<'_>,
) -> Result<Option<i64>, rusqlite::Error> {
    let mut stmt = tx.prepare(DELETE_META_BY_KEY_SQL)?;
    let deleted: Option<(i64, i64)> = stmt
        .query_row(enc.key_params(), |r| Ok((r.get(0)?, r.get(1)?)))
        .optional()?;
    let Some((tile_id, sz)) = deleted else {
        return Ok(None);
    };
    decrement_cache_meta(tx, sz.max(0), 1)?;
    Ok(Some(tile_id))
}

pub(super) fn delete_entry_in_tx(
    tx: &Transaction<'_>,
    tile_id: i64,
    expected_last_used_ms: i64,
) -> Result<bool, rusqlite::Error> {
    let mut stmt = tx.prepare(DELETE_META_BY_ID_AND_TS_SQL)?;
    let deleted_sz: Option<i64> = stmt
        .query_row(params![tile_id, expected_last_used_ms], |r| r.get(0))
        .optional()?;
    let Some(sz) = deleted_sz else {
        return Ok(false);
    };
    decrement_cache_meta(tx, sz.max(0), 1)?;
    Ok(true)
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

pub(super) fn clear_source_in_tx(
    tx: &Transaction<'_>,
    source_key: &str,
) -> Result<Vec<i64>, rusqlite::Error> {
    let (ids, removed_bytes) = {
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
        (ids, removed_bytes)
    };
    if !ids.is_empty() {
        decrement_cache_meta(tx, removed_bytes, ids.len() as i64)?;
    }
    Ok(ids)
}

pub(super) fn clear_all_in_tx(tx: &Transaction<'_>) -> Result<(), rusqlite::Error> {
    // Delete blobs first to avoid per-row cascade work and allow SQLite's
    // table-truncate optimization.
    tx.execute(CLEAR_BLOBS_SQL, [])?;
    tx.execute(CLEAR_META_SQL, [])?;
    tx.execute(RESET_CACHE_META_SQL, [])?;
    Ok(())
}

fn collect_evict_rows(
    stmt: &mut Statement<'_>,
    params: impl Params,
    capacity: usize,
) -> Result<Vec<(i64, u64, i64)>, rusqlite::Error> {
    let mut batch = Vec::with_capacity(capacity);
    let rows = stmt.query_map(params, |r| {
        let tile_id: i64 = r.get(0)?;
        let sz: i64 = r.get(1)?;
        let last_used_ms: i64 = r.get(2)?;
        Ok((tile_id, sz.max(0) as u64, last_used_ms))
    })?;
    for row in rows {
        batch.push(row?);
    }
    Ok(batch)
}

pub(super) fn select_evict_batch(
    tx: &Transaction<'_>,
    limit: i64,
    protect_id: Option<i64>,
) -> Result<Vec<(i64, u64, i64)>, rusqlite::Error> {
    // `ORDER BY last_used_ms ASC` (without `, tile_id ASC`) lets SQLite satisfy
    // the query directly from covering index `idx_tile_meta_lru (last_used_ms,
    // byte_size)` with zero temp B-tree sorting and zero table lookups.
    let capacity = limit.max(0) as usize;
    if let Some(protected) = protect_id {
        let mut stmt = tx.prepare(SELECT_EVICT_BATCH_PROTECTED_SQL)?;
        collect_evict_rows(&mut stmt, params![protected, limit], capacity)
    } else {
        let mut stmt = tx.prepare(SELECT_EVICT_BATCH_SQL)?;
        collect_evict_rows(&mut stmt, params![limit], capacity)
    }
}

pub(super) fn read_cache_meta_totals(conn: &Connection) -> Result<(u64, u64), rusqlite::Error> {
    let mut stmt = conn.prepare(READ_CACHE_META_SQL)?;
    let (total_bytes, tile_count): (i64, i64) =
        stmt.query_row([], |r| Ok((r.get(0)?, r.get(1)?)))?;
    Ok((total_bytes.max(0) as u64, tile_count.max(0) as u64))
}

pub(super) fn set_cache_meta_totals(
    tx: &Transaction<'_>,
    total_bytes: u64,
    tile_count: u64,
) -> Result<(), rusqlite::Error> {
    let mut stmt = tx.prepare(SET_CACHE_META_SQL)?;
    stmt.execute(params![total_bytes as i64, tile_count as i64])?;
    Ok(())
}

fn decrement_cache_meta(
    tx: &Transaction<'_>,
    removed_bytes: i64,
    removed_count: i64,
) -> Result<(), rusqlite::Error> {
    let mut stmt = tx.prepare(DECREMENT_CACHE_META_SQL)?;
    stmt.execute(params![removed_bytes, removed_count])?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::TestTempDir;
    use crate::tile_cache::evict::EVICT_BATCH_ROWS;
    use crate::tile_cache::schema::open_and_init_file_db;

    #[test]
    fn eviction_query_uses_covering_lru_index_without_temp_btree() {
        let dir = TestTempDir::new("queries-covering-idx");
        let db_path = dir.path().join("tiles.sqlite3");
        let (conn, _) = open_and_init_file_db(&db_path).expect("open db");

        let mut stmt = conn
            .prepare(&format!(
                "EXPLAIN QUERY PLAN {SELECT_EVICT_BATCH_PROTECTED_SQL}"
            ))
            .expect("explain");
        let details: Vec<String> = stmt
            .query_map(params![1i64, EVICT_BATCH_ROWS], |r| r.get::<_, String>(3))
            .expect("query plan")
            .map(|r| r.expect("plan row"))
            .collect();
        let joined = details.join(" | ");
        assert!(
            joined.contains("idx_tile_meta_lru") && joined.contains("COVERING"),
            "expected covering idx_tile_meta_lru scan, got: {joined}"
        );
        assert!(
            !joined.contains("TEMP B-TREE"),
            "unexpected temp B-tree sort in eviction plan: {joined}"
        );
    }
}
