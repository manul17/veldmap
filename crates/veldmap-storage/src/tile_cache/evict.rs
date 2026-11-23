//! Batched O(log N) SQLite LRU eviction and incremental vacuum / WAL
//! reclamation helpers for [`super::TileDiskCache`].

use rusqlite::{Connection, Transaction};

use super::EvictionReport;
use super::schema::{delete_meta_ids, select_evict_batch, set_cache_meta_totals};

/// Eviction headroom divisor (`10` => evict down to `90%` of budget so
/// subsequent inserts do not trigger eviction on every single tile).
const EVICT_HEADROOM_DIVISOR: u64 = 10;

/// Maximum SQLite pages reclaimed per incremental eviction pass.
const INCREMENTAL_VACUUM_PAGES: i64 = 128;

/// Maximum batch size pulled per SQLite LRU eviction iteration.
pub(super) const EVICT_BATCH_ROWS: i64 = 128;

pub(super) fn reclaim_space(conn: &Connection, is_file_backed: bool, full_vacuum: bool) {
    if full_vacuum {
        let _ = conn.execute_batch("PRAGMA incremental_vacuum;");
        if is_file_backed {
            let _ = conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);");
        }
    } else {
        let _ = conn.execute_batch(&format!(
            "PRAGMA incremental_vacuum({INCREMENTAL_VACUUM_PAGES});"
        ));
    }
}

pub(super) struct EvictionOutcome {
    pub(super) report: EvictionReport,
    pub(super) evicted_ids: Vec<i64>,
}

pub(super) fn evict_in_tx(
    tx: &Transaction<'_>,
    mut total_bytes: u64,
    mut tile_count: u64,
    max_bytes: u64,
    protect_id: Option<i64>,
) -> Result<EvictionOutcome, rusqlite::Error> {
    if total_bytes <= max_bytes {
        return Ok(EvictionOutcome {
            report: EvictionReport {
                evicted_tiles: 0,
                evicted_bytes: 0,
                total_bytes_after: total_bytes,
            },
            evicted_ids: Vec::new(),
        });
    }

    let target = max_bytes.saturating_sub(max_bytes / EVICT_HEADROOM_DIVISOR);
    let mut evicted_tiles = 0usize;
    let mut evicted_bytes = 0u64;
    let mut evicted_ids = Vec::new();

    while total_bytes > target {
        let batch = select_evict_batch(tx, EVICT_BATCH_ROWS, protect_id)?;
        if batch.is_empty() {
            break;
        }

        let mut to_delete = Vec::with_capacity(batch.len());
        for (tile_id, sz) in batch {
            if total_bytes <= target {
                break;
            }
            to_delete.push(tile_id);
            evicted_ids.push(tile_id);
            total_bytes = total_bytes.saturating_sub(sz);
            tile_count = tile_count.saturating_sub(1);
            evicted_bytes = evicted_bytes.saturating_add(sz);
            evicted_tiles += 1;
        }
        delete_meta_ids(tx, &to_delete)?;
    }

    set_cache_meta_totals(tx, total_bytes, tile_count)?;

    Ok(EvictionOutcome {
        report: EvictionReport {
            evicted_tiles,
            evicted_bytes,
            total_bytes_after: total_bytes,
        },
        evicted_ids,
    })
}
