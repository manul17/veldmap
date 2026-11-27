//! Batched O(log N) SQLite LRU eviction and incremental vacuum / WAL
//! reclamation helpers for [`super::TileDiskCache`].

use rusqlite::{Connection, Transaction};

use super::queries::{TileEvictor, set_cache_meta_totals};

/// Eviction headroom divisor (`10` => evict down to `90%` of budget so
/// subsequent inserts do not trigger eviction on every single tile).
const EVICT_HEADROOM_DIVISOR: u64 = 10;

/// Maximum SQLite pages reclaimed per incremental eviction pass.
const INCREMENTAL_VACUUM_PAGES: i64 = 128;

/// Maximum batch size pulled per SQLite LRU eviction iteration.
pub(super) const EVICT_BATCH_ROWS: i64 = 128;

pub(super) fn reclaim_space(conn: &Connection, full_vacuum: bool) {
    if full_vacuum {
        let _ = conn.execute_batch("PRAGMA incremental_vacuum;");
        let _ = conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);");
    } else {
        let _ = conn.execute_batch(&format!(
            "PRAGMA incremental_vacuum({INCREMENTAL_VACUUM_PAGES});"
        ));
    }
}

/// Optional behavioral knobs for [`evict_in_tx`].
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct EvictOptions {
    /// Optional `tile_id` excluded from eviction (e.g. the newest inserted row).
    pub protect_id: Option<i64>,
    /// Start timestamp of the current multi-tile batch; entries at or after
    /// this timestamp only evict above `max_bytes`, never into the 90% headroom.
    pub batch_start_ts: Option<i64>,
    /// Write `cache_meta` even when zero rows were evicted (used after inserts).
    pub always_write_meta: bool,
}

/// Evict oldest tiles while `total_bytes > max_bytes`, returning the evicted
/// `tile_id`s.
///
/// Updates `cache_meta` once when `opts.always_write_meta` is true or when at
/// least one tile was evicted, avoiding duplicate `UPDATE cache_meta` writes on
/// evicting inserts.
pub(super) fn evict_in_tx(
    tx: &Transaction<'_>,
    mut total_bytes: u64,
    mut tile_count: u64,
    max_bytes: u64,
    opts: EvictOptions,
) -> Result<Vec<i64>, rusqlite::Error> {
    if total_bytes <= max_bytes {
        if opts.always_write_meta {
            set_cache_meta_totals(tx, total_bytes, tile_count)?;
        }
        return Ok(Vec::new());
    }

    let target = max_bytes.saturating_sub(max_bytes / EVICT_HEADROOM_DIVISOR);
    let mut evicted_ids = Vec::new();

    {
        let mut evictor = TileEvictor::new(tx, opts.protect_id)?;
        while total_bytes > target {
            let batch = evictor.select_batch(EVICT_BATCH_ROWS)?;
            if batch.is_empty() {
                break;
            }

            let mut to_delete = Vec::with_capacity(batch.len());
            let mut stopped_at_batch_boundary = false;
            for (tile_id, sz, last_used_ms) in batch {
                // Pre-batch entries evict down to `target` (90% headroom); entries
                // inserted by the current batch only evict while `total_bytes > max_bytes`
                // so a batch never self-evicts into headroom while still honoring the
                // hard `max_bytes` cap.
                let is_current_batch = opts
                    .batch_start_ts
                    .is_some_and(|start| last_used_ms >= start);
                let limit = if is_current_batch { max_bytes } else { target };
                if total_bytes <= limit {
                    if is_current_batch {
                        stopped_at_batch_boundary = true;
                    }
                    break;
                }
                to_delete.push(tile_id);
                evicted_ids.push(tile_id);
                total_bytes = total_bytes.saturating_sub(sz);
                tile_count = tile_count.saturating_sub(1);
            }
            if to_delete.is_empty() {
                break;
            }
            evictor.delete_ids(&to_delete)?;
            if stopped_at_batch_boundary {
                break;
            }
        }
    }

    if opts.always_write_meta || !evicted_ids.is_empty() {
        set_cache_meta_totals(tx, total_bytes, tile_count)?;
    }

    Ok(evicted_ids)
}
