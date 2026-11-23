use std::sync::Arc;
use std::thread;
use std::time::Duration;

use veldmap_core::tile::{BandKey, PixelFormat, TileKey, TileScheme};

use super::evict::EVICT_BATCH_ROWS;
use super::schema::{SCHEMA_VERSION, SQLITE_BUSY_TIMEOUT_MS, SQLITE_PAGE_SIZE};
use super::*;
use crate::test_support::TestTempDir;

fn open_in_memory_cache(max_bytes: u64) -> Result<TileDiskCache, StorageError> {
    let conn = Connection::open_in_memory()?;
    let max_ts = init_connection(&conn, false, true)?;
    Ok(TileDiskCache::from_connection(
        conn, None, max_bytes, max_ts,
    ))
}

fn key(source: &str, level: u8, x: u32, y: u32) -> DiskTileKey {
    DiskTileKey::new(source, TileScheme::WebMercator, level, x, y)
}

fn count_rows(cache: &TileDiskCache, table: &str) -> i64 {
    let conn = lock_unpoisoned(&cache.writer);
    conn.query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |r| r.get(0))
        .expect("count rows")
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
    let cache = open_in_memory_cache(100).expect("open mem cache");
    let k_a = key("eox", 12, 0, 0);
    let k_b = key("eox", 12, 1, 0);
    let k_c = key("eox", 12, 2, 0);
    let k_d = key("eox", 12, 3, 0);

    cache.put(&k_a, &[1u8; 40]).expect("put a");
    cache.put(&k_b, &[2u8; 20]).expect("put b");
    cache.put(&k_c, &[3u8; 40]).expect("put c");
    assert_eq!(cache.stats().expect("stats").payload_bytes, 100);
    assert_eq!(cache.stats().expect("stats").tile_count, 3);
    assert_eq!(cache.write_generation(), 3);

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
    assert_eq!(count_rows(&cache, "tile_meta"), 2);
    assert_eq!(count_rows(&cache, "tile_blobs"), 2);
}

#[test]
fn pragmas_schema_version_and_covering_lru_index_are_configured() {
    let cache = open_in_memory_cache(4096).expect("open mem");
    let conn = lock_unpoisoned(&cache.writer);

    let page_size: i64 = conn
        .query_row("PRAGMA page_size;", [], |r| r.get(0))
        .expect("page_size");
    let foreign_keys: i64 = conn
        .query_row("PRAGMA foreign_keys;", [], |r| r.get(0))
        .expect("foreign_keys");
    let user_version: i64 = conn
        .query_row("PRAGMA user_version;", [], |r| r.get(0))
        .expect("user_version");
    let busy_timeout: i64 = conn
        .query_row("PRAGMA busy_timeout;", [], |r| r.get(0))
        .expect("busy_timeout");
    assert_eq!(page_size, SQLITE_PAGE_SIZE);
    assert_eq!(foreign_keys, 1);
    assert_eq!(user_version, SCHEMA_VERSION);
    assert_eq!(busy_timeout, SQLITE_BUSY_TIMEOUT_MS);

    let mut stmt = conn
        .prepare(
            "EXPLAIN QUERY PLAN
             SELECT tile_id, byte_size FROM tile_meta
             WHERE tile_id != ?2
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
        joined.contains("idx_tile_meta_lru") && joined.contains("COVERING"),
        "expected covering idx_tile_meta_lru scan, got: {joined}"
    );
    assert!(
        !joined.contains("TEMP B-TREE"),
        "unexpected temp B-tree sort in eviction plan: {joined}"
    );
}

#[test]
fn recovers_from_corrupt_or_legacy_v1_sqlite_file_and_persists_across_reopen() {
    let dir = TestTempDir::new("sqlite-recovery");
    let db_path = dir.path().join("tiles.sqlite3");

    // 1. Legacy v1 WAL DB (`user_version = 0`, single `tiles` table, 4 KiB pages)
    //    upgrades cleanly to v2 with 16 KiB pages and zero stale data.
    {
        let legacy = Connection::open(&db_path).expect("open legacy");
        legacy
            .execute_batch(
                "PRAGMA journal_mode = WAL;
                 CREATE TABLE tiles (
                     source_key TEXT, scheme INTEGER, level INTEGER, x INTEGER, y INTEGER,
                     payload BLOB, byte_size INTEGER, last_used_ms INTEGER
                 );
                 INSERT INTO tiles VALUES ('old', 1, 0, 0, 0, X'DEADBEEF', 4, 1);",
            )
            .expect("seed legacy");
    }
    {
        let upgraded = TileDiskCache::open(&db_path, 4096).expect("upgrade legacy db");
        assert_eq!(upgraded.stats().expect("stats").tile_count, 0);
        let conn = lock_unpoisoned(&upgraded.writer);
        let page_size: i64 = conn
            .query_row("PRAGMA page_size;", [], |r| r.get(0))
            .expect("upgraded page_size");
        let user_version: i64 = conn
            .query_row("PRAGMA user_version;", [], |r| r.get(0))
            .expect("upgraded user_version");
        assert_eq!(page_size, SQLITE_PAGE_SIZE);
        assert_eq!(user_version, SCHEMA_VERSION);
    }

    // 2. Garbage bytes simulating a corrupted cache file recover cleanly.
    std::fs::write(&db_path, b"not a valid sqlite database file!").expect("write corrupt");

    let cache = TileDiskCache::open(&db_path, 4096).expect("recover corrupt db");
    assert_eq!(cache.stats().expect("stats").tile_count, 0);

    let k1 = key("gibs", 3, 2, 1);
    cache.put(&k1, b"png-bytes").expect("put k1");
    let ts_before = cache.last_ts_ms.load(Ordering::Acquire);
    drop(cache);

    // 3. Reopen and verify persistence + monotonic timestamp seed + shrinking + clear.
    let reopened = TileDiskCache::open(&db_path, 4096).expect("reopen");
    assert!(
        reopened.last_ts_ms.load(Ordering::Acquire) >= ts_before,
        "reopen must seed last_ts_ms from MAX(last_used_ms)"
    );
    assert_eq!(
        reopened.get(&k1).expect("get k1"),
        Some(b"png-bytes".to_vec())
    );

    let ev = reopened.set_max_bytes(4).expect("shrink cap");
    assert_eq!(ev.evicted_tiles, 1);
    assert_eq!(reopened.stats().expect("stats").payload_bytes, 0);
    assert_eq!(reopened.stats().expect("stats").tile_count, 0);
    assert_eq!(count_rows(&reopened, "tile_blobs"), 0);

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
    assert_eq!(count_rows(&reopened, "tile_blobs"), 0);
    assert!(
        after_clear.file_bytes < before_clear,
        "clear() should reclaim space (before={before_clear}, after={})",
        after_clear.file_bytes
    );
}

#[test]
fn file_backed_open_waits_for_brief_external_lock_and_serves_concurrent_wal_readers() {
    let dir = TestTempDir::new("sqlite-wal-busy");
    let db_path = dir.path().join("tiles.sqlite3");

    let initial = TileDiskCache::open(&db_path, 1024 * 1024).expect("initial open");
    let k0 = key("eox", 2, 1, 1);
    initial.put(&k0, b"cached-payload").expect("seed tile");
    drop(initial);

    // Hold an exclusive transaction briefly on a separate connection; opening
    // `TileDiskCache` must wait via `busy_timeout` instead of failing immediately
    // with `SQLITE_BUSY` on `PRAGMA page_count`/`user_version`.
    let lock_path = db_path.clone();
    let (ready_tx, ready_rx) = std::sync::mpsc::channel();
    let locker = thread::spawn(move || {
        let mut conn = Connection::open(&lock_path).expect("open locker");
        let tx = conn
            .transaction_with_behavior(TransactionBehavior::Exclusive)
            .expect("exclusive tx");
        let _ = ready_tx.send(());
        thread::sleep(Duration::from_millis(80));
        tx.commit().expect("commit locker");
    });
    ready_rx.recv().expect("locker acquired");

    let cache =
        Arc::new(TileDiskCache::open(&db_path, 1024 * 1024).expect("open under brief lock"));
    locker.join().expect("join locker");

    // Verify file-backed reads work concurrently even while the writer lock is held.
    let writer_guard = lock_unpoisoned(&cache.writer);
    let reader_cache = Arc::clone(&cache);
    let reader = thread::spawn(move || {
        assert_eq!(
            reader_cache.get(&k0).expect("concurrent wal read"),
            Some(b"cached-payload".to_vec())
        );
        assert_eq!(
            reader_cache.stats().expect("concurrent stats").tile_count,
            1
        );
    });
    reader
        .join()
        .expect("reader completed without blocking on writer");
    drop(writer_guard);
}

#[test]
fn multi_band_format_gutter_and_encoding_never_alias() {
    let cache = open_in_memory_cache(16 * 1024).expect("open mem");
    let source = "sentinel2-s36uvb";
    let base = TileKey {
        source: veldmap_core::tile::SourceId::from_stable_key(source),
        band: BandKey::from_name("B04"),
        scheme: TileScheme::RasterPyramid { max_level: 4 },
        format: PixelFormat::R32Float,
        gutter: 0,
        decode_rev: 1,
        level: 0,
        x: 3,
        y: 5,
    };
    let k_b04 = DiskTileKey::from_tile_key(source, &base);
    let k_b08 = DiskTileKey::from_tile_key(
        source,
        &TileKey {
            band: BandKey::from_name("B08"),
            ..base
        },
    );
    let k_scl = DiskTileKey::from_tile_key(
        source,
        &TileKey {
            band: BandKey::from_name("SCL"),
            format: PixelFormat::R8Uint,
            ..base
        },
    );
    let k_dem_g1 = DiskTileKey::from_tile_key(source, &TileKey { gutter: 1, ..base });

    cache
        .put_encoded(&k_b04, TileBlobEncoding::Lz4Raw, b"b04-lz4")
        .expect("put b04");
    cache
        .put_encoded(&k_b08, TileBlobEncoding::Lz4Raw, b"b08-lz4")
        .expect("put b08");
    cache
        .put_encoded(&k_scl, TileBlobEncoding::Lz4Raw, b"scl-lz4")
        .expect("put scl");
    cache
        .put_encoded(&k_dem_g1, TileBlobEncoding::Qoi, b"dem-g1")
        .expect("put dem");

    assert_eq!(
        cache.get_encoded(&k_b04).expect("get b04"),
        Some((TileBlobEncoding::Lz4Raw, b"b04-lz4".to_vec()))
    );
    assert_eq!(
        cache.get_encoded(&k_b08).expect("get b08"),
        Some((TileBlobEncoding::Lz4Raw, b"b08-lz4".to_vec()))
    );
    assert_eq!(
        cache.get_encoded(&k_scl).expect("get scl"),
        Some((TileBlobEncoding::Lz4Raw, b"scl-lz4".to_vec()))
    );
    assert_eq!(
        cache.get_encoded(&k_dem_g1).expect("get dem"),
        Some((TileBlobEncoding::Qoi, b"dem-g1".to_vec()))
    );
    // `get()` only returns `EncodedImage` entries and must never hand Qoi/Lz4Raw
    // payloads to a raw-image caller.
    assert_eq!(cache.get(&k_b04).expect("get b04 raw"), None);
    assert_eq!(cache.get(&k_dem_g1).expect("get dem raw"), None);
    assert_eq!(cache.stats().expect("stats").tile_count, 4);
    assert_eq!(count_rows(&cache, "tile_blobs"), 4);
}

#[test]
fn overwrite_accounting_skipped_puts_and_multi_batch_eviction_stay_exact() {
    let cache = open_in_memory_cache(10_000).expect("open mem");
    let k = key("eox", 5, 1, 1);

    // Overwrite with larger then smaller payloads; byte & tile_count stay exact.
    cache.put(&k, &[1u8; 100]).expect("put 100");
    assert_eq!(cache.stats().expect("s1").payload_bytes, 100);
    assert_eq!(cache.stats().expect("s1").tile_count, 1);
    cache.put(&k, &[2u8; 350]).expect("put 350");
    assert_eq!(cache.stats().expect("s2").payload_bytes, 350);
    assert_eq!(cache.stats().expect("s2").tile_count, 1);
    cache.put(&k, &[3u8; 50]).expect("put 50");
    assert_eq!(cache.stats().expect("s3").payload_bytes, 50);
    assert_eq!(cache.stats().expect("s3").tile_count, 1);
    assert_eq!(count_rows(&cache, "tile_blobs"), 1);

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
    let post = cache.stats().expect("post-evict");
    assert_eq!(post.payload_bytes, ev.total_bytes_after);
    assert_eq!(post.tile_count, (200 - ev.evicted_tiles) as u64);
    assert_eq!(count_rows(&cache, "tile_blobs") as u64, post.tile_count);
}

#[test]
fn clear_source_removes_only_matching_source_without_orphan_blobs() {
    let cache = open_in_memory_cache(4096).expect("open mem");
    let k1 = key("gibs", 1, 0, 0);
    let k2 = key("eox", 1, 0, 0);
    cache.put(&k1, &[1u8; 100]).expect("put k1");
    cache.put(&k2, &[2u8; 200]).expect("put k2");

    let removed = cache.clear_source("gibs").expect("clear gibs");
    assert_eq!(removed, 1);
    assert!(cache.get(&k1).expect("get k1").is_none());
    assert!(cache.get(&k2).expect("get k2").is_some());
    let stats = cache.stats().expect("stats");
    assert_eq!(stats.payload_bytes, 200);
    assert_eq!(stats.tile_count, 1);
    assert_eq!(count_rows(&cache, "tile_blobs"), 1);
    assert_eq!(cache.pending_touches.lock().unwrap().len(), 1);

    assert_eq!(cache.clear_source("eox").expect("clear eox"), 1);
    assert!(
        cache.pending_touches.lock().unwrap().is_empty(),
        "clear_source must leave pending_touches empty for deleted tile_ids"
    );
}

#[test]
fn large_tile_near_budget_never_self_evicts_and_delete_updates_total() {
    let cache = open_in_memory_cache(100).expect("open mem");
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
    let stats = cache.stats().expect("stats");
    assert_eq!(stats.payload_bytes, 0);
    assert_eq!(stats.tile_count, 0);
    assert_eq!(count_rows(&cache, "tile_blobs"), 0);
}

#[test]
fn concurrent_reads_and_writes_keep_budget_and_accounting_consistent() {
    let cache = Arc::new(open_in_memory_cache(4096).expect("open mem"));
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
    assert_eq!(count_rows(&cache, "tile_blobs") as u64, stats.tile_count);
}
