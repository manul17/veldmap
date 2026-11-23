//! Grouped public-API integration tests for `veldmap-storage`:
//! `LocalFileSource` + BLAKE3 `LocalFingerprintCache`, durable atomic JSON
//! persistence, and multi-format `TileDiskCache` persistence across reopen.

#![expect(
    clippy::expect_used,
    reason = "INVARIANT: integration test harness helpers panic on setup failure"
)]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use veldmap_core::io::ByteSource;
use veldmap_core::sync::CancelToken;
use veldmap_core::tile::{BandKey, PixelFormat, SourceId, TileKey, TileScheme};
use veldmap_storage::{
    DiskTileKey, LocalFileSource, LocalFingerprintCache, StoragePaths, TileBlobEncoding,
    TileDiskCache, fingerprint_byte_source, read_json_or_default_with_backup,
    write_json_durable_private,
};

struct TempDir(PathBuf);

impl TempDir {
    fn new(label: &str) -> Self {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "veldmap-storage-integ-{label}-{}-{nanos}-{seq}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        Self(dir)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[derive(Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
struct TestDoc {
    name: String,
    limit: u64,
}

#[test]
fn local_file_source_fingerprinting_and_bypass_cache_work_end_to_end() {
    let dir = TempDir::new("fp");
    let path = dir.path().join("sample.bin");
    let mut bytes = vec![0xAAu8; 256 * 1024];
    std::fs::write(&path, &bytes).expect("write initial");

    let cancel = CancelToken::new();
    let src = LocalFileSource::open(&path).expect("open LocalFileSource");
    let direct_fp = fingerprint_byte_source(&src, &cancel).expect("fingerprint ByteSource");

    let fp_cache = LocalFingerprintCache::new();
    let cached_fp = fp_cache
        .get_or_compute(&path, &cancel)
        .expect("cached fingerprint");
    assert_eq!(cached_fp, direct_fp);
    assert_eq!(cached_fp.to_hex().len(), 64);

    // Mutate midpoint bytes with identical length; bypass recomputes a distinct fingerprint.
    bytes[128 * 1024] ^= 0x55;
    std::fs::write(&path, &bytes).expect("rewrite same length");
    let refreshed_fp = fp_cache
        .get_or_compute_with_bypass(&path, true, &cancel)
        .expect("bypass recomputes fingerprint");
    assert_ne!(refreshed_fp, direct_fp);
    assert_eq!(src.len(), 256 * 1024);
}

#[test]
fn durable_json_and_sqlite_disk_cache_persist_across_reopen() {
    let dir = TempDir::new("persist");
    let paths = StoragePaths::with_root(dir.path());
    let doc_path = paths.config_dir.join("session.json");
    std::fs::create_dir_all(&paths.config_dir).expect("create config dir");
    std::fs::create_dir_all(&paths.cache_dir).expect("create cache dir");

    let doc = TestDoc {
        name: "sentinel-aoi".to_owned(),
        limit: 42,
    };
    write_json_durable_private(&doc_path, &doc).expect("durable json write");
    let loaded: TestDoc = read_json_or_default_with_backup(&doc_path);
    assert_eq!(loaded, doc);

    let key = TileKey {
        source: SourceId::new(77),
        band: BandKey::from_name("B08"),
        scheme: TileScheme::RasterPyramid { max_level: 3 },
        format: PixelFormat::R32Float,
        gutter: 0,
        decode_rev: 1,
        level: 0,
        x: 2,
        y: 1,
    };
    let disk_key = DiskTileKey::from_tile_key("deadbeef", &key);
    let payload = b"lz4-scientific-tile-bytes";

    {
        let cache =
            TileDiskCache::open(paths.tile_cache_db_path(), 1024 * 1024).expect("open cache");
        cache
            .put_encoded(&disk_key, TileBlobEncoding::Lz4Raw, payload)
            .expect("put encoded tile");
        assert_eq!(cache.stats().expect("stats").tile_count, 1);
    }

    let reopened =
        TileDiskCache::open(paths.tile_cache_db_path(), 1024 * 1024).expect("reopen cache");
    assert_eq!(
        reopened.get_encoded(&disk_key).expect("get encoded"),
        Some((TileBlobEncoding::Lz4Raw, payload.to_vec()))
    );
    assert!(reopened.get(&disk_key).expect("filtered get").is_none());
    assert_eq!(reopened.clear_source("deadbeef").expect("clear source"), 1);
    assert_eq!(reopened.stats().expect("post clear stats").tile_count, 0);
}
