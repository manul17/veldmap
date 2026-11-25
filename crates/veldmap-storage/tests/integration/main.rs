//! Grouped public-API integration tests for `veldmap-storage`:
//! `LocalFileSource` + BLAKE3 `LocalFingerprintCache`, durable atomic JSON
//! persistence, and multi-format `TileDiskCache` persistence across reopen.

use serde::{Deserialize, Serialize};
use veldmap_core::io::ByteSource;
use veldmap_core::sync::CancelToken;
use veldmap_core::tile::{BandKey, PixelFormat, SourceId, TileKey, TileScheme};
use veldmap_storage::test_support::TestTempDir;
use veldmap_storage::{
    DiskTileKey, LocalFileSource, LocalFingerprintCache, StoragePaths, TileBlobEncoding,
    TileDiskCache, fingerprint_byte_source, read_json_or_default_with_backup,
    write_json_durable_private,
};

#[derive(Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
struct TestDoc {
    name: String,
    limit: u64,
}

#[test]
fn local_file_source_fingerprinting_and_bypass_cache_work_end_to_end() {
    let dir = TestTempDir::new("storage-integ-fp");
    let path = dir.path().join("sample.bin");
    let mut bytes = vec![0xAAu8; 256 * 1024];
    std::fs::write(&path, &bytes).expect("write initial");

    let cancel = CancelToken::new();
    let src = LocalFileSource::open(&path).expect("open LocalFileSource");
    let direct_fp = fingerprint_byte_source(&src, &cancel).expect("fingerprint ByteSource");

    let fp_cache = LocalFingerprintCache::new();
    let (cached_src, cached_fp) = fp_cache
        .open_and_fingerprint(&path, false, &cancel)
        .expect("cached fingerprint");
    assert_eq!(cached_fp, direct_fp);
    assert_eq!(cached_fp.to_hex().len(), 64);
    assert_eq!(cached_src.len(), 256 * 1024);

    // Mutate midpoint bytes with identical length; bypass recomputes a distinct fingerprint.
    bytes[128 * 1024] ^= 0x55;
    std::fs::write(&path, &bytes).expect("rewrite same length");
    let (refreshed_src, refreshed_fp) = fp_cache
        .open_and_fingerprint(&path, true, &cancel)
        .expect("bypass recomputes fingerprint");
    assert_ne!(refreshed_fp, direct_fp);
    assert_eq!(refreshed_src.len(), 256 * 1024);
    assert_eq!(src.len(), 256 * 1024);
}

#[test]
fn durable_json_and_sqlite_disk_cache_persist_across_reopen() {
    let dir = TestTempDir::new("storage-integ-persist");
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

    let cancel = CancelToken::new();
    {
        let cache =
            TileDiskCache::open(paths.tile_cache_db_path(), 1024 * 1024).expect("open cache");
        cache
            .put_encoded_if_uncancelled(&disk_key, TileBlobEncoding::Lz4Raw, payload, &cancel)
            .expect("put encoded tile");
        assert_eq!(cache.stats().expect("stats").tile_count, 1);
    }

    let reopened =
        TileDiskCache::open(paths.tile_cache_db_path(), 1024 * 1024).expect("reopen cache");
    let (entry, encoding, got_payload) = reopened
        .get_encoded_entry(&disk_key, None)
        .expect("get encoded")
        .expect("tile present");
    reopened.touch_entry(entry);
    assert_eq!(encoding, TileBlobEncoding::Lz4Raw);
    assert_eq!(got_payload, payload);
    assert!(
        reopened
            .get_encoded_entry(&disk_key, Some(TileBlobEncoding::EncodedImage))
            .expect("filtered get")
            .is_none()
    );
    assert_eq!(reopened.clear_source("deadbeef").expect("clear source"), 1);
    assert_eq!(reopened.stats().expect("post clear stats").tile_count, 0);
}
