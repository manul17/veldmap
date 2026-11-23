//! Durable atomic state persistence, injectable storage paths, and crash-safe
//! SQLite WAL LRU tile disk caching for VeldMap.
//!
//! Focused strictly on storage mechanics: application-level preference and
//! session schemas live in `veldmap-app` and persist through
//! [`read_json_or_default_with_backup`] / [`write_json_durable_private`].

use thiserror::Error;

mod atomic;
mod paths;
mod tile_cache;

pub use atomic::{
    read_json_or_default, read_json_or_default_with_backup, write_atomic_durable,
    write_atomic_durable_private, write_json_durable, write_json_durable_private,
};
pub use paths::StoragePaths;
pub use tile_cache::{
    DEFAULT_DISK_CACHE_MAX_MB, DiskCacheStats, DiskTileKey, EvictionReport, MAX_CACHED_TILE_BYTES,
    MAX_DISK_CACHE_MAX_MB, MIN_DISK_CACHE_MAX_MB, TileBlobEncoding, TileDiskCache,
    valid_source_key,
};

#[cfg(test)]
pub(crate) struct TestTempDir(pub(crate) std::path::PathBuf);

#[cfg(test)]
impl TestTempDir {
    pub(crate) fn new(label: &str) -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        use std::time::{SystemTime, UNIX_EPOCH};
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "veldmap-storage-{label}-{}-{nanos}-{seq}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("create temp test dir");
        Self(dir)
    }

    pub(crate) fn path(&self) -> &std::path::Path {
        &self.0
    }
}

#[cfg(test)]
impl Drop for TestTempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Errors produced by durable storage and disk cache operations.
#[derive(Debug, Error)]
pub enum StorageError {
    #[error("unable to determine platform application directories")]
    NoPlatformDirs,
    #[error("invalid storage path: {0}")]
    InvalidPath(String),
    #[error("invalid tile cache source key: '{0}'")]
    InvalidSourceKey(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("SQLite cache error: {0}")]
    Sqlite(#[from] rusqlite::Error),
}
