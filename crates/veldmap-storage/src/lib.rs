//! Durable atomic state persistence, injectable storage paths, user settings,
//! and crash-safe SQLite WAL LRU tile disk caching for VeldMap.

use thiserror::Error;

pub mod atomic;
pub mod paths;
pub mod settings;
pub mod tile_cache;

pub use atomic::{
    read_json_or_default, write_atomic_durable, write_atomic_durable_private, write_json_durable,
    write_json_durable_private,
};
pub use paths::StoragePaths;
pub use settings::{AppSettings, BasemapSelection, CustomXyzSettings, DEFAULT_GIBS_DATE};
pub use tile_cache::{
    DEFAULT_DISK_CACHE_BYTES, DiskCacheStats, DiskTileKey, EvictionReport, MAX_CACHED_TILE_BYTES,
    TileDiskCache, valid_source_key,
};

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
