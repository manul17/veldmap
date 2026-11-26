//! Durable atomic state persistence, injectable storage paths, platform host
//! RAM probing, positioned local file I/O, BLAKE3 fingerprinting, and
//! crash-safe SQLite WAL LRU tile disk caching for VeldMap.
//!
//! Focused strictly on storage mechanics: application-level preference and
//! session schemas live in `veldmap-app` and persist through
//! [`read_json_or_default_with_backup`] / [`write_json_durable_private`].

#![cfg_attr(not(target_os = "windows"), forbid(unsafe_code))]
#![cfg_attr(target_os = "windows", deny(unsafe_code))]

use thiserror::Error;
use veldmap_core::io::IoError;

mod atomic;
mod file_source;
mod fingerprint;
mod host;
mod paths;
mod tile_cache;

pub use atomic::{
    read_json_or_default_with_backup, write_atomic_durable, write_json_durable_private,
};
pub use file_source::{LocalFileSource, read_regular_file_utf8_limited};
pub use fingerprint::{LocalFingerprintCache, fingerprint_byte_source, fingerprint_remote_source};
pub use host::detect_memory_budget;
pub use paths::StoragePaths;
pub use tile_cache::{DiskCacheStats, DiskTileEntry, DiskTileKey, TileBlobEncoding, TileDiskCache};

#[cfg(any(test, feature = "test-support"))]
pub mod test_support;

/// Errors produced by durable storage and disk cache operations.
#[derive(Debug, Error)]
pub enum StorageError {
    #[error("storage operation cancelled")]
    Cancelled,
    #[error("unable to determine platform application directories")]
    NoPlatformDirs,
    #[error("invalid storage path: {0}")]
    InvalidPath(String),
    #[error("file '{path}' exceeds {limit_bytes} B limit")]
    FileTooLarge { path: String, limit_bytes: u64 },
    #[error("invalid tile cache source key: '{0}'")]
    InvalidSourceKey(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON serialization error: {0}")]
    Json(String),
    #[error("SQLite cache error: {0}")]
    Sqlite(String),
}

impl StorageError {
    /// Return whether this error represents task cancellation.
    pub const fn is_cancelled(&self) -> bool {
        matches!(self, Self::Cancelled)
    }
}

impl From<serde_json::Error> for StorageError {
    fn from(err: serde_json::Error) -> Self {
        Self::Json(err.to_string())
    }
}

impl From<rusqlite::Error> for StorageError {
    fn from(err: rusqlite::Error) -> Self {
        Self::Sqlite(err.to_string())
    }
}

impl From<IoError> for StorageError {
    fn from(err: IoError) -> Self {
        match err {
            IoError::Cancelled => Self::Cancelled,
            IoError::Io(io_err) => Self::Io(io_err),
            other => Self::Io(other.into_std_io()),
        }
    }
}
