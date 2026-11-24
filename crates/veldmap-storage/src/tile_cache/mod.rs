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
mod key;
mod queries;
mod schema;
mod store;

pub use self::key::{DiskTileKey, TileBlobEncoding, valid_source_key};
pub use self::store::{DiskCacheStats, EvictionReport, MAX_CACHED_TILE_BYTES, TileDiskCache};
