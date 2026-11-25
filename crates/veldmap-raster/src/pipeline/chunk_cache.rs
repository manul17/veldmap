//! Bounded LRU decoded-chunk cache with panic- and cancel-safe singleflight
//! deduplication across concurrent tile decode workers.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use veldmap_core::sync::{CancelToken, lock_unpoisoned};
use veldmap_core::tile::{SourceId, TileCache, TileStreamSpec};

use crate::RasterError;

use super::singleflight::{SingleflightRole, SingleflightSlot, run_singleflight};

/// Composite key identifying one decoded source chunk (TIFF tile/strip)
/// partitioned by `(source, spec, image, chunk_index)` so incompatible bands,
/// formats, gutters, or revisions never alias.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ChunkKey {
    pub(crate) source: SourceId,
    pub(crate) spec: TileStreamSpec,
    pub(crate) image: usize,
    pub(crate) chunk_index: u32,
}

/// Decoded chunk payload with its pixel format and valid dimensions
/// `(width, height)`.
#[derive(Clone, Debug)]
pub(crate) struct DecodedChunk {
    pub(crate) data: Vec<u8>,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) format: veldmap_core::tile::PixelFormat,
}

impl DecodedChunk {
    /// Byte size accounted against [`SharedChunkCache`].
    fn byte_size(&self) -> usize {
        self.data.len()
    }
}

type ChunkSlot = SingleflightSlot<Arc<DecodedChunk>>;

struct CacheState {
    lru: TileCache<ChunkKey, Arc<DecodedChunk>>,
    in_flight: HashMap<ChunkKey, Arc<ChunkSlot>>,
    hits: u64,
    misses: u64,
    dedup_waits: u64,
}

/// Thread-safe bounded LRU decoded-chunk cache with panic/cancel-safe
/// singleflight deduplication.
pub struct SharedChunkCache {
    state: Mutex<CacheState>,
}

impl SharedChunkCache {
    /// Construct a new decoded-chunk cache bounded by `max_bytes`.
    pub fn new(max_bytes: usize) -> Self {
        Self {
            state: Mutex::new(CacheState {
                lru: TileCache::new(max_bytes),
                in_flight: HashMap::new(),
                hits: 0,
                misses: 0,
                dedup_waits: 0,
            }),
        }
    }

    /// Cache statistics `(hits, misses, dedup_waits)`.
    pub fn stats(&self) -> (u64, u64, u64) {
        let guard = lock_unpoisoned(&self.state);
        (guard.hits, guard.misses, guard.dedup_waits)
    }

    /// Remove all cached chunks belonging to `source`.
    pub fn invalidate_source(&self, source: SourceId) {
        let removed = lock_unpoisoned(&self.state)
            .lru
            .remove_where(|k| k.source == source);
        drop(removed);
    }

    /// Look up `key` in the cache, join an active singleflight decode if one is
    /// already running, or run `decode` once as the leader and publish to all
    /// waiters.
    ///
    /// If the leader cancels via its own `cancel` token, uncancelled followers
    /// retry and elect a new leader instead of inheriting a spurious
    /// [`RasterError::Cancelled`].
    pub(crate) fn get_or_decode(
        &self,
        key: ChunkKey,
        cancel: &CancelToken,
        decode: impl FnOnce() -> Result<DecodedChunk, RasterError>,
    ) -> Result<Arc<DecodedChunk>, RasterError> {
        run_singleflight(
            cancel,
            "chunk decode leader panicked",
            || {
                let mut guard = lock_unpoisoned(&self.state);
                if let Some(chunk) = guard.lru.get(&key).map(Arc::clone) {
                    guard.hits += 1;
                    SingleflightRole::Ready(chunk)
                } else if let Some(slot) = guard.in_flight.get(&key).map(Arc::clone) {
                    guard.dedup_waits += 1;
                    SingleflightRole::Follower(slot)
                } else {
                    guard.misses += 1;
                    let slot = Arc::new(SingleflightSlot::new());
                    guard.in_flight.insert(key, Arc::clone(&slot));
                    SingleflightRole::Leader(slot)
                }
            },
            || decode().map(Arc::new),
            |slot, arc| {
                let byte_size = arc.byte_size();
                let evicted = {
                    let mut st = lock_unpoisoned(&self.state);
                    if let Some(cur) = st.in_flight.get(&key)
                        && Arc::ptr_eq(cur, slot)
                    {
                        st.in_flight.remove(&key);
                    }
                    st.lru.insert(key, Arc::clone(arc), byte_size).ok()
                };
                drop(evicted);
            },
            |slot| {
                let mut st = lock_unpoisoned(&self.state);
                if let Some(cur) = st.in_flight.get(&key)
                    && Arc::ptr_eq(cur, slot)
                {
                    st.in_flight.remove(&key);
                }
            },
        )
    }
}

#[cfg(test)]
#[path = "chunk_cache_tests.rs"]
mod tests;
