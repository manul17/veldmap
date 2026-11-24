//! Bounded LRU decoded-chunk cache with panic- and cancel-safe singleflight
//! deduplication across concurrent tile decode workers.

use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Mutex};

use veldmap_core::sync::{CancelToken, lock_unpoisoned};
use veldmap_core::tile::{SourceId, TileCache, TileStreamSpec};

use crate::RasterError;

use super::singleflight::SingleflightSlot;

/// Composite key identifying one decoded source chunk (TIFF tile/strip)
/// partitioned by `(source, spec, image, chunk_index)` so incompatible bands,
/// formats, gutters, or revisions never alias.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ChunkKey {
    pub source: SourceId,
    pub spec: TileStreamSpec,
    pub image: usize,
    pub chunk_index: u32,
}

/// Decoded RGBA8 chunk payload with its valid dimensions `(width, height)`.
#[derive(Clone, Debug)]
pub struct DecodedChunk {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

impl DecodedChunk {
    /// Byte size accounted against [`SharedChunkCache`].
    pub fn byte_size(&self) -> usize {
        self.data.len()
    }
}

#[derive(Clone)]
enum ChunkOutcome {
    Ready(Arc<DecodedChunk>),
    /// The leader's own `CancelToken` fired during decode; uncancelled
    /// followers should loop and elect a new leader instead of failing.
    LeaderCancelled,
    Failed(RasterError),
}

type ChunkSlot = SingleflightSlot<ChunkOutcome>;

struct LeaderGuard<'a> {
    state: &'a Mutex<CacheState>,
    key: ChunkKey,
    slot: Arc<ChunkSlot>,
    completed: bool,
}

impl Drop for LeaderGuard<'_> {
    fn drop(&mut self) {
        if self.completed {
            return;
        }
        // Remove the in-flight slot only if it still points to this leader's
        // `SingleflightSlot`, then wake any waiting followers cleanly.
        {
            let mut guard = lock_unpoisoned(self.state);
            if let Some(current) = guard.in_flight.get(&self.key)
                && Arc::ptr_eq(current, &self.slot)
            {
                guard.in_flight.remove(&self.key);
            }
        }
        self.slot
            .resolve(ChunkOutcome::Failed(RasterError::DecoderPanicked(
                "chunk decode leader panicked or exited prematurely".to_string(),
            )));
    }
}

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

    /// Configured maximum byte budget.
    pub fn max_bytes(&self) -> usize {
        lock_unpoisoned(&self.state).lru.max_bytes()
    }

    /// Currently cached bytes across all stored chunks.
    pub fn total_bytes(&self) -> usize {
        lock_unpoisoned(&self.state).lru.total_bytes()
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
    pub fn get_or_decode(
        &self,
        key: ChunkKey,
        cancel: &CancelToken,
        decode: impl FnOnce() -> Result<DecodedChunk, RasterError>,
    ) -> Result<Arc<DecodedChunk>, RasterError> {
        enum Role {
            Hit(Arc<DecodedChunk>),
            Leader(Arc<ChunkSlot>),
            Follower(Arc<ChunkSlot>),
        }

        loop {
            if cancel.is_cancelled() {
                return Err(RasterError::Cancelled);
            }

            let role = {
                let mut guard = lock_unpoisoned(&self.state);
                if let Some(chunk) = guard.lru.get(&key).map(Arc::clone) {
                    guard.hits += 1;
                    Role::Hit(chunk)
                } else if let Some(slot) = guard.in_flight.get(&key).map(Arc::clone) {
                    guard.dedup_waits += 1;
                    Role::Follower(slot)
                } else {
                    guard.misses += 1;
                    let slot = Arc::new(SingleflightSlot::new());
                    guard.in_flight.insert(key, Arc::clone(&slot));
                    Role::Leader(slot)
                }
            };

            match role {
                Role::Hit(chunk) => return Ok(chunk),
                Role::Follower(slot) => match slot.wait(cancel)? {
                    ChunkOutcome::Ready(chunk) => return Ok(chunk),
                    ChunkOutcome::LeaderCancelled => {
                        // The previous leader's token cancelled; loop to check
                        // our own token and either hit cache or elect a leader.
                        continue;
                    }
                    ChunkOutcome::Failed(_) if cancel.is_cancelled() => {
                        return Err(RasterError::Cancelled);
                    }
                    ChunkOutcome::Failed(err) => return Err(err),
                },
                Role::Leader(slot) => {
                    let mut guard = LeaderGuard {
                        state: &self.state,
                        key,
                        slot: Arc::clone(&slot),
                        completed: false,
                    };

                    let outcome = catch_unwind(AssertUnwindSafe(decode));
                    return match outcome {
                        Ok(Ok(chunk)) => {
                            let byte_size = chunk.byte_size();
                            let arc = Arc::new(chunk);
                            let evicted = {
                                let mut st = lock_unpoisoned(&self.state);
                                if let Some(cur) = st.in_flight.get(&key)
                                    && Arc::ptr_eq(cur, &slot)
                                {
                                    st.in_flight.remove(&key);
                                }
                                st.lru.insert(key, Arc::clone(&arc), byte_size).ok()
                            };
                            drop(evicted);
                            slot.resolve(ChunkOutcome::Ready(Arc::clone(&arc)));
                            guard.completed = true;
                            Ok(arc)
                        }
                        Ok(Err(err)) => {
                            {
                                let mut st = lock_unpoisoned(&self.state);
                                if let Some(cur) = st.in_flight.get(&key)
                                    && Arc::ptr_eq(cur, &slot)
                                {
                                    st.in_flight.remove(&key);
                                }
                            }
                            // Only treat cancellation as retryable when this
                            // leader's own token fired; an uncancelled leader
                            // returning `Err(Cancelled)` propagates once to
                            // avoid follower livelock.
                            let published = if err.is_cancelled() && cancel.is_cancelled() {
                                ChunkOutcome::LeaderCancelled
                            } else {
                                ChunkOutcome::Failed(err.clone())
                            };
                            slot.resolve(published);
                            guard.completed = true;
                            Err(err)
                        }
                        Err(_) => {
                            // Drop `guard` cleanly: removes `in_flight` and notifies waiters.
                            drop(guard);
                            Err(RasterError::DecoderPanicked(
                                "chunk decoder panicked".to_string(),
                            ))
                        }
                    };
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "chunk_cache_tests.rs"]
mod tests;
