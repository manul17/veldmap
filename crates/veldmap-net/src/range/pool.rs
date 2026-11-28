//! Fixed host-RAM-scaled 64 KiB O(1) LRU block pool and upfront async memory
//! permit tracker for pinned HTTP Range prefetch batches.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use tokio::sync::Notify;
use veldmap_core::io::IoError;
use veldmap_core::sync::{CancelToken, lock_unpoisoned};
use veldmap_core::tile::TileCache;

/// Fixed aligned block size (`64 KiB`) for HTTP Range caching and singleflight.
pub const RANGE_BLOCK_SIZE: usize = 64 * 1024;

/// Cache key inside the shared [`RangeBlockPool`].
///
/// Sources with a validated strong RFC 9110 ETag share immutable blocks across
/// re-opens via [`Self::Strong`]; sources with missing or weak (`W/"..."`) ETags
/// are strictly isolated per open instance via [`Self::Isolated`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) enum BlockPoolKey {
    Strong {
        canonical_url: Arc<str>,
        strong_etag: Arc<str>,
        block_idx: u64,
    },
    Isolated {
        instance_id: u64,
        block_idx: u64,
    },
}

/// Thread-safe O(1) LRU cache of aligned 64 KiB HTTP Range blocks bounded by
/// `MemoryBudget::range_block_pool_bytes`.
pub(crate) struct RangeBlockPool {
    inner: Mutex<TileCache<BlockPoolKey, Arc<[u8]>>>,
}

impl RangeBlockPool {
    pub(crate) fn new(cap_bytes: usize) -> Self {
        let clamped = if cap_bytes == 0 {
            0
        } else {
            cap_bytes.max(RANGE_BLOCK_SIZE)
        };
        Self {
            inner: Mutex::new(TileCache::new(clamped)),
        }
    }

    pub(super) fn get(&self, key: &BlockPoolKey) -> Option<Arc<[u8]>> {
        lock_unpoisoned(&self.inner).get(key).cloned()
    }

    pub(super) fn insert(&self, key: BlockPoolKey, data: Arc<[u8]>) {
        let len = data.len();
        let _evicted = lock_unpoisoned(&self.inner).insert(key, data, len);
    }

    #[cfg(test)]
    pub(super) fn stats(&self) -> (usize, usize) {
        let inner = lock_unpoisoned(&self.inner);
        (inner.len(), inner.total_bytes())
    }
}

struct PrefetchBudgetState {
    active_bytes: usize,
    next_ticket: u64,
    waiters: VecDeque<u64>,
}

/// RAII waiter guard removing a queued ticket from [`PrefetchBudgetState`] on
/// early cancel or future drop and waking remaining waiters outside the lock so
/// dropping the head waiter never stalls tasks queued behind it.
struct PrefetchWaiterGuard<'a> {
    budget: &'a PrefetchBudget,
    ticket: Option<u64>,
}

impl PrefetchWaiterGuard<'_> {
    fn disarm(&mut self) {
        self.ticket = None;
    }
}

impl Drop for PrefetchWaiterGuard<'_> {
    fn drop(&mut self) {
        let Some(ticket) = self.ticket.take() else {
            return;
        };
        let wake = {
            let mut st = lock_unpoisoned(&self.budget.state);
            let removed_head = if let Some(pos) = st.waiters.iter().position(|&t| t == ticket) {
                st.waiters.remove(pos);
                pos == 0
            } else {
                false
            };
            removed_head && !st.waiters.is_empty()
        };
        if wake {
            self.budget.notify.notify_waiters();
        }
    }
}

/// Async cancel-aware FIFO memory permit tracker bounding pinned coalesced
/// prefetch bytes across concurrent remote COG tile fetches.
pub(crate) struct PrefetchBudget {
    cap_bytes: usize,
    state: Mutex<PrefetchBudgetState>,
    notify: Notify,
}

impl PrefetchBudget {
    pub(crate) fn new(cap_bytes: usize) -> Self {
        Self {
            cap_bytes: cap_bytes.max(RANGE_BLOCK_SIZE),
            state: Mutex::new(PrefetchBudgetState {
                active_bytes: 0,
                next_ticket: 0,
                waiters: VecDeque::new(),
            }),
            notify: Notify::new(),
        }
    }

    pub(super) const fn cap_bytes(&self) -> usize {
        self.cap_bytes
    }

    pub(super) async fn reserve(
        self: &Arc<Self>,
        bytes: usize,
        cancel: &CancelToken,
    ) -> Result<PrefetchBudgetLease, IoError> {
        if bytes == 0 {
            return Ok(PrefetchBudgetLease {
                budget: Arc::clone(self),
                bytes: 0,
            });
        }
        if bytes > self.cap_bytes {
            return Err(IoError::InvalidInput(format!(
                "remote prefetch batch ({bytes} B) exceeds cap ({} B)",
                self.cap_bytes
            )));
        }
        if cancel.is_cancelled() {
            return Err(IoError::Cancelled);
        }

        // Fast path when no waiters are queued and capacity is available;
        // otherwise enqueue a FIFO ticket so smaller batches cannot starve a
        // waiting multi-megabyte prefetch.
        let ticket = {
            let mut st = lock_unpoisoned(&self.state);
            if st.waiters.is_empty() && bytes <= self.cap_bytes.saturating_sub(st.active_bytes) {
                st.active_bytes += bytes;
                return Ok(PrefetchBudgetLease {
                    budget: Arc::clone(self),
                    bytes,
                });
            }
            let ticket = st.next_ticket;
            st.next_ticket = st.next_ticket.wrapping_add(1);
            st.waiters.push_back(ticket);
            ticket
        };
        let mut waiter_guard = PrefetchWaiterGuard {
            budget: self,
            ticket: Some(ticket),
        };

        loop {
            if cancel.is_cancelled() {
                return Err(IoError::Cancelled);
            }
            let notified = self.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            let granted = {
                let mut st = lock_unpoisoned(&self.state);
                if st.waiters.front() == Some(&ticket)
                    && bytes <= self.cap_bytes.saturating_sub(st.active_bytes)
                {
                    st.waiters.pop_front();
                    st.active_bytes += bytes;
                    Some(!st.waiters.is_empty())
                } else {
                    None
                }
            };
            if let Some(wake_next) = granted {
                waiter_guard.disarm();
                if wake_next {
                    self.notify.notify_waiters();
                }
                return Ok(PrefetchBudgetLease {
                    budget: Arc::clone(self),
                    bytes,
                });
            }
            tokio::select! {
                biased;
                _ = cancel.cancelled() => return Err(IoError::Cancelled),
                _ = notified => {}
            }
        }
    }
}

/// RAII lease releasing reserved pinned prefetch bytes on drop.
pub(super) struct PrefetchBudgetLease {
    budget: Arc<PrefetchBudget>,
    bytes: usize,
}

impl Drop for PrefetchBudgetLease {
    fn drop(&mut self) {
        if self.bytes == 0 {
            return;
        }
        {
            let mut st = lock_unpoisoned(&self.budget.state);
            st.active_bytes = st.active_bytes.saturating_sub(self.bytes);
        }
        self.budget.notify.notify_waiters();
    }
}

#[cfg(test)]
#[path = "pool_tests.rs"]
mod tests;
