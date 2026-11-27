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
mod tests {
    use super::*;

    #[test]
    fn lru_pool_evicts_least_recently_used_and_respects_cap() {
        let pool = RangeBlockPool::new(2 * RANGE_BLOCK_SIZE);
        let k0 = BlockPoolKey::Isolated {
            instance_id: 1,
            block_idx: 0,
        };
        let k1 = BlockPoolKey::Isolated {
            instance_id: 1,
            block_idx: 1,
        };
        let k2 = BlockPoolKey::Isolated {
            instance_id: 1,
            block_idx: 2,
        };
        pool.insert(k0.clone(), Arc::from(vec![0u8; RANGE_BLOCK_SIZE]));
        pool.insert(k1.clone(), Arc::from(vec![1u8; RANGE_BLOCK_SIZE]));
        // Promote k0 so k1 becomes LRU.
        assert!(pool.get(&k0).is_some());
        pool.insert(k2.clone(), Arc::from(vec![2u8; RANGE_BLOCK_SIZE]));
        assert!(pool.get(&k0).is_some());
        assert!(pool.get(&k1).is_none());
        assert!(pool.get(&k2).is_some());
        assert_eq!(pool.stats(), (2, 2 * RANGE_BLOCK_SIZE));
    }

    #[test]
    fn prefetch_budget_enforces_cap_wakes_waiters_and_honors_cancel() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("rt");
        rt.block_on(async {
            let budget = Arc::new(PrefetchBudget::new(2 * RANGE_BLOCK_SIZE));
            let cancel = CancelToken::new();

            // Exceeding cap returns InvalidInput immediately.
            let over = budget
                .reserve(3 * RANGE_BLOCK_SIZE, &cancel)
                .await
                .err()
                .expect("over cap");
            assert!(matches!(over, IoError::InvalidInput(_)));

            // Acquire full budget, verify cancelled waiter aborts, then release wakes waiter.
            let lease_a = budget
                .reserve(2 * RANGE_BLOCK_SIZE, &cancel)
                .await
                .expect("lease_a");

            let cancel_waiter = CancelToken::new();
            let b_clone = Arc::clone(&budget);
            let cw_clone = cancel_waiter.clone();
            let cancelled_task =
                tokio::spawn(async move { b_clone.reserve(RANGE_BLOCK_SIZE, &cw_clone).await });
            tokio::task::yield_now().await;
            cancel_waiter.cancel();
            let res = cancelled_task.await.expect("join");
            assert!(matches!(res, Err(IoError::Cancelled)));

            let b_clone2 = Arc::clone(&budget);
            let ok_cancel = CancelToken::new();
            let ok_task =
                tokio::spawn(async move { b_clone2.reserve(RANGE_BLOCK_SIZE, &ok_cancel).await });
            tokio::task::yield_now().await;
            drop(lease_a);
            let lease_b = ok_task.await.expect("join").expect("woken lease");
            drop(lease_b);
        });
    }

    #[test]
    fn prefetch_budget_enforces_fifo_order_and_wakes_on_head_drop() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("rt");
        rt.block_on(async {
            let budget = Arc::new(PrefetchBudget::new(4 * RANGE_BLOCK_SIZE));
            let cancel = CancelToken::new();

            // Hold 2 blocks so 2 blocks remain free.
            let initial = budget
                .reserve(2 * RANGE_BLOCK_SIZE, &cancel)
                .await
                .expect("initial");

            // Large waiter (3 blocks) cannot fit yet and enters the FIFO queue.
            let b_large = Arc::clone(&budget);
            let c_large = CancelToken::new();
            let large_task = tokio::spawn(async move {
                b_large
                    .reserve(3 * RANGE_BLOCK_SIZE, &c_large)
                    .await
                    .expect("large")
            });
            tokio::task::yield_now().await;

            // Smaller waiter (1 block) would fit in the remaining 2 blocks, but
            // must wait behind the queued large waiter (FIFO non-starvation).
            let b_small = Arc::clone(&budget);
            let c_small = CancelToken::new();
            let small_task = tokio::spawn(async move {
                b_small
                    .reserve(RANGE_BLOCK_SIZE, &c_small)
                    .await
                    .expect("small")
            });
            tokio::task::yield_now().await;
            assert!(
                !small_task.is_finished(),
                "small waiter must not jump ahead of queued large waiter"
            );

            // Releasing `initial` wakes `large_task`, which acquires 3 blocks
            // and immediately wakes `small_task` for the remaining 1 block.
            drop(initial);
            let large_lease = large_task.await.expect("large join");
            let small_lease = small_task.await.expect("small join");
            drop(small_lease);
            drop(large_lease);

            // Dropping/aborting a queued head waiter must wake the next waiter.
            let hold = budget
                .reserve(2 * RANGE_BLOCK_SIZE, &cancel)
                .await
                .expect("hold");
            let b_drop = Arc::clone(&budget);
            let c_drop = CancelToken::new();
            let head_task =
                tokio::spawn(async move { b_drop.reserve(3 * RANGE_BLOCK_SIZE, &c_drop).await });
            tokio::task::yield_now().await;

            let b_tail = Arc::clone(&budget);
            let c_tail = CancelToken::new();
            let tail_task =
                tokio::spawn(async move { b_tail.reserve(RANGE_BLOCK_SIZE, &c_tail).await });
            tokio::task::yield_now().await;
            assert!(!tail_task.is_finished());

            head_task.abort();
            let _ = head_task.await;
            let tail_lease = tail_task.await.expect("tail join").expect("tail woken");
            drop(tail_lease);
            drop(hold);
        });
    }
}
