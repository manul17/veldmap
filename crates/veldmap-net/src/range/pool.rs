//! Fixed host-RAM-scaled 64 KiB O(1) LRU block pool and upfront async memory
//! permit tracker for pinned HTTP Range prefetch batches.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use tokio::sync::Notify;
use veldmap_core::io::IoError;
use veldmap_core::sync::{CancelToken, lock_unpoisoned};

/// Fixed aligned block size (`64 KiB`) for HTTP Range caching and singleflight.
pub const RANGE_BLOCK_SIZE: usize = 64 * 1024;

/// Cache key inside the shared [`RangeBlockPool`].
///
/// Sources with a validated strong RFC 9110 ETag share immutable blocks across
/// re-opens via [`Self::Strong`]; sources with missing or weak (`W/"..."`) ETags
/// are strictly isolated per open instance via [`Self::Isolated`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum BlockPoolKey {
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

struct LruNode {
    key: BlockPoolKey,
    data: Arc<[u8]>,
    prev: Option<usize>,
    next: Option<usize>,
}

struct LruInner {
    cap_bytes: usize,
    bytes: usize,
    nodes: Vec<Option<LruNode>>,
    free: Vec<usize>,
    index: HashMap<BlockPoolKey, usize>,
    head: Option<usize>,
    tail: Option<usize>,
}

impl LruInner {
    fn new(cap_bytes: usize) -> Self {
        Self {
            cap_bytes,
            bytes: 0,
            nodes: Vec::new(),
            free: Vec::new(),
            index: HashMap::new(),
            head: None,
            tail: None,
        }
    }

    fn detach(&mut self, idx: usize) {
        let (prev, next) = match &self.nodes[idx] {
            Some(node) => (node.prev, node.next),
            None => return,
        };
        if let Some(p) = prev {
            if let Some(pnode) = &mut self.nodes[p] {
                pnode.next = next;
            }
        } else {
            self.head = next;
        }
        if let Some(n) = next {
            if let Some(nnode) = &mut self.nodes[n] {
                nnode.prev = prev;
            }
        } else {
            self.tail = prev;
        }
        if let Some(node) = &mut self.nodes[idx] {
            node.prev = None;
            node.next = None;
        }
    }

    fn attach_front(&mut self, idx: usize) {
        let old_head = self.head;
        if let Some(node) = &mut self.nodes[idx] {
            node.prev = None;
            node.next = old_head;
        }
        if let Some(h) = old_head {
            if let Some(hnode) = &mut self.nodes[h] {
                hnode.prev = Some(idx);
            }
        } else {
            self.tail = Some(idx);
        }
        self.head = Some(idx);
    }

    fn promote(&mut self, idx: usize) {
        if self.head == Some(idx) {
            return;
        }
        self.detach(idx);
        self.attach_front(idx);
    }

    fn evict_tail(&mut self) -> bool {
        let Some(tail_idx) = self.tail else {
            return false;
        };
        self.detach(tail_idx);
        if let Some(node) = self.nodes[tail_idx].take() {
            self.index.remove(&node.key);
            self.bytes = self.bytes.saturating_sub(node.data.len());
            self.free.push(tail_idx);
            true
        } else {
            false
        }
    }

    fn get(&mut self, key: &BlockPoolKey) -> Option<Arc<[u8]>> {
        let &idx = self.index.get(key)?;
        self.promote(idx);
        self.nodes[idx].as_ref().map(|n| Arc::clone(&n.data))
    }

    fn insert(&mut self, key: BlockPoolKey, data: Arc<[u8]>) {
        let len = data.len();
        if self.cap_bytes == 0 || len > self.cap_bytes {
            return;
        }
        if let Some(&idx) = self.index.get(&key) {
            if let Some(node) = &mut self.nodes[idx] {
                self.bytes = self.bytes.saturating_sub(node.data.len());
                node.data = data;
                self.bytes = self.bytes.saturating_add(len);
            }
            self.promote(idx);
        } else {
            let idx = if let Some(slot) = self.free.pop() {
                self.nodes[slot] = Some(LruNode {
                    key: key.clone(),
                    data,
                    prev: None,
                    next: None,
                });
                slot
            } else {
                let slot = self.nodes.len();
                self.nodes.push(Some(LruNode {
                    key: key.clone(),
                    data,
                    prev: None,
                    next: None,
                }));
                slot
            };
            self.index.insert(key, idx);
            self.bytes = self.bytes.saturating_add(len);
            self.attach_front(idx);
        }

        while self.bytes > self.cap_bytes {
            if !self.evict_tail() {
                break;
            }
        }
    }
}

/// Thread-safe O(1) LRU cache of aligned 64 KiB HTTP Range blocks bounded by
/// `MemoryBudget::range_block_pool_bytes`.
pub(crate) struct RangeBlockPool {
    inner: Mutex<LruInner>,
}

impl RangeBlockPool {
    pub(crate) fn new(cap_bytes: usize) -> Self {
        let clamped = if cap_bytes == 0 {
            0
        } else {
            cap_bytes.max(RANGE_BLOCK_SIZE)
        };
        Self {
            inner: Mutex::new(LruInner::new(clamped)),
        }
    }

    pub(crate) fn get(&self, key: &BlockPoolKey) -> Option<Arc<[u8]>> {
        lock_unpoisoned(&self.inner).get(key)
    }

    pub(crate) fn insert(&self, key: BlockPoolKey, data: Arc<[u8]>) {
        lock_unpoisoned(&self.inner).insert(key, data);
    }

    #[cfg(test)]
    pub(crate) fn stats(&self) -> (usize, usize) {
        let inner = lock_unpoisoned(&self.inner);
        (inner.index.len(), inner.bytes)
    }
}

struct PrefetchBudgetState {
    active_bytes: usize,
}

/// Async cancel-aware upfront memory permit tracker bounding pinned coalesced
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
            state: Mutex::new(PrefetchBudgetState { active_bytes: 0 }),
            notify: Notify::new(),
        }
    }

    pub(crate) const fn cap_bytes(&self) -> usize {
        self.cap_bytes
    }

    pub(crate) async fn reserve(
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
        loop {
            if cancel.is_cancelled() {
                return Err(IoError::Cancelled);
            }
            let notified = self.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            {
                let mut st = lock_unpoisoned(&self.state);
                if bytes <= self.cap_bytes.saturating_sub(st.active_bytes) {
                    st.active_bytes += bytes;
                    return Ok(PrefetchBudgetLease {
                        budget: Arc::clone(self),
                        bytes,
                    });
                }
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
pub(crate) struct PrefetchBudgetLease {
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
}
