//! Upfront-budgeted, coalesced, pinned HTTP Range block prefetching for
//! [`super::HttpRangeSource`].

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use veldmap_core::io::IoError;
use veldmap_core::sync::{CancelToken, lock_unpoisoned};

use crate::http::link_cancel_token;

use super::pool::{PrefetchBudgetLease, RANGE_BLOCK_SIZE};
use super::{HttpRangeSource, RangeInner};

/// Maximum gap (in blocks) bridged when coalescing remote COG chunk prefetch
/// ranges (`1` block = `64 KiB`).
const MAX_PREFETCH_GAP_BLOCKS: u64 = 1;

/// RAII guard pinning prefetched 64 KiB blocks in memory for the duration of a
/// CPU decode job so LRU eviction in the shared range block pool can never
/// cause unexpected HTTP I/O on the fast decode lane.
pub struct PrefetchPin {
    inner: Arc<RangeInner>,
    pinned_blocks: Vec<u64>,
    _budget_lease: PrefetchBudgetLease,
}

impl PrefetchPin {
    /// Number of 64 KiB blocks pinned by this guard.
    pub fn block_count(&self) -> usize {
        self.pinned_blocks.len()
    }
}

impl Drop for PrefetchPin {
    fn drop(&mut self) {
        if self.pinned_blocks.is_empty() {
            return;
        }
        let mut pins = lock_unpoisoned(&self.inner.active_pins);
        for &blk in &self.pinned_blocks {
            if let Some((refcount, _)) = pins.get_mut(&blk) {
                *refcount = refcount.saturating_sub(1);
                if *refcount == 0 {
                    pins.remove(&blk);
                }
            }
        }
    }
}

fn plan_prefetch_blocks(
    inner: &RangeInner,
    ranges: &[(u64, u64)],
    cap_bytes: usize,
) -> Result<(Vec<u64>, usize), IoError> {
    let max_blocks_in_cap = (cap_bytes.div_ceil(RANGE_BLOCK_SIZE) as u64).saturating_add(1);
    let mut unique_blocks = BTreeSet::new();
    let mut total_pin_bytes = 0usize;

    for &(offset, len) in ranges {
        if len == 0 || offset >= inner.len {
            continue;
        }
        let clamped_end = offset.saturating_add(len).min(inner.len);
        if clamped_end <= inner.pinned_head.len() as u64 {
            continue;
        }
        let first = offset / RANGE_BLOCK_SIZE as u64;
        let last = (clamped_end - 1) / RANGE_BLOCK_SIZE as u64;
        if last.saturating_sub(first).saturating_add(1) > max_blocks_in_cap {
            return Err(IoError::InvalidInput(format!(
                "remote prefetch range ({len} B) exceeds cap ({cap_bytes} B)"
            )));
        }
        for blk in first..=last {
            let start = blk.saturating_mul(RANGE_BLOCK_SIZE as u64);
            let blen = inner.expected_block_len(blk);
            if start.saturating_add(blen as u64) <= inner.pinned_head.len() as u64 {
                continue;
            }
            if unique_blocks.insert(blk) {
                total_pin_bytes = total_pin_bytes.saturating_add(blen);
                if total_pin_bytes > cap_bytes {
                    return Err(IoError::InvalidInput(format!(
                        "remote prefetch batch ({total_pin_bytes} B) exceeds cap ({cap_bytes} B)"
                    )));
                }
            }
        }
    }

    Ok((unique_blocks.into_iter().collect(), total_pin_bytes))
}

impl HttpRangeSource {
    /// Asynchronously coalesce, fetch, and pin all 64 KiB blocks covering
    /// `ranges` (`(offset, len)` pairs) under an upfront memory permit from
    /// [`crate::NetContext`].
    ///
    /// While the returned [`PrefetchPin`] is alive, subsequent synchronous
    /// [`veldmap_core::io::ByteSource::read_at`] calls for any byte inside
    /// `ranges` are guaranteed to hit RAM without issuing HTTP requests.
    pub async fn prefetch_ranges_async(
        &self,
        ranges: &[(u64, u64)],
        cancel: &CancelToken,
    ) -> Result<PrefetchPin, IoError> {
        let (linked_cancel, _cancel_guard) = link_cancel_token(&self.inner.shutdown, cancel);
        if linked_cancel.is_cancelled() {
            return Err(IoError::Cancelled);
        }

        let cap_bytes = self.inner.prefetch_budget.cap_bytes();
        let (needed_blocks, total_pin_bytes) =
            plan_prefetch_blocks(&self.inner, ranges, cap_bytes)?;

        let budget_lease = self
            .inner
            .prefetch_budget
            .reserve(total_pin_bytes, &linked_cancel)
            .await?;

        let mut resolved: HashMap<u64, Arc<[u8]>> = HashMap::with_capacity(needed_blocks.len());
        let mut missing = Vec::new();
        for &blk in &needed_blocks {
            if let Some(data) = self.inner.lookup_cached_block(blk) {
                resolved.insert(blk, data);
            } else {
                missing.push(blk);
            }
        }

        if !missing.is_empty() {
            let fetched = self
                .inner
                .fetch_blocks_singleflight(&missing, MAX_PREFETCH_GAP_BLOCKS, &linked_cancel)
                .await?;
            for (blk, data) in fetched {
                resolved.insert(blk, data);
            }
        }

        let mut pinned_blocks = Vec::with_capacity(resolved.len());
        {
            let mut pins = lock_unpoisoned(&self.inner.active_pins);
            for &blk in &needed_blocks {
                if let Some(data) = resolved.get(&blk) {
                    pins.entry(blk)
                        .and_modify(|(rc, _)| *rc = rc.saturating_add(1))
                        .or_insert_with(|| (1, Arc::clone(data)));
                    pinned_blocks.push(blk);
                }
            }
        }

        Ok(PrefetchPin {
            inner: Arc::clone(&self.inner),
            pinned_blocks,
            _budget_lease: budget_lease,
        })
    }

    /// Blocking wrapper around [`Self::prefetch_ranges_async`], enforcing that
    /// it is never invoked from inside a Tokio runtime thread.
    pub fn prefetch_ranges(
        &self,
        ranges: &[(u64, u64)],
        cancel: &CancelToken,
    ) -> Result<PrefetchPin, IoError> {
        if tokio::runtime::Handle::try_current().is_ok() {
            return Err(IoError::InvalidInput(
                "blocking HttpRangeSource::prefetch_ranges called inside Tokio runtime".to_owned(),
            ));
        }
        self.inner
            .handle
            .block_on(self.prefetch_ranges_async(ranges, cancel))
    }
}
