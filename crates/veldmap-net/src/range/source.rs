//! Thread-safe HTTP Range [`ByteSource`] implementation with consumed-only
//! sequential readahead, pinned head caching, shared LRU pooling, and
//! multi-block singleflight.

use std::collections::HashMap;
use std::fmt;
#[cfg(any(test, feature = "test-support"))]
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use reqwest::header::{HeaderName, HeaderValue};
use veldmap_core::io::{ByteSource, IoError};
use veldmap_core::sync::{CancelToken, lock_unpoisoned};
use veldmap_core::tile::SourceFingerprint;

use crate::context::NetContext;
use crate::http::{link_cancel_token, parse_custom_headers};
use crate::limiter::HostLimiters;
use crate::redact::{EncodedSecret, sanitize_url_for_display};
use crate::url::extract_host;

use super::config::{HttpRangeConfig, RangeOpenError};
use super::pool::{BlockPoolKey, PrefetchBudget, RANGE_BLOCK_SIZE, RangeBlockPool};
use super::protocol::fingerprint_remote_source;
use super::singleflight::{
    BlockFlight, LeaderCleanupGuard, SharedRangeError, SingleflightTable, WaiterGuard,
};
use super::transport::{RangeFetchMode, RangeFetchRequest, fetch_range_http};

/// Maximum blocks coalesced into one HTTP Range GET (`64 * 64 KiB = 4 MiB`).
const MAX_COALESCED_SPAN_BLOCKS: u64 = 64;

#[derive(Default)]
struct ReadaheadTracker {
    last_consumed_block: Option<u64>,
    sequential_streak: u64,
}

impl ReadaheadTracker {
    fn extra_blocks(&mut self, first_block: u64, last_block: u64, max_ra: u64) -> u64 {
        if self
            .last_consumed_block
            .is_some_and(|prev| first_block == prev.saturating_add(1))
        {
            self.sequential_streak = self.sequential_streak.saturating_add(1);
        } else if self
            .last_consumed_block
            .is_some_and(|prev| first_block == prev && last_block == prev)
        {
            // Repeated sub-block read within the same 64 KiB block keeps the
            // current sequential streak intact without advancing it.
        } else {
            self.sequential_streak = 0;
        }
        self.last_consumed_block = Some(last_block);

        if max_ra == 0 || self.sequential_streak == 0 {
            0
        } else {
            (1u64 << (self.sequential_streak.min(4) - 1)).min(max_ra)
        }
    }
}

type ActivePinMap = HashMap<u64, (usize, Arc<[u8]>)>;

/// Immutable remote object identity and HTTP transport metadata for a single
/// [`HttpRangeSource`] instance.
struct RangeMeta {
    url: String,
    canonical_url: Arc<str>,
    host: String,
    len: u64,
    strong_etag: Option<Arc<str>>,
    instance_id: u64,
    fingerprint: SourceFingerprint,
    extra_headers: Vec<(HeaderName, HeaderValue)>,
    secrets: Vec<EncodedSecret>,
    max_readahead_blocks: u64,
    max_retries: usize,
}

/// Internal shared state for [`HttpRangeSource`].
///
/// Intentionally holds cloned inner `Arc`s / handles rather than
/// `Arc<NetContext>` so spawned Tokio singleflight tasks can never drop the
/// last `NetContext` (and its `tokio::runtime::Runtime`) on a Tokio worker.
pub(super) struct RangeInner {
    handle: tokio::runtime::Handle,
    shutdown: CancelToken,
    client: reqwest::Client,
    host_limiters: Arc<HostLimiters>,
    range_pool: Arc<RangeBlockPool>,
    prefetch_budget: Arc<PrefetchBudget>,
    meta: RangeMeta,
    pinned_head: Arc<[u8]>,
    singleflight: Arc<SingleflightTable>,
    active_pins: Mutex<ActivePinMap>,
    readahead: Mutex<ReadaheadTracker>,
    #[cfg(any(test, feature = "test-support"))]
    http_requests: AtomicU64,
    #[cfg(any(test, feature = "test-support"))]
    bytes_delivered: AtomicU64,
}

impl RangeInner {
    #[cfg(any(test, feature = "test-support"))]
    pub(super) fn handle(&self) -> &tokio::runtime::Handle {
        &self.handle
    }

    pub(super) fn shutdown(&self) -> &CancelToken {
        &self.shutdown
    }

    pub(super) fn prefetch_budget(&self) -> &Arc<PrefetchBudget> {
        &self.prefetch_budget
    }

    pub(super) const fn len(&self) -> u64 {
        self.meta.len
    }

    pub(super) fn pinned_head_len(&self) -> u64 {
        self.pinned_head.len() as u64
    }

    fn pool_key(&self, block_idx: u64) -> BlockPoolKey {
        match &self.meta.strong_etag {
            Some(etag) => BlockPoolKey::Strong {
                canonical_url: Arc::clone(&self.meta.canonical_url),
                strong_etag: Arc::clone(etag),
                block_idx,
            },
            None => BlockPoolKey::Isolated {
                instance_id: self.meta.instance_id,
                block_idx,
            },
        }
    }

    fn total_blocks(&self) -> u64 {
        self.meta.len.div_ceil(RANGE_BLOCK_SIZE as u64)
    }

    pub(super) fn expected_block_len(&self, block_idx: u64) -> usize {
        let start = block_idx.saturating_mul(RANGE_BLOCK_SIZE as u64);
        if start >= self.meta.len {
            return 0;
        }
        (self.meta.len - start).min(RANGE_BLOCK_SIZE as u64) as usize
    }

    pub(super) fn lookup_cached_block(&self, block_idx: u64) -> Option<Arc<[u8]>> {
        let start = block_idx.saturating_mul(RANGE_BLOCK_SIZE as u64);
        let expected_len = self.expected_block_len(block_idx);
        if expected_len == 0 {
            return None;
        }
        if start.saturating_add(expected_len as u64) <= self.pinned_head_len() {
            let s = start as usize;
            return Some(Arc::from(&self.pinned_head[s..s + expected_len]));
        }
        if let Some((_, data)) = lock_unpoisoned(&self.active_pins).get(&block_idx) {
            return Some(Arc::clone(data));
        }
        self.range_pool.get(&self.pool_key(block_idx))
    }

    pub(super) fn pin_blocks(
        &self,
        needed_blocks: &[u64],
        resolved: &HashMap<u64, Arc<[u8]>>,
    ) -> Vec<u64> {
        let mut pinned_blocks = Vec::with_capacity(resolved.len());
        let mut pins = lock_unpoisoned(&self.active_pins);
        for &blk in needed_blocks {
            if let Some(data) = resolved.get(&blk) {
                pins.entry(blk)
                    .and_modify(|(rc, _)| *rc = rc.saturating_add(1))
                    .or_insert_with(|| (1, Arc::clone(data)));
                pinned_blocks.push(blk);
            }
        }
        pinned_blocks
    }

    pub(super) fn unpin_blocks(&self, blocks: &[u64]) {
        if blocks.is_empty() {
            return;
        }
        let mut pins = lock_unpoisoned(&self.active_pins);
        for &blk in blocks {
            if let Some((refcount, _)) = pins.get_mut(&blk) {
                *refcount = refcount.saturating_sub(1);
                if *refcount == 0 {
                    pins.remove(&blk);
                }
            }
        }
    }

    fn insert_into_pool(&self, block_idx: u64, data: &Arc<[u8]>) {
        self.range_pool
            .insert(self.pool_key(block_idx), Arc::clone(data));
    }

    async fn fetch_block_span_http(
        &self,
        start_block: u64,
        end_block: u64,
        cancel: &CancelToken,
    ) -> Result<HashMap<u64, Arc<[u8]>>, SharedRangeError> {
        let byte_start = start_block.saturating_mul(RANGE_BLOCK_SIZE as u64);
        let byte_end_excl = end_block
            .saturating_add(1)
            .saturating_mul(RANGE_BLOCK_SIZE as u64)
            .min(self.meta.len);
        if byte_start >= byte_end_excl {
            return Ok(HashMap::new());
        }
        let byte_end_incl = byte_end_excl - 1;

        let req_cfg = RangeFetchRequest {
            client: &self.client,
            host_limiters: &self.host_limiters,
            url: &self.meta.url,
            host: &self.meta.host,
            extra_headers: &self.meta.extra_headers,
            secrets: &self.meta.secrets,
            max_retries: self.meta.max_retries,
            mode: RangeFetchMode::BlockSpan {
                byte_start,
                byte_end_incl,
                total_len: self.meta.len,
                strong_etag: self.meta.strong_etag.as_deref(),
            },
            #[cfg(any(test, feature = "test-support"))]
            request_counter: Some(&self.http_requests),
        };
        let resp = fetch_range_http(&req_cfg, cancel).await?;
        let body = resp.body;

        #[cfg(any(test, feature = "test-support"))]
        self.bytes_delivered
            .fetch_add(body.len() as u64, Ordering::Relaxed);

        let mut blocks = HashMap::with_capacity((end_block - start_block + 1) as usize);
        let mut cursor = 0usize;
        for blk in start_block..=end_block {
            let blen = self.expected_block_len(blk);
            if blen == 0 || cursor + blen > body.len() {
                break;
            }
            let slice: Arc<[u8]> = Arc::from(&body[cursor..cursor + blen]);
            self.insert_into_pool(blk, &slice);
            blocks.insert(blk, slice);
            cursor += blen;
        }
        Ok(blocks)
    }

    pub(super) async fn fetch_blocks_singleflight(
        self: &Arc<Self>,
        missing_blocks: &[u64],
        max_gap_blocks: u64,
        caller_cancel: &CancelToken,
    ) -> Result<HashMap<u64, Arc<[u8]>>, IoError> {
        if missing_blocks.is_empty() {
            return Ok(HashMap::new());
        }
        let plan = self.singleflight.claim_spans(
            missing_blocks,
            max_gap_blocks,
            MAX_COALESCED_SPAN_BLOCKS,
            &self.shutdown,
        );

        let mut flights_to_wait: Vec<(Arc<BlockFlight>, WaiterGuard)> = plan.joined;

        for span in plan.started {
            let start_block = span.blocks[0];
            let end_block = *span.blocks.last().unwrap_or(&start_block);
            let inner = Arc::clone(self);
            let flight = Arc::clone(&span.flight);
            let leader_guard: LeaderCleanupGuard = span.leader_guard;
            self.handle.spawn(async move {
                let outcome = inner
                    .fetch_block_span_http(start_block, end_block, flight.cancel_token())
                    .await
                    .map(Arc::new);
                leader_guard.finish(outcome);
            });
            flights_to_wait.push((span.flight, span.waiter));
        }

        let mut collected = HashMap::with_capacity(missing_blocks.len());
        for (flight, waiter) in flights_to_wait {
            let blocks = flight.wait(waiter, caller_cancel).await?;
            for &blk in missing_blocks {
                if let Some(data) = blocks.get(&blk) {
                    collected.insert(blk, Arc::clone(data));
                }
            }
        }
        Ok(collected)
    }
}

/// Thread-safe HTTP Range [`ByteSource`] backed by [`NetContext`].
#[derive(Clone)]
pub struct HttpRangeSource {
    inner: Arc<RangeInner>,
}

impl fmt::Debug for HttpRangeSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HttpRangeSource")
            .field("url", &sanitize_url_for_display(&self.inner.meta.url))
            .field("len", &self.inner.meta.len)
            .field("has_strong_etag", &self.inner.meta.strong_etag.is_some())
            .field("fingerprint", &self.inner.meta.fingerprint)
            .finish()
    }
}

impl HttpRangeSource {
    pub(super) fn inner(&self) -> &Arc<RangeInner> {
        &self.inner
    }

    /// Open a remote HTTP(S) Range object asynchronously with a single adaptive
    /// `0..head_probe_bytes` GET (416-safe on 0-byte files).
    pub async fn open_async(
        net: Arc<NetContext>,
        url: &str,
        config: HttpRangeConfig,
        cancel: &CancelToken,
    ) -> Result<Self, RangeOpenError> {
        let (linked_cancel, _cancel_guard) = link_cancel_token(net.shutdown_token(), cancel);
        if linked_cancel.is_cancelled() {
            return Err(RangeOpenError::Cancelled);
        }
        let trimmed_url = url.trim();
        let canonical = crate::url::sanitize_url_for_persistence(trimmed_url)
            .ok_or_else(|| RangeOpenError::InvalidUrl(sanitize_url_for_display(trimmed_url)))?;
        let host = extract_host(trimmed_url);

        let parsed = parse_custom_headers(&config.extra_headers)
            .map_err(|(name, reason)| RangeOpenError::InvalidHeader { name, reason })?;
        let parsed_headers = parsed.headers;
        let secrets = parsed.secrets;

        let client = net.range_client_for(!parsed_headers.is_empty()).clone();
        let host_limiters = net.host_limiters();
        let req_cfg = RangeFetchRequest {
            client: &client,
            host_limiters: &host_limiters,
            url: trimmed_url,
            host: &host,
            extra_headers: &parsed_headers,
            secrets: &secrets,
            max_retries: config.max_retries,
            mode: RangeFetchMode::OpenProbe {
                probe_len: config.head_probe_bytes.max(1),
            },
            #[cfg(any(test, feature = "test-support"))]
            request_counter: None,
        };
        let resp = fetch_range_http(&req_cfg, &linked_cancel).await?;

        let fingerprint = fingerprint_remote_source(
            &canonical,
            resp.total_len,
            resp.strong_etag.as_deref(),
            &resp.body,
        );
        let instance_id = net.alloc_source_instance_id();
        #[cfg(any(test, feature = "test-support"))]
        let delivered = resp.body.len() as u64;

        let meta = RangeMeta {
            url: trimmed_url.to_owned(),
            canonical_url: Arc::from(canonical),
            host,
            len: resp.total_len,
            strong_etag: resp.strong_etag.map(Arc::from),
            instance_id,
            fingerprint,
            extra_headers: parsed_headers,
            secrets,
            max_readahead_blocks: config.max_readahead_blocks,
            max_retries: config.max_retries,
        };

        let inner = Arc::new(RangeInner {
            handle: net.handle().clone(),
            shutdown: net.shutdown_token().clone(),
            client,
            host_limiters,
            range_pool: net.range_pool(),
            prefetch_budget: net.prefetch_budget(),
            meta,
            pinned_head: Arc::from(resp.body),
            singleflight: Arc::new(SingleflightTable::new()),
            active_pins: Mutex::new(HashMap::new()),
            readahead: Mutex::new(ReadaheadTracker::default()),
            #[cfg(any(test, feature = "test-support"))]
            http_requests: AtomicU64::new(resp.requests_made),
            #[cfg(any(test, feature = "test-support"))]
            bytes_delivered: AtomicU64::new(delivered),
        });

        Ok(Self { inner })
    }

    /// Blocking wrapper around [`Self::open_async`], enforcing that it is never
    /// invoked from inside a Tokio worker thread.
    #[cfg(any(test, feature = "test-support"))]
    pub fn open(
        net: Arc<NetContext>,
        url: &str,
        config: HttpRangeConfig,
        cancel: &CancelToken,
    ) -> Result<Self, RangeOpenError> {
        if tokio::runtime::Handle::try_current().is_ok() {
            return Err(RangeOpenError::CalledInsideTokioRuntime);
        }
        let handle = net.handle().clone();
        handle.block_on(Self::open_async(net, url, config, cancel))
    }

    /// Canonical URL with lowercased scheme/authority and credentials/query
    /// stripped.
    #[cfg(any(test, feature = "test-support"))]
    pub fn canonical_url(&self) -> &str {
        &self.inner.meta.canonical_url
    }

    /// Validated strong RFC 9110 `ETag`, if supplied by the server.
    #[cfg(any(test, feature = "test-support"))]
    pub fn strong_etag(&self) -> Option<&str> {
        self.inner.meta.strong_etag.as_deref()
    }

    /// Deterministic [`SourceFingerprint`] derived during open with zero extra
    /// HTTP round-trips.
    pub fn fingerprint(&self) -> SourceFingerprint {
        self.inner.meta.fingerprint
    }

    /// Total HTTP GET requests issued by this source (including open probe).
    #[cfg(any(test, feature = "test-support"))]
    pub fn http_requests(&self) -> u64 {
        self.inner.http_requests.load(Ordering::Relaxed)
    }

    /// Total HTTP response payload bytes delivered to this source.
    #[cfg(any(test, feature = "test-support"))]
    pub fn bytes_delivered(&self) -> u64 {
        self.inner.bytes_delivered.load(Ordering::Relaxed)
    }
}

fn copy_blocks_into_buf(
    offset: u64,
    avail: usize,
    first_block: u64,
    cached: &[Option<Arc<[u8]>>],
    buf: &mut [u8],
) -> Result<usize, IoError> {
    let mut dst_pos = 0usize;
    let mut cursor = offset;
    for (i, slot) in cached.iter().enumerate() {
        let blk = first_block + i as u64;
        let data = slot.as_ref().ok_or(IoError::UnexpectedEof {
            offset,
            requested: avail,
            read: dst_pos,
        })?;
        let blk_start = blk * RANGE_BLOCK_SIZE as u64;
        let in_blk_start = (cursor - blk_start) as usize;
        let take = (avail - dst_pos).min(data.len().saturating_sub(in_blk_start));
        if take == 0 {
            return Err(IoError::UnexpectedEof {
                offset,
                requested: avail,
                read: dst_pos,
            });
        }
        buf[dst_pos..dst_pos + take].copy_from_slice(&data[in_blk_start..in_blk_start + take]);
        dst_pos += take;
        cursor += take as u64;
    }
    Ok(dst_pos)
}

impl ByteSource for HttpRangeSource {
    fn len(&self) -> u64 {
        self.inner.len()
    }

    fn is_remote(&self) -> bool {
        true
    }

    fn read_at(&self, offset: u64, buf: &mut [u8], cancel: &CancelToken) -> Result<usize, IoError> {
        if cancel.is_cancelled() || self.inner.shutdown.is_cancelled() {
            return Err(IoError::Cancelled);
        }
        let total_len = self.inner.len();
        if buf.is_empty() || offset >= total_len {
            return Ok(0);
        }
        let avail = (total_len - offset).min(buf.len() as u64) as usize;
        let end = offset + avail as u64;

        // Fast path 1: completely inside pinned head window.
        if end <= self.inner.pinned_head_len() {
            let s = offset as usize;
            buf[..avail].copy_from_slice(&self.inner.pinned_head[s..s + avail]);
            return Ok(avail);
        }

        let first_block = offset / RANGE_BLOCK_SIZE as u64;
        let last_block = (end - 1) / RANGE_BLOCK_SIZE as u64;
        let block_count = (last_block - first_block + 1) as usize;

        // Fast path 2: check pinned head, active pins, and shared LRU pool.
        let mut cached: Vec<Option<Arc<[u8]>>> = Vec::with_capacity(block_count);
        let mut missing = Vec::new();
        for blk in first_block..=last_block {
            if let Some(data) = self.inner.lookup_cached_block(blk) {
                cached.push(Some(data));
            } else {
                cached.push(None);
                missing.push(blk);
            }
        }

        if !missing.is_empty() {
            // Never block a Tokio runtime worker thread.
            if tokio::runtime::Handle::try_current().is_ok() {
                return Err(IoError::InvalidInput(
                    "blocking HttpRangeSource::read_at called inside Tokio runtime".to_owned(),
                ));
            }

            let (linked_cancel, _cancel_guard) = link_cancel_token(&self.inner.shutdown, cancel);

            // Consumed-only sequential readahead: only extend past `last_block`
            // when sequential block progression is observed.
            let extra = lock_unpoisoned(&self.inner.readahead).extra_blocks(
                first_block,
                last_block,
                self.inner.meta.max_readahead_blocks,
            );
            if extra > 0 {
                let total_blocks = self.inner.total_blocks();
                let ra_end = last_block
                    .saturating_add(extra)
                    .min(total_blocks.saturating_sub(1));
                for blk in (last_block + 1)..=ra_end {
                    if self.inner.lookup_cached_block(blk).is_none() {
                        missing.push(blk);
                    }
                }
            }

            let fetched = self
                .inner
                .handle
                .block_on(
                    self.inner
                        .fetch_blocks_singleflight(&missing, 0, &linked_cancel),
                )?;
            for (i, slot) in cached.iter_mut().enumerate() {
                if slot.is_none() {
                    let blk = first_block + i as u64;
                    if let Some(data) = fetched.get(&blk) {
                        *slot = Some(Arc::clone(data));
                    } else if let Some(data) = self.inner.lookup_cached_block(blk) {
                        *slot = Some(data);
                    }
                }
            }
        } else {
            // Even on cache hit, advance or reset the sequential tracker so
            // readahead engages smoothly on sequential walks and resets on random jumps.
            let _ = lock_unpoisoned(&self.inner.readahead).extra_blocks(first_block, last_block, 0);
        }

        copy_blocks_into_buf(offset, avail, first_block, &cached, buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn readahead_tracker_ramps_sequentially_and_resets_on_random_cache_hits() {
        let mut tr = ReadaheadTracker::default();
        // Initial miss starts at 0.
        assert_eq!(tr.extra_blocks(10, 10, 4), 0);
        // Sequential misses ramp 1 -> 2 -> 4.
        assert_eq!(tr.extra_blocks(11, 11, 4), 1);
        assert_eq!(tr.extra_blocks(12, 12, 4), 2);
        // Intra-block repeat keeps streak alive.
        assert_eq!(tr.extra_blocks(12, 12, 0), 0);
        assert_eq!(tr.extra_blocks(13, 13, 4), 4);
        // Random jump on a cache hit (`max_ra == 0`) must reset streak.
        assert_eq!(tr.extra_blocks(50, 50, 0), 0);
        // Next sequential miss after the jump starts back at 1 (not 4).
        assert_eq!(tr.extra_blocks(51, 51, 4), 1);
    }
}
