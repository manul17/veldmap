//! HTTP Range [`ByteSource`] with adaptive 1-RTT head probing, fixed
//! host-RAM-scaled 64 KiB O(1) LRU pooling, consumed-only readahead, strong-ETag
//! `If-Match` validation, refcounted multi-block singleflight, and upfront-budgeted
//! pinned coalesced prefetch batches.

mod pool;
mod protocol;
mod singleflight;
mod transport;

use std::collections::{BTreeSet, HashMap};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use reqwest::header::{HeaderName, HeaderValue};
use thiserror::Error;
use veldmap_core::io::{ByteSource, IoError};
use veldmap_core::sync::{CancelToken, lock_unpoisoned};
use veldmap_core::tile::SourceFingerprint;

use crate::context::NetContext;
use crate::http::{HostLimiters, link_cancel_token, parse_custom_headers};
use crate::redact::{EncodedSecret, sanitize_url_for_display};
use crate::url::{canonical_url_without_query, extract_host};

pub use self::pool::RANGE_BLOCK_SIZE;
pub(crate) use self::pool::{BlockPoolKey, PrefetchBudget, PrefetchBudgetLease, RangeBlockPool};
use self::protocol::fingerprint_remote_source;
use self::singleflight::{BlockFlight, LeaderCleanupGuard, SharedRangeError, SingleflightTable};
use self::transport::{RangeFetchMode, RangeFetchRequest, fetch_range_http};

/// Default initial head probe window (`256 KiB`), covering typical 128–300 KiB
/// Sentinel-2 COG IFD and tile-offset headers in a single RTT.
pub const DEFAULT_HEAD_PROBE_BYTES: usize = 256 * 1024;

/// Maximum blocks coalesced into one HTTP Range GET (`64 * 64 KiB = 4 MiB`).
const MAX_COALESCED_SPAN_BLOCKS: u64 = 64;

/// Maximum gap (in blocks) bridged when coalescing remote COG chunk prefetch
/// ranges (`1` block = `64 KiB`).
const MAX_PREFETCH_GAP_BLOCKS: u64 = 1;

/// Default maximum consumed-only sequential readahead (`4 * 64 KiB = 256 KiB`).
const DEFAULT_MAX_READAHEAD_BLOCKS: u64 = 4;

/// Default transient retry limit for HTTP Range requests.
const DEFAULT_MAX_RETRIES: usize = 2;

/// Errors returned when opening an [`HttpRangeSource`].
#[derive(Debug, Error)]
pub enum RangeOpenError {
    #[error("operation cancelled")]
    Cancelled,
    #[error("invalid HTTP URL: {0}")]
    InvalidUrl(String),
    #[error("invalid HTTP header '{name}': {reason}")]
    InvalidHeader { name: String, reason: String },
    #[error("HTTP Range requests not supported by server: {0}")]
    RangeNotSupported(String),
    #[error("HTTP {code} opening remote raster: {detail}")]
    HttpStatus { code: u16, detail: String },
    #[error("invalid HTTP Range response: {0}")]
    InvalidResponse(String),
    #[error("network error: {0}")]
    Network(String),
    #[error("blocking HttpRangeSource::open called inside a Tokio runtime")]
    CalledInsideTokioRuntime,
}

impl RangeOpenError {
    pub const fn is_cancelled(&self) -> bool {
        matches!(self, Self::Cancelled)
    }
}

/// Configuration for opening an [`HttpRangeSource`].
#[derive(Clone)]
pub struct HttpRangeConfig {
    /// Bytes requested in the initial open probe (`0..head_probe_bytes`).
    pub head_probe_bytes: usize,
    /// Upper bound on consumed-only sequential readahead in 64 KiB blocks.
    pub max_readahead_blocks: u64,
    /// Maximum retries on transient network/5xx/429 errors (`412` and `200` are
    /// never retried).
    pub max_retries: usize,
    /// Optional custom HTTP headers (e.g. `Authorization: Bearer ...` or
    /// `x-api-key`) attached to every request for this source; redacted from
    /// `Debug` output and errors.
    pub extra_headers: Vec<(String, String)>,
}

impl Default for HttpRangeConfig {
    fn default() -> Self {
        Self {
            head_probe_bytes: DEFAULT_HEAD_PROBE_BYTES,
            max_readahead_blocks: DEFAULT_MAX_READAHEAD_BLOCKS,
            max_retries: DEFAULT_MAX_RETRIES,
            extra_headers: Vec::new(),
        }
    }
}

impl fmt::Debug for HttpRangeConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let header_names: Vec<&str> = self.extra_headers.iter().map(|(k, _)| k.as_str()).collect();
        f.debug_struct("HttpRangeConfig")
            .field("head_probe_bytes", &self.head_probe_bytes)
            .field("max_readahead_blocks", &self.max_readahead_blocks)
            .field("max_retries", &self.max_retries)
            .field("extra_header_names", &header_names)
            .finish()
    }
}

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

/// Internal shared state for [`HttpRangeSource`].
///
/// Intentionally holds cloned inner `Arc`s / handles rather than
/// `Arc<NetContext>` so spawned Tokio singleflight tasks can never drop the
/// last `NetContext` (and its `tokio::runtime::Runtime`) on a Tokio worker.
struct RangeInner {
    handle: tokio::runtime::Handle,
    shutdown: CancelToken,
    client: reqwest::Client,
    host_limiters: Arc<HostLimiters>,
    range_pool: Arc<RangeBlockPool>,
    prefetch_budget: Arc<PrefetchBudget>,
    url: String,
    canonical_url: Arc<str>,
    host: String,
    len: u64,
    strong_etag: Option<Arc<str>>,
    instance_id: u64,
    fingerprint: SourceFingerprint,
    pinned_head: Arc<[u8]>,
    extra_headers: Vec<(HeaderName, HeaderValue)>,
    secrets: Vec<EncodedSecret>,
    max_readahead_blocks: u64,
    max_retries: usize,
    singleflight: Arc<SingleflightTable>,
    active_pins: Mutex<ActivePinMap>,
    readahead: Mutex<ReadaheadTracker>,
    http_requests: AtomicU64,
    bytes_delivered: AtomicU64,
}

impl RangeInner {
    fn pool_key(&self, block_idx: u64) -> BlockPoolKey {
        match &self.strong_etag {
            Some(etag) => BlockPoolKey::Strong {
                canonical_url: Arc::clone(&self.canonical_url),
                strong_etag: Arc::clone(etag),
                block_idx,
            },
            None => BlockPoolKey::Isolated {
                instance_id: self.instance_id,
                block_idx,
            },
        }
    }

    fn total_blocks(&self) -> u64 {
        self.len.div_ceil(RANGE_BLOCK_SIZE as u64)
    }

    fn expected_block_len(&self, block_idx: u64) -> usize {
        let start = block_idx.saturating_mul(RANGE_BLOCK_SIZE as u64);
        if start >= self.len {
            return 0;
        }
        (self.len - start).min(RANGE_BLOCK_SIZE as u64) as usize
    }

    fn lookup_cached_block(&self, block_idx: u64) -> Option<Arc<[u8]>> {
        let start = block_idx.saturating_mul(RANGE_BLOCK_SIZE as u64);
        let expected_len = self.expected_block_len(block_idx);
        if expected_len == 0 {
            return None;
        }
        if start.saturating_add(expected_len as u64) <= self.pinned_head.len() as u64 {
            let s = start as usize;
            return Some(Arc::from(&self.pinned_head[s..s + expected_len]));
        }
        if let Some((_, data)) = lock_unpoisoned(&self.active_pins).get(&block_idx) {
            return Some(Arc::clone(data));
        }
        self.range_pool.get(&self.pool_key(block_idx))
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
            .min(self.len);
        if byte_start >= byte_end_excl {
            return Ok(HashMap::new());
        }
        let byte_end_incl = byte_end_excl - 1;

        let req_cfg = RangeFetchRequest {
            client: &self.client,
            host_limiters: &self.host_limiters,
            url: &self.url,
            host: &self.host,
            extra_headers: &self.extra_headers,
            secrets: &self.secrets,
            max_retries: self.max_retries,
            mode: RangeFetchMode::BlockSpan {
                byte_start,
                byte_end_incl,
                total_len: self.len,
                strong_etag: self.strong_etag.as_deref(),
            },
            request_counter: Some(&self.http_requests),
        };
        let resp = fetch_range_http(&req_cfg, cancel).await?;
        let body = resp.body;

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

    async fn fetch_blocks_singleflight(
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

        let mut flights_to_wait: Vec<(Arc<BlockFlight>, singleflight::WaiterGuard)> = plan.joined;

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
            .field("url", &sanitize_url_for_display(&self.inner.url))
            .field("len", &self.inner.len)
            .field("has_strong_etag", &self.inner.strong_etag.is_some())
            .field("fingerprint", &self.inner.fingerprint)
            .finish()
    }
}

impl HttpRangeSource {
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
        let canonical = canonical_url_without_query(trimmed_url)
            .ok_or_else(|| RangeOpenError::InvalidUrl(sanitize_url_for_display(trimmed_url)))?;
        let host = extract_host(trimmed_url);

        let parsed = parse_custom_headers(&config.extra_headers, &[])
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
        let delivered = resp.body.len() as u64;

        let inner = Arc::new(RangeInner {
            handle: net.handle().clone(),
            shutdown: net.shutdown_token().clone(),
            client,
            host_limiters,
            range_pool: net.range_pool(),
            prefetch_budget: net.prefetch_budget(),
            url: trimmed_url.to_owned(),
            canonical_url: Arc::from(canonical),
            host,
            len: resp.total_len,
            strong_etag: resp.strong_etag.map(Arc::from),
            instance_id,
            fingerprint,
            pinned_head: Arc::from(resp.body),
            extra_headers: parsed_headers,
            secrets,
            max_readahead_blocks: config.max_readahead_blocks,
            max_retries: config.max_retries,
            singleflight: Arc::new(SingleflightTable::new()),
            active_pins: Mutex::new(HashMap::new()),
            readahead: Mutex::new(ReadaheadTracker::default()),
            http_requests: AtomicU64::new(resp.requests_made),
            bytes_delivered: AtomicU64::new(delivered),
        });

        Ok(Self { inner })
    }

    /// Blocking wrapper around [`Self::open_async`], enforcing that it is never
    /// invoked from inside a Tokio worker thread.
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
    pub fn canonical_url(&self) -> &str {
        &self.inner.canonical_url
    }

    /// Validated strong RFC 9110 `ETag`, if supplied by the server.
    pub fn strong_etag(&self) -> Option<&str> {
        self.inner.strong_etag.as_deref()
    }

    /// Deterministic [`SourceFingerprint`] derived during open with zero extra
    /// HTTP round-trips.
    pub fn fingerprint(&self) -> SourceFingerprint {
        self.inner.fingerprint
    }

    /// Initial head bytes pinned in RAM during the open probe.
    pub fn pinned_head(&self) -> &[u8] {
        &self.inner.pinned_head
    }

    /// Total HTTP GET requests issued by this source (including open probe).
    pub fn http_requests(&self) -> u64 {
        self.inner.http_requests.load(Ordering::Relaxed)
    }

    /// Total HTTP response payload bytes delivered to this source.
    pub fn bytes_delivered(&self) -> u64 {
        self.inner.bytes_delivered.load(Ordering::Relaxed)
    }

    /// Asynchronously coalesce, fetch, and pin all 64 KiB blocks covering
    /// `ranges` (`(offset, len)` pairs) under an upfront memory permit from
    /// [`NetContext`].
    ///
    /// While the returned [`PrefetchPin`] is alive, subsequent synchronous
    /// [`ByteSource::read_at`] calls for any byte inside `ranges` are guaranteed
    /// to hit RAM without issuing HTTP requests.
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
        let max_blocks_in_cap = (cap_bytes.div_ceil(RANGE_BLOCK_SIZE) as u64).saturating_add(1);
        let mut unique_blocks = BTreeSet::new();
        let mut total_pin_bytes = 0usize;

        for &(offset, len) in ranges {
            if len == 0 || offset >= self.inner.len {
                continue;
            }
            let clamped_end = offset.saturating_add(len).min(self.inner.len);
            if clamped_end <= self.inner.pinned_head.len() as u64 {
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
                let blen = self.inner.expected_block_len(blk);
                if start.saturating_add(blen as u64) <= self.inner.pinned_head.len() as u64 {
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
        let needed_blocks: Vec<u64> = unique_blocks.into_iter().collect();

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

impl ByteSource for HttpRangeSource {
    fn len(&self) -> u64 {
        self.inner.len
    }

    fn is_remote(&self) -> bool {
        true
    }

    fn read_at(&self, offset: u64, buf: &mut [u8], cancel: &CancelToken) -> Result<usize, IoError> {
        if cancel.is_cancelled() || self.inner.shutdown.is_cancelled() {
            return Err(IoError::Cancelled);
        }
        if buf.is_empty() || offset >= self.inner.len {
            return Ok(0);
        }
        let avail = (self.inner.len - offset).min(buf.len() as u64) as usize;
        let end = offset + avail as u64;

        // Fast path 1: completely inside pinned head window.
        if end <= self.inner.pinned_head.len() as u64 {
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
                self.inner.max_readahead_blocks,
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
