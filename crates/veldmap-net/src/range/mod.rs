//! HTTP Range [`veldmap_core::io::ByteSource`] with adaptive 1-RTT head probing,
//! fixed host-RAM-scaled 64 KiB O(1) LRU pooling, consumed-only readahead,
//! strong-ETag `If-Match` validation, refcounted multi-block singleflight, and
//! upfront-budgeted pinned coalesced prefetch batches.

mod config;
mod pool;
mod prefetch;
mod protocol;
mod singleflight;
mod source;
mod transport;

pub use self::config::{HttpRangeConfig, RangeOpenError};
#[cfg(any(test, feature = "test-support"))]
pub use self::pool::RANGE_BLOCK_SIZE;
pub(crate) use self::pool::{PrefetchBudget, RangeBlockPool};
pub use self::prefetch::PrefetchPin;
pub use self::source::HttpRangeSource;
