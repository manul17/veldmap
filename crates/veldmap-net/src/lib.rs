//! Asynchronous HTTP networking for VeldMap: shared [`NetContext`], live
//! basemap streaming ([`LiveTilePipeline`]), HTTP Range COG random-access
//! reading ([`HttpRangeSource`]), and zero-auth/custom-auth STAC search
//! ([`StacClient`]).
//!
//! Uses `reqwest` + `rustls` with the explicit `ring` cryptography provider
//! (zero OpenSSL or `aws-lc-sys` CMake/NASM build requirements), enforces
//! per-host concurrency and rate-limit cooldowns, supports immediate
//! `CancelToken` abort, stays decoupled from `veldmap-storage` via
//! [`TileBlobStore`], and stays decoupled from `veldmap-raster` codecs via
//! [`TileDecodeExecutor`].

#![forbid(unsafe_code)]

use thiserror::Error;

mod basemap;
mod context;
mod fetcher;
mod http;
mod limiter;
mod range;
mod redact;
mod stac;
mod url;

pub use basemap::{
    CUSTOM_MAX_LEVEL_MAX, CUSTOM_MAX_LEVEL_MIN, CustomBasemapConfig, LiveBasemapSpec,
    is_valid_gibs_date, validate_custom_url_template,
};
pub use context::{NetConfig, NetContext};
#[cfg(any(test, feature = "test-support"))]
pub use fetcher::test_support;
pub use fetcher::{
    BlobLease, BlobStoreError, DecodeCompletion, DecodeFailure, DecodeRequest, FetcherConfig,
    LiveTilePipeline, NetworkStatus, TileBlobStore, TileDecodeExecutor,
};
pub use http::is_valid_custom_header_name;
#[cfg(any(test, feature = "test-support"))]
pub use range::RANGE_BLOCK_SIZE;
pub use range::{HttpRangeConfig, HttpRangeSource, PrefetchPin, RangeOpenError};
pub use redact::sanitize_url_for_display;
pub use stac::{
    MAX_STAC_PAGINATED_ITEMS, StacAsset, StacAuth, StacCatalogConfig, StacClient, StacError,
    StacItem, StacItemFlightCache, StacNextLink, StacNextMethod, StacRasterBandMeta,
    StacSearchQuery, StacSearchResult, StacSortBy,
};
pub use url::{is_valid_http_endpoint, same_http_origin, sanitize_url_for_persistence};

/// Errors produced by live basemap configuration and HTTP streaming pipeline
/// setup.
#[derive(Debug, Error)]
pub enum NetError {
    #[error("invalid custom tile URL template: {0}")]
    InvalidUrlTemplate(String),
    #[error("invalid UTC date '{0}' (expected YYYY-MM-DD)")]
    InvalidDate(String),
    #[error("failed to initialize HTTP client: {0}")]
    HttpClient(String),
    #[error("failed to initialize tokio runtime: {0}")]
    Runtime(#[from] std::io::Error),
}
