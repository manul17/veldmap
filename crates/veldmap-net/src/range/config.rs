//! Configuration and open-error types for [`crate::HttpRangeSource`].

use std::fmt;

use thiserror::Error;

/// Default initial head probe window (`256 KiB`), covering typical 128–300 KiB
/// Sentinel-2 COG IFD and tile-offset headers in a single RTT.
const DEFAULT_HEAD_PROBE_BYTES: usize = 256 * 1024;

/// Default maximum consumed-only sequential readahead (`4 * 64 KiB = 256 KiB`).
const DEFAULT_MAX_READAHEAD_BLOCKS: u64 = 4;

/// Default transient retry limit for HTTP Range requests.
const DEFAULT_MAX_RETRIES: usize = 2;

/// Errors returned when opening an [`crate::HttpRangeSource`].
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
    #[cfg(any(test, feature = "test-support"))]
    #[error("blocking HttpRangeSource::open called inside a Tokio runtime")]
    CalledInsideTokioRuntime,
}

impl RangeOpenError {
    pub const fn is_cancelled(&self) -> bool {
        matches!(self, Self::Cancelled)
    }
}

/// Configuration for opening an [`crate::HttpRangeSource`].
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
