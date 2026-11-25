//! Asynchronous HTTP tile streaming, live basemap specifications, per-host
//! concurrency limiting, and immediate `CancelToken` abort for VeldMap.
//!
//! Uses `reqwest` + `rustls` with the explicit `ring` cryptography provider
//! (zero OpenSSL or `aws-lc-sys` CMake/NASM build requirements), stays
//! decoupled from `veldmap-storage` via [`TileBlobStore`], and stays decoupled
//! from `veldmap-raster` codecs via [`TileDecodeExecutor`].

#![forbid(unsafe_code)]

use std::sync::Arc;

use thiserror::Error;
use veldmap_core::sync::CancelToken;
use veldmap_core::tile::{QuadTileId, TileKey};

mod basemap;
mod fetcher;
mod http;
mod redact;

pub use basemap::{
    CUSTOM_MAX_LEVEL_MAX, CUSTOM_MAX_LEVEL_MIN, CustomBasemapConfig, LiveBasemapSpec,
    is_valid_gibs_date, validate_custom_url_template,
};
pub use fetcher::{FetcherConfig, LiveTilePipeline, NetworkStatus};
pub use redact::{redact_secret_label, sanitize_url_for_display};

/// Structured error returned across the [`TileBlobStore`] seam.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum BlobStoreError {
    #[error("blob store error: {0}")]
    Backend(String),
}

/// Minimal encoded-tile persistence boundary injected into
/// [`LiveTilePipeline`], keeping `veldmap-net` decoupled from concrete disk-cache
/// backends (implementations may enforce source-key formatting rules and
/// maximum blob-size limits).
pub trait TileBlobStore: Send + Sync + 'static {
    /// Fetch cached encoded tile bytes for `(source_key, id)`, if present.
    fn get(&self, source_key: &str, id: QuadTileId) -> Result<Option<Vec<u8>>, BlobStoreError>;

    /// Persist decode-validated encoded tile bytes for `(source_key, id)`.
    fn put(&self, source_key: &str, id: QuadTileId, payload: &[u8]) -> Result<(), BlobStoreError>;

    /// Delete one cached tile entry (e.g. if a cached payload fails decode).
    fn delete(&self, source_key: &str, id: QuadTileId) -> Result<(), BlobStoreError>;
}

/// Typed decode failure reported through [`DecodeCompletion`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DecodeFailure {
    /// Encoded payload was malformed or exceeded decoder bounds (terminal for
    /// HTTP fetches; deletes corrupt disk entries).
    InvalidPayload(String),
    /// Decode execution was cancelled, evicted from the decode queue under
    /// pressure, or panicked before finishing (never marked terminal).
    Aborted,
}

/// Receiver half of [`DecodeCompletion::channel`].
pub struct DecodeCompletionRx(tokio::sync::oneshot::Receiver<Result<(), DecodeFailure>>);

impl DecodeCompletionRx {
    /// Non-blocking poll for completion, returning `Some(Err(DecodeFailure::Aborted))`
    /// if the sender was dropped without calling [`DecodeCompletion::complete`].
    pub fn try_recv(&mut self) -> Option<Result<(), DecodeFailure>> {
        match self.0.try_recv() {
            Ok(res) => Some(res),
            Err(tokio::sync::oneshot::error::TryRecvError::Closed) => {
                Some(Err(DecodeFailure::Aborted))
            }
            Err(tokio::sync::oneshot::error::TryRecvError::Empty) => None,
        }
    }

    pub(crate) async fn recv(self) -> Result<(), DecodeFailure> {
        self.0.await.unwrap_or(Err(DecodeFailure::Aborted))
    }
}

/// Oneshot completion handle passed inside [`DecodeRequest`].
///
/// Dropping this handle without calling [`Self::complete`] automatically
/// signals [`DecodeFailure::Aborted`] to the awaiting [`LiveTilePipeline`]
/// worker, ensuring queue evictions or worker panics never orphan `in_flight`
/// tiles.
pub struct DecodeCompletion(tokio::sync::oneshot::Sender<Result<(), DecodeFailure>>);

impl DecodeCompletion {
    /// Create a linked `(DecodeCompletion, DecodeCompletionRx)` pair.
    pub fn channel() -> (Self, DecodeCompletionRx) {
        let (tx, rx) = tokio::sync::oneshot::channel();
        (Self(tx), DecodeCompletionRx(rx))
    }

    /// Report the decode result back to the awaiting [`LiveTilePipeline`] task.
    pub fn complete(self, outcome: Result<(), DecodeFailure>) {
        let _ = self.0.send(outcome);
    }
}

/// Single encoded-tile decode request submitted from [`LiveTilePipeline`] to
/// an injected [`TileDecodeExecutor`].
pub struct DecodeRequest {
    pub key: TileKey,
    pub bytes: Arc<[u8]>,
    pub cancel: CancelToken,
    pub on_decoded: DecodeCompletion,
}

/// Narrow CPU decode submission boundary injected into [`LiveTilePipeline`],
/// keeping `veldmap-net` permanently free of `veldmap-raster` and image codecs.
pub trait TileDecodeExecutor: Send + Sync + 'static {
    /// Enqueue `request` for asynchronous CPU decoding. Returns `false` if the
    /// request was rejected (e.g. queue closed, cancelled, or full under
    /// backpressure); rejected requests are dropped by the executor.
    fn submit(&self, request: DecodeRequest) -> bool;
}

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
