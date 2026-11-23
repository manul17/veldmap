//! Asynchronous HTTP tile streaming, live basemap specifications, per-host
//! concurrency limiting, and immediate `CancelToken` abort for VeldMap.
//!
//! Uses `reqwest` + `rustls` with the explicit `ring` cryptography provider
//! (zero OpenSSL or `aws-lc-sys` CMake/NASM build requirements), stays
//! decoupled from `veldmap-storage` via [`TileBlobStore`], and stays decoupled
//! from `veldmap-raster` codecs via [`TileDecodeExecutor`].

use std::sync::Arc;

use thiserror::Error;
use veldmap_core::io::CancelToken;
use veldmap_core::tile::{PlanetaryScheme, QuadTileId, TileKey};

mod basemap;
mod fetcher;
mod http;
mod redact;

pub use basemap::{
    CUSTOM_MAX_LEVEL_MAX, CUSTOM_MAX_LEVEL_MIN, CustomBasemapConfig, LiveBasemapSpec,
    is_valid_gibs_date,
};
pub use fetcher::{FetcherConfig, LiveTilePipeline, NetworkStatus};
pub use redact::sanitize_url_for_display;

/// Minimal encoded-tile persistence boundary injected into
/// [`LiveTilePipeline`], keeping `veldmap-net` decoupled from SQLite and
/// application settings types in `veldmap-storage`.
pub trait TileBlobStore: Send + Sync + 'static {
    /// Fetch cached encoded tile bytes for `(source_key, id)`, if present.
    fn get(&self, source_key: &str, id: QuadTileId) -> Result<Option<Vec<u8>>, String>;

    /// Persist decode-validated encoded tile bytes for `(source_key, id)`.
    fn put(&self, source_key: &str, id: QuadTileId, payload: &[u8]) -> Result<(), String>;

    /// Delete one cached tile entry (e.g. if a cached payload fails decode).
    fn delete(&self, source_key: &str, id: QuadTileId) -> Result<(), String>;
}

/// Completion callback executed by a [`TileDecodeExecutor`] after decode
/// finishes (`Ok(())` after forwarding the decoded tile, or `Err(msg)` if
/// decode failed or panicked).
pub type DecodeCompletion = Box<dyn FnOnce(Result<(), String>) + Send + 'static>;

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

/// Errors produced by live basemap URL resolution and HTTP streaming setup.
#[derive(Debug, Error)]
pub enum NetError {
    #[error(
        "live HTTP requests for PolarCap tiles are forbidden (route to offline polar fallback)"
    )]
    PolarCapForbidden,
    #[error("tile scheme mismatch: requested {requested:?}, expected {expected:?}")]
    SchemeMismatch {
        requested: PlanetaryScheme,
        expected: PlanetaryScheme,
    },
    #[error("tile level {level} exceeds source max_level {max_level}")]
    LevelAboveMax { level: u8, max_level: u8 },
    #[error("tile coordinates ({x}, {y}) out of range for {scheme:?} z={level}")]
    TileOutOfRange {
        scheme: PlanetaryScheme,
        level: u8,
        x: u32,
        y: u32,
    },
    #[error("invalid custom tile URL template: {0}")]
    InvalidUrlTemplate(String),
    #[error("invalid UTC date '{0}' (expected YYYY-MM-DD)")]
    InvalidDate(String),
    #[error("failed to initialize HTTP client: {0}")]
    HttpClient(String),
    #[error("failed to initialize tokio runtime: {0}")]
    Runtime(#[from] std::io::Error),
}
