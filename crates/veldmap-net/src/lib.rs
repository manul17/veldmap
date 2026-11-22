//! Asynchronous HTTP tile streaming, live basemap specifications, per-host
//! concurrency limiting, immediate `CancelToken` abort, and bounded CPU decode
//! pool for VeldMap.
//!
//! Uses `reqwest` + `rustls` with the explicit `ring` cryptography provider
//! (zero OpenSSL or `aws-lc-sys` CMake/NASM build requirements) and routes
//! decoded `Rgba8UnormSrgb` tiles with CPU linear-light mipmaps into
//! `PendingUploads`'s completion channel.

use thiserror::Error;
use veldmap_core::tile::TileScheme;

pub mod basemap;
pub mod decode_pool;
pub mod fetcher;

pub use basemap::{
    EOX_S2_MAX_LEVEL, EOX_S2_URL_TEMPLATE, GIBS_BLUE_MARBLE_MAX_LEVEL,
    GIBS_BLUE_MARBLE_URL_TEMPLATE, GIBS_VIIRS_MAX_LEVEL, GIBS_VIIRS_URL_TEMPLATE, LiveBasemapSpec,
    is_valid_iso_date,
};
pub use decode_pool::{DecodeCompletion, DecodeJob, DecodePool, DecodePoolHandle, Wakeup};
pub use fetcher::{
    DEFAULT_MAX_CONCURRENT_PER_HOST, FetcherConfig, LiveTilePipeline, NetworkStatus,
    default_user_agent,
};

/// Errors produced by live basemap URL resolution and HTTP streaming setup.
#[derive(Debug, Error)]
pub enum NetError {
    #[error(
        "live HTTP requests for PolarCap tiles are forbidden (route to offline polar fallback)"
    )]
    PolarCapForbidden,
    #[error("tile scheme mismatch: requested {requested:?}, expected {expected:?}")]
    SchemeMismatch {
        requested: TileScheme,
        expected: TileScheme,
    },
    #[error("tile level {level} exceeds source max_level {max_level}")]
    LevelAboveMax { level: u8, max_level: u8 },
    #[error("tile coordinates ({x}, {y}) out of range for {scheme:?} z={level}")]
    TileOutOfRange {
        scheme: TileScheme,
        level: u8,
        x: u32,
        y: u32,
    },
    #[error("invalid custom tile URL template: {0}")]
    InvalidUrlTemplate(String),
    #[error("failed to initialize HTTP client: {0}")]
    HttpClient(String),
    #[error("failed to initialize tokio runtime: {0}")]
    Runtime(#[from] std::io::Error),
}
