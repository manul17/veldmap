//! Shared control-plane and immutable runtime context for
//! [`crate::LiveTilePipeline`].

use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::sync::Notify;
use veldmap_core::sync::{CancelToken, Wakeup};

use crate::TileDecodeExecutor;
use crate::http::HostLimiters;

use super::disk::DiskCacheGate;
use super::state::PipelineState;

/// Shared control-plane handles held by both [`crate::LiveTilePipeline`] and
/// [`PipelineContext`].
///
/// Kept distinct from [`PipelineContext`] so the frontend handle never extends
/// the lifetimes of `reqwest::Client`, `DiskCacheGate`, `TileDecodeExecutor`,
/// or `Wakeup` past `dispatcher_loop` exit.
///
/// Lock ordering: always acquire `state` BEFORE `host_limiters` (never
/// reverse; `HostLimiters` is a leaf lock).
pub(super) struct PipelineShared {
    pub(super) shutdown: CancelToken,
    pub(super) state: Mutex<PipelineState>,
    pub(super) host_limiters: Arc<HostLimiters>,
    pub(super) notify: Notify,
}

/// Shared immutable context passed across the dispatcher and per-tile tasks.
pub(super) struct PipelineContext {
    pub(super) shared: Arc<PipelineShared>,
    pub(super) client: reqwest::Client,
    pub(super) disk_cache: Option<DiskCacheGate>,
    pub(super) decoder: Arc<dyn TileDecodeExecutor>,
    pub(super) wakeup: Wakeup,
    pub(super) max_tile_bytes: usize,
    pub(super) transient_retry_backoff: Duration,
}
