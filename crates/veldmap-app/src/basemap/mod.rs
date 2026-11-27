//! Live + offline basemap orchestration ([`BasemapManager`]) and UI-facing
//! status DTOs.
//!
//! Coordinates the lazy bundled offline NASA Blue Marble provider
//! ([`bundled::BundledBasemapProvider`]) and the async HTTP streaming pipeline
//! (`LiveTilePipeline`) over an injected fast-lane `DecodePoolHandle`,
//! enforcing:
//! - `0` live HTTP requests for `PolarCap` keys (always routed to bundled
//!   `SourceId::BUNDLED_BASEMAP` offline polar caps)
//! - `0` live HTTP requests above per-source `max_level`
//! - Seamless Frame-1 & offline fallback to bundled Blue Marble tiles
//! - Clean fallback to full `z = 0..3` bundled Blue Marble if live pipeline
//!   initialization fails
//! - Shared `< 50%` VRAM appetite capping across bundled + live tiles

mod adapters;
mod bundled;
mod manager;
#[cfg(not(any(test, feature = "test-support")))]
mod policy;
#[cfg(any(test, feature = "test-support"))]
pub mod policy;

use veldmap_core::tile::SourceId;
use veldmap_net::NetworkStatus;

#[cfg(any(test, feature = "test-support"))]
pub use bundled::{BUNDLED_BASEMAP_CONFIG, BUNDLED_BASEMAP_JPEG, BUNDLED_MAX_LEVEL};
pub(crate) use manager::{BasemapManager, BasemapManagerDeps};

/// App-owned view of live HTTP streaming status so `ui::basemap` never depends
/// on `veldmap-net` types directly.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct LiveStreamStatus {
    pub(crate) in_flight: usize,
    pub(crate) queued: usize,
    pub(crate) failed_count: u64,
    pub(crate) offline: bool,
    pub(crate) last_error: Option<String>,
}

impl From<NetworkStatus> for LiveStreamStatus {
    fn from(status: NetworkStatus) -> Self {
        Self {
            in_flight: status.in_flight,
            queued: status.queued,
            failed_count: status.failed_count,
            offline: status.offline,
            last_error: status.last_error,
        }
    }
}

/// Plain read-only basemap status view model passed to UI toolbar controls so
/// `ui::basemap` never depends directly on [`BasemapManager`] or `veldmap-net`.
#[derive(Clone, Debug, Default)]
pub(crate) struct BasemapStatusSnapshot {
    pub(crate) stream: LiveStreamStatus,
    pub(crate) live_pipeline_error: Option<String>,
    pub(crate) config_error: Option<String>,
}

/// Explicit outcome returned by [`BasemapManager::sync_settings`] so the
/// caller (`SceneState`) owns GPU cache and upload queue mutations directly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BasemapSyncOutcome {
    pub(crate) changed: bool,
    pub(crate) evict_source: Option<SourceId>,
}
