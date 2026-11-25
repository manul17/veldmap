//! Raster overlay subsystem: ordered raster overlay stack, two-lane
//! `PassKey`-scoped scheduling with QOI disk caching, non-blocking local file
//! open/reload I/O, and zoom-invariant CPU cursor readouts.
//!
//! Read-only GeoJSON vector layers live in [`crate::vectors`] and interactive
//! geodesic measurement lives in [`crate::measure`].

mod crs_override;
mod cursor;
mod frame;
mod io_worker;
mod jobs;
mod manager;
mod opener;
mod qoi_cache;
mod runtime;
mod stack;

use std::path::{Path, PathBuf};

use veldmap_core::geo::FootprintBounds;
use veldmap_core::tile::RASTER_TILE_SIZE;

use crate::gpu_tiles::display_tile_mip_bytes;

pub(crate) use crs_override::CrsOverride;
pub(crate) use manager::{OverlayFrameCtx, OverlayManager, OverlayMutation};

/// Default unpacked VRAM byte weight for an 8-bit `512 x 512` display tile
/// with full linear-light mip chain (`~1.33 MiB`), matching `BasemapManager`.
pub(super) fn default_display_tile_bytes() -> usize {
    display_tile_mip_bytes(RASTER_TILE_SIZE)
}

/// Unique identifier for a loaded overlay layer in [`stack::LayerStack`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct LayerId(u64);

/// Read-only summary of an overlay layer rendered in the Layer List UI.
#[derive(Clone, Debug)]
pub(crate) struct LayerSummary {
    pub(crate) id: LayerId,
    pub(crate) name: String,
    pub(crate) path: PathBuf,
    pub(crate) visible: bool,
    pub(crate) opacity: f32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) source_pixel_size_m: Option<(f64, f64)>,
    pub(crate) footprint: Option<FootprintBounds>,
    pub(crate) in_flight: usize,
    pub(crate) pass_active: bool,
    pub(crate) reloading: bool,
    pub(crate) crs_label: String,
    pub(crate) datum_warning: Option<String>,
    pub(crate) sample_info: String,
    pub(crate) detail_limit_level: u8,
    pub(crate) can_override_crs: bool,
    pub(crate) has_crs_override: bool,
    pub(crate) last_error: Option<String>,
    pub(crate) can_move_up: bool,
    pub(crate) can_move_down: bool,
}

/// Borrowed persistence view of a loaded raster overlay layer for
/// `session.json` serialization without coupling `layers` to
/// `crate::storage::session`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct OverlayPersistEntry<'a> {
    pub(crate) path: &'a Path,
    pub(crate) visible: bool,
    pub(crate) opacity: f32,
    pub(crate) crs_override: Option<&'a CrsOverride>,
}

/// Classified value returned by a zoom-invariant CPU cursor probe so decode
/// backpressure or giant-strip rejections are never misreported as `"nodata"`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CursorReadoutValue {
    Rgba([u8; 4]),
    Nodata,
    Pending,
    Unavailable,
}

/// Zoom-invariant CPU cursor readout for the topmost visible overlay covering
/// the cursor position.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CursorReadout {
    pub(crate) layer_name: String,
    pub(crate) pixel: (u32, u32),
    pub(crate) source_pixel_size_m: Option<(f64, f64)>,
    pub(crate) value: CursorReadoutValue,
}

/// Asynchronous result of a background local raster open or reload operation,
/// drained by `SceneState` at the start of each frame.
#[derive(Clone, Debug)]
pub(crate) enum OverlayIoEvent {
    Opened {
        id: LayerId,
        path: PathBuf,
        bounds: Option<FootprintBounds>,
        fly_to: bool,
    },
    Reloaded {
        id: LayerId,
    },
    Failed {
        path: PathBuf,
        error: String,
        can_assign_crs: bool,
    },
}
