//! Ordered raster overlay stack, two-lane `PassKey`-scoped scheduling with QOI
//! disk caching, non-blocking local file open/reload I/O, and zoom-invariant
//! CPU cursor readouts.

mod cursor;
mod jobs;
mod manager;
mod runtime;
mod stack;

use std::path::PathBuf;

use veldmap_core::geo::footprint::FootprintBounds;
use veldmap_core::tile::{PixelFormat, RASTER_TILE_SIZE, max_mip_levels, mip_chain_byte_size};

pub(crate) use manager::{OverlayManager, OverlayMutation};

/// Default unpacked VRAM byte weight for an 8-bit `512 x 512` display tile
/// with full linear-light mip chain (`~1.33 MiB`), matching `BasemapManager`.
pub(crate) fn default_display_tile_bytes() -> usize {
    let mips = max_mip_levels(RASTER_TILE_SIZE, RASTER_TILE_SIZE);
    mip_chain_byte_size(
        RASTER_TILE_SIZE,
        RASTER_TILE_SIZE,
        PixelFormat::Rgba8UnormSrgb,
        mips,
    )
}

/// Unique identifier for a loaded overlay layer in [`stack::LayerStack`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct LayerId(pub(crate) u64);

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

/// Asynchronous result of a background local-file open or reload operation,
/// drained by `VeldmapApp` at the start of each frame.
#[derive(Clone, Debug)]
pub(crate) enum OverlayIoEvent {
    Opened {
        id: LayerId,
        bounds: Option<FootprintBounds>,
        fly_to: bool,
    },
    Reloaded {
        id: LayerId,
    },
    Failed {
        path: PathBuf,
        error: String,
    },
}
