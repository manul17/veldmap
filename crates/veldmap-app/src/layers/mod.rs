//! Raster overlay subsystem: ordered multi-stream raster overlay stack,
//! two-lane `StreamKey`-scoped scheduling with QOI + byte-shuffled LZ4 disk
//! caching, non-blocking local file open/reload I/O, background band statistics,
//! and typed zoom-invariant CPU cursor readouts.

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
mod style;

use std::path::{Path, PathBuf};

use veldmap_core::geo::FootprintBounds;
use veldmap_core::tile::TileStreamSpec;
use veldmap_raster::BandStats;

use crate::render::{CategoricalPaletteId, ColormapId};

pub(crate) use crs_override::CrsOverride;
pub(crate) use manager::{OverlayFrameCtx, OverlayManager, OverlayMutation};
pub(crate) use style::{ColorbarInfo, LayerBandOption, SavedLayerStyle, StretchPreset};

/// Unique identifier for a loaded overlay layer in [`stack::LayerStack`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct LayerId(u64);

/// Initial display, style, and georeference override options when inserting an
/// opened raster layer into [`stack::LayerStack`].
#[derive(Clone, Debug)]
pub(super) struct LayerInitOptions {
    pub(super) visible: bool,
    pub(super) opacity: f32,
    pub(super) crs_override: Option<CrsOverride>,
    pub(super) style: Option<SavedLayerStyle>,
}

/// Read-only scientific track summary projected into [`LayerSummary`].
#[derive(Clone, Debug)]
pub(crate) struct ScientificSummary {
    pub(crate) preset: StretchPreset,
    pub(crate) min: f64,
    pub(crate) max: f64,
    pub(crate) colormap: ColormapId,
    pub(crate) invert_colormap: bool,
    pub(crate) units: Option<String>,
    pub(crate) scale: f64,
    pub(crate) offset: f64,
    pub(crate) stats: Option<BandStats>,
    pub(crate) stats_loading: bool,
}

/// Read-only categorical track summary projected into [`LayerSummary`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct CategoricalSummary {
    pub(crate) palette: CategoricalPaletteId,
    pub(crate) has_source_color_map: bool,
    pub(crate) nodata_value: Option<u32>,
}

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
    pub(crate) active_spec: TileStreamSpec,
    pub(crate) band_options: Vec<LayerBandOption>,
    pub(crate) scientific: Option<ScientificSummary>,
    pub(crate) categorical: Option<CategoricalSummary>,
}

/// Borrowed persistence view of a loaded raster overlay layer for
/// `session.json` serialization without coupling `layers` to
/// `crate::storage::session`.
#[derive(Clone, Debug)]
pub(crate) struct OverlayPersistEntry<'a> {
    pub(crate) path: &'a Path,
    pub(crate) visible: bool,
    pub(crate) opacity: f32,
    pub(crate) crs_override: Option<&'a CrsOverride>,
    pub(crate) style: SavedLayerStyle,
}

/// Classified value returned by a zoom-invariant CPU cursor probe.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum CursorReadoutValue {
    Rgba([u8; 4]),
    Scalar { raw: f64, physical: f64 },
    Categorical(u16),
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
    pub(crate) units: Option<String>,
    pub(crate) has_affine_scaling: bool,
    pub(crate) log10_db: bool,
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
