//! Raster overlay subsystem: ordered multi-asset local & remote COG overlay
//! stack (`SingleRaster` + `BandMath`), two-lane `StreamKey`-scoped scheduling
//! with offloaded QOI + byte-shuffled LZ4 disk caching, non-blocking
//! local/remote single & multi-asset open/reload I/O, background band
//! statistics, and typed zoom-invariant CPU cursor readouts.

mod asset;
mod band_math;
mod crs_override;
mod cursor;
mod disk_cache;
mod frame;
mod io_worker;
mod jobs;
mod layer;
mod manager;
mod opener;
mod remote_opener;
mod runtime;
mod stack;
mod style;
mod summary;

use std::path::{Path, PathBuf};

use veldmap_core::geo::FootprintBounds;
use veldmap_raster::ExternalBandMeta;

pub(crate) use crate::band_math::BandMathFormula;
pub(crate) use asset::StacCogAuthMode;
pub(crate) use band_math::{
    BandChannelSample, BandMathPreset, BandMathReadout, BandMathStyleSpec, BandMathSummary,
};
pub(crate) use crs_override::CrsOverride;
pub(crate) use cursor::{CursorReadout, CursorReadoutValue};
pub(crate) use manager::{
    OverlayEdit, OverlayFrameCtx, OverlayManager, OverlayManagerDeps, OverlayMutation,
};
pub(crate) use style::{ColorbarInfo, LayerStyleSpec, StretchPreset, TrackKind};
pub(crate) use summary::{CategoricalSummary, LayerSummary, ScientificSummary};

/// Unique identifier for a loaded overlay layer in [`stack::LayerStack`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct LayerId(u64);

impl LayerId {
    /// Return the raw numeric identifier for renderer scissor/swipe matching.
    pub(crate) const fn raw(self) -> u64 {
        self.0
    }

    #[cfg(test)]
    pub(crate) const fn from_raw_for_test(raw: u64) -> Self {
        Self(raw)
    }
}

/// Initial display, style, and georeference override options when inserting an
/// opened raster layer into [`stack::LayerStack`].
#[derive(Clone, Debug)]
struct LayerInitOptions {
    visible: bool,
    opacity: f32,
    crs_override: Option<CrsOverride>,
    style: Option<LayerStyleSpec>,
}

/// Parameters for queuing a non-blocking local raster overlay open.
#[derive(Clone, Debug)]
pub(crate) struct OpenOverlayRequest<'a> {
    pub(crate) path: &'a Path,
    pub(crate) fly_to: bool,
    pub(crate) visible: bool,
    pub(crate) opacity: f32,
    pub(crate) crs_override: Option<CrsOverride>,
    pub(crate) style: Option<LayerStyleSpec>,
    pub(crate) restore_id: Option<u64>,
}

/// Parameters for queuing a non-blocking remote Cloud-Optimized GeoTIFF open.
#[derive(Clone, Debug)]
pub(crate) struct RemoteCogRequest {
    pub(crate) url: String,
    pub(crate) title: String,
    pub(crate) auth_mode: StacCogAuthMode,
    pub(crate) extra_headers: Vec<(String, String)>,
    pub(crate) external_bands: Vec<ExternalBandMeta>,
    /// When `true`, explicitly opens the asset on the `R8Uint` track with
    /// [`crate::colormap::CategoricalPaletteId::Sentinel2Scl`].
    pub(crate) is_scl: bool,
    pub(crate) stac_item_href: Option<String>,
    pub(crate) asset_key: Option<String>,
    pub(crate) visible: bool,
    pub(crate) opacity: f32,
    pub(crate) style: Option<LayerStyleSpec>,
    pub(crate) fly_to: bool,
    pub(crate) restore_id: Option<u64>,
}

/// One constituent local or remote asset inside a [`RemoteBandMathRequest`],
/// supporting both live STAC recipe opens and heterogeneous local/remote
/// workspace restoration.
#[derive(Clone, Debug)]
pub(crate) enum BandMathAssetOpenSpec {
    Local {
        label: String,
        path: PathBuf,
        crs_override: Option<CrsOverride>,
        band_index: u16,
    },
    Remote {
        label: String,
        url: String,
        auth_mode: StacCogAuthMode,
        extra_headers: Vec<(String, String)>,
        external_bands: Vec<ExternalBandMeta>,
        stac_item_href: Option<String>,
        asset_key: Option<String>,
        band_index: u16,
    },
}

impl BandMathAssetOpenSpec {
    pub(crate) fn is_remote(&self) -> bool {
        matches!(self, Self::Remote { .. })
    }
}

/// Parameters for opening a multi-asset Band Math / RGB Composite recipe
/// (remote STAC or heterogeneous local/remote restored workspace) as a single
/// `LayerKind::BandMath` layer.
#[derive(Clone, Debug)]
pub(crate) struct RemoteBandMathRequest {
    pub(crate) title: String,
    pub(crate) preset: BandMathPreset,
    pub(crate) bands: Vec<BandMathAssetOpenSpec>,
    pub(crate) scl: Option<BandMathAssetOpenSpec>,
    pub(crate) visible: bool,
    pub(crate) opacity: f32,
    pub(crate) style: Option<BandMathStyleSpec>,
    pub(crate) fly_to: bool,
    pub(crate) restore_id: Option<u64>,
}

/// One constituent band reference selected from an already-loaded raster layer
/// inside the Layers-panel Band Math / RGB Composite builder.
#[derive(Clone, Debug)]
pub(crate) struct LocalBandMathInput {
    pub(crate) layer_id: LayerId,
    pub(crate) band_index: u16,
    pub(crate) label: String,
}

/// Parameters for constructing a `LayerKind::BandMath` layer from already-loaded
/// local or remote raster layers (e.g., a 4-band RGB+NIR orthophoto or distinct
/// single-band layers).
#[derive(Clone, Debug)]
pub(crate) struct LocalBandMathSpec {
    pub(crate) title: String,
    pub(crate) formula: BandMathFormula,
    pub(crate) inputs: Vec<LocalBandMathInput>,
    pub(crate) scl: Option<LocalBandMathInput>,
}

/// Constituent local or remote asset snapshot inside
/// [`OverlayPersistEntry::BandMath`].
#[derive(Clone, Debug)]
pub(crate) enum BandAssetPersistEntry {
    Local {
        label: String,
        path: PathBuf,
        crs_override: Option<CrsOverride>,
        band_index: u16,
    },
    Remote {
        label: String,
        canonical_url: String,
        auth_mode: StacCogAuthMode,
        external_bands: Vec<ExternalBandMeta>,
        stac_item_href: Option<String>,
        asset_key: Option<String>,
        band_index: u16,
    },
}

/// Owned persistence snapshot of a loaded raster overlay layer (local raster,
/// remote COG, or heterogeneous local/remote Band Math recipe) for
/// `session.json` and `.veldmap.json` serialization without coupling `layers`
/// to `crate::storage::session`.
#[derive(Clone, Debug)]
pub(crate) enum OverlayPersistEntry {
    LocalRaster {
        path: PathBuf,
        visible: bool,
        opacity: f32,
        crs_override: Option<CrsOverride>,
        style: Option<LayerStyleSpec>,
    },
    RemoteCog {
        canonical_url: String,
        title: String,
        auth_mode: StacCogAuthMode,
        external_bands: Vec<ExternalBandMeta>,
        is_scl: bool,
        stac_item_href: Option<String>,
        asset_key: Option<String>,
        visible: bool,
        opacity: f32,
        style: Option<LayerStyleSpec>,
    },
    BandMath {
        title: String,
        visible: bool,
        opacity: f32,
        bands: Vec<BandAssetPersistEntry>,
        scl: Option<BandAssetPersistEntry>,
        style: BandMathStyleSpec,
    },
}

/// Asynchronous result of a background local or remote raster open or reload
/// operation, drained by `SceneState` at the start of each frame.
#[derive(Clone, Debug)]
pub(crate) enum OverlayIoEvent {
    Opened {
        id: LayerId,
        path: PathBuf,
        bounds: Option<FootprintBounds>,
        fly_to: bool,
        restore_id: Option<u64>,
    },
    Reloaded {
        id: LayerId,
    },
    ReloadFailed {
        path: PathBuf,
        error: String,
        can_assign_crs: bool,
    },
    Failed {
        path: PathBuf,
        error: String,
        can_assign_crs: bool,
        restore_id: Option<u64>,
    },
    RemoteOpened {
        id: LayerId,
        title: String,
        display_url: String,
        is_band_math: bool,
        bounds: Option<FootprintBounds>,
        fly_to: bool,
        restore_id: Option<u64>,
    },
    RemoteFailed {
        title: String,
        display_url: String,
        is_band_math: bool,
        error: String,
        restore_id: Option<u64>,
    },
}
