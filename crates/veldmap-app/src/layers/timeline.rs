//! Multi-temporal `Timeline` overlay layer (`LayerKind::Timeline`) supporting
//! dated local folders and STAC AOI time series with bounded LRU open epochs,
//! cloud-cover filtering, adjacent-date prefetch (`active_idx ± 1`), instant
//! stale-request cancellation on scrub, and dual-epoch GPU scalar diffing /
//! 0%-bleed-through RGB cross-fading.

use std::path::PathBuf;

use veldmap_core::geo::FootprintBounds;
use veldmap_core::sync::CancelToken;
use veldmap_core::tile::{PixelFormat, SourceId, StreamKey};
use veldmap_net::sanitize_url_for_display;
use veldmap_raster::ExternalBandMeta;

use crate::colormap::ColormapId;
use crate::render::ScratchDemand;

use super::asset::{LayerStream, RasterAsset, StacCogAuthMode};
use super::band_math::{
    BandMathInputSummary, BandMathLayerData, BandMathPreset, BandMathStyle, BandMathStyleSpec,
    BandMathSummary,
};
use super::crs_override::CrsOverride;
use super::style::{LayerStyle, LayerStyleSpec};
use super::{BandAssetPersistEntry, BandMathAssetOpenSpec, StacItemRefreshPolicy};

/// Maximum number of epochs kept open concurrently inside one `Timeline` layer.
pub(super) const MAX_OPEN_TIMELINE_EPOCHS: usize = 6;

/// Tile budget cap per adjacent prefetched epoch stream so background warming
/// of `active_idx ± 1` stays coarse/lightweight and never starves the active
/// epoch's LOD.
pub(super) const TIMELINE_PREFETCH_CAP_TILES: usize = 4;

/// Comparison rendering mode for a multi-temporal `Timeline` layer.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) enum TimelineCompareMode {
    /// Render only `active_idx`.
    #[default]
    Single,
    /// Dual-epoch GPU scalar difference (`B - A`, where `A = ref_idx`, `B = active_idx`).
    ScalarDiff { ref_idx: usize },
    /// Dual-epoch GPU 0%-bleed-through RGB cross-fade (`blend_t = 0.0` shows `ref_idx`,
    /// `1.0` shows `active_idx`).
    CrossFade { ref_idx: usize, blend_t: f32 },
}

impl TimelineCompareMode {
    pub(crate) const fn ref_idx(self) -> Option<usize> {
        match self {
            Self::Single => None,
            Self::ScalarDiff { ref_idx } | Self::CrossFade { ref_idx, .. } => Some(ref_idx),
        }
    }
}

/// Source specification for opening one epoch inside a [`TimelineOpenRequest`].
#[derive(Clone, Debug)]
pub(crate) enum TimelineEpochSourceSpec {
    LocalSingle {
        path: PathBuf,
        crs_override: Option<CrsOverride>,
    },
    RemoteSingle {
        url: String,
        auth_mode: StacCogAuthMode,
        extra_headers: Vec<(String, String)>,
        external_bands: Vec<ExternalBandMeta>,
        is_scl: bool,
        stac_item_href: Option<String>,
        asset_key: Option<String>,
        refresh_policy: Option<StacItemRefreshPolicy>,
    },
    BandMath {
        bands: Vec<BandMathAssetOpenSpec>,
        scl: Option<BandMathAssetOpenSpec>,
    },
}

impl TimelineEpochSourceSpec {
    pub(crate) fn is_remote(&self) -> bool {
        match self {
            Self::LocalSingle { .. } => false,
            Self::RemoteSingle { .. } => true,
            Self::BandMath { bands, scl } => {
                bands.iter().any(BandMathAssetOpenSpec::is_remote)
                    || scl.as_ref().is_some_and(BandMathAssetOpenSpec::is_remote)
            }
        }
    }

    pub(crate) fn to_persist(&self) -> TimelineEpochPersistSource {
        match self {
            Self::LocalSingle { path, crs_override } => TimelineEpochPersistSource::LocalSingle {
                path: path.clone(),
                crs_override: crs_override.clone(),
            },
            Self::RemoteSingle {
                url,
                auth_mode,
                external_bands,
                is_scl,
                stac_item_href,
                asset_key,
                ..
            } => TimelineEpochPersistSource::RemoteSingle {
                canonical_url: veldmap_net::sanitize_url_for_persistence(url)
                    .unwrap_or_else(|| sanitize_url_for_display(url)),
                auth_mode: *auth_mode,
                external_bands: external_bands.clone(),
                is_scl: *is_scl,
                stac_item_href: stac_item_href.clone(),
                asset_key: asset_key.clone(),
            },
            Self::BandMath { bands, scl } => TimelineEpochPersistSource::BandMath {
                bands: bands.iter().map(persist_from_band_open_spec).collect(),
                scl: scl.as_ref().map(persist_from_band_open_spec),
            },
        }
    }
}

fn persist_from_band_open_spec(spec: &BandMathAssetOpenSpec) -> BandAssetPersistEntry {
    match spec {
        BandMathAssetOpenSpec::Local {
            label,
            path,
            crs_override,
            band_index,
        } => BandAssetPersistEntry::Local {
            label: label.clone(),
            path: path.clone(),
            crs_override: crs_override.clone(),
            band_index: *band_index,
        },
        BandMathAssetOpenSpec::Remote {
            label,
            url,
            auth_mode,
            external_bands,
            stac_item_href,
            asset_key,
            band_index,
            ..
        } => BandAssetPersistEntry::Remote {
            label: label.clone(),
            canonical_url: veldmap_net::sanitize_url_for_persistence(url)
                .unwrap_or_else(|| sanitize_url_for_display(url)),
            auth_mode: *auth_mode,
            external_bands: external_bands.clone(),
            stac_item_href: stac_item_href.clone(),
            asset_key: asset_key.clone(),
            band_index: *band_index,
        },
    }
}

/// One chronologically ordered epoch specification inside [`TimelineOpenRequest`].
#[derive(Clone, Debug)]
pub(crate) struct TimelineEpochSpec {
    pub(crate) label: String,
    pub(crate) date_label: String,
    pub(crate) sort_key: String,
    pub(crate) cloud_cover: Option<f32>,
    pub(crate) source: TimelineEpochSourceSpec,
}

/// Parameters for creating a `LayerKind::Timeline` overlay layer (from a local
/// dated folder scan, a STAC AOI tile group, or session/project restore).
#[derive(Clone, Debug)]
pub(crate) struct TimelineOpenRequest {
    pub(crate) title: String,
    pub(crate) epochs: Vec<TimelineEpochSpec>,
    pub(crate) recipe_preset: Option<BandMathPreset>,
    pub(crate) band_math_style: Option<BandMathStyleSpec>,
    pub(crate) single_style: Option<LayerStyleSpec>,
    pub(crate) visible: bool,
    pub(crate) opacity: f32,
    pub(crate) active_idx: Option<usize>,
    pub(crate) compare: Option<TimelineCompareMode>,
    pub(crate) max_cloud_cover: Option<f32>,
    pub(crate) diff_colormap: Option<ColormapId>,
    pub(crate) diff_invert_colormap: bool,
    pub(crate) diff_stretch_min: Option<f64>,
    pub(crate) diff_stretch_max: Option<f64>,
    pub(crate) fly_to: bool,
    pub(crate) restore_id: Option<u64>,
}

/// Owned persistence snapshot of one epoch source inside
/// [`super::OverlayPersistEntry::Timeline`].
#[derive(Clone, Debug)]
pub(crate) enum TimelineEpochPersistSource {
    LocalSingle {
        path: PathBuf,
        crs_override: Option<CrsOverride>,
    },
    RemoteSingle {
        canonical_url: String,
        auth_mode: StacCogAuthMode,
        external_bands: Vec<ExternalBandMeta>,
        is_scl: bool,
        stac_item_href: Option<String>,
        asset_key: Option<String>,
    },
    BandMath {
        bands: Vec<BandAssetPersistEntry>,
        scl: Option<BandAssetPersistEntry>,
    },
}

/// Owned persistence snapshot of one epoch inside
/// [`super::OverlayPersistEntry::Timeline`].
#[derive(Clone, Debug)]
pub(crate) struct TimelineEpochPersistEntry {
    pub(crate) label: String,
    pub(crate) date_label: String,
    pub(crate) sort_key: String,
    pub(crate) cloud_cover: Option<f32>,
    pub(crate) source: TimelineEpochPersistSource,
}

/// UI-facing load status of one epoch in a `Timeline` layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TimelineEpochLoadState {
    Unloaded,
    Loading,
    Ready,
    Failed,
}

/// UI-facing summary of one epoch inside [`TimelineSummary`].
#[derive(Clone, Debug)]
pub(crate) struct TimelineEpochSummary {
    pub(crate) label: String,
    pub(crate) date_label: String,
    pub(crate) cloud_cover: Option<f32>,
    pub(crate) load_state: TimelineEpochLoadState,
    pub(crate) error: Option<String>,
}

/// Read-only `Timeline`-specific summary embedded in [`super::LayerSummaryKind`].
#[derive(Clone, Debug)]
pub(crate) struct TimelineSummary {
    pub(crate) width: Option<u32>,
    pub(crate) height: Option<u32>,
    pub(crate) epochs: Vec<TimelineEpochSummary>,
    pub(crate) filtered_indices: Vec<usize>,
    pub(crate) active_idx: usize,
    pub(crate) compare: TimelineCompareMode,
    pub(crate) max_cloud_cover: Option<f32>,
    pub(crate) has_cloud_cover_data: bool,
    pub(crate) supports_scalar_diff: bool,
    pub(crate) supports_cross_fade: bool,
    pub(crate) diff_colormap: ColormapId,
    pub(crate) diff_invert_colormap: bool,
    pub(crate) diff_stretch_min: f64,
    pub(crate) diff_stretch_max: f64,
    pub(crate) band_math: Option<BandMathSummary>,
}

/// Opened runtime content for one epoch in a `Timeline` layer.
pub(super) enum TimelineEpochContent {
    Single {
        asset: Box<RasterAsset>,
        style: LayerStyle,
    },
    BandMath(Box<BandMathLayerData>),
}

impl TimelineEpochContent {
    pub(super) fn reference_asset(&self) -> &RasterAsset {
        match self {
            Self::Single { asset, .. } => asset,
            Self::BandMath(bm) => bm.reference_asset(),
        }
    }

    pub(super) fn footprint(&self) -> Option<FootprintBounds> {
        match self {
            Self::Single { asset, .. } => asset.geo().footprint(),
            Self::BandMath(bm) => bm.footprint(),
        }
    }

    pub(super) fn source_pixel_size_m(&self) -> Option<(f64, f64)> {
        match self {
            Self::Single { asset, .. } => asset.geo().source_pixel_size_m(),
            Self::BandMath(bm) => bm.source_pixel_size_m(),
        }
    }

    pub(super) fn for_each_asset(&self, mut f: impl FnMut(&RasterAsset)) {
        match self {
            Self::Single { asset, .. } => f(asset),
            Self::BandMath(bm) => {
                for a in bm.assets() {
                    f(a);
                }
            }
        }
    }

    pub(super) fn for_each_asset_mut(&mut self, mut f: impl FnMut(&mut RasterAsset)) {
        match self {
            Self::Single { asset, .. } => f(asset),
            Self::BandMath(bm) => {
                for a in bm.assets_mut() {
                    f(a);
                }
            }
        }
    }

    pub(super) fn push_active_streams<'a>(&'a self, out: &mut Vec<&'a LayerStream>) {
        match self {
            Self::Single { asset, .. } => out.push(asset.stream()),
            Self::BandMath(bm) => {
                for b in &bm.bands {
                    out.push(b.stream());
                }
                if bm.style.scl_mask_enabled
                    && let Some(scl) = &bm.scl
                {
                    out.push(scl.stream());
                }
            }
        }
    }

    pub(super) fn active_stream_count(&self) -> usize {
        match self {
            Self::Single { .. } => 1,
            Self::BandMath(bm) => {
                bm.bands.len() + usize::from(bm.style.scl_mask_enabled && bm.scl.is_some())
            }
        }
    }

    pub(super) fn clear_wanted(&mut self) {
        self.for_each_asset_mut(RasterAsset::clear_wanted);
    }

    pub(super) fn supports_scalar_diff(&self) -> bool {
        match self {
            Self::Single { asset, .. } => asset.meta().spec.format == PixelFormat::R32Float,
            Self::BandMath(bm) => bm.style.formula.is_scalar(),
        }
    }

    pub(super) fn supports_cross_fade(&self) -> bool {
        match self {
            Self::Single { asset, .. } => asset.meta().spec.format == PixelFormat::Rgba8UnormSrgb,
            Self::BandMath(bm) => !bm.style.formula.is_scalar() && bm.bands.len() >= 3,
        }
    }
}

/// Lifecycle state of one epoch inside [`TimelineLayerData`].
pub(super) enum TimelineEpochState {
    Unloaded,
    Loading {
        open_seq: u64,
        cancel: CancelToken,
    },
    Ready {
        content: TimelineEpochContent,
        last_used_seq: u64,
    },
    Failed(String),
}

/// One chronological epoch managed by [`TimelineLayerData`].
pub(super) struct TimelineEpoch {
    pub(super) label: String,
    pub(super) date_label: String,
    pub(super) sort_key: String,
    pub(super) cloud_cover: Option<f32>,
    pub(super) source: TimelineEpochSourceSpec,
    pub(super) state: TimelineEpochState,
}

impl TimelineEpoch {
    pub(super) fn ready_content(&self) -> Option<&TimelineEpochContent> {
        match &self.state {
            TimelineEpochState::Ready { content, .. } => Some(content),
            _ => None,
        }
    }

    pub(super) fn ready_content_mut(&mut self) -> Option<&mut TimelineEpochContent> {
        match &mut self.state {
            TimelineEpochState::Ready { content, .. } => Some(content),
            _ => None,
        }
    }

    pub(super) fn passes_cloud_filter(&self, max_cloud_cover: Option<f32>) -> bool {
        match (max_cloud_cover, self.cloud_cover) {
            (Some(limit), Some(cc)) => cc <= limit + 1e-3,
            _ => true,
        }
    }
}

/// Data payload for `LayerKind::Timeline`.
pub(super) struct TimelineLayerData {
    pub(super) epochs: Vec<TimelineEpoch>,
    pub(super) recipe_preset: Option<BandMathPreset>,
    pub(super) band_math_style: Option<BandMathStyle>,
    pub(super) single_style: Option<LayerStyleSpec>,
    pub(super) active_idx: usize,
    pub(super) compare: TimelineCompareMode,
    pub(super) max_cloud_cover: Option<f32>,
    pub(super) diff_colormap: ColormapId,
    pub(super) diff_invert_colormap: bool,
    pub(super) diff_stretch_min: f64,
    pub(super) diff_stretch_max: f64,
    pub(super) last_known_footprint: Option<FootprintBounds>,
    pub(super) fly_to_on_first_ready: bool,
    pub(super) restore_id: Option<u64>,
    use_seq: u64,
}

impl TimelineLayerData {
    pub(super) fn from_request(req: TimelineOpenRequest) -> Self {
        let len = req.epochs.len();
        let active_idx = req
            .active_idx
            .unwrap_or_else(|| len.saturating_sub(1))
            .min(len.saturating_sub(1));
        let has_scl = req.epochs.iter().any(|e| {
            matches!(
                &e.source,
                TimelineEpochSourceSpec::BandMath { scl: Some(_), .. }
            )
        });
        let band_math_style = req
            .band_math_style
            .as_ref()
            .map(|spec| BandMathStyle::from_spec(spec, has_scl))
            .or_else(|| {
                req.recipe_preset
                    .map(|preset| BandMathStyle::from_preset(preset, has_scl))
            });
        let (default_diff_min, default_diff_max) = match &band_math_style {
            Some(s) if s.formula.is_scalar() => (-0.5, 0.5),
            _ => (-10.0, 10.0),
        };
        let compare = match req.compare.unwrap_or(TimelineCompareMode::Single) {
            TimelineCompareMode::Single => TimelineCompareMode::Single,
            TimelineCompareMode::ScalarDiff { ref_idx } => TimelineCompareMode::ScalarDiff {
                ref_idx: ref_idx.min(len.saturating_sub(1)),
            },
            TimelineCompareMode::CrossFade { ref_idx, blend_t } => TimelineCompareMode::CrossFade {
                ref_idx: ref_idx.min(len.saturating_sub(1)),
                blend_t: if blend_t.is_finite() {
                    blend_t.clamp(0.0, 1.0)
                } else {
                    0.5
                },
            },
        };
        let epochs = req
            .epochs
            .into_iter()
            .map(|e| TimelineEpoch {
                label: e.label,
                date_label: e.date_label,
                sort_key: e.sort_key,
                cloud_cover: e.cloud_cover,
                source: e.source,
                state: TimelineEpochState::Unloaded,
            })
            .collect();

        let mut data = Self {
            epochs,
            recipe_preset: req.recipe_preset,
            band_math_style,
            single_style: req.single_style,
            active_idx,
            compare,
            max_cloud_cover: req
                .max_cloud_cover
                .filter(|v| v.is_finite())
                .map(|v| v.clamp(0.0, 100.0)),
            diff_colormap: req.diff_colormap.unwrap_or(ColormapId::RdBu),
            diff_invert_colormap: req.diff_invert_colormap,
            diff_stretch_min: req
                .diff_stretch_min
                .filter(|v| v.is_finite())
                .unwrap_or(default_diff_min),
            diff_stretch_max: req
                .diff_stretch_max
                .filter(|v| v.is_finite())
                .unwrap_or(default_diff_max),
            last_known_footprint: None,
            fly_to_on_first_ready: req.fly_to,
            restore_id: req.restore_id,
            use_seq: 1,
        };
        data.clamp_active_to_filter();
        data
    }

    pub(super) fn next_use_seq(&mut self) -> u64 {
        let seq = self.use_seq;
        self.use_seq = self.use_seq.saturating_add(1);
        seq
    }

    pub(super) fn is_remote(&self) -> bool {
        self.epochs.iter().any(|e| e.source.is_remote())
    }

    pub(super) fn remote_url(&self) -> Option<&str> {
        self.active_ready_content()
            .and_then(|c| c.reference_asset().remote_url())
            .or_else(|| {
                self.epochs
                    .iter()
                    .find_map(|e| e.ready_content()?.reference_asset().remote_url())
            })
    }

    pub(super) fn filtered_indices(&self) -> Vec<usize> {
        let filtered: Vec<usize> = self
            .epochs
            .iter()
            .enumerate()
            .filter(|(_, e)| e.passes_cloud_filter(self.max_cloud_cover))
            .map(|(i, _)| i)
            .collect();
        if filtered.is_empty() && !self.epochs.is_empty() {
            // Never trap the user on an empty selection if all scenes exceed threshold.
            return (0..self.epochs.len()).collect();
        }
        filtered
    }

    pub(super) fn clamp_active_to_filter(&mut self) {
        let filtered = self.filtered_indices();
        if filtered.is_empty() {
            self.active_idx = 0;
            return;
        }
        if !filtered.contains(&self.active_idx) {
            // Snap to nearest passing epoch.
            let nearest = filtered
                .iter()
                .copied()
                .min_by_key(|&idx| idx.abs_diff(self.active_idx))
                .unwrap_or(filtered[0]);
            self.active_idx = nearest;
        }
    }

    /// Return the primary rendered epoch indices: `[Some(active_idx), ref_idx_if_distinct]`.
    pub(super) fn render_epoch_indices(&self) -> [Option<usize>; 2] {
        if self.epochs.is_empty() {
            return [None, None];
        }
        let active = self.active_idx.min(self.epochs.len() - 1);
        let ref_idx = self
            .compare
            .ref_idx()
            .map(|r| r.min(self.epochs.len() - 1))
            .filter(|&r| r != active);
        [Some(active), ref_idx]
    }

    /// Return the adjacent filtered prefetch epoch indices (`active_idx ± 1`),
    /// excluding indices already in `render_epoch_indices()`.
    pub(super) fn adjacent_prefetch_indices(&self) -> [Option<usize>; 2] {
        let filtered = self.filtered_indices();
        let Some(pos) = filtered.iter().position(|&i| i == self.active_idx) else {
            return [None, None];
        };
        let render = self.render_epoch_indices();
        let prev = pos
            .checked_sub(1)
            .and_then(|p| filtered.get(p).copied())
            .filter(|&idx| !render.contains(&Some(idx)));
        let next = filtered
            .get(pos + 1)
            .copied()
            .filter(|&idx| !render.contains(&Some(idx)) && Some(idx) != prev);
        [prev, next]
    }

    /// Return all epoch indices that should stay open/prefetched (`active`, `ref`,
    /// `prev`, `next`), deduplicated in priority order.
    pub(super) fn needed_epoch_indices(&self) -> Vec<usize> {
        let mut out = Vec::with_capacity(4);
        for idx in self
            .render_epoch_indices()
            .into_iter()
            .chain(self.adjacent_prefetch_indices())
            .flatten()
        {
            if !out.contains(&idx) {
                out.push(idx);
            }
        }
        out
    }

    pub(super) fn active_epoch(&self) -> Option<&TimelineEpoch> {
        self.epochs.get(self.active_idx)
    }

    pub(super) fn active_ready_content(&self) -> Option<&TimelineEpochContent> {
        self.active_epoch().and_then(TimelineEpoch::ready_content)
    }

    pub(super) fn ref_ready_content(&self) -> Option<&TimelineEpochContent> {
        self.compare
            .ref_idx()
            .filter(|&r| r != self.active_idx)
            .and_then(|i| self.epochs.get(i))
            .and_then(TimelineEpoch::ready_content)
    }

    pub(super) fn is_scalar_diff_ready(&self) -> bool {
        matches!(self.compare, TimelineCompareMode::ScalarDiff { .. })
            && self.active_ready_content().is_some()
            && self.ref_ready_content().is_some()
    }

    pub(super) fn any_ready_content(&self) -> Option<&TimelineEpochContent> {
        self.active_ready_content()
            .or_else(|| self.ref_ready_content())
            .or_else(|| self.epochs.iter().find_map(TimelineEpoch::ready_content))
    }

    pub(super) fn footprint(&self) -> Option<FootprintBounds> {
        self.active_ready_content()
            .and_then(TimelineEpochContent::footprint)
            .or_else(|| {
                self.any_ready_content()
                    .and_then(TimelineEpochContent::footprint)
            })
            .or(self.last_known_footprint)
    }

    pub(super) fn source_pixel_size_m(&self) -> Option<(f64, f64)> {
        self.active_ready_content()
            .and_then(TimelineEpochContent::source_pixel_size_m)
            .or_else(|| {
                self.any_ready_content()
                    .and_then(TimelineEpochContent::source_pixel_size_m)
            })
    }

    pub(super) fn active_streams(&self) -> Vec<&LayerStream> {
        let mut out = Vec::new();
        for idx in self.needed_epoch_indices() {
            if let Some(content) = self.epochs.get(idx).and_then(TimelineEpoch::ready_content) {
                content.push_active_streams(&mut out);
            }
        }
        out
    }

    pub(super) fn active_stream_count(&self) -> usize {
        self.needed_epoch_indices()
            .into_iter()
            .filter_map(|idx| self.epochs.get(idx)?.ready_content())
            .map(TimelineEpochContent::active_stream_count)
            .sum()
    }

    pub(super) fn all_source_ids(&self) -> Vec<SourceId> {
        let mut out = Vec::new();
        for epoch in &self.epochs {
            if let Some(content) = epoch.ready_content() {
                content.for_each_asset(|a| {
                    let src = a.source_id();
                    if !out.contains(&src) {
                        out.push(src);
                    }
                });
            }
        }
        out
    }

    pub(super) fn has_source(&self, source_id: SourceId) -> bool {
        self.epochs.iter().any(|epoch| {
            let mut found = false;
            if let Some(content) = epoch.ready_content() {
                content.for_each_asset(|a| {
                    if a.source_id() == source_id {
                        found = true;
                    }
                });
            }
            found
        })
    }

    pub(super) fn has_remote_source(&self, source_id: SourceId) -> bool {
        self.epochs.iter().any(|epoch| {
            let mut found = false;
            if let Some(content) = epoch.ready_content() {
                content.for_each_asset(|a| {
                    if a.source_id() == source_id && a.source().is_remote() {
                        found = true;
                    }
                });
            }
            found
        })
    }

    pub(super) fn all_stream_keys(&self) -> Vec<StreamKey> {
        let mut out = Vec::new();
        for epoch in &self.epochs {
            if let Some(content) = epoch.ready_content() {
                content.for_each_asset(|a| {
                    let sk = a.meta().stream_key();
                    if !out.contains(&sk) {
                        out.push(sk);
                    }
                });
            }
        }
        out
    }

    pub(super) fn cancel_all_loading(&mut self) {
        for epoch in &mut self.epochs {
            if let TimelineEpochState::Loading { cancel, .. } = &epoch.state {
                cancel.cancel();
                epoch.state = TimelineEpochState::Unloaded;
            }
        }
    }

    /// Cancel in-flight opens for epochs no longer in `needed_epoch_indices()`
    /// and evict oldest non-needed ready epochs above [`MAX_OPEN_TIMELINE_EPOCHS`],
    /// returning the `StreamKey`s of evicted epochs so the manager can cancel
    /// their decode jobs (while intentionally leaving `GpuTileCache` warm).
    pub(super) fn trim_and_cancel_unneeded(&mut self) -> Vec<StreamKey> {
        let needed = self.needed_epoch_indices();
        let mut cancelled_streams = Vec::new();

        for (idx, epoch) in self.epochs.iter_mut().enumerate() {
            if needed.contains(&idx) {
                continue;
            }
            match &mut epoch.state {
                TimelineEpochState::Loading { cancel, .. } => {
                    cancel.cancel();
                    epoch.state = TimelineEpochState::Unloaded;
                }
                TimelineEpochState::Ready { content, .. } => {
                    content.clear_wanted();
                    content.for_each_asset(|a| {
                        let sk = a.meta().stream_key();
                        if !cancelled_streams.contains(&sk) {
                            cancelled_streams.push(sk);
                        }
                    });
                }
                TimelineEpochState::Unloaded | TimelineEpochState::Failed(_) => {}
            }
        }

        // Enforce bounded LRU of open epochs (never evicting needed epochs).
        loop {
            let ready_count = self
                .epochs
                .iter()
                .filter(|e| matches!(e.state, TimelineEpochState::Ready { .. }))
                .count();
            if ready_count <= MAX_OPEN_TIMELINE_EPOCHS {
                break;
            }
            let victim = self
                .epochs
                .iter()
                .enumerate()
                .filter(|(idx, e)| {
                    !needed.contains(idx) && matches!(e.state, TimelineEpochState::Ready { .. })
                })
                .min_by_key(|(_, e)| match &e.state {
                    TimelineEpochState::Ready { last_used_seq, .. } => *last_used_seq,
                    _ => u64::MAX,
                })
                .map(|(idx, _)| idx);
            let Some(victim_idx) = victim else {
                break;
            };
            if let Some(content) = self.epochs[victim_idx].ready_content() {
                content.for_each_asset(|a| {
                    let sk = a.meta().stream_key();
                    if !cancelled_streams.contains(&sk) {
                        cancelled_streams.push(sk);
                    }
                });
            }
            self.epochs[victim_idx].state = TimelineEpochState::Unloaded;
        }

        cancelled_streams
    }

    /// Touch `last_used_seq` on all currently needed ready epochs so LRU
    /// preserves active, ref, and adjacent epochs.
    pub(super) fn touch_needed_ready(&mut self) {
        let needed = self.needed_epoch_indices();
        for idx in needed {
            let seq = self.next_use_seq();
            if let Some(epoch) = self.epochs.get_mut(idx)
                && let TimelineEpochState::Ready { last_used_seq, .. } = &mut epoch.state
            {
                *last_used_seq = seq;
            }
        }
    }

    pub(super) fn clear_all_wanted(&mut self) {
        for epoch in &mut self.epochs {
            if let Some(content) = epoch.ready_content_mut() {
                content.clear_wanted();
            }
        }
    }

    pub(super) fn observe_stream_tile_bytes(&mut self, stream: StreamKey, bytes: usize) {
        for epoch in &mut self.epochs {
            if let Some(content) = epoch.ready_content_mut() {
                content.for_each_asset_mut(|a| a.observe_stream_tile_bytes(stream, bytes));
            }
        }
    }

    fn single_epoch_scratch_requirements(active: &TimelineEpochContent) -> ScratchDemand {
        match active {
            TimelineEpochContent::Single { .. } => ScratchDemand::NONE,
            TimelineEpochContent::BandMath(bm) => ScratchDemand {
                float_bands: bm.bands.len().min(4),
                needs_u8_mask: bm.scl.is_some() && bm.style.scl_mask_enabled,
                rgba_targets: 0,
            },
        }
    }

    pub(super) fn scratch_requirements(&self) -> ScratchDemand {
        let Some(active) = self.active_ready_content() else {
            return ScratchDemand::NONE;
        };
        let fallback = || Self::single_epoch_scratch_requirements(active);
        match self.compare {
            TimelineCompareMode::Single => fallback(),
            TimelineCompareMode::ScalarDiff { .. } => {
                let Some(ref_c) = self.ref_ready_content() else {
                    return fallback();
                };
                match (ref_c, active) {
                    (
                        TimelineEpochContent::BandMath(bm_a),
                        TimelineEpochContent::BandMath(bm_b),
                    ) if bm_a.style.formula.is_scalar() && bm_b.style.formula.is_scalar() => {
                        let n = bm_b.style.formula.required_bands().clamp(1, 3);
                        ScratchDemand {
                            float_bands: (n + 1).min(4),
                            needs_u8_mask: bm_b.style.scl_mask_enabled
                                && (bm_a.scl.is_some() || bm_b.scl.is_some()),
                            rgba_targets: 0,
                        }
                    }
                    (
                        TimelineEpochContent::Single { asset: a, .. },
                        TimelineEpochContent::Single { asset: b, .. },
                    ) if a.meta().spec.format == PixelFormat::R32Float
                        && b.meta().spec.format == PixelFormat::R32Float =>
                    {
                        ScratchDemand {
                            float_bands: 2,
                            needs_u8_mask: false,
                            rgba_targets: 0,
                        }
                    }
                    _ => fallback(),
                }
            }
            TimelineCompareMode::CrossFade { .. } => {
                let Some(ref_c) = self.ref_ready_content() else {
                    return fallback();
                };
                match (ref_c, active) {
                    (
                        TimelineEpochContent::BandMath(bm_a),
                        TimelineEpochContent::BandMath(bm_b),
                    ) if !bm_a.style.formula.is_scalar() && !bm_b.style.formula.is_scalar() => {
                        ScratchDemand {
                            float_bands: 3,
                            needs_u8_mask: bm_b.style.scl_mask_enabled
                                && (bm_a.scl.is_some() || bm_b.scl.is_some()),
                            rgba_targets: 1,
                        }
                    }
                    (
                        TimelineEpochContent::Single { asset: a, .. },
                        TimelineEpochContent::Single { asset: b, .. },
                    ) if a.meta().spec.format == PixelFormat::Rgba8UnormSrgb
                        && b.meta().spec.format == PixelFormat::Rgba8UnormSrgb =>
                    {
                        ScratchDemand {
                            float_bands: 0,
                            needs_u8_mask: false,
                            rgba_targets: 2,
                        }
                    }
                    _ => fallback(),
                }
            }
        }
    }

    /// Synchronize shared `BandMathStyle` into all currently loaded BandMath epochs.
    pub(super) fn sync_band_math_style_to_epochs(&mut self) -> Vec<StreamKey> {
        let Some(style) = self.band_math_style.clone() else {
            return Vec::new();
        };
        let mut disabled_scl = Vec::new();
        for epoch in &mut self.epochs {
            if let Some(TimelineEpochContent::BandMath(bm)) = epoch.ready_content_mut() {
                let was_enabled = bm.style.scl_mask_enabled;
                bm.style = style.clone();
                if was_enabled
                    && !bm.style.scl_mask_enabled
                    && let Some(scl) = &mut bm.scl
                {
                    scl.clear_wanted();
                    disabled_scl.push(scl.meta().stream_key());
                }
            }
        }
        disabled_scl
    }

    pub(super) fn supports_scalar_diff(&self) -> bool {
        if let Some(content) = self.any_ready_content() {
            return content.supports_scalar_diff();
        }
        self.band_math_style
            .as_ref()
            .is_some_and(|s| s.formula.is_scalar())
    }

    pub(super) fn supports_cross_fade(&self) -> bool {
        if let Some(content) = self.any_ready_content() {
            return content.supports_cross_fade();
        }
        self.band_math_style
            .as_ref()
            .is_none_or(|s| !s.formula.is_scalar())
    }

    pub(super) fn pick_default_ref_idx(&self) -> usize {
        let filtered = self.filtered_indices();
        if let Some(pos) = filtered.iter().position(|&i| i == self.active_idx) {
            if pos > 0 {
                return filtered[pos - 1];
            }
            if pos + 1 < filtered.len() {
                return filtered[pos + 1];
            }
        }
        if self.active_idx > 0 {
            self.active_idx - 1
        } else {
            1.min(self.epochs.len().saturating_sub(1))
        }
    }

    pub(super) fn set_active_epoch(&mut self, idx: usize) -> bool {
        if idx >= self.epochs.len() || idx == self.active_idx {
            return false;
        }
        self.active_idx = idx;
        self.touch_needed_ready();
        true
    }

    pub(super) fn step_epoch(&mut self, delta: i32) -> bool {
        let filtered = self.filtered_indices();
        if filtered.len() < 2 || delta == 0 {
            return false;
        }
        let cur_pos = filtered
            .iter()
            .position(|&i| i == self.active_idx)
            .unwrap_or(0);
        let target_pos = if delta > 0 {
            cur_pos
                .saturating_add(delta as usize)
                .min(filtered.len() - 1)
        } else {
            cur_pos.saturating_sub(delta.unsigned_abs() as usize)
        };
        self.set_active_epoch(filtered[target_pos])
    }

    pub(super) fn set_cloud_filter(&mut self, max_cloud_cover: Option<f32>) -> bool {
        let next = max_cloud_cover
            .filter(|v| v.is_finite())
            .map(|v| v.clamp(0.0, 100.0));
        if self.max_cloud_cover == next {
            return false;
        }
        self.max_cloud_cover = next;
        self.clamp_active_to_filter();
        self.touch_needed_ready();
        true
    }

    pub(super) fn set_compare_mode(&mut self, mode: TimelineCompareMode) -> bool {
        let len = self.epochs.len();
        let normalized = match mode {
            TimelineCompareMode::Single => TimelineCompareMode::Single,
            TimelineCompareMode::ScalarDiff { ref_idx } => {
                if !self.supports_scalar_diff() || len < 2 {
                    return false;
                }
                let r = if ref_idx < len && ref_idx != self.active_idx {
                    ref_idx
                } else {
                    self.pick_default_ref_idx()
                };
                TimelineCompareMode::ScalarDiff { ref_idx: r }
            }
            TimelineCompareMode::CrossFade { ref_idx, blend_t } => {
                if !self.supports_cross_fade() || len < 2 {
                    return false;
                }
                let r = if ref_idx < len && ref_idx != self.active_idx {
                    ref_idx
                } else {
                    self.pick_default_ref_idx()
                };
                let t = if blend_t.is_finite() {
                    blend_t.clamp(0.0, 1.0)
                } else {
                    0.5
                };
                TimelineCompareMode::CrossFade {
                    ref_idx: r,
                    blend_t: t,
                }
            }
        };
        if self.compare == normalized {
            return false;
        }
        self.compare = normalized;
        self.touch_needed_ready();
        true
    }

    pub(super) fn set_blend_t(&mut self, blend_t: f32) -> bool {
        let TimelineCompareMode::CrossFade {
            ref_idx,
            blend_t: cur,
        } = self.compare
        else {
            return false;
        };
        let next = if blend_t.is_finite() {
            blend_t.clamp(0.0, 1.0)
        } else {
            cur
        };
        if (cur - next).abs() <= f32::EPSILON {
            return false;
        }
        self.compare = TimelineCompareMode::CrossFade {
            ref_idx,
            blend_t: next,
        };
        true
    }

    pub(super) fn set_diff_stretch_range(&mut self, min: f64, max: f64) -> bool {
        if !min.is_finite() || !max.is_finite() || (max - min).abs() <= 1e-12 {
            return false;
        }
        let (lo, hi) = if min < max { (min, max) } else { (max, min) };
        if (self.diff_stretch_min - lo).abs() <= 1e-12
            && (self.diff_stretch_max - hi).abs() <= 1e-12
        {
            return false;
        }
        self.diff_stretch_min = lo;
        self.diff_stretch_max = hi;
        true
    }

    pub(super) fn set_diff_colormap(&mut self, colormap: ColormapId, invert: bool) -> bool {
        if self.diff_colormap == colormap && self.diff_invert_colormap == invert {
            return false;
        }
        self.diff_colormap = colormap;
        self.diff_invert_colormap = invert;
        true
    }

    pub(super) fn to_summary(&self) -> TimelineSummary {
        let filtered_indices = self.filtered_indices();
        let has_cloud_cover_data = self.epochs.iter().any(|e| e.cloud_cover.is_some());
        let epochs = self
            .epochs
            .iter()
            .map(|e| {
                let (load_state, error) = match &e.state {
                    TimelineEpochState::Unloaded => (TimelineEpochLoadState::Unloaded, None),
                    TimelineEpochState::Loading { .. } => (TimelineEpochLoadState::Loading, None),
                    TimelineEpochState::Ready { .. } => (TimelineEpochLoadState::Ready, None),
                    TimelineEpochState::Failed(err) => {
                        (TimelineEpochLoadState::Failed, Some(err.clone()))
                    }
                };
                TimelineEpochSummary {
                    label: e.label.clone(),
                    date_label: e.date_label.clone(),
                    cloud_cover: e.cloud_cover,
                    load_state,
                    error,
                }
            })
            .collect();

        let dims = self.any_ready_content().map(|c| {
            let d = c.reference_asset().source().description();
            (d.width, d.height)
        });

        let band_math = self.band_math_style.as_ref().map(|style| {
            let ready_bm = self.any_ready_content().and_then(|c| match c {
                TimelineEpochContent::BandMath(bm) => Some(bm.as_ref()),
                TimelineEpochContent::Single { .. } => None,
            });
            let (width, height) = dims.unwrap_or((0, 0));
            let has_scl = ready_bm.map_or_else(
                || {
                    self.epochs.iter().any(|e| {
                        matches!(
                            &e.source,
                            TimelineEpochSourceSpec::BandMath { scl: Some(_), .. }
                        )
                    })
                },
                |bm| bm.scl.is_some(),
            );
            let inputs = ready_bm
                .map(|bm| {
                    bm.bands
                        .iter()
                        .enumerate()
                        .map(|(idx, a)| BandMathInputSummary {
                            slot_name: style
                                .formula
                                .slot_names()
                                .get(idx)
                                .copied()
                                .unwrap_or("Band")
                                .to_owned(),
                            label: a.label().to_owned(),
                            source_pixel_size_m: a.geo().source_pixel_size_m(),
                        })
                        .collect()
                })
                .unwrap_or_default();
            BandMathSummary {
                width,
                height,
                formula: style.formula,
                recipe_label: style.recipe_label.clone(),
                colormap: style.colormap,
                invert_colormap: style.invert_colormap,
                stretch_min: style.stretch_min,
                stretch_max: style.stretch_max,
                rgb_min: style.rgb_min,
                rgb_max: style.rgb_max,
                rgb_gamma: style.rgb_gamma,
                has_scl,
                scl_mask_enabled: style.scl_mask_enabled,
                scl_mask_bits: style.scl_mask_bits,
                inputs,
                scl_label: has_scl.then(|| "SCL".to_owned()),
            }
        });

        TimelineSummary {
            width: dims.map(|(w, _)| w),
            height: dims.map(|(_, h)| h),
            epochs,
            filtered_indices,
            active_idx: self.active_idx,
            compare: self.compare,
            max_cloud_cover: self.max_cloud_cover,
            has_cloud_cover_data,
            supports_scalar_diff: self.supports_scalar_diff(),
            supports_cross_fade: self.supports_cross_fade(),
            diff_colormap: self.diff_colormap,
            diff_invert_colormap: self.diff_invert_colormap,
            diff_stretch_min: self.diff_stretch_min,
            diff_stretch_max: self.diff_stretch_max,
            band_math,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_dummy_epoch(date: &str, cloud: Option<f32>) -> TimelineEpochSpec {
        TimelineEpochSpec {
            label: format!("scene_{date}"),
            date_label: date.to_owned(),
            sort_key: format!("0:{date}"),
            cloud_cover: cloud,
            source: TimelineEpochSourceSpec::LocalSingle {
                path: PathBuf::from(format!("{date}.tif")),
                crs_override: None,
            },
        }
    }

    #[test]
    fn cloud_filter_and_adjacent_prefetch_indices_skip_cloudy_epochs() {
        let req = TimelineOpenRequest {
            title: "Test Timeline".to_owned(),
            epochs: vec![
                make_dummy_epoch("2024-05-01", Some(5.0)),
                make_dummy_epoch("2024-05-15", Some(80.0)), // filtered out under <=20%
                make_dummy_epoch("2024-06-01", Some(10.0)),
                make_dummy_epoch("2024-06-15", Some(15.0)),
            ],
            recipe_preset: Some(BandMathPreset::Ndvi),
            band_math_style: None,
            single_style: None,
            visible: true,
            opacity: 1.0,
            active_idx: Some(2),
            compare: None,
            max_cloud_cover: Some(20.0),
            diff_colormap: None,
            diff_invert_colormap: false,
            diff_stretch_min: None,
            diff_stretch_max: None,
            fly_to: false,
            restore_id: None,
        };
        let mut tl = TimelineLayerData::from_request(req);
        assert_eq!(tl.filtered_indices(), vec![0, 2, 3]);
        assert_eq!(tl.active_idx, 2);
        // Adjacent filtered epochs around 2 are 0 (skipping cloudy 1) and 3!
        assert_eq!(tl.adjacent_prefetch_indices(), [Some(0), Some(3)]);

        // Stepping backwards from 2 skips cloudy index 1 and lands on 0.
        assert!(tl.step_epoch(-1));
        assert_eq!(tl.active_idx, 0);
        assert_eq!(tl.adjacent_prefetch_indices(), [None, Some(2)]);
    }
}
