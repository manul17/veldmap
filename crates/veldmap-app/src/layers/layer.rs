//! Georeferenced raster overlay (`OverlayLayer`) with tagged [`LayerKind`]
//! (`SingleRaster` | `BandMath`, extensible to Slice 5B `DemMosaic` and
//! `Timeline`) backed by cohesive [`RasterAsset`] units.

#[cfg(test)]
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
#[cfg(test)]
use std::sync::Mutex;

use veldmap_core::geo::{FootprintBounds, Frame};
use veldmap_core::sync::CancelToken;
#[cfg(test)]
use veldmap_core::tile::TileKey;
use veldmap_core::tile::{BandKey, PixelFormat, SourceId, StreamKey, TileStreamSpec};
use veldmap_net::HttpRangeSource;
use veldmap_raster::{BandStats, PyramidMeta, RasterSource, stream_spec_for_band};

use crate::colormap::CategoricalPaletteId;
use crate::render::ScratchDemand;
use crate::{clamp_opacity, display_name_from_path};

use super::asset::{GeoBinding, LayerOrigin, LayerStream, RasterAsset};
use super::band_math::{BandMathLayerData, BandMathStyle};
use super::style::{LayerStyle, LayerStyleSpec, ScientificStyle, TrackKind};
use super::{CrsOverride, LayerId, LayerInitOptions, OverlayPersistEntry};

/// Zero-allocation iterator over every actively rendered [`LayerStream`] owned
/// by an [`OverlayLayer`] (omitting a `BandMath` layer's `SCL` stream while its
/// cloud mask is toggled off).
pub(super) enum ActiveStreams<'a> {
    Single(Option<&'a LayerStream>),
    BandMath {
        bands: std::slice::Iter<'a, RasterAsset>,
        scl: Option<&'a LayerStream>,
    },
}

impl<'a> Iterator for ActiveStreams<'a> {
    type Item = &'a LayerStream;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Single(slot) => slot.take(),
            Self::BandMath { bands, scl } => {
                bands.next().map(RasterAsset::stream).or_else(|| scl.take())
            }
        }
    }
}

/// Layer-domain state transition record returned by [`OverlayLayer::begin_reload`].
pub(super) struct ReloadPrep {
    pub(super) reload_seq: u64,
    pub(super) old_source: SourceId,
    pub(super) old_cancel: Option<CancelToken>,
}

/// Cohesive per-layer reload lifecycle state.
#[derive(Default)]
struct ReloadState {
    in_progress: bool,
    seq: u64,
    cancel: Option<CancelToken>,
}

/// Input parameters for constructing a remote Cloud-Optimized GeoTIFF
/// [`OverlayLayer`] via [`OverlayLayer::from_remote_source`].
pub(super) struct RemoteLayerSpec {
    pub(super) id: LayerId,
    pub(super) title: String,
    pub(super) display_url: String,
    pub(super) range_source: HttpRangeSource,
    pub(super) source: Arc<dyn RasterSource>,
    pub(super) frame: Frame,
    pub(super) options: LayerInitOptions,
    pub(super) is_scl: bool,
    pub(super) relaid: u64,
}

/// Tagged overlay kind distinguishing single-asset rasters from multi-asset
/// GPU band-math / RGB composite layers (and future Slice 5B DEM mosaics and
/// multi-temporal timelines).
pub(super) enum LayerKind {
    SingleRaster {
        asset: Box<RasterAsset>,
        style: LayerStyle,
    },
    BandMath(Box<BandMathLayerData>),
}

/// Georeferenced raster overlay inside [`super::stack::LayerStack`].
pub(super) struct OverlayLayer {
    id: LayerId,
    name: String,
    kind: LayerKind,
    /// Opt-in flag controlling whether this layer's root tiles stay pinned in
    /// `PendingUploads` (defaults `true` for local files; remote COGs and future
    /// DEM mosaics / timelines opt out).
    pin_root: bool,
    last_error: Option<String>,
    visible: bool,
    opacity: f32,
    reload: ReloadState,
}

impl OverlayLayer {
    #[expect(
        clippy::too_many_arguments,
        reason = "Private constructor unifies local and remote SingleRaster layer initialization."
    )]
    fn from_single_parts(
        id: LayerId,
        name: String,
        origin: LayerOrigin,
        source: Arc<dyn RasterSource>,
        frame: Frame,
        mut options: LayerInitOptions,
        is_scl: bool,
        relaid: u64,
    ) -> Self {
        let desc = source.description();
        if is_scl && options.style.is_none() {
            // Explicitly request `PixelFormat::R8Uint` + `Sentinel2Scl` for
            // ESA SCL assets (`U8 BlackIsZero` otherwise defaults to Display).
            options.style = Some(LayerStyleSpec {
                band_index: Some(0),
                track: Some(TrackKind::Categorical),
                categorical_palette: Some(CategoricalPaletteId::Sentinel2Scl),
                ..LayerStyleSpec::default()
            });
        }
        let preferred_spec = options
            .style
            .as_ref()
            .and_then(|s| s.resolve_stream_spec(desc))
            .unwrap_or_else(|| {
                if is_scl {
                    stream_spec_for_band(BandKey::Index(0), PixelFormat::R8Uint)
                } else {
                    source.default_stream_spec()
                }
            });
        let meta = source
            .pyramid_meta(preferred_spec)
            .or_else(|_| source.pyramid_meta(veldmap_raster::DISPLAY_STREAM_SPEC))
            .unwrap_or_else(|_| PyramidMeta {
                source: source.source_id(),
                spec: veldmap_raster::DISPLAY_STREAM_SPEC,
                levels: desc.levels.clone(),
            });
        let style = LayerStyle::new(source.as_ref(), meta.spec, options.style.as_ref());
        let geo = GeoBinding::new(
            frame,
            desc.width,
            desc.height,
            meta.max_level(),
            options.crs_override,
            relaid,
        );
        let pin_root = matches!(origin, LayerOrigin::Local { .. });
        let asset = RasterAsset::new(name.clone(), origin, source, geo, meta);

        Self {
            id,
            name,
            kind: LayerKind::SingleRaster {
                asset: Box::new(asset),
                style,
            },
            pin_root,
            last_error: None,
            visible: options.visible,
            opacity: clamp_opacity(options.opacity),
            reload: ReloadState::default(),
        }
    }

    pub(super) fn from_source(
        id: LayerId,
        path: PathBuf,
        source: Arc<dyn RasterSource>,
        frame: Frame,
        options: LayerInitOptions,
        relaid: u64,
    ) -> Self {
        let name = display_name_from_path(&path);
        Self::from_single_parts(
            id,
            name,
            LayerOrigin::Local { path },
            source,
            frame,
            options,
            false,
            relaid,
        )
    }

    #[cfg(test)]
    pub(super) fn from_scl_source_for_test(
        id: LayerId,
        path: PathBuf,
        source: Arc<dyn RasterSource>,
        frame: Frame,
    ) -> Self {
        Self::from_single_parts(
            id,
            "SCL".to_owned(),
            LayerOrigin::Local { path },
            source,
            frame,
            LayerInitOptions {
                visible: true,
                opacity: 1.0,
                crs_override: None,
                style: None,
            },
            true,
            1,
        )
    }

    /// Construct a remote Cloud-Optimized GeoTIFF overlay layer. Remote layers
    /// do not pin root tiles in `PendingUploads` (`pin_root = false`) and carry
    /// their [`HttpRangeSource`] for asynchronous pinned prefetching.
    pub(super) fn from_remote_source(spec: RemoteLayerSpec) -> Self {
        let name = if spec.title.trim().is_empty() {
            spec.display_url.clone()
        } else {
            spec.title
        };
        Self::from_single_parts(
            spec.id,
            name,
            LayerOrigin::Remote {
                display_url: spec.display_url,
                range: spec.range_source,
            },
            spec.source,
            spec.frame,
            spec.options,
            spec.is_scl,
            spec.relaid,
        )
    }

    /// Construct a multi-asset GPU Band Math / RGB Composite overlay layer.
    ///
    /// Precondition: `bands` is non-empty (`2` for scalar formulas, `3` for
    /// RGB composites).
    pub(super) fn from_band_math(
        id: LayerId,
        name: String,
        bands: Vec<RasterAsset>,
        scl: Option<RasterAsset>,
        style: BandMathStyle,
    ) -> Self {
        debug_assert!(!bands.is_empty());
        let any_remote = bands.iter().any(RasterAsset::is_remote)
            || scl.as_ref().is_some_and(RasterAsset::is_remote);
        Self {
            id,
            name,
            kind: LayerKind::BandMath(Box::new(BandMathLayerData { bands, scl, style })),
            pin_root: !any_remote,
            last_error: None,
            visible: true,
            opacity: 1.0,
            reload: ReloadState::default(),
        }
    }

    pub(super) fn id(&self) -> LayerId {
        self.id
    }

    pub(super) fn name(&self) -> &str {
        &self.name
    }

    pub(super) fn kind(&self) -> &LayerKind {
        &self.kind
    }

    pub(super) fn kind_mut(&mut self) -> &mut LayerKind {
        &mut self.kind
    }

    pub(super) fn primary_asset(&self) -> &RasterAsset {
        match &self.kind {
            LayerKind::SingleRaster { asset, .. } => asset,
            LayerKind::BandMath(bm) => bm.primary_asset(),
        }
    }

    pub(super) fn single_asset(&self) -> Option<&RasterAsset> {
        match &self.kind {
            LayerKind::SingleRaster { asset, .. } => Some(asset),
            LayerKind::BandMath(_) => None,
        }
    }

    #[cfg(test)]
    pub(super) fn style(&self) -> &LayerStyle {
        match &self.kind {
            LayerKind::SingleRaster { style, .. } => style,
            LayerKind::BandMath(_) => panic!("style() called on BandMath layer"),
        }
    }

    #[cfg(test)]
    pub(super) fn style_mut(&mut self) -> &mut LayerStyle {
        match &mut self.kind {
            LayerKind::SingleRaster { style, .. } => style,
            LayerKind::BandMath(_) => panic!("style_mut() called on BandMath layer"),
        }
    }

    pub(super) fn local_path(&self) -> Option<&Path> {
        match &self.kind {
            LayerKind::SingleRaster { asset, .. } => asset.local_path(),
            LayerKind::BandMath(_) => None,
        }
    }

    pub(super) fn is_remote(&self) -> bool {
        match &self.kind {
            LayerKind::SingleRaster { asset, .. } => asset.is_remote(),
            LayerKind::BandMath(bm) => bm.assets().any(RasterAsset::is_remote),
        }
    }

    pub(super) fn remote_url(&self) -> Option<&str> {
        match &self.kind {
            LayerKind::SingleRaster { asset, .. } => asset.remote_url(),
            LayerKind::BandMath(bm) => bm.bands.first().and_then(RasterAsset::remote_url),
        }
    }

    /// Collect all distinct [`SourceId`]s referenced by this layer (1 for
    /// `SingleRaster`, 1–4 for `BandMath`).
    pub(super) fn all_source_ids(&self) -> Vec<SourceId> {
        let mut out = Vec::new();
        match &self.kind {
            LayerKind::SingleRaster { asset, .. } => out.push(asset.source_id()),
            LayerKind::BandMath(bm) => {
                for a in bm.assets() {
                    let src = a.source_id();
                    if !out.contains(&src) {
                        out.push(src);
                    }
                }
            }
        }
        out
    }

    pub(super) fn has_source(&self, source_id: SourceId) -> bool {
        match &self.kind {
            LayerKind::SingleRaster { asset, .. } => asset.source_id() == source_id,
            LayerKind::BandMath(bm) => bm.assets().any(|a| a.source_id() == source_id),
        }
    }

    pub(super) fn has_remote_source(&self, source_id: SourceId) -> bool {
        match &self.kind {
            LayerKind::SingleRaster { asset, .. } => {
                asset.source_id() == source_id && asset.source().is_remote()
            }
            LayerKind::BandMath(bm) => bm
                .assets()
                .any(|a| a.source_id() == source_id && a.source().is_remote()),
        }
    }

    /// Iterate over every actively rendered [`LayerStream`] owned by this layer
    /// without heap allocation (omitting a `BandMath` layer's `SCL` stream
    /// while its cloud mask is toggled off).
    pub(super) fn streams(&self) -> ActiveStreams<'_> {
        match &self.kind {
            LayerKind::SingleRaster { asset, .. } => ActiveStreams::Single(Some(asset.stream())),
            LayerKind::BandMath(bm) => ActiveStreams::BandMath {
                bands: bm.bands.iter(),
                scl: bm
                    .style
                    .scl_mask_enabled
                    .then(|| bm.scl.as_ref().map(RasterAsset::stream))
                    .flatten(),
            },
        }
    }

    /// Number of actively rendered streams yielded by [`Self::streams`] (O(1)).
    pub(super) fn active_stream_count(&self) -> usize {
        match &self.kind {
            LayerKind::SingleRaster { .. } => 1,
            LayerKind::BandMath(bm) => {
                bm.bands.len() + usize::from(bm.style.scl_mask_enabled && bm.scl.is_some())
            }
        }
    }

    /// Collect every [`StreamKey`] owned by this layer (including disabled
    /// `SCL`), used for cancelling orphaned jobs when hiding or removing.
    pub(super) fn all_stream_keys(&self) -> Vec<StreamKey> {
        match &self.kind {
            LayerKind::SingleRaster { asset, .. } => vec![asset.meta().stream_key()],
            LayerKind::BandMath(bm) => {
                let mut keys = Vec::with_capacity(bm.bands.len() + usize::from(bm.scl.is_some()));
                for a in bm.assets() {
                    let key = a.meta().stream_key();
                    if !keys.contains(&key) {
                        keys.push(key);
                    }
                }
                keys
            }
        }
    }

    pub(super) fn is_visible(&self) -> bool {
        self.visible
    }

    pub(super) fn opacity(&self) -> f32 {
        self.opacity
    }

    pub(super) fn last_error(&self) -> Option<&str> {
        self.last_error.as_deref()
    }

    pub(super) fn pin_root(&self) -> bool {
        self.pin_root
    }

    #[cfg(test)]
    pub(super) fn set_pin_root(&mut self, pin_root: bool) {
        self.pin_root = pin_root;
    }

    /// Return the shared viewport [`ScratchDemand`] required when this layer is
    /// active.
    pub(super) fn scratch_requirements(&self) -> ScratchDemand {
        if !self.is_active() {
            return ScratchDemand::NONE;
        }
        match &self.kind {
            LayerKind::SingleRaster { .. } => ScratchDemand::NONE,
            LayerKind::BandMath(bm) => ScratchDemand {
                float_bands: bm.bands.len().min(4),
                needs_u8_mask: bm.scl.is_some() && bm.style.scl_mask_enabled,
            },
        }
    }

    pub(super) fn needs_band_stats(&self) -> Option<BandKey> {
        if !self.is_active() {
            return None;
        }
        match &self.kind {
            LayerKind::SingleRaster { asset, style, .. } => {
                let spec = asset.meta().spec;
                if spec.format == PixelFormat::R32Float && style.scientific().needs_stats() {
                    Some(spec.band)
                } else {
                    None
                }
            }
            LayerKind::BandMath(_) => None,
        }
    }

    pub(super) fn begin_stats_job(&mut self, job_id: u64, cancel: CancelToken) {
        if let LayerKind::SingleRaster { style, .. } = &mut self.kind {
            style.scientific_mut().begin_loading(job_id, cancel);
        }
    }

    pub(super) fn cancel_stats_job(&mut self) {
        if let LayerKind::SingleRaster { style, .. } = &mut self.kind {
            style.scientific_mut().cancel_loading();
        }
    }

    fn matching_scientific_style_mut(&mut self, band: BandKey) -> Option<&mut ScientificStyle> {
        let LayerKind::SingleRaster { asset, style, .. } = &mut self.kind else {
            return None;
        };
        if asset.meta().spec.band != band || asset.meta().spec.format != PixelFormat::R32Float {
            return None;
        }
        Some(style.scientific_mut())
    }

    pub(super) fn apply_band_stats_ready(&mut self, band: BandKey, stats: BandStats) -> bool {
        let Some(sci) = self.matching_scientific_style_mut(band) else {
            return false;
        };
        sci.apply_ready(stats);
        true
    }

    pub(super) fn apply_band_stats_empty(&mut self, band: BandKey, job_id: u64) -> bool {
        self.matching_scientific_style_mut(band)
            .is_some_and(|sci| sci.apply_empty(job_id))
    }

    pub(super) fn apply_band_stats_cancelled(&mut self, band: BandKey, job_id: u64) -> bool {
        self.matching_scientific_style_mut(band)
            .is_some_and(|sci| sci.apply_cancelled(job_id))
    }

    /// Switch a single-raster layer's active stream (`Display` / `Scientific` /
    /// `Categorical` + band index). Returns `true` if the active stream changed.
    pub(super) fn set_active_stream(&mut self, spec: TileStreamSpec) -> Result<bool, String> {
        let LayerKind::SingleRaster { asset, style } = &mut self.kind else {
            return Err("band-math layers do not support switching single-band streams".to_owned());
        };
        let changed = asset.set_active_stream(spec)?;
        if changed {
            style.on_stream_switched(spec);
        }
        Ok(changed)
    }

    pub(super) fn saved_style(&self) -> Option<LayerStyleSpec> {
        match &self.kind {
            LayerKind::SingleRaster { asset, style, .. } => {
                style.saved_style(asset.source().as_ref(), asset.meta().spec)
            }
            LayerKind::BandMath(_) => None,
        }
    }

    pub(super) fn persist_entry(&self) -> Option<OverlayPersistEntry<'_>> {
        let LayerKind::SingleRaster { asset, .. } = &self.kind else {
            return None;
        };
        let path = asset.local_path()?;
        Some(OverlayPersistEntry {
            path,
            visible: self.visible,
            opacity: self.opacity,
            crs_override: asset.geo().crs_override(),
            style: self.saved_style(),
        })
    }

    pub(super) fn set_visible(&mut self, visible: bool) -> bool {
        if self.visible == visible {
            return false;
        }
        self.visible = visible;
        if !visible {
            self.clear_wanted();
        }
        true
    }

    pub(super) fn set_opacity(&mut self, opacity: f32) -> bool {
        let clamped = clamp_opacity(opacity);
        if (self.opacity - clamped).abs() <= f32::EPSILON {
            return false;
        }
        self.opacity = clamped;
        true
    }

    pub(super) fn set_stretch_preset(&mut self, preset: super::StretchPreset) -> bool {
        match &mut self.kind {
            LayerKind::SingleRaster { style, .. } => style.set_stretch_preset(preset),
            LayerKind::BandMath(_) => false,
        }
    }

    pub(super) fn set_stretch_range(&mut self, min: f64, max: f64) -> bool {
        match &mut self.kind {
            LayerKind::SingleRaster { style, .. } => style.set_stretch_range(min, max),
            LayerKind::BandMath(bm) => bm.style.set_stretch_range(min, max),
        }
    }

    pub(super) fn set_colormap(
        &mut self,
        colormap: crate::colormap::ColormapId,
        invert: bool,
    ) -> bool {
        match &mut self.kind {
            LayerKind::SingleRaster { style, .. } => style.set_colormap(colormap, invert),
            LayerKind::BandMath(bm) => bm.style.set_colormap(colormap, invert),
        }
    }

    pub(super) fn set_categorical_palette(&mut self, palette: CategoricalPaletteId) -> bool {
        match &mut self.kind {
            LayerKind::SingleRaster { style, .. } => style.set_categorical_palette(palette),
            LayerKind::BandMath(_) => false,
        }
    }

    pub(super) fn set_band_math_rgb_stretch(
        &mut self,
        min: [f64; 3],
        max: [f64; 3],
        gamma: f32,
    ) -> bool {
        match &mut self.kind {
            LayerKind::BandMath(bm) => bm.style.set_rgb_stretch(min, max, gamma),
            LayerKind::SingleRaster { .. } => false,
        }
    }

    pub(super) fn reset_band_math_stretch(&mut self) -> bool {
        match &mut self.kind {
            LayerKind::BandMath(bm) => bm.style.reset_stretch(),
            LayerKind::SingleRaster { .. } => false,
        }
    }

    pub(super) fn set_band_math_scl_enabled(&mut self, enabled: bool) -> (bool, Option<StreamKey>) {
        let LayerKind::BandMath(bm) = &mut self.kind else {
            return (false, None);
        };
        if !bm.style.set_scl_mask_enabled(enabled) {
            return (false, None);
        }
        let disabled_stream = if !enabled {
            if let Some(scl) = &mut bm.scl {
                scl.clear_wanted();
                Some(scl.meta().stream_key())
            } else {
                None
            }
        } else {
            None
        };
        (true, disabled_stream)
    }

    pub(super) fn set_band_math_scl_mask_bits(&mut self, bits: u32) -> bool {
        match &mut self.kind {
            LayerKind::BandMath(bm) => bm.style.set_scl_mask_bits(bits),
            LayerKind::SingleRaster { .. } => false,
        }
    }

    pub(super) fn observe_stream_tile_bytes(&mut self, stream: StreamKey, bytes: usize) {
        match &mut self.kind {
            LayerKind::SingleRaster { asset, .. } => {
                asset.observe_stream_tile_bytes(stream, bytes);
            }
            LayerKind::BandMath(bm) => {
                for a in bm.assets_mut() {
                    a.observe_stream_tile_bytes(stream, bytes);
                }
            }
        }
    }

    pub(super) fn apply_rebound_frame(
        &mut self,
        frame: Frame,
        crs_override: Option<CrsOverride>,
        relaid: u64,
    ) {
        if let LayerKind::SingleRaster { asset, .. } = &mut self.kind {
            asset.apply_rebound_frame(frame, crs_override, relaid);
            self.last_error = None;
        }
    }

    pub(super) fn footprint(&self) -> Option<FootprintBounds> {
        self.primary_asset().geo().footprint()
    }

    pub(super) fn is_active(&self) -> bool {
        self.visible && !self.reload.in_progress
    }

    pub(super) fn is_reloading(&self) -> bool {
        self.reload.in_progress
    }

    pub(super) fn begin_reload(&mut self, cancel: CancelToken) -> ReloadPrep {
        let old_cancel = self.reload.cancel.replace(cancel);
        self.reload.seq = self.reload.seq.saturating_add(1);
        self.reload.in_progress = true;
        self.clear_wanted();
        ReloadPrep {
            reload_seq: self.reload.seq,
            old_source: self.primary_asset().source_id(),
            old_cancel,
        }
    }

    pub(super) fn abort_reload(&mut self) {
        self.reload.in_progress = false;
        self.reload.cancel = None;
    }

    pub(super) fn matches_reload_seq(&self, seq: u64) -> bool {
        self.reload.seq == seq
    }

    pub(super) fn fail_reload(&mut self, error: String) {
        self.reload.in_progress = false;
        self.reload.cancel = None;
        self.last_error = Some(error);
    }

    pub(super) fn replace_reloaded(
        &mut self,
        path: PathBuf,
        source: Arc<dyn RasterSource>,
        frame: Frame,
        relaid: u64,
    ) -> SourceId {
        let primary = self.primary_asset();
        let old_src = primary.source_id();
        let old_override = primary.geo().crs_override().cloned();
        let old_epoch = primary.geo().crs_override_epoch();
        let saved_style = self.saved_style();
        let mut replacement = Self::from_source(
            self.id,
            path,
            source,
            frame,
            LayerInitOptions {
                crs_override: old_override,
                visible: self.visible,
                opacity: self.opacity,
                style: saved_style,
            },
            relaid,
        );
        replacement.pin_root = self.pin_root;
        replacement.reload.seq = self.reload.seq;
        if let LayerKind::SingleRaster { asset, .. } = &mut replacement.kind {
            asset.geo_mut().set_crs_override_epoch(old_epoch);
        }
        *self = replacement;
        old_src
    }

    pub(super) fn take_reload_cancel(&mut self) -> Option<CancelToken> {
        self.reload.cancel.take()
    }

    #[cfg(test)]
    pub(super) fn wanted(&self) -> &HashSet<TileKey> {
        self.primary_asset().stream().wanted()
    }

    #[cfg(test)]
    pub(super) fn wanted_shared(&self) -> Arc<Mutex<HashSet<TileKey>>> {
        self.primary_asset().stream().wanted_shared()
    }

    #[cfg(test)]
    pub(super) fn replace_wanted(&mut self, tiles: impl IntoIterator<Item = TileKey>) {
        match &mut self.kind {
            LayerKind::SingleRaster { asset, .. } => asset.stream_mut().replace_wanted(tiles),
            LayerKind::BandMath(bm) => bm.bands[0].stream_mut().replace_wanted(tiles),
        }
    }

    pub(super) fn clear_wanted(&mut self) {
        match &mut self.kind {
            LayerKind::SingleRaster { asset, .. } => {
                asset.clear_wanted();
            }
            LayerKind::BandMath(bm) => {
                for a in bm.assets_mut() {
                    a.clear_wanted();
                }
            }
        }
    }
}
