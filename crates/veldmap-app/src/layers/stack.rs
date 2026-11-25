//! Ordered [`LayerStack`] and multi-stream per-layer state (`OverlayLayer`,
//! `LayerStream`).
//!
//! `OverlayLayer` tracks a list of active [`LayerStream`]s (currently 1 primary
//! stream, ready for multi-band virtual math, DEM mosaic, and timeline layers),
//! summing per-stream VRAM tile weights in [`OverlayLayer::appetite_weight_bytes`]
//! and supporting opt-in `pin_root` control.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use veldmap_core::geo::{CullFrustum, FootprintBounds, Frame};
use veldmap_core::sync::{CancelToken, lock_unpoisoned};
use veldmap_core::tile::{
    BandKey, OverlayCullTree, PixelFormat, RASTER_TILE_SIZE, SourceId, StreamKey, TileKey,
    TileStreamSpec,
};
use veldmap_raster::{BandStats, PyramidMeta, RasterSource};

use crate::gpu_tiles::tile_format_bytes;
use crate::render::{CategoricalPaletteId, ColormapId, DrawOverlayTrack};
use crate::{clamp_opacity, display_name_from_path};

use super::style::{
    CategoricalStyle, ColorbarInfo, LayerBandOption, SavedLayerStyle, ScientificStyle,
    StretchPreset, TrackKind, build_band_options, format_band_label,
};
use super::{
    CategoricalSummary, CrsOverride, LayerId, LayerInitOptions, LayerSummary, OverlayPersistEntry,
    ScientificSummary,
};

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

/// One active tile stream inside an [`OverlayLayer`].
pub(super) struct LayerStream {
    meta: PyramidMeta,
    wanted_shared: Arc<Mutex<HashSet<TileKey>>>,
    wanted: HashSet<TileKey>,
    observed_tile_bytes: usize,
}

impl LayerStream {
    fn new(meta: PyramidMeta) -> Self {
        let initial_bytes = tile_format_bytes(RASTER_TILE_SIZE, meta.spec.format);
        Self {
            meta,
            wanted_shared: Arc::new(Mutex::new(HashSet::new())),
            wanted: HashSet::new(),
            observed_tile_bytes: initial_bytes,
        }
    }

    pub(super) fn meta(&self) -> &PyramidMeta {
        &self.meta
    }

    pub(super) fn stream_key(&self) -> StreamKey {
        self.meta.stream_key()
    }

    pub(super) fn observed_tile_bytes(&self) -> usize {
        self.observed_tile_bytes
    }

    pub(super) fn observe_tile_bytes(&mut self, bytes: usize) {
        self.observed_tile_bytes = self.observed_tile_bytes.max(bytes);
    }

    pub(super) fn wanted(&self) -> &HashSet<TileKey> {
        &self.wanted
    }

    pub(super) fn wanted_shared(&self) -> Arc<Mutex<HashSet<TileKey>>> {
        Arc::clone(&self.wanted_shared)
    }

    pub(super) fn replace_wanted(&mut self, tiles: impl IntoIterator<Item = TileKey>) {
        self.wanted.clear();
        self.wanted.extend(tiles);
        lock_unpoisoned(&self.wanted_shared).clone_from(&self.wanted);
    }

    pub(super) fn clear_wanted(&mut self) {
        self.wanted.clear();
        lock_unpoisoned(&self.wanted_shared).clear();
    }
}

/// Single georeferenced raster overlay inside [`LayerStack`].
pub(super) struct OverlayLayer {
    id: LayerId,
    name: String,
    path: PathBuf,
    source: Arc<dyn RasterSource>,
    streams: Vec<LayerStream>,
    band_options: Vec<LayerBandOption>,
    custom_color_map: Option<Arc<[[u8; 4]]>>,
    scientific: ScientificStyle,
    categorical: CategoricalStyle,
    stats_cancel: Option<CancelToken>,
    /// Opt-in flag controlling whether this layer's root tiles stay pinned in
    /// `PendingUploads` (defaults `true` for local files; future DEM mosaics
    /// and multi-temporal timelines opt out).
    pin_root: bool,
    fingerprint_hex: String,
    frame: Arc<Frame>,
    crs_override: Option<CrsOverride>,
    crs_override_epoch: u64,
    last_error: Option<String>,
    relaid: u64,
    cull: OverlayCullTree,
    visible: bool,
    opacity: f32,
    source_pixel_size_m: Option<(f64, f64)>,
    footprint: Option<FootprintBounds>,
    reload: ReloadState,
}

impl OverlayLayer {
    pub(super) fn from_source(
        id: LayerId,
        path: PathBuf,
        source: Arc<dyn RasterSource>,
        frame: Frame,
        options: LayerInitOptions,
        relaid: u64,
    ) -> Self {
        let name = display_name_from_path(&path);
        let desc = source.description();
        let band_options = build_band_options(desc);

        let preferred_spec = options
            .style
            .as_ref()
            .and_then(|s| s.resolve_stream_spec(desc))
            .unwrap_or_else(|| source.default_stream_spec());
        let meta = source
            .pyramid_meta_for(preferred_spec)
            .unwrap_or_else(|_| source.pyramid_meta());
        let max_level = meta.max_level();

        let custom_color_map: Option<Arc<[[u8; 4]]>> = source.color_map().map(Arc::from);
        let mut scientific = ScientificStyle::default();
        let mut categorical = CategoricalStyle {
            palette: if custom_color_map.is_some() {
                CategoricalPaletteId::SourceColorMap
            } else {
                CategoricalPaletteId::DistinctClasses
            },
        };
        if let Some(saved) = &options.style {
            if let Some(preset) = saved.stretch_preset {
                scientific.preset = preset;
            }
            if let (Some(lo), Some(hi)) = (saved.stretch_min, saved.stretch_max)
                && lo.is_finite()
                && hi.is_finite()
                && hi > lo
            {
                scientific.min = lo;
                scientific.max = hi;
                if saved.custom_stretch_range || saved.stretch_preset == Some(StretchPreset::Custom)
                {
                    scientific.custom_range_applied = true;
                }
            }
            if let Some(cm) = saved.colormap {
                scientific.colormap = cm;
            }
            scientific.invert_colormap = saved.invert_colormap;
            if let Some(pal) = saved.categorical_palette {
                categorical.palette = pal;
            }
        }

        let fingerprint_hex = source.fingerprint().to_hex();
        let source_pixel_size_m = frame.pixel_size_m(desc.width, desc.height);
        let footprint = frame.footprint_bounds();
        let cull = OverlayCullTree::new(desc.width, desc.height, max_level);

        Self {
            id,
            name,
            path,
            source,
            streams: vec![LayerStream::new(meta)],
            band_options,
            custom_color_map,
            scientific,
            categorical,
            stats_cancel: None,
            pin_root: true,
            fingerprint_hex,
            frame: Arc::new(frame),
            crs_override: options.crs_override,
            crs_override_epoch: 0,
            last_error: None,
            relaid,
            cull,
            visible: options.visible,
            opacity: clamp_opacity(options.opacity),
            source_pixel_size_m,
            footprint,
            reload: ReloadState::default(),
        }
    }

    pub(super) fn name(&self) -> &str {
        &self.name
    }

    pub(super) fn source(&self) -> &Arc<dyn RasterSource> {
        &self.source
    }

    /// Primary active stream (index 0).
    pub(super) fn primary_stream(&self) -> &LayerStream {
        &self.streams[0]
    }

    pub(super) fn primary_stream_mut(&mut self) -> &mut LayerStream {
        &mut self.streams[0]
    }

    pub(super) fn streams(&self) -> &[LayerStream] {
        &self.streams
    }

    pub(super) fn meta(&self) -> &PyramidMeta {
        self.primary_stream().meta()
    }

    pub(super) fn scientific_style(&self) -> &ScientificStyle {
        &self.scientific
    }

    pub(super) fn fingerprint_hex(&self) -> &str {
        &self.fingerprint_hex
    }

    pub(super) fn frame(&self) -> &Arc<Frame> {
        &self.frame
    }

    pub(super) fn crs_override(&self) -> Option<&CrsOverride> {
        self.crs_override.as_ref()
    }

    pub(super) fn crs_override_epoch(&self) -> u64 {
        self.crs_override_epoch
    }

    pub(super) fn relaid(&self) -> u64 {
        self.relaid
    }

    pub(super) fn opacity(&self) -> f32 {
        self.opacity
    }

    pub(super) fn source_pixel_size_m(&self) -> Option<(f64, f64)> {
        self.source_pixel_size_m
    }

    pub(super) fn footprint(&self) -> Option<FootprintBounds> {
        self.footprint
    }

    pub(super) fn observed_tile_bytes(&self) -> usize {
        self.primary_stream().observed_tile_bytes()
    }

    pub(super) fn pin_root(&self) -> bool {
        self.pin_root
    }

    pub(super) fn needs_band_stats(&self) -> Option<BandKey> {
        let spec = self.meta().spec;
        if spec.format == PixelFormat::R32Float
            && self.scientific.stats.is_none()
            && !self.scientific.stats_loading
        {
            Some(spec.band)
        } else {
            None
        }
    }

    pub(super) fn set_stats_loading(&mut self, cancel: CancelToken) {
        if let Some(prev) = self.stats_cancel.replace(cancel) {
            prev.cancel();
        }
        self.scientific.stats_loading = true;
    }

    pub(super) fn clear_stats_loading(&mut self) {
        if let Some(prev) = self.stats_cancel.take() {
            prev.cancel();
        }
        self.scientific.stats_loading = false;
    }

    pub(super) fn cancel_stats_job(&mut self) {
        if let Some(prev) = self.stats_cancel.take() {
            prev.cancel();
        }
        self.scientific.stats_loading = false;
    }

    pub(super) fn apply_band_stats(&mut self, band: BandKey, stats: Option<BandStats>) -> bool {
        if self.meta().spec.band != band || self.meta().spec.format != PixelFormat::R32Float {
            return false;
        }
        self.stats_cancel = None;
        self.scientific.apply_stats(stats);
        true
    }

    /// Switch this layer's active stream (`Display` / `Scientific` / `Categorical`
    /// + band index). Returns `true` if the active stream changed.
    pub(super) fn set_active_stream(&mut self, spec: TileStreamSpec) -> Result<bool, String> {
        if self.meta().spec == spec {
            return Ok(false);
        }
        let meta = self
            .source
            .pyramid_meta_for(spec)
            .map_err(|e| e.to_string())?;
        let desc = self.source.description();
        let prev_band = self.meta().spec.band;
        self.cull = OverlayCullTree::new(desc.width, desc.height, meta.max_level());
        self.streams = vec![LayerStream::new(meta)];
        if prev_band != spec.band {
            self.cancel_stats_job();
            self.scientific.stats = None;
            self.scientific.custom_range_applied = false;
        }
        Ok(true)
    }

    pub(super) fn set_stretch_preset(&mut self, preset: StretchPreset) -> bool {
        if self.scientific.preset == preset && !self.scientific.custom_range_applied {
            return false;
        }
        self.scientific.preset = preset;
        if preset != StretchPreset::Custom {
            self.scientific.custom_range_applied = false;
            self.scientific.recompute_bounds_from_stats();
        }
        true
    }

    pub(super) fn set_stretch_range(&mut self, min: f64, max: f64) -> bool {
        if !min.is_finite() || !max.is_finite() {
            return false;
        }
        let (lo, hi) = if max > min {
            (min, max)
        } else {
            (min, min + min.abs().max(1.0) * 1e-6)
        };
        self.scientific.min = lo;
        self.scientific.max = hi;
        self.scientific.custom_range_applied = true;
        if !self.scientific.preset.is_log10_db() {
            self.scientific.preset = StretchPreset::Custom;
        }
        true
    }

    pub(super) fn set_colormap(&mut self, colormap: ColormapId, invert: bool) -> bool {
        if self.scientific.colormap == colormap && self.scientific.invert_colormap == invert {
            return false;
        }
        self.scientific.colormap = colormap;
        self.scientific.invert_colormap = invert;
        true
    }

    pub(super) fn set_categorical_palette(&mut self, palette: CategoricalPaletteId) -> bool {
        if self.categorical.palette == palette {
            return false;
        }
        self.categorical.palette = palette;
        true
    }

    pub(super) fn draw_track(&self) -> DrawOverlayTrack {
        let spec = self.meta().spec;
        let band_info = self.source.description().band_info(spec.band);
        match spec.format {
            PixelFormat::Rgba8UnormSrgb => DrawOverlayTrack::Display,
            PixelFormat::R32Float => {
                let (scale, offset) = band_info.map_or((1.0, 0.0), |b| (b.scale, b.offset));
                DrawOverlayTrack::Scientific {
                    scale: scale as f32,
                    offset: offset as f32,
                    stretch_min: self.scientific.min as f32,
                    stretch_max: self.scientific.max as f32,
                    log10_db: self.scientific.preset.is_log10_db(),
                    colormap: self.scientific.colormap,
                    invert_colormap: self.scientific.invert_colormap,
                }
            }
            PixelFormat::R8Uint | PixelFormat::R16Uint => {
                let nodata_value = band_info
                    .and_then(|b| b.nodata)
                    .filter(|v| v.is_finite() && *v >= 0.0 && *v <= f64::from(u32::MAX))
                    .map(|v| v.round() as u32);
                DrawOverlayTrack::Categorical {
                    source: self.source_id(),
                    palette: self.categorical.palette,
                    custom_color_map: self.custom_color_map.clone(),
                    nodata_value,
                }
            }
        }
    }

    pub(super) fn colorbar_info(&self) -> Option<ColorbarInfo> {
        let spec = self.meta().spec;
        if !self.is_active() || spec.format != PixelFormat::R32Float {
            return None;
        }
        let info = self.source.description().band_info(spec.band);
        let band_title = info.map(format_band_label);
        let title = match band_title {
            Some(bt) => format!("{} · {bt}", self.name),
            None => self.name.clone(),
        };
        Some(ColorbarInfo {
            title,
            units: info.and_then(|b| b.units.clone()),
            min: self.scientific.min,
            max: self.scientific.max,
            log10_db: self.scientific.preset.is_log10_db(),
            colormap: self.scientific.colormap,
            invert_colormap: self.scientific.invert_colormap,
        })
    }

    pub(super) fn saved_style(&self) -> SavedLayerStyle {
        let spec = self.meta().spec;
        let (band_index, track) = match (spec.band, TrackKind::from_format(spec.format)) {
            (_, TrackKind::Display) => (None, Some("display".to_owned())),
            (BandKey::Index(i), TrackKind::Scientific) => (Some(i), Some("scientific".to_owned())),
            (BandKey::Index(i), TrackKind::Categorical) => {
                (Some(i), Some("categorical".to_owned()))
            }
            _ => (None, None),
        };
        SavedLayerStyle {
            band_index,
            track,
            stretch_preset: Some(self.scientific.preset),
            stretch_min: Some(self.scientific.min),
            stretch_max: Some(self.scientific.max),
            custom_stretch_range: self.scientific.custom_range_applied,
            colormap: Some(self.scientific.colormap),
            invert_colormap: self.scientific.invert_colormap,
            categorical_palette: Some(self.categorical.palette),
        }
    }

    pub(super) fn persist_entry(&self) -> OverlayPersistEntry<'_> {
        OverlayPersistEntry {
            path: &self.path,
            visible: self.visible,
            opacity: self.opacity,
            crs_override: self.crs_override.as_ref(),
            style: self.saved_style(),
        }
    }

    pub(super) fn visible_cells(&mut self, level: u8, frustum: &CullFrustum) -> Vec<(u32, u32)> {
        self.cull.visible_cells(&self.frame, level, frustum)
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

    pub(super) fn observe_stream_tile_bytes(&mut self, stream: StreamKey, bytes: usize) {
        for s in &mut self.streams {
            if s.stream_key() == stream {
                s.observe_tile_bytes(bytes);
            }
        }
    }

    pub(super) fn to_summary(
        &self,
        in_flight: usize,
        pass_active: bool,
        can_move_up: bool,
        can_move_down: bool,
    ) -> LayerSummary {
        let desc = self.source.description();
        let active_spec = self.meta().spec;
        let sample_info = desc.bands.first().map_or_else(
            || "Display RGBA".to_owned(),
            |b| {
                let nodata_str = b
                    .nodata
                    .map(|v| format!(", nodata={v}"))
                    .unwrap_or_default();
                format!("{} ch, {:?}{nodata_str}", b.channels, b.sample_type)
            },
        );
        let active_band_info = desc.band_info(active_spec.band);
        let scientific = (active_spec.format == PixelFormat::R32Float).then(|| {
            let (scale, offset) = active_band_info.map_or((1.0, 0.0), |b| (b.scale, b.offset));
            ScientificSummary {
                preset: self.scientific.preset,
                min: self.scientific.min,
                max: self.scientific.max,
                colormap: self.scientific.colormap,
                invert_colormap: self.scientific.invert_colormap,
                units: active_band_info.and_then(|b| b.units.clone()),
                scale,
                offset,
                stats: self.scientific.stats.clone(),
                stats_loading: self.scientific.stats_loading,
            }
        });
        let categorical = matches!(
            active_spec.format,
            PixelFormat::R8Uint | PixelFormat::R16Uint
        )
        .then(|| {
            let nodata_value = active_band_info
                .and_then(|b| b.nodata)
                .filter(|v| v.is_finite() && *v >= 0.0 && *v <= f64::from(u32::MAX))
                .map(|v| v.round() as u32);
            CategoricalSummary {
                palette: self.categorical.palette,
                has_source_color_map: self.custom_color_map.is_some(),
                nodata_value,
            }
        });

        LayerSummary {
            id: self.id,
            name: self.name.clone(),
            path: self.path.clone(),
            visible: self.visible,
            opacity: self.opacity,
            width: desc.width,
            height: desc.height,
            source_pixel_size_m: self.source_pixel_size_m,
            footprint: self.footprint,
            in_flight,
            pass_active,
            reloading: self.is_reloading(),
            crs_label: self.frame.crs_label(),
            datum_warning: self.frame.datum_warning().map(|w| w.message()),
            sample_info,
            detail_limit_level: self
                .meta()
                .finest_fitting_level()
                .unwrap_or(self.meta().max_level()),
            can_override_crs: desc.can_override_crs(),
            has_crs_override: self.crs_override.is_some(),
            last_error: self.last_error.clone(),
            can_move_up,
            can_move_down,
            active_spec,
            band_options: self.band_options.clone(),
            scientific,
            categorical,
        }
    }

    pub(super) fn apply_rebound_frame(
        &mut self,
        frame: Frame,
        crs_override: Option<CrsOverride>,
        relaid: u64,
    ) {
        let desc = self.source.description();
        self.source_pixel_size_m = frame.pixel_size_m(desc.width, desc.height);
        self.footprint = frame.footprint_bounds();
        self.frame = Arc::new(frame);
        self.crs_override = crs_override;
        self.crs_override_epoch = self.crs_override_epoch.saturating_add(1);
        self.last_error = None;
        self.relaid = relaid;
        self.cull = OverlayCullTree::new(desc.width, desc.height, self.meta().max_level());
    }

    pub(super) fn source_id(&self) -> SourceId {
        self.meta().source
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
            old_source: self.source_id(),
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
        let old_src = self.source_id();
        let saved_style = self.saved_style();
        let mut replacement = Self::from_source(
            self.id,
            path,
            source,
            frame,
            LayerInitOptions {
                crs_override: self.crs_override.clone(),
                visible: self.visible,
                opacity: self.opacity,
                style: Some(saved_style),
            },
            relaid,
        );
        replacement.reload.seq = self.reload.seq;
        replacement.crs_override_epoch = self.crs_override_epoch;
        *self = replacement;
        old_src
    }

    pub(super) fn take_reload_cancel(&mut self) -> Option<CancelToken> {
        self.reload.cancel.take()
    }

    /// Total proportional VRAM appetite weight across all active streams of
    /// this layer (summing heterogeneous multi-band / multi-format streams).
    pub(super) fn appetite_weight_bytes(&self) -> u64 {
        if !self.is_active() {
            return 0;
        }
        self.streams.iter().fold(0u64, |acc, s| {
            acc.saturating_add((s.observed_tile_bytes() as u64).max(1024))
        })
    }

    pub(super) fn wanted(&self) -> &HashSet<TileKey> {
        self.primary_stream().wanted()
    }

    pub(super) fn wanted_shared(&self) -> Arc<Mutex<HashSet<TileKey>>> {
        self.primary_stream().wanted_shared()
    }

    pub(super) fn replace_wanted(&mut self, tiles: impl IntoIterator<Item = TileKey>) {
        self.primary_stream_mut().replace_wanted(tiles);
    }

    pub(super) fn clear_wanted(&mut self) {
        for s in &mut self.streams {
            s.clear_wanted();
        }
    }

    pub(super) fn path(&self) -> &Path {
        &self.path
    }
}

/// Ordered collection of draped raster overlays in painter's order.
#[derive(Default)]
pub(super) struct LayerStack {
    layers: Vec<OverlayLayer>,
}

impl LayerStack {
    pub(super) fn push(&mut self, layer: OverlayLayer) {
        self.layers.push(layer);
    }

    pub(super) fn remove(&mut self, id: LayerId) -> Option<OverlayLayer> {
        let idx = self.layers.iter().position(|l| l.id == id)?;
        Some(self.layers.remove(idx))
    }

    pub(super) fn move_up(&mut self, id: LayerId) -> bool {
        let Some(idx) = self.layers.iter().position(|l| l.id == id) else {
            return false;
        };
        if idx + 1 >= self.layers.len() {
            return false;
        }
        self.layers.swap(idx, idx + 1);
        true
    }

    pub(super) fn move_down(&mut self, id: LayerId) -> bool {
        let Some(idx) = self.layers.iter().position(|l| l.id == id) else {
            return false;
        };
        if idx == 0 {
            return false;
        }
        self.layers.swap(idx - 1, idx);
        true
    }

    pub(super) fn get(&self, id: LayerId) -> Option<&OverlayLayer> {
        self.layers.iter().find(|l| l.id == id)
    }

    pub(super) fn get_mut(&mut self, id: LayerId) -> Option<&mut OverlayLayer> {
        self.layers.iter_mut().find(|l| l.id == id)
    }

    pub(super) fn layers(&self) -> &[OverlayLayer] {
        &self.layers
    }

    pub(super) fn layers_mut(&mut self) -> &mut [OverlayLayer] {
        &mut self.layers
    }

    pub(super) fn count_sharing_source(&self, source_id: SourceId) -> usize {
        self.layers
            .iter()
            .filter(|l| l.source_id() == source_id)
            .count()
    }

    pub(super) fn has_visible_source(&self, source_id: SourceId) -> bool {
        self.layers
            .iter()
            .any(|l| l.is_active() && l.source_id() == source_id)
    }

    pub(super) fn total_appetite_weight_bytes(&self) -> u64 {
        self.layers
            .iter()
            .fold(0u64, |acc, l| acc.saturating_add(l.appetite_weight_bytes()))
    }

    pub(super) fn topmost_visible_source_pixel_size_m(&self) -> Option<(f64, f64)> {
        self.layers
            .iter()
            .rev()
            .find(|l| l.is_active())
            .and_then(|l| l.source_pixel_size_m)
    }

    pub(super) fn topmost_colorbar_info(&self) -> Option<ColorbarInfo> {
        self.layers
            .iter()
            .rev()
            .find_map(OverlayLayer::colorbar_info)
    }

    pub(super) fn extend_pinned_root_sources(&self, out: &mut Vec<SourceId>) {
        for layer in &self.layers {
            if layer.is_active() && layer.pin_root() {
                let src = layer.source_id();
                if !out.contains(&src) {
                    out.push(src);
                }
            }
        }
    }

    pub(super) fn extend_wanted_tile_sets<'a>(&'a self, out: &mut Vec<&'a HashSet<TileKey>>) {
        for layer in &self.layers {
            if layer.is_active() {
                for stream in layer.streams() {
                    if !stream.wanted().is_empty() {
                        out.push(stream.wanted());
                    }
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "stack_tests.rs"]
mod tests;
