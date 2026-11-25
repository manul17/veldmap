//! Single georeferenced raster overlay (`OverlayLayer`), per-stream tile state
//! (`LayerStream`), reload lifecycle, and analytical style/stats transitions.

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
    BandStatsState, CategoricalStyle, ColorbarInfo, LayerBandOption, SavedLayerStyle,
    ScientificStyle, StretchPreset, TrackKind, build_band_options, format_band_label,
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

fn default_categorical_palette(has_color_map: bool) -> CategoricalPaletteId {
    if has_color_map {
        CategoricalPaletteId::SourceColorMap
    } else {
        CategoricalPaletteId::DistinctClasses
    }
}

/// Single georeferenced raster overlay inside [`super::stack::LayerStack`].
///
/// Invariant: `self.streams` is always non-empty (`self.streams[0]` is the
/// primary active stream).
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

        let custom_color_map = source.color_map();
        let mut scientific = ScientificStyle {
            last_scientific_band: (meta.spec.format == PixelFormat::R32Float)
                .then_some(meta.spec.band),
            ..ScientificStyle::default()
        };
        let mut categorical = CategoricalStyle {
            palette: default_categorical_palette(custom_color_map.is_some()),
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
            if let Some(pal) = saved.categorical_palette
                && (pal != CategoricalPaletteId::SourceColorMap || custom_color_map.is_some())
            {
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

    pub(super) fn id(&self) -> LayerId {
        self.id
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

    pub(super) fn pin_root(&self) -> bool {
        self.pin_root
    }

    #[cfg(test)]
    pub(super) fn set_pin_root(&mut self, pin_root: bool) {
        self.pin_root = pin_root;
    }

    #[cfg(test)]
    pub(super) fn push_aux_stream_for_test(&mut self, meta: PyramidMeta) {
        self.streams.push(LayerStream::new(meta));
    }

    pub(super) fn needs_band_stats(&self) -> Option<BandKey> {
        let spec = self.meta().spec;
        if self.is_active() && spec.format == PixelFormat::R32Float && self.scientific.needs_stats()
        {
            Some(spec.band)
        } else {
            None
        }
    }

    pub(super) fn begin_stats_job(&mut self, job_id: u64, cancel: CancelToken) {
        self.scientific.begin_loading(job_id, cancel);
    }

    pub(super) fn cancel_stats_job(&mut self) {
        self.scientific.cancel_loading();
    }

    pub(super) fn apply_band_stats_ready(&mut self, band: BandKey, stats: BandStats) -> bool {
        if self.meta().spec.band != band || self.meta().spec.format != PixelFormat::R32Float {
            return false;
        }
        self.scientific.apply_ready(stats);
        true
    }

    pub(super) fn apply_band_stats_empty(&mut self, band: BandKey, job_id: u64) -> bool {
        if self.meta().spec.band != band || self.meta().spec.format != PixelFormat::R32Float {
            return false;
        }
        self.scientific.apply_empty(job_id)
    }

    pub(super) fn apply_band_stats_cancelled(&mut self, band: BandKey, job_id: u64) -> bool {
        if self.meta().spec.band != band || self.meta().spec.format != PixelFormat::R32Float {
            return false;
        }
        self.scientific.apply_cancelled(job_id)
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
        self.cull = OverlayCullTree::new(desc.width, desc.height, meta.max_level());
        self.streams = vec![LayerStream::new(meta)];
        if spec.format == PixelFormat::R32Float {
            if self.scientific.last_scientific_band != Some(spec.band) {
                self.cancel_stats_job();
                self.scientific.stats_state = BandStatsState::NotRequested;
                self.scientific.custom_range_applied = false;
                if self.scientific.preset == StretchPreset::Custom {
                    self.scientific.preset = StretchPreset::Percentile2_98;
                }
                self.scientific.last_scientific_band = Some(spec.band);
            }
        } else {
            // Cancel any in-flight stats scan while away from Scientific,
            // preserving Ready stats/custom bounds if the user returns to the
            // same band.
            self.cancel_stats_job();
        }
        Ok(true)
    }

    pub(super) fn set_stretch_preset(&mut self, preset: StretchPreset) -> bool {
        if self.scientific.preset == preset
            && (preset == StretchPreset::Custom || !self.scientific.custom_range_applied)
        {
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
        let eps = lo.abs().max(hi.abs()).max(1e-30) * 1e-12;
        if (self.scientific.min - lo).abs() <= eps && (self.scientific.max - hi).abs() <= eps {
            return false;
        }
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
        if palette == CategoricalPaletteId::SourceColorMap && self.custom_color_map.is_none() {
            return false;
        }
        if self.categorical.palette == palette {
            return false;
        }
        self.categorical.palette = palette;
        true
    }

    fn categorical_nodata_value(&self, band: BandKey) -> Option<u32> {
        self.source
            .description()
            .band_info(band)
            .and_then(|b| b.nodata)
            .filter(|v| v.is_finite() && *v >= 0.0 && *v <= f64::from(u32::MAX))
            .map(|v| v.round() as u32)
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
            PixelFormat::R8Uint | PixelFormat::R16Uint => DrawOverlayTrack::Categorical {
                source: self.source_id(),
                palette: self.categorical.palette,
                custom_color_map: self.custom_color_map.clone(),
                nodata_value: self.categorical_nodata_value(spec.band),
            },
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

    pub(super) fn saved_style(&self) -> Option<SavedLayerStyle> {
        let spec = self.meta().spec;
        let default_cat = default_categorical_palette(self.custom_color_map.is_some());
        if spec == self.source.default_stream_spec()
            && spec.format == PixelFormat::Rgba8UnormSrgb
            && self.scientific.is_default_untouched()
            && self.categorical.palette == default_cat
        {
            return None;
        }
        let (band_index, track) = match (spec.band, TrackKind::from_format(spec.format)) {
            (_, TrackKind::Display) => (None, Some("display".to_owned())),
            (BandKey::Index(i), TrackKind::Scientific) => (Some(i), Some("scientific".to_owned())),
            (BandKey::Index(i), TrackKind::Categorical) => {
                (Some(i), Some("categorical".to_owned()))
            }
            _ => (None, None),
        };
        Some(SavedLayerStyle {
            band_index,
            track,
            stretch_preset: Some(self.scientific.preset),
            stretch_min: Some(self.scientific.min),
            stretch_max: Some(self.scientific.max),
            custom_stretch_range: self.scientific.custom_range_applied,
            colormap: Some(self.scientific.colormap),
            invert_colormap: self.scientific.invert_colormap,
            categorical_palette: Some(self.categorical.palette),
        })
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
                stats: self.scientific.stats().cloned(),
                stats_loading: self.scientific.is_loading(),
            }
        });
        let categorical = matches!(
            active_spec.format,
            PixelFormat::R8Uint | PixelFormat::R16Uint
        )
        .then(|| CategoricalSummary {
            palette: self.categorical.palette,
            has_source_color_map: self.custom_color_map.is_some(),
            nodata_value: self.categorical_nodata_value(active_spec.band),
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
                style: saved_style,
            },
            relaid,
        );
        replacement.pin_root = self.pin_root;
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

    /// Total observed bytes for one cell across all streams of this layer.
    pub(super) fn cell_observed_bytes(&self) -> usize {
        self.streams
            .iter()
            .fold(0usize, |acc, s| acc.saturating_add(s.observed_tile_bytes()))
            .max(1)
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
