//! Per-layer analytical rendering style (`ScientificStyle`, `CategoricalStyle`,
//! `StretchPreset`, `LayerStyleSpec`), band/track option descriptors, and
//! colorbar HUD projection.
//!
//! Kept strictly free of `serde` and `session.json` wire-format concerns;
//! persistence proxies live in `crate::storage::session`.

use std::sync::Arc;

use veldmap_core::sync::CancelToken;
use veldmap_core::tile::{BandKey, PixelFormat, TileStreamSpec};
use veldmap_raster::{
    BandInfo, BandStats, DISPLAY_STREAM_SPEC, RasterDescription, RasterSource, stream_spec_for_band,
};

use crate::colormap::{CategoricalPaletteId, ColormapId};
use crate::render::DrawOverlayTrack;

/// High-level track classification for a selectable raster stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TrackKind {
    Display,
    Scientific,
    Categorical,
}

impl TrackKind {
    pub(super) const fn from_format(format: PixelFormat) -> Self {
        match format {
            PixelFormat::Rgba8UnormSrgb => Self::Display,
            PixelFormat::R32Float => Self::Scientific,
            PixelFormat::R8Uint | PixelFormat::R16Uint => Self::Categorical,
        }
    }
}

/// Live contrast stretch mode for `R32Float` scientific layers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum StretchPreset {
    #[default]
    Percentile2_98,
    MinMax,
    Log10Db,
    Custom,
}

impl StretchPreset {
    pub(crate) const ALL: [Self; 4] = [
        Self::Percentile2_98,
        Self::MinMax,
        Self::Log10Db,
        Self::Custom,
    ];

    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Percentile2_98 => "Linear 2–98%",
            Self::MinMax => "Linear Min–Max",
            Self::Log10Db => "dB (10·log₁₀)",
            Self::Custom => "Custom Linear",
        }
    }

    pub(crate) const fn is_log10_db(self) -> bool {
        matches!(self, Self::Log10Db)
    }
}

/// Async lifecycle state for a layer's sampled physical [`BandStats`].
#[derive(Clone, Debug, Default)]
enum BandStatsState {
    #[default]
    NotRequested,
    Loading {
        job_id: u64,
        cancel: CancelToken,
    },
    Ready(Box<BandStats>),
    /// Stats job completed and confirmed no finite samples (or unsupported).
    Empty,
}

/// Live GPU stretch & colormap state for a scientific (`R32Float`) layer.
#[derive(Clone, Debug)]
pub(super) struct ScientificStyle {
    pub(super) preset: StretchPreset,
    pub(super) min: f64,
    pub(super) max: f64,
    pub(super) custom_range_applied: bool,
    pub(super) colormap: ColormapId,
    pub(super) invert_colormap: bool,
    pub(super) last_scientific_band: Option<BandKey>,
    stats_state: BandStatsState,
}

impl Default for ScientificStyle {
    fn default() -> Self {
        Self {
            preset: StretchPreset::Percentile2_98,
            min: 0.0,
            max: 1.0,
            custom_range_applied: false,
            colormap: ColormapId::Viridis,
            invert_colormap: false,
            last_scientific_band: None,
            stats_state: BandStatsState::NotRequested,
        }
    }
}

impl ScientificStyle {
    pub(super) fn stats(&self) -> Option<&BandStats> {
        match &self.stats_state {
            BandStatsState::Ready(stats) => Some(stats),
            _ => None,
        }
    }

    pub(super) fn is_loading(&self) -> bool {
        matches!(self.stats_state, BandStatsState::Loading { .. })
    }

    pub(super) fn needs_stats(&self) -> bool {
        matches!(self.stats_state, BandStatsState::NotRequested)
    }

    pub(super) fn is_default_untouched(&self) -> bool {
        self.preset == StretchPreset::Percentile2_98
            && !self.custom_range_applied
            && self.colormap == ColormapId::Viridis
            && !self.invert_colormap
    }

    pub(super) fn cancel_loading(&mut self) {
        if let BandStatsState::Loading { cancel, .. } = &self.stats_state {
            cancel.cancel();
            self.stats_state = BandStatsState::NotRequested;
        }
    }

    pub(super) fn begin_loading(&mut self, job_id: u64, cancel: CancelToken) {
        self.cancel_loading();
        self.stats_state = BandStatsState::Loading { job_id, cancel };
    }

    /// Apply newly computed [`BandStats`] to the stretch bounds unless the user
    /// already pinned a custom range, cancelling any superseded in-flight token.
    pub(super) fn apply_ready(&mut self, stats: BandStats) {
        if let BandStatsState::Loading { cancel, .. } = &self.stats_state {
            cancel.cancel();
        }
        self.stats_state = BandStatsState::Ready(Box::new(stats));
        if !self.custom_range_applied {
            self.recompute_bounds_from_stats();
        }
    }

    /// Mark stats empty only if `job_id` matches the currently active job.
    pub(super) fn apply_empty(&mut self, job_id: u64) -> bool {
        match &self.stats_state {
            BandStatsState::Loading { job_id: active, .. } if *active == job_id => {
                self.stats_state = BandStatsState::Empty;
                true
            }
            _ => false,
        }
    }

    /// Reset a cancelled loading job back to `NotRequested` without poisoning `Empty`.
    pub(super) fn apply_cancelled(&mut self, job_id: u64) -> bool {
        match &self.stats_state {
            BandStatsState::Loading { job_id: active, .. } if *active == job_id => {
                self.stats_state = BandStatsState::NotRequested;
                true
            }
            _ => false,
        }
    }

    /// Recompute `(min, max)` from cached `stats` for the current `preset`.
    fn recompute_bounds_from_stats(&mut self) {
        let Some(stats) = self.stats().cloned() else {
            return;
        };
        match self.preset {
            StretchPreset::Percentile2_98 => {
                self.min = stats.p2;
                self.max = stats.p98;
                self.custom_range_applied = false;
            }
            StretchPreset::MinMax => {
                self.min = stats.min;
                self.max = stats.max;
                self.custom_range_applied = false;
            }
            StretchPreset::Log10Db => {
                if let (Some(lo), Some(hi)) = (stats.db_p2, stats.db_p98) {
                    self.min = lo;
                    self.max = hi;
                } else {
                    self.min = -30.0;
                    self.max = 0.0;
                }
                self.custom_range_applied = false;
            }
            StretchPreset::Custom => {}
        }
    }

    pub(super) fn apply_saved(&mut self, saved: &LayerStyleSpec) {
        if let Some(preset) = saved.stretch_preset {
            self.preset = preset;
        }
        if let (Some(lo), Some(hi)) = (saved.stretch_min, saved.stretch_max)
            && lo.is_finite()
            && hi.is_finite()
            && hi > lo
        {
            self.min = lo;
            self.max = hi;
            if saved.custom_stretch_range || saved.stretch_preset == Some(StretchPreset::Custom) {
                self.custom_range_applied = true;
            }
        }
        if let Some(cm) = saved.colormap {
            self.colormap = cm;
        }
        self.invert_colormap = saved.invert_colormap;
    }

    pub(super) fn on_stream_switched(&mut self, spec: TileStreamSpec) {
        if spec.format == PixelFormat::R32Float {
            if self.last_scientific_band != Some(spec.band) {
                self.cancel_loading();
                self.stats_state = BandStatsState::NotRequested;
                self.custom_range_applied = false;
                if self.preset == StretchPreset::Custom {
                    self.preset = StretchPreset::Percentile2_98;
                }
                self.last_scientific_band = Some(spec.band);
            }
        } else {
            // Cancel any in-flight stats scan while away from Scientific,
            // preserving Ready stats/custom bounds if the user returns to the
            // same band.
            self.cancel_loading();
        }
    }

    pub(super) fn set_stretch_preset(&mut self, preset: StretchPreset) -> bool {
        if self.preset == preset && (preset == StretchPreset::Custom || !self.custom_range_applied)
        {
            return false;
        }
        self.preset = preset;
        if preset != StretchPreset::Custom {
            self.custom_range_applied = false;
            self.recompute_bounds_from_stats();
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
        if (self.min - lo).abs() <= eps && (self.max - hi).abs() <= eps {
            return false;
        }
        self.min = lo;
        self.max = hi;
        self.custom_range_applied = true;
        if !self.preset.is_log10_db() {
            self.preset = StretchPreset::Custom;
        }
        true
    }

    pub(super) fn set_colormap(&mut self, colormap: ColormapId, invert: bool) -> bool {
        if self.colormap == colormap && self.invert_colormap == invert {
            return false;
        }
        self.colormap = colormap;
        self.invert_colormap = invert;
        true
    }
}

/// Live categorical (`R8Uint`/`R16Uint`) palette state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct CategoricalStyle {
    pub(super) palette: CategoricalPaletteId,
}

fn default_categorical_palette(has_color_map: bool) -> CategoricalPaletteId {
    if has_color_map {
        CategoricalPaletteId::SourceColorMap
    } else {
        CategoricalPaletteId::DistinctClasses
    }
}

/// Cohesive per-layer analytical style state: selectable band options, optional
/// embedded TIFF ColorMap, live scientific stretch/colormap/stats state, and
/// categorical palette selection.
pub(super) struct LayerStyle {
    band_options: Vec<LayerBandOption>,
    custom_color_map: Option<Arc<[[u8; 4]]>>,
    scientific: ScientificStyle,
    categorical: CategoricalStyle,
}

impl LayerStyle {
    pub(super) fn new(
        source: &dyn RasterSource,
        active_spec: TileStreamSpec,
        saved: Option<&LayerStyleSpec>,
    ) -> Self {
        let band_options = build_band_options(source.description());
        let custom_color_map = source.color_map();
        let mut scientific = ScientificStyle {
            last_scientific_band: (active_spec.format == PixelFormat::R32Float)
                .then_some(active_spec.band),
            ..ScientificStyle::default()
        };
        let mut categorical = CategoricalStyle {
            palette: default_categorical_palette(custom_color_map.is_some()),
        };
        if let Some(saved) = saved {
            scientific.apply_saved(saved);
            if let Some(pal) = saved.categorical_palette
                && (pal != CategoricalPaletteId::SourceColorMap || custom_color_map.is_some())
            {
                categorical.palette = pal;
            }
        }
        Self {
            band_options,
            custom_color_map,
            scientific,
            categorical,
        }
    }

    pub(super) fn scientific(&self) -> &ScientificStyle {
        &self.scientific
    }

    pub(super) fn scientific_mut(&mut self) -> &mut ScientificStyle {
        &mut self.scientific
    }

    pub(super) fn categorical_palette(&self) -> CategoricalPaletteId {
        self.categorical.palette
    }

    pub(super) fn has_source_color_map(&self) -> bool {
        self.custom_color_map.is_some()
    }

    pub(super) fn band_options(&self) -> &[LayerBandOption] {
        &self.band_options
    }

    pub(super) fn on_stream_switched(&mut self, spec: TileStreamSpec) {
        self.scientific.on_stream_switched(spec);
    }

    pub(super) fn set_stretch_preset(&mut self, preset: StretchPreset) -> bool {
        self.scientific.set_stretch_preset(preset)
    }

    pub(super) fn set_stretch_range(&mut self, min: f64, max: f64) -> bool {
        self.scientific.set_stretch_range(min, max)
    }

    pub(super) fn set_colormap(&mut self, colormap: ColormapId, invert: bool) -> bool {
        self.scientific.set_colormap(colormap, invert)
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

    pub(super) fn categorical_nodata_value(desc: &RasterDescription, band: BandKey) -> Option<u32> {
        desc.band_info(band)
            .and_then(|b| b.nodata)
            .filter(|v| v.is_finite() && *v >= 0.0 && *v <= f64::from(u32::MAX))
            .map(|v| v.round() as u32)
    }

    pub(super) fn draw_track(
        &self,
        source: &dyn RasterSource,
        spec: TileStreamSpec,
    ) -> DrawOverlayTrack {
        let desc = source.description();
        let band_info = desc.band_info(spec.band);
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
                source: source.source_id(),
                palette: self.categorical.palette,
                custom_color_map: self.custom_color_map.clone(),
                nodata_value: Self::categorical_nodata_value(desc, spec.band),
            },
        }
    }

    pub(super) fn saved_style(
        &self,
        source: &dyn RasterSource,
        spec: TileStreamSpec,
    ) -> Option<LayerStyleSpec> {
        let default_cat = default_categorical_palette(self.custom_color_map.is_some());
        if spec == source.default_stream_spec()
            && spec.format == PixelFormat::Rgba8UnormSrgb
            && self.scientific.is_default_untouched()
            && self.categorical.palette == default_cat
        {
            return None;
        }
        let track_kind = TrackKind::from_format(spec.format);
        let (band_index, track) = match (spec.band, track_kind) {
            (_, TrackKind::Display) => (None, Some(track_kind)),
            (BandKey::Index(i), TrackKind::Scientific | TrackKind::Categorical) => {
                (Some(i), Some(track_kind))
            }
            _ => (None, None),
        };
        Some(LayerStyleSpec {
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
}

/// One selectable band/track entry in the Layer Card dropdown.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LayerBandOption {
    pub(crate) label: String,
    pub(crate) short_label: String,
    pub(crate) spec: TileStreamSpec,
}

/// Build the ordered list of selectable streams for `desc`.
pub(super) fn build_band_options(desc: &RasterDescription) -> Vec<LayerBandOption> {
    let mut opts = Vec::new();
    opts.push(LayerBandOption {
        label: "Display (RGB)".to_owned(),
        short_label: "Display".to_owned(),
        spec: DISPLAY_STREAM_SPEC,
    });
    for info in desc.indexed_bands() {
        let base_label = format_band_label(info);
        opts.push(LayerBandOption {
            label: format!("{base_label} · Scientific (f32)"),
            short_label: base_label.clone(),
            spec: stream_spec_for_band(info.band_key, PixelFormat::R32Float),
        });
        if let Some(cat_fmt) = info.sample_type.categorical_pixel_format() {
            let fmt_tag = match cat_fmt {
                PixelFormat::R8Uint => "u8",
                PixelFormat::R16Uint => "u16",
                _ => "uint",
            };
            opts.push(LayerBandOption {
                label: format!("{base_label} · Categorical ({fmt_tag})"),
                short_label: base_label,
                spec: stream_spec_for_band(info.band_key, cat_fmt),
            });
        }
    }
    opts
}

pub(super) fn format_band_label(info: &BandInfo) -> String {
    match &info.name {
        Some(name) if !name.is_empty() => format!("Band {} ({name})", info.index + 1),
        _ => format!("Band {}", info.index + 1),
    }
}

/// Viewport HUD colorbar snapshot for the topmost visible scientific layer.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ColorbarInfo {
    pub(crate) title: String,
    pub(crate) units: Option<String>,
    pub(crate) min: f64,
    pub(crate) max: f64,
    pub(crate) log10_db: bool,
    pub(crate) colormap: ColormapId,
    pub(crate) invert_colormap: bool,
}

/// Pure domain snapshot of a layer's analytical rendering style used for
/// session restore/persistence and in-memory reload preservation.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct LayerStyleSpec {
    /// Selected band index (`None` = Display composite).
    pub(crate) band_index: Option<u16>,
    /// Selected track (`Display`, `Scientific`, or `Categorical`).
    pub(crate) track: Option<TrackKind>,
    pub(crate) stretch_preset: Option<StretchPreset>,
    pub(crate) stretch_min: Option<f64>,
    pub(crate) stretch_max: Option<f64>,
    pub(crate) custom_stretch_range: bool,
    pub(crate) colormap: Option<ColormapId>,
    pub(crate) invert_colormap: bool,
    pub(crate) categorical_palette: Option<CategoricalPaletteId>,
}

impl LayerStyleSpec {
    /// Resolve a target [`TileStreamSpec`] against `desc`, if valid.
    pub(super) fn resolve_stream_spec(&self, desc: &RasterDescription) -> Option<TileStreamSpec> {
        let track = self.track?;
        match track {
            TrackKind::Display => Some(DISPLAY_STREAM_SPEC),
            TrackKind::Scientific => {
                let idx = self.band_index.unwrap_or(0);
                let band = BandKey::Index(idx);
                desc.band_info(band)?;
                Some(stream_spec_for_band(band, PixelFormat::R32Float))
            }
            TrackKind::Categorical => {
                let idx = self.band_index.unwrap_or(0);
                let band = BandKey::Index(idx);
                let info = desc.band_info(band)?;
                let fmt = info.sample_type.categorical_pixel_format()?;
                Some(stream_spec_for_band(band, fmt))
            }
        }
    }
}
