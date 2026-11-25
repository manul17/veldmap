//! Per-layer analytical rendering style (`ScientificStyle`, `CategoricalStyle`,
//! `StretchPreset`), band/track option descriptors, colorbar HUD projection,
//! and optional `session.json` persistence snapshots.

use serde::{Deserialize, Serialize};
use veldmap_core::tile::{BandKey, PixelFormat, TileStreamSpec};
use veldmap_raster::{
    BandInfo, BandStats, DISPLAY_STREAM_SPEC, RasterDescription, stream_spec_for_band,
};

use crate::render::{CategoricalPaletteId, ColormapId};

/// High-level track classification for a selectable raster stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TrackKind {
    Display,
    Scientific,
    Categorical,
}

impl TrackKind {
    pub(crate) const fn from_format(format: PixelFormat) -> Self {
        match format {
            PixelFormat::Rgba8UnormSrgb => Self::Display,
            PixelFormat::R32Float => Self::Scientific,
            PixelFormat::R8Uint | PixelFormat::R16Uint => Self::Categorical,
        }
    }
}

/// Live contrast stretch mode for `R32Float` scientific layers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
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

/// Live GPU stretch & colormap state for a scientific (`R32Float`) layer.
#[derive(Clone, Debug)]
pub(crate) struct ScientificStyle {
    pub(crate) preset: StretchPreset,
    pub(crate) min: f64,
    pub(crate) max: f64,
    pub(crate) custom_range_applied: bool,
    pub(crate) colormap: ColormapId,
    pub(crate) invert_colormap: bool,
    pub(crate) stats: Option<BandStats>,
    pub(crate) stats_loading: bool,
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
            stats: None,
            stats_loading: false,
        }
    }
}

impl ScientificStyle {
    /// Apply newly computed [`BandStats`] to the stretch bounds unless the user
    /// already pinned a custom range.
    pub(crate) fn apply_stats(&mut self, stats: Option<BandStats>) {
        self.stats_loading = false;
        self.stats = stats;
        if !self.custom_range_applied {
            self.recompute_bounds_from_stats();
        }
    }

    /// Recompute `(min, max)` from cached `stats` for the current `preset`.
    pub(crate) fn recompute_bounds_from_stats(&mut self) {
        let Some(ref stats) = self.stats else {
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
}

/// Live categorical (`R8Uint`/`R16Uint`) palette state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct CategoricalStyle {
    pub(crate) palette: CategoricalPaletteId,
}

/// One selectable band/track entry in the Layer Card dropdown.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LayerBandOption {
    pub(crate) label: String,
    pub(crate) spec: TileStreamSpec,
    pub(crate) track_kind: TrackKind,
}

/// Build the ordered list of selectable streams for `desc`.
pub(crate) fn build_band_options(desc: &RasterDescription) -> Vec<LayerBandOption> {
    let mut opts = Vec::new();
    opts.push(LayerBandOption {
        label: "Display (RGB)".to_owned(),
        spec: DISPLAY_STREAM_SPEC,
        track_kind: TrackKind::Display,
    });
    for info in desc.indexed_bands() {
        let base_label = format_band_label(info);
        opts.push(LayerBandOption {
            label: format!("{base_label} · Scientific (f32)"),
            spec: stream_spec_for_band(info.band_key, PixelFormat::R32Float),
            track_kind: TrackKind::Scientific,
        });
        if let Some(cat_fmt) = info.sample_type.categorical_pixel_format() {
            let fmt_tag = match cat_fmt {
                PixelFormat::R8Uint => "u8",
                PixelFormat::R16Uint => "u16",
                _ => "uint",
            };
            opts.push(LayerBandOption {
                label: format!("{base_label} · Categorical ({fmt_tag})"),
                spec: stream_spec_for_band(info.band_key, cat_fmt),
                track_kind: TrackKind::Categorical,
            });
        }
    }
    opts
}

pub(crate) fn format_band_label(info: &BandInfo) -> String {
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

/// Optional persisted analytical rendering style stored inside `session.json`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct SavedLayerStyle {
    /// Selected band index (`None` = Display composite).
    pub(crate) band_index: Option<u16>,
    /// Selected track (`"display"`, `"scientific"`, or `"categorical"`).
    pub(crate) track: Option<String>,
    pub(crate) stretch_preset: Option<StretchPreset>,
    pub(crate) stretch_min: Option<f64>,
    pub(crate) stretch_max: Option<f64>,
    #[serde(default)]
    pub(crate) custom_stretch_range: bool,
    pub(crate) colormap: Option<ColormapId>,
    pub(crate) invert_colormap: bool,
    pub(crate) categorical_palette: Option<CategoricalPaletteId>,
}

impl SavedLayerStyle {
    /// Resolve a persisted [`TileStreamSpec`] against `desc`, if valid.
    pub(crate) fn resolve_stream_spec(&self, desc: &RasterDescription) -> Option<TileStreamSpec> {
        let track = self.track.as_deref()?;
        match track {
            "display" => Some(DISPLAY_STREAM_SPEC),
            "scientific" => {
                let idx = self.band_index.unwrap_or(0);
                let band = BandKey::Index(idx);
                desc.band_info(band)?;
                Some(stream_spec_for_band(band, PixelFormat::R32Float))
            }
            "categorical" => {
                let idx = self.band_index.unwrap_or(0);
                let band = BandKey::Index(idx);
                let info = desc.band_info(band)?;
                let fmt = info.sample_type.categorical_pixel_format()?;
                Some(stream_spec_for_band(band, fmt))
            }
            _ => None,
        }
    }
}
