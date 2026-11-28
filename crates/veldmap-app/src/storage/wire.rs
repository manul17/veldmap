//! Forward-compatible serde wire proxies for `session.json` and `.veldmap.json`
//! layer styles, CRS overrides, external band metadata, and `BandMath` recipes.
//!
//! Kept private to `crate::storage` so internal wire enums never leak outside
//! persistence boundaries.

use serde::{Deserialize, Deserializer, Serialize};
use veldmap_core::geo::CustomTmParams;
use veldmap_raster::ExternalBandMeta;

use crate::band_math::{BandMathFormula, clamp_rgb_gamma};
use crate::colormap::{CategoricalPaletteId, ColormapId};
use crate::layers::{
    BandMathPreset, BandMathStyleSpec, CrsOverride, LayerStyleSpec, StacCogAuthMode, StretchPreset,
    TrackKind,
};

/// Forward-compatible `Option<T>` deserializer for string-encoded style enums:
/// unknown variants written by newer versions fall back to `None` instead of
/// failing the entire `session.json` load into `.corrupt`.
pub(super) fn deserialize_lossy_opt<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: for<'a> Deserialize<'a>,
{
    let raw = Option::<String>::deserialize(deserializer)?;
    Ok(raw.and_then(|s| {
        let de = serde::de::value::StrDeserializer::<serde::de::value::Error>::new(&s);
        T::deserialize(de).ok()
    }))
}

macro_rules! wire_enum {
    ($wire:ident <-> $domain:ident { $($variant:ident),+ $(,)? }) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
        #[serde(rename_all = "snake_case")]
        pub(crate) enum $wire {
            $($variant,)+
        }

        impl From<$domain> for $wire {
            fn from(value: $domain) -> Self {
                match value {
                    $($domain::$variant => Self::$variant,)+
                }
            }
        }

        impl From<$wire> for $domain {
            fn from(value: $wire) -> Self {
                match value {
                    $($wire::$variant => Self::$variant,)+
                }
            }
        }
    };
}

wire_enum!(WireTrackKind <-> TrackKind {
    Display,
    Scientific,
    Categorical,
});

wire_enum!(WireStretchPreset <-> StretchPreset {
    Percentile2_98,
    MinMax,
    Log10Db,
    Custom,
});

wire_enum!(WireColormapId <-> ColormapId {
    Grayscale,
    Viridis,
    Inferno,
    Plasma,
    Turbo,
    RdBu,
    RdYlGn,
    BrBG,
    Terrain,
});

wire_enum!(WireCategoricalPaletteId <-> CategoricalPaletteId {
    SourceColorMap,
    Sentinel2Scl,
    DistinctClasses,
});

wire_enum!(WireStacCogAuthMode <-> StacCogAuthMode {
    None,
    FromSearchResult,
    DirectCustom,
});

wire_enum!(WireBandMathFormula <-> BandMathFormula {
    NormalizedDiff,
    Difference,
    Ratio,
    RgbComposite,
});

wire_enum!(WireBandMathPreset <-> BandMathPreset {
    TrueColor,
    Ndvi,
    Ndwi,
    Nbr,
    Ndsi,
    FalseColorNir,
    Swir,
    CustomNormalizedDiff,
    CustomDifference,
    CustomRatio,
    CustomRgb,
});

/// Forward-compatible `session.json` wire proxy for [`LayerStyleSpec`].
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub(super) struct SavedLayerStyle {
    band_index: Option<u16>,
    #[serde(deserialize_with = "deserialize_lossy_opt")]
    track: Option<WireTrackKind>,
    #[serde(deserialize_with = "deserialize_lossy_opt")]
    stretch_preset: Option<WireStretchPreset>,
    stretch_min: Option<f64>,
    stretch_max: Option<f64>,
    #[serde(default)]
    custom_stretch_range: bool,
    #[serde(deserialize_with = "deserialize_lossy_opt")]
    colormap: Option<WireColormapId>,
    invert_colormap: bool,
    #[serde(deserialize_with = "deserialize_lossy_opt")]
    categorical_palette: Option<WireCategoricalPaletteId>,
}

impl From<&LayerStyleSpec> for SavedLayerStyle {
    fn from(spec: &LayerStyleSpec) -> Self {
        Self {
            band_index: spec.band_index,
            track: spec.track.map(WireTrackKind::from),
            stretch_preset: spec.stretch_preset.map(WireStretchPreset::from),
            stretch_min: spec.stretch_min,
            stretch_max: spec.stretch_max,
            custom_stretch_range: spec.custom_stretch_range,
            colormap: spec.colormap.map(WireColormapId::from),
            invert_colormap: spec.invert_colormap,
            categorical_palette: spec.categorical_palette.map(WireCategoricalPaletteId::from),
        }
    }
}

impl From<&SavedLayerStyle> for LayerStyleSpec {
    fn from(saved: &SavedLayerStyle) -> Self {
        Self {
            band_index: saved.band_index,
            track: saved.track.map(TrackKind::from),
            stretch_preset: saved.stretch_preset.map(StretchPreset::from),
            stretch_min: saved.stretch_min,
            stretch_max: saved.stretch_max,
            custom_stretch_range: saved.custom_stretch_range,
            colormap: saved.colormap.map(ColormapId::from),
            invert_colormap: saved.invert_colormap,
            categorical_palette: saved.categorical_palette.map(CategoricalPaletteId::from),
        }
    }
}

/// Persisted per-band radiometric metadata (`scale`, `offset`, `nodata`, `unit`).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct SavedExternalBandMeta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) scale: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) offset: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) nodata: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) unit: Option<String>,
}

impl From<&ExternalBandMeta> for SavedExternalBandMeta {
    fn from(m: &ExternalBandMeta) -> Self {
        Self {
            scale: m.scale.filter(|v| v.is_finite()),
            offset: m.offset.filter(|v| v.is_finite()),
            nodata: m.nodata.filter(|v| v.is_finite()),
            unit: m.unit.clone(),
        }
    }
}

impl From<&SavedExternalBandMeta> for ExternalBandMeta {
    fn from(m: &SavedExternalBandMeta) -> Self {
        Self {
            scale: m.scale.filter(|v| v.is_finite()),
            offset: m.offset.filter(|v| v.is_finite()),
            nodata: m.nodata.filter(|v| v.is_finite()),
            unit: m.unit.clone(),
        }
    }
}

/// Minimal serde proxy preserving the exact flat `session.json` wire format
/// (`{"kind": "...", ...}`) for [`CrsOverride`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum SavedCrsOverride {
    Epsg { code: u32 },
    Wkt { wkt: String },
    Proj { proj: String },
    CustomTm(CustomTmParams),
}

impl From<&CrsOverride> for SavedCrsOverride {
    fn from(ov: &CrsOverride) -> Self {
        match ov {
            CrsOverride::Epsg(code) => Self::Epsg { code: *code },
            CrsOverride::Wkt(wkt) => Self::Wkt { wkt: wkt.clone() },
            CrsOverride::Proj(proj) => Self::Proj { proj: proj.clone() },
            CrsOverride::CustomTm(params) => Self::CustomTm(params.clone()),
        }
    }
}

impl From<&SavedCrsOverride> for CrsOverride {
    fn from(saved: &SavedCrsOverride) -> Self {
        match saved {
            SavedCrsOverride::Epsg { code } => Self::Epsg(*code),
            SavedCrsOverride::Wkt { wkt } => Self::Wkt(wkt.clone()),
            SavedCrsOverride::Proj { proj } => Self::Proj(proj.clone()),
            SavedCrsOverride::CustomTm(params) => Self::CustomTm(params.clone()),
        }
    }
}

/// Persisted style payload for a `BandMath` layer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct SavedBandMathStyle {
    #[serde(deserialize_with = "deserialize_lossy_opt")]
    preset: Option<WireBandMathPreset>,
    #[serde(deserialize_with = "deserialize_lossy_opt")]
    formula: Option<WireBandMathFormula>,
    recipe_label: String,
    #[serde(deserialize_with = "deserialize_lossy_opt")]
    colormap: Option<WireColormapId>,
    invert_colormap: bool,
    stretch_min: f64,
    stretch_max: f64,
    rgb_min: [f64; 3],
    rgb_max: [f64; 3],
    default_rgb_min: [f64; 3],
    default_rgb_max: [f64; 3],
    rgb_gamma: f32,
    scl_mask_enabled: bool,
    scl_mask_bits: u32,
}

impl Default for SavedBandMathStyle {
    fn default() -> Self {
        Self {
            preset: Some(WireBandMathPreset::Ndvi),
            formula: Some(WireBandMathFormula::NormalizedDiff),
            recipe_label: "NDVI".to_owned(),
            colormap: Some(WireColormapId::RdYlGn),
            invert_colormap: false,
            stretch_min: -0.2,
            stretch_max: 0.85,
            rgb_min: [0.0; 3],
            rgb_max: [0.35; 3],
            default_rgb_min: [0.0; 3],
            default_rgb_max: [0.35; 3],
            rgb_gamma: 1.0,
            scl_mask_enabled: false,
            scl_mask_bits: 0,
        }
    }
}

impl From<&BandMathStyleSpec> for SavedBandMathStyle {
    fn from(spec: &BandMathStyleSpec) -> Self {
        Self {
            preset: Some(WireBandMathPreset::from(spec.preset)),
            formula: Some(WireBandMathFormula::from(spec.formula)),
            recipe_label: spec.recipe_label.clone(),
            colormap: Some(WireColormapId::from(spec.colormap)),
            invert_colormap: spec.invert_colormap,
            stretch_min: spec.stretch_min,
            stretch_max: spec.stretch_max,
            rgb_min: spec.rgb_min,
            rgb_max: spec.rgb_max,
            default_rgb_min: spec.default_rgb_range.0,
            default_rgb_max: spec.default_rgb_range.1,
            rgb_gamma: spec.rgb_gamma,
            scl_mask_enabled: spec.scl_mask_enabled,
            scl_mask_bits: spec.scl_mask_bits,
        }
    }
}

impl SavedBandMathStyle {
    pub(super) fn sanitize(&mut self) {
        let defaults = Self::default();
        if !self.stretch_min.is_finite() {
            self.stretch_min = defaults.stretch_min;
        }
        if !self.stretch_max.is_finite() {
            self.stretch_max = defaults.stretch_max;
        }
        for i in 0..3 {
            if !self.rgb_min[i].is_finite() {
                self.rgb_min[i] = defaults.rgb_min[i];
            }
            if !self.rgb_max[i].is_finite() {
                self.rgb_max[i] = defaults.rgb_max[i];
            }
            if !self.default_rgb_min[i].is_finite() {
                self.default_rgb_min[i] = defaults.default_rgb_min[i];
            }
            if !self.default_rgb_max[i].is_finite() {
                self.default_rgb_max[i] = defaults.default_rgb_max[i];
            }
        }
        self.rgb_gamma = clamp_rgb_gamma(self.rgb_gamma);
    }

    pub(crate) fn to_spec(&self) -> BandMathStyleSpec {
        let preset = self
            .preset
            .map_or(BandMathPreset::CustomNormalizedDiff, BandMathPreset::from);
        let formula = self
            .formula
            .map_or_else(|| preset.formula(), BandMathFormula::from);
        let colormap = self
            .colormap
            .map_or_else(|| preset.default_colormap(), ColormapId::from);
        BandMathStyleSpec {
            preset,
            formula,
            recipe_label: self.recipe_label.clone(),
            colormap,
            invert_colormap: self.invert_colormap,
            stretch_min: self.stretch_min,
            stretch_max: self.stretch_max,
            rgb_min: self.rgb_min,
            rgb_max: self.rgb_max,
            default_rgb_range: (self.default_rgb_min, self.default_rgb_max),
            rgb_gamma: self.rgb_gamma,
            scl_mask_enabled: self.scl_mask_enabled,
            scl_mask_bits: self.scl_mask_bits,
        }
    }
}
