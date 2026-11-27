//! Domain model for multi-band analytical Band Math & RGB Composite layers
//! (`BandMathFormula`, `BandMathPreset`, `BandMathStyle`, `BandMathLayerData`),
//! shared between GPU scratch/composite rendering and zoom-invariant CPU cursor
//! readouts so physical-space formulas never diverge.

use crate::colormap::{ColormapId, DEFAULT_SCL_MASK_BITS};

use super::asset::RasterAsset;

/// Epsilon guarding division-by-zero in `(A - B) / (A + B)` and `A / B`
/// across both GPU (`band_math_composite.wgsl`) and CPU (`CursorProber`).
pub(crate) const BAND_MATH_DIV_EPS: f64 = 1e-6;

/// Mathematical band-combination mode evaluated in canonical physical
/// `raw * scale + offset` space.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BandMathFormula {
    /// Normalized difference `(A - B) / (A + B)` (NDVI, NDWI, NBR, NDSI).
    NormalizedDiff,
    /// Physical difference `A - B`.
    Difference,
    /// Physical ratio `A / B`.
    Ratio,
    /// 3-channel physical RGB composite `(R, G, B)`.
    RgbComposite,
}

impl BandMathFormula {
    pub(crate) const ALL: [Self; 4] = [
        Self::NormalizedDiff,
        Self::Difference,
        Self::Ratio,
        Self::RgbComposite,
    ];

    /// Number of float input bands required by this formula (`2` for scalar
    /// modes, `3` for RGB composite).
    pub(crate) const fn required_bands(self) -> usize {
        match self {
            Self::NormalizedDiff | Self::Difference | Self::Ratio => 2,
            Self::RgbComposite => 3,
        }
    }

    /// Numeric shader mode ID matching `band_math_composite.wgsl`.
    pub(crate) const fn shader_mode(self) -> u32 {
        match self {
            Self::NormalizedDiff => 0,
            Self::Difference => 1,
            Self::Ratio => 2,
            Self::RgbComposite => 3,
        }
    }

    /// Return whether this formula produces a 1-channel scalar output styled
    /// via colormap LUT (as opposed to a 3-channel RGB composite).
    pub(crate) const fn is_scalar(self) -> bool {
        !matches!(self, Self::RgbComposite)
    }

    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::NormalizedDiff => "Normalized Diff (A−B)/(A+B)",
            Self::Difference => "Difference (A − B)",
            Self::Ratio => "Ratio (A / B)",
            Self::RgbComposite => "RGB Composite (R, G, B)",
        }
    }

    pub(crate) const fn short_label(self) -> &'static str {
        match self {
            Self::NormalizedDiff => "Index",
            Self::Difference => "A − B",
            Self::Ratio => "A / B",
            Self::RgbComposite => "RGB",
        }
    }

    /// Evaluate a scalar formula on two physical `f64` values, preserving valid
    /// `0.0` inputs while returning `None` on non-finite values or near-zero
    /// denominators.
    pub(crate) fn evaluate_scalar(self, a: f64, b: f64) -> Option<f64> {
        if !a.is_finite() || !b.is_finite() {
            return None;
        }
        let val = match self {
            Self::NormalizedDiff => {
                let denom = a + b;
                if denom.abs() < BAND_MATH_DIV_EPS {
                    return None;
                }
                (a - b) / denom
            }
            Self::Difference => a - b,
            Self::Ratio => {
                if b.abs() < BAND_MATH_DIV_EPS {
                    return None;
                }
                a / b
            }
            Self::RgbComposite => return None,
        };
        val.is_finite().then_some(val)
    }
}

/// Curated EO scene recipe presets driving default formulas, stretch ranges,
/// colormaps, and display titles.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum BandMathPreset {
    TrueColor,
    Ndvi,
    Ndwi,
    FalseColorNir,
    Swir,
    CustomNormalizedDiff,
    CustomDifference,
    CustomRatio,
    CustomRgb,
}

impl BandMathPreset {
    pub(crate) const fn formula(self) -> BandMathFormula {
        match self {
            Self::TrueColor | Self::FalseColorNir | Self::Swir | Self::CustomRgb => {
                BandMathFormula::RgbComposite
            }
            Self::Ndvi | Self::Ndwi | Self::CustomNormalizedDiff => BandMathFormula::NormalizedDiff,
            Self::CustomDifference => BandMathFormula::Difference,
            Self::CustomRatio => BandMathFormula::Ratio,
        }
    }

    pub(crate) const fn short_label(self) -> &'static str {
        match self {
            Self::TrueColor => "True Color",
            Self::Ndvi => "NDVI",
            Self::Ndwi => "NDWI",
            Self::FalseColorNir => "False Color NIR",
            Self::Swir => "SWIR",
            Self::CustomNormalizedDiff => "Norm. Diff",
            Self::CustomDifference => "Diff",
            Self::CustomRatio => "Ratio",
            Self::CustomRgb => "RGB",
        }
    }

    pub(crate) const fn default_colormap(self) -> ColormapId {
        match self {
            Self::Ndvi | Self::CustomNormalizedDiff => ColormapId::RdYlGn,
            Self::Ndwi => ColormapId::BrBG,
            Self::CustomDifference => ColormapId::RdBu,
            _ => ColormapId::Viridis,
        }
    }

    pub(crate) const fn default_scalar_range(self) -> (f64, f64) {
        match self {
            Self::Ndvi => (-0.2, 0.9),
            Self::Ndwi => (-0.5, 0.5),
            Self::CustomNormalizedDiff => (-1.0, 1.0),
            Self::CustomDifference => (-0.5, 0.5),
            Self::CustomRatio => (0.0, 3.0),
            Self::TrueColor | Self::FalseColorNir | Self::Swir | Self::CustomRgb => (0.0, 0.35),
        }
    }

    pub(crate) const fn default_rgb_range(self) -> ([f64; 3], [f64; 3]) {
        match self {
            Self::TrueColor => ([0.0, 0.0, 0.0], [0.30, 0.30, 0.30]),
            Self::FalseColorNir | Self::Swir | Self::CustomRgb => {
                ([0.0, 0.0, 0.0], [0.35, 0.35, 0.35])
            }
            _ => ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
        }
    }

    pub(crate) const fn from_sample_kind(kind: veldmap_net::SampleRecipeKind) -> Self {
        match kind {
            veldmap_net::SampleRecipeKind::TrueColor => Self::TrueColor,
            veldmap_net::SampleRecipeKind::Ndvi => Self::Ndvi,
            veldmap_net::SampleRecipeKind::Ndwi => Self::Ndwi,
            veldmap_net::SampleRecipeKind::FalseColorNir => Self::FalseColorNir,
            veldmap_net::SampleRecipeKind::Swir => Self::Swir,
        }
    }
}

/// Live GPU styling, stretch, colormap, and `SCL` mask configuration for a
/// [`BandMathLayerData`] layer.
#[derive(Clone, Debug)]
pub(super) struct BandMathStyle {
    pub(super) preset: BandMathPreset,
    pub(super) formula: BandMathFormula,
    pub(super) recipe_label: String,
    pub(super) colormap: ColormapId,
    pub(super) invert_colormap: bool,
    pub(super) stretch_min: f64,
    pub(super) stretch_max: f64,
    pub(super) rgb_min: [f64; 3],
    pub(super) rgb_max: [f64; 3],
    default_rgb_range: ([f64; 3], [f64; 3]),
    pub(super) rgb_gamma: f32,
    pub(super) scl_mask_enabled: bool,
    pub(super) scl_mask_bits: u32,
    pub(super) style_rev: u64,
}

impl BandMathStyle {
    pub(super) fn from_preset(preset: BandMathPreset, has_scl: bool) -> Self {
        let (stretch_min, stretch_max) = preset.default_scalar_range();
        let (rgb_min, rgb_max) = preset.default_rgb_range();
        Self {
            preset,
            formula: preset.formula(),
            recipe_label: preset.short_label().to_owned(),
            colormap: preset.default_colormap(),
            invert_colormap: false,
            stretch_min,
            stretch_max,
            rgb_min,
            rgb_max,
            default_rgb_range: (rgb_min, rgb_max),
            rgb_gamma: 1.0,
            scl_mask_enabled: has_scl,
            scl_mask_bits: DEFAULT_SCL_MASK_BITS,
            style_rev: 1,
        }
    }

    /// Construct a custom local band-math style with automatic RGB range
    /// adapted to integer sample scales (e.g. `0..255` for unscaled 8-bit
    /// orthophotos vs `0.0..0.35` for physical reflectance).
    pub(super) fn for_custom(
        formula: BandMathFormula,
        has_scl: bool,
        rgb_max_hint: Option<f64>,
    ) -> Self {
        let preset = match formula {
            BandMathFormula::NormalizedDiff => BandMathPreset::CustomNormalizedDiff,
            BandMathFormula::Difference => BandMathPreset::CustomDifference,
            BandMathFormula::Ratio => BandMathPreset::CustomRatio,
            BandMathFormula::RgbComposite => BandMathPreset::CustomRgb,
        };
        let mut style = Self::from_preset(preset, has_scl);
        style.formula = formula;
        style.recipe_label = match formula {
            BandMathFormula::NormalizedDiff => "Norm. Diff".to_owned(),
            BandMathFormula::Difference => "Difference".to_owned(),
            BandMathFormula::Ratio => "Ratio".to_owned(),
            BandMathFormula::RgbComposite => "RGB Composite".to_owned(),
        };
        if formula == BandMathFormula::RgbComposite
            && let Some(hi) = rgb_max_hint
            && hi.is_finite()
            && hi > 0.0
        {
            style.rgb_min = [0.0; 3];
            style.rgb_max = [hi; 3];
            style.default_rgb_range = (style.rgb_min, style.rgb_max);
        }
        style
    }

    fn bump_rev(&mut self) {
        self.style_rev = self.style_rev.wrapping_add(1);
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
        if (self.stretch_min - lo).abs() <= eps && (self.stretch_max - hi).abs() <= eps {
            return false;
        }
        self.stretch_min = lo;
        self.stretch_max = hi;
        true
    }

    pub(super) fn set_rgb_stretch(&mut self, min: [f64; 3], max: [f64; 3], gamma: f32) -> bool {
        if !min.iter().all(|v| v.is_finite())
            || !max.iter().all(|v| v.is_finite())
            || !gamma.is_finite()
        {
            return false;
        }
        let gamma = gamma.clamp(0.2, 4.0);
        let mut next_max = max;
        for i in 0..3 {
            if next_max[i] <= min[i] {
                next_max[i] = min[i] + min[i].abs().max(1.0) * 1e-6;
            }
        }
        if self.rgb_min == min
            && self.rgb_max == next_max
            && (self.rgb_gamma - gamma).abs() <= f32::EPSILON
        {
            return false;
        }
        self.rgb_min = min;
        self.rgb_max = next_max;
        self.rgb_gamma = gamma;
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

    pub(super) fn set_scl_mask_enabled(&mut self, enabled: bool) -> bool {
        if self.scl_mask_enabled == enabled {
            return false;
        }
        self.scl_mask_enabled = enabled;
        self.bump_rev();
        true
    }

    pub(super) fn set_scl_mask_bits(&mut self, bits: u32) -> bool {
        if self.scl_mask_bits == bits {
            return false;
        }
        self.scl_mask_bits = bits;
        self.bump_rev();
        true
    }

    pub(super) fn reset_stretch(&mut self) -> bool {
        if self.formula.is_scalar() {
            let (lo, hi) = self.preset.default_scalar_range();
            self.set_stretch_range(lo, hi)
        } else {
            let (lo, hi) = self.default_rgb_range;
            self.set_rgb_stretch(lo, hi, 1.0)
        }
    }

    /// Return whether `scl_class` is currently masked out by this layer's `SCL`
    /// configuration.
    #[cfg(test)]
    pub(crate) const fn is_scl_masked(&self, scl_class: u16) -> bool {
        is_scl_class_masked(self.scl_mask_enabled, self.scl_mask_bits, scl_class)
    }
}

/// Evaluate whether `scl_class` is masked out under `enabled` + `mask_bits`.
pub(crate) const fn is_scl_class_masked(enabled: bool, mask_bits: u32, scl_class: u16) -> bool {
    enabled && scl_class < 32 && ((mask_bits & (1u32 << scl_class)) != 0)
}

/// Multi-asset data owned by [`super::layer::LayerKind::BandMath`].
pub(super) struct BandMathLayerData {
    /// Active float band assets (`2` for index/diff/ratio, `3` for RGB).
    pub(super) bands: Vec<RasterAsset>,
    /// Optional nearest-neighbor `R8Uint` Scene Classification Layer (`SCL`) mask.
    pub(super) scl: Option<RasterAsset>,
    pub(super) style: BandMathStyle,
}

impl BandMathLayerData {
    pub(super) fn primary_asset(&self) -> &RasterAsset {
        &self.bands[0]
    }

    pub(super) fn assets(&self) -> impl Iterator<Item = &RasterAsset> {
        self.bands.iter().chain(self.scl.iter())
    }

    pub(super) fn assets_mut(&mut self) -> impl Iterator<Item = &mut RasterAsset> {
        self.bands.iter_mut().chain(self.scl.iter_mut())
    }
}

/// Summary of one constituent input band in a `BandMath` layer, projected into
/// [`super::BandMathSummary`] for the Layer Card UI.
#[derive(Clone, Debug)]
pub(crate) struct BandMathInputSummary {
    pub(crate) slot_name: String,
    pub(crate) label: String,
    pub(crate) source_pixel_size_m: Option<(f64, f64)>,
}

/// Read-only UI summary of a `BandMath` layer projected into
/// [`super::LayerSummary`].
#[derive(Clone, Debug)]
pub(crate) struct BandMathSummary {
    pub(crate) formula: BandMathFormula,
    pub(crate) recipe_label: String,
    pub(crate) colormap: ColormapId,
    pub(crate) invert_colormap: bool,
    pub(crate) stretch_min: f64,
    pub(crate) stretch_max: f64,
    pub(crate) rgb_min: [f64; 3],
    pub(crate) rgb_max: [f64; 3],
    pub(crate) rgb_gamma: f32,
    pub(crate) has_scl: bool,
    pub(crate) scl_mask_enabled: bool,
    pub(crate) scl_mask_bits: u32,
    pub(crate) inputs: Vec<BandMathInputSummary>,
    pub(crate) scl_label: Option<String>,
}

/// Per-channel sample state in a [`BandMathReadout`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum BandChannelSample {
    Value(f64),
    Nodata,
    Unavailable,
}

impl BandChannelSample {
    pub(crate) const fn value(self) -> Option<f64> {
        match self {
            Self::Value(v) => Some(v),
            Self::Nodata | Self::Unavailable => None,
        }
    }
}

/// Zoom-invariant CPU readout payload for a `BandMath` layer at the cursor.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct BandMathReadout {
    pub(crate) formula_label: String,
    pub(crate) is_scalar: bool,
    pub(crate) computed: Option<f64>,
    pub(crate) channels: Vec<(String, BandChannelSample)>,
    pub(crate) scl_class: Option<u16>,
    pub(crate) scl_masked: bool,
    pub(crate) has_unavailable: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn band_math_scalar_formulas_preserve_zero_and_guard_singularities() {
        assert_eq!(
            BandMathFormula::NormalizedDiff.evaluate_scalar(0.5, 0.5),
            Some(0.0)
        );
        assert!(
            (BandMathFormula::NormalizedDiff
                .evaluate_scalar(0.8, 0.2)
                .unwrap()
                - 0.6)
                .abs()
                < 1e-12
        );
        assert_eq!(
            BandMathFormula::NormalizedDiff.evaluate_scalar(0.0, 0.0),
            None
        );
        assert_eq!(
            BandMathFormula::NormalizedDiff.evaluate_scalar(0.5, -0.5),
            None
        );
        assert_eq!(
            BandMathFormula::NormalizedDiff.evaluate_scalar(f64::NAN, 0.2),
            None
        );
        assert_eq!(
            BandMathFormula::Difference.evaluate_scalar(0.25, 0.25),
            Some(0.0)
        );
        assert_eq!(BandMathFormula::Ratio.evaluate_scalar(0.0, 0.5), Some(0.0));
        assert_eq!(BandMathFormula::Ratio.evaluate_scalar(0.5, 0.0), None);

        let mut style = BandMathStyle::from_preset(BandMathPreset::Ndvi, true);
        assert!(style.is_scl_masked(9));
        assert!(!style.is_scl_masked(4));
        style.set_scl_mask_enabled(false);
        assert!(!style.is_scl_masked(9));

        // Custom U8 RGB composite retains its 0..255 default range across Reset.
        let mut rgb_style =
            BandMathStyle::for_custom(BandMathFormula::RgbComposite, false, Some(255.0));
        assert_eq!(rgb_style.rgb_max, [255.0; 3]);
        rgb_style.set_rgb_stretch([10.0; 3], [120.0; 3], 1.5);
        assert!(rgb_style.reset_stretch());
        assert_eq!(rgb_style.rgb_min, [0.0; 3]);
        assert_eq!(rgb_style.rgb_max, [255.0; 3]);
        assert_eq!(rgb_style.rgb_gamma, 1.0);

        // Custom NormalizedDiff uses CustomNormalizedDiff (-1.0..1.0).
        let mut nd_style = BandMathStyle::for_custom(BandMathFormula::NormalizedDiff, false, None);
        assert_eq!(nd_style.preset, BandMathPreset::CustomNormalizedDiff);
        assert_eq!((nd_style.stretch_min, nd_style.stretch_max), (-1.0, 1.0));
        nd_style.set_stretch_range(0.0, 0.5);
        assert!(nd_style.reset_stretch());
        assert_eq!((nd_style.stretch_min, nd_style.stretch_max), (-1.0, 1.0));
    }
}
