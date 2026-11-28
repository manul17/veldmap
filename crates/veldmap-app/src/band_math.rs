//! Dependency-free analytical Band Math formula contract and shared RGB gamma
//! bounds used across `layers/`, `render/`, `storage/`, and `ui/`.
//!
//! Kept at crate root so `render/` and `storage/` never depend back on
//! `layers/` (preserving a strictly acyclic module graph).

/// Epsilon guarding division-by-zero in `(A - B) / (A + B)` and `A / B`
/// across both GPU (`band_math_composite.wgsl`) and CPU (`CursorProber`).
const BAND_MATH_DIV_EPS: f64 = 1e-6;

/// Minimum supported RGB composite display gamma.
const MIN_RGB_GAMMA: f32 = 0.2;
/// Maximum supported RGB composite display gamma.
const MAX_RGB_GAMMA: f32 = 4.0;

/// Clamp an RGB composite gamma into `0.2..=4.0`, falling back to `1.0` for
/// non-finite or non-positive values across live styling, GPU rendering, and
/// persisted session/project normalization.
pub(crate) fn clamp_rgb_gamma(gamma: f32) -> f32 {
    if gamma.is_finite() && gamma > 0.0 {
        gamma.clamp(MIN_RGB_GAMMA, MAX_RGB_GAMMA)
    } else {
        1.0
    }
}

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

    /// Canonical UI input slot labels matching `required_bands()`.
    pub(crate) const fn slot_names(self) -> &'static [&'static str] {
        match self {
            Self::RgbComposite => &["Red", "Green", "Blue"],
            Self::NormalizedDiff | Self::Difference | Self::Ratio => &["Band A", "Band B"],
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamp_rgb_gamma_bounds_and_sanitizes_invalid_values() {
        assert_eq!(clamp_rgb_gamma(1.5), 1.5);
        assert_eq!(clamp_rgb_gamma(0.05), 0.2);
        assert_eq!(clamp_rgb_gamma(5.0), 4.0);
        assert_eq!(clamp_rgb_gamma(0.0), 1.0);
        assert_eq!(clamp_rgb_gamma(-1.0), 1.0);
        assert_eq!(clamp_rgb_gamma(f32::NAN), 1.0);
        assert_eq!(clamp_rgb_gamma(f32::INFINITY), 1.0);
    }

    #[test]
    fn band_math_formula_slot_names_match_required_bands() {
        for formula in BandMathFormula::ALL {
            assert_eq!(formula.slot_names().len(), formula.required_bands());
        }
    }
}
