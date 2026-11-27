//! Library surface of `veldmap-app` exposing [`VeldmapApp`], [`AppError`],
//! and [`clamp_to_primary_backends`] for the binary entry point, plus
//! a `test-support`-gated headless rendering/upload harness for GPU smoke tests
//! and `naga` shader validation.

#![forbid(unsafe_code)]

use std::path::Path;

mod app;
mod basemap;
mod colormap;
#[cfg(not(any(test, feature = "test-support")))]
mod gpu_tiles;
#[cfg(any(test, feature = "test-support"))]
pub mod gpu_tiles;
mod layers;
mod measure;
#[cfg(not(any(test, feature = "test-support")))]
mod render;
#[cfg(any(test, feature = "test-support"))]
pub mod render;
mod scene;
mod stac;
mod storage;
mod ui;
mod vectors;
mod worker;

pub use app::{AppError, VeldmapApp};
#[cfg(any(test, feature = "test-support"))]
pub use basemap::policy as basemap_policy;
#[cfg(any(test, feature = "test-support"))]
pub use basemap::{BUNDLED_BASEMAP_CONFIG, BUNDLED_BASEMAP_JPEG, BUNDLED_MAX_LEVEL};
pub use render::clamp_to_primary_backends;

/// Clamp a display opacity into `0.0..=1.0`, falling back to `1.0` for
/// non-finite values (`NaN` / infinities) so `f32::NAN.clamp(0.0, 1.0)` can
/// never propagate `NaN` into layer or renderer state.
pub(crate) fn clamp_opacity(opacity: f32) -> f32 {
    if opacity.is_finite() {
        opacity.clamp(0.0, 1.0)
    } else {
        1.0
    }
}

/// Extract a human-readable display name from `path` (falling back to its full
/// display string when no valid UTF-8 filename component is present).
pub(crate) fn display_name_from_path(path: &Path) -> String {
    path.file_name()
        .and_then(|n| n.to_str())
        .map_or_else(|| path.display().to_string(), str::to_owned)
}

#[cfg(test)]
mod tests {
    use super::clamp_opacity;

    #[test]
    fn clamp_opacity_clamps_and_guards_non_finite_values() {
        assert_eq!(clamp_opacity(0.4), 0.4);
        assert_eq!(clamp_opacity(-0.5), 0.0);
        assert_eq!(clamp_opacity(1.5), 1.0);
        assert_eq!(clamp_opacity(f32::NAN), 1.0);
        assert_eq!(clamp_opacity(f32::INFINITY), 1.0);
        assert_eq!(clamp_opacity(f32::NEG_INFINITY), 1.0);
    }
}
