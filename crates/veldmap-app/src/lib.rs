//! Library surface of `veldmap-app` exposing [`VeldmapApp`], [`AppError`],
//! and [`clamp_to_primary_backends`] for the binary entry point, plus a
//! `test-support`-gated headless rendering/upload harness for GPU smoke tests
//! and `naga` shader validation.
//!
//! ## Internal Modules
//! - `app` — Top-level `eframe::App` shell, input/tool state, native `rfd`
//!   dialogs, `egui_wgpu::CallbackResources` bridge, and session restoration.
//! - `band_math` — Dependency-free Band Math formula contract and shared RGB
//!   gamma bounds.
//! - `basemap` — Bundled NASA Blue Marble & live HTTP basemap policy/streaming.
//! - `colormap` — Linear-light scientific colormaps & categorical palettes.
//! - `export` — Pure-Rust CPU cartographic PNG overlay compositor (metric scale
//!   bar, rotated North arrow, scientific colorbar, and caption pill).
//! - `format` — Pure-Rust GIS display formatters (coordinates, MGRS, metric
//!   distances, GSD, areas, and 1–2–5 scale bar geometry).
//! - `gpu_tiles` — Bounded `GpuTileCache` and enqueue-bounded `PendingUploads`.
//! - `layers` — `OverlayManager`, `LayerStack`, multi-asset `BandMath`,
//!   background I/O/decode/tile-write orchestration, and CPU cursor probing.
//! - `measure` — Interactive Karney WGS84 distance/area measurement session.
//! - `render` — Offscreen two-pass `wgpu` `GlobeRenderer` (Pass 1
//!   body + segmented depth-less Pass 2 overlays, `BandMath` scratch pool,
//!   split-screen swipe scissors, vectors, and procedural graticule).
//! - `scene` — `SceneState` mediator coordinating camera snapshots, caches,
//!   VRAM budgeting, and deterministic shutdown.
//! - `stac` — Cloud/STAC search session, AOI state, and read-only UI DTOs.
//! - `storage` — `settings.json`, `session.json`, portable `.veldmap.json`
//!   project files, and `DiskCacheWorker`.
//! - `ui` — Presentation-only `egui` views emitting typed actions (zero
//!   `veldmap-net`/`veldmap-raster`/`veldmap-storage` imports).
//! - `vectors` — Retained GeoJSON vector layers and background tessellation.
//! - `worker` — Lightweight single-thread background worker primitive.

#![forbid(unsafe_code)]

use std::path::Path;

mod app;
mod band_math;
mod basemap;
mod colormap;
mod export;
mod format;
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
