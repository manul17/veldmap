//! Offscreen `wgpu` WGS84 quadtree globe renderer decoupled from `egui`.
//!
//! Renders per-tile CPU `f64` RTC quadtree meshes (textured sRGB basemap or
//! untextured WGS84 ellipsoid shading) in Pass 1 (`Rgba8UnormSrgb` +
//! `Depth32Float` Reverse-Z), followed by segmented depth-less Pass 2 post-body
//! rendering in exact painter's order: direct display / scientific (`R32Float`)
//! / categorical (`R8Uint`/`R16Uint`) raster warp meshes, multi-asset
//! screen-space `BandMath` + `SCL` masking (backed by the lazy shared
//! quiet-`NaN`-filled `R32Float` + `R8Uint` `ScratchTargetPool` with
//! co-grid MRT & clamped scissor bounds), draped GeoJSON / measurement / STAC
//! AOI & HiDPI ribbon/hatch vector overlays, and the post-overlay procedural
//! `fwidth()` graticule, then composites 1:1 with `FilterMode::Nearest` onto
//! `egui_wgpu`'s target format via `view_formats`.

mod bind_groups;
mod globe;
mod graticule;
mod mesh;
mod mesh_cache;
mod overlay;
mod palette_cache;
mod pipelines;
mod scratch;
mod snapshot;
mod swipe;
mod uniforms;
mod vector;
mod vector_pack;
mod viewport;

#[cfg(any(test, feature = "test-support"))]
pub use crate::colormap::{CategoricalPaletteId, ColormapId};
#[cfg(any(test, feature = "test-support"))]
pub use crate::layers::BandMathFormula;
pub use bind_groups::TileBindGroupCache;
pub use globe::{GlobeRenderer, OffscreenReadback};
#[cfg(any(test, feature = "test-support"))]
pub use scratch::compute_band_math_scissor_rect;
pub(crate) use scratch::{ScratchDemand, estimate_viewport_target_vram_bytes};
#[cfg(any(test, feature = "test-support"))]
pub use snapshot::DrawTile;
pub use snapshot::{
    DrawBandMathLayer, DrawOverlayContent, DrawOverlayLayer, DrawOverlayPatch, DrawOverlayTrack,
    DrawScratchBand, DrawScratchMask, DrawSnapshot, DrawVectorId, DrawVectorLayer, OverlayMeshKey,
    SnapshotBuildParams, TileCarrierPolicy,
};
pub use swipe::{SwipeRenderState, snap_split_x};
#[cfg(any(test, feature = "test-support"))]
pub use uniforms::{UNIFORM_LAYOUT_SPECS, UniformLayoutSpec};
pub use viewport::RenderParams;

/// Native desktop primary `wgpu` backends (Vulkan, DX12, Metal), explicitly
/// excluding OpenGL/GLES and WebGPU.
const NATIVE_PRIMARY_BACKENDS: wgpu::Backends =
    wgpu::Backends::PRIMARY.difference(wgpu::Backends::BROWSER_WEBGPU);

/// Restrict environment-resolved `wgpu` backends to native desktop primary
/// backends (`Vulkan`, `DX12`, `Metal`), warning when unsupported entries are
/// ignored or when falling back.
pub fn clamp_to_primary_backends(backends: wgpu::Backends) -> wgpu::Backends {
    let unsupported = backends - NATIVE_PRIMARY_BACKENDS;
    if !unsupported.is_empty() {
        tracing::warn!(
            ?unsupported,
            "Ignoring unsupported WGPU_BACKEND entries (only Vulkan, DX12, and Metal are supported)"
        );
    }
    let primary = backends & NATIVE_PRIMARY_BACKENDS;
    if primary.is_empty() {
        tracing::warn!(
            "WGPU_BACKEND selected no supported backends; falling back to native primary backends"
        );
        NATIVE_PRIMARY_BACKENDS
    } else {
        primary
    }
}

/// Embedded WGSL source for the quadtree globe body and procedural graticule.
pub const GLOBE_WGSL: &str = include_str!("../shaders/globe.wgsl");
/// Embedded WGSL source for depth-less post-body display raster overlay warp meshes.
pub const OVERLAY_WGSL: &str = include_str!("../shaders/overlay.wgsl");
/// Embedded WGSL source for depth-less post-body scientific (`R32Float`) raster overlays.
pub const OVERLAY_SCIENTIFIC_WGSL: &str = concat!(
    include_str!("../shaders/r32f_common.wgsl"),
    "\n",
    include_str!("../shaders/overlay_scientific.wgsl"),
);
/// Embedded WGSL source for depth-less post-body categorical (`R8Uint`/`R16Uint`) raster overlays.
pub const OVERLAY_CATEGORICAL_WGSL: &str = include_str!("../shaders/overlay_categorical.wgsl");
/// Embedded WGSL source for fullscreen quiet-NaN scratch target initialization.
pub const SCRATCH_NAN_FILL_WGSL: &str = include_str!("../shaders/scratch_nan_fill.wgsl");
/// Embedded WGSL source for single-band `R32Float` Band Math scratch rasterization.
pub const BAND_MATH_SCRATCH_WGSL: &str = concat!(
    include_str!("../shaders/r32f_common.wgsl"),
    "\n",
    include_str!("../shaders/band_math_scratch.wgsl"),
);
/// Embedded WGSL source for 2..=4 attachment MRT `R32Float` Band Math scratch rasterization.
pub const BAND_MATH_SCRATCH_MRT_WGSL: &str = concat!(
    include_str!("../shaders/r32f_common.wgsl"),
    "\n",
    include_str!("../shaders/band_math_scratch_mrt.wgsl"),
);
/// Embedded WGSL source for nearest-neighbor `R8Uint` `SCL` mask scratch rasterization.
pub const BAND_MATH_MASK_WGSL: &str = include_str!("../shaders/band_math_mask.wgsl");
/// Embedded WGSL source for fullscreen screen-space Band Math + `SCL` mask + colormap compositing.
pub const BAND_MATH_COMPOSITE_WGSL: &str = include_str!("../shaders/band_math_composite.wgsl");
/// Embedded WGSL source for depth-less post-body draped vector geometry.
pub const VECTOR_WGSL: &str = include_str!("../shaders/vector.wgsl");
/// Embedded WGSL source for 1:1 nearest-neighbor offscreen compositing.
pub const COMPOSITE_WGSL: &str = include_str!("../shaders/composite.wgsl");

/// Deep-space clear color in exact linear RGB (`#0B1320` in sRGB).
const CLEAR_COLOR_LINEAR: wgpu::Color = wgpu::Color {
    r: 0.003_347,
    g: 0.006_512,
    b: 0.014_444,
    a: 1.0,
};

#[cfg(test)]
mod tests {
    use super::{NATIVE_PRIMARY_BACKENDS, clamp_to_primary_backends};

    #[test]
    fn clamps_backends_to_primary_and_falls_back_when_empty() {
        assert_eq!(
            clamp_to_primary_backends(wgpu::Backends::VULKAN),
            wgpu::Backends::VULKAN
        );
        assert_eq!(
            clamp_to_primary_backends(wgpu::Backends::VULKAN | wgpu::Backends::GL),
            wgpu::Backends::VULKAN
        );
        assert_eq!(
            clamp_to_primary_backends(wgpu::Backends::GL | wgpu::Backends::BROWSER_WEBGPU),
            NATIVE_PRIMARY_BACKENDS
        );
        assert_eq!(
            clamp_to_primary_backends(wgpu::Backends::empty()),
            NATIVE_PRIMARY_BACKENDS
        );
    }
}
