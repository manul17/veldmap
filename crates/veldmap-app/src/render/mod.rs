//! Offscreen `wgpu` WGS84 quadtree globe renderer decoupled from `egui`.
//!
//! Renders per-tile CPU `f64` RTC quadtree meshes (textured sRGB basemap or
//! untextured WGS84 ellipsoid shading) and the post-overlay procedural
//! `fwidth()` graticule into an offscreen `Rgba8UnormSrgb` + `Depth32Float`
//! Reverse-Z target, then composites 1:1 with `FilterMode::Nearest` onto
//! `egui_wgpu`'s target format via `view_formats`.

mod bind_groups;
mod globe;
mod graticule;
mod mesh;
mod mesh_cache;
mod overlay;
mod palette_cache;
mod pipelines;
mod snapshot;
mod uniforms;
mod vector;
mod viewport;

#[cfg(any(test, feature = "test-support"))]
pub use crate::colormap::{CategoricalPaletteId, ColormapId};
#[cfg(any(test, feature = "test-support"))]
pub use bind_groups::TileBindGroupCache;
#[cfg(not(any(test, feature = "test-support")))]
pub(crate) use bind_groups::TileBindGroupCache;
#[cfg(any(test, feature = "test-support"))]
pub use globe::GlobeRenderer;
#[cfg(not(any(test, feature = "test-support")))]
pub(crate) use globe::GlobeRenderer;
#[cfg(any(test, feature = "test-support"))]
pub use snapshot::{
    DrawOverlayLayer, DrawOverlayPatch, DrawOverlayTrack, DrawSnapshot, DrawTile, DrawVectorId,
    DrawVectorLayer, OverlayMeshKey, SnapshotBuildParams, TileCarrierPolicy,
};
#[cfg(not(any(test, feature = "test-support")))]
pub(crate) use snapshot::{
    DrawOverlayLayer, DrawOverlayPatch, DrawOverlayTrack, DrawSnapshot, DrawVectorId,
    DrawVectorLayer, OverlayMeshKey, SnapshotBuildParams, TileCarrierPolicy,
};
#[cfg(any(test, feature = "test-support"))]
pub use viewport::RenderParams;
#[cfg(not(any(test, feature = "test-support")))]
pub(crate) use viewport::RenderParams;

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

macro_rules! shader_source {
    ($(#[$meta:meta])* const $name:ident = $path:literal;) => {
        $(#[$meta])*
        #[cfg(any(test, feature = "test-support"))]
        pub const $name: &str = include_str!($path);
        $(#[$meta])*
        #[cfg(not(any(test, feature = "test-support")))]
        pub(crate) const $name: &str = include_str!($path);
    };
}

shader_source! {
    /// Embedded WGSL source for the quadtree globe body and procedural graticule.
    const GLOBE_WGSL = "../shaders/globe.wgsl";
}
shader_source! {
    /// Embedded WGSL source for depth-less post-body display raster overlay warp meshes.
    const OVERLAY_WGSL = "../shaders/overlay.wgsl";
}
shader_source! {
    /// Embedded WGSL source for depth-less post-body scientific (`R32Float`) raster overlays.
    const OVERLAY_SCIENTIFIC_WGSL = "../shaders/overlay_scientific.wgsl";
}
shader_source! {
    /// Embedded WGSL source for depth-less post-body categorical (`R8Uint`/`R16Uint`) raster overlays.
    const OVERLAY_CATEGORICAL_WGSL = "../shaders/overlay_categorical.wgsl";
}
shader_source! {
    /// Embedded WGSL source for depth-less post-body draped vector geometry.
    const VECTOR_WGSL = "../shaders/vector.wgsl";
}
shader_source! {
    /// Embedded WGSL source for 1:1 nearest-neighbor offscreen compositing.
    const COMPOSITE_WGSL = "../shaders/composite.wgsl";
}

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
