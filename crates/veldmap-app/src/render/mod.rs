//! Offscreen `wgpu` WGS84 quadtree globe renderer decoupled from `egui`.
//!
//! Renders per-tile CPU `f64` RTC quadtree meshes (textured sRGB basemap or
//! untextured WGS84 ellipsoid shading) and the post-overlay procedural
//! `fwidth()` graticule into an offscreen `Rgba8UnormSrgb` + `Depth32Float`
//! Reverse-Z target, then composites 1:1 with `FilterMode::Nearest` onto
//! `egui_wgpu`'s target format via `view_formats`.

mod bind_groups;
mod globe;
mod mesh;
mod mesh_cache;
mod overlay;
mod pipelines;
mod snapshot;
mod uniforms;
mod vector;
mod viewport;

pub use globe::GlobeRenderer;
pub use overlay::OverlayMeshKey;
pub use snapshot::{
    DrawOverlayLayer, DrawOverlayPatch, DrawSnapshot, DrawTile, DrawVectorLayer,
    SnapshotBuildParams, TileCarrierPolicy,
};
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

/// Choose a clean graticule step in degrees (dividing `90°` evenly) such that
/// lines stay roughly `~85 pt` apart on screen across orbital-to-2 m zoom.
fn adaptive_grid_step_deg(metres_per_pt: f64) -> f64 {
    const STEPS_DEG: [f64; 13] = [
        0.000_01, 0.000_05, 0.000_1, 0.000_5, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 15.0,
    ];
    if !metres_per_pt.is_finite() || metres_per_pt <= 0.0 {
        return 15.0;
    }
    let target_deg = (metres_per_pt * 85.0) / 111_320.0;
    for &step in &STEPS_DEG {
        if step >= target_deg {
            return step;
        }
    }
    15.0
}

/// Embedded WGSL source for the quadtree globe body and procedural graticule.
pub const GLOBE_WGSL: &str = include_str!("../shaders/globe.wgsl");
/// Embedded WGSL source for depth-less post-body raster overlay warp meshes.
pub const OVERLAY_WGSL: &str = include_str!("../shaders/overlay.wgsl");
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
    use super::{NATIVE_PRIMARY_BACKENDS, adaptive_grid_step_deg, clamp_to_primary_backends};

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

    #[test]
    fn adaptive_grid_step_divides_ninety_degrees_cleanly() {
        for m_per_pt in [1e-4, 0.002, 0.5, 10.0, 500.0, 16_000.0, 100_000.0] {
            let step = adaptive_grid_step_deg(m_per_pt);
            let ratio = 90.0 / step;
            assert!(
                (ratio - ratio.round()).abs() < 1e-6,
                "step {step} does not cleanly divide 90 deg (ratio={ratio})"
            );
        }
        assert_eq!(adaptive_grid_step_deg(16_320.0), 15.0);
        assert_eq!(adaptive_grid_step_deg(0.002), 0.000_01);
    }
}
