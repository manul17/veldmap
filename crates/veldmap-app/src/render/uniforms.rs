//! GPU uniform buffer layouts shared across [`super::pipelines`],
//! [`super::viewport`], and [`super::globe`].

use bytemuck::{Pod, Zeroable};

/// `TileUniform::tile_kind` tag for `Geodetic` / `WebMercator` tiles.
pub(super) const TILE_KIND_GEO: f32 = 0.0;
/// `TileUniform::tile_kind` tag for North PolarCap tiles near the pole.
pub(super) const TILE_KIND_POLAR_NORTH: f32 = 1.0;
/// `TileUniform::tile_kind` tag for South PolarCap tiles near the pole.
pub(super) const TILE_KIND_POLAR_SOUTH: f32 = -1.0;
/// Scale multiplier applied to `TILE_KIND_POLAR_*` (`±2.0`) when a PolarCap
/// tile is strictly clear of the pole and uses cancellation-free relative
/// longitude deltas in `shaders/globe.wgsl` (`abs(tile.tile_kind) > 1.5`).
pub(super) const TILE_KIND_POLAR_CLEAR_SCALE: f32 = 2.0;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct CameraUniform {
    pub(super) view_proj: [[f32; 4]; 4],
    pub(super) grid_step_deg: f32,
    pub(super) grid_width_px: f32,
    pub(super) polar_r_cap: f32,
    pub(super) _pad0: f32,
}

const _: () = {
    assert!(std::mem::size_of::<CameraUniform>() == 80);
    assert!(std::mem::offset_of!(CameraUniform, view_proj) == 0);
    assert!(std::mem::offset_of!(CameraUniform, grid_step_deg) == 64);
    assert!(std::mem::offset_of!(CameraUniform, grid_width_px) == 68);
    assert!(std::mem::offset_of!(CameraUniform, polar_r_cap) == 72);
};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct TileUniform {
    pub(super) offset_eye: [f32; 3],
    pub(super) tile_kind: f32,
    pub(super) uv_rect: [f32; 4],
    pub(super) grid_anchor: [f32; 4],
}

const _: () = {
    assert!(std::mem::size_of::<TileUniform>() == 48);
    assert!(std::mem::offset_of!(TileUniform, offset_eye) == 0);
    assert!(std::mem::offset_of!(TileUniform, tile_kind) == 12);
    assert!(std::mem::offset_of!(TileUniform, uv_rect) == 16);
    assert!(std::mem::offset_of!(TileUniform, grid_anchor) == 32);
};
