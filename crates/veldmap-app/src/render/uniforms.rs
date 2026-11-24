//! GPU uniform buffer layouts shared across [`super::pipelines`],
//! [`super::viewport`], and [`super::globe`].

use bytemuck::{Pod, Zeroable};
use veldmap_core::tile::{PlanetaryScheme, QuadTileId};

use super::mesh::TileGridAnchor;

/// `TileUniform::tile_kind` tag for `Geodetic` / `WebMercator` tiles.
pub(super) const TILE_KIND_GEO: f32 = 0.0;
/// `TileUniform::tile_kind` tag for North PolarCap tiles near the pole.
const TILE_KIND_POLAR_NORTH: f32 = 1.0;
/// `TileUniform::tile_kind` tag for South PolarCap tiles near the pole.
const TILE_KIND_POLAR_SOUTH: f32 = -1.0;
/// Scale multiplier applied to `TILE_KIND_POLAR_*` (`±2.0`) when a PolarCap
/// tile is strictly clear of the pole and uses cancellation-free relative
/// longitude deltas in `shaders/globe.wgsl` (`abs(tile.tile_kind) > 1.5`).
const TILE_KIND_POLAR_CLEAR_SCALE: f32 = 2.0;

/// Compute per-tile `(tile_kind, grid_anchor)` uniform values from a cached
/// [`TileGridAnchor`] so `GlobeRenderer` never re-samples `sample_tile_surface`
/// on every dirty frame.
pub(super) fn tile_graticule_uniform(
    id: QuadTileId,
    cached_anchor: Option<TileGridAnchor>,
    step_deg: f64,
) -> (f32, [f32; 4]) {
    let anchor = cached_anchor.unwrap_or_else(|| TileGridAnchor::for_tile(id));
    match id.scheme {
        PlanetaryScheme::PolarCap { north } => {
            let colat_phase = (90.0 - anchor.lat0.abs()).rem_euclid(step_deg);
            let lon_phase = anchor.lon0.rem_euclid(step_deg);

            // Use cancellation-free relative longitude deltas when the tile's
            // disk center is strictly clear of the pole (`rho0 > 4 * tile_half_diag`).
            let rho0 = anchor.dx0.hypot(anchor.dy0);
            let n = f64::from(1u32 << id.level.min(30));
            let tile_half_diag = std::f64::consts::SQRT_2 / n;
            let clear_of_pole = id.level >= 2 && rho0 > 4.0 * tile_half_diag;
            let sign = if north {
                TILE_KIND_POLAR_NORTH
            } else {
                TILE_KIND_POLAR_SOUTH
            };
            let kind = if clear_of_pole {
                sign * TILE_KIND_POLAR_CLEAR_SCALE
            } else {
                sign
            };
            (
                kind,
                [
                    anchor.dx0 as f32,
                    anchor.dy0 as f32,
                    colat_phase as f32,
                    lon_phase as f32,
                ],
            )
        }
        PlanetaryScheme::Geodetic | PlanetaryScheme::WebMercator => {
            // Wrap tile anchor phase modulo `grid_step_deg` in f64 on CPU so
            // `f32` fragment arithmetic preserves sub-millimetre accuracy even
            // at `2 m` altitude.
            let lon_phase = anchor.lon0.rem_euclid(step_deg);
            let lat_phase = anchor.lat0.rem_euclid(step_deg);
            (
                TILE_KIND_GEO,
                [lon_phase as f32, lat_phase as f32, 0.0, 0.0],
            )
        }
    }
}

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
