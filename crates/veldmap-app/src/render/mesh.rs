//! Per-tile CPU `f64` RTC WGS84 quadtree geometry with per-cell sag-scaled
//! skirts and a single shared static index buffer.
//!
//! Both the globe surface (`body_*_pipeline`) and the post-overlay procedural
//! `fwidth()` graticule (`grid_pipeline`) consume the exact same per-tile
//! vertex buffer and shared static index buffer (`full` range for body +
//! skirts, `surface` range for graticule).

use std::ops::Range;

use bytemuck::{Pod, Zeroable};
use veldmap_core::geo::geodesy;
use veldmap_core::tile::{PlanetaryScheme, QuadTileId, VisibleTile, sample_tile_surface};

/// Tile-local `f64` graticule anchor `(anchor_a, anchor_b)`:
/// - For `Geodetic` / `WebMercator`: `(lon_center_deg, lat_center_deg)`
/// - For `PolarCap`: `(dx_center, dy_center)` in unit-disk coordinates
pub(super) fn tile_grid_anchor(id: QuadTileId) -> Option<(f64, f64)> {
    let (lat, lon, _, _, dx, dy) = sample_tile_surface(id, 0.5, 0.5)?;
    match id.scheme {
        PlanetaryScheme::PolarCap { .. } => Some((dx, dy)),
        PlanetaryScheme::Geodetic | PlanetaryScheme::WebMercator => Some((lon, lat)),
    }
}

/// Vertex layout shared by textured/untextured quadtree tiles and the
/// post-overlay procedural graticule.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct TileVertex {
    /// Surface (or skirt) position relative to the tile's CPU `f64` RTC origin
    /// (`(P_ecef_m - origin_ecef_m) as f32`, in metres).
    pub rel_pos: [f32; 3],
    /// Outward unit WGS84 ellipsoidal normal.
    pub normal: [f32; 3],
    /// Tile-local UV coordinates in `[0, 1]^2` (transformed by per-draw
    /// `uv_rect` when sampling a coarser ancestor texture).
    pub uv: [f32; 2],
    /// Tile-relative graticule coordinates computed in `f64` before casting to
    /// `f32`:
    /// - `Geodetic` / `WebMercator`: `(lon_deg - anchor_lon, lat_deg - anchor_lat)`
    /// - `PolarCap`: `(dx - anchor_dx, dy - anchor_dy)` in unit-disk coordinates
    pub grid_rel: [f32; 2],
}

const _: () = {
    assert!(std::mem::size_of::<TileVertex>() == 40);
};

impl TileVertex {
    pub const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &wgpu::vertex_attr_array![
            0 => Float32x3,
            1 => Float32x3,
            2 => Float32x2,
            3 => Float32x2,
        ],
    };
}

/// Shared static triangle-list index buffer for every `grid_cells x grid_cells`
/// quadtree tile (both main planetary tiles and adaptive Polar Stereographic
/// cap tiles).
#[derive(Clone, Debug)]
pub struct SharedTileIndices {
    pub indices: Vec<u32>,
    /// Index range covering the `grid_cells x grid_cells` surface quads only
    /// (used by the post-overlay `grid_pipeline` so skirts never draw graticule
    /// artifacts).
    pub surface: Range<u32>,
    /// Full index range covering the surface quads plus all four downward
    /// skirts (used by `body_ellipsoid_pipeline` and `body_textured_pipeline`).
    pub full: Range<u32>,
}

impl SharedTileIndices {
    /// Build the shared static index buffer for a `grid_cells x grid_cells`
    /// tile with four perimeter skirt strips.
    pub fn build(grid_cells: u32) -> Self {
        let n = grid_cells.max(1);
        let stride = n + 1;
        let surface_verts = stride * stride;

        let surface_idx_count = (n * n * 6) as usize;
        let skirt_idx_count = (4 * n * 6) as usize;
        let mut indices = Vec::with_capacity(surface_idx_count + skirt_idx_count);

        // 1. Surface quads (CCW outward when viewed from outside the ellipsoid).
        for row in 0..n {
            for col in 0..n {
                let top = row * stride + col;
                let bottom = top + stride;
                indices.extend_from_slice(&[top, bottom, top + 1]);
                indices.extend_from_slice(&[top + 1, bottom, bottom + 1]);
            }
        }
        let surface_end = indices.len() as u32;

        // Skirt vertex ring offsets immediately follow the surface vertices:
        //   north: [surface_verts .. surface_verts + stride)
        //   south: [surface_verts + stride .. surface_verts + 2 * stride)
        //   west:  [surface_verts + 2 * stride .. surface_verts + 3 * stride)
        //   east:  [surface_verts + 3 * stride .. surface_verts + 4 * stride)
        let skirt_edges = [
            (0u32, 1u32, surface_verts, false),
            (n * stride, 1u32, surface_verts + stride, true),
            (0u32, stride, surface_verts + 2 * stride, true),
            (n, stride, surface_verts + 3 * stride, false),
        ];
        for (surf_start, surf_step, skirt_base, flip_winding) in skirt_edges {
            for i in 0..n {
                let s0 = surf_start + i * surf_step;
                let s1 = s0 + surf_step;
                let k0 = skirt_base + i;
                let k1 = skirt_base + i + 1;
                if flip_winding {
                    indices.extend_from_slice(&[s0, k0, s1, s1, k0, k1]);
                } else {
                    indices.extend_from_slice(&[s0, s1, k0, s1, k1, k0]);
                }
            }
        }

        let full_end = indices.len() as u32;
        Self {
            indices,
            surface: 0..surface_end,
            full: 0..full_end,
        }
    }
}

/// Build per-tile CPU `f64` RTC vertex data (surface grid + 4 downward skirts)
/// together with its cached `f64` graticule anchor `(anchor_a, anchor_b)` so
/// `GlobeRenderer::upload_uniforms` does not re-sample
/// `sample_tile_surface(id, 0.5, 0.5)`.
pub fn build_tile_mesh_data(tile: &VisibleTile, grid_cells: u32) -> (Vec<TileVertex>, (f64, f64)) {
    let n = grid_cells.max(1);
    let stride = (n + 1) as usize;
    let total_verts = stride * stride + 4 * stride;
    let mut vertices = Vec::with_capacity(total_verts);

    let anchor = tile_grid_anchor(tile.id).unwrap_or((0.0, 0.0));
    let (anchor_a, anchor_b) = anchor;
    let is_polar = matches!(tile.id.scheme, PlanetaryScheme::PolarCap { .. });
    let origin = tile.origin_ecef_m;
    let n_f = f64::from(n);

    // 1. Surface grid `(n + 1) x (n + 1)`.
    for row in 0..=n {
        let t = f64::from(row) / n_f;
        for col in 0..=n {
            let s = f64::from(col) / n_f;
            let (lat, lon, u_tex, v_tex, dx, dy) =
                sample_tile_surface(tile.id, s, t).unwrap_or((0.0, 0.0, s, t, 0.0, 0.0));
            let (pos_ecef, unit_normal) = geodesy::surface_ecef_and_normal(lat, lon);
            let rel_pos = (pos_ecef - origin).as_vec3().to_array();
            let normal = unit_normal.as_vec3().to_array();
            let grid_rel = if is_polar {
                [(dx - anchor_a) as f32, (dy - anchor_b) as f32]
            } else {
                [(lon - anchor_a) as f32, (lat - anchor_b) as f32]
            };
            vertices.push(TileVertex {
                rel_pos,
                normal,
                uv: [u_tex as f32, v_tex as f32],
                grid_rel,
            });
        }
    }

    // Helper: push a skirt vertex displaced inward along `-normal` by
    // `tile.skirt_depth_m` metres plus a tiny outward tangential flare so
    // sub-millimetre f32 RTC edge rounding gaps never leak directly beneath a
    // nadir camera (while staying strictly below adjacent surfaces in depth).
    let drop_m = tile.skirt_depth_m as f32;
    let flare_m = (0.05 * drop_m).max(0.002);
    let push_skirt = |out: &mut Vec<TileVertex>, src_idx: usize, inner_idx: usize| {
        let v = out[src_idx];
        let inner = out[inner_idx];
        let raw_x = v.rel_pos[0] - inner.rel_pos[0];
        let raw_y = v.rel_pos[1] - inner.rel_pos[1];
        let raw_z = v.rel_pos[2] - inner.rel_pos[2];
        let dot_n = raw_x * v.normal[0] + raw_y * v.normal[1] + raw_z * v.normal[2];
        let tx = raw_x - v.normal[0] * dot_n;
        let ty = raw_y - v.normal[1] * dot_n;
        let tz = raw_z - v.normal[2] * dot_n;
        let t_len = (tx * tx + ty * ty + tz * tz).sqrt();
        let (fx, fy, fz) = if t_len > 1e-9 {
            let s = flare_m / t_len;
            (tx * s, ty * s, tz * s)
        } else {
            (0.0, 0.0, 0.0)
        };
        out.push(TileVertex {
            rel_pos: [
                v.rel_pos[0] - v.normal[0] * drop_m + fx,
                v.rel_pos[1] - v.normal[1] * drop_m + fy,
                v.rel_pos[2] - v.normal[2] * drop_m + fz,
            ],
            ..v
        });
    };

    let n_usize = n as usize;
    let edge_strips = [
        (0usize, 1usize, stride, 1usize),                           // North
        (n_usize * stride, 1usize, (n_usize - 1) * stride, 1usize), // South
        (0usize, stride, 1usize, stride),                           // West
        (n_usize, stride, n_usize - 1, stride),                     // East
    ];
    for (surf_start, surf_step, inner_start, inner_step) in edge_strips {
        for i in 0..stride {
            push_skirt(
                &mut vertices,
                surf_start + i * surf_step,
                inner_start + i * inner_step,
            );
        }
    }

    (vertices, anchor)
}

#[cfg(test)]
#[path = "mesh_tests.rs"]
mod tests;
