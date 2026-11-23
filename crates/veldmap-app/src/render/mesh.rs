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
        let north_base = surface_verts;
        let south_base = surface_verts + stride;
        let west_base = surface_verts + 2 * stride;
        let east_base = surface_verts + 3 * stride;

        // 2. North edge skirt (row = 0, col = 0..n): faces outward north.
        for col in 0..n {
            let s0 = col;
            let s1 = col + 1;
            let k0 = north_base + col;
            let k1 = north_base + col + 1;
            indices.extend_from_slice(&[s0, s1, k0]);
            indices.extend_from_slice(&[s1, k1, k0]);
        }

        // 3. South edge skirt (row = n, col = 0..n): faces outward south.
        let south_row = n * stride;
        for col in 0..n {
            let s0 = south_row + col;
            let s1 = south_row + col + 1;
            let k0 = south_base + col;
            let k1 = south_base + col + 1;
            indices.extend_from_slice(&[s0, k0, s1]);
            indices.extend_from_slice(&[s1, k0, k1]);
        }

        // 4. West edge skirt (col = 0, row = 0..n): faces outward west.
        for row in 0..n {
            let s0 = row * stride;
            let s1 = (row + 1) * stride;
            let k0 = west_base + row;
            let k1 = west_base + row + 1;
            indices.extend_from_slice(&[s0, k0, s1]);
            indices.extend_from_slice(&[s1, k0, k1]);
        }

        // 5. East edge skirt (col = n, row = 0..n): faces outward east.
        for row in 0..n {
            let s0 = row * stride + n;
            let s1 = (row + 1) * stride + n;
            let k0 = east_base + row;
            let k1 = east_base + row + 1;
            indices.extend_from_slice(&[s0, s1, k0]);
            indices.extend_from_slice(&[s1, k1, k0]);
        }

        let full_end = indices.len() as u32;
        Self {
            indices,
            surface: 0..surface_end,
            full: 0..full_end,
        }
    }
}

/// Build the per-tile CPU `f64` RTC vertex buffer (surface grid + 4 downward
/// skirts) for `tile`.
pub fn build_tile_vertices(tile: &VisibleTile, grid_cells: u32) -> Vec<TileVertex> {
    let n = grid_cells.max(1);
    let stride = (n + 1) as usize;
    let total_verts = stride * stride + 4 * stride;
    let mut vertices = Vec::with_capacity(total_verts);

    let (anchor_a, anchor_b) = tile_grid_anchor(tile.id).unwrap_or((0.0, 0.0));
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

    // 2. North skirt (row = 0, col = 0..=n).
    for col in 0..stride {
        push_skirt(&mut vertices, col, stride + col);
    }
    // 3. South skirt (row = n, col = 0..=n).
    let south_row = (n as usize) * stride;
    let south_inner = ((n - 1) as usize) * stride;
    for col in 0..stride {
        push_skirt(&mut vertices, south_row + col, south_inner + col);
    }
    // 4. West skirt (col = 0, row = 0..=n).
    for row in 0..stride {
        push_skirt(&mut vertices, row * stride, row * stride + 1);
    }
    // 5. East skirt (col = n, row = 0..=n).
    for row in 0..stride {
        push_skirt(
            &mut vertices,
            row * stride + (n as usize),
            row * stride + ((n - 1) as usize),
        );
    }

    vertices
}

#[cfg(test)]
#[path = "mesh_tests.rs"]
mod tests;
