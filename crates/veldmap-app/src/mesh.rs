//! Per-tile CPU `f64` RTC WGS84 quadtree geometry with per-cell sag-scaled
//! skirts and a single shared static index buffer.
//!
//! Both the globe surface (`body_*_pipeline`) and the post-overlay procedural
//! `fwidth()` graticule (`grid_pipeline`) consume the exact same per-tile
//! vertex buffer and shared static index buffer (`full` range for body +
//! skirts, `surface` range for graticule).

use std::ops::Range;

use bytemuck::{Pod, Zeroable};
use veldmap_core::geo::geodesy::{self, Geodetic};
use veldmap_core::tile::{TileScheme, VisibleTile, sample_tile_surface, tile_grid_anchor};

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
    let is_polar = matches!(tile.id.scheme, TileScheme::PolarCap { .. });
    let origin = tile.origin_ecef_m;
    let n_f = f64::from(n);

    // 1. Surface grid `(n + 1) x (n + 1)`.
    for row in 0..=n {
        let t = f64::from(row) / n_f;
        for col in 0..=n {
            let s = f64::from(col) / n_f;
            let (lat, lon, u_tex, v_tex, dx, dy) =
                sample_tile_surface(tile.id, s, t).unwrap_or((0.0, 0.0, s, t, 0.0, 0.0));
            let pos_ecef = geodesy::world(Geodetic::surface(lat, lon));
            let rel_pos = (pos_ecef - origin).as_vec3().to_array();
            let normal = geodesy::normal(lat, lon);
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
mod tests {
    use glam::Vec3;
    use veldmap_core::tile::{
        DEFAULT_GRID_CELLS, QuadTileId, QuadtreeConfig, TileScheme, describe_tile,
    };

    use super::*;

    #[test]
    fn all_planetary_and_polar_surface_triangles_have_outward_ccw_winding() {
        let cfg = QuadtreeConfig::bundled_geodetic();
        let shared = SharedTileIndices::build(DEFAULT_GRID_CELLS);
        let samples = [
            QuadTileId {
                scheme: TileScheme::Geodetic,
                level: 1,
                x: 2,
                y: 0,
            },
            QuadTileId {
                scheme: TileScheme::Geodetic,
                level: 3,
                x: 0,
                y: 7,
            },
            QuadTileId {
                scheme: TileScheme::WebMercator,
                level: 2,
                x: 1,
                y: 2,
            },
            QuadTileId {
                scheme: TileScheme::PolarCap { north: true },
                level: 0,
                x: 0,
                y: 0,
            },
            QuadTileId {
                scheme: TileScheme::PolarCap { north: false },
                level: 0,
                x: 0,
                y: 0,
            },
            QuadTileId {
                scheme: TileScheme::PolarCap { north: true },
                level: 2,
                x: 2,
                y: 1,
            },
        ];

        for id in samples {
            let tile = describe_tile(id, &cfg).expect("valid tile");
            let verts = build_tile_vertices(&tile, DEFAULT_GRID_CELLS);
            let vcount = verts.len() as u32;
            assert!(shared.indices.iter().all(|&i| i < vcount));

            // Every surface triangle must face outward (dot with vertex normal > 0).
            for tri in shared.indices[shared.surface.start as usize..shared.surface.end as usize]
                .chunks_exact(3)
            {
                let v0 = verts[tri[0] as usize];
                let v1 = verts[tri[1] as usize];
                let v2 = verts[tri[2] as usize];
                let p0 = Vec3::from(v0.rel_pos);
                let p1 = Vec3::from(v1.rel_pos);
                let p2 = Vec3::from(v2.rel_pos);
                let face = (p1 - p0).cross(p2 - p0);
                let normal =
                    (Vec3::from(v0.normal) + Vec3::from(v1.normal) + Vec3::from(v2.normal))
                        .normalize();
                assert!(
                    face.dot(normal) > 0.0,
                    "inverted surface triangle on {:?}: dot={}",
                    id,
                    face.dot(normal)
                );
            }

            // Every skirt triangle must face laterally outward away from the
            // tile's local RTC origin (dot with tangential edge offset > 0).
            for tri in shared.indices[shared.surface.end as usize..shared.full.end as usize]
                .chunks_exact(3)
            {
                let p0 = Vec3::from(verts[tri[0] as usize].rel_pos);
                let p1 = Vec3::from(verts[tri[1] as usize].rel_pos);
                let p2 = Vec3::from(verts[tri[2] as usize].rel_pos);
                let face = (p1 - p0).cross(p2 - p0);
                let centroid = (p0 + p1 + p2) / 3.0;
                assert!(
                    face.dot(centroid) > 0.0,
                    "inverted skirt triangle on {:?}: dot={}",
                    id,
                    face.dot(centroid)
                );
            }
        }
    }

    #[test]
    fn per_tile_rtc_preserves_sub_millimetre_precision_at_2m_zoom() {
        let cfg = QuadtreeConfig::bundled_geodetic();
        // z = 16 tile containing (47.2357 N, 39.7015 E).
        let level = 16u8;
        let (cols, rows) = TileScheme::Geodetic.grid_dims(level).expect("dims");
        let x = (((39.7015 + 180.0) / 360.0) * f64::from(cols)).floor() as u32;
        let y = (((90.0 - 47.2357) / 180.0) * f64::from(rows)).floor() as u32;
        let tile = describe_tile(
            QuadTileId {
                scheme: TileScheme::Geodetic,
                level,
                x,
                y,
            },
            &cfg,
        )
        .expect("z=16 tile");
        let verts = build_tile_vertices(&tile, DEFAULT_GRID_CELLS);

        // Reconstruct two adjacent surface vertices via RTC and compare against f64 ECEF.
        let v0 = verts[0];
        let v1 = verts[1];
        let step_rtc_m = f64::from((Vec3::from(v1.rel_pos) - Vec3::from(v0.rel_pos)).length());
        let (lat0, lon0, _, _, _, _) = sample_tile_surface(tile.id, 0.0, 0.0).expect("sample 0");
        let (lat1, lon1, _, _, _, _) =
            sample_tile_surface(tile.id, 1.0 / f64::from(DEFAULT_GRID_CELLS), 0.0)
                .expect("sample 1");
        let step_exact_m = geodesy::world(Geodetic::surface(lat0, lon0))
            .distance(geodesy::world(Geodetic::surface(lat1, lon1)));
        assert!(
            (step_rtc_m - step_exact_m).abs() < 1e-4,
            "RTC vertex step drifted: rtc={step_rtc_m} m vs exact={step_exact_m} m"
        );
    }

    #[test]
    fn skirts_drop_inward_by_at_least_four_times_cell_sag() {
        let cfg = QuadtreeConfig::bundled_geodetic();
        let tile = describe_tile(
            QuadTileId {
                scheme: TileScheme::Geodetic,
                level: 2,
                x: 3,
                y: 1,
            },
            &cfg,
        )
        .expect("tile");
        let verts = build_tile_vertices(&tile, DEFAULT_GRID_CELLS);
        let stride = (DEFAULT_GRID_CELLS + 1) as usize;
        let north_skirt_start = stride * stride;

        for col in 0..stride {
            let surf = verts[col];
            let skirt = verts[north_skirt_start + col];
            let delta = Vec3::from(skirt.rel_pos) - Vec3::from(surf.rel_pos);
            let inward = -delta.dot(Vec3::from(surf.normal));
            let tol = (tile.skirt_depth_m * 1e-4).max(1e-3);
            assert!(
                (f64::from(inward) - tile.skirt_depth_m).abs() < tol,
                "skirt drop {inward} m != expected {} m (tol {tol})",
                tile.skirt_depth_m
            );
            assert!(f64::from(inward) >= (4.0 * tile.sag_cell_diag_m).max(0.001) - tol);
        }
    }
}
