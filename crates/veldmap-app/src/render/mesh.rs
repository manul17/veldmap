//! Per-tile CPU `f64` RTC WGS84 quadtree geometry with per-cell sag-scaled
//! skirts and a single shared static index buffer.
//!
//! Both the globe surface (`BodyPipelines::{ellipsoid, textured}`) and the
//! post-overlay procedural `fwidth()` graticule (`BodyPipelines::grid`) consume
//! the exact same per-tile vertex buffer and shared static index buffer (`full`
//! range for body + skirts, `surface` range for graticule).

use std::ops::Range;

use bytemuck::{Pod, Zeroable};
use veldmap_core::geo::WGS84;
use veldmap_core::glam::Vec3;
use veldmap_core::tile::{
    DEFAULT_GRID_CELLS, PlanetaryScheme, QuadTileId, TileSurfaceSample, VisibleTile,
    sample_tile_surface,
};
use wgpu::util::DeviceExt;

use super::graticule::TileGridAnchor;
use super::mesh_cache::{DEFAULT_MESH_CACHE_BYTES, ElasticMeshCache, push_grid_quad_indices};
use super::snapshot::DrawTile;

pub(super) struct CachedTileMesh {
    pub(super) buffer: wgpu::Buffer,
    pub(super) grid_anchor: TileGridAnchor,
}

/// Bounded LRU cache of per-tile CPU `f64` RTC vertex buffers paired with the
/// shared static quadtree index buffer.
pub(super) struct TileMeshCache {
    grid_cells: u32,
    surface_indices: Range<u32>,
    full_indices: Range<u32>,
    index_buffer: wgpu::Buffer,
    meshes: ElasticMeshCache<QuadTileId, CachedTileMesh>,
}

fn create_shared_index_buffer(device: &wgpu::Device, indices: &[u32]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("quadtree shared index buffer"),
        contents: bytemuck::cast_slice(indices),
        usage: wgpu::BufferUsages::INDEX,
    })
}

impl TileMeshCache {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        let grid_cells = DEFAULT_GRID_CELLS;
        let shared = SharedTileIndices::build(grid_cells);
        let index_buffer = create_shared_index_buffer(device, &shared.indices);
        Self {
            grid_cells,
            surface_indices: shared.surface,
            full_indices: shared.full,
            index_buffer,
            meshes: ElasticMeshCache::new(DEFAULT_MESH_CACHE_BYTES),
        }
    }

    pub(super) fn index_buffer(&self) -> &wgpu::Buffer {
        &self.index_buffer
    }

    pub(super) fn surface_indices(&self) -> Range<u32> {
        self.surface_indices.clone()
    }

    pub(super) fn full_indices(&self) -> Range<u32> {
        self.full_indices.clone()
    }

    pub(super) fn peek(&self, id: &QuadTileId) -> Option<&CachedTileMesh> {
        self.meshes.peek(id)
    }

    fn ensure_grid_cells(&mut self, device: &wgpu::Device, grid_cells: u32) {
        let cells = grid_cells.max(1);
        if cells == self.grid_cells {
            return;
        }
        let shared = SharedTileIndices::build(cells);
        self.grid_cells = cells;
        self.surface_indices = shared.surface;
        self.full_indices = shared.full;
        self.index_buffer = create_shared_index_buffer(device, &shared.indices);
        self.meshes.clear_and_reset_cap();
    }

    /// Ensure every tile in `tiles` has an uploaded vertex buffer without
    /// evicting earlier tiles of the current frame, while relaxing the cache
    /// ceiling back toward [`DEFAULT_MESH_CACHE_BYTES`] once demand subsides.
    pub(super) fn prepare_for_tiles(
        &mut self,
        device: &wgpu::Device,
        grid_cells: u32,
        tiles: &[DrawTile],
    ) {
        self.ensure_grid_cells(device, grid_cells);
        let stride = (self.grid_cells + 1) as usize;
        let single_tile_bytes = (stride * stride + 4 * stride) * std::mem::size_of::<TileVertex>();
        let needed_bytes = tiles.len().saturating_mul(single_tile_bytes);

        self.meshes
            .begin_frame(needed_bytes, tiles.iter().map(|draw| &draw.visible.id));

        for draw in tiles {
            let id = draw.visible.id;
            if self.meshes.peek(&id).is_some() {
                continue;
            }
            let (verts, grid_anchor) = build_tile_mesh_data(&draw.visible, self.grid_cells);
            let byte_size = std::mem::size_of_val(verts.as_slice());
            let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("quadtree tile RTC vertex buffer"),
                contents: bytemuck::cast_slice(&verts),
                usage: wgpu::BufferUsages::VERTEX,
            });
            self.meshes.insert(
                id,
                CachedTileMesh {
                    buffer,
                    grid_anchor,
                },
                byte_size,
            );
        }

        self.meshes.end_frame(needed_bytes);
    }
}

/// Vertex layout shared by textured/untextured quadtree tiles and the
/// post-overlay procedural graticule.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(super) struct TileVertex {
    /// Surface (or skirt) position relative to the tile's CPU `f64` RTC origin
    /// (`(P_ecef_m - origin_ecef_m) as f32`, in metres).
    rel_pos: [f32; 3],
    /// Outward unit WGS84 ellipsoidal normal.
    normal: [f32; 3],
    /// Tile-local UV coordinates in `[0, 1]^2` (transformed by per-draw
    /// `uv_rect` when sampling a coarser ancestor texture).
    uv: [f32; 2],
    /// Tile-relative graticule coordinates computed in `f64` before casting to
    /// `f32`:
    /// - `Geodetic` / `WebMercator`: `(lon_deg - anchor_lon, lat_deg - anchor_lat)`
    /// - `PolarCap`: `(dx - anchor_dx, dy - anchor_dy)` in unit-disk coordinates
    grid_rel: [f32; 2],
}

const _: () = {
    assert!(std::mem::size_of::<TileVertex>() == 40);
};

impl TileVertex {
    pub(super) const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
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
struct SharedTileIndices {
    indices: Vec<u32>,
    /// Index range covering the `grid_cells x grid_cells` surface quads only
    /// (used by the post-overlay `grid_pipeline` so skirts never draw graticule
    /// artifacts).
    surface: Range<u32>,
    /// Full index range covering the surface quads plus all four downward
    /// skirts (used by `body_ellipsoid_pipeline` and `body_textured_pipeline`).
    full: Range<u32>,
}

impl SharedTileIndices {
    /// Build the shared static index buffer for a `grid_cells x grid_cells`
    /// tile with four perimeter skirt strips.
    fn build(grid_cells: u32) -> Self {
        let n = grid_cells.max(1);
        let stride = n + 1;
        let surface_verts = stride * stride;

        let surface_idx_count = (n * n * 6) as usize;
        let skirt_idx_count = (4 * n * 6) as usize;
        let mut indices = Vec::with_capacity(surface_idx_count + skirt_idx_count);

        // 1. Surface quads (CCW outward when viewed from outside the ellipsoid).
        push_grid_quad_indices(&mut indices, n, false);
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
/// together with its cached `f64` [`TileGridAnchor`] so
/// `GlobeRenderer::upload_uniforms` does not re-sample
/// `sample_tile_surface(id, 0.5, 0.5)`.
fn build_tile_mesh_data(tile: &VisibleTile, grid_cells: u32) -> (Vec<TileVertex>, TileGridAnchor) {
    let n = grid_cells.max(1);
    let stride = (n + 1) as usize;
    let total_verts = stride * stride + 4 * stride;
    let mut vertices = Vec::with_capacity(total_verts);

    let anchor = TileGridAnchor::for_tile(tile.id);
    let is_polar = matches!(tile.id.scheme, PlanetaryScheme::PolarCap { .. });
    let origin = tile.origin_ecef_m;
    let n_f = f64::from(n);

    // 1. Surface grid `(n + 1) x (n + 1)`.
    for row in 0..=n {
        let t = f64::from(row) / n_f;
        for col in 0..=n {
            let s = f64::from(col) / n_f;
            let TileSurfaceSample {
                lat_deg: lat,
                lon_deg: lon,
                u_tex,
                v_tex,
                polar_dx: dx,
                polar_dy: dy,
            } = sample_tile_surface(tile.id, s, t).unwrap_or(TileSurfaceSample {
                lat_deg: 0.0,
                lon_deg: 0.0,
                u_tex: s,
                v_tex: t,
                polar_dx: 0.0,
                polar_dy: 0.0,
            });
            let (pos_ecef, unit_normal) = WGS84.surface_ecef_and_normal(lat, lon);
            let rel_pos = (pos_ecef - origin).as_vec3().to_array();
            let normal = unit_normal.as_vec3().to_array();
            let grid_rel = if is_polar {
                [(dx - anchor.dx0) as f32, (dy - anchor.dy0) as f32]
            } else {
                [(lon - anchor.lon0) as f32, (lat - anchor.lat0) as f32]
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
        let pos = Vec3::from_array(v.rel_pos);
        let normal = Vec3::from_array(v.normal);
        let raw = pos - Vec3::from_array(out[inner_idx].rel_pos);
        let tangent = raw - normal * raw.dot(normal);
        let t_len = tangent.length();
        let flare = if t_len > 1e-9 {
            tangent * (flare_m / t_len)
        } else {
            Vec3::ZERO
        };
        out.push(TileVertex {
            rel_pos: (pos - normal * drop_m + flare).to_array(),
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
