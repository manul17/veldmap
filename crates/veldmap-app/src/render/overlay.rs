//! Depth-less post-body warp-mesh overlay renderer.
//!
//! Renders per-cell CPU `f64` RTC raster overlay patches in painter's order
//! inside Pass 2 (`depth_stencil_attachment: None`, `LoadOp::Load`) using the
//! frame-wide [`DynamicUniformArena`] and [`TileBindGroupCache`].

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use veldmap_core::glam::DVec3;
use veldmap_core::tile::{MAX_WARP_SEGMENTS, PixelFormat, SourceId, build_warp_patch};
use wgpu::util::DeviceExt;

use super::bind_groups::TileBindGroupCache;
use super::mesh_cache::{DEFAULT_MESH_CACHE_BYTES, ElasticMeshCache, push_grid_quad_indices};
use super::pipelines::GlobePipelines;
use super::snapshot::DrawSnapshot;
use super::uniforms::{DynamicUniformArena, OverlayPatchUniform};

/// Cache identity for an uploaded overlay cell warp-mesh vertex buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OverlayMeshKey {
    pub source: SourceId,
    pub relaid: u64,
    pub level: u8,
    pub x: u32,
    pub y: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(super) struct OverlayVertex {
    rel_pos: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
}

const _: () = {
    assert!(std::mem::size_of::<OverlayVertex>() == 32);
};

impl OverlayVertex {
    pub(super) const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &wgpu::vertex_attr_array![
            0 => Float32x3,
            1 => Float32x3,
            2 => Float32x2,
        ],
    };
}

struct CachedOverlayMesh {
    buffer: wgpu::Buffer,
    origin_ecef_m: DVec3,
    segments: u32,
    flip_winding: bool,
}

/// Manages shared per-density index buffers, bounded LRU cache of lazily built
/// per-cell RTC warp-mesh vertex buffers, and per-frame dynamic uniform
/// offsets for draped raster overlays.
pub(super) struct OverlayRenderer {
    index_buffers: HashMap<(u32, bool), (wgpu::Buffer, u32)>,
    meshes: ElasticMeshCache<OverlayMeshKey, CachedOverlayMesh>,
    uniform_offsets: Vec<u32>,
}

impl Default for OverlayRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl OverlayRenderer {
    pub(super) fn new() -> Self {
        Self {
            index_buffers: HashMap::new(),
            meshes: ElasticMeshCache::new(DEFAULT_MESH_CACHE_BYTES),
            uniform_offsets: Vec::new(),
        }
    }

    fn ensure_index_buffer(&mut self, device: &wgpu::Device, segments: u32, flip_winding: bool) {
        let segs = segments.clamp(1, MAX_WARP_SEGMENTS);
        if self.index_buffers.contains_key(&(segs, flip_winding)) {
            return;
        }
        let mut indices = Vec::with_capacity((segs * segs * 6) as usize);
        push_grid_quad_indices(&mut indices, segs, flip_winding);
        let count = indices.len() as u32;
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("overlay warp-mesh shared index buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        self.index_buffers
            .insert((segs, flip_winding), (buffer, count));
    }

    /// Ensure vertex buffers (built lazily on cache miss), shared density index
    /// buffers, and texture bind groups exist for every overlay patch in
    /// `snapshot`. Releases cached overlay vertex buffers when no overlays are
    /// present so closed layers never linger in VRAM.
    pub(super) fn prepare_resources(
        &mut self,
        device: &wgpu::Device,
        pipelines: &GlobePipelines,
        bind_groups: &mut TileBindGroupCache,
        snapshot: &DrawSnapshot,
    ) {
        if snapshot.overlays.is_empty() {
            self.meshes.clear_and_reset_cap();
            return;
        }

        let needed_bytes: usize = snapshot
            .overlays
            .iter()
            .map(|layer| {
                let side = (layer.segments.clamp(1, MAX_WARP_SEGMENTS) + 1) as usize;
                layer
                    .patches
                    .len()
                    .saturating_mul(side * side * std::mem::size_of::<OverlayVertex>())
            })
            .sum();
        self.meshes.begin_frame(
            needed_bytes,
            snapshot
                .overlays
                .iter()
                .flat_map(|layer| layer.patches.iter().map(|patch| &patch.mesh_key)),
        );

        for layer in &snapshot.overlays {
            let segs = layer.segments.clamp(1, MAX_WARP_SEGMENTS);
            for patch in &layer.patches {
                bind_groups.ensure(device, pipelines, &patch.texture);

                if let Some(existing) = self.meshes.peek(&patch.mesh_key)
                    && existing.segments == segs
                {
                    let flip = existing.flip_winding;
                    self.ensure_index_buffer(device, segs, flip);
                    continue;
                }

                let warp = build_warp_patch(
                    &layer.frame,
                    layer.image_width,
                    layer.image_height,
                    patch.cell_rect,
                    segs,
                );
                self.ensure_index_buffer(device, warp.segments, warp.flip_winding);
                let Some(vertices) =
                    build_overlay_vertices(warp.segments, &warp.rel_positions, &warp.normals)
                else {
                    continue;
                };
                let byte_size = std::mem::size_of_val(vertices.as_slice());
                let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("overlay warp-mesh RTC vertex buffer"),
                    contents: bytemuck::cast_slice(&vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                });
                self.meshes.insert(
                    patch.mesh_key,
                    CachedOverlayMesh {
                        buffer,
                        origin_ecef_m: warp.origin_ecef_m,
                        segments: warp.segments,
                        flip_winding: warp.flip_winding,
                    },
                    byte_size,
                );
            }
        }

        self.meshes.end_frame(needed_bytes);
    }

    /// Push per-patch uniforms for every overlay patch into the frame-wide
    /// `arena`, recording their aligned dynamic byte offsets in draw order.
    pub(super) fn push_uniforms(
        &mut self,
        arena: &mut DynamicUniformArena,
        snapshot: &DrawSnapshot,
        eye: DVec3,
    ) {
        self.uniform_offsets.clear();
        for layer in &snapshot.overlays {
            let opacity = layer.opacity.clamp(0.0, 1.0);
            for patch in &layer.patches {
                let origin = self
                    .meshes
                    .peek(&patch.mesh_key)
                    .map_or(DVec3::ZERO, |m| m.origin_ecef_m);
                let offset_eye = (origin - eye).as_vec3().to_array();
                let u = OverlayPatchUniform {
                    offset_eye,
                    opacity,
                    uv_rect: patch.uv_rect,
                };
                self.uniform_offsets.push(arena.push(&u));
            }
        }
    }

    /// Record all ordered overlay layers bottom-to-top inside Pass 2
    /// (`depth_stencil_attachment: None`).
    pub(super) fn record_pass<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        pipeline: &'a wgpu::RenderPipeline,
        arena: &'a DynamicUniformArena,
        bind_groups: &'a TileBindGroupCache,
        snapshot: &'a DrawSnapshot,
    ) {
        if snapshot.overlays.is_empty() {
            return;
        }
        pass.set_pipeline(pipeline);
        let mut last_index_key: Option<(u32, bool)> = None;
        let mut last_tex_id: Option<u64> = None;
        let mut offset_idx = 0usize;

        for layer in &snapshot.overlays {
            for patch in &layer.patches {
                let dyn_offset = self.uniform_offsets[offset_idx];
                offset_idx += 1;

                let Some(mesh) = self.meshes.peek(&patch.mesh_key) else {
                    continue;
                };
                let Some(tex_bg) = bind_groups.get(&patch.texture, PixelFormat::Rgba8UnormSrgb)
                else {
                    continue;
                };
                let idx_key = (mesh.segments, mesh.flip_winding);
                let Some((idx_buf, idx_count)) = self.index_buffers.get(&idx_key) else {
                    continue;
                };
                if last_index_key != Some(idx_key) {
                    pass.set_index_buffer(idx_buf.slice(..), wgpu::IndexFormat::Uint32);
                    last_index_key = Some(idx_key);
                }
                if last_tex_id != Some(patch.texture.tile_id) {
                    pass.set_bind_group(2, tex_bg, &[]);
                    last_tex_id = Some(patch.texture.tile_id);
                }
                pass.set_vertex_buffer(0, mesh.buffer.slice(..));
                pass.set_bind_group(1, arena.overlay_bind_group(), &[dyn_offset]);
                pass.draw_indexed(0..*idx_count, 0, 0..1);
            }
        }
    }
}

fn build_overlay_vertices(
    segments: u32,
    rel_positions: &[[f32; 3]],
    normals: &[[f32; 3]],
) -> Option<Vec<OverlayVertex>> {
    let side = (segments + 1) as usize;
    let total = side.checked_mul(side)?;
    if rel_positions.len() < total || normals.len() < total {
        return None;
    }
    let mut vertices = Vec::with_capacity(total);
    let inv = 1.0 / (segments as f32);
    for row in 0..side {
        let v = (row as f32) * inv;
        for col in 0..side {
            let u = (col as f32) * inv;
            let idx = row * side + col;
            vertices.push(OverlayVertex {
                rel_pos: rel_positions[idx],
                normal: normals[idx],
                uv: [u, v],
            });
        }
    }
    Some(vertices)
}
