//! Warp-mesh vertex layout, shared per-density index buffers, and bounded LRU
//! vertex-buffer cache ([`OverlayMeshPool`]) for direct and `BandMath` raster
//! overlay passes.

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use veldmap_core::glam::DVec3;
use veldmap_core::tile::{MAX_WARP_SEGMENTS, build_warp_patch};
use wgpu::util::DeviceExt;

use super::super::mesh_cache::{
    DEFAULT_MESH_CACHE_BYTES, ElasticMeshCache, push_grid_quad_indices,
};
use super::super::snapshot::{DrawOverlayPatch, OverlayMeshKey, WarpPatchSetRef};

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(in crate::render) struct OverlayVertex {
    rel_pos: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
}

const _: () = {
    assert!(std::mem::size_of::<OverlayVertex>() == 32);
};

impl OverlayVertex {
    pub(in crate::render) const LAYOUT: wgpu::VertexBufferLayout<'static> =
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![
                0 => Float32x3,
                1 => Float32x3,
                2 => Float32x2,
            ],
        };
}

pub(super) fn clamp_warp_segments(segments: u32) -> u32 {
    segments.clamp(1, MAX_WARP_SEGMENTS)
}

pub(super) struct CachedOverlayMesh {
    pub(super) buffer: wgpu::Buffer,
    pub(super) origin_ecef_m: DVec3,
    pub(super) segments: u32,
    pub(super) flip_winding: bool,
}

/// Manages shared per-density index buffers and the elastic LRU cache of
/// per-cell RTC warp-mesh vertex buffers.
pub(super) struct OverlayMeshPool {
    index_buffers: HashMap<(u32, bool), (wgpu::Buffer, u32)>,
    meshes: ElasticMeshCache<OverlayMeshKey, CachedOverlayMesh>,
}

impl OverlayMeshPool {
    pub(super) fn new() -> Self {
        Self {
            index_buffers: HashMap::new(),
            meshes: ElasticMeshCache::new(DEFAULT_MESH_CACHE_BYTES),
        }
    }

    pub(super) fn relax_to_default(&mut self) {
        self.meshes.relax_to_default();
    }

    pub(super) fn begin_frame<'a>(
        &mut self,
        needed_bytes: usize,
        live_keys: impl Iterator<Item = &'a OverlayMeshKey>,
    ) {
        self.meshes.begin_frame(needed_bytes, live_keys);
    }

    pub(super) fn end_frame(&mut self, needed_bytes: usize) {
        self.meshes.end_frame(needed_bytes);
    }

    pub(super) fn peek(&self, key: &OverlayMeshKey) -> Option<&CachedOverlayMesh> {
        self.meshes.peek(key)
    }

    pub(super) fn index_buffer(&self, key: &(u32, bool)) -> Option<&(wgpu::Buffer, u32)> {
        self.index_buffers.get(key)
    }

    /// Accumulate vertex-buffer byte demand and live mesh keys for one warp
    /// patch set.
    pub(super) fn account_patch_set(
        set: WarpPatchSetRef<'_>,
        needed_bytes: &mut usize,
        live_keys: &mut Vec<OverlayMeshKey>,
    ) {
        let side = (clamp_warp_segments(set.segments) + 1) as usize;
        *needed_bytes = needed_bytes.saturating_add(
            set.patches
                .len()
                .saturating_mul(side * side * std::mem::size_of::<OverlayVertex>()),
        );
        live_keys.extend(set.patches.iter().map(|p| p.mesh_key));
    }

    fn ensure_index_buffer(&mut self, device: &wgpu::Device, segments: u32, flip_winding: bool) {
        let segs = clamp_warp_segments(segments);
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

    pub(super) fn ensure_patch_mesh(
        &mut self,
        device: &wgpu::Device,
        set: WarpPatchSetRef<'_>,
        patch: &DrawOverlayPatch,
    ) {
        let segs = clamp_warp_segments(set.segments);
        if self
            .meshes
            .peek(&patch.mesh_key)
            .is_some_and(|existing| existing.segments == segs)
        {
            return;
        }
        let warp = build_warp_patch(
            set.frame,
            set.image_width,
            set.image_height,
            patch.cell_rect,
            segs,
        );
        self.ensure_index_buffer(device, warp.segments, warp.flip_winding);
        let Some(vertices) =
            build_overlay_vertices(warp.segments, &warp.rel_positions, &warp.normals)
        else {
            return;
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

    /// Bind index buffer (deduplicated), vertex buffer, dynamic uniform slot,
    /// and issue the indexed draw for `mesh`.
    pub(super) fn draw_indexed_patch<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        mesh: &'a CachedOverlayMesh,
        last_index_key: &mut Option<(u32, bool)>,
        uniform_bg: &'a wgpu::BindGroup,
        dyn_offset: u32,
    ) -> bool {
        let idx_key = (mesh.segments, mesh.flip_winding);
        let Some((idx_buf, idx_count)) = self.index_buffer(&idx_key) else {
            return false;
        };
        if *last_index_key != Some(idx_key) {
            pass.set_index_buffer(idx_buf.slice(..), wgpu::IndexFormat::Uint32);
            *last_index_key = Some(idx_key);
        }
        pass.set_vertex_buffer(0, mesh.buffer.slice(..));
        pass.set_bind_group(1, uniform_bg, &[dyn_offset]);
        pass.draw_indexed(0..*idx_count, 0, 0..1);
        true
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
