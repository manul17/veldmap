//! Depth-less post-body warp-mesh overlay renderer extracted from `globe.rs`.
//!
//! Renders per-cell CPU `f64` RTC raster overlay patches in painter's order
//! inside Pass 2 (`depth_stencil_attachment: None`, `LoadOp::Load`) using the
//! frame-wide [`DynamicUniformArena`] and [`TileBindGroupCache`].

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use veldmap_core::glam::DVec3;
use veldmap_core::tile::{PixelFormat, SourceId, TileCache};
use wgpu::util::DeviceExt;

use super::OVERLAY_WGSL;
use super::pipelines::{GlobePipelines, TileBindGroupCache};
use super::snapshot::DrawSnapshot;
use super::uniforms::{DynamicUniformArena, OverlayPatchUniform};

/// Default VRAM budget for cached per-cell overlay warp-mesh vertex buffers
/// (`16 MiB`).
const DEFAULT_OVERLAY_MESH_CACHE_BYTES: usize = 16 * 1024 * 1024;

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
struct OverlayVertex {
    rel_pos: [f32; 3],
    uv: [f32; 2],
}

const _: () = {
    assert!(std::mem::size_of::<OverlayVertex>() == 20);
};

impl OverlayVertex {
    const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &wgpu::vertex_attr_array![
            0 => Float32x3,
            1 => Float32x2,
        ],
    };
}

struct CachedOverlayMesh {
    buffer: wgpu::Buffer,
    segments: u32,
    flip_winding: bool,
}

/// Manages the depth-less overlay pipeline, shared per-density index buffers,
/// and bounded LRU cache of uploaded per-cell RTC warp-mesh vertex buffers.
pub(super) struct OverlayRenderer {
    pipeline: wgpu::RenderPipeline,
    index_buffers: HashMap<(u32, bool), (wgpu::Buffer, u32)>,
    meshes: TileCache<OverlayMeshKey, CachedOverlayMesh>,
}

impl OverlayRenderer {
    pub(super) fn new(device: &wgpu::Device, pipelines: &GlobePipelines) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("overlay warp-mesh wgsl"),
            source: wgpu::ShaderSource::Wgsl(OVERLAY_WGSL.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("overlay warp-mesh pipeline layout"),
            bind_group_layouts: &[
                Some(&pipelines.camera_layout),
                Some(&pipelines.overlay_uniform_layout),
                Some(&pipelines.tile_texture_layout),
            ],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("overlay warp-mesh pipeline (depth-less)"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Some(OverlayVertex::LAYOUT)],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_overlay"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                front_face: wgpu::FrontFace::Ccw,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            pipeline,
            index_buffers: HashMap::new(),
            meshes: TileCache::new(DEFAULT_OVERLAY_MESH_CACHE_BYTES),
        }
    }

    fn ensure_index_buffer(&mut self, device: &wgpu::Device, segments: u32, flip_winding: bool) {
        let segs = segments.max(1);
        if self.index_buffers.contains_key(&(segs, flip_winding)) {
            return;
        }
        let stride = segs + 1;
        let mut indices = Vec::with_capacity((segs * segs * 6) as usize);
        for row in 0..segs {
            for col in 0..segs {
                let top = row * stride + col;
                let bottom = top + stride;
                if flip_winding {
                    indices.extend_from_slice(&[top, top + 1, bottom]);
                    indices.extend_from_slice(&[top + 1, bottom + 1, bottom]);
                } else {
                    indices.extend_from_slice(&[top, bottom, top + 1]);
                    indices.extend_from_slice(&[top + 1, bottom, bottom + 1]);
                }
            }
        }
        let count = indices.len() as u32;
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("overlay warp-mesh shared index buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        self.index_buffers
            .insert((segs, flip_winding), (buffer, count));
    }

    /// Ensure vertex buffers, shared density index buffers, and texture bind
    /// groups exist for every overlay patch in `snapshot`.
    pub(super) fn prepare_resources(
        &mut self,
        device: &wgpu::Device,
        pipelines: &GlobePipelines,
        bind_groups: &mut TileBindGroupCache,
        snapshot: &DrawSnapshot,
    ) {
        if snapshot.overlays.is_empty() {
            if self.meshes.max_bytes() > DEFAULT_OVERLAY_MESH_CACHE_BYTES {
                let _ = self.meshes.set_max_bytes(DEFAULT_OVERLAY_MESH_CACHE_BYTES);
            }
            return;
        }

        let needed_bytes: usize = snapshot
            .overlays
            .iter()
            .flat_map(|layer| &layer.patches)
            .map(|patch| {
                let side = (patch.segments.max(1) + 1) as usize;
                side * side * std::mem::size_of::<OverlayVertex>()
            })
            .sum();
        let active_floor = needed_bytes.max(DEFAULT_OVERLAY_MESH_CACHE_BYTES);
        if active_floor > self.meshes.max_bytes() {
            let _ = self.meshes.set_max_bytes(active_floor);
        }

        for layer in &snapshot.overlays {
            for patch in &layer.patches {
                self.meshes.touch(&patch.mesh_key);
            }
        }

        for layer in &snapshot.overlays {
            for patch in &layer.patches {
                bind_groups.ensure(device, pipelines, &patch.texture);
                let segs = patch.segments.max(1);
                self.ensure_index_buffer(device, segs, patch.flip_winding);

                let already_valid = self
                    .meshes
                    .peek(&patch.mesh_key)
                    .is_some_and(|m| m.segments == segs && m.flip_winding == patch.flip_winding);
                if already_valid {
                    continue;
                }

                let vertices = build_overlay_vertices(segs, &patch.rel_positions);
                let byte_size = std::mem::size_of_val(vertices.as_slice());
                let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("overlay warp-mesh RTC vertex buffer"),
                    contents: bytemuck::cast_slice(&vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                });
                let _ = self.meshes.insert(
                    patch.mesh_key,
                    CachedOverlayMesh {
                        buffer,
                        segments: segs,
                        flip_winding: patch.flip_winding,
                    },
                    byte_size,
                );
            }
        }

        let target_cap = needed_bytes.max(DEFAULT_OVERLAY_MESH_CACHE_BYTES);
        if target_cap < self.meshes.max_bytes() {
            let _ = self.meshes.set_max_bytes(target_cap);
        }
    }

    /// Push per-patch uniforms for every overlay patch into the frame-wide
    /// `arena`, returning their aligned dynamic byte offsets in draw order.
    pub(super) fn push_uniforms(
        &self,
        arena: &mut DynamicUniformArena,
        snapshot: &DrawSnapshot,
        eye: DVec3,
        out_offsets: &mut Vec<u32>,
    ) {
        out_offsets.clear();
        for layer in &snapshot.overlays {
            let opacity = layer.opacity.clamp(0.0, 1.0);
            for patch in &layer.patches {
                let offset_eye = (patch.origin_ecef_m - eye).as_vec3().to_array();
                let u = OverlayPatchUniform {
                    offset_eye,
                    opacity,
                    uv_rect: patch.uv_rect,
                };
                out_offsets.push(arena.push(&u));
            }
        }
    }

    /// Record all ordered overlay layers bottom-to-top inside Pass 2
    /// (`depth_stencil_attachment: None`).
    pub(super) fn record_pass<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        arena: &'a DynamicUniformArena,
        bind_groups: &'a TileBindGroupCache,
        snapshot: &'a DrawSnapshot,
        offsets: &[u32],
    ) {
        if snapshot.overlays.is_empty() {
            return;
        }
        pass.set_pipeline(&self.pipeline);
        let mut last_index_key: Option<(u32, bool)> = None;
        let mut last_tex_id: Option<u64> = None;
        let mut offset_idx = 0usize;

        for layer in &snapshot.overlays {
            for patch in &layer.patches {
                let dyn_offset = offsets[offset_idx];
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

fn build_overlay_vertices(segments: u32, rel_positions: &[[f32; 3]]) -> Vec<OverlayVertex> {
    let side = (segments + 1) as usize;
    let total = side * side;
    let mut vertices = Vec::with_capacity(total);
    let inv = 1.0 / (segments as f32);
    for row in 0..side {
        let v = (row as f32) * inv;
        for col in 0..side {
            let u = (col as f32) * inv;
            let idx = row * side + col;
            let rel_pos = rel_positions.get(idx).copied().unwrap_or([0.0; 3]);
            vertices.push(OverlayVertex {
                rel_pos,
                uv: [u, v],
            });
        }
    }
    vertices
}
