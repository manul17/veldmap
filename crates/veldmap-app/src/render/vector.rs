//! Depth-less post-body GPU vector renderer for GeoJSON overlays and
//! interactive geodesic measurement geometries.

use std::collections::{HashMap, HashSet};

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use super::snapshot::DrawSnapshot;
use super::uniforms::{DynamicUniformArena, VectorUniform};

/// Compact 24-byte GPU vertex matching `shaders/vector.wgsl` (`@location(0..1)`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct GpuVectorVertex {
    pub(super) rel_pos: [f32; 3],
    pub(super) normal: [f32; 3],
}

const _: () = {
    assert!(std::mem::size_of::<GpuVectorVertex>() == 24);
};

impl GpuVectorVertex {
    const ATTRS: [wgpu::VertexAttribute; 2] = wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Float32x3,
    ];

    pub(super) const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &Self::ATTRS,
    };
}

struct CachedVectorBuffers {
    rev: u64,
    fill_vb: Option<wgpu::Buffer>,
    fill_ib: Option<wgpu::Buffer>,
    fill_index_count: u32,
    line_vb: Option<wgpu::Buffer>,
    line_vertex_count: u32,
    point_vb: Option<wgpu::Buffer>,
    point_vertex_count: u32,
    point_radius_m: f32,
}

struct VectorDrawOffsets {
    fill_offset: Option<u32>,
    line_offset: Option<u32>,
}

fn point_marker_radius_m(camera_height_m: f64) -> f32 {
    // Scale point markers with camera altitude (~0.9% of eye altitude) so
    // GeoJSON Point / MultiPoint features stay clearly visible from global
    // orbit down to centimetre drone inspection without crowding the map.
    (camera_height_m * 0.009).clamp(0.15, 250_000.0) as f32
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize3(v: [f32; 3]) -> Option<[f32; 3]> {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 1e-6 && len.is_finite() {
        Some([v[0] / len, v[1] / len, v[2] / len])
    } else {
        None
    }
}

fn add_scaled3(a: [f32; 3], b: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] + b[0] * s, a[1] + b[1] * s, a[2] + b[2] * s]
}

fn build_point_diamond_vertices(
    points: &[veldmap_core::geo::VectorVertex],
    radius_m: f32,
) -> Vec<GpuVectorVertex> {
    let mut verts = Vec::with_capacity(points.len().saturating_mul(6));
    for pt in points {
        let Some(normal) = normalize3(pt.normal) else {
            continue;
        };
        let ref_axis = if normal[2].abs() < 0.9 {
            [0.0, 0.0, 1.0]
        } else {
            [1.0, 0.0, 0.0]
        };
        let Some(ex) = normalize3(cross3(ref_axis, normal)) else {
            continue;
        };
        let Some(ey) = normalize3(cross3(normal, ex)) else {
            continue;
        };
        // Slight outward lift (~0.5% of radius) prevents tangent corners from
        // dipping under curved globe facets at orbital altitudes.
        let base = add_scaled3(pt.rel_pos, normal, radius_m * 0.005);
        let p0 = add_scaled3(base, ex, radius_m);
        let p1 = add_scaled3(base, ey, radius_m);
        let p2 = add_scaled3(base, ex, -radius_m);
        let p3 = add_scaled3(base, ey, -radius_m);
        // Outward-facing CCW diamond (2 triangles).
        for rel_pos in [p0, p1, p2, p0, p2, p3] {
            verts.push(GpuVectorVertex { rel_pos, normal });
        }
    }
    verts
}

/// GPU buffer cache and Pass-2 recorder for draped vector layers.
pub(super) struct VectorRenderer {
    buffers: HashMap<u64, CachedVectorBuffers>,
    draw_offsets: Vec<VectorDrawOffsets>,
}

impl VectorRenderer {
    pub(super) fn new() -> Self {
        Self {
            buffers: HashMap::new(),
            draw_offsets: Vec::new(),
        }
    }

    pub(super) fn prepare_resources(&mut self, device: &wgpu::Device, snapshot: &DrawSnapshot) {
        if snapshot.vectors.is_empty() {
            self.buffers.clear();
            return;
        }
        let active_ids: HashSet<u64> = snapshot.vectors.iter().map(|v| v.id).collect();
        self.buffers.retain(|id, _| active_ids.contains(id));

        let marker_radius_m = point_marker_radius_m(snapshot.camera.height_m());

        for layer in &snapshot.vectors {
            let needs_full_upload = self
                .buffers
                .get(&layer.id)
                .is_none_or(|cached| cached.rev != layer.rev);
            if needs_full_upload {
                let (fill_vb, fill_ib, fill_index_count) = if !layer.mesh.fill_vertices.is_empty()
                    && !layer.mesh.fill_indices.is_empty()
                {
                    let verts: Vec<GpuVectorVertex> = layer
                        .mesh
                        .fill_vertices
                        .iter()
                        .map(|v| GpuVectorVertex {
                            rel_pos: v.rel_pos,
                            normal: v.normal,
                        })
                        .collect();
                    let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("vector fill vertex buffer"),
                        contents: bytemuck::cast_slice(&verts),
                        usage: wgpu::BufferUsages::VERTEX,
                    });
                    let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("vector fill index buffer"),
                        contents: bytemuck::cast_slice(&layer.mesh.fill_indices),
                        usage: wgpu::BufferUsages::INDEX,
                    });
                    (Some(vb), Some(ib), layer.mesh.fill_indices.len() as u32)
                } else {
                    (None, None, 0)
                };

                let (line_vb, line_vertex_count) = if !layer.mesh.line_vertices.is_empty() {
                    let verts: Vec<GpuVectorVertex> = layer
                        .mesh
                        .line_vertices
                        .iter()
                        .map(|v| GpuVectorVertex {
                            rel_pos: v.rel_pos,
                            normal: v.normal,
                        })
                        .collect();
                    let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("vector line vertex buffer"),
                        contents: bytemuck::cast_slice(&verts),
                        usage: wgpu::BufferUsages::VERTEX,
                    });
                    (Some(vb), verts.len() as u32)
                } else {
                    (None, 0)
                };

                let (point_vb, point_vertex_count, point_radius_m) =
                    if !layer.mesh.point_vertices.is_empty() {
                        let verts = build_point_diamond_vertices(
                            &layer.mesh.point_vertices,
                            marker_radius_m,
                        );
                        if verts.is_empty() {
                            (None, 0, 0.0)
                        } else {
                            let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("vector point marker vertex buffer"),
                                contents: bytemuck::cast_slice(&verts),
                                usage: wgpu::BufferUsages::VERTEX,
                            });
                            (Some(vb), verts.len() as u32, marker_radius_m)
                        }
                    } else {
                        (None, 0, 0.0)
                    };

                self.buffers.insert(
                    layer.id,
                    CachedVectorBuffers {
                        rev: layer.rev,
                        fill_vb,
                        fill_ib,
                        fill_index_count,
                        line_vb,
                        line_vertex_count,
                        point_vb,
                        point_vertex_count,
                        point_radius_m,
                    },
                );
            } else if !layer.mesh.point_vertices.is_empty()
                && let Some(cached) = self.buffers.get_mut(&layer.id)
                && (cached.point_radius_m <= 0.0
                    || (marker_radius_m / cached.point_radius_m - 1.0).abs() > 0.15)
            {
                let verts =
                    build_point_diamond_vertices(&layer.mesh.point_vertices, marker_radius_m);
                if verts.is_empty() {
                    cached.point_vb = None;
                    cached.point_vertex_count = 0;
                    cached.point_radius_m = 0.0;
                } else {
                    cached.point_vb = Some(device.create_buffer_init(
                        &wgpu::util::BufferInitDescriptor {
                            label: Some("vector point marker vertex buffer"),
                            contents: bytemuck::cast_slice(&verts),
                            usage: wgpu::BufferUsages::VERTEX,
                        },
                    ));
                    cached.point_vertex_count = verts.len() as u32;
                    cached.point_radius_m = marker_radius_m;
                }
            }
        }
    }

    pub(super) fn push_uniforms(
        &mut self,
        arena: &mut DynamicUniformArena,
        snapshot: &DrawSnapshot,
    ) {
        self.draw_offsets.clear();
        let eye = snapshot.camera.eye_ecef();
        for layer in &snapshot.vectors {
            let delta = layer.mesh.origin_ecef_m - eye;
            let offset_eye = [delta.x as f32, delta.y as f32, delta.z as f32];
            let opacity = layer.opacity.clamp(0.0, 1.0);
            let fill_alpha = layer.fill_color[3] * opacity;
            let line_alpha = layer.line_color[3] * opacity;

            let fill_offset =
                (fill_alpha > 0.001 && !layer.mesh.fill_indices.is_empty()).then(|| {
                    arena.push(&VectorUniform {
                        offset_eye,
                        _pad0: 0.0,
                        color: [
                            layer.fill_color[0],
                            layer.fill_color[1],
                            layer.fill_color[2],
                            fill_alpha,
                        ],
                    })
                });

            let has_strokes_or_points =
                !layer.mesh.line_vertices.is_empty() || !layer.mesh.point_vertices.is_empty();
            let line_offset = (line_alpha > 0.001 && has_strokes_or_points).then(|| {
                arena.push(&VectorUniform {
                    offset_eye,
                    _pad0: 0.0,
                    color: [
                        layer.line_color[0],
                        layer.line_color[1],
                        layer.line_color[2],
                        line_alpha,
                    ],
                })
            });

            self.draw_offsets.push(VectorDrawOffsets {
                fill_offset,
                line_offset,
            });
        }
    }

    pub(super) fn record_pass<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        fill_pipeline: &'a wgpu::RenderPipeline,
        line_pipeline: &'a wgpu::RenderPipeline,
        arena: &'a DynamicUniformArena,
        snapshot: &'a DrawSnapshot,
    ) {
        if snapshot.vectors.is_empty() {
            return;
        }
        for (layer, offsets) in snapshot.vectors.iter().zip(self.draw_offsets.iter()) {
            let Some(bufs) = self.buffers.get(&layer.id) else {
                continue;
            };
            if let (Some(dyn_off), Some(vb), Some(ib)) = (
                offsets.fill_offset,
                bufs.fill_vb.as_ref(),
                bufs.fill_ib.as_ref(),
            ) && bufs.fill_index_count > 0
            {
                pass.set_pipeline(fill_pipeline);
                pass.set_bind_group(1, arena.overlay_bind_group(), &[dyn_off]);
                pass.set_vertex_buffer(0, vb.slice(..));
                pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..bufs.fill_index_count, 0, 0..1);
            }
            if let (Some(dyn_off), Some(vb)) = (offsets.line_offset, bufs.line_vb.as_ref())
                && bufs.line_vertex_count > 0
            {
                pass.set_pipeline(line_pipeline);
                pass.set_bind_group(1, arena.overlay_bind_group(), &[dyn_off]);
                pass.set_vertex_buffer(0, vb.slice(..));
                pass.draw(0..bufs.line_vertex_count, 0..1);
            }
            if let (Some(dyn_off), Some(vb)) = (offsets.line_offset, bufs.point_vb.as_ref())
                && bufs.point_vertex_count > 0
            {
                pass.set_pipeline(fill_pipeline);
                pass.set_bind_group(1, arena.overlay_bind_group(), &[dyn_off]);
                pass.set_vertex_buffer(0, vb.slice(..));
                pass.draw(0..bufs.point_vertex_count, 0..1);
            }
        }
    }
}
