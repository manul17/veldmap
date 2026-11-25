//! Depth-less post-body GPU vector renderer for GeoJSON overlays and
//! interactive geodesic measurement geometries with per-cluster CPU `f64` RTC
//! offsets and GPU-instanced altitude-scaled point markers.

use std::collections::HashMap;
use std::ops::Range;

use bytemuck::{Pod, Zeroable};
use veldmap_core::glam::{DVec3, Vec3};
use wgpu::util::DeviceExt;

use super::snapshot::{DrawSnapshot, DrawVectorId, DrawVectorLayer};
use super::uniforms::{DynamicUniformArena, VectorUniform};
use crate::clamp_opacity;
use crate::colormap::srgb_rgba_f32_to_linear;

/// Borrowed Pass-2 vector render pipelines passed into
/// [`VectorRenderer::record_pass`].
#[derive(Clone, Copy)]
pub(super) struct VectorPipelines<'a> {
    pub(super) fill: &'a wgpu::RenderPipeline,
    pub(super) line: &'a wgpu::RenderPipeline,
    pub(super) point: &'a wgpu::RenderPipeline,
}

/// Compact 24-byte GPU vertex matching `shaders/vector.wgsl` (`@location(0..1)`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct GpuVectorVertex {
    rel_pos: [f32; 3],
    normal: [f32; 3],
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

    pub(super) const INSTANCE_LAYOUT: wgpu::VertexBufferLayout<'static> =
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &Self::ATTRS,
        };
}

struct ClusterDrawRange {
    origin_ecef_m: DVec3,
    fill_indices: Range<u32>,
    line_vertices: Range<u32>,
    point_instances: Range<u32>,
}

struct CachedVectorBuffers {
    rev: u64,
    fill_vb: Option<wgpu::Buffer>,
    fill_ib: Option<wgpu::Buffer>,
    line_vb: Option<wgpu::Buffer>,
    point_vb: Option<wgpu::Buffer>,
    clusters: Vec<ClusterDrawRange>,
}

struct ClusterDrawOffsets {
    fill_offset: Option<u32>,
    line_offset: Option<u32>,
}

fn point_marker_radius_m(camera_height_m: f64) -> f32 {
    // Scale point markers continuously with camera altitude (~0.9% of eye
    // altitude) directly in `vs_point` so markers stay crisp from global orbit
    // down to centimetre drone inspection without CPU buffer reallocation.
    (camera_height_m * 0.009).clamp(0.03, 250_000.0) as f32
}

fn create_checked_buffer<T: Pod>(
    device: &wgpu::Device,
    label: &str,
    data: &[T],
    usage: wgpu::BufferUsages,
) -> Option<wgpu::Buffer> {
    if data.is_empty() {
        return None;
    }
    let bytes = bytemuck::cast_slice(data);
    let max_size = device.limits().max_buffer_size;
    if (bytes.len() as u64) > max_size {
        tracing::warn!(
            label,
            bytes = bytes.len(),
            max_buffer_size = max_size,
            "Vector GPU buffer exceeds device max_buffer_size; skipping upload"
        );
        return None;
    }
    Some(
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytes,
            usage,
        }),
    )
}

fn pack_layer_buffers(device: &wgpu::Device, layer: &DrawVectorLayer) -> CachedVectorBuffers {
    let mut all_fill_verts: Vec<GpuVectorVertex> = Vec::new();
    let mut all_fill_indices: Vec<u32> = Vec::new();
    let mut all_line_verts: Vec<GpuVectorVertex> = Vec::new();
    let mut all_point_verts: Vec<GpuVectorVertex> = Vec::new();
    let mut clusters = Vec::with_capacity(layer.mesh.clusters.len());

    for cluster in &layer.mesh.clusters {
        let fill_start = all_fill_indices.len() as u32;
        if !cluster.fill_vertices.is_empty() && !cluster.fill_indices.is_empty() {
            let base_v = all_fill_verts.len() as u32;
            all_fill_verts.extend(cluster.fill_vertices.iter().map(|v| GpuVectorVertex {
                rel_pos: v.rel_pos,
                normal: v.normal,
            }));
            all_fill_indices.extend(cluster.fill_indices.iter().map(|&idx| base_v + idx));
        }
        let fill_end = all_fill_indices.len() as u32;

        let line_start = all_line_verts.len() as u32;
        all_line_verts.extend(cluster.line_vertices.iter().map(|v| GpuVectorVertex {
            rel_pos: v.rel_pos,
            normal: v.normal,
        }));
        let line_end = all_line_verts.len() as u32;

        let point_start = all_point_verts.len() as u32;
        for v in &cluster.point_vertices {
            let n = Vec3::from_array(v.normal);
            if n.is_finite() && n.length_squared() > 1e-12 {
                let unit_n = n.normalize();
                all_point_verts.push(GpuVectorVertex {
                    rel_pos: v.rel_pos,
                    normal: unit_n.to_array(),
                });
            }
        }
        let point_end = all_point_verts.len() as u32;

        clusters.push(ClusterDrawRange {
            origin_ecef_m: cluster.origin_ecef_m,
            fill_indices: fill_start..fill_end,
            line_vertices: line_start..line_end,
            point_instances: point_start..point_end,
        });
    }

    let fill_vb = create_checked_buffer(
        device,
        "vector fill vertex buffer",
        &all_fill_verts,
        wgpu::BufferUsages::VERTEX,
    );
    let fill_ib = if fill_vb.is_some() {
        create_checked_buffer(
            device,
            "vector fill index buffer",
            &all_fill_indices,
            wgpu::BufferUsages::INDEX,
        )
    } else {
        None
    };
    let line_vb = create_checked_buffer(
        device,
        "vector line vertex buffer",
        &all_line_verts,
        wgpu::BufferUsages::VERTEX,
    );
    let point_vb = create_checked_buffer(
        device,
        "vector point instance buffer",
        &all_point_verts,
        wgpu::BufferUsages::VERTEX,
    );

    CachedVectorBuffers {
        rev: layer.rev,
        fill_vb,
        fill_ib,
        line_vb,
        point_vb,
        clusters,
    }
}

enum VectorDrawKind<'a> {
    Fill { ib: &'a wgpu::Buffer },
    Line,
    Point,
}

struct ClusterDrawTarget<'a> {
    pipeline: &'a wgpu::RenderPipeline,
    vb: &'a wgpu::Buffer,
    kind: VectorDrawKind<'a>,
}

fn record_cluster_draws<'a>(
    pass: &mut wgpu::RenderPass<'a>,
    target: ClusterDrawTarget<'a>,
    arena: &'a DynamicUniformArena,
    clusters: &[ClusterDrawRange],
    offsets: &[ClusterDrawOffsets],
) {
    let mut bound = false;
    for (cluster, off) in clusters.iter().zip(offsets.iter()) {
        let (dyn_off, range) = match target.kind {
            VectorDrawKind::Fill { .. } => (off.fill_offset, cluster.fill_indices.clone()),
            VectorDrawKind::Line => (off.line_offset, cluster.line_vertices.clone()),
            VectorDrawKind::Point => (off.line_offset, cluster.point_instances.clone()),
        };
        let Some(dyn_off) = dyn_off else {
            continue;
        };
        if range.is_empty() {
            continue;
        }
        if !bound {
            pass.set_pipeline(target.pipeline);
            pass.set_vertex_buffer(0, target.vb.slice(..));
            if let VectorDrawKind::Fill { ib } = target.kind {
                pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
            }
            bound = true;
        }
        pass.set_bind_group(1, arena.vector_bind_group(), &[dyn_off]);
        match target.kind {
            VectorDrawKind::Fill { .. } => pass.draw_indexed(range, 0, 0..1),
            VectorDrawKind::Line => pass.draw(range, 0..1),
            VectorDrawKind::Point => pass.draw(0..6, range),
        }
    }
}

/// GPU buffer cache and Pass-2 recorder for draped vector layers.
pub(super) struct VectorRenderer {
    buffers: HashMap<DrawVectorId, CachedVectorBuffers>,
    draw_offsets: Vec<ClusterDrawOffsets>,
    layer_offset_ranges: Vec<Range<usize>>,
}

impl VectorRenderer {
    pub(super) fn new() -> Self {
        Self {
            buffers: HashMap::new(),
            draw_offsets: Vec::new(),
            layer_offset_ranges: Vec::new(),
        }
    }

    pub(super) fn prepare_resources(&mut self, device: &wgpu::Device, snapshot: &DrawSnapshot) {
        if snapshot.vectors.is_empty() {
            self.buffers.clear();
            return;
        }
        self.buffers
            .retain(|id, _| snapshot.vectors.iter().any(|v| v.id == *id));

        for layer in &snapshot.vectors {
            let needs_upload = self
                .buffers
                .get(&layer.id)
                .is_none_or(|cached| cached.rev != layer.rev);
            if !needs_upload {
                continue;
            }
            let packed = pack_layer_buffers(device, layer);
            self.buffers.insert(layer.id, packed);
        }
    }

    pub(super) fn push_uniforms(
        &mut self,
        arena: &mut DynamicUniformArena,
        snapshot: &DrawSnapshot,
        eye: DVec3,
    ) {
        self.draw_offsets.clear();
        self.layer_offset_ranges.clear();
        let marker_radius_m = point_marker_radius_m(snapshot.camera.height_m());

        for layer in &snapshot.vectors {
            let start = self.draw_offsets.len();
            let opacity = clamp_opacity(layer.opacity);
            let fill_linear = srgb_rgba_f32_to_linear(layer.fill_color, opacity);
            let line_linear = srgb_rgba_f32_to_linear(layer.line_color, opacity);
            let fill_alpha = fill_linear[3];
            let line_alpha = line_linear[3];

            let Some(bufs) = self.buffers.get(&layer.id) else {
                self.layer_offset_ranges.push(start..start);
                continue;
            };

            self.draw_offsets.reserve(bufs.clusters.len());
            for cluster in &bufs.clusters {
                let offset_eye = (cluster.origin_ecef_m - eye).as_vec3().to_array();

                let fill_offset =
                    (fill_alpha > 0.001 && !cluster.fill_indices.is_empty()).then(|| {
                        arena.push(&VectorUniform {
                            offset_eye,
                            point_radius_m: 0.0,
                            color: fill_linear,
                        })
                    });

                let has_strokes_or_points =
                    !cluster.line_vertices.is_empty() || !cluster.point_instances.is_empty();
                let line_offset = (line_alpha > 0.001 && has_strokes_or_points).then(|| {
                    arena.push(&VectorUniform {
                        offset_eye,
                        point_radius_m: marker_radius_m,
                        color: line_linear,
                    })
                });

                self.draw_offsets.push(ClusterDrawOffsets {
                    fill_offset,
                    line_offset,
                });
            }
            self.layer_offset_ranges
                .push(start..self.draw_offsets.len());
        }
    }

    pub(super) fn record_pass<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        pipelines: VectorPipelines<'a>,
        arena: &'a DynamicUniformArena,
        snapshot: &'a DrawSnapshot,
    ) {
        if snapshot.vectors.is_empty() {
            return;
        }
        for (layer, range) in snapshot.vectors.iter().zip(self.layer_offset_ranges.iter()) {
            let Some(bufs) = self.buffers.get(&layer.id) else {
                continue;
            };
            let Some(cluster_offsets) = self.draw_offsets.get(range.clone()) else {
                continue;
            };

            if let (Some(vb), Some(ib)) = (bufs.fill_vb.as_ref(), bufs.fill_ib.as_ref()) {
                record_cluster_draws(
                    pass,
                    ClusterDrawTarget {
                        pipeline: pipelines.fill,
                        vb,
                        kind: VectorDrawKind::Fill { ib },
                    },
                    arena,
                    &bufs.clusters,
                    cluster_offsets,
                );
            }

            if let Some(vb) = bufs.line_vb.as_ref() {
                record_cluster_draws(
                    pass,
                    ClusterDrawTarget {
                        pipeline: pipelines.line,
                        vb,
                        kind: VectorDrawKind::Line,
                    },
                    arena,
                    &bufs.clusters,
                    cluster_offsets,
                );
            }

            if let Some(vb) = bufs.point_vb.as_ref() {
                record_cluster_draws(
                    pass,
                    ClusterDrawTarget {
                        pipeline: pipelines.point,
                        vb,
                        kind: VectorDrawKind::Point,
                    },
                    arena,
                    &bufs.clusters,
                    cluster_offsets,
                );
            }
        }
    }
}
