//! GPU vertex/ribbon layouts and per-layer cluster buffer packing for
//! [`super::vector::VectorRenderer`].

use std::ops::Range;

use bytemuck::{Pod, Zeroable};
use veldmap_core::glam::{DVec3, Vec3};
use wgpu::util::DeviceExt;

use super::snapshot::DrawVectorLayer;

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

/// Compact 48-byte instanced line-segment record matching `RibbonIn` in
/// `shaders/vector.wgsl` (`@location(0..3)`), extruded on GPU into a
/// HiDPI-scaled screen-space ribbon quad.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct GpuRibbonSegment {
    p0: [f32; 3],
    n0: [f32; 3],
    p1: [f32; 3],
    n1: [f32; 3],
}

const _: () = {
    assert!(std::mem::size_of::<GpuRibbonSegment>() == 48);
};

impl GpuRibbonSegment {
    const ATTRS: [wgpu::VertexAttribute; 4] = wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Float32x3,
        2 => Float32x3,
        3 => Float32x3,
    ];

    pub(super) const INSTANCE_LAYOUT: wgpu::VertexBufferLayout<'static> =
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &Self::ATTRS,
        };
}

pub(super) struct ClusterDrawRange {
    pub(super) origin_ecef_m: DVec3,
    pub(super) fill_indices: Range<u32>,
    pub(super) line_segments: Range<u32>,
    pub(super) point_instances: Range<u32>,
}

pub(super) struct CachedVectorBuffers {
    pub(super) fill_vb: Option<wgpu::Buffer>,
    pub(super) fill_ib: Option<wgpu::Buffer>,
    pub(super) line_vb: Option<wgpu::Buffer>,
    pub(super) point_vb: Option<wgpu::Buffer>,
    pub(super) clusters: Vec<ClusterDrawRange>,
    pub(super) byte_size: usize,
}

struct LayerElementCounts {
    fill_verts: usize,
    fill_indices: usize,
    line_segments: usize,
    point_verts: usize,
}

fn sum_layer_element_counts(layer: &DrawVectorLayer) -> LayerElementCounts {
    let mut counts = LayerElementCounts {
        fill_verts: 0,
        fill_indices: 0,
        line_segments: 0,
        point_verts: 0,
    };
    for cluster in &layer.mesh.clusters {
        if !cluster.fill_vertices.is_empty() && !cluster.fill_indices.is_empty() {
            counts.fill_verts = counts
                .fill_verts
                .saturating_add(cluster.fill_vertices.len());
            counts.fill_indices = counts
                .fill_indices
                .saturating_add(cluster.fill_indices.len());
        }
        counts.line_segments = counts
            .line_segments
            .saturating_add(cluster.line_vertices.len() / 2);
        counts.point_verts = counts
            .point_verts
            .saturating_add(cluster.point_vertices.len());
    }
    counts
}

pub(super) fn estimate_layer_gpu_bytes(layer: &DrawVectorLayer) -> usize {
    let counts = sum_layer_element_counts(layer);
    counts
        .fill_verts
        .saturating_mul(std::mem::size_of::<GpuVectorVertex>())
        .saturating_add(
            counts
                .fill_indices
                .saturating_mul(std::mem::size_of::<u32>()),
        )
        .saturating_add(
            counts
                .line_segments
                .saturating_mul(std::mem::size_of::<GpuRibbonSegment>()),
        )
        .saturating_add(
            counts
                .point_verts
                .saturating_mul(std::mem::size_of::<GpuVectorVertex>()),
        )
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

pub(super) fn pack_layer_buffers(
    device: &wgpu::Device,
    layer: &DrawVectorLayer,
) -> CachedVectorBuffers {
    let counts = sum_layer_element_counts(layer);
    let mut all_fill_verts = Vec::with_capacity(counts.fill_verts);
    let mut all_fill_indices = Vec::with_capacity(counts.fill_indices);
    let mut all_line_segments = Vec::with_capacity(counts.line_segments);
    let mut all_point_verts = Vec::with_capacity(counts.point_verts);
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

        let line_start = all_line_segments.len() as u32;
        for pair in cluster.line_vertices.chunks_exact(2) {
            all_line_segments.push(GpuRibbonSegment {
                p0: pair[0].rel_pos,
                n0: pair[0].normal,
                p1: pair[1].rel_pos,
                n1: pair[1].normal,
            });
        }
        let line_end = all_line_segments.len() as u32;

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
            line_segments: line_start..line_end,
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
        "vector ribbon segment instance buffer",
        &all_line_segments,
        wgpu::BufferUsages::VERTEX,
    );
    let point_vb = create_checked_buffer(
        device,
        "vector point instance buffer",
        &all_point_verts,
        wgpu::BufferUsages::VERTEX,
    );

    let byte_size = [&fill_vb, &fill_ib, &line_vb, &point_vb]
        .into_iter()
        .flatten()
        .fold(0usize, |acc, buf| acc.saturating_add(buf.size() as usize));

    CachedVectorBuffers {
        fill_vb,
        fill_ib,
        line_vb,
        point_vb,
        clusters,
        byte_size,
    }
}
