//! Depth-less post-body GPU vector renderer for GeoJSON overlays, interactive
//! geodesic measurement geometries, and STAC AOI / scene-footprint overlays
//! with per-cluster CPU `f64` RTC offsets, HiDPI screen-space ribbon outlines,
//! diagonal hatch fills, and GPU-instanced altitude-scaled point markers.

use std::collections::HashSet;
use std::ops::Range;

use veldmap_core::glam::DVec3;

use super::mesh_cache::{DEFAULT_MESH_CACHE_BYTES, ElasticMeshCache};
use super::pipelines::VectorPipelines;
use super::snapshot::{DrawSnapshot, DrawVectorId};
use super::uniforms::{DynamicUniformArena, UniformSlot, VectorUniform};
use super::vector_pack::{
    CachedVectorBuffers, ClusterDrawRange, estimate_layer_gpu_bytes, pack_layer_buffers,
};
use crate::clamp_opacity;
use crate::colormap::srgb_rgba_f32_to_linear;

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
            VectorDrawKind::Line => (off.line_offset, cluster.line_segments.clone()),
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
        pass.set_bind_group(1, arena.bind_group(UniformSlot::Vector), &[dyn_off]);
        match target.kind {
            VectorDrawKind::Fill { .. } => pass.draw_indexed(range, 0, 0..1),
            VectorDrawKind::Line | VectorDrawKind::Point => pass.draw(0..6, range),
        }
    }
}

/// Elastic LRU GPU buffer cache and Pass-2 recorder for draped vector layers.
pub(super) struct VectorRenderer {
    buffers: ElasticMeshCache<(DrawVectorId, u64), CachedVectorBuffers>,
    draw_offsets: Vec<ClusterDrawOffsets>,
    layer_offset_ranges: Vec<Range<usize>>,
}

impl VectorRenderer {
    pub(super) fn new() -> Self {
        Self {
            buffers: ElasticMeshCache::new(DEFAULT_MESH_CACHE_BYTES),
            draw_offsets: Vec::new(),
            layer_offset_ranges: Vec::new(),
        }
    }

    pub(super) fn prepare_resources(&mut self, device: &wgpu::Device, snapshot: &DrawSnapshot) {
        if snapshot.vectors.is_empty() {
            // Relax any elastic cap expansion back to `DEFAULT_MESH_CACHE_BYTES`
            // without clearing warm entries so toggling visibility avoids full
            // CPU repacking and GPU buffer re-upload.
            self.buffers.begin_frame(0, std::iter::empty());
            self.buffers.end_frame(0);
            return;
        }

        let mut seen = HashSet::with_capacity(snapshot.vectors.len());
        let mut active_keys = Vec::with_capacity(snapshot.vectors.len());
        let mut estimated_needed = 0usize;
        for layer in &snapshot.vectors {
            let key = (layer.id, layer.rev);
            if seen.insert(key) {
                active_keys.push(key);
                let bytes = self
                    .buffers
                    .peek(&key)
                    .map_or_else(|| estimate_layer_gpu_bytes(layer), |b| b.byte_size);
                estimated_needed = estimated_needed.saturating_add(bytes);
            }
        }

        self.buffers.begin_frame(estimated_needed, &active_keys);
        for layer in &snapshot.vectors {
            let key = (layer.id, layer.rev);
            if self.buffers.peek(&key).is_some() {
                continue;
            }
            let packed = pack_layer_buffers(device, layer);
            let byte_size = packed.byte_size;
            // Evict superseded revisions for this vector layer ID so interactive
            // dynamic layers (AOI drag, STAC highlight, measurement) never accumulate
            // dead GPU buffers and evict warm static GeoJSON layers.
            self.buffers
                .remove_where(|k| k.0 == layer.id && k.1 != layer.rev);
            self.buffers.insert(key, packed, byte_size);
        }

        let actual_needed = active_keys.iter().fold(0usize, |acc, key| {
            acc.saturating_add(self.buffers.peek(key).map_or(0, |b| b.byte_size))
        });
        self.buffers.end_frame(actual_needed);
    }

    pub(super) fn push_uniforms(
        &mut self,
        arena: &mut DynamicUniformArena,
        snapshot: &DrawSnapshot,
        eye: DVec3,
        width_px: u32,
        height_px: u32,
    ) {
        self.draw_offsets.clear();
        self.layer_offset_ranges.clear();
        let marker_radius_m = point_marker_radius_m(snapshot.camera.height_m());
        let ppp = snapshot.pixels_per_point.max(0.5);
        let viewport_inv_size = [
            2.0 / (width_px.max(1) as f32),
            2.0 / (height_px.max(1) as f32),
        ];

        for layer in &snapshot.vectors {
            let start = self.draw_offsets.len();
            let opacity = clamp_opacity(layer.opacity);
            let fill_linear = srgb_rgba_f32_to_linear(layer.fill_color, opacity);
            let line_linear = srgb_rgba_f32_to_linear(layer.line_color, opacity);
            let fill_alpha = fill_linear[3];
            let line_alpha = line_linear[3];
            let stroke_half_width_px =
                (layer.stroke_width_px.max(1.0) * ppp * 0.5).clamp(0.5, 16.0);
            let hatch_spacing_px = if layer.hatch_spacing_px > 0.5 {
                (layer.hatch_spacing_px * ppp).clamp(4.0, 128.0)
            } else {
                0.0
            };

            let Some(bufs) = self.buffers.peek(&(layer.id, layer.rev)) else {
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
                            viewport_inv_size,
                            stroke_half_width_px: 0.0,
                            hatch_spacing_px,
                        })
                    });

                let has_strokes_or_points =
                    !cluster.line_segments.is_empty() || !cluster.point_instances.is_empty();
                let line_offset = (line_alpha > 0.001 && has_strokes_or_points).then(|| {
                    arena.push(&VectorUniform {
                        offset_eye,
                        point_radius_m: marker_radius_m,
                        color: line_linear,
                        viewport_inv_size,
                        stroke_half_width_px,
                        hatch_spacing_px: 0.0,
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
        pipelines: &'a VectorPipelines,
        arena: &'a DynamicUniformArena,
        snapshot: &'a DrawSnapshot,
    ) {
        if snapshot.vectors.is_empty() {
            return;
        }
        for (layer, range) in snapshot.vectors.iter().zip(self.layer_offset_ranges.iter()) {
            let Some(bufs) = self.buffers.peek(&(layer.id, layer.rev)) else {
                continue;
            };
            let Some(cluster_offsets) = self.draw_offsets.get(range.clone()) else {
                continue;
            };

            if let (Some(vb), Some(ib)) = (bufs.fill_vb.as_ref(), bufs.fill_ib.as_ref()) {
                record_cluster_draws(
                    pass,
                    ClusterDrawTarget {
                        pipeline: &pipelines.fill,
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
                        pipeline: &pipelines.line,
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
                        pipeline: &pipelines.point,
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
