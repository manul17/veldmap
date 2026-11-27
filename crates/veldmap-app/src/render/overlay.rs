//! Depth-less post-body warp-mesh overlay renderer supporting `Display`
//! (`Rgba8UnormSrgb`), `Scientific` (`R32Float` 8-tap nodata-trilinear + live
//! linear/dB stretch + colormap), and `Categorical` (`R8Uint`/`R16Uint`
//! nearest-neighbor class palettes).

use std::collections::HashMap;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use veldmap_core::glam::DVec3;
use veldmap_core::tile::{MAX_WARP_SEGMENTS, PixelFormat, build_warp_patch};
use wgpu::util::DeviceExt;

use crate::clamp_opacity;
use crate::gpu_tiles::GpuTile;

use super::bind_groups::TileBindGroupCache;
use super::mesh_cache::{DEFAULT_MESH_CACHE_BYTES, ElasticMeshCache, push_grid_quad_indices};
use super::palette_cache::PaletteCache;
use super::pipelines::OverlayPipelines;
use super::snapshot::{DrawOverlayTrack, DrawSnapshot, OverlayMeshKey};
use super::uniforms::{
    CategoricalPatchUniform, DynamicUniformArena, OverlayPatchUniform, ScientificPatchUniform,
    UniformSlot,
};

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

fn clamp_warp_segments(segments: u32) -> u32 {
    segments.clamp(1, MAX_WARP_SEGMENTS)
}

struct CachedOverlayMesh {
    buffer: wgpu::Buffer,
    origin_ecef_m: DVec3,
    segments: u32,
    flip_winding: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ActiveOverlayPipeline {
    Display,
    Scientific,
    Categorical,
}

type TileTextureLookup =
    for<'a, 'b> fn(&'a TileBindGroupCache, &'b Arc<GpuTile>) -> Option<&'a wgpu::BindGroup>;

fn lookup_rgba8_tile<'a>(
    bind_groups: &'a TileBindGroupCache,
    tile: &Arc<GpuTile>,
) -> Option<&'a wgpu::BindGroup> {
    bind_groups.get(tile, PixelFormat::Rgba8UnormSrgb)
}

fn lookup_r32f_tile<'a>(
    bind_groups: &'a TileBindGroupCache,
    tile: &Arc<GpuTile>,
) -> Option<&'a wgpu::BindGroup> {
    bind_groups.get(tile, PixelFormat::R32Float)
}

/// Per-layer Pass-2 pipeline, dynamic uniform slot, optional palette LUT, and
/// tile-format bind-group lookup resolved once per overlay layer.
struct LayerDrawBindings<'a> {
    pipeline_kind: ActiveOverlayPipeline,
    pipeline: &'a wgpu::RenderPipeline,
    uniform_bg: &'a wgpu::BindGroup,
    palette_bg: Option<&'a wgpu::BindGroup>,
    texture_lookup: TileTextureLookup,
}

impl<'a> LayerDrawBindings<'a> {
    fn resolve(
        track: &DrawOverlayTrack,
        pipelines: &'a OverlayPipelines,
        arena: &'a DynamicUniformArena,
        palettes: &'a PaletteCache,
    ) -> Option<Self> {
        match track {
            DrawOverlayTrack::Display => Some(Self {
                pipeline_kind: ActiveOverlayPipeline::Display,
                pipeline: &pipelines.display,
                uniform_bg: arena.bind_group(UniformSlot::Overlay),
                palette_bg: None,
                texture_lookup: lookup_rgba8_tile,
            }),
            DrawOverlayTrack::Scientific { .. } => Some(Self {
                pipeline_kind: ActiveOverlayPipeline::Scientific,
                pipeline: &pipelines.scientific,
                uniform_bg: arena.bind_group(UniformSlot::Scientific),
                palette_bg: Some(palettes.bind_group(track)?),
                texture_lookup: lookup_r32f_tile,
            }),
            DrawOverlayTrack::Categorical { .. } => Some(Self {
                pipeline_kind: ActiveOverlayPipeline::Categorical,
                pipeline: &pipelines.categorical,
                uniform_bg: arena.bind_group(UniformSlot::Categorical),
                palette_bg: Some(palettes.bind_group(track)?),
                texture_lookup: TileBindGroupCache::get_uint,
            }),
        }
    }

    fn texture_bind_group(
        &self,
        bind_groups: &'a TileBindGroupCache,
        texture: &Arc<GpuTile>,
    ) -> Option<&'a wgpu::BindGroup> {
        (self.texture_lookup)(bind_groups, texture)
    }
}

/// Layer-invariant uniform parameters sanitized once per layer outside the
/// per-patch loop in [`OverlayRenderer::push_uniforms`].
#[derive(Clone, Copy)]
enum PreparedTrackParams {
    Display,
    Scientific {
        scale: f32,
        offset: f32,
        stretch_min: f32,
        stretch_inv_range: f32,
        stretch_mode: u32,
        invert_colormap: u32,
    },
    Categorical {
        has_nodata: u32,
        nodata_value: u32,
    },
}

impl PreparedTrackParams {
    fn from_track(track: &DrawOverlayTrack) -> Self {
        match track {
            DrawOverlayTrack::Display => Self::Display,
            DrawOverlayTrack::Scientific {
                scale,
                offset,
                stretch_min,
                stretch_max,
                log10_db,
                invert_colormap,
                ..
            } => {
                let safe_scale = if scale.is_finite() { *scale } else { 1.0 };
                let safe_offset = if offset.is_finite() { *offset } else { 0.0 };
                let safe_min = if stretch_min.is_finite() {
                    *stretch_min
                } else {
                    0.0
                };
                let safe_max = if stretch_max.is_finite() {
                    *stretch_max
                } else {
                    safe_min + 1.0
                };
                let span = safe_max - safe_min;
                let stretch_inv_range = if span.is_finite() && span.abs() > 1e-12 {
                    1.0 / span
                } else {
                    0.0
                };
                Self::Scientific {
                    scale: safe_scale,
                    offset: safe_offset,
                    stretch_min: safe_min,
                    stretch_inv_range,
                    stretch_mode: u32::from(*log10_db),
                    invert_colormap: u32::from(*invert_colormap),
                }
            }
            DrawOverlayTrack::Categorical { nodata_value, .. } => Self::Categorical {
                has_nodata: u32::from(nodata_value.is_some()),
                nodata_value: nodata_value.unwrap_or(0),
            },
        }
    }
}

/// Manages shared per-density index buffers, bounded LRU cache of lazily built
/// per-cell RTC warp-mesh vertex buffers, cached 256-entry palette LUT uniform
/// bind groups, and per-frame dynamic uniform offsets for draped raster overlays.
pub(super) struct OverlayRenderer {
    index_buffers: HashMap<(u32, bool), (wgpu::Buffer, u32)>,
    meshes: ElasticMeshCache<OverlayMeshKey, CachedOverlayMesh>,
    palettes: PaletteCache,
    uniform_offsets: Vec<Option<u32>>,
}

impl OverlayRenderer {
    pub(super) fn new() -> Self {
        Self {
            index_buffers: HashMap::new(),
            meshes: ElasticMeshCache::new(DEFAULT_MESH_CACHE_BYTES),
            palettes: PaletteCache::new(),
            uniform_offsets: Vec::new(),
        }
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

    /// Ensure vertex buffers, index buffers, palette LUT bind groups, and
    /// format-appropriate texture bind groups exist for every overlay patch.
    pub(super) fn prepare_resources(
        &mut self,
        device: &wgpu::Device,
        palette_layout: &wgpu::BindGroupLayout,
        bind_groups: &mut TileBindGroupCache,
        snapshot: &DrawSnapshot,
    ) {
        self.palettes.sweep_stale();

        if snapshot.overlays.is_empty() {
            // Relax any elastic cap expansion back to `DEFAULT_MESH_CACHE_BYTES`
            // without clearing warm warp meshes so temporary layer hides or
            // off-screen pans avoid full CPU warp rebuilding and GPU re-upload.
            self.meshes.relax_to_default();
            return;
        }

        let needed_bytes: usize = snapshot
            .overlays
            .iter()
            .map(|layer| {
                let side = (clamp_warp_segments(layer.segments) + 1) as usize;
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
            self.palettes.ensure(device, palette_layout, &layer.track);
            let segs = clamp_warp_segments(layer.segments);
            for patch in &layer.patches {
                bind_groups.ensure(device, &patch.texture);

                if self
                    .meshes
                    .peek(&patch.mesh_key)
                    .is_some_and(|existing| existing.segments == segs)
                {
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

    /// Push per-patch uniforms for every overlay patch with a cached mesh into
    /// the frame-wide `arena`, recording aligned dynamic byte offsets in draw
    /// order.
    pub(super) fn push_uniforms(
        &mut self,
        arena: &mut DynamicUniformArena,
        snapshot: &DrawSnapshot,
        eye: DVec3,
    ) {
        self.uniform_offsets.clear();
        for layer in &snapshot.overlays {
            let opacity = clamp_opacity(layer.opacity);
            let params = PreparedTrackParams::from_track(&layer.track);
            for patch in &layer.patches {
                let offset = self.meshes.peek(&patch.mesh_key).map(|mesh| {
                    let offset_eye = (mesh.origin_ecef_m - eye).as_vec3().to_array();
                    match params {
                        PreparedTrackParams::Display => arena.push(&OverlayPatchUniform {
                            offset_eye,
                            opacity,
                            uv_rect: patch.uv_rect,
                        }),
                        PreparedTrackParams::Scientific {
                            scale,
                            offset,
                            stretch_min,
                            stretch_inv_range,
                            stretch_mode,
                            invert_colormap,
                        } => arena.push(&ScientificPatchUniform {
                            offset_eye,
                            opacity,
                            uv_rect: patch.uv_rect,
                            scale,
                            offset,
                            stretch_min,
                            stretch_inv_range,
                            stretch_mode,
                            invert_colormap,
                            mip_count: patch.texture.mip_level_count.max(1),
                            _pad0: 0,
                        }),
                        PreparedTrackParams::Categorical {
                            has_nodata,
                            nodata_value,
                        } => arena.push(&CategoricalPatchUniform {
                            offset_eye,
                            opacity,
                            uv_rect: patch.uv_rect,
                            has_nodata,
                            nodata_value,
                            _pad0: 0,
                            _pad1: 0,
                        }),
                    }
                });
                self.uniform_offsets.push(offset);
            }
        }
    }

    /// Record all ordered overlay layers bottom-to-top inside Pass 2
    /// (`depth_stencil_attachment: None`).
    pub(super) fn record_pass<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        pipelines: &'a OverlayPipelines,
        arena: &'a DynamicUniformArena,
        bind_groups: &'a TileBindGroupCache,
        snapshot: &'a DrawSnapshot,
    ) {
        if snapshot.overlays.is_empty() {
            return;
        }
        let mut last_pipeline: Option<ActiveOverlayPipeline> = None;
        let mut last_index_key: Option<(u32, bool)> = None;
        let mut last_tex_id: Option<u64> = None;
        let mut offset_idx = 0usize;

        for layer in &snapshot.overlays {
            let Some(bindings) =
                LayerDrawBindings::resolve(&layer.track, pipelines, arena, &self.palettes)
            else {
                offset_idx += layer.patches.len();
                continue;
            };

            let mut layer_bound = false;
            for patch in &layer.patches {
                let dyn_offset = self.uniform_offsets.get(offset_idx).copied().flatten();
                offset_idx += 1;
                let Some(dyn_offset) = dyn_offset else {
                    continue;
                };
                let Some(mesh) = self.meshes.peek(&patch.mesh_key) else {
                    continue;
                };
                let Some(tex_bg) = bindings.texture_bind_group(bind_groups, &patch.texture) else {
                    continue;
                };
                let idx_key = (mesh.segments, mesh.flip_winding);
                let Some((idx_buf, idx_count)) = self.index_buffers.get(&idx_key) else {
                    continue;
                };

                if last_pipeline != Some(bindings.pipeline_kind) {
                    pass.set_pipeline(bindings.pipeline);
                    last_pipeline = Some(bindings.pipeline_kind);
                    last_tex_id = None;
                }
                if !layer_bound {
                    if let Some(pbg) = bindings.palette_bg {
                        pass.set_bind_group(3, pbg, &[]);
                    }
                    layer_bound = true;
                }
                if last_index_key != Some(idx_key) {
                    pass.set_index_buffer(idx_buf.slice(..), wgpu::IndexFormat::Uint32);
                    last_index_key = Some(idx_key);
                }
                if last_tex_id != Some(patch.texture.tile_id) {
                    pass.set_bind_group(2, tex_bg, &[]);
                    last_tex_id = Some(patch.texture.tile_id);
                }
                pass.set_vertex_buffer(0, mesh.buffer.slice(..));
                pass.set_bind_group(1, bindings.uniform_bg, &[dyn_offset]);
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
