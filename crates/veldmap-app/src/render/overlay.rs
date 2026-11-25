//! Depth-less post-body warp-mesh overlay renderer supporting `Display`
//! (`Rgba8UnormSrgb`), `Scientific` (`R32Float` 8-tap nodata-trilinear + live
//! linear/dB stretch + colormap), and `Categorical` (`R8Uint`/`R16Uint`
//! nearest-neighbor class palettes).

use std::collections::HashMap;
use std::sync::{Arc, Weak};

use bytemuck::{Pod, Zeroable};
use veldmap_core::glam::DVec3;
use veldmap_core::tile::{MAX_WARP_SEGMENTS, PixelFormat, SourceId, build_warp_patch};
use wgpu::util::DeviceExt;

use super::bind_groups::TileBindGroupCache;
use super::colormap::{
    CategoricalPaletteId, ColormapId, categorical_lut_linear, scientific_lut_linear,
};
use super::mesh_cache::{DEFAULT_MESH_CACHE_BYTES, ElasticMeshCache, push_grid_quad_indices};
use super::snapshot::{DrawOverlayTrack, DrawSnapshot, OverlayMeshKey};
use super::uniforms::{
    CategoricalPatchUniform, DynamicUniformArena, OverlayPatchUniform, PaletteUniform,
    ScientificPatchUniform,
};
use crate::clamp_opacity;

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

/// Pipelines passed to [`OverlayRenderer::record_pass`].
pub(super) struct OverlayPipelines<'a> {
    pub(super) display: &'a wgpu::RenderPipeline,
    pub(super) scientific: &'a wgpu::RenderPipeline,
    pub(super) categorical: &'a wgpu::RenderPipeline,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum PaletteKey {
    Scientific(ColormapId),
    CategoricalPreset(CategoricalPaletteId),
    CustomSource(SourceId),
}

struct CachedPalette {
    _buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    custom_weak: Option<Weak<[[u8; 4]]>>,
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

/// Manages shared per-density index buffers, bounded LRU cache of lazily built
/// per-cell RTC warp-mesh vertex buffers, cached 256-entry palette LUT uniform
/// bind groups, and per-frame dynamic uniform offsets for draped raster overlays.
pub(super) struct OverlayRenderer {
    index_buffers: HashMap<(u32, bool), (wgpu::Buffer, u32)>,
    meshes: ElasticMeshCache<OverlayMeshKey, CachedOverlayMesh>,
    palettes: HashMap<PaletteKey, CachedPalette>,
    uniform_offsets: Vec<Option<u32>>,
}

impl OverlayRenderer {
    pub(super) fn new() -> Self {
        Self {
            index_buffers: HashMap::new(),
            meshes: ElasticMeshCache::new(DEFAULT_MESH_CACHE_BYTES),
            palettes: HashMap::new(),
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

    fn ensure_palette(
        &mut self,
        device: &wgpu::Device,
        palette_layout: &wgpu::BindGroupLayout,
        track: &DrawOverlayTrack,
    ) {
        let (key, lut, custom_weak) = match track {
            DrawOverlayTrack::Display => return,
            DrawOverlayTrack::Scientific { colormap, .. } => {
                let key = PaletteKey::Scientific(*colormap);
                if self.palettes.contains_key(&key) {
                    return;
                }
                (key, scientific_lut_linear(*colormap), None)
            }
            DrawOverlayTrack::Categorical {
                source,
                palette,
                custom_color_map,
                ..
            } => {
                if *palette == CategoricalPaletteId::SourceColorMap
                    && let Some(cmap) = custom_color_map
                {
                    let key = PaletteKey::CustomSource(*source);
                    let reuse = self.palettes.get(&key).is_some_and(|entry| {
                        entry.custom_weak.as_ref().is_some_and(|w| {
                            w.strong_count() > 0 && std::ptr::eq(w.as_ptr(), Arc::as_ptr(cmap))
                        })
                    });
                    if reuse {
                        return;
                    }
                    (
                        key,
                        categorical_lut_linear(*palette, Some(cmap.as_ref())),
                        Some(Arc::downgrade(cmap)),
                    )
                } else {
                    let key = PaletteKey::CategoricalPreset(*palette);
                    if self.palettes.contains_key(&key) {
                        return;
                    }
                    (key, categorical_lut_linear(*palette, None), None)
                }
            }
        };

        let uniform = PaletteUniform { entries: lut };
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("overlay palette LUT uniform buffer"),
            contents: bytemuck::bytes_of(&uniform),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("overlay palette LUT bind group"),
            layout: palette_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });
        self.palettes.insert(
            key,
            CachedPalette {
                _buffer: buffer,
                bind_group,
                custom_weak,
            },
        );
    }

    fn palette_bind_group(&self, track: &DrawOverlayTrack) -> Option<&wgpu::BindGroup> {
        let key = match track {
            DrawOverlayTrack::Display => return None,
            DrawOverlayTrack::Scientific { colormap, .. } => PaletteKey::Scientific(*colormap),
            DrawOverlayTrack::Categorical {
                source,
                palette,
                custom_color_map,
                ..
            } => {
                if *palette == CategoricalPaletteId::SourceColorMap && custom_color_map.is_some() {
                    PaletteKey::CustomSource(*source)
                } else {
                    PaletteKey::CategoricalPreset(*palette)
                }
            }
        };
        self.palettes.get(&key).map(|p| &p.bind_group)
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
        // Sweep custom TIFF ColorMap palette entries whose owning source dropped.
        self.palettes
            .retain(|_, p| p.custom_weak.as_ref().is_none_or(|w| w.strong_count() > 0));

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
            self.ensure_palette(device, palette_layout, &layer.track);
            let segs = layer.segments.clamp(1, MAX_WARP_SEGMENTS);
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
            for patch in &layer.patches {
                let offset = self.meshes.peek(&patch.mesh_key).map(|mesh| {
                    let offset_eye = (mesh.origin_ecef_m - eye).as_vec3().to_array();
                    match &layer.track {
                        DrawOverlayTrack::Display => {
                            let u = OverlayPatchUniform {
                                offset_eye,
                                opacity,
                                uv_rect: patch.uv_rect,
                            };
                            arena.push(&u)
                        }
                        DrawOverlayTrack::Scientific {
                            scale,
                            offset: phys_offset,
                            stretch_min,
                            stretch_max,
                            log10_db,
                            invert_colormap,
                            ..
                        } => {
                            let safe_scale = if scale.is_finite() { *scale } else { 1.0 };
                            let safe_offset = if phys_offset.is_finite() {
                                *phys_offset
                            } else {
                                0.0
                            };
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
                            let u = ScientificPatchUniform {
                                offset_eye,
                                opacity,
                                uv_rect: patch.uv_rect,
                                scale: safe_scale,
                                offset: safe_offset,
                                stretch_min: safe_min,
                                stretch_inv_range,
                                stretch_mode: u32::from(*log10_db),
                                invert_colormap: u32::from(*invert_colormap),
                                mip_count: patch.texture.mip_level_count.max(1),
                                _pad0: 0,
                            };
                            arena.push(&u)
                        }
                        DrawOverlayTrack::Categorical { nodata_value, .. } => {
                            let u = CategoricalPatchUniform {
                                offset_eye,
                                opacity,
                                uv_rect: patch.uv_rect,
                                has_nodata: u32::from(nodata_value.is_some()),
                                nodata_value: nodata_value.unwrap_or(0),
                                _pad0: 0,
                                _pad1: 0,
                            };
                            arena.push(&u)
                        }
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
        pipelines: OverlayPipelines<'a>,
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
            let desired_pipeline = match &layer.track {
                DrawOverlayTrack::Display => ActiveOverlayPipeline::Display,
                DrawOverlayTrack::Scientific { .. } => ActiveOverlayPipeline::Scientific,
                DrawOverlayTrack::Categorical { .. } => ActiveOverlayPipeline::Categorical,
            };
            let palette_bg = match &layer.track {
                DrawOverlayTrack::Display => None,
                DrawOverlayTrack::Scientific { .. } | DrawOverlayTrack::Categorical { .. } => {
                    let Some(bg) = self.palette_bind_group(&layer.track) else {
                        offset_idx += layer.patches.len();
                        continue;
                    };
                    Some(bg)
                }
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
                let tex_bg = match desired_pipeline {
                    ActiveOverlayPipeline::Display => {
                        bind_groups.get(&patch.texture, PixelFormat::Rgba8UnormSrgb)
                    }
                    ActiveOverlayPipeline::Scientific => {
                        bind_groups.get(&patch.texture, PixelFormat::R32Float)
                    }
                    ActiveOverlayPipeline::Categorical => bind_groups.get_uint(&patch.texture),
                };
                let Some(tex_bg) = tex_bg else {
                    continue;
                };
                let idx_key = (mesh.segments, mesh.flip_winding);
                let Some((idx_buf, idx_count)) = self.index_buffers.get(&idx_key) else {
                    continue;
                };

                if last_pipeline != Some(desired_pipeline) {
                    let rp = match desired_pipeline {
                        ActiveOverlayPipeline::Display => pipelines.display,
                        ActiveOverlayPipeline::Scientific => pipelines.scientific,
                        ActiveOverlayPipeline::Categorical => pipelines.categorical,
                    };
                    pass.set_pipeline(rp);
                    last_pipeline = Some(desired_pipeline);
                    last_tex_id = None;
                }
                if !layer_bound {
                    if let Some(pbg) = palette_bg {
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
                let uniform_bg = match desired_pipeline {
                    ActiveOverlayPipeline::Display => arena.overlay_bind_group(),
                    ActiveOverlayPipeline::Scientific => arena.scientific_overlay_bind_group(),
                    ActiveOverlayPipeline::Categorical => arena.categorical_overlay_bind_group(),
                };
                pass.set_bind_group(1, uniform_bg, &[dyn_offset]);
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
