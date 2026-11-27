//! Depth-less post-body warp-mesh overlay renderer supporting direct `Display`
//! (`Rgba8UnormSrgb`), `Scientific` (`R32Float` 8-tap nodata-trilinear + live
//! linear/dB stretch + colormap), `Categorical` (`R8Uint`/`R16Uint`), and
//! multi-asset screen-space `BandMath` (`1..=4` `R32Float` scratch passes with
//! MRT fast path + optional `R8Uint` `SCL` mask + fullscreen composite).

use std::collections::HashMap;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use veldmap_core::glam::DVec3;
use veldmap_core::tile::{MAX_WARP_SEGMENTS, PixelFormat, build_warp_patch};
use wgpu::util::DeviceExt;

use crate::clamp_opacity;
use crate::colormap::ColormapId;
use crate::gpu_tiles::GpuTile;

use super::bind_groups::TileBindGroupCache;
use super::mesh_cache::{DEFAULT_MESH_CACHE_BYTES, ElasticMeshCache, push_grid_quad_indices};
use super::palette_cache::PaletteCache;
use super::pipelines::{BandMathPipelines, OverlayPipelines};
use super::scratch::{
    MASK_UNCOVERED_SENTINEL, QUIET_NAN_BITS, ScratchTargetPool, compute_band_math_scissor_rect,
};
use super::snapshot::{
    DrawBandMathLayer, DrawOverlayContent, DrawOverlayLayer, DrawOverlayPatch, DrawOverlayTrack,
    DrawSnapshot, OverlayMeshKey,
};
use super::uniforms::{
    BandMathCompositeUniform, CategoricalPatchUniform, DynamicUniformArena, OverlayPatchUniform,
    ScientificPatchUniform, ScratchMrtPatchUniform, ScratchSinglePatchUniform, UniformSlot,
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
/// tile-format bind-group lookup resolved once per direct overlay layer.
struct LayerDrawBindings<'a> {
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
                pipeline: &pipelines.display,
                uniform_bg: arena.bind_group(UniformSlot::Overlay),
                palette_bg: None,
                texture_lookup: lookup_rgba8_tile,
            }),
            DrawOverlayTrack::Scientific { .. } => Some(Self {
                pipeline: &pipelines.scientific,
                uniform_bg: arena.bind_group(UniformSlot::Scientific),
                palette_bg: Some(palettes.bind_group(track)?),
                texture_lookup: lookup_r32f_tile,
            }),
            DrawOverlayTrack::Categorical { .. } => Some(Self {
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

fn safe_inv_range(min_v: f32, max_v: f32) -> (f32, f32) {
    let safe_min = if min_v.is_finite() { min_v } else { 0.0 };
    let safe_max = if max_v.is_finite() {
        max_v
    } else {
        safe_min + 1.0
    };
    let span = safe_max - safe_min;
    let inv = if span.is_finite() && span.abs() > 1e-12 {
        1.0 / span
    } else {
        0.0
    };
    (safe_min, inv)
}

fn sanitize_scale_offset(scale: f32, offset: f32) -> (f32, f32) {
    (
        if scale.is_finite() { scale } else { 1.0 },
        if offset.is_finite() { offset } else { 0.0 },
    )
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
                let (safe_scale, safe_offset) = sanitize_scale_offset(*scale, *offset);
                let (stretch_min, stretch_inv_range) = safe_inv_range(*stretch_min, *stretch_max);
                Self::Scientific {
                    scale: safe_scale,
                    offset: safe_offset,
                    stretch_min,
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

/// Prepared per-layer GPU uniform offsets & scissor bounds built once in
/// [`OverlayRenderer::push_uniforms`] and consumed during segmented Pass-2
/// recording.
enum PreparedLayerDraw {
    Direct {
        patch_offsets: Vec<Option<u32>>,
    },
    BandMath {
        /// `None` if the clamped scissor rect is empty (skip entire layer).
        scissor: Option<[u32; 4]>,
        /// When `Some`, co-grid MRT mode is active with one uniform offset per
        /// shared patch index.
        mrt_patch_offsets: Option<Vec<Option<u32>>>,
        /// Per-band single-attachment patch offsets when not using MRT.
        per_band_offsets: Vec<Vec<Option<u32>>>,
        /// Optional `R8Uint` `SCL` mask patch offsets.
        mask_offsets: Vec<Option<u32>>,
        /// Fullscreen composite dynamic uniform offset.
        composite_offset: Option<u32>,
        colormap: ColormapId,
    },
}

fn mrt_tiles_for_patch(bm: &DrawBandMathLayer, patch_idx: usize) -> Option<[&Arc<GpuTile>; 4]> {
    let first = &bm.bands.first()?.patches.get(patch_idx)?.texture;
    let t0 = first;
    let t1 = bm
        .bands
        .get(1)
        .and_then(|b| b.patches.get(patch_idx))
        .map_or(first, |p| &p.texture);
    let t2 = bm
        .bands
        .get(2)
        .and_then(|b| b.patches.get(patch_idx))
        .map_or(first, |p| &p.texture);
    let t3 = bm
        .bands
        .get(3)
        .and_then(|b| b.patches.get(patch_idx))
        .map_or(first, |p| &p.texture);
    Some([t0, t1, t2, t3])
}

/// Manages shared per-density index buffers, bounded LRU cache of lazily built
/// per-cell RTC warp-mesh vertex buffers, cached 256-entry palette LUT uniform
/// bind groups, and per-frame dynamic uniform offsets for draped raster overlays.
pub(super) struct OverlayRenderer {
    index_buffers: HashMap<(u32, bool), (wgpu::Buffer, u32)>,
    meshes: ElasticMeshCache<OverlayMeshKey, CachedOverlayMesh>,
    palettes: PaletteCache,
    prepared_layers: Vec<PreparedLayerDraw>,
}

impl OverlayRenderer {
    pub(super) fn new() -> Self {
        Self {
            index_buffers: HashMap::new(),
            meshes: ElasticMeshCache::new(DEFAULT_MESH_CACHE_BYTES),
            palettes: PaletteCache::new(),
            prepared_layers: Vec::new(),
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

    fn ensure_patch_mesh(
        &mut self,
        device: &wgpu::Device,
        frame: &veldmap_core::geo::Frame,
        image_width: u32,
        image_height: u32,
        segments: u32,
        patch: &DrawOverlayPatch,
    ) {
        let segs = clamp_warp_segments(segments);
        if self
            .meshes
            .peek(&patch.mesh_key)
            .is_some_and(|existing| existing.segments == segs)
        {
            return;
        }
        let warp = build_warp_patch(frame, image_width, image_height, patch.cell_rect, segs);
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

    /// Ensure vertex buffers, index buffers, palette LUT bind groups, and
    /// format-appropriate single & MRT texture bind groups exist for every
    /// overlay patch.
    pub(super) fn prepare_resources(
        &mut self,
        device: &wgpu::Device,
        palette_layout: &wgpu::BindGroupLayout,
        bind_groups: &mut TileBindGroupCache,
        snapshot: &DrawSnapshot,
    ) {
        self.palettes.sweep_stale();

        if snapshot.overlays.is_empty() {
            self.meshes.relax_to_default();
            return;
        }

        let mut needed_bytes = 0usize;
        let mut live_keys = Vec::new();
        for layer in &snapshot.overlays {
            match &layer.content {
                DrawOverlayContent::Direct {
                    segments, patches, ..
                } => {
                    let side = (clamp_warp_segments(*segments) + 1) as usize;
                    needed_bytes = needed_bytes.saturating_add(
                        patches
                            .len()
                            .saturating_mul(side * side * std::mem::size_of::<OverlayVertex>()),
                    );
                    live_keys.extend(patches.iter().map(|p| p.mesh_key));
                }
                DrawOverlayContent::BandMath(bm) => {
                    let mrt = bm.can_use_mrt();
                    for (idx, band) in bm.bands.iter().enumerate() {
                        if mrt && idx > 0 {
                            continue;
                        }
                        let side = (clamp_warp_segments(band.segments) + 1) as usize;
                        needed_bytes =
                            needed_bytes.saturating_add(band.patches.len().saturating_mul(
                                side * side * std::mem::size_of::<OverlayVertex>(),
                            ));
                        live_keys.extend(band.patches.iter().map(|p| p.mesh_key));
                    }
                    if let Some(scl) = &bm.scl
                        && bm.scl_mask_enabled
                    {
                        let side = (clamp_warp_segments(scl.segments) + 1) as usize;
                        needed_bytes =
                            needed_bytes.saturating_add(scl.patches.len().saturating_mul(
                                side * side * std::mem::size_of::<OverlayVertex>(),
                            ));
                        live_keys.extend(scl.patches.iter().map(|p| p.mesh_key));
                    }
                }
            }
        }
        self.meshes.begin_frame(needed_bytes, live_keys.iter());

        for layer in &snapshot.overlays {
            match &layer.content {
                DrawOverlayContent::Direct {
                    track,
                    frame,
                    image_width,
                    image_height,
                    segments,
                    patches,
                } => {
                    self.palettes.ensure(device, palette_layout, track);
                    for patch in patches {
                        bind_groups.ensure(device, &patch.texture);
                        self.ensure_patch_mesh(
                            device,
                            frame,
                            *image_width,
                            *image_height,
                            *segments,
                            patch,
                        );
                    }
                }
                DrawOverlayContent::BandMath(bm) => {
                    self.palettes
                        .ensure_colormap(device, palette_layout, bm.colormap);
                    if bm.can_use_mrt() {
                        let first = &bm.bands[0];
                        for (patch_idx, patch) in first.patches.iter().enumerate() {
                            if let Some(tiles) = mrt_tiles_for_patch(bm, patch_idx) {
                                bind_groups.ensure_mrt(device, tiles);
                            }
                            self.ensure_patch_mesh(
                                device,
                                &first.frame,
                                first.image_width,
                                first.image_height,
                                first.segments,
                                patch,
                            );
                        }
                    } else {
                        for band in &bm.bands {
                            for patch in &band.patches {
                                bind_groups.ensure(device, &patch.texture);
                                self.ensure_patch_mesh(
                                    device,
                                    &band.frame,
                                    band.image_width,
                                    band.image_height,
                                    band.segments,
                                    patch,
                                );
                            }
                        }
                    }
                    if let Some(scl) = &bm.scl
                        && bm.scl_mask_enabled
                    {
                        for patch in &scl.patches {
                            bind_groups.ensure(device, &patch.texture);
                            self.ensure_patch_mesh(
                                device,
                                &scl.frame,
                                scl.image_width,
                                scl.image_height,
                                scl.segments,
                                patch,
                            );
                        }
                    }
                }
            }
        }

        self.meshes.end_frame(needed_bytes);
    }

    /// Push per-patch and per-composite uniforms for every overlay layer into
    /// the frame-wide `arena`, recording aligned dynamic byte offsets and
    /// projected-AABB scissor rects.
    pub(super) fn push_uniforms(
        &mut self,
        arena: &mut DynamicUniformArena,
        snapshot: &DrawSnapshot,
        eye: DVec3,
        width: u32,
        height: u32,
    ) {
        self.prepared_layers.clear();
        for layer in &snapshot.overlays {
            let opacity = clamp_opacity(layer.opacity);
            match &layer.content {
                DrawOverlayContent::Direct { track, patches, .. } => {
                    let params = PreparedTrackParams::from_track(track);
                    let mut patch_offsets = Vec::with_capacity(patches.len());
                    for patch in patches {
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
                        patch_offsets.push(offset);
                    }
                    self.prepared_layers
                        .push(PreparedLayerDraw::Direct { patch_offsets });
                }
                DrawOverlayContent::BandMath(bm) => {
                    let has_required_bands = bm.bands.len() >= bm.formula.required_bands();
                    let scissor = has_required_bands
                        .then(|| {
                            compute_band_math_scissor_rect(&snapshot.camera, width, height, bm)
                        })
                        .flatten();
                    if scissor.is_none() {
                        self.prepared_layers.push(PreparedLayerDraw::BandMath {
                            scissor: None,
                            mrt_patch_offsets: None,
                            per_band_offsets: Vec::new(),
                            mask_offsets: Vec::new(),
                            composite_offset: None,
                            colormap: bm.colormap,
                        });
                        continue;
                    }

                    let (mrt_patch_offsets, per_band_offsets) = if bm.can_use_mrt() {
                        let first = &bm.bands[0];
                        let mut offsets = Vec::with_capacity(first.patches.len());
                        for (patch_idx, patch) in first.patches.iter().enumerate() {
                            let off = self.meshes.peek(&patch.mesh_key).map(|mesh| {
                                let offset_eye = (mesh.origin_ecef_m - eye).as_vec3().to_array();
                                let mut uv_rects = [[0.0f32; 4]; 4];
                                let mut scale_offsets = [[1.0f32, 0.0, 0.0, 0.0]; 4];
                                let mut mip_counts = [1u32; 4];
                                for (b_idx, band) in bm.bands.iter().take(4).enumerate() {
                                    if let Some(bp) = band.patches.get(patch_idx) {
                                        uv_rects[b_idx] = bp.uv_rect;
                                        let (sc, of) =
                                            sanitize_scale_offset(band.scale, band.offset);
                                        scale_offsets[b_idx] = [sc, of, 0.0, 0.0];
                                        mip_counts[b_idx] = bp.texture.mip_level_count.max(1);
                                    }
                                }
                                arena.push(&ScratchMrtPatchUniform {
                                    offset_eye,
                                    nan_bits: QUIET_NAN_BITS,
                                    uv_rects,
                                    scale_offsets,
                                    mip_counts,
                                })
                            });
                            offsets.push(off);
                        }
                        (Some(offsets), Vec::new())
                    } else {
                        let mut all_band_offs = Vec::with_capacity(bm.bands.len());
                        for band in &bm.bands {
                            let (sc, of) = sanitize_scale_offset(band.scale, band.offset);
                            let mut b_offs = Vec::with_capacity(band.patches.len());
                            for patch in &band.patches {
                                let off = self.meshes.peek(&patch.mesh_key).map(|mesh| {
                                    let offset_eye =
                                        (mesh.origin_ecef_m - eye).as_vec3().to_array();
                                    arena.push(&ScratchSinglePatchUniform {
                                        offset_eye,
                                        scale: sc,
                                        uv_rect: patch.uv_rect,
                                        offset: of,
                                        mip_count: patch.texture.mip_level_count.max(1),
                                        has_nodata: 0,
                                        nodata_value: 0,
                                    })
                                });
                                b_offs.push(off);
                            }
                            all_band_offs.push(b_offs);
                        }
                        (None, all_band_offs)
                    };

                    let mut mask_offsets = Vec::new();
                    let has_active_mask = bm
                        .scl
                        .as_ref()
                        .is_some_and(|m| bm.scl_mask_enabled && !m.patches.is_empty());
                    if has_active_mask && let Some(scl) = &bm.scl {
                        let has_nodata = u32::from(scl.nodata_value.is_some());
                        let nodata_value = scl.nodata_value.unwrap_or(0);
                        mask_offsets.reserve(scl.patches.len());
                        for patch in &scl.patches {
                            let off = self.meshes.peek(&patch.mesh_key).map(|mesh| {
                                let offset_eye = (mesh.origin_ecef_m - eye).as_vec3().to_array();
                                arena.push(&ScratchSinglePatchUniform {
                                    offset_eye,
                                    scale: 1.0,
                                    uv_rect: patch.uv_rect,
                                    offset: 0.0,
                                    mip_count: 1,
                                    has_nodata,
                                    nodata_value,
                                })
                            });
                            mask_offsets.push(off);
                        }
                    }

                    let (stretch_min, stretch_inv_range) =
                        safe_inv_range(bm.stretch_min, bm.stretch_max);
                    let (r_min, r_inv) = safe_inv_range(bm.rgb_min[0], bm.rgb_max[0]);
                    let (g_min, g_inv) = safe_inv_range(bm.rgb_min[1], bm.rgb_max[1]);
                    let (b_min, b_inv) = safe_inv_range(bm.rgb_min[2], bm.rgb_max[2]);
                    let safe_gamma = if bm.rgb_gamma.is_finite() {
                        bm.rgb_gamma.clamp(0.2, 4.0)
                    } else {
                        1.0
                    };
                    let inv_gamma = 1.0 / safe_gamma;
                    let composite_offset = arena.push(&BandMathCompositeUniform {
                        opacity,
                        mode: bm.formula.shader_mode(),
                        stretch_min,
                        stretch_inv_range,
                        rgb_min: [r_min, g_min, b_min, inv_gamma],
                        rgb_inv_range: [r_inv, g_inv, b_inv, 0.0],
                        invert_colormap: u32::from(bm.invert_colormap),
                        scl_mask_enabled: u32::from(has_active_mask),
                        scl_mask_bits: bm.scl_mask_bits,
                        _pad0: 0,
                    });

                    self.prepared_layers.push(PreparedLayerDraw::BandMath {
                        scissor,
                        mrt_patch_offsets,
                        per_band_offsets,
                        mask_offsets,
                        composite_offset: Some(composite_offset),
                        colormap: bm.colormap,
                    });
                }
            }
        }
    }

    /// Record a single direct overlay layer inside an already-open depth-less
    /// color render pass (`@group(0)` camera already bound).
    pub(super) fn record_direct_layer<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        layer_idx: usize,
        layer: &'a DrawOverlayLayer,
        pipelines: &'a OverlayPipelines,
        arena: &'a DynamicUniformArena,
        bind_groups: &'a TileBindGroupCache,
    ) {
        let DrawOverlayContent::Direct { track, patches, .. } = &layer.content else {
            return;
        };
        let Some(PreparedLayerDraw::Direct { patch_offsets }) = self.prepared_layers.get(layer_idx)
        else {
            return;
        };
        let Some(bindings) = LayerDrawBindings::resolve(track, pipelines, arena, &self.palettes)
        else {
            return;
        };

        let mut pipeline_bound = false;
        let mut last_index_key: Option<(u32, bool)> = None;
        let mut last_tex_id: Option<u64> = None;

        for (patch, &dyn_offset) in patches.iter().zip(patch_offsets.iter()) {
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

            if !pipeline_bound {
                pass.set_pipeline(bindings.pipeline);
                if let Some(pbg) = bindings.palette_bg {
                    pass.set_bind_group(3, pbg, &[]);
                }
                pipeline_bound = true;
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

    fn record_scratch_single_patches<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        patches: &'a [DrawOverlayPatch],
        offsets: &[Option<u32>],
        arena: &'a DynamicUniformArena,
        mut resolve_tex_bg: impl FnMut(
            &'a Arc<crate::gpu_tiles::GpuTile>,
        ) -> Option<&'a wgpu::BindGroup>,
    ) {
        let mut last_idx_key: Option<(u32, bool)> = None;
        let mut last_tex_id: Option<u64> = None;
        for (patch, &dyn_offset) in patches.iter().zip(offsets.iter()) {
            let Some(dyn_offset) = dyn_offset else {
                continue;
            };
            let Some(mesh) = self.meshes.peek(&patch.mesh_key) else {
                continue;
            };
            let Some(tex_bg) = resolve_tex_bg(&patch.texture) else {
                continue;
            };
            let idx_key = (mesh.segments, mesh.flip_winding);
            let Some((idx_buf, idx_count)) = self.index_buffers.get(&idx_key) else {
                continue;
            };
            if last_idx_key != Some(idx_key) {
                pass.set_index_buffer(idx_buf.slice(..), wgpu::IndexFormat::Uint32);
                last_idx_key = Some(idx_key);
            }
            if last_tex_id != Some(patch.texture.tile_id) {
                pass.set_bind_group(2, tex_bg, &[]);
                last_tex_id = Some(patch.texture.tile_id);
            }
            pass.set_vertex_buffer(0, mesh.buffer.slice(..));
            pass.set_bind_group(
                1,
                arena.bind_group(UniformSlot::ScratchSingle),
                &[dyn_offset],
            );
            pass.draw_indexed(0..*idx_count, 0, 0..1);
        }
    }

    /// Record the shared viewport scratch passes (quiet-NaN initialization +
    /// MRT or per-band `R32Float` draws + optional `R8Uint` `SCL` mask pass)
    /// for a `BandMath` layer, constrained to its clamped projected-AABB
    /// scissor rect. Returns `true` if the layer should proceed to fullscreen
    /// composite.
    #[expect(
        clippy::too_many_arguments,
        reason = "Explicit borrowed GPU state avoids allocating a transient pass-context struct per BandMath layer."
    )]
    pub(super) fn record_band_math_scratch_passes<'a>(
        &'a self,
        encoder: &mut wgpu::CommandEncoder,
        layer_idx: usize,
        bm: &'a DrawBandMathLayer,
        camera_bg: &'a wgpu::BindGroup,
        pipelines: &'a BandMathPipelines,
        scratch: &'a ScratchTargetPool,
        arena: &'a DynamicUniformArena,
        bind_groups: &'a TileBindGroupCache,
    ) -> bool {
        let Some(PreparedLayerDraw::BandMath {
            scissor: Some(scissor),
            mrt_patch_offsets,
            per_band_offsets,
            mask_offsets,
            ..
        }) = self.prepared_layers.get(layer_idx)
        else {
            return false;
        };
        let band_count = bm.bands.len().min(4);
        let float_views = scratch.float_views();
        if band_count < bm.formula.required_bands() || float_views.len() < band_count {
            return false;
        }

        let [sx, sy, sw, sh] = *scissor;

        // 1. Initialize active `1..=band_count` `R32Float` scratch targets via
        //    `LoadOp::Clear(TRANSPARENT)` (eliminating TBDR full-viewport DRAM
        //    readback) + scissored quiet-NaN fill, merged directly with either
        //    the single co-grid MRT pass or each per-band single-target pass.
        if let Some(mrt_offsets) = mrt_patch_offsets {
            let attachments: Vec<Option<wgpu::RenderPassColorAttachment<'_>>> = float_views
                [..band_count]
                .iter()
                .map(|view| {
                    Some(wgpu::RenderPassColorAttachment {
                        view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    })
                })
                .collect();
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("band-math scratch NaN fill + MRT"),
                color_attachments: &attachments,
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_scissor_rect(sx, sy, sw, sh);
            pass.set_pipeline(pipelines.clear_for_count(band_count));
            pass.set_bind_group(0, scratch.nan_clear_bind_group(), &[]);
            pass.draw(0..3, 0..1);

            pass.set_pipeline(pipelines.mrt_for_count(band_count));
            pass.set_bind_group(0, camera_bg, &[]);
            let first = &bm.bands[0];
            let mut last_idx_key: Option<(u32, bool)> = None;
            for (patch_idx, (patch, &dyn_offset)) in
                first.patches.iter().zip(mrt_offsets.iter()).enumerate()
            {
                let Some(dyn_offset) = dyn_offset else {
                    continue;
                };
                let Some(mesh) = self.meshes.peek(&patch.mesh_key) else {
                    continue;
                };
                let Some(tiles) = mrt_tiles_for_patch(bm, patch_idx) else {
                    continue;
                };
                let Some(mrt_bg) = bind_groups.get_mrt(tiles) else {
                    continue;
                };
                let idx_key = (mesh.segments, mesh.flip_winding);
                let Some((idx_buf, idx_count)) = self.index_buffers.get(&idx_key) else {
                    continue;
                };
                if last_idx_key != Some(idx_key) {
                    pass.set_index_buffer(idx_buf.slice(..), wgpu::IndexFormat::Uint32);
                    last_idx_key = Some(idx_key);
                }
                pass.set_vertex_buffer(0, mesh.buffer.slice(..));
                pass.set_bind_group(1, arena.bind_group(UniformSlot::ScratchMrt), &[dyn_offset]);
                pass.set_bind_group(2, mrt_bg, &[]);
                pass.draw_indexed(0..*idx_count, 0, 0..1);
            }
        } else {
            for ((band, offsets), view) in bm
                .bands
                .iter()
                .zip(per_band_offsets.iter())
                .zip(float_views[..band_count].iter())
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("band-math single-band NaN fill + R32Float scratch pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_scissor_rect(sx, sy, sw, sh);
                pass.set_pipeline(pipelines.clear_for_count(1));
                pass.set_bind_group(0, scratch.nan_clear_bind_group(), &[]);
                pass.draw(0..3, 0..1);

                pass.set_pipeline(&pipelines.band_single);
                pass.set_bind_group(0, camera_bg, &[]);
                self.record_scratch_single_patches(&mut pass, &band.patches, offsets, arena, |t| {
                    bind_groups.get(t, PixelFormat::R32Float)
                });
            }
        }

        // 2. Optional nearest-neighbor `R8Uint` `SCL` mask pass (cleared to 255u sentinel).
        if !mask_offsets.is_empty()
            && let Some(scl) = &bm.scl
            && let Some(mask_view) = scratch.mask_view()
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("band-math R8Uint SCL mask scratch pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: mask_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: f64::from(MASK_UNCOVERED_SENTINEL),
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_scissor_rect(sx, sy, sw, sh);
            pass.set_pipeline(&pipelines.mask_u8);
            pass.set_bind_group(0, camera_bg, &[]);
            self.record_scratch_single_patches(&mut pass, &scl.patches, mask_offsets, arena, |t| {
                bind_groups.get_uint(t)
            });
        }

        true
    }

    /// Record the scissored fullscreen Band Math + `SCL` mask + colormap
    /// composite draw inside an open depth-less `Rgba8UnormSrgb` color pass,
    /// then restore the full-viewport scissor rect for subsequent draws.
    #[expect(
        clippy::too_many_arguments,
        reason = "Explicit borrowed GPU state matches record_band_math_scratch_passes."
    )]
    pub(super) fn record_band_math_composite<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        layer_idx: usize,
        full_width: u32,
        full_height: u32,
        pipelines: &'a BandMathPipelines,
        scratch: &'a ScratchTargetPool,
        arena: &'a DynamicUniformArena,
    ) {
        let Some(PreparedLayerDraw::BandMath {
            scissor: Some([sx, sy, sw, sh]),
            composite_offset: Some(comp_offset),
            colormap,
            ..
        }) = self.prepared_layers.get(layer_idx)
        else {
            return;
        };
        let Some(scratch_bg) = scratch.composite_bind_group() else {
            return;
        };
        let Some(palette_bg) = self.palettes.colormap_bind_group(*colormap) else {
            return;
        };

        pass.set_scissor_rect(*sx, *sy, *sw, *sh);
        pass.set_pipeline(&pipelines.composite);
        pass.set_bind_group(0, scratch_bg, &[]);
        pass.set_bind_group(
            1,
            arena.bind_group(UniformSlot::BandMathComposite),
            &[*comp_offset],
        );
        pass.set_bind_group(2, palette_bg, &[]);
        pass.draw(0..3, 0..1);
        pass.set_scissor_rect(0, 0, full_width, full_height);
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
