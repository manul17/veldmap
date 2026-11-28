//! Depth-less post-body warp-mesh overlay renderer supporting direct `Display`
//! (`Rgba8UnormSrgb`), `Scientific` (`R32Float` 8-tap nodata-trilinear + live
//! linear/dB stretch + colormap), `Categorical` (`R8Uint`/`R16Uint`), and
//! multi-asset screen-space `BandMath` (`1..=4` `R32Float` scratch passes with
//! MRT fast path + optional `R8Uint` `SCL` mask + fullscreen composite).

mod band_math;
mod mesh;

use std::sync::Arc;

use veldmap_core::glam::DVec3;
use veldmap_core::tile::PixelFormat;

use crate::clamp_opacity;
use crate::gpu_tiles::GpuTile;

use super::bind_groups::TileBindGroupCache;
use super::palette_cache::PaletteCache;
use super::pipelines::OverlayPipelines;
use super::snapshot::{
    DrawBandMathLayer, DrawOverlayContent, DrawOverlayLayer, DrawOverlayTrack, DrawSnapshot,
    WarpPatchSetRef,
};
use super::swipe::ResolvedSwipe;
use super::uniforms::{
    CategoricalPatchUniform, DynamicUniformArena, OverlayPatchUniform, ScientificPatchUniform,
    UniformSlot,
};

pub(super) use band_math::BandMathPassCtx;
use band_math::{
    PreparedBandMath, mrt_tiles_for_patch, push_band_math_uniforms, record_composite,
    record_scratch_passes,
};
use mesh::OverlayMeshPool;
pub(super) use mesh::OverlayVertex;

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
    Direct { patch_offsets: Vec<Option<u32>> },
    BandMath(PreparedBandMath),
}

/// Manages shared per-density index buffers, bounded LRU cache of lazily built
/// per-cell RTC warp-mesh vertex buffers, cached 256-entry palette LUT uniform
/// bind groups, and per-frame dynamic uniform offsets for draped raster overlays.
pub(super) struct OverlayRenderer {
    meshes: OverlayMeshPool,
    palettes: PaletteCache,
    /// Per-layer cached `can_use_mrt()` flag evaluated once in
    /// [`Self::prepare_resources`] and reused in [`Self::push_uniforms`].
    layer_mrt_flags: Vec<bool>,
    prepared_layers: Vec<PreparedLayerDraw>,
}

impl OverlayRenderer {
    pub(super) fn new() -> Self {
        Self {
            meshes: OverlayMeshPool::new(),
            palettes: PaletteCache::new(),
            layer_mrt_flags: Vec::new(),
            prepared_layers: Vec::new(),
        }
    }

    fn ensure_patch_set_resources(
        &mut self,
        device: &wgpu::Device,
        bind_groups: &mut TileBindGroupCache,
        set: WarpPatchSetRef<'_>,
    ) {
        for patch in set.patches {
            bind_groups.ensure(device, &patch.texture);
            self.meshes.ensure_patch_mesh(device, set, patch);
        }
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
        self.layer_mrt_flags.clear();

        if snapshot.overlays.is_empty() {
            self.meshes.relax_to_default();
            return;
        }

        self.layer_mrt_flags.reserve(snapshot.overlays.len());
        let mut needed_bytes = 0usize;
        let mut live_keys = Vec::new();
        for layer in &snapshot.overlays {
            match &layer.content {
                DrawOverlayContent::Direct { .. } => {
                    self.layer_mrt_flags.push(false);
                    if let Some(set) = layer.content.direct_warp_patch_set() {
                        OverlayMeshPool::account_patch_set(set, &mut needed_bytes, &mut live_keys);
                    }
                }
                DrawOverlayContent::BandMath(bm) => {
                    let mrt = bm.can_use_mrt();
                    self.layer_mrt_flags.push(mrt);
                    for set in bm.active_warp_patch_sets(mrt) {
                        OverlayMeshPool::account_patch_set(set, &mut needed_bytes, &mut live_keys);
                    }
                }
            }
        }
        self.meshes.begin_frame(needed_bytes, live_keys.iter());

        for (layer_idx, layer) in snapshot.overlays.iter().enumerate() {
            match &layer.content {
                DrawOverlayContent::Direct { track, .. } => {
                    self.palettes.ensure(device, palette_layout, track);
                    if let Some(set) = layer.content.direct_warp_patch_set() {
                        self.ensure_patch_set_resources(device, bind_groups, set);
                    }
                }
                DrawOverlayContent::BandMath(bm) => {
                    self.palettes
                        .ensure_colormap(device, palette_layout, bm.colormap);
                    let mrt = self.layer_mrt_flags[layer_idx];
                    if mrt {
                        let first_set = bm.bands[0].warp_patch_set();
                        for (patch_idx, patch) in first_set.patches.iter().enumerate() {
                            if let Some(tiles) = mrt_tiles_for_patch(bm, patch_idx) {
                                bind_groups.ensure_mrt(device, tiles);
                            }
                            self.meshes.ensure_patch_mesh(device, first_set, patch);
                        }
                    } else {
                        for band in &bm.bands {
                            self.ensure_patch_set_resources(
                                device,
                                bind_groups,
                                band.warp_patch_set(),
                            );
                        }
                    }
                    if let Some(scl) = &bm.scl
                        && bm.scl_mask_enabled
                    {
                        self.ensure_patch_set_resources(device, bind_groups, scl.warp_patch_set());
                    }
                }
            }
        }

        self.meshes.end_frame(needed_bytes);
    }

    /// Push per-patch and per-composite uniforms for every overlay layer into
    /// the frame-wide `arena`, recording aligned dynamic byte offsets and
    /// projected-AABB scissor rects (intersected with any active swipe
    /// half-plane before scratch passes run).
    pub(super) fn push_uniforms(
        &mut self,
        arena: &mut DynamicUniformArena,
        snapshot: &DrawSnapshot,
        eye: DVec3,
        width: u32,
        height: u32,
        swipe: Option<ResolvedSwipe>,
    ) {
        self.prepared_layers.clear();
        for (layer_idx, layer) in snapshot.overlays.iter().enumerate() {
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
                    let use_mrt = self
                        .layer_mrt_flags
                        .get(layer_idx)
                        .copied()
                        .unwrap_or_else(|| bm.can_use_mrt());
                    let uniform_ctx = band_math::BandMathUniformCtx {
                        meshes: &self.meshes,
                        snapshot,
                        eye,
                        width,
                        height,
                    };
                    let prepared = push_band_math_uniforms(
                        &uniform_ctx,
                        arena,
                        layer.layer_id,
                        bm,
                        use_mrt,
                        opacity,
                        swipe,
                    );
                    self.prepared_layers
                        .push(PreparedLayerDraw::BandMath(prepared));
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

            if !pipeline_bound {
                pass.set_pipeline(bindings.pipeline);
                if let Some(pbg) = bindings.palette_bg {
                    pass.set_bind_group(3, pbg, &[]);
                }
                pipeline_bound = true;
            }
            if last_tex_id != Some(patch.texture.tile_id) {
                pass.set_bind_group(2, tex_bg, &[]);
                last_tex_id = Some(patch.texture.tile_id);
            }
            self.meshes.draw_indexed_patch(
                pass,
                mesh,
                &mut last_index_key,
                bindings.uniform_bg,
                dyn_offset,
            );
        }
    }

    /// Record the shared viewport scratch passes (quiet-NaN initialization +
    /// MRT or per-band `R32Float` draws + optional `R8Uint` `SCL` mask pass)
    /// for a `BandMath` layer, constrained to its clamped projected-AABB
    /// scissor rect. Returns `true` if the layer should proceed to fullscreen
    /// composite.
    pub(super) fn record_band_math_scratch_passes<'a>(
        &'a self,
        encoder: &mut wgpu::CommandEncoder,
        layer_idx: usize,
        bm: &'a DrawBandMathLayer,
        ctx: &BandMathPassCtx<'a>,
    ) -> bool {
        let Some(PreparedLayerDraw::BandMath(prepared)) = self.prepared_layers.get(layer_idx)
        else {
            return false;
        };
        record_scratch_passes(&self.meshes, encoder, prepared, bm, ctx)
    }

    /// Record the scissored fullscreen Band Math + `SCL` mask + colormap
    /// composite draw inside an open depth-less `Rgba8UnormSrgb` color pass,
    /// then restore the full-viewport scissor rect for subsequent draws.
    pub(super) fn record_band_math_composite<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        layer_idx: usize,
        full_width: u32,
        full_height: u32,
        ctx: &BandMathPassCtx<'a>,
    ) {
        let Some(PreparedLayerDraw::BandMath(prepared)) = self.prepared_layers.get(layer_idx)
        else {
            return;
        };
        record_composite(&self.palettes, pass, prepared, full_width, full_height, ctx);
    }
}
