//! Multi-asset screen-space `BandMath` scratch-target passes (quiet-NaN fill,
//! co-grid MRT or per-band `R32Float` draws, optional `R8Uint` `SCL` mask pass)
//! and scissored fullscreen composite recording.

use std::sync::Arc;

use veldmap_core::glam::DVec3;
use veldmap_core::tile::PixelFormat;

use crate::colormap::ColormapId;
use crate::gpu_tiles::GpuTile;

use super::super::bind_groups::TileBindGroupCache;
use super::super::palette_cache::PaletteCache;
use super::super::pipelines::BandMathPipelines;
use super::super::scratch::{
    MASK_UNCOVERED_SENTINEL, QUIET_NAN_BITS, ScratchTargetPool, compute_band_math_scissor_rect,
};
use super::super::snapshot::{DrawBandMathLayer, DrawOverlayPatch, DrawSnapshot};
use super::super::uniforms::{
    BandMathCompositeUniform, DynamicUniformArena, ScratchMrtPatchUniform,
    ScratchSinglePatchUniform, UniformSlot,
};
use super::mesh::OverlayMeshPool;
use super::{safe_inv_range, sanitize_scale_offset};

/// Borrowed GPU resources shared across `BandMath` scratch and composite pass
/// recording.
pub(in crate::render) struct BandMathPassCtx<'a> {
    pub(in crate::render) camera_bg: &'a wgpu::BindGroup,
    pub(in crate::render) pipelines: &'a BandMathPipelines,
    pub(in crate::render) scratch: &'a ScratchTargetPool,
    pub(in crate::render) arena: &'a DynamicUniformArena,
    pub(in crate::render) bind_groups: &'a TileBindGroupCache,
}

/// Prepared dynamic uniform offsets and clamped scissor bounds for one
/// `BandMath` layer.
pub(super) struct PreparedBandMath {
    /// `None` if the clamped scissor rect is empty (skip entire layer).
    pub(super) scissor: Option<[u32; 4]>,
    /// When `Some`, co-grid MRT mode is active with one uniform offset per
    /// shared patch index.
    pub(super) mrt_patch_offsets: Option<Vec<Option<u32>>>,
    /// Per-band single-attachment patch offsets when not using MRT.
    pub(super) per_band_offsets: Vec<Vec<Option<u32>>>,
    /// Optional `R8Uint` `SCL` mask patch offsets.
    pub(super) mask_offsets: Vec<Option<u32>>,
    /// Fullscreen composite dynamic uniform offset.
    pub(super) composite_offset: Option<u32>,
    pub(super) colormap: ColormapId,
}

pub(super) fn mrt_tiles_for_patch(
    bm: &DrawBandMathLayer,
    patch_idx: usize,
) -> Option<[&Arc<GpuTile>; 4]> {
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

/// Frame-level inputs shared across all BandMath uniform pushes in a frame.
pub(super) struct BandMathUniformCtx<'a> {
    pub(super) meshes: &'a OverlayMeshPool,
    pub(super) snapshot: &'a DrawSnapshot,
    pub(super) eye: DVec3,
    pub(super) width: u32,
    pub(super) height: u32,
}

pub(super) fn push_band_math_uniforms(
    ctx: &BandMathUniformCtx<'_>,
    arena: &mut DynamicUniformArena,
    bm: &DrawBandMathLayer,
    use_mrt: bool,
    opacity: f32,
) -> PreparedBandMath {
    let meshes = ctx.meshes;
    let snapshot = ctx.snapshot;
    let eye = ctx.eye;
    let width = ctx.width;
    let height = ctx.height;
    let has_required_bands = bm.bands.len() >= bm.formula.required_bands();
    let scissor = has_required_bands
        .then(|| compute_band_math_scissor_rect(&snapshot.camera, width, height, bm))
        .flatten();
    if scissor.is_none() {
        return PreparedBandMath {
            scissor: None,
            mrt_patch_offsets: None,
            per_band_offsets: Vec::new(),
            mask_offsets: Vec::new(),
            composite_offset: None,
            colormap: bm.colormap,
        };
    }

    let (mrt_patch_offsets, per_band_offsets) = if use_mrt {
        let first = &bm.bands[0];
        let mut offsets = Vec::with_capacity(first.patches.len());
        for (patch_idx, patch) in first.patches.iter().enumerate() {
            let off = meshes.peek(&patch.mesh_key).map(|mesh| {
                let offset_eye = (mesh.origin_ecef_m - eye).as_vec3().to_array();
                let mut uv_rects = [[0.0f32; 4]; 4];
                let mut scale_offsets = [[1.0f32, 0.0, 0.0, 0.0]; 4];
                let mut mip_counts = [1u32; 4];
                for (b_idx, band) in bm.bands.iter().take(4).enumerate() {
                    if let Some(bp) = band.patches.get(patch_idx) {
                        uv_rects[b_idx] = bp.uv_rect;
                        let (sc, of) = sanitize_scale_offset(band.scale, band.offset);
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
                let off = meshes.peek(&patch.mesh_key).map(|mesh| {
                    let offset_eye = (mesh.origin_ecef_m - eye).as_vec3().to_array();
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
            let off = meshes.peek(&patch.mesh_key).map(|mesh| {
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

    let (stretch_min, stretch_inv_range) = safe_inv_range(bm.stretch_min, bm.stretch_max);
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

    PreparedBandMath {
        scissor,
        mrt_patch_offsets,
        per_band_offsets,
        mask_offsets,
        composite_offset: Some(composite_offset),
        colormap: bm.colormap,
    }
}

fn color_clear_attachment(
    view: &wgpu::TextureView,
    clear_color: wgpu::Color,
) -> Option<wgpu::RenderPassColorAttachment<'_>> {
    Some(wgpu::RenderPassColorAttachment {
        view,
        depth_slice: None,
        resolve_target: None,
        ops: wgpu::Operations {
            load: wgpu::LoadOp::Clear(clear_color),
            store: wgpu::StoreOp::Store,
        },
    })
}

fn begin_color_only_pass<'a>(
    encoder: &'a mut wgpu::CommandEncoder,
    label: &'static str,
    color_attachments: &[Option<wgpu::RenderPassColorAttachment<'a>>],
) -> wgpu::RenderPass<'a> {
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments,
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    })
}

fn record_scratch_single_patches<'a>(
    meshes: &'a OverlayMeshPool,
    pass: &mut wgpu::RenderPass<'a>,
    patches: &'a [DrawOverlayPatch],
    offsets: &[Option<u32>],
    arena: &'a DynamicUniformArena,
    mut resolve_tex_bg: impl FnMut(&'a Arc<GpuTile>) -> Option<&'a wgpu::BindGroup>,
) {
    let uniform_bg = arena.bind_group(UniformSlot::ScratchSingle);
    let mut last_idx_key: Option<(u32, bool)> = None;
    let mut last_tex_id: Option<u64> = None;
    for (patch, &dyn_offset) in patches.iter().zip(offsets.iter()) {
        let Some(dyn_offset) = dyn_offset else {
            continue;
        };
        let Some(mesh) = meshes.peek(&patch.mesh_key) else {
            continue;
        };
        let Some(tex_bg) = resolve_tex_bg(&patch.texture) else {
            continue;
        };
        if last_tex_id != Some(patch.texture.tile_id) {
            pass.set_bind_group(2, tex_bg, &[]);
            last_tex_id = Some(patch.texture.tile_id);
        }
        meshes.draw_indexed_patch(pass, mesh, &mut last_idx_key, uniform_bg, dyn_offset);
    }
}

pub(super) fn record_scratch_passes<'a>(
    meshes: &'a OverlayMeshPool,
    encoder: &mut wgpu::CommandEncoder,
    prepared: &PreparedBandMath,
    bm: &'a DrawBandMathLayer,
    ctx: &BandMathPassCtx<'a>,
) -> bool {
    let Some([sx, sy, sw, sh]) = prepared.scissor else {
        return false;
    };
    let band_count = bm.bands.len().min(4);
    let float_views = ctx.scratch.float_views();
    if band_count < bm.formula.required_bands() || float_views.len() < band_count {
        return false;
    }

    // 1. Initialize active `1..=band_count` `R32Float` scratch targets via
    //    `LoadOp::Clear(TRANSPARENT)` (eliminating TBDR full-viewport DRAM
    //    readback) + scissored quiet-NaN fill, merged directly with either the
    //    single co-grid MRT pass or each per-band single-target pass.
    if let Some(mrt_offsets) = &prepared.mrt_patch_offsets {
        let attachments: Vec<Option<wgpu::RenderPassColorAttachment<'_>>> = float_views
            [..band_count]
            .iter()
            .map(|view| color_clear_attachment(view, wgpu::Color::TRANSPARENT))
            .collect();
        let mut pass =
            begin_color_only_pass(encoder, "band-math scratch NaN fill + MRT", &attachments);
        pass.set_scissor_rect(sx, sy, sw, sh);
        pass.set_pipeline(ctx.pipelines.clear_for_count(band_count));
        pass.set_bind_group(0, ctx.scratch.nan_clear_bind_group(), &[]);
        pass.draw(0..3, 0..1);

        pass.set_pipeline(ctx.pipelines.mrt_for_count(band_count));
        pass.set_bind_group(0, ctx.camera_bg, &[]);
        let uniform_bg = ctx.arena.bind_group(UniformSlot::ScratchMrt);
        let first = &bm.bands[0];
        let mut last_idx_key: Option<(u32, bool)> = None;
        let mut last_mrt_ids: Option<[u64; 4]> = None;
        for (patch_idx, (patch, &dyn_offset)) in
            first.patches.iter().zip(mrt_offsets.iter()).enumerate()
        {
            let Some(dyn_offset) = dyn_offset else {
                continue;
            };
            let Some(mesh) = meshes.peek(&patch.mesh_key) else {
                continue;
            };
            let Some(tiles) = mrt_tiles_for_patch(bm, patch_idx) else {
                continue;
            };
            let Some(mrt_bg) = ctx.bind_groups.get_mrt(tiles) else {
                continue;
            };
            let mrt_ids = tiles.map(|t| t.tile_id);
            if last_mrt_ids != Some(mrt_ids) {
                pass.set_bind_group(2, mrt_bg, &[]);
                last_mrt_ids = Some(mrt_ids);
            }
            meshes.draw_indexed_patch(&mut pass, mesh, &mut last_idx_key, uniform_bg, dyn_offset);
        }
    } else {
        for ((band, offsets), view) in bm
            .bands
            .iter()
            .zip(prepared.per_band_offsets.iter())
            .zip(float_views[..band_count].iter())
        {
            let attachments = [color_clear_attachment(view, wgpu::Color::TRANSPARENT)];
            let mut pass = begin_color_only_pass(
                encoder,
                "band-math single-band NaN fill + R32Float scratch pass",
                &attachments,
            );
            pass.set_scissor_rect(sx, sy, sw, sh);
            pass.set_pipeline(ctx.pipelines.clear_for_count(1));
            pass.set_bind_group(0, ctx.scratch.nan_clear_bind_group(), &[]);
            pass.draw(0..3, 0..1);

            pass.set_pipeline(&ctx.pipelines.band_single);
            pass.set_bind_group(0, ctx.camera_bg, &[]);
            record_scratch_single_patches(
                meshes,
                &mut pass,
                &band.patches,
                offsets,
                ctx.arena,
                |t| ctx.bind_groups.get(t, PixelFormat::R32Float),
            );
        }
    }

    // 2. Optional nearest-neighbor `R8Uint` `SCL` mask pass (cleared to 255u sentinel).
    if !prepared.mask_offsets.is_empty()
        && let Some(scl) = &bm.scl
        && let Some(mask_view) = ctx.scratch.mask_view()
    {
        let clear = wgpu::Color {
            r: f64::from(MASK_UNCOVERED_SENTINEL),
            g: 0.0,
            b: 0.0,
            a: 0.0,
        };
        let attachments = [color_clear_attachment(mask_view, clear)];
        let mut pass = begin_color_only_pass(
            encoder,
            "band-math R8Uint SCL mask scratch pass",
            &attachments,
        );
        pass.set_scissor_rect(sx, sy, sw, sh);
        pass.set_pipeline(&ctx.pipelines.mask_u8);
        pass.set_bind_group(0, ctx.camera_bg, &[]);
        record_scratch_single_patches(
            meshes,
            &mut pass,
            &scl.patches,
            &prepared.mask_offsets,
            ctx.arena,
            |t| ctx.bind_groups.get_uint(t),
        );
    }

    true
}

pub(super) fn record_composite<'a>(
    palettes: &'a PaletteCache,
    pass: &mut wgpu::RenderPass<'a>,
    prepared: &PreparedBandMath,
    full_width: u32,
    full_height: u32,
    ctx: &BandMathPassCtx<'a>,
) {
    let (Some([sx, sy, sw, sh]), Some(comp_offset)) = (prepared.scissor, prepared.composite_offset)
    else {
        return;
    };
    let Some(scratch_bg) = ctx.scratch.composite_bind_group() else {
        return;
    };
    let Some(palette_bg) = palettes.colormap_bind_group(prepared.colormap) else {
        return;
    };

    pass.set_scissor_rect(sx, sy, sw, sh);
    pass.set_pipeline(&ctx.pipelines.composite);
    pass.set_bind_group(0, scratch_bg, &[]);
    pass.set_bind_group(
        1,
        ctx.arena.bind_group(UniformSlot::BandMathComposite),
        &[comp_offset],
    );
    pass.set_bind_group(2, palette_bg, &[]);
    pass.draw(0..3, 0..1);
    pass.set_scissor_rect(0, 0, full_width, full_height);
}
