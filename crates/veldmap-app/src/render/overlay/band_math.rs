//! Multi-asset screen-space `BandMath`, dual-epoch scalar `TemporalDiff`, and
//! 0%-bleed-through dual-epoch `TemporalCrossFade` scratch-target passes
//! (quiet-NaN fill, co-grid MRT or per-band `R32Float` draws, optional `R8Uint`
//! `SCL` mask pass, non-overlapping Epoch A scalar reduce, `Rgba8UnormSrgb`
//! scratch bake) and scissored fullscreen composite recording.

use std::sync::Arc;

use veldmap_core::glam::DVec3;
use veldmap_core::tile::PixelFormat;

use crate::band_math::clamp_rgb_gamma;
use crate::colormap::ColormapId;
use crate::gpu_tiles::GpuTile;

use super::super::bind_groups::TileBindGroupCache;
use super::super::palette_cache::PaletteCache;
use super::super::pipelines::BandMathPipelines;
use super::super::scratch::{
    MASK_UNCOVERED_SENTINEL, QUIET_NAN_BITS, ScratchTargetPool, compute_band_math_scissor_rect,
    compute_patch_sets_scissor_rect,
};
use super::super::snapshot::{
    DrawBandMathLayer, DrawDisplayEpoch, DrawOverlayPatch, DrawScratchBand, DrawScratchMask,
    DrawSnapshot, DrawTemporalCrossFadeLayer, DrawTemporalCrossFadeSource, DrawTemporalDiffLayer,
};
use super::super::swipe::{ResolvedSwipe, intersect_scissor_rects};
use super::super::uniforms::{
    BandMathCompositeUniform, BandMathReduceUniform, DynamicUniformArena, OverlayPatchUniform,
    ScratchMrtPatchUniform, ScratchSinglePatchUniform, UniformSlot,
};
use super::mesh::OverlayMeshPool;
use super::{safe_inv_range, sanitize_scale_offset};

/// Borrowed GPU resources shared across `BandMath` and dual-epoch temporal
/// scratch and composite pass recording.
pub(in crate::render) struct BandMathPassCtx<'a> {
    pub(in crate::render) camera_bg: &'a wgpu::BindGroup,
    pub(in crate::render) pipelines: &'a BandMathPipelines,
    pub(in crate::render) display_scratch: &'a wgpu::RenderPipeline,
    pub(in crate::render) scratch: &'a ScratchTargetPool,
    pub(in crate::render) arena: &'a DynamicUniformArena,
    pub(in crate::render) bind_groups: &'a TileBindGroupCache,
}

/// Prepared dynamic uniform offsets for one float-band + optional `SCL` epoch.
pub(super) struct PreparedBandEpoch {
    pub(super) mrt_patch_offsets: Option<Vec<Option<u32>>>,
    pub(super) per_band_offsets: Vec<Vec<Option<u32>>>,
    pub(super) mask_offsets: Vec<Option<u32>>,
    pub(super) has_active_mask: bool,
}

/// Prepared dynamic uniform offsets and clamped scissor bounds for one
/// `BandMath` layer.
pub(super) struct PreparedBandMath {
    pub(super) scissor: Option<[u32; 4]>,
    pub(super) epoch: PreparedBandEpoch,
    pub(super) composite_offset: Option<u32>,
    pub(super) colormap: ColormapId,
}

/// Prepared dynamic uniform offsets and union scissor bounds `S_AB` for one
/// dual-epoch scalar `TemporalDiff` layer.
pub(super) struct PreparedTemporalDiff {
    pub(super) scissor: Option<[u32; 4]>,
    pub(super) epoch_a: PreparedBandEpoch,
    pub(super) reduce_offset: Option<u32>,
    pub(super) epoch_b: PreparedBandEpoch,
    pub(super) composite_offset: Option<u32>,
    pub(super) colormap: ColormapId,
}

/// Source-specific prepared offsets for a dual-epoch RGB cross-fade layer.
pub(super) enum PreparedCrossFadeKind {
    BandMathRgb {
        epoch_a: PreparedBandEpoch,
        bake_a_offset: u32,
        epoch_b: PreparedBandEpoch,
    },
    DirectDisplay {
        epoch_a_offsets: Vec<Option<u32>>,
        epoch_b_offsets: Vec<Option<u32>>,
    },
}

/// Prepared dynamic uniform offsets and union scissor bounds `S_AB` for one
/// dual-epoch RGB `TemporalCrossFade` layer.
pub(super) struct PreparedTemporalCrossFade {
    pub(super) scissor: Option<[u32; 4]>,
    pub(super) kind: Option<PreparedCrossFadeKind>,
    pub(super) composite_offset: Option<u32>,
}

pub(super) fn mrt_tiles_for_bands(
    bands: &[DrawScratchBand],
    patch_idx: usize,
) -> Option<[&Arc<GpuTile>; 4]> {
    let first = &bands.first()?.patches.get(patch_idx)?.texture;
    let t0 = first;
    let t1 = bands
        .get(1)
        .and_then(|b| b.patches.get(patch_idx))
        .map_or(first, |p| &p.texture);
    let t2 = bands
        .get(2)
        .and_then(|b| b.patches.get(patch_idx))
        .map_or(first, |p| &p.texture);
    let t3 = bands
        .get(3)
        .and_then(|b| b.patches.get(patch_idx))
        .map_or(first, |p| &p.texture);
    Some([t0, t1, t2, t3])
}

/// Frame-level inputs shared across all BandMath / temporal uniform pushes in a frame.
pub(super) struct BandMathUniformCtx<'a> {
    pub(super) meshes: &'a OverlayMeshPool,
    pub(super) snapshot: &'a DrawSnapshot,
    pub(super) eye: DVec3,
    pub(super) width: u32,
    pub(super) height: u32,
}

fn push_mrt_uniforms(
    meshes: &OverlayMeshPool,
    arena: &mut DynamicUniformArena,
    eye: DVec3,
    bands: &[DrawScratchBand],
) -> Vec<Option<u32>> {
    let first = &bands[0];
    let mut offsets = Vec::with_capacity(first.patches.len());
    for (patch_idx, patch) in first.patches.iter().enumerate() {
        let off = meshes.peek(&patch.mesh_key).map(|mesh| {
            let offset_eye = (mesh.origin_ecef_m - eye).as_vec3().to_array();
            let mut uv_rects = [[0.0f32; 4]; 4];
            let mut scale_offsets = [[1.0f32, 0.0, 0.0, 0.0]; 4];
            let mut mip_counts = [1u32; 4];
            for (b_idx, band) in bands.iter().take(4).enumerate() {
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
    offsets
}

fn push_per_band_uniforms(
    meshes: &OverlayMeshPool,
    arena: &mut DynamicUniformArena,
    eye: DVec3,
    bands: &[DrawScratchBand],
) -> Vec<Vec<Option<u32>>> {
    let mut all_band_offs = Vec::with_capacity(bands.len());
    for band in bands {
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
    all_band_offs
}

fn push_mask_uniforms(
    meshes: &OverlayMeshPool,
    arena: &mut DynamicUniformArena,
    eye: DVec3,
    scl: Option<&DrawScratchMask>,
    scl_mask_enabled: bool,
) -> (Vec<Option<u32>>, bool) {
    let has_active_mask = scl.is_some_and(|m| scl_mask_enabled && !m.patches.is_empty());
    let mut mask_offsets = Vec::new();
    if has_active_mask && let Some(scl) = scl {
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
    (mask_offsets, has_active_mask)
}

fn push_band_epoch_uniforms(
    meshes: &OverlayMeshPool,
    arena: &mut DynamicUniformArena,
    eye: DVec3,
    bands: &[DrawScratchBand],
    scl: Option<&DrawScratchMask>,
    scl_mask_enabled: bool,
    use_mrt: bool,
) -> PreparedBandEpoch {
    let (mrt_patch_offsets, per_band_offsets) = if use_mrt && bands.len() >= 2 {
        (
            Some(push_mrt_uniforms(meshes, arena, eye, bands)),
            Vec::new(),
        )
    } else {
        (None, push_per_band_uniforms(meshes, arena, eye, bands))
    };
    let (mask_offsets, has_active_mask) =
        push_mask_uniforms(meshes, arena, eye, scl, scl_mask_enabled);
    PreparedBandEpoch {
        mrt_patch_offsets,
        per_band_offsets,
        mask_offsets,
        has_active_mask,
    }
}

fn empty_prepared_epoch() -> PreparedBandEpoch {
    PreparedBandEpoch {
        mrt_patch_offsets: None,
        per_band_offsets: Vec::new(),
        mask_offsets: Vec::new(),
        has_active_mask: false,
    }
}

fn push_display_epoch_uniforms(
    meshes: &OverlayMeshPool,
    arena: &mut DynamicUniformArena,
    eye: DVec3,
    epoch: &DrawDisplayEpoch,
) -> Vec<Option<u32>> {
    let mut offsets = Vec::with_capacity(epoch.patches.len());
    for patch in &epoch.patches {
        let off = meshes.peek(&patch.mesh_key).map(|mesh| {
            let offset_eye = (mesh.origin_ecef_m - eye).as_vec3().to_array();
            arena.push(&OverlayPatchUniform {
                offset_eye,
                // Outer layer opacity applies once during the final fullscreen
                // cross-fade composite so Epoch A/B scratch buffers stay full alpha.
                opacity: 1.0,
                uv_rect: patch.uv_rect,
            })
        });
        offsets.push(off);
    }
    offsets
}

fn apply_swipe_scissor(
    projected: Option<[u32; 4]>,
    swipe: Option<ResolvedSwipe>,
    layer_id: u64,
    width: u32,
    height: u32,
) -> Option<[u32; 4]> {
    match (projected, swipe) {
        (Some(proj), Some(sw)) => sw
            .scissor_for_layer(layer_id, width, height)
            .and_then(|half| intersect_scissor_rects(proj, half)),
        (proj, None) => proj,
        (None, _) => None,
    }
}

struct RgbCompositeParams {
    mode: u32,
    opacity: f32,
    rgb_min: [f32; 3],
    rgb_max: [f32; 3],
    rgb_gamma: f32,
    blend_t: f32,
    has_active_mask: bool,
    scl_mask_bits: u32,
}

fn push_rgb_composite_uniform(arena: &mut DynamicUniformArena, p: RgbCompositeParams) -> u32 {
    let (r_min, r_inv) = safe_inv_range(p.rgb_min[0], p.rgb_max[0]);
    let (g_min, g_inv) = safe_inv_range(p.rgb_min[1], p.rgb_max[1]);
    let (b_min, b_inv) = safe_inv_range(p.rgb_min[2], p.rgb_max[2]);
    let inv_gamma = 1.0 / clamp_rgb_gamma(p.rgb_gamma);
    arena.push(&BandMathCompositeUniform {
        opacity: p.opacity,
        mode: p.mode,
        stretch_min: 0.0,
        stretch_inv_range: 1.0,
        rgb_min: [r_min, g_min, b_min, inv_gamma],
        rgb_inv_range: [r_inv, g_inv, b_inv, p.blend_t.clamp(0.0, 1.0)],
        invert_colormap: 0,
        scl_mask_enabled: u32::from(p.has_active_mask),
        scl_mask_bits: p.scl_mask_bits,
        diff_slot: 0,
    })
}

pub(super) fn push_band_math_uniforms(
    ctx: &BandMathUniformCtx<'_>,
    arena: &mut DynamicUniformArena,
    layer_id: u64,
    bm: &DrawBandMathLayer,
    use_mrt: bool,
    opacity: f32,
    swipe: Option<ResolvedSwipe>,
) -> PreparedBandMath {
    let has_required_bands = opacity > 0.001 && bm.bands.len() >= bm.formula.required_bands();
    let projected_scissor = has_required_bands
        .then(|| compute_band_math_scissor_rect(&ctx.snapshot.camera, ctx.width, ctx.height, bm))
        .flatten();
    let scissor = apply_swipe_scissor(projected_scissor, swipe, layer_id, ctx.width, ctx.height);
    if scissor.is_none() {
        return PreparedBandMath {
            scissor: None,
            epoch: empty_prepared_epoch(),
            composite_offset: None,
            colormap: bm.colormap,
        };
    }

    let epoch = push_band_epoch_uniforms(
        ctx.meshes,
        arena,
        ctx.eye,
        bm.active_bands(),
        bm.scl.as_ref(),
        bm.scl_mask_enabled,
        use_mrt,
    );
    let (stretch_min, stretch_inv_range) = safe_inv_range(bm.stretch_min, bm.stretch_max);
    let (r_min, r_inv) = safe_inv_range(bm.rgb_min[0], bm.rgb_max[0]);
    let (g_min, g_inv) = safe_inv_range(bm.rgb_min[1], bm.rgb_max[1]);
    let (b_min, b_inv) = safe_inv_range(bm.rgb_min[2], bm.rgb_max[2]);
    let inv_gamma = 1.0 / clamp_rgb_gamma(bm.rgb_gamma);
    let composite_offset = arena.push(&BandMathCompositeUniform {
        opacity,
        mode: bm.formula.shader_mode(),
        stretch_min,
        stretch_inv_range,
        rgb_min: [r_min, g_min, b_min, inv_gamma],
        rgb_inv_range: [r_inv, g_inv, b_inv, 0.0],
        invert_colormap: u32::from(bm.invert_colormap),
        scl_mask_enabled: u32::from(epoch.has_active_mask),
        scl_mask_bits: bm.scl_mask_bits,
        diff_slot: 0,
    });

    PreparedBandMath {
        scissor,
        epoch,
        composite_offset: Some(composite_offset),
        colormap: bm.colormap,
    }
}

pub(super) fn push_temporal_diff_uniforms(
    ctx: &BandMathUniformCtx<'_>,
    arena: &mut DynamicUniformArena,
    layer_id: u64,
    diff: &DrawTemporalDiffLayer,
    (use_mrt_a, use_mrt_b): (bool, bool),
    opacity: f32,
    swipe: Option<ResolvedSwipe>,
) -> PreparedTemporalDiff {
    let n = diff.required_input_bands();
    let ready = opacity > 0.001
        && diff.is_valid_scalar()
        && diff.epoch_a.bands.len() >= n
        && diff.epoch_b.bands.len() >= n;
    let projected = ready
        .then(|| {
            let sets = diff
                .epoch_a
                .active_warp_patch_sets_for(n, false, diff.scl_mask_enabled)
                .chain(
                    diff.epoch_b
                        .active_warp_patch_sets_for(n, false, diff.scl_mask_enabled),
                );
            compute_patch_sets_scissor_rect(&ctx.snapshot.camera, ctx.width, ctx.height, sets)
        })
        .flatten();
    let scissor = apply_swipe_scissor(projected, swipe, layer_id, ctx.width, ctx.height);
    if scissor.is_none() {
        return PreparedTemporalDiff {
            scissor: None,
            epoch_a: empty_prepared_epoch(),
            reduce_offset: None,
            epoch_b: empty_prepared_epoch(),
            composite_offset: None,
            colormap: diff.colormap,
        };
    }

    let epoch_a = push_band_epoch_uniforms(
        ctx.meshes,
        arena,
        ctx.eye,
        &diff.epoch_a.bands[..n],
        diff.epoch_a.scl.as_ref(),
        diff.scl_mask_enabled,
        use_mrt_a,
    );
    let reduce_offset = arena.push(&BandMathReduceUniform {
        mode: diff.shader_mode(),
        scl_mask_enabled: u32::from(epoch_a.has_active_mask),
        scl_mask_bits: diff.scl_mask_bits,
        nan_bits: QUIET_NAN_BITS,
    });
    let epoch_b = push_band_epoch_uniforms(
        ctx.meshes,
        arena,
        ctx.eye,
        &diff.epoch_b.bands[..n],
        diff.epoch_b.scl.as_ref(),
        diff.scl_mask_enabled,
        use_mrt_b,
    );
    let (stretch_min, stretch_inv_range) = safe_inv_range(diff.stretch_min, diff.stretch_max);
    let composite_offset = arena.push(&BandMathCompositeUniform {
        opacity,
        mode: diff.shader_mode(),
        stretch_min,
        stretch_inv_range,
        rgb_min: [0.0, 0.0, 0.0, 1.0],
        rgb_inv_range: [1.0, 1.0, 1.0, 0.0],
        invert_colormap: u32::from(diff.invert_colormap),
        scl_mask_enabled: u32::from(epoch_b.has_active_mask),
        scl_mask_bits: diff.scl_mask_bits,
        diff_slot: n as u32,
    });

    PreparedTemporalDiff {
        scissor,
        epoch_a,
        reduce_offset: Some(reduce_offset),
        epoch_b,
        composite_offset: Some(composite_offset),
        colormap: diff.colormap,
    }
}

pub(super) fn push_temporal_cross_fade_uniforms(
    ctx: &BandMathUniformCtx<'_>,
    arena: &mut DynamicUniformArena,
    layer_id: u64,
    cf: &DrawTemporalCrossFadeLayer,
    (use_mrt_a, use_mrt_b): (bool, bool),
    opacity: f32,
    swipe: Option<ResolvedSwipe>,
) -> PreparedTemporalCrossFade {
    let projected = if opacity <= 0.001 {
        None
    } else {
        match &cf.source {
            DrawTemporalCrossFadeSource::BandMathRgb {
                scl_mask_enabled,
                epoch_a,
                epoch_b,
                ..
            } => {
                let a_ready = epoch_a.bands.len() >= 3;
                let b_ready = epoch_b.bands.len() >= 3;
                if !a_ready && !b_ready {
                    None
                } else {
                    let sets_a = a_ready
                        .then(|| epoch_a.active_warp_patch_sets_for(3, false, *scl_mask_enabled))
                        .into_iter()
                        .flatten();
                    let sets_b = b_ready
                        .then(|| epoch_b.active_warp_patch_sets_for(3, false, *scl_mask_enabled))
                        .into_iter()
                        .flatten();
                    compute_patch_sets_scissor_rect(
                        &ctx.snapshot.camera,
                        ctx.width,
                        ctx.height,
                        sets_a.chain(sets_b),
                    )
                }
            }
            DrawTemporalCrossFadeSource::DirectDisplay { epoch_a, epoch_b } => {
                if epoch_a.patches.is_empty() && epoch_b.patches.is_empty() {
                    None
                } else {
                    compute_patch_sets_scissor_rect(
                        &ctx.snapshot.camera,
                        ctx.width,
                        ctx.height,
                        [epoch_a.warp_patch_set(), epoch_b.warp_patch_set()],
                    )
                }
            }
        }
    };
    let scissor = apply_swipe_scissor(projected, swipe, layer_id, ctx.width, ctx.height);
    if scissor.is_none() {
        return PreparedTemporalCrossFade {
            scissor: None,
            kind: None,
            composite_offset: None,
        };
    }

    let blend_t = if cf.blend_t.is_finite() {
        cf.blend_t.clamp(0.0, 1.0)
    } else {
        0.0
    };
    match &cf.source {
        DrawTemporalCrossFadeSource::BandMathRgb {
            rgb_min,
            rgb_max,
            rgb_gamma,
            scl_mask_enabled,
            scl_mask_bits,
            epoch_a,
            epoch_b,
        } => {
            let prep_a = if epoch_a.bands.len() >= 3 {
                push_band_epoch_uniforms(
                    ctx.meshes,
                    arena,
                    ctx.eye,
                    &epoch_a.bands[..3],
                    epoch_a.scl.as_ref(),
                    *scl_mask_enabled,
                    use_mrt_a,
                )
            } else {
                empty_prepared_epoch()
            };
            let bake_a_offset = push_rgb_composite_uniform(
                arena,
                RgbCompositeParams {
                    mode: 3,
                    opacity: 1.0,
                    rgb_min: *rgb_min,
                    rgb_max: *rgb_max,
                    rgb_gamma: *rgb_gamma,
                    blend_t: 0.0,
                    has_active_mask: prep_a.has_active_mask,
                    scl_mask_bits: *scl_mask_bits,
                },
            );
            let prep_b = if epoch_b.bands.len() >= 3 {
                push_band_epoch_uniforms(
                    ctx.meshes,
                    arena,
                    ctx.eye,
                    &epoch_b.bands[..3],
                    epoch_b.scl.as_ref(),
                    *scl_mask_enabled,
                    use_mrt_b,
                )
            } else {
                empty_prepared_epoch()
            };
            let composite_offset = push_rgb_composite_uniform(
                arena,
                RgbCompositeParams {
                    mode: 5,
                    opacity,
                    rgb_min: *rgb_min,
                    rgb_max: *rgb_max,
                    rgb_gamma: *rgb_gamma,
                    blend_t,
                    has_active_mask: prep_b.has_active_mask,
                    scl_mask_bits: *scl_mask_bits,
                },
            );
            PreparedTemporalCrossFade {
                scissor,
                kind: Some(PreparedCrossFadeKind::BandMathRgb {
                    epoch_a: prep_a,
                    bake_a_offset,
                    epoch_b: prep_b,
                }),
                composite_offset: Some(composite_offset),
            }
        }
        DrawTemporalCrossFadeSource::DirectDisplay { epoch_a, epoch_b } => {
            let epoch_a_offsets = push_display_epoch_uniforms(ctx.meshes, arena, ctx.eye, epoch_a);
            let epoch_b_offsets = push_display_epoch_uniforms(ctx.meshes, arena, ctx.eye, epoch_b);
            let composite_offset = push_rgb_composite_uniform(
                arena,
                RgbCompositeParams {
                    mode: 6,
                    opacity,
                    rgb_min: [0.0; 3],
                    rgb_max: [1.0; 3],
                    rgb_gamma: 1.0,
                    blend_t,
                    has_active_mask: false,
                    scl_mask_bits: 0,
                },
            );
            PreparedTemporalCrossFade {
                scissor,
                kind: Some(PreparedCrossFadeKind::DirectDisplay {
                    epoch_a_offsets,
                    epoch_b_offsets,
                }),
                composite_offset: Some(composite_offset),
            }
        }
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
    uniform_slot: UniformSlot,
    mut resolve_tex_bg: impl FnMut(&'a Arc<GpuTile>) -> Option<&'a wgpu::BindGroup>,
) {
    let uniform_bg = arena.bind_group(uniform_slot);
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

/// Rasterize `band_count` `R32Float` bands (quiet-NaN initialized via merged
/// `LoadOp::Clear(TRANSPARENT)` + scissored NaN fill) into `float_views[0..band_count]`
/// plus optional `R8Uint` `SCL` mask into `mask_view`.
fn record_band_epoch_scratch<'a>(
    meshes: &'a OverlayMeshPool,
    encoder: &mut wgpu::CommandEncoder,
    scissor: [u32; 4],
    bands: &'a [DrawScratchBand],
    scl: Option<&'a DrawScratchMask>,
    prepared: &PreparedBandEpoch,
    ctx: &BandMathPassCtx<'a>,
) -> bool {
    let [sx, sy, sw, sh] = scissor;
    let band_count = bands.len().min(4);
    let float_views = ctx.scratch.float_views();
    if band_count == 0 || float_views.len() < band_count {
        return false;
    }

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
        let first = &bands[0];
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
            let Some(tiles) = mrt_tiles_for_bands(bands, patch_idx) else {
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
        for ((band, offsets), view) in bands
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
                UniformSlot::ScratchSingle,
                |t| ctx.bind_groups.get(t, PixelFormat::R32Float),
            );
        }
    }

    if !prepared.mask_offsets.is_empty()
        && let Some(scl) = scl
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
            UniformSlot::ScratchSingle,
            |t| ctx.bind_groups.get_uint(t),
        );
    }

    true
}

/// Fill `float_views[0..count]` with quiet-NaN across `scissor` (used when one
/// epoch of a `BandMathRgb` cross-fade has no loaded bands yet).
fn clear_float_scratch_to_nan<'a>(
    encoder: &mut wgpu::CommandEncoder,
    scissor: [u32; 4],
    count: usize,
    ctx: &BandMathPassCtx<'a>,
) {
    let [sx, sy, sw, sh] = scissor;
    let float_views = ctx.scratch.float_views();
    let n = count.min(float_views.len()).min(4);
    if n == 0 {
        return;
    }
    let attachments: Vec<Option<wgpu::RenderPassColorAttachment<'_>>> = float_views[..n]
        .iter()
        .map(|view| color_clear_attachment(view, wgpu::Color::TRANSPARENT))
        .collect();
    let mut pass = begin_color_only_pass(encoder, "scratch quiet-NaN clear", &attachments);
    pass.set_scissor_rect(sx, sy, sw, sh);
    pass.set_pipeline(ctx.pipelines.clear_for_count(n));
    pass.set_bind_group(0, ctx.scratch.nan_clear_bind_group(), &[]);
    pass.draw(0..3, 0..1);
}

pub(super) fn record_scratch_passes<'a>(
    meshes: &'a OverlayMeshPool,
    encoder: &mut wgpu::CommandEncoder,
    prepared: &PreparedBandMath,
    bm: &'a DrawBandMathLayer,
    ctx: &BandMathPassCtx<'a>,
) -> bool {
    let Some(scissor) = prepared.scissor else {
        return false;
    };
    if bm.bands.len() < bm.formula.required_bands() {
        return false;
    }
    record_band_epoch_scratch(
        meshes,
        encoder,
        scissor,
        bm.active_bands(),
        bm.scl.as_ref(),
        &prepared.epoch,
        ctx,
    )
}

/// Record the 3-step dual-epoch scalar diff scratch pipeline:
/// 1. Rasterize Epoch A (`b0..bN` -> `scratch[0..N]`, `SCL_A` -> `mask`).
/// 2. Reduce Epoch A via the non-overlapping read-only `reduce_bind_group(N)`
///    into `scratch[N]` (baking masked/uncovered pixels to quiet-NaN).
/// 3. Rasterize Epoch B (`b0..bN` -> `scratch[0..N]`, `SCL_B` -> `mask`,
///    leaving `scratch[N]` intact).
pub(super) fn record_temporal_diff_scratch_passes<'a>(
    meshes: &'a OverlayMeshPool,
    encoder: &mut wgpu::CommandEncoder,
    prepared: &PreparedTemporalDiff,
    diff: &'a DrawTemporalDiffLayer,
    ctx: &BandMathPassCtx<'a>,
) -> bool {
    let (Some(scissor), Some(reduce_offset)) = (prepared.scissor, prepared.reduce_offset) else {
        return false;
    };
    if !diff.is_valid_scalar() {
        return false;
    }
    let n = diff.required_input_bands();
    let float_views = ctx.scratch.float_views();
    if diff.epoch_a.bands.len() < n || diff.epoch_b.bands.len() < n || float_views.len() <= n {
        return false;
    }
    let Some(reduce_bg) = ctx.scratch.reduce_bind_group(n) else {
        return false;
    };

    // 1. Rasterize Epoch A into `scratch[0..N]` + `mask`.
    if !record_band_epoch_scratch(
        meshes,
        encoder,
        scissor,
        &diff.epoch_a.bands[..n],
        diff.epoch_a.scl.as_ref(),
        &prepared.epoch_a,
        ctx,
    ) {
        return false;
    }

    // 2. Reduce Epoch A from non-overlapping `reduce_bg` into `scratch[N]`.
    {
        let [sx, sy, sw, sh] = scissor;
        let attachments = [color_clear_attachment(
            &float_views[n],
            wgpu::Color::TRANSPARENT,
        )];
        let mut pass = begin_color_only_pass(
            encoder,
            "temporal-diff Epoch A scalar reduce pass",
            &attachments,
        );
        pass.set_scissor_rect(sx, sy, sw, sh);
        pass.set_pipeline(&ctx.pipelines.reduce);
        pass.set_bind_group(0, reduce_bg, &[]);
        pass.set_bind_group(
            1,
            ctx.arena.bind_group(UniformSlot::BandMathReduce),
            &[reduce_offset],
        );
        pass.draw(0..3, 0..1);
    }

    // 3. Rasterize Epoch B into `scratch[0..N]` + `mask` (leaving `scratch[N]` intact).
    record_band_epoch_scratch(
        meshes,
        encoder,
        scissor,
        &diff.epoch_b.bands[..n],
        diff.epoch_b.scl.as_ref(),
        &prepared.epoch_b,
        ctx,
    )
}

/// Record the scratch passes for a 0%-bleed-through dual-epoch RGB cross-fade:
/// - `BandMathRgb`: rasterize Epoch A -> bake into `rgba_scratch[0]` (`blend: None`,
///   binding `dummy_rgba_bind_group` on `@group(3)` so zero read/write hazard)
///   -> rasterize Epoch B into `scratch[0..3]` + `mask`.
/// - `DirectDisplay`: rasterize Epoch A into `rgba_scratch[0]` and Epoch B into
///   `rgba_scratch[1]` (`blend: None`, `8 B/px`, zero `R32Float`).
pub(super) fn record_temporal_cross_fade_scratch_passes<'a>(
    meshes: &'a OverlayMeshPool,
    palettes: &'a PaletteCache,
    encoder: &mut wgpu::CommandEncoder,
    prepared: &PreparedTemporalCrossFade,
    cf: &'a DrawTemporalCrossFadeLayer,
    ctx: &BandMathPassCtx<'a>,
) -> bool {
    let (Some(scissor), Some(kind)) = (prepared.scissor, &prepared.kind) else {
        return false;
    };
    let [sx, sy, sw, sh] = scissor;
    let rgba_views = ctx.scratch.rgba_views();

    match (&cf.source, kind) {
        (
            DrawTemporalCrossFadeSource::BandMathRgb {
                epoch_a, epoch_b, ..
            },
            PreparedCrossFadeKind::BandMathRgb {
                epoch_a: prep_a,
                bake_a_offset,
                epoch_b: prep_b,
            },
        ) => {
            if rgba_views.is_empty() || ctx.scratch.float_views().len() < 3 {
                return false;
            }
            let Some(scratch_bg) = ctx.scratch.composite_bind_group() else {
                return false;
            };
            let Some(palette_bg) = palettes.colormap_bind_group(ColormapId::Viridis) else {
                return false;
            };

            // 1. Rasterize Epoch A and bake stretched RGB + mask into
            //    `rgba_scratch[0]` (`blend: None`), or leave `rgba_scratch[0]`
            //    transparent if Epoch A is not yet loaded.
            let baked_a = epoch_a.bands.len() >= 3
                && record_band_epoch_scratch(
                    meshes,
                    encoder,
                    scissor,
                    &epoch_a.bands[..3],
                    epoch_a.scl.as_ref(),
                    prep_a,
                    ctx,
                );
            {
                let attachments = [color_clear_attachment(
                    &rgba_views[0],
                    wgpu::Color::TRANSPARENT,
                )];
                let mut pass = begin_color_only_pass(
                    encoder,
                    "temporal-crossfade BandMath RGB Epoch A bake",
                    &attachments,
                );
                if baked_a {
                    pass.set_scissor_rect(sx, sy, sw, sh);
                    pass.set_pipeline(&ctx.pipelines.composite_rgba_bake);
                    pass.set_bind_group(0, scratch_bg, &[]);
                    pass.set_bind_group(
                        1,
                        ctx.arena.bind_group(UniformSlot::BandMathComposite),
                        &[*bake_a_offset],
                    );
                    pass.set_bind_group(2, palette_bg, &[]);
                    // Bind immutable 1x1 dummy on @group(3) while writing rgba_views[0]!
                    pass.set_bind_group(3, ctx.scratch.dummy_rgba_bind_group(), &[]);
                    pass.draw(0..3, 0..1);
                }
            }

            // 2. Rasterize Epoch B into `scratch[0..3]` + `mask` (or quiet-NaN clear).
            let wrote_b = epoch_b.bands.len() >= 3
                && record_band_epoch_scratch(
                    meshes,
                    encoder,
                    scissor,
                    &epoch_b.bands[..3],
                    epoch_b.scl.as_ref(),
                    prep_b,
                    ctx,
                );
            if !wrote_b {
                clear_float_scratch_to_nan(encoder, scissor, 3, ctx);
            }
            true
        }
        (
            DrawTemporalCrossFadeSource::DirectDisplay { epoch_a, epoch_b },
            PreparedCrossFadeKind::DirectDisplay {
                epoch_a_offsets,
                epoch_b_offsets,
            },
        ) => {
            if rgba_views.len() < 2 {
                return false;
            }
            for (slot_idx, (epoch, offsets)) in [
                (epoch_a, epoch_a_offsets.as_slice()),
                (epoch_b, epoch_b_offsets.as_slice()),
            ]
            .into_iter()
            .enumerate()
            {
                let attachments = [color_clear_attachment(
                    &rgba_views[slot_idx],
                    wgpu::Color::TRANSPARENT,
                )];
                let mut pass = begin_color_only_pass(
                    encoder,
                    if slot_idx == 0 {
                        "temporal-crossfade Direct Display Epoch A scratch"
                    } else {
                        "temporal-crossfade Direct Display Epoch B scratch"
                    },
                    &attachments,
                );
                if !epoch.patches.is_empty() {
                    pass.set_scissor_rect(sx, sy, sw, sh);
                    pass.set_pipeline(ctx.display_scratch);
                    pass.set_bind_group(0, ctx.camera_bg, &[]);
                    record_scratch_single_patches(
                        meshes,
                        &mut pass,
                        &epoch.patches,
                        offsets,
                        ctx.arena,
                        UniformSlot::Overlay,
                        |t| ctx.bind_groups.get(t, PixelFormat::Rgba8UnormSrgb),
                    );
                }
            }
            true
        }
        _ => false,
    }
}

/// Draw parameters for a scissored fullscreen composite draw.
pub(super) struct CompositeDrawSpec {
    pub(super) scissor: Option<[u32; 4]>,
    pub(super) composite_offset: Option<u32>,
    pub(super) colormap: ColormapId,
    pub(super) use_rgba_scratch: bool,
    pub(super) viewport: [u32; 2],
}

pub(super) fn record_composite_draw<'a>(
    palettes: &'a PaletteCache,
    pass: &mut wgpu::RenderPass<'a>,
    spec: CompositeDrawSpec,
    ctx: &BandMathPassCtx<'a>,
) {
    let (Some([sx, sy, sw, sh]), Some(comp_offset)) = (spec.scissor, spec.composite_offset) else {
        return;
    };
    let Some(scratch_bg) = ctx.scratch.composite_bind_group() else {
        return;
    };
    let Some(palette_bg) = palettes.colormap_bind_group(spec.colormap) else {
        return;
    };
    let rgba_bg = if spec.use_rgba_scratch {
        let Some(bg) = ctx.scratch.rgba_bind_group() else {
            return;
        };
        bg
    } else {
        ctx.scratch.dummy_rgba_bind_group()
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
    pass.set_bind_group(3, rgba_bg, &[]);
    pass.draw(0..3, 0..1);
    pass.set_scissor_rect(0, 0, spec.viewport[0], spec.viewport[1]);
}
