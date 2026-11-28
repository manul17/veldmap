//! Per-frame UI-thread overlay LOD selection, frustum culling, wanted-set
//! reconciliation, and immutable [`DrawOverlayLayer`] patch assembly across
//! both `SingleRaster` and multi-asset `BandMath` layers.

use std::collections::HashSet;
use std::sync::Arc;

use veldmap_core::geo::CullFrustum;
use veldmap_core::tile::{
    CarrierLayout, TileKey, cell_image_rect, cell_uv_rect, grid_dim, level_segments, level_size,
};

use crate::gpu_tiles::{GpuTileCache, PendingUploads};
use crate::render::{
    DrawBandMathLayer, DrawOverlayContent, DrawOverlayLayer, DrawOverlayPatch, DrawScratchBand,
    DrawScratchMask, OverlayMeshKey,
};

use super::asset::RasterAsset;
use super::jobs::schedule_asset_rungs;
use super::layer::{LayerKind, OverlayLayer};
use super::runtime::OverlayRuntime;
use super::style::LayerStyle;

/// Shared read-only inputs for Phase 1 (`prepare_layer_wanted`), before the
/// frame-wide unioned `reconcile_pointwise_streams` pass runs.
pub(super) struct PrepareFrameCtx<'a> {
    pub(super) frustum: &'a CullFrustum,
    pub(super) viewport_mpp: f64,
    pub(super) gpu_cache: &'a GpuTileCache,
    pub(super) total_appetite_weight: u64,
}

/// Per-frame shared inputs passed down to [`finish_scheduled_layer`].
pub(super) struct LayerFrameCtx<'a> {
    pub(super) runtime: &'a Arc<OverlayRuntime>,
    pub(super) gpu_cache: &'a mut GpuTileCache,
    pub(super) pending: &'a PendingUploads,
    pub(super) carrier_keys: &'a mut HashSet<TileKey>,
}

pub(super) struct PreparedAssetPlan {
    img_w: u32,
    img_h: u32,
    target_level: u8,
    target_cells: Vec<(u32, u32)>,
    ladder_cells: Vec<(u8, Vec<(u32, u32)>)>,
}

pub(super) enum PreparedLayerPlan {
    Single(PreparedAssetPlan),
    BandMath {
        bands: Vec<PreparedAssetPlan>,
        scl: Option<PreparedAssetPlan>,
    },
}

struct ScheduledAsset {
    image_width: u32,
    image_height: u32,
    segments: u32,
    patches: Vec<DrawOverlayPatch>,
}

/// Select the finest fitting pyramid rung whose visible cells fit within
/// `cap_tiles`, returning `(target_level, visible_cells)`.
fn select_target_rung(
    asset: &mut RasterAsset,
    sharpest: u8,
    cap_tiles: usize,
    frustum: &CullFrustum,
) -> (u8, Vec<(u32, u32)>) {
    let max_level = asset.meta().max_level();
    let finest = asset
        .meta()
        .finest_fitting_level()
        .map_or(max_level, |fit| sharpest.max(fit));

    let mut level = finest.min(max_level);
    loop {
        let cells = asset.geo_mut().visible_cells(level, frustum);
        if cells.len() <= cap_tiles || level >= max_level {
            return (level, cells);
        }
        level = level.saturating_add(1);
    }
}

fn prepare_single_asset(asset: &mut RasterAsset, ctx: &PrepareFrameCtx<'_>) -> PreparedAssetPlan {
    let desc = asset.source().description();
    let (img_w, img_h) = (desc.width, desc.height);
    let max_level = asset.meta().max_level();

    let weight = asset.appetite_weight_bytes();
    let appetite_bytes = ctx
        .gpu_cache
        .proportional_pyramid_byte_cap(weight, ctx.total_appetite_weight);
    let cap_tiles = (appetite_bytes / asset.observed_tile_bytes()).max(1);

    let sharpest = asset
        .geo()
        .frame()
        .sharpest_level(img_w, img_h, ctx.viewport_mpp, max_level);
    let (target_level, target_cells) = select_target_rung(asset, sharpest, cap_tiles, ctx.frustum);

    let ladder = asset.meta().ladder(target_level);
    let mut ladder_cells: Vec<(u8, Vec<(u32, u32)>)> = Vec::with_capacity(ladder.len());
    for &rung in &ladder {
        let cells = if rung == target_level {
            target_cells.clone()
        } else {
            asset.geo_mut().visible_cells(rung, ctx.frustum)
        };
        ladder_cells.push((rung, cells));
    }

    update_wanted(asset, img_w, img_h, &ladder_cells);

    PreparedAssetPlan {
        img_w,
        img_h,
        target_level,
        target_cells,
        ladder_cells,
    }
}

fn finish_single_asset(
    asset: &RasterAsset,
    plan: PreparedAssetPlan,
    ctx: &mut LayerFrameCtx<'_>,
) -> ScheduledAsset {
    let segments = level_segments(
        asset.geo().frame(),
        plan.img_w,
        plan.img_h,
        plan.target_level,
    );
    let patches = build_draw_patches(
        asset,
        plan.img_w,
        plan.img_h,
        plan.target_level,
        &plan.target_cells,
        ctx.gpu_cache,
        ctx.carrier_keys,
    );

    if !plan.target_cells.is_empty() {
        schedule_asset_rungs(
            asset,
            &plan.ladder_cells,
            ctx.runtime,
            ctx.gpu_cache,
            ctx.pending,
        );
    }

    ScheduledAsset {
        image_width: plan.img_w,
        image_height: plan.img_h,
        segments,
        patches,
    }
}

/// Phase 1: compute visible target + ladder rungs and update `wanted` for all
/// active assets of `layer` (clearing `wanted` on inactive layers or disabled
/// `SCL` masks) before the frame-wide unioned reconcile runs.
pub(super) fn prepare_layer_wanted(
    layer: &mut OverlayLayer,
    ctx: &PrepareFrameCtx<'_>,
) -> Option<PreparedLayerPlan> {
    if !layer.is_active() {
        layer.clear_wanted();
        return None;
    }

    match layer.kind_mut() {
        LayerKind::SingleRaster { asset, .. } => {
            Some(PreparedLayerPlan::Single(prepare_single_asset(asset, ctx)))
        }
        LayerKind::BandMath(bm) => {
            let bands = bm
                .bands
                .iter_mut()
                .map(|band_asset| prepare_single_asset(band_asset, ctx))
                .collect();
            let scl = if let Some(scl_asset) = &mut bm.scl {
                if bm.style.scl_mask_enabled {
                    Some(prepare_single_asset(scl_asset, ctx))
                } else {
                    scl_asset.clear_wanted();
                    None
                }
            } else {
                None
            };
            Some(PreparedLayerPlan::BandMath { bands, scl })
        }
    }
}

/// Phase 3: build the immutable [`DrawOverlayLayer`] from cached carrier tiles
/// and submit missing tiles onto `fast_pool` / `bulk_pool`.
pub(super) fn finish_scheduled_layer(
    layer: &OverlayLayer,
    plan: PreparedLayerPlan,
    ctx: &mut LayerFrameCtx<'_>,
) -> Option<DrawOverlayLayer> {
    let opacity = layer.opacity();
    match (layer.kind(), plan) {
        (LayerKind::SingleRaster { asset, style }, PreparedLayerPlan::Single(asset_plan)) => {
            let scheduled = finish_single_asset(asset, asset_plan, ctx);
            let track = style.draw_track(asset.source().as_ref(), asset.meta().spec);
            (!scheduled.patches.is_empty()).then(|| DrawOverlayLayer {
                layer_id: layer.id().raw(),
                opacity,
                content: DrawOverlayContent::Direct {
                    track,
                    frame: Arc::clone(asset.geo().frame()),
                    image_width: scheduled.image_width,
                    image_height: scheduled.image_height,
                    segments: scheduled.segments,
                    patches: scheduled.patches,
                },
            })
        }
        (
            LayerKind::BandMath(bm),
            PreparedLayerPlan::BandMath {
                bands: band_plans,
                scl: scl_plan,
            },
        ) => {
            let mut draw_bands = Vec::with_capacity(bm.bands.len());
            let mut all_bands_have_patches = true;
            for (band_asset, band_plan) in bm.bands.iter().zip(band_plans) {
                let (scale, offset) = band_asset.band_scale_offset();
                let scheduled = finish_single_asset(band_asset, band_plan, ctx);
                if scheduled.patches.is_empty() {
                    all_bands_have_patches = false;
                }
                draw_bands.push(DrawScratchBand {
                    frame: Arc::clone(band_asset.geo().frame()),
                    image_width: scheduled.image_width,
                    image_height: scheduled.image_height,
                    segments: scheduled.segments,
                    scale: scale as f32,
                    offset: offset as f32,
                    patches: scheduled.patches,
                });
            }

            let draw_scl = if let (Some(scl_asset), Some(scl_plan)) = (&bm.scl, scl_plan) {
                // Never discard maskable SCL class IDs (`0..32`, notably class
                // `0` = ESA "No Data" declared as `GDAL_NODATA = 0` on real
                // Sentinel-2 COGs): write them into the `R8Uint` scratch target
                // so `scl_mask_bits` controls them while `255u` stays strictly
                // the uncovered-footprint sentinel.
                let nodata_value = LayerStyle::categorical_nodata_value(
                    scl_asset.source().description(),
                    scl_asset.meta().spec.band,
                )
                .filter(|&nd| nd >= super::band_math::MAX_SCL_MASK_CLASSES);
                let scheduled = finish_single_asset(scl_asset, scl_plan, ctx);
                (!scheduled.patches.is_empty()).then(|| DrawScratchMask {
                    frame: Arc::clone(scl_asset.geo().frame()),
                    image_width: scheduled.image_width,
                    image_height: scheduled.image_height,
                    segments: scheduled.segments,
                    nodata_value,
                    patches: scheduled.patches,
                })
            } else {
                None
            };

            // Only emit a BandMath draw once every required input band has at
            // least one cached carrier tile so we avoid allocating scratch
            // targets for all-NaN frames while bands are still opening/streaming.
            (all_bands_have_patches && !draw_bands.is_empty()).then(|| DrawOverlayLayer {
                layer_id: layer.id().raw(),
                opacity,
                content: DrawOverlayContent::BandMath(Box::new(DrawBandMathLayer {
                    formula: bm.style.formula,
                    colormap: bm.style.colormap,
                    invert_colormap: bm.style.invert_colormap,
                    stretch_min: bm.style.stretch_min as f32,
                    stretch_max: bm.style.stretch_max as f32,
                    rgb_min: bm.style.rgb_min.map(|v| v as f32),
                    rgb_max: bm.style.rgb_max.map(|v| v as f32),
                    rgb_gamma: bm.style.rgb_gamma,
                    scl_mask_enabled: bm.style.scl_mask_enabled,
                    scl_mask_bits: bm.style.scl_mask_bits,
                    bands: draw_bands,
                    scl: draw_scl,
                })),
            })
        }
        _ => None,
    }
}

fn update_wanted(
    asset: &mut RasterAsset,
    img_w: u32,
    img_h: u32,
    ladder_cells: &[(u8, Vec<(u32, u32)>)],
) {
    let max_level = asset.meta().max_level();
    // Populate `wanted` with apex root tiles AND visible cells across ALL
    // ladder rungs so `PendingUploads` never prunes intermediate coarse rungs
    // during progressive refinement.
    let top_w = grid_dim(level_size(img_w, u32::from(max_level)));
    let top_h = grid_dim(level_size(img_h, u32::from(max_level)));
    let mut next_wanted = Vec::with_capacity(
        (top_w as usize) * (top_h as usize)
            + ladder_cells
                .iter()
                .map(|(_, cells)| cells.len())
                .sum::<usize>(),
    );
    for ry in 0..top_h {
        for rx in 0..top_w {
            next_wanted.push(asset.meta().tile_key(max_level, rx, ry));
        }
    }
    for (rung, cells) in ladder_cells {
        for &(x, y) in cells {
            next_wanted.push(asset.meta().tile_key(*rung, x, y));
        }
    }
    asset.stream_mut().replace_wanted(next_wanted);
}

fn build_draw_patches(
    asset: &RasterAsset,
    img_w: u32,
    img_h: u32,
    target_level: u8,
    target_cells: &[(u32, u32)],
    gpu_cache: &mut GpuTileCache,
    carrier_keys: &mut HashSet<TileKey>,
) -> Vec<DrawOverlayPatch> {
    let mut patches = Vec::with_capacity(target_cells.len());
    for &(x, y) in target_cells {
        let want_key = asset.meta().tile_key(target_level, x, y);
        let Some((carrier_key, gpu_tile)) = gpu_cache.find_carrier(&want_key) else {
            continue;
        };
        carrier_keys.insert(carrier_key);

        let cell_rect = cell_image_rect(x, y, u32::from(target_level), img_w, img_h);
        let uv_rect = cell_uv_rect(
            cell_rect,
            &carrier_key,
            CarrierLayout {
                width: gpu_tile.width,
                height: gpu_tile.height,
                valid_w: gpu_tile.valid_w,
                valid_h: gpu_tile.valid_h,
                gutter: gpu_tile.gutter,
            },
        );
        patches.push(DrawOverlayPatch {
            mesh_key: OverlayMeshKey {
                source: asset.source_id(),
                relaid: asset.geo().relaid(),
                level: target_level,
                x,
                y,
            },
            cell_rect,
            uv_rect,
            texture: gpu_tile,
        });
    }
    patches
}
