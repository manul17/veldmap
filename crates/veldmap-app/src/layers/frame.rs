//! Per-frame UI-thread overlay LOD selection, frustum culling, wanted-set
//! reconciliation, and immutable [`DrawOverlayLayer`] patch assembly.

use std::collections::HashSet;
use std::sync::Arc;

use veldmap_core::geo::CullFrustum;
use veldmap_core::tile::{
    CarrierLayout, TileKey, cell_image_rect, cell_uv_rect, grid_dim, level_segments, level_size,
};

use crate::gpu_tiles::{GpuTileCache, PendingUploads};
use crate::render::{DrawOverlayLayer, DrawOverlayPatch, OverlayMeshKey};

use super::jobs::schedule_layer_rungs;
use super::runtime::OverlayRuntime;
use super::stack::OverlayLayer;

/// Per-frame shared inputs passed down to [`build_and_schedule_layer`].
pub(super) struct LayerFrameCtx<'a> {
    pub(super) runtime: &'a Arc<OverlayRuntime>,
    pub(super) frustum: &'a CullFrustum,
    pub(super) viewport_mpp: f64,
    pub(super) gpu_cache: &'a mut GpuTileCache,
    pub(super) pending: &'a PendingUploads,
    pub(super) total_appetite_weight: u64,
    pub(super) carrier_keys: &'a mut HashSet<TileKey>,
}

/// Select the finest fitting pyramid rung whose visible cells fit within
/// `cap_tiles`, returning `(target_level, visible_cells)`.
fn select_target_rung(
    layer: &mut OverlayLayer,
    sharpest: u8,
    cap_tiles: usize,
    frustum: &CullFrustum,
) -> (u8, Vec<(u32, u32)>) {
    let max_level = layer.meta().max_level();
    let finest = layer
        .meta()
        .finest_fitting_level()
        .map_or(max_level, |fit| sharpest.max(fit));

    let mut level = finest.min(max_level);
    loop {
        let cells = layer.visible_cells(level, frustum);
        if cells.len() <= cap_tiles || level >= max_level {
            return (level, cells);
        }
        level = level.saturating_add(1);
    }
}

/// Build the immutable [`DrawOverlayLayer`] (if any carriers are cached),
/// update `layer.wanted` across all active ladder rungs, reconcile stale
/// pointwise jobs, and schedule missing tiles onto `fast_pool` / `bulk_pool`.
pub(super) fn build_and_schedule_layer(
    layer: &mut OverlayLayer,
    ctx: &mut LayerFrameCtx<'_>,
) -> Option<DrawOverlayLayer> {
    if !layer.is_active() {
        layer.clear_wanted();
        return None;
    }

    let desc = layer.source().description();
    let (img_w, img_h) = (desc.width, desc.height);
    let max_level = layer.meta().max_level();

    let weight = layer.appetite_weight_bytes();
    let appetite_bytes = ctx
        .gpu_cache
        .proportional_pyramid_byte_cap(weight, ctx.total_appetite_weight);
    let cap_tiles = (appetite_bytes / layer.observed_tile_bytes().max(1)).max(1);

    let sharpest = layer
        .frame()
        .sharpest_level(img_w, img_h, ctx.viewport_mpp, max_level);
    let (target_level, target_cells) = select_target_rung(layer, sharpest, cap_tiles, ctx.frustum);

    let ladder = layer.meta().ladder(target_level);
    let mut ladder_cells: Vec<(u8, Vec<(u32, u32)>)> = Vec::with_capacity(ladder.len());
    for &rung in &ladder {
        let cells = if rung == target_level {
            target_cells.clone()
        } else {
            layer.visible_cells(rung, ctx.frustum)
        };
        ladder_cells.push((rung, cells));
    }

    update_wanted_and_reconcile(
        layer,
        img_w,
        img_h,
        &ladder_cells,
        ctx.runtime,
        ctx.gpu_cache,
        ctx.pending,
    );

    let segments = level_segments(layer.frame(), img_w, img_h, target_level);
    let patches = build_draw_patches(
        layer,
        img_w,
        img_h,
        target_level,
        &target_cells,
        ctx.gpu_cache,
        ctx.carrier_keys,
    );

    if !target_cells.is_empty() {
        schedule_layer_rungs(
            layer,
            &ladder_cells,
            ctx.runtime,
            ctx.gpu_cache,
            ctx.pending,
        );
    }

    (!patches.is_empty()).then(|| DrawOverlayLayer {
        opacity: layer.opacity(),
        track: layer.draw_track(),
        frame: Arc::clone(layer.frame()),
        image_width: img_w,
        image_height: img_h,
        segments,
        patches,
    })
}

fn update_wanted_and_reconcile(
    layer: &mut OverlayLayer,
    img_w: u32,
    img_h: u32,
    ladder_cells: &[(u8, Vec<(u32, u32)>)],
    runtime: &OverlayRuntime,
    gpu_cache: &GpuTileCache,
    pending: &PendingUploads,
) {
    let max_level = layer.meta().max_level();
    // Populate `layer.wanted` with apex root tiles AND visible cells across ALL
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
            next_wanted.push(layer.meta().tile_key(max_level, rx, ry));
        }
    }
    for (rung, cells) in ladder_cells {
        for &(x, y) in cells {
            next_wanted.push(layer.meta().tile_key(*rung, x, y));
        }
    }
    layer.replace_wanted(next_wanted);

    runtime.reconcile_pointwise_streams(
        [(layer.meta().stream_key(), layer.wanted())],
        gpu_cache,
        pending,
    );
}

fn build_draw_patches(
    layer: &OverlayLayer,
    img_w: u32,
    img_h: u32,
    target_level: u8,
    target_cells: &[(u32, u32)],
    gpu_cache: &mut GpuTileCache,
    carrier_keys: &mut HashSet<TileKey>,
) -> Vec<DrawOverlayPatch> {
    let mut patches = Vec::with_capacity(target_cells.len());
    for &(x, y) in target_cells {
        let want_key = layer.meta().tile_key(target_level, x, y);
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
                source: layer.source_id(),
                relaid: layer.relaid(),
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
