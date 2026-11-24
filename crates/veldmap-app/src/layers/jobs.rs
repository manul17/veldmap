//! Per-frame overlay LOD selection, two-lane pointwise/pass decode scheduling,
//! linear-light mipmap generation, and ABA-safe QOI SQLite disk caching.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use veldmap_core::geo::CullFrustum;
use veldmap_core::sync::{CancelToken, lock_unpoisoned};
use veldmap_core::tile::{
    DecodedTile, PixelFormat, TileKey, cell_image_rect, cell_uv, grid_dim, level_segments,
    level_size,
};
use veldmap_raster::{
    AdmittedTile, DecodeContext, DecodeJob, DecodeJobKey, MAX_QOI_DECODED_BYTES, PassKey,
    PyramidMeta, RasterError, RasterSource, ServeMode, SubmitError, decode_qoi_rgba8,
    encode_qoi_rgba8, generate_linear_mipmaps,
};
use veldmap_storage::{DiskTileKey, TileBlobEncoding, TileDiskCache};

use crate::gpu_tiles::{GpuTileCache, PendingUploads};
use crate::render::{DrawOverlayLayer, DrawOverlayPatch, OverlayMeshKey};

use super::default_display_tile_bytes;
use super::runtime::{OverlayRuntime, PassFlightGuard, PointwiseFlightGuard};
use super::stack::OverlayLayer;

/// Maximum QOI tiles buffered before committing a single SQLite batch write
/// during `ServeMode::Pass`.
const PASS_QOI_BATCH_MAX_TILES: usize = 16;

/// Maximum encoded QOI bytes buffered before flushing a batch write (`4 MiB`).
const PASS_QOI_BATCH_MAX_BYTES: usize = 4 * 1024 * 1024;

/// Look up and decode a lossless QOI RGBA8 tile from `disk`, recording its
/// debounced LRU touch ONLY after validation succeeds and purging corrupt rows
/// safely by `(tile_id, row_last_used_ms)`.
fn load_qoi_display_tile(disk: &TileDiskCache, disk_key: &DiskTileKey) -> Option<DecodedTile> {
    let (entry, _, payload) = disk
        .get_encoded_entry(disk_key, Some(TileBlobEncoding::Qoi))
        .ok()??;
    match decode_qoi_rgba8(&payload, MAX_QOI_DECODED_BYTES) {
        Ok((w, h, rgba)) => {
            if let Ok(mut tile) = DecodedTile::try_new(w, h, PixelFormat::Rgba8UnormSrgb, rgba) {
                disk.touch_entry(entry);
                generate_linear_mipmaps(&mut tile);
                Some(tile)
            } else {
                let _ = disk.delete_entry(entry);
                None
            }
        }
        Err(err) => {
            tracing::warn!(
                source_key = %disk_key.source_key,
                level = disk_key.level,
                x = disk_key.x,
                y = disk_key.y,
                error = %err,
                "Purging corrupt QOI disk tile entry"
            );
            let _ = disk.delete_entry(entry);
            None
        }
    }
}

fn store_qoi_display_tile(
    disk: &TileDiskCache,
    disk_key: &DiskTileKey,
    width: u32,
    height: u32,
    rgba: &[u8],
    cancel: &CancelToken,
) {
    if disk.max_bytes() == 0 || cancel.is_cancelled() {
        return;
    }
    if let Ok(encoded) = encode_qoi_rgba8(width, height, rgba, MAX_QOI_DECODED_BYTES) {
        let _ = disk.put_encoded_if_uncancelled(disk_key, TileBlobEncoding::Qoi, &encoded, cancel);
    }
}

fn flush_qoi_batch(
    disk: &TileDiskCache,
    batch: &mut Vec<(DiskTileKey, Vec<u8>)>,
    batch_bytes: &mut usize,
    cancel: &CancelToken,
) {
    if batch.is_empty() {
        return;
    }
    if !cancel.is_cancelled() && disk.max_bytes() > 0 {
        let refs: Vec<(&DiskTileKey, TileBlobEncoding, &[u8])> = batch
            .iter()
            .map(|(k, bytes)| (k, TileBlobEncoding::Qoi, bytes.as_slice()))
            .collect();
        let _ = disk.put_encoded_batch_if_uncancelled(&refs, cancel);
    }
    batch.clear();
    *batch_bytes = 0;
}

/// Select the finest fitting pyramid rung whose visible cells fit within
/// `cap_tiles`, returning `(target_level, visible_cells)`.
pub(super) fn select_target_rung(
    layer: &mut OverlayLayer,
    sharpest: u8,
    cap_tiles: usize,
    frustum: &CullFrustum,
) -> (u8, Vec<(u32, u32)>) {
    let max_level = layer.meta.max_level();
    let finest = layer
        .meta
        .finest_fitting_level()
        .map_or(max_level, |fit| sharpest.max(fit));

    let mut level = finest.min(max_level);
    loop {
        let cells = layer.cull.visible_cells(&layer.frame, level, frustum);
        if cells.len() <= cap_tiles || level >= max_level {
            return (level, cells);
        }
        level = level.saturating_add(1);
    }
}

/// Build the immutable [`DrawOverlayLayer`] (if any carriers are cached),
/// update `layer.wanted` across all active ladder rungs, reconcile stale
/// pointwise jobs, and schedule missing tiles onto `fast_pool` / `bulk_pool`.
#[expect(
    clippy::too_many_arguments,
    reason = "frame orchestration crosses cache, frustum, budget, and runtime"
)]
pub(super) fn build_and_schedule_layer(
    layer: &mut OverlayLayer,
    runtime: &Arc<OverlayRuntime>,
    frustum: &CullFrustum,
    viewport_mpp: f64,
    gpu_cache: &mut GpuTileCache,
    pending: &PendingUploads,
    total_appetite_weight: u64,
    carrier_keys: &mut HashSet<TileKey>,
) -> Option<DrawOverlayLayer> {
    if !layer.visible || layer.reloading {
        layer.clear_wanted();
        return None;
    }

    let desc = layer.source.description();
    let (img_w, img_h) = (desc.width, desc.height);
    let max_level = layer.meta.max_level();

    let weight = layer.appetite_weight_bytes();
    let appetite_bytes = gpu_cache.proportional_pyramid_byte_cap(weight, total_appetite_weight);
    let cap_tiles = (appetite_bytes / layer.observed_tile_bytes.max(1)).max(1);

    let sharpest = layer
        .frame
        .sharpest_level(img_w, img_h, viewport_mpp, max_level);
    let (target_level, target_cells) = select_target_rung(layer, sharpest, cap_tiles, frustum);

    let ladder = layer.meta.ladder(target_level);
    let mut ladder_cells: Vec<(u8, Vec<(u32, u32)>)> = Vec::with_capacity(ladder.len());
    for &rung in &ladder {
        let cells = if rung == target_level {
            target_cells.clone()
        } else {
            layer.cull.visible_cells(&layer.frame, rung, frustum)
        };
        ladder_cells.push((rung, cells));
    }

    // Populate `layer.wanted` with apex root tiles AND visible cells across ALL
    // ladder rungs so `PendingUploads` never prunes intermediate coarse rungs
    // during progressive refinement.
    layer.wanted.clear();
    let top_w = grid_dim(level_size(img_w, u32::from(max_level)));
    let top_h = grid_dim(level_size(img_h, u32::from(max_level)));
    for ry in 0..top_h {
        for rx in 0..top_w {
            layer.wanted.insert(layer.meta.tile_key(max_level, rx, ry));
        }
    }
    for (rung, cells) in &ladder_cells {
        for &(x, y) in cells {
            layer.wanted.insert(layer.meta.tile_key(*rung, x, y));
        }
    }
    layer.sync_wanted_shared();

    // Reconcile off-screen or already-cached pointwise jobs (cancelling tokens
    // strictly outside the scheduler lock).
    let stale_tokens = {
        let mut sched = lock_unpoisoned(&runtime.scheduler);
        sched.reconcile_pointwise(layer.source_id(), &layer.wanted, |k| {
            gpu_cache.contains(k) || pending.contains(k)
        })
    };
    for tok in stale_tokens {
        tok.cancel();
    }

    // Assemble draw patches using uniform per-level `segments` (geometry is
    // built lazily inside `OverlayRenderer` only on GPU cache miss).
    let segments = level_segments(&layer.frame, img_w, img_h, target_level);
    let mut patches = Vec::with_capacity(target_cells.len());
    for &(x, y) in &target_cells {
        let want_key = layer.meta.tile_key(target_level, x, y);
        let Some((carrier_key, gpu_tile)) = gpu_cache.find_carrier(&want_key) else {
            continue;
        };
        carrier_keys.insert(carrier_key);

        let cell_rect = cell_image_rect(x, y, u32::from(target_level), img_w, img_h);
        let uv_corners = cell_uv(
            cell_rect,
            &carrier_key,
            gpu_tile.width,
            gpu_tile.height,
            gpu_tile.valid_w,
            gpu_tile.valid_h,
            gpu_tile.gutter,
        );
        let uv_rect = [
            uv_corners[0],
            uv_corners[1],
            uv_corners[2] - uv_corners[0],
            uv_corners[3] - uv_corners[1],
        ];
        patches.push(DrawOverlayPatch {
            mesh_key: OverlayMeshKey {
                source: layer.source_id(),
                relaid: layer.relaid,
                level: target_level,
                x,
                y,
            },
            cell_rect,
            uv_rect,
            texture: gpu_tile,
        });
    }

    if !target_cells.is_empty() {
        schedule_layer_rungs(layer, &ladder_cells, runtime, gpu_cache, pending);
    }

    (!patches.is_empty()).then(|| DrawOverlayLayer {
        opacity: layer.opacity,
        frame: Arc::clone(&layer.frame),
        image_width: img_w,
        image_height: img_h,
        segments,
        patches,
    })
}

fn schedule_layer_rungs(
    layer: &OverlayLayer,
    ladder_cells: &[(u8, Vec<(u32, u32)>)],
    runtime: &Arc<OverlayRuntime>,
    gpu_cache: &GpuTileCache,
    pending: &PendingUploads,
) {
    let is_cached = |key: &TileKey| gpu_cache.contains(key) || pending.contains(key);

    // Walk the ladder coarsest-first so coarse root coverage lands before fine
    // tiles queue up, while never letting a terminally failed tile on a coarse
    // rung block finer rungs.
    for (rung, rung_cells) in ladder_cells {
        let rung = *rung;
        let Some(plan) = layer.meta.level(rung) else {
            continue;
        };
        if !plan.fits || rung_cells.is_empty() {
            continue;
        }
        let candidates: Vec<TileKey> = rung_cells
            .iter()
            .map(|&(x, y)| layer.meta.tile_key(rung, x, y))
            .collect();

        match plan.serve {
            ServeMode::Pointwise => {
                let (admitted, rung_unfinished) = {
                    let mut sched = lock_unpoisoned(&runtime.scheduler);
                    let admitted = sched.admit_pointwise(
                        &layer.meta,
                        rung,
                        &candidates,
                        is_cached,
                        &runtime.shutdown,
                    );
                    let unfinished = candidates
                        .iter()
                        .any(|k| !gpu_cache.contains(k) && !sched.is_terminal_failed(k));
                    (admitted, unfinished)
                };
                for tile in admitted {
                    if !submit_pointwise_job(tile, &layer.source, &layer.fingerprint_hex, runtime) {
                        break;
                    }
                }
                if rung_unfinished {
                    break;
                }
            }
            ServeMode::Pass { from } => {
                let pass_key = layer.meta.pass_key();
                let (cancelled_old, pass_info, pass_active) = {
                    let mut sched = lock_unpoisoned(&runtime.scheduler);
                    let cancelled_old = sched.update_target(pass_key, rung, Some(from));
                    let missing =
                        sched.check_pass_candidates(&layer.meta, rung, &candidates, is_cached);
                    if missing.is_empty() {
                        (cancelled_old, None, sched.has_active_pass(pass_key))
                    } else {
                        let pass_cancel = runtime.shutdown.child_token();
                        match sched.begin_pass(
                            pass_key,
                            rung,
                            Some(from),
                            pass_cancel.clone(),
                            missing.clone(),
                        ) {
                            Ok(pass_id) => {
                                (cancelled_old, Some((pass_id, pass_cancel, missing)), true)
                            }
                            Err(_) => (cancelled_old, None, true),
                        }
                    }
                };
                if let Some(tok) = cancelled_old {
                    tok.cancel();
                }
                if let Some((pass_id, pass_cancel, missing)) = pass_info {
                    submit_pass_job(
                        pass_key,
                        pass_id,
                        pass_cancel,
                        missing,
                        Arc::clone(&layer.wanted_shared),
                        &layer.source,
                        &layer.meta,
                        &layer.fingerprint_hex,
                        runtime,
                    );
                }
                if pass_active {
                    break;
                }
            }
        }
    }
}

/// Submit a single pointwise overlay decode job to `fast_pool`.
/// Returns `false` if the lane is full or closed so the caller stops submitting
/// this rung and backs off cleanly without spinning.
fn submit_pointwise_job(
    admitted: AdmittedTile,
    source: &Arc<dyn RasterSource>,
    fingerprint_hex: &str,
    runtime: &Arc<OverlayRuntime>,
) -> bool {
    let tile_key = admitted.key;
    let job_id = admitted.job_id;
    let cancel = admitted.cancel;
    let source = Arc::clone(source);
    let fp = fingerprint_hex.to_owned();
    let rt = Arc::clone(runtime);
    let guard = PointwiseFlightGuard::new(Arc::clone(runtime), tile_key, job_id);

    let job = DecodeJob {
        key: DecodeJobKey::Tile(tile_key),
        queued_bytes: default_display_tile_bytes(),
        cancel: cancel.clone(),
        work: Box::new(move || {
            if cancel.is_cancelled() {
                guard.mark_cancelled();
                return;
            }
            let disk_key = DiskTileKey::from_tile_key(fp, &tile_key);
            if let Some(ref disk) = rt.disk_cache
                && let Some(tile) = load_qoi_display_tile(disk, &disk_key)
            {
                if matches!(
                    rt.done_tx.send_if_uncancelled((tile_key, tile), &cancel),
                    Ok(true)
                ) {
                    guard.mark_arrived();
                } else {
                    guard.mark_cancelled();
                }
                return;
            }

            let ctx = DecodeContext {
                chunk_cache: &rt.chunk_cache,
                budget_tracker: &rt.budget_tracker,
                budget: &rt.budget,
                cancel: &cancel,
            };
            match source.decode_pointwise_tile(tile_key.level, tile_key.x, tile_key.y, &ctx) {
                Ok(mut tile) => {
                    if cancel.is_cancelled() {
                        guard.mark_cancelled();
                        return;
                    }
                    if let Some(ref disk) = rt.disk_cache {
                        store_qoi_display_tile(
                            disk,
                            &disk_key,
                            tile.width(),
                            tile.height(),
                            tile.data(),
                            &cancel,
                        );
                    }
                    generate_linear_mipmaps(&mut tile);
                    if matches!(
                        rt.done_tx.send_if_uncancelled((tile_key, tile), &cancel),
                        Ok(true)
                    ) {
                        guard.mark_arrived();
                    } else {
                        guard.mark_cancelled();
                    }
                }
                Err(err) if err.is_cancelled() => {
                    guard.mark_cancelled();
                }
                Err(err) => {
                    tracing::warn!(
                        source = ?tile_key.source,
                        level = tile_key.level,
                        x = tile_key.x,
                        y = tile_key.y,
                        error = %err,
                        "Pointwise overlay tile decode failed"
                    );
                    let terminal = matches!(
                        err,
                        RasterError::UnsupportedFormat
                            | RasterError::InvalidDimensions { .. }
                            | RasterError::LimitExceeded { .. }
                            | RasterError::TileOutOfRange { .. }
                            | RasterError::Binding(_)
                    );
                    guard.mark_failed(terminal);
                }
            }
        }),
    };

    match runtime.fast_pool.try_submit(job) {
        Ok(()) => true,
        Err(err) => {
            // Unregister BEFORE dropping the rejected job so `PointwiseFlightGuard::drop`
            // does not trigger an immediate 100% CPU repaint loop.
            let _ = lock_unpoisoned(&runtime.scheduler).cancel_in_flight(&tile_key, job_id);
            let is_full = matches!(err, SubmitError::Full(_));
            drop(err);
            if is_full {
                runtime.note_submit_backoff();
            }
            false
        }
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "sequential pass submission binds pass identity, wanted set, and runtime"
)]
fn submit_pass_job(
    pass_key: PassKey,
    pass_id: u64,
    pass_cancel: CancelToken,
    missing: Vec<TileKey>,
    wanted_shared: Arc<Mutex<HashSet<TileKey>>>,
    source: &Arc<dyn RasterSource>,
    meta: &PyramidMeta,
    fingerprint_hex: &str,
    runtime: &Arc<OverlayRuntime>,
) {
    let source = Arc::clone(source);
    let meta = meta.clone();
    let fp = fingerprint_hex.to_owned();
    let rt = Arc::clone(runtime);
    let queued_bytes = source
        .pass_peak_bytes()
        .min(runtime.budget.bulk_lane_pending_bytes as u64) as usize;
    let guard = PassFlightGuard::new(Arc::clone(runtime), pass_key, pass_id);

    let job = DecodeJob {
        key: DecodeJobKey::Pass(pass_key),
        queued_bytes: queued_bytes.max(default_display_tile_bytes()),
        cancel: pass_cancel.clone(),
        work: Box::new(move || {
            // Stream any QOI disk-cache hits immediately (never buffering an
            // entire rung in RAM or discarding partial hits on 1 miss).
            let mut already_hit: HashSet<TileKey> = HashSet::new();
            let mut any_miss = false;
            if let Some(ref disk) = rt.disk_cache {
                for &tk in &missing {
                    if pass_cancel.is_cancelled() {
                        guard.abort();
                        return;
                    }
                    let dk = DiskTileKey::from_tile_key(fp.clone(), &tk);
                    if let Some(tile) = load_qoi_display_tile(disk, &dk) {
                        if matches!(
                            rt.done_tx.send_if_uncancelled((tk, tile), &pass_cancel),
                            Ok(true)
                        ) {
                            lock_unpoisoned(&rt.scheduler)
                                .on_pass_tile_arrived(pass_key, pass_id, &tk);
                            already_hit.insert(tk);
                            (rt.wakeup)();
                        }
                    } else {
                        any_miss = true;
                    }
                }
                if !any_miss {
                    guard.finish(true);
                    return;
                }
            }

            let initial_missing: HashSet<TileKey> = missing
                .into_iter()
                .filter(|tk| !already_hit.contains(tk))
                .collect();
            let max_level = meta.max_level();
            let mut qoi_batch: Vec<(DiskTileKey, Vec<u8>)> =
                Vec::with_capacity(PASS_QOI_BATCH_MAX_TILES);
            let mut qoi_batch_bytes = 0usize;

            let ctx = DecodeContext {
                chunk_cache: &rt.chunk_cache,
                budget_tracker: &rt.budget_tracker,
                budget: &rt.budget,
                cancel: &pass_cancel,
            };

            let result = source.produce_pass(&ctx, &mut |level, tx, ty, tw, th, rgba| {
                if pass_cancel.is_cancelled() {
                    return Err(RasterError::Cancelled);
                }
                let tk = meta.tile_key(level, tx, ty);
                if already_hit.contains(&tk) {
                    return Ok(());
                }
                if let Some(ref disk) = rt.disk_cache
                    && disk.max_bytes() > 0
                    && let Ok(encoded) = encode_qoi_rgba8(tw, th, &rgba, MAX_QOI_DECODED_BYTES)
                {
                    qoi_batch_bytes = qoi_batch_bytes.saturating_add(encoded.len());
                    qoi_batch.push((DiskTileKey::from_tile_key(fp.clone(), &tk), encoded));
                    if qoi_batch.len() >= PASS_QOI_BATCH_MAX_TILES
                        || qoi_batch_bytes >= PASS_QOI_BATCH_MAX_BYTES
                    {
                        flush_qoi_batch(disk, &mut qoi_batch, &mut qoi_batch_bytes, &pass_cancel);
                    }
                }

                // Read live `wanted_shared` briefly (never nesting locks) so
                // mid-pass zooms upload tiles matching the current viewport.
                let should_upload = level == max_level
                    || initial_missing.contains(&tk)
                    || lock_unpoisoned(&wanted_shared).contains(&tk);
                if should_upload {
                    let mut tile = DecodedTile::try_new(tw, th, PixelFormat::Rgba8UnormSrgb, rgba)
                        .map_err(|e| RasterError::Decode(e.to_string()))?;
                    generate_linear_mipmaps(&mut tile);
                    if matches!(
                        rt.done_tx.send_if_uncancelled((tk, tile), &pass_cancel),
                        Ok(true)
                    ) {
                        lock_unpoisoned(&rt.scheduler).on_pass_tile_arrived(pass_key, pass_id, &tk);
                        (rt.wakeup)();
                    }
                }
                Ok(())
            });

            if let Some(ref disk) = rt.disk_cache {
                if result.is_ok() {
                    flush_qoi_batch(disk, &mut qoi_batch, &mut qoi_batch_bytes, &pass_cancel);
                } else {
                    qoi_batch.clear();
                }
            }

            match result {
                Ok(()) => guard.finish(true),
                Err(err) if err.is_cancelled() => guard.abort(),
                Err(err) => {
                    tracing::warn!(
                        source = ?meta.source,
                        error = %err,
                        "Sequential overlay pass failed"
                    );
                    guard.finish(false);
                }
            }
        }),
    };

    if let Err(err) = runtime.bulk_pool.try_submit(job) {
        let token = lock_unpoisoned(&runtime.scheduler).abort_pass(pass_key, pass_id);
        let is_full = matches!(err, SubmitError::Full(_));
        drop(err);
        if let Some(tok) = token {
            tok.cancel();
        }
        if is_full {
            runtime.note_submit_backoff();
        }
    }
}
