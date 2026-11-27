//! Runtime execution drivers (`Chunked`, `DirectRead`, `direct`, `pass`)
//! bridging chunked raster backends to [`Cascade`] and fractional window
//! resampling across `Display`, `Scientific`, and `Categorical` tracks while
//! honoring [`DecodeBudgetTracker`] and [`CancelToken`].

use std::sync::Arc;

use veldmap_core::budget::{DecodeBudgetTracker, MemoryBudget};
use veldmap_core::sync::CancelToken;
use veldmap_core::tile::{PixelFormat, level_size, tile_extent};

use crate::RasterError;

use super::cascade::{Cascade, TileEmit, bleed_display_emit};
use super::chunk_cache::DecodedChunk;
use super::grid::{Grid, Region, region_of};
use super::resample::{
    Window, bleed_alpha, resample_f32_window, resample_rgba8_window, resample_uint_nearest_window,
};

/// Chunked raster backend driving [`direct`] and [`pass`].
pub(crate) trait Chunked {
    /// Decode chunk `index` of IFD/image `image` in the stream's pixel format.
    fn chunk(
        &mut self,
        image: usize,
        index: u32,
        cancel: &CancelToken,
    ) -> Result<Arc<DecodedChunk>, RasterError>;
}

/// Parameters governing a [`direct`] pointwise tile read.
pub(crate) struct DirectRead<'a> {
    pub(crate) grid: &'a Grid,
    pub(crate) base: (u32, u32),
    pub(crate) format: PixelFormat,
    pub(crate) level: u8,
    pub(crate) wants: &'a [(u32, u32)],
    pub(crate) budget: &'a MemoryBudget,
    pub(crate) budget_tracker: &'a DecodeBudgetTracker,
    pub(crate) cancel: &'a CancelToken,
}

pub(crate) fn alloc_empty_surface(pixels: usize, format: PixelFormat) -> Vec<u8> {
    let bpp = format.bytes_per_pixel();
    let byte_len = pixels.saturating_mul(bpp);
    match format {
        PixelFormat::R32Float => {
            let nan = f32::NAN.to_ne_bytes();
            let mut buf = Vec::with_capacity(byte_len);
            for _ in 0..pixels {
                buf.extend_from_slice(&nan);
            }
            buf
        }
        PixelFormat::Rgba8UnormSrgb | PixelFormat::R8Uint | PixelFormat::R16Uint => {
            vec![0u8; byte_len]
        }
    }
}

fn blit_chunk(
    dst: &mut [u8],
    dst_origin: (u32, u32),
    dst_size: (u32, u32),
    chunk_origin: (u32, u32),
    chunk: &DecodedChunk,
    bpp: usize,
) {
    let (dx0, dy0) = dst_origin;
    let (dw, dh) = dst_size;
    let (cx0, cy0) = chunk_origin;
    let ix0 = dx0.max(cx0);
    let iy0 = dy0.max(cy0);
    let ix1 = dx0.saturating_add(dw).min(cx0.saturating_add(chunk.width));
    let iy1 = dy0.saturating_add(dh).min(cy0.saturating_add(chunk.height));
    if ix0 >= ix1 || iy0 >= iy1 {
        return;
    }
    let run = ((ix1 - ix0) as usize) * bpp;
    let chunk_stride = (chunk.width as usize).saturating_mul(bpp);
    let dst_stride = (dw as usize).saturating_mul(bpp);
    for y in iy0..iy1 {
        let src_off = ((y - cy0) as usize) * chunk_stride + ((ix0 - cx0) as usize) * bpp;
        let dst_off = ((y - dy0) as usize) * dst_stride + ((ix0 - dx0) as usize) * bpp;
        if let (Some(src_slice), Some(dst_slice)) = (
            chunk.data.get(src_off..src_off + run),
            dst.get_mut(dst_off..dst_off + run),
        ) {
            dst_slice.copy_from_slice(src_slice);
        }
    }
}

/// Pure per-tile chunk geometry plan extracted from [`direct`] so remote COG
/// prefetch planners and pointwise execution share identical math.
struct TileChunkPlan {
    tx: u32,
    ty: u32,
    tw: u32,
    th: u32,
    sx0: u32,
    sy0: u32,
    rw: u32,
    rh: u32,
    window: Window,
    cx_start: u32,
    cx_end: u32,
    cy_start: u32,
    cy_end: u32,
}

/// Pure pointwise read plan for one level & `wants` tile batch.
pub(crate) struct DirectPlan {
    image: usize,
    cw: u32,
    ch: u32,
    across: u32,
    tiles: Vec<TileChunkPlan>,
}

impl DirectPlan {
    /// Target IFD/image index chosen for this pointwise plan.
    pub(crate) const fn image(&self) -> usize {
        self.image
    }

    /// Sorted, deduplicated list of chunk indices touched by this plan.
    pub(crate) fn unique_chunk_indices(&self) -> Vec<u32> {
        let mut indices = Vec::new();
        for tile in &self.tiles {
            for cy in tile.cy_start..=tile.cy_end {
                for cx in tile.cx_start..=tile.cx_end {
                    if let Some(idx) = cy.checked_mul(self.across).and_then(|r| r.checked_add(cx)) {
                        indices.push(idx);
                    }
                }
            }
        }
        indices.sort_unstable();
        indices.dedup();
        indices
    }
}

/// Compute the pure chunk-index plan for a pointwise read without issuing I/O.
pub(crate) fn plan_direct(
    grid: &Grid,
    base: (u32, u32),
    format: PixelFormat,
    level: u8,
    wants: &[(u32, u32)],
    budget: &MemoryBudget,
) -> Result<DirectPlan, RasterError> {
    if base.0 == 0 || base.1 == 0 {
        return Err(RasterError::InvalidDimensions {
            width: base.0,
            height: base.1,
        });
    }
    let bpp_u64 = format.bytes_per_pixel() as u64;
    let lw = level_size(base.0, u32::from(level));
    let lh = level_size(base.1, u32::from(level));
    let (image, sw, sh) = grid.source_for(base, lw, lh);
    let (cw, ch) = grid.chunk_of(image);
    if cw == 0 || ch == 0 {
        return Err(RasterError::InvalidDimensions {
            width: cw,
            height: ch,
        });
    }
    let across = sw.div_ceil(cw);
    let region_cap_pixels = budget.region_cap_pixels_for(format);
    let mut tiles = Vec::with_capacity(wants.len());

    for &(tx, ty) in wants {
        let tw = tile_extent(tx, lw);
        let th = tile_extent(ty, lh);
        if tw == 0 || th == 0 {
            continue;
        }
        let Region {
            sx0,
            sx1,
            sy0,
            sy1,
            window,
        } = region_of((tx, ty), (tw, th), base, (sw, sh), level);
        let (rw, rh) = ((sx1 - sx0) as u32, (sy1 - sy0) as u32);
        if rw == 0 || rh == 0 {
            return Err(RasterError::Decode(format!(
                "tile ({tx}, {ty}) at level {level} produced empty source region"
            )));
        }
        if u64::from(rw) * u64::from(rh) > region_cap_pixels {
            return Err(RasterError::LimitExceeded {
                width: rw,
                height: rh,
                estimated_bytes: u64::from(rw) * u64::from(rh) * bpp_u64,
                limit_bytes: budget.region_cap_bytes,
            });
        }
        let (sx0, sy0) = (sx0 as u32, sy0 as u32);
        tiles.push(TileChunkPlan {
            tx,
            ty,
            tw,
            th,
            sx0,
            sy0,
            rw,
            rh,
            window,
            cx_start: sx0 / cw,
            cx_end: (sx0 + rw - 1) / cw,
            cy_start: sy0 / ch,
            cy_end: (sy0 + rh - 1) / ch,
        });
    }

    Ok(DirectPlan {
        image,
        cw,
        ch,
        across,
        tiles,
    })
}

/// Pointwise reading: decode only the chunks intersecting `req.wants` at
/// `req.level`, reserving `region + tile` working memory against
/// `req.budget_tracker` without nesting leases across chunk singleflight.
pub(crate) fn direct<S: Chunked>(
    source: &mut S,
    req: &DirectRead<'_>,
    emit: TileEmit<'_>,
) -> Result<(), RasterError> {
    let format = req.format;
    let bpp = format.bytes_per_pixel();
    let bpp_u64 = bpp as u64;
    let plan = plan_direct(req.grid, req.base, format, req.level, req.wants, req.budget)?;

    for tile_plan in plan.tiles {
        if req.cancel.is_cancelled() {
            return Err(RasterError::Cancelled);
        }
        let TileChunkPlan {
            tx,
            ty,
            tw,
            th,
            sx0,
            sy0,
            rw,
            rh,
            window,
            cx_start,
            cx_end,
            cy_start,
            cy_end,
        } = tile_plan;

        // Fetch intersecting chunks BEFORE acquiring the region+tile working
        // lease so chunk singleflight and per-chunk reservations never nest
        // leases or risk AB-BA deadlock under tight decode budgets.
        let mut chunks = Vec::new();
        for cy in cy_start..=cy_end {
            for cx in cx_start..=cx_end {
                if req.cancel.is_cancelled() {
                    return Err(RasterError::Cancelled);
                }
                let chunk = source.chunk(plan.image, cy * plan.across + cx, req.cancel)?;
                chunks.push((cx * plan.cw, cy * plan.ch, chunk));
            }
        }

        let region_bytes = u64::from(rw)
            .saturating_mul(u64::from(rh))
            .saturating_mul(bpp_u64);
        let tile_bytes = u64::from(tw)
            .saturating_mul(u64::from(th))
            .saturating_mul(bpp_u64);
        let _work_lease = req
            .budget_tracker
            .reserve_blocking(region_bytes.saturating_add(tile_bytes), req.cancel)?;

        let mut region = alloc_empty_surface((rw as usize) * (rh as usize), format);
        for (chunk_x, chunk_y, chunk) in chunks {
            debug_assert_eq!(chunk.format, format);
            blit_chunk(
                &mut region,
                (sx0, sy0),
                (rw, rh),
                (chunk_x, chunk_y),
                &chunk,
                bpp,
            );
        }

        let mut tile = match format {
            PixelFormat::Rgba8UnormSrgb => resample_rgba8_window(&region, rw, rh, window, tw, th),
            PixelFormat::R32Float => resample_f32_window(&region, rw, rh, window, tw, th),
            PixelFormat::R8Uint | PixelFormat::R16Uint => {
                resample_uint_nearest_window(&region, rw, rh, window, tw, th, bpp)
            }
        };
        drop(region);
        // Alpha bleeding is strictly for RGBA8 display tiles; scientific and
        // categorical tracks preserve exact nodata / class boundaries.
        if format == PixelFormat::Rgba8UnormSrgb {
            bleed_alpha(&mut tile, tw, th);
        }
        emit(req.level, tx, ty, tw, th, tile)?;
    }

    Ok(())
}

/// Sequential top-to-bottom pass over the base IFD feeding [`Cascade`].
///
/// Callers must admit/reserve `grid.admit_pass_peak(base, format, limit)` before invoking `pass`.
pub(crate) fn pass<S: Chunked>(
    source: &mut S,
    grid: &Grid,
    base: (u32, u32),
    format: PixelFormat,
    cancel: &CancelToken,
    emit: TileEmit<'_>,
) -> Result<(), RasterError> {
    let (width, height) = base;
    let (cw, ch) = grid.chunk;
    if width == 0 || height == 0 || cw == 0 || ch == 0 {
        return Err(RasterError::InvalidDimensions { width, height });
    }
    let bpp = format.bytes_per_pixel();
    let across = width.div_ceil(cw);
    let down = height.div_ceil(ch);

    let mut cascade = Cascade::new(0, width, height, format);
    let mut bled_holder;
    let sink: TileEmit<'_> = if format == PixelFormat::Rgba8UnormSrgb {
        bled_holder = bleed_display_emit(emit);
        &mut bled_holder
    } else {
        emit
    };
    for cy in 0..down {
        if cancel.is_cancelled() {
            return Err(RasterError::Cancelled);
        }
        let rows = ch.min(height - cy * ch);
        let mut band = alloc_empty_surface((width as usize) * (rows as usize), format);
        for cx in 0..across {
            if cancel.is_cancelled() {
                return Err(RasterError::Cancelled);
            }
            let chunk = source.chunk(0, cy * across + cx, cancel)?;
            blit_chunk(&mut band, (0, 0), (width, rows), (cx * cw, 0), &chunk, bpp);
        }
        cascade.push_rows(&band, rows, sink)?;
    }
    cascade.finish(sink)
}
