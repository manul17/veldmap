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
    bleed_alpha, resample_f32_window, resample_rgba8_window, resample_uint_nearest_window,
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

fn alloc_empty_surface(pixels: usize, format: PixelFormat) -> Vec<u8> {
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
    for y in iy0..iy1 {
        let src_off =
            (((y - cy0) as usize) * (chunk.width as usize) + ((ix0 - cx0) as usize)) * bpp;
        let dst_off = (((y - dy0) as usize) * (dw as usize) + ((ix0 - dx0) as usize)) * bpp;
        dst[dst_off..dst_off + run].copy_from_slice(&chunk.data[src_off..src_off + run]);
    }
}

/// Pointwise reading: decode only the chunks intersecting `req.wants` at
/// `req.level`, reserving `region + tile` working memory against
/// `req.budget_tracker` without nesting leases across chunk singleflight.
pub(crate) fn direct<S: Chunked>(
    source: &mut S,
    req: &DirectRead<'_>,
    emit: TileEmit<'_>,
) -> Result<(), RasterError> {
    if req.base.0 == 0 || req.base.1 == 0 {
        return Err(RasterError::InvalidDimensions {
            width: req.base.0,
            height: req.base.1,
        });
    }
    let format = req.format;
    let bpp = format.bytes_per_pixel();
    let bpp_u64 = bpp as u64;
    let lw = level_size(req.base.0, u32::from(req.level));
    let lh = level_size(req.base.1, u32::from(req.level));
    let (image, sw, sh) = req.grid.source_for(req.base, lw, lh);
    let (cw, ch) = req.grid.chunk_of(image);
    if cw == 0 || ch == 0 {
        return Err(RasterError::InvalidDimensions {
            width: cw,
            height: ch,
        });
    }
    let across = sw.div_ceil(cw);
    let region_cap_pixels = req.budget.region_cap_pixels_for(format);

    for &(tx, ty) in req.wants {
        if req.cancel.is_cancelled() {
            return Err(RasterError::Cancelled);
        }
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
        } = region_of((tx, ty), (tw, th), req.base, (sw, sh), req.level);
        let (rw, rh) = ((sx1 - sx0) as u32, (sy1 - sy0) as u32);
        if rw == 0 || rh == 0 {
            return Err(RasterError::Decode(format!(
                "tile ({tx}, {ty}) at level {} produced empty source region",
                req.level
            )));
        }
        if u64::from(rw) * u64::from(rh) > region_cap_pixels {
            return Err(RasterError::LimitExceeded {
                width: rw,
                height: rh,
                estimated_bytes: u64::from(rw) * u64::from(rh) * bpp_u64,
                limit_bytes: req.budget.region_cap_bytes,
            });
        }
        let (sx0, sy0) = (sx0 as u32, sy0 as u32);

        // Fetch intersecting chunks BEFORE acquiring the region+tile working
        // lease so chunk singleflight and per-chunk reservations never nest
        // leases or risk AB-BA deadlock under tight decode budgets.
        let mut chunks = Vec::new();
        for cy in sy0 / ch..=(sy0 + rh - 1) / ch {
            for cx in sx0 / cw..=(sx0 + rw - 1) / cw {
                if req.cancel.is_cancelled() {
                    return Err(RasterError::Cancelled);
                }
                let chunk = source.chunk(image, cy * across + cx, req.cancel)?;
                chunks.push((cx * cw, cy * ch, chunk));
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

/// Sequential top-to-bottom pass over the base IFD feeding `Cascade`.
///
/// Callers must admit/reserve `grid.pass_peak(base, format)` before invoking `pass`.
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

    let mut cascade = Cascade::new_for_format(0, width, height, format);
    if format == PixelFormat::Rgba8UnormSrgb {
        let mut bled_emit = bleed_display_emit(emit);
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
            cascade.push_rows(&band, rows, &mut bled_emit)?;
        }
        cascade.finish(&mut bled_emit)
    } else {
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
            cascade.push_rows(&band, rows, emit)?;
        }
        cascade.finish(emit)
    }
}
