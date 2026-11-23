//! Chunk grid driver (`Grid`, `Overview`, `Chunked`, `direct`, `pass`):
//! drives both pointwise random chunk access and single-pass sequential
//! cascade emission while honoring `MemoryBudget` and `CancelToken`.

use std::sync::Arc;

use veldmap_core::budget::{DecodePeak, MemoryBudget};
use veldmap_core::io::CancelToken;

use crate::RasterError;
use crate::cascade::{self, Cascade, TileEmit};
use crate::chunk_cache::DecodedChunk;
use crate::pyramid::{self, RASTER_TILE_SIZE};
use crate::resample::{Window, bleed_alpha, resample_rgba8_window};

/// Minimum chunk pixel area eligible for pointwise random access.
pub const MIN_CHUNK_PIXELS: u64 = 32 * 32;

/// Cost of reading one tile at a pointwise level.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Footprint {
    pub image: usize,
    pub region_pixels: u64,
    pub held_pixels: u64,
}

/// Reduced overview IFD inside a multi-resolution source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Overview {
    pub image: usize,
    pub width: u32,
    pub height: u32,
    pub chunk: (u32, u32),
}

/// Chunk layout of a raster source (base image + overviews).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Grid {
    pub tiled: bool,
    pub chunk: (u32, u32),
    pub overviews: Vec<Overview>,
    /// Bytes per pixel in raw encoded samples (`channels * sample_bytes`).
    pub depth: u32,
}

impl Grid {
    /// Return whether this layout supports cheap random access (`tiled && !overviews.is_empty()`).
    pub fn random_access(&self) -> bool {
        self.tiled && !self.overviews.is_empty()
    }

    /// Return the chunk grid `(cw, ch)` for `image` (`0` = base IFD).
    pub fn chunk_of(&self, image: usize) -> (u32, u32) {
        self.overviews
            .iter()
            .find(|o| o.image == image)
            .map_or(self.chunk, |o| o.chunk)
    }

    /// Choose the smallest overview covering `(level_width, level_height)`
    /// within the 1-pixel rounding tolerance between `ceil(side / 2^k)` and
    /// GDAL's `floor(side / 2)`.
    pub fn source_for(
        &self,
        base: (u32, u32),
        level_width: u32,
        level_height: u32,
    ) -> (usize, u32, u32) {
        let fits = |copy: u32, level: u32| {
            copy >= level.saturating_sub(1) && u64::from(copy) * 2 > u64::from(level)
        };

        let mut best = (0usize, base.0, base.1);
        for overview in &self.overviews {
            if fits(overview.width, level_width)
                && fits(overview.height, level_height)
                && overview.width < best.1
            {
                best = (overview.image, overview.width, overview.height);
            }
        }
        best
    }

    /// Return whether `level` qualifies for pointwise random access under `budget`.
    pub fn pointwise(&self, base: (u32, u32), level: u8, budget: &MemoryBudget) -> bool {
        self.footprint(base, level, budget).is_some()
    }

    /// Compute the pointwise reading footprint for `level`, or `None` when the
    /// level must be served via sequential pass.
    pub fn footprint(
        &self,
        base: (u32, u32),
        level: u8,
        budget: &MemoryBudget,
    ) -> Option<Footprint> {
        let lw = pyramid::level_size(base.0, u32::from(level)).max(1);
        let lh = pyramid::level_size(base.1, u32::from(level)).max(1);
        let (image, sw, sh) = self.source_for(base, lw, lh);
        let (cw, ch) = self.chunk_of(image);
        if u64::from(cw) * u64::from(ch) < MIN_CHUNK_PIXELS {
            return None;
        }
        let aligned = lw.abs_diff(sw) <= 1 && lh.abs_diff(sh) <= 1;

        let side = |level_side: u32, source_side: u32| -> u64 {
            let ideal = (u64::from(RASTER_TILE_SIZE) * u64::from(source_side))
                .div_ceil(u64::from(level_side));
            if aligned { ideal } else { ideal + 1 }
        };
        let rw = side(lw, sw).min(u64::from(sw));
        let rh = side(lh, sh).min(u64::from(sh));
        let region_pixels = rw.saturating_mul(rh);
        if region_pixels > budget.region_cap_pixels {
            return None;
        }

        let held_pixels = spanned(rw, cw, sw, aligned).saturating_mul(spanned(rh, ch, sh, aligned));
        if held_pixels.saturating_mul(4) > budget.decoded_chunk_cache_bytes as u64 {
            return None;
        }
        Some(Footprint {
            image,
            region_pixels,
            held_pixels,
        })
    }

    /// Peak memory of decoding one fresh chunk `(cw, ch)` before expansion.
    pub fn chunk_bytes(&self, chunk: (u32, u32)) -> u64 {
        u64::from(chunk.0) * u64::from(chunk.1) * (u64::from(self.depth) + 4)
    }

    /// Estimated working memory peak for pointwise reading at `level`.
    pub fn direct_peak(
        &self,
        base: (u32, u32),
        level: u8,
        budget: &MemoryBudget,
    ) -> Option<DecodePeak> {
        let fp = self.footprint(base, level, budget)?;
        let tile = u64::from(RASTER_TILE_SIZE) * u64::from(RASTER_TILE_SIZE) * 4;
        Some(
            DecodePeak::new()
                .with("chunk cache", budget.decoded_chunk_cache_bytes as u64)
                .with("region", fp.region_pixels * 4)
                .with("fresh chunk", self.chunk_bytes(self.chunk_of(fp.image)))
                .with("tile", tile * 2),
        )
    }

    /// Estimated working memory peak for a full sequential pass over `base`.
    pub fn pass_peak(&self, base: (u32, u32)) -> DecodePeak {
        let (width, height) = base;
        DecodePeak::new()
            .with("band", u64::from(width) * u64::from(self.chunk.1) * 4)
            .with("fresh chunk", self.chunk_bytes(self.chunk))
            .with("cascade", cascade::bytes(width, height))
    }
}

pub(crate) fn spanned(side: u64, chunk: u32, source: u32, aligned: bool) -> u64 {
    if side == 0 {
        return 0;
    }
    let chunk = u64::from(chunk.max(1));
    let tile = u64::from(RASTER_TILE_SIZE);
    let flush = aligned && (chunk % tile == 0 || tile % chunk == 0);
    let offset = if flush {
        chunk - tile.min(chunk)
    } else {
        chunk - 1
    };
    let touched = ((offset + side - 1) / chunk + 1).min(u64::from(source).div_ceil(chunk));
    touched.saturating_mul(chunk)
}

pub(crate) struct Region {
    pub sx0: u64,
    pub sx1: u64,
    pub sy0: u64,
    pub sy1: u64,
    pub window: Window,
}

pub(crate) fn region_of(
    (tx, ty): (u32, u32),
    (tw, th): (u32, u32),
    (lw, lh): (u32, u32),
    (sw, sh): (u32, u32),
    aligned: bool,
) -> Region {
    let (x0, y0) = (
        u64::from(tx) * u64::from(RASTER_TILE_SIZE),
        u64::from(ty) * u64::from(RASTER_TILE_SIZE),
    );
    if aligned {
        let x0 = x0.min(u64::from(sw).saturating_sub(1));
        let y0 = y0.min(u64::from(sh).saturating_sub(1));
        let x1 = (x0 + u64::from(tw)).min(u64::from(sw));
        let y1 = (y0 + u64::from(th)).min(u64::from(sh));
        let window = Window {
            x0: 0.0,
            y0: 0.0,
            x1: (x1 - x0) as f64,
            y1: (y1 - y0) as f64,
        };
        return Region {
            sx0: x0,
            sx1: x1,
            sy0: y0,
            sy1: y1,
            window,
        };
    }

    let exact =
        |at: u64, side: u32, level_side: u32| at as f64 * f64::from(side) / f64::from(level_side);
    let sx0 = (x0 * u64::from(sw)) / u64::from(lw);
    let sy0 = (y0 * u64::from(sh)) / u64::from(lh);
    let sx1 = ((u64::from(tx) * u64::from(RASTER_TILE_SIZE) + u64::from(tw)) * u64::from(sw))
        .div_ceil(u64::from(lw))
        .min(u64::from(sw));
    let sy1 = ((u64::from(ty) * u64::from(RASTER_TILE_SIZE) + u64::from(th)) * u64::from(sh))
        .div_ceil(u64::from(lh))
        .min(u64::from(sh));
    let window = Window {
        x0: exact(x0, sw, lw) - sx0 as f64,
        y0: exact(y0, sh, lh) - sy0 as f64,
        x1: exact(
            u64::from(tx) * u64::from(RASTER_TILE_SIZE) + u64::from(tw),
            sw,
            lw,
        ) - sx0 as f64,
        y1: exact(
            u64::from(ty) * u64::from(RASTER_TILE_SIZE) + u64::from(th),
            sh,
            lh,
        ) - sy0 as f64,
    };
    Region {
        sx0,
        sx1,
        sy0,
        sy1,
        window,
    }
}

/// Chunked raster backend driving [`direct`] and [`pass`].
pub trait Chunked {
    /// Decode chunk `index` of IFD/image `image` as tightly packed RGBA8.
    fn chunk(
        &mut self,
        image: usize,
        index: u32,
        cancel: &CancelToken,
    ) -> Result<Arc<DecodedChunk>, RasterError>;
}

/// Parameters governing a [`direct`] pointwise tile read.
pub struct DirectRead<'a> {
    pub grid: &'a Grid,
    pub base: (u32, u32),
    pub level: u8,
    pub wants: &'a [(u32, u32)],
    pub budget: &'a MemoryBudget,
    pub cancel: &'a CancelToken,
}

/// Pointwise reading: decode only the chunks intersecting `req.wants` at
/// `req.level`.
pub fn direct<S: Chunked>(
    source: &mut S,
    req: &DirectRead<'_>,
    emit: TileEmit<'_>,
) -> Result<(), RasterError> {
    let lw = pyramid::level_size(req.base.0, u32::from(req.level));
    let lh = pyramid::level_size(req.base.1, u32::from(req.level));
    let (image, sw, sh) = req.grid.source_for(req.base, lw, lh);
    let (cw, ch) = req.grid.chunk_of(image);
    let across = sw.div_ceil(cw);
    let aligned = lw.abs_diff(sw) <= 1 && lh.abs_diff(sh) <= 1;

    for &(tx, ty) in req.wants {
        if req.cancel.is_cancelled() {
            return Err(RasterError::Cancelled);
        }
        let tw = pyramid::tile_extent(tx, lw);
        let th = pyramid::tile_extent(ty, lh);
        if tw == 0 || th == 0 {
            continue;
        }

        let Region {
            sx0,
            sx1,
            sy0,
            sy1,
            window,
        } = region_of((tx, ty), (tw, th), (lw, lh), (sw, sh), aligned);
        let (rw, rh) = ((sx1 - sx0) as u32, (sy1 - sy0) as u32);
        if rw == 0 || rh == 0 {
            return Err(RasterError::InvalidFormat(format!(
                "tile ({tx}, {ty}) at level {} produced empty source region",
                req.level
            )));
        }
        if u64::from(rw) * u64::from(rh) > req.budget.region_cap_pixels {
            return Err(RasterError::LimitExceeded {
                width: rw,
                height: rh,
                estimated_bytes: u64::from(rw) * u64::from(rh) * 4,
                limit_bytes: req.budget.region_cap_pixels * 4,
            });
        }
        let (sx0, sy0) = (sx0 as u32, sy0 as u32);

        let mut region = vec![0u8; (rw as usize) * (rh as usize) * 4];
        for cy in sy0 / ch..=(sy0 + rh - 1) / ch {
            for cx in sx0 / cw..=(sx0 + rw - 1) / cw {
                if req.cancel.is_cancelled() {
                    return Err(RasterError::Cancelled);
                }
                let chunk = source.chunk(image, cy * across + cx, req.cancel)?;
                let (chunk_x, chunk_y) = (cx * cw, cy * ch);
                let ix0 = sx0.max(chunk_x);
                let iy0 = sy0.max(chunk_y);
                let ix1 = (sx0 + rw).min(chunk_x + chunk.width);
                let iy1 = (sy0 + rh).min(chunk_y + chunk.height);
                if ix0 >= ix1 || iy0 >= iy1 {
                    continue;
                }
                let run = ((ix1 - ix0) as usize) * 4;
                for y in iy0..iy1 {
                    let src_off = (((y - chunk_y) as usize) * (chunk.width as usize)
                        + ((ix0 - chunk_x) as usize))
                        * 4;
                    let dst_off =
                        (((y - sy0) as usize) * (rw as usize) + ((ix0 - sx0) as usize)) * 4;
                    region[dst_off..dst_off + run]
                        .copy_from_slice(&chunk.data[src_off..src_off + run]);
                }
            }
        }

        let mut tile = resample_rgba8_window(&region, rw, rh, window, tw, th);
        bleed_alpha(&mut tile, tw, th);
        emit(req.level, tx, ty, tw, th, tile)?;
    }

    Ok(())
}

/// Sequential top-to-bottom pass over the base IFD feeding `Cascade`.
pub fn pass<S: Chunked>(
    source: &mut S,
    grid: &Grid,
    base: (u32, u32),
    budget: &MemoryBudget,
    cancel: &CancelToken,
    emit: TileEmit<'_>,
) -> Result<(), RasterError> {
    let (width, height) = base;
    let (cw, ch) = grid.chunk;
    let across = width.div_ceil(cw);
    let down = height.div_ceil(ch);
    grid.pass_peak(base).admit(budget.decode_working_bytes)?;

    let mut cascade = Cascade::new(0, width, height);
    let mut bled_emit = |lvl: u8,
                         tx: u32,
                         ty: u32,
                         tw: u32,
                         th: u32,
                         mut rgba: Vec<u8>|
     -> Result<(), RasterError> {
        bleed_alpha(&mut rgba, tw, th);
        emit(lvl, tx, ty, tw, th, rgba)
    };

    for cy in 0..down {
        if cancel.is_cancelled() {
            return Err(RasterError::Cancelled);
        }
        let rows = ch.min(height - cy * ch);
        let mut band = vec![0u8; (width as usize) * (rows as usize) * 4];
        for cx in 0..across {
            if cancel.is_cancelled() {
                return Err(RasterError::Cancelled);
            }
            let chunk = source.chunk(0, cy * across + cx, cancel)?;
            let copy_w = chunk.width.min(width - cx * cw) as usize;
            let run = copy_w * 4;
            for y in 0..chunk.height.min(rows) as usize {
                let src_row = y * (chunk.width as usize) * 4;
                let dst_row = (y * (width as usize) + (cx * cw) as usize) * 4;
                band[dst_row..dst_row + run].copy_from_slice(&chunk.data[src_row..src_row + run]);
            }
        }
        cascade.push_rows(&band, rows, &mut bled_emit)?;
    }
    cascade.finish(&mut bled_emit)
}
