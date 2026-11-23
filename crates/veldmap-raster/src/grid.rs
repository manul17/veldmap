//! Chunk grid driver (`Grid`, `Overview`, `Chunked`, `direct`, `pass`):
//! drives both pointwise random chunk access and single-pass sequential
//! cascade emission while honoring `MemoryBudget` and `CancelToken`.

use std::sync::Arc;

use veldmap_core::budget::{DecodePeak, MemoryBudget};
use veldmap_core::io::CancelToken;

use crate::RasterError;
pub use crate::cascade::TileEmit;
use crate::cascade::{self, Cascade};
use crate::chunk_cache::DecodedChunk;
use crate::pyramid::{self, RASTER_TILE_SIZE};
use crate::resample::{Window, bleed_alpha, resample_rgba8_window};

/// Minimum chunk pixel area eligible for pointwise random access.
pub const MIN_CHUNK_PIXELS: u64 = 32 * 32;

/// Working memory in bytes for decoding a `dw x dh` chunk with `depth` raw
/// bytes/pixel, optionally adding 4 bytes/pixel for RGBA8 display expansion.
pub(crate) fn chunk_working_bytes(
    dw: u32,
    dh: u32,
    depth: u32,
    includes_rgba8_output: bool,
) -> u64 {
    let per_pixel = u64::from(depth).saturating_add(if includes_rgba8_output { 4 } else { 0 });
    u64::from(dw)
        .saturating_mul(u64::from(dh))
        .saturating_mul(per_pixel)
}

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
                && (overview.width, overview.height) < (best.1, best.2)
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
        if base.0 == 0 || base.1 == 0 {
            return None;
        }
        let lw = pyramid::level_size(base.0, u32::from(level)).max(1);
        let lh = pyramid::level_size(base.1, u32::from(level)).max(1);
        let (image, sw, sh) = self.source_for(base, lw, lh);
        if sw == 0 || sh == 0 {
            return None;
        }
        let (cw, ch) = self.chunk_of(image);
        if cw == 0 || ch == 0 || u64::from(cw) * u64::from(ch) < MIN_CHUNK_PIXELS {
            return None;
        }
        let ((num_x, den_x), (num_y, den_y)) = source_scale(base, (sw, sh), level);
        let aligned_x = sw == lw && num_x == den_x;
        let aligned_y = sh == lh && num_y == den_y;

        let side = |num: u128, den: u128, source_side: u32, aligned: bool| -> u64 {
            let ideal = (u128::from(RASTER_TILE_SIZE) * num).div_ceil(den);
            let span = if aligned { ideal } else { ideal + 1 };
            span.min(u128::from(source_side)) as u64
        };
        let rw = side(num_x, den_x, sw, aligned_x);
        let rh = side(num_y, den_y, sh, aligned_y);
        let region_pixels = rw.saturating_mul(rh);
        if region_pixels > budget.region_cap_pixels {
            return None;
        }

        let held_pixels =
            spanned(rw, cw, sw, aligned_x).saturating_mul(spanned(rh, ch, sh, aligned_y));
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
        chunk_working_bytes(chunk.0, chunk.1, self.depth, true)
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
            .with(
                "band",
                u64::from(width)
                    .saturating_mul(u64::from(self.chunk.1))
                    .saturating_mul(4),
            )
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

/// Exact rational `(num, den)` source pixels per level-`level` texel along `(x, y)`
/// matching `pyramid::cell_uv`'s `2^level` base-pixel stride across exact, `ceil`,
/// and GDAL `floor` overviews.
pub(crate) fn source_scale(
    base: (u32, u32),
    source: (u32, u32),
    level: u8,
) -> ((u128, u128), (u128, u128)) {
    let (bw, bh) = base;
    let (sw, sh) = source;
    let lvl = u32::from(level).min(62);
    let lw = pyramid::level_size(bw, lvl).max(1);
    let lh = pyramid::level_size(bh, lvl).max(1);

    if sw == lw && sh == lh {
        return ((1, 1), (1, 1));
    }

    for k in 0..=lvl {
        if sw == pyramid::level_size(bw, k) && sh == pyramid::level_size(bh, k) {
            let stride = 1u128 << (lvl - k);
            return ((stride, 1), (stride, 1));
        }
    }

    let step = 1u128 << lvl;
    (
        (u128::from(sw) * step, u128::from(bw.max(1))),
        (u128::from(sh) * step, u128::from(bh.max(1))),
    )
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
    base: (u32, u32),
    source: (u32, u32),
    level: u8,
) -> Region {
    let (sw, sh) = source;
    let ((num_x, den_x), (num_y, den_y)) = source_scale(base, source, level);
    let (x0, y0) = (
        u128::from(tx) * u128::from(RASTER_TILE_SIZE),
        u128::from(ty) * u128::from(RASTER_TILE_SIZE),
    );
    if num_x == den_x && num_y == den_y {
        let sx0 = x0.min(u128::from(sw).saturating_sub(1)) as u64;
        let sy0 = y0.min(u128::from(sh).saturating_sub(1)) as u64;
        let sx1 = (u128::from(sx0) + u128::from(tw)).min(u128::from(sw)) as u64;
        let sy1 = (u128::from(sy0) + u128::from(th)).min(u128::from(sh)) as u64;
        let window = Window {
            x0: 0.0,
            y0: 0.0,
            x1: (sx1 - sx0) as f64,
            y1: (sy1 - sy0) as f64,
        };
        return Region {
            sx0,
            sx1,
            sy0,
            sy1,
            window,
        };
    }

    let axis =
        |start_t: u128, extent_t: u32, num: u128, den: u128, side: u32| -> (u64, u64, f64, f64) {
            let end_t = start_t + u128::from(extent_t);
            let start_num = start_t * num;
            let end_num = end_t * num;
            let max_start = u128::from(side).saturating_sub(1);
            let s0 = (start_num / den).min(max_start);
            let s1 = end_num.div_ceil(den).min(u128::from(side)).max(s0);
            let s0_num = s0 * den;
            let w0 = (start_num.saturating_sub(s0_num) as f64) / (den as f64);
            let w1 = (end_num.saturating_sub(s0_num) as f64) / (den as f64);
            (s0 as u64, s1 as u64, w0, w1)
        };

    let (sx0, sx1, x0_win, x1_win) = axis(x0, tw, num_x, den_x, sw);
    let (sy0, sy1, y0_win, y1_win) = axis(y0, th, num_y, den_y, sh);
    Region {
        sx0,
        sx1,
        sy0,
        sy1,
        window: Window {
            x0: x0_win,
            y0: y0_win,
            x1: x1_win,
            y1: y1_win,
        },
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
    if req.base.0 == 0 || req.base.1 == 0 {
        return Err(RasterError::InvalidDimensions {
            width: req.base.0,
            height: req.base.1,
        });
    }
    let lw = pyramid::level_size(req.base.0, u32::from(req.level));
    let lh = pyramid::level_size(req.base.1, u32::from(req.level));
    let (image, sw, sh) = req.grid.source_for(req.base, lw, lh);
    let (cw, ch) = req.grid.chunk_of(image);
    if cw == 0 || ch == 0 {
        return Err(RasterError::InvalidDimensions {
            width: cw,
            height: ch,
        });
    }
    let across = sw.div_ceil(cw);

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
        } = region_of((tx, ty), (tw, th), req.base, (sw, sh), req.level);
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
    if width == 0 || height == 0 || cw == 0 || ch == 0 {
        return Err(RasterError::InvalidDimensions { width, height });
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_for_and_region_of_align_npot_base_ceil_and_gdal_floor_overviews() {
        // Narrow 1x2048 raster: overview (1, 1024) has equal width to base, must still be picked.
        let narrow = Grid {
            tiled: true,
            chunk: (1, 512),
            overviews: vec![Overview {
                image: 1,
                width: 1,
                height: 1024,
                chunk: (1, 512),
            }],
            depth: 4,
        };
        assert_eq!(narrow.source_for((1, 2048), 1, 1024), (1, 1, 1024));

        // 8000x5000 at level 4 (lw=500, lh=313, 2^4 = 16):
        // 1) Base IFD (8000x5000): window.y1 = 313 * 16 = 5008.0 so row 312 spans [4992, 5000).
        let r_base = region_of((0, 0), (500, 313), (8000, 5000), (8000, 5000), 4);
        assert_eq!(
            (r_base.sx0, r_base.sx1, r_base.sy0, r_base.sy1),
            (0, 8000, 0, 5000)
        );
        assert!((r_base.window.y1 - 5008.0).abs() < 1e-12);

        // 2) VeldMap `ceil` overview at level 3 (1000x625): window.y1 = 313 * 2 = 626.0.
        let r_ceil = region_of((0, 0), (500, 313), (8000, 5000), (1000, 625), 4);
        assert!((r_ceil.window.y1 - 626.0).abs() < 1e-12);

        // 3) GDAL `floor` overview at level 4 (500x312): 312 rows cover 5000 base px (312.5 level-4 texels),
        // so 313 level-4 texels span `313 * 312 * 16 / 5000 = 312.4992` overview rows.
        let r_gdal = region_of((0, 0), (500, 313), (8000, 5000), (500, 312), 4);
        assert_eq!(r_gdal.sy1, 312);
        assert!((r_gdal.window.y1 - 312.4992).abs() < 1e-9);
    }
}
