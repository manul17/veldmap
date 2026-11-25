//! Single-pass pyramid band cascade (`Cascade`): builds every coarser pyramid
//! level from a top-to-bottom stream of base-level row bands with exact 2x2
//! block alignment and predictable working memory (`bytes_for_format`) across
//! `Rgba8UnormSrgb`, `R32Float`, `R8Uint`, and `R16Uint` tracks.

use veldmap_core::tile::{PixelFormat, RASTER_TILE_SIZE, grid_dim, tile_extent};

use crate::RasterError;

use super::resample::{bleed_alpha, halve_f32, halve_rgba8, halve_uint_nearest};

/// Callback invoked whenever a cascade tile `(level, tx, ty, w, h, bytes)`
/// is completed.
pub type TileEmit<'a> =
    &'a mut dyn FnMut(u8, u32, u32, u32, u32, Vec<u8>) -> Result<(), RasterError>;

/// Wrap a display-track [`TileEmit`] callback so every emitted RGBA8 tile runs
/// [`bleed_alpha`] before delivery (kept composable outside [`Cascade`] so
/// non-display scientific/categorical tracks never bleed alpha).
pub(crate) fn bleed_display_emit<'a>(
    emit: TileEmit<'a>,
) -> impl FnMut(u8, u32, u32, u32, u32, Vec<u8>) -> Result<(), RasterError> + 'a {
    move |lvl, tx, ty, tw, th, mut rgba| {
        bleed_alpha(&mut rgba, tw, th);
        emit(lvl, tx, ty, tw, th, rgba)
    }
}

struct CascadeBand {
    level: u8,
    width: u32,
    height: u32,
    top: u32,
    filled: u32,
    buf: Vec<u8>,
}

impl CascadeBand {
    fn new(level: u8, width: u32, height: u32, bpp: usize) -> Self {
        let rows = height.min(RASTER_TILE_SIZE);
        Self {
            level,
            width,
            height,
            top: 0,
            filled: 0,
            buf: vec![0; (width as usize) * (rows as usize) * bpp],
        }
    }

    fn rows(&self) -> u32 {
        (self.height - self.top).min(RASTER_TILE_SIZE)
    }
}

/// Exact working memory in bytes required by [`Cascade`] for `(base_w, base_h)`
/// in RGBA8 display format.
pub(crate) fn bytes(base_w: u32, base_h: u32) -> u64 {
    bytes_for_format(base_w, base_h, PixelFormat::Rgba8UnormSrgb)
}

/// Exact working memory in bytes required by [`Cascade`] for `(base_w, base_h)`
/// in `format`.
pub(crate) fn bytes_for_format(base_w: u32, base_h: u32, format: PixelFormat) -> u64 {
    let bpp = format.bytes_per_pixel() as u64;
    bands_bytes(base_w, base_h, bpp).saturating_add(flush_bytes(base_w, base_h, bpp))
}

fn bands_bytes(base_w: u32, base_h: u32, bpp: u64) -> u64 {
    levels(base_w, base_h)
        .map(|(w, h, _)| u64::from(w) * u64::from(h.min(RASTER_TILE_SIZE)) * bpp)
        .sum()
}

fn flush_bytes(base_w: u32, base_h: u32, bpp: u64) -> u64 {
    let halves: u64 = levels(base_w, base_h)
        .filter(|(_, _, is_apex)| !is_apex)
        .map(|(w, h, _)| {
            let rows = u64::from(h.min(RASTER_TILE_SIZE));
            u64::from(w.div_ceil(2)) * rows.div_ceil(2) * bpp
        })
        .sum();
    let tile = levels(base_w, base_h)
        .map(|(w, h, _)| {
            u64::from(w.min(RASTER_TILE_SIZE)) * u64::from(h.min(RASTER_TILE_SIZE)) * bpp
        })
        .max()
        .unwrap_or(0);
    halves.saturating_add(tile)
}

fn levels(base_w: u32, base_h: u32) -> impl Iterator<Item = (u32, u32, bool)> {
    let (mut w, mut h) = (base_w, base_h);
    let mut done = false;
    std::iter::from_fn(move || {
        if done || w == 0 || h == 0 {
            return None;
        }
        let is_apex = w.max(h) <= RASTER_TILE_SIZE;
        let item = (w, h, is_apex);
        done = is_apex;
        w = w.div_ceil(2);
        h = h.div_ceil(2);
        Some(item)
    })
}

/// Top-to-bottom streaming typed pyramid builder.
pub(crate) struct Cascade {
    format: PixelFormat,
    bands: Vec<CascadeBand>,
}

impl Cascade {
    /// Allocate display (`Rgba8UnormSrgb`) cascade bands from `base_level` up
    /// to the single-tile apex.
    pub(crate) fn new(base_level: u8, base_w: u32, base_h: u32) -> Self {
        Self::new_for_format(base_level, base_w, base_h, PixelFormat::Rgba8UnormSrgb)
    }

    /// Allocate cascade bands for `format` from `base_level` up to the
    /// single-tile apex.
    pub(crate) fn new_for_format(
        base_level: u8,
        base_w: u32,
        base_h: u32,
        format: PixelFormat,
    ) -> Self {
        let bpp = format.bytes_per_pixel();
        let bands = levels(base_w, base_h)
            .enumerate()
            .map(|(idx, (w, h, _))| {
                CascadeBand::new(base_level.saturating_add(idx as u8), w, h, bpp)
            })
            .collect();
        Self { format, bands }
    }

    /// Feed `nrows` contiguous rows at `base_level`.
    pub(crate) fn push_rows(
        &mut self,
        rows: &[u8],
        nrows: u32,
        emit: TileEmit<'_>,
    ) -> Result<(), RasterError> {
        if self.bands.is_empty() {
            return Ok(());
        }
        self.feed(0, rows, nrows, emit)
    }

    /// Flush all incomplete bottom-edge bands top-down.
    pub(crate) fn finish(mut self, emit: TileEmit<'_>) -> Result<(), RasterError> {
        for i in 0..self.bands.len() {
            self.flush(i, emit)?;
        }
        Ok(())
    }

    fn feed(
        &mut self,
        i: usize,
        data: &[u8],
        nrows: u32,
        emit: TileEmit<'_>,
    ) -> Result<(), RasterError> {
        let bpp = self.format.bytes_per_pixel();
        let width = self.bands[i].width as usize;
        let expected_len = (nrows as usize)
            .checked_mul(width)
            .and_then(|px| px.checked_mul(bpp))
            .ok_or_else(|| RasterError::Decode("cascade row slice size overflow".into()))?;
        if data.len() < expected_len {
            return Err(RasterError::Decode(format!(
                "cascade level {} expected at least {expected_len} bytes for {nrows} rows, got {}",
                self.bands[i].level,
                data.len()
            )));
        }
        let mut fed = 0u32;
        while fed < nrows {
            let band = &mut self.bands[i];
            let take = (band.rows() - band.filled).min(nrows - fed);
            if take == 0 {
                return Err(RasterError::Decode(format!(
                    "cascade level {} received more rows than declared height {}",
                    band.level, band.height
                )));
            }
            let src_start = (fed as usize) * width * bpp;
            let copy_len = (take as usize) * width * bpp;
            let dst_start = (band.filled as usize) * width * bpp;
            band.buf[dst_start..dst_start + copy_len]
                .copy_from_slice(&data[src_start..src_start + copy_len]);
            band.filled += take;
            fed += take;
            if band.filled == band.rows() {
                self.flush(i, emit)?;
            }
        }
        Ok(())
    }

    fn flush(&mut self, i: usize, emit: TileEmit<'_>) -> Result<(), RasterError> {
        let bpp = self.format.bytes_per_pixel();
        let (level, width, top, rows) = {
            let band = &self.bands[i];
            (band.level, band.width, band.top, band.filled)
        };
        if rows == 0 {
            return Ok(());
        }

        let ty = top / RASTER_TILE_SIZE;
        for tx in 0..grid_dim(width) {
            let tw = tile_extent(tx, width);
            let mut tile = Vec::with_capacity((tw as usize) * (rows as usize) * bpp);
            let buf = &self.bands[i].buf;
            for row in 0..rows as usize {
                let from = (row * (width as usize) + (tx * RASTER_TILE_SIZE) as usize) * bpp;
                tile.extend_from_slice(&buf[from..from + (tw as usize) * bpp]);
            }
            emit(level, tx, ty, tw, rows, tile)?;
        }

        let half = if i + 1 < self.bands.len() {
            let band = &self.bands[i];
            let filled = &band.buf[..(width as usize) * (rows as usize) * bpp];
            debug_assert_eq!(width.div_ceil(2), self.bands[i + 1].width);
            let halved = match self.format {
                PixelFormat::Rgba8UnormSrgb => halve_rgba8(filled, width, rows),
                PixelFormat::R32Float => halve_f32(filled, width, rows),
                PixelFormat::R8Uint | PixelFormat::R16Uint => {
                    halve_uint_nearest(filled, width, rows, bpp)
                }
            };
            Some((halved, rows.div_ceil(2)))
        } else {
            None
        };

        {
            let band = &mut self.bands[i];
            band.top += rows;
            band.filled = 0;
        }
        if let Some((half_buf, hrows)) = half {
            self.feed(i + 1, &half_buf, hrows, emit)?;
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "cascade_tests.rs"]
mod tests;
