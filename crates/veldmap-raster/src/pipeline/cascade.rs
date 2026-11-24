//! Single-pass pyramid band cascade (`Cascade`): builds every coarser pyramid
//! level from a top-to-bottom stream of base-level RGBA8 row bands with exact
//! 2x2 block alignment and predictable working memory (`bytes`).

use crate::RasterError;

use super::pyramid::{self, RASTER_TILE_SIZE};
use super::resample::halve_rgba8;

/// Callback invoked whenever a cascade tile `(level, tx, ty, w, h, rgba8)`
/// is completed.
pub type TileEmit<'a> =
    &'a mut dyn FnMut(u8, u32, u32, u32, u32, Vec<u8>) -> Result<(), RasterError>;

struct CascadeBand {
    level: u8,
    width: u32,
    height: u32,
    top: u32,
    filled: u32,
    buf: Vec<u8>,
}

impl CascadeBand {
    fn new(level: u8, width: u32, height: u32) -> Self {
        let rows = height.min(RASTER_TILE_SIZE);
        Self {
            level,
            width,
            height,
            top: 0,
            filled: 0,
            buf: vec![0; (width as usize) * (rows as usize) * 4],
        }
    }

    fn rows(&self) -> u32 {
        (self.height - self.top).min(RASTER_TILE_SIZE)
    }
}

/// Exact working memory in bytes required by [`Cascade`] for `(base_w, base_h)`.
pub(crate) fn bytes(base_w: u32, base_h: u32) -> u64 {
    bands_bytes(base_w, base_h).saturating_add(flush_bytes(base_w, base_h))
}

fn bands_bytes(base_w: u32, base_h: u32) -> u64 {
    levels(base_w, base_h)
        .map(|(w, h, _)| u64::from(w) * u64::from(h.min(RASTER_TILE_SIZE)) * 4)
        .sum()
}

fn flush_bytes(base_w: u32, base_h: u32) -> u64 {
    let halves: u64 = levels(base_w, base_h)
        .filter(|(_, _, is_apex)| !is_apex)
        .map(|(w, h, _)| {
            let rows = u64::from(h.min(RASTER_TILE_SIZE));
            u64::from(w.div_ceil(2)) * rows.div_ceil(2) * 4
        })
        .sum();
    let tile = levels(base_w, base_h)
        .map(|(w, h, _)| {
            u64::from(w.min(RASTER_TILE_SIZE)) * u64::from(h.min(RASTER_TILE_SIZE)) * 4
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

/// Top-to-bottom streaming RGBA8 pyramid builder.
pub(crate) struct Cascade {
    bands: Vec<CascadeBand>,
}

impl Cascade {
    /// Allocate cascade bands from `base_level` up to the single-tile apex.
    pub(crate) fn new(base_level: u8, base_w: u32, base_h: u32) -> Self {
        let bands = levels(base_w, base_h)
            .enumerate()
            .map(|(idx, (w, h, _))| CascadeBand::new(base_level.saturating_add(idx as u8), w, h))
            .collect();
        Self { bands }
    }

    /// Feed `nrows` contiguous RGBA8 rows at `base_level`.
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
        let width = self.bands[i].width as usize;
        let expected_len = (nrows as usize)
            .checked_mul(width)
            .and_then(|px| px.checked_mul(4))
            .ok_or_else(|| RasterError::InvalidFormat("cascade row slice size overflow".into()))?;
        if data.len() < expected_len {
            return Err(RasterError::InvalidFormat(format!(
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
                return Err(RasterError::InvalidFormat(format!(
                    "cascade level {} received more rows than declared height {}",
                    band.level, band.height
                )));
            }
            let src_start = (fed as usize) * width * 4;
            let copy_len = (take as usize) * width * 4;
            let dst_start = (band.filled as usize) * width * 4;
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
        let (level, width, top, rows) = {
            let band = &self.bands[i];
            (band.level, band.width, band.top, band.filled)
        };
        if rows == 0 {
            return Ok(());
        }

        let ty = top / RASTER_TILE_SIZE;
        for tx in 0..pyramid::grid_dim(width) {
            let tw = pyramid::tile_extent(tx, width);
            let mut tile = Vec::with_capacity((tw as usize) * (rows as usize) * 4);
            let buf = &self.bands[i].buf;
            for row in 0..rows as usize {
                let from = (row * (width as usize) + (tx * RASTER_TILE_SIZE) as usize) * 4;
                tile.extend_from_slice(&buf[from..from + (tw as usize) * 4]);
            }
            emit(level, tx, ty, tw, rows, tile)?;
        }

        let half = if i + 1 < self.bands.len() {
            let band = &self.bands[i];
            let filled = &band.buf[..(width as usize) * (rows as usize) * 4];
            debug_assert_eq!(width.div_ceil(2), self.bands[i + 1].width);
            Some((halve_rgba8(filled, width, rows), rows.div_ceil(2)))
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
mod tests {
    use super::*;

    #[test]
    fn cascade_handles_non_tile_row_batches_npot_dims_and_short_slices() {
        let (w, h) = (700u32, 600u32);
        let mut cascade = Cascade::new(0, w, h);
        assert!(
            cascade
                .push_rows(&[0u8; 8], 1, &mut |_, _, _, _, _, _| Ok(()))
                .is_err()
        );

        let mut emitted = Vec::new();
        let row_bytes = (w as usize) * 4;
        // Feed in 130-row batches (non-divisor of 512) to exercise multi-band flushing.
        let mut fed = 0u32;
        while fed < h {
            let n = (h - fed).min(130);
            let buf = vec![42u8; (n as usize) * row_bytes];
            cascade
                .push_rows(&buf, n, &mut |lvl, tx, ty, tw, th, rgba| {
                    emitted.push((lvl, tx, ty, tw, th, rgba.len()));
                    Ok(())
                })
                .expect("push_rows");
            fed += n;
        }
        cascade
            .finish(&mut |lvl, tx, ty, tw, th, rgba| {
                emitted.push((lvl, tx, ty, tw, th, rgba.len()));
                Ok(())
            })
            .expect("finish");

        // Level 0: 2x2 = 4 tiles; Level 1 (350x300 <= 512): 1 apex tile.
        assert_eq!(emitted.len(), 5);
        assert!(emitted.contains(&(1, 0, 0, 350, 300, 350 * 300 * 4)));
        assert!(bytes(w, h) > bands_bytes(w, h));
        // Apex-only raster has zero halving buffer cost.
        assert_eq!(
            flush_bytes(256, 256),
            256 * 256 * 4,
            "apex-only flush_bytes must not count halving"
        );
    }
}
