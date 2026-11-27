//! Shared 2x box downsamplers and CPU mipmap generation across raster tracks:
//! - Linear-light alpha-weighted downsampling & mipmaps for `Rgba8UnormSrgb`
//! - NaN-skipping `f64`-accumulated downsampling & mipmaps for `R32Float`
//! - 1-mip clearing for `R8Uint`/`R16Uint` categorical tiles.
//!
//! Kept at crate root above [`crate::color`] so both `basemap/` and `pipeline/`
//! depend unidirectionally on this pixel leaf without layering inversions.

use veldmap_core::tile::{DecodedTile, PixelFormat, max_mip_levels, mip_dimensions};

use crate::color::{average_srgb_rgba8_block, linear_to_srgb_lut, srgb_to_linear_lut};

/// Target-dimension rounding policy for 2x box downsampling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DownsampleRounding {
    /// GPU mipmap chain convention (`max(side / 2, 1)`), folding any trailing
    /// odd row or column into the final destination texel (`1..=3` footprint).
    Floor,
    /// Raster pyramid cascade convention (`ceil(side / 2)`), keeping strict
    /// `1..=2` footprints aligned to `2^k` base-pixel blocks.
    Ceil,
}

impl DownsampleRounding {
    const fn target_dim(self, side: u32) -> u32 {
        match self {
            Self::Floor => {
                let half = side / 2;
                if half > 0 { half } else { 1 }
            }
            Self::Ceil => side.div_ceil(2),
        }
    }

    #[inline]
    fn span(self, d: u32, dst: u32, src: u32) -> (u32, u32) {
        let start = d * 2;
        let end = match self {
            Self::Floor if d + 1 == dst => src,
            Self::Floor | Self::Ceil => (start + 2).min(src),
        };
        (start, end)
    }
}

/// Alpha-weighted linear-light 2x box downsample for `Rgba8UnormSrgb` under
/// `rounding` (`Ceil` for `1..=2` pyramid cascades; `Floor` for `1..=3` GPU
/// mipmap chains folding trailing odd edges).
///
/// Keeps fully transparent blocks deterministic `[0, 0, 0, 0]`.
pub(crate) fn downsample_rgba8_2x(
    src: &[u8],
    w: u32,
    h: u32,
    rounding: DownsampleRounding,
) -> Vec<u8> {
    if w == 0 || h == 0 {
        return Vec::new();
    }
    debug_assert_eq!(src.len(), (w as usize) * (h as usize) * 4);

    let to_lin = srgb_to_linear_lut();
    let to_srgb = linear_to_srgb_lut();
    let (dw, dh) = (rounding.target_dim(w), rounding.target_dim(h));
    let mut out = vec![0u8; (dw as usize) * (dh as usize) * 4];
    for dy in 0..dh {
        let (y0, y1) = rounding.span(dy, dh, h);
        for dx in 0..dw {
            let (x0, x1) = rounding.span(dx, dw, w);
            // Both `Ceil` (`1..=2`) and `Floor` (`1..=3`) produce at most `3 * 3 = 9` samples.
            let mut offsets = [0usize; 9];
            let mut n = 0usize;
            for y in y0..y1 {
                let row = (y as usize) * (w as usize);
                for x in x0..x1 {
                    offsets[n] = (row + (x as usize)) * 4;
                    n += 1;
                }
            }
            let out_px = ((dy as usize) * (dw as usize) + (dx as usize)) * 4;
            average_srgb_rgba8_block(
                src,
                &offsets[..n],
                to_lin,
                to_srgb,
                &mut out[out_px..out_px + 4],
            );
        }
    }
    out
}

#[inline]
pub(crate) fn read_f32_ne(buf: &[u8], pixel_index: usize) -> f32 {
    let off = pixel_index * 4;
    f32::from_ne_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

#[inline]
pub(crate) fn write_f32_ne(buf: &mut [u8], pixel_index: usize, val: f32) {
    let off = pixel_index * 4;
    buf[off..off + 4].copy_from_slice(&val.to_ne_bytes());
}

/// NaN-skipping 2x box downsample for `R32Float` under `rounding` (`Ceil` for
/// `1..=2` pyramid cascades; `Floor` for `1..=3` GPU mipmap chains).
///
/// Finite samples are averaged in `f64`; all-non-finite blocks emit `f32::NAN`.
pub(crate) fn downsample_r32f_2x(
    src: &[u8],
    w: u32,
    h: u32,
    rounding: DownsampleRounding,
) -> Vec<u8> {
    if w == 0 || h == 0 {
        return Vec::new();
    }
    debug_assert_eq!(src.len(), (w as usize) * (h as usize) * 4);
    let (dw, dh) = (rounding.target_dim(w), rounding.target_dim(h));
    let mut out = vec![0u8; (dw as usize) * (dh as usize) * 4];
    for dy in 0..dh {
        let (y0, y1) = rounding.span(dy, dh, h);
        for dx in 0..dw {
            let (x0, x1) = rounding.span(dx, dw, w);
            let mut sum = 0.0f64;
            let mut count = 0u32;
            for y in y0..y1 {
                let row = (y as usize) * (w as usize);
                for x in x0..x1 {
                    let v = read_f32_ne(src, row + (x as usize));
                    if v.is_finite() {
                        sum += f64::from(v);
                        count += 1;
                    }
                }
            }
            let val = if count > 0 {
                (sum / f64::from(count)) as f32
            } else {
                f32::NAN
            };
            write_f32_ne(&mut out, (dy as usize) * (dw as usize) + (dx as usize), val);
        }
    }
    out
}

fn generate_mips_with(
    tile: &mut DecodedTile,
    expected_format: PixelFormat,
    mut step: impl FnMut(&[u8], u32, u32) -> Vec<u8>,
) {
    tile.clear_mips();
    let width = tile.width();
    let height = tile.height();
    if tile.format() != expected_format
        || tile.gutter() != 0
        || tile.valid_w() != width
        || tile.valid_h() != height
    {
        return;
    }

    let total_levels = max_mip_levels(width, height);
    if total_levels <= 1 {
        return;
    }

    let mut mips: Vec<Vec<u8>> = Vec::with_capacity((total_levels - 1) as usize);
    let mut prev_w = width;
    let mut prev_h = height;
    for level in 1..total_levels {
        let (dst_w, dst_h) = mip_dimensions(width, height, level);
        debug_assert_eq!(dst_w, (prev_w / 2).max(1));
        debug_assert_eq!(dst_h, (prev_h / 2).max(1));
        let src = if level == 1 {
            tile.data()
        } else {
            mips[(level - 2) as usize].as_slice()
        };
        mips.push(step(src, prev_w, prev_h));
        prev_w = dst_w;
        prev_h = dst_h;
    }
    let res = tile.try_set_mips(mips);
    debug_assert!(
        res.is_ok(),
        "generated mip chain must match tile layout: {res:?}"
    );
}

/// Populate the complete CPU linear-light mipmap chain on `tile` when
/// `tile.format() == PixelFormat::Rgba8UnormSrgb`, `tile.gutter() == 0`, the
/// valid interior covers the full texture (`tile.valid_w() == tile.width() &&
/// tile.valid_h() == tile.height()`), and `tile.data()` matches
/// `width x height x 4`.
///
/// Non-display formats (`R32Float`, `R8Uint`, `R16Uint`), guttered tiles, and
/// padded partial-edge tiles are left untouched with `mips` cleared so 2x box
/// filtering never shifts gutters or bleeds padding into valid pixels.
fn generate_linear_mipmaps(tile: &mut DecodedTile) {
    generate_mips_with(tile, PixelFormat::Rgba8UnormSrgb, |src, pw, ph| {
        downsample_rgba8_2x(src, pw, ph, DownsampleRounding::Floor)
    });
}

/// Populate the complete CPU NaN-skipping mipmap chain on an unguttered,
/// unpadded `R32Float` scientific tile (`gutter == 0`, `valid_w == width`,
/// `valid_h == height`, including NPOT clipped edge tiles).
///
/// Finite samples inside each downsample footprint are averaged in `f64`, while
/// `NaN` / non-finite nodata samples are excluded; a footprint containing only
/// non-finite samples produces `f32::NAN`.
fn generate_scientific_mipmaps(tile: &mut DecodedTile) {
    generate_mips_with(tile, PixelFormat::R32Float, |src, pw, ph| {
        downsample_r32f_2x(src, pw, ph, DownsampleRounding::Floor)
    });
}

/// Generate format-appropriate CPU mipmaps on `tile`:
/// - `Rgba8UnormSrgb`: linear-light alpha-weighted mip chain
/// - `R32Float`: NaN-skipping numeric mip chain
/// - `R8Uint` / `R16Uint`: 1 mip only (discrete class IDs are never averaged)
pub fn generate_tile_mipmaps(tile: &mut DecodedTile) {
    match tile.format() {
        PixelFormat::Rgba8UnormSrgb => generate_linear_mipmaps(tile),
        PixelFormat::R32Float => generate_scientific_mipmaps(tile),
        PixelFormat::R8Uint | PixelFormat::R16Uint => tile.clear_mips(),
    }
}

#[cfg(test)]
#[path = "mipmap_tests.rs"]
mod tests;
