//! CPU mipmap generation across raster tracks:
//! - Linear-light alpha-weighted mipmaps for `Rgba8UnormSrgb` display tiles
//! - NaN-skipping `f64`-accumulated mipmaps for `R32Float` scientific tiles
//! - 1-mip clearing for `R8Uint`/`R16Uint` categorical tiles
//! - Equirectangular horizontal-wrap 2x RGB8 downsampling for bundled basemaps.

use veldmap_core::tile::{DecodedTile, PixelFormat, max_mip_levels, mip_dimensions};

use crate::color::{
    LINEAR_TO_SRGB_LUT_LEN, average_srgb_rgba8_block, linear_to_srgb_lut, mix_2x2_srgb_rgb,
    srgb_to_linear_lut,
};

/// Downsample an opaque packed `Rgb8` sRGB image 2x in linear light, wrapping
/// horizontally for equirectangular continuity.
pub(super) fn downsample_rgb8_srgb_wrap_x_2x(
    src: &[u8],
    src_w: u32,
    src_h: u32,
) -> (u32, u32, Vec<u8>) {
    let to_lin = srgb_to_linear_lut();
    let to_srgb = linear_to_srgb_lut();
    let sw = src_w as usize;
    let sh = src_h as usize;
    let dw = (sw / 2).max(1);
    let dh = (sh / 2).max(1);
    let mut dst = vec![0u8; dw * dh * 3];

    for y in 0..dh {
        let sy0 = (y * 2).min(sh - 1);
        let sy1 = (sy0 + 1).min(sh - 1);
        for x in 0..dw {
            let sx0 = (x * 2).min(sw - 1);
            let sx1 = (sx0 + 1) % sw;
            let offsets = [
                (sy0 * sw + sx0) * 3,
                (sy0 * sw + sx1) * 3,
                (sy1 * sw + sx0) * 3,
                (sy1 * sw + sx1) * 3,
            ];
            let out = (y * dw + x) * 3;
            mix_2x2_srgb_rgb(
                src,
                offsets,
                [0.25; 4],
                to_lin,
                to_srgb,
                &mut dst[out..out + 3],
            );
        }
    }

    (dw as u32, dh as u32, dst)
}

fn downsample_rgba8_srgb_2x(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    to_lin: &[f32; 256],
    to_srgb: &[u8; LINEAR_TO_SRGB_LUT_LEN],
) -> Vec<u8> {
    let sw = src_w as usize;
    let sh = src_h as usize;
    let dw = dst_w as usize;
    let dh = dst_h as usize;
    let mut dst = vec![0u8; dw * dh * 4];

    for y in 0..dh {
        let sy0 = (y * 2).min(sh - 1);
        let sy1 = (sy0 + 1).min(sh - 1);
        for x in 0..dw {
            let sx0 = (x * 2).min(sw - 1);
            let sx1 = (sx0 + 1).min(sw - 1);

            let offsets = [
                (sy0 * sw + sx0) * 4,
                (sy0 * sw + sx1) * 4,
                (sy1 * sw + sx0) * 4,
                (sy1 * sw + sx1) * 4,
            ];

            let out = (y * dw + x) * 4;
            average_srgb_rgba8_block(src, &offsets, to_lin, to_srgb, &mut dst[out..out + 4]);
        }
    }

    dst
}

fn generate_mips_with(
    tile: &mut DecodedTile,
    expected_format: PixelFormat,
    mut step: impl FnMut(&[u8], u32, u32, u32, u32) -> Vec<u8>,
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
        let src = if level == 1 {
            tile.data()
        } else {
            mips[(level - 2) as usize].as_slice()
        };
        mips.push(step(src, prev_w, prev_h, dst_w, dst_h));
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
pub(super) fn generate_linear_mipmaps(tile: &mut DecodedTile) {
    let to_lin = srgb_to_linear_lut();
    let to_srgb = linear_to_srgb_lut();
    generate_mips_with(tile, PixelFormat::Rgba8UnormSrgb, |src, pw, ph, dw, dh| {
        downsample_rgba8_srgb_2x(src, pw, ph, dw, dh, to_lin, to_srgb)
    });
}

fn downsample_r32f_nan_skipping_2x(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
) -> Vec<u8> {
    let sw = src_w as usize;
    let sh = src_h as usize;
    let dw = dst_w as usize;
    let dh = dst_h as usize;
    let mut dst = vec![0u8; dw * dh * 4];

    for y in 0..dh {
        let sy0 = y * 2;
        let sy1 = if y + 1 == dh { sh } else { (sy0 + 2).min(sh) };
        for x in 0..dw {
            let sx0 = x * 2;
            let sx1 = if x + 1 == dw { sw } else { (sx0 + 2).min(sw) };
            let mut sum = 0.0f64;
            let mut count = 0u32;
            for sy in sy0..sy1 {
                let row = sy * sw;
                for sx in sx0..sx1 {
                    let off = (row + sx) * 4;
                    let v =
                        f32::from_ne_bytes([src[off], src[off + 1], src[off + 2], src[off + 3]]);
                    if v.is_finite() {
                        sum += f64::from(v);
                        count += 1;
                    }
                }
            }
            let out_val = if count > 0 {
                (sum / f64::from(count)) as f32
            } else {
                f32::NAN
            };
            let out_off = (y * dw + x) * 4;
            dst[out_off..out_off + 4].copy_from_slice(&out_val.to_ne_bytes());
        }
    }
    dst
}

/// Populate the complete CPU NaN-skipping mipmap chain on an unguttered,
/// unpadded `R32Float` scientific tile (`gutter == 0`, `valid_w == width`,
/// `valid_h == height`, including NPOT clipped edge tiles).
///
/// Finite samples inside each downsample footprint are averaged in `f64`, while
/// `NaN` / non-finite nodata samples are excluded; a footprint containing only
/// non-finite samples produces `f32::NAN`.
fn generate_scientific_mipmaps(tile: &mut DecodedTile) {
    generate_mips_with(tile, PixelFormat::R32Float, downsample_r32f_nan_skipping_2x);
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
