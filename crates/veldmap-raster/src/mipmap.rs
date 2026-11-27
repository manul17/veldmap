//! CPU mipmap generation across raster tracks:
//! - Linear-light alpha-weighted mipmaps for `Rgba8UnormSrgb` display tiles
//! - NaN-skipping `f64`-accumulated mipmaps for `R32Float` scientific tiles
//! - 1-mip clearing for `R8Uint`/`R16Uint` categorical tiles.

use veldmap_core::tile::{DecodedTile, PixelFormat, max_mip_levels, mip_dimensions};

use crate::pipeline::{DownsampleRounding, downsample_r32f_2x, downsample_rgba8_2x};

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
