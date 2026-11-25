//! CPU linear-light mipmap generation for `Rgba8UnormSrgb` tiles and
//! equirectangular horizontal-wrap 2x RGB8 downsampling.

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

/// Populate the complete CPU linear-light mipmap chain on `tile` when
/// `tile.format() == PixelFormat::Rgba8UnormSrgb`, `tile.gutter() == 0`, the
/// valid interior covers the full texture (`tile.valid_w() == tile.width() &&
/// tile.valid_h() == tile.height()`), and `tile.data()` matches
/// `width x height x 4`.
///
/// Non-display formats (`R32Float`, `R8Uint`, `R16Uint`), guttered tiles, and
/// padded partial-edge tiles are left untouched with `mips` cleared so 2x box
/// filtering never shifts gutters or bleeds padding into valid pixels.
pub fn generate_linear_mipmaps(tile: &mut DecodedTile) {
    tile.clear_mips();
    let width = tile.width();
    let height = tile.height();
    if tile.format() != PixelFormat::Rgba8UnormSrgb
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

    let to_lin = srgb_to_linear_lut();
    let to_srgb = linear_to_srgb_lut();
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
        let next = downsample_rgba8_srgb_2x(src, prev_w, prev_h, dst_w, dst_h, to_lin, to_srgb);
        mips.push(next);
        prev_w = dst_w;
        prev_h = dst_h;
    }
    let _ = tile.try_set_mips(mips);
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
pub fn generate_scientific_mipmaps(tile: &mut DecodedTile) {
    tile.clear_mips();
    let width = tile.width();
    let height = tile.height();
    if tile.format() != PixelFormat::R32Float
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
        let next = downsample_r32f_nan_skipping_2x(src, prev_w, prev_h, dst_w, dst_h);
        mips.push(next);
        prev_w = dst_w;
        prev_h = dst_h;
    }
    let _ = tile.try_set_mips(mips);
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
mod tests {
    use super::*;

    #[test]
    fn generates_full_linear_light_mip_chain_and_avoids_dark_srgb_collapse() {
        let mut data = Vec::with_capacity(4 * 4 * 4);
        for y in 0..4 {
            for x in 0..4 {
                let v = if (x + y) % 2 == 0 { 0u8 } else { 255u8 };
                data.extend_from_slice(&[v, v, v, 255]);
            }
        }
        let mut tile =
            DecodedTile::try_new(4, 4, PixelFormat::Rgba8UnormSrgb, data).expect("valid tile");
        generate_linear_mipmaps(&mut tile);

        assert_eq!(tile.mip_level_count(), 3);
        assert_eq!(tile.mip_dims(1), (2, 2));
        assert_eq!(tile.mip_dims(2), (1, 1));

        let mip1 = tile.mip_data(1).expect("mip 1");
        assert!(
            mip1[0].abs_diff(188) <= 2 && mip1[3] == 255,
            "expected linear-light ~188 sRGB, got {:?}",
            &mip1[0..4]
        );
        let mip2 = tile.mip_data(2).expect("mip 2");
        assert!(
            mip2[0].abs_diff(188) <= 2 && mip2[3] == 255,
            "expected linear-light ~188 sRGB at 1x1, got {:?}",
            &mip2[0..4]
        );
    }

    #[test]
    fn transparent_neighbors_do_not_darken_opaque_rgb_and_non_srgb_is_untouched() {
        let data = vec![
            0, 255, 0, 255, 0, 0, 0, 0, //
            0, 0, 0, 0, 0, 0, 0, 0,
        ];
        let mut tile =
            DecodedTile::try_new(2, 2, PixelFormat::Rgba8UnormSrgb, data).expect("valid tile");
        generate_linear_mipmaps(&mut tile);
        let mip1 = tile.mip_data(1).expect("mip 1");
        assert_eq!(mip1[0], 0);
        assert_eq!(mip1[1], 255, "RGB must stay pure green without black halo");
        assert_eq!(mip1[2], 0);
        assert!(mip1[3].abs_diff(64) <= 1, "alpha={}", mip1[3]);

        let mut npot =
            DecodedTile::try_new(5, 3, PixelFormat::Rgba8UnormSrgb, vec![200u8; 5 * 3 * 4])
                .expect("valid npot");
        generate_linear_mipmaps(&mut npot);
        assert_eq!(npot.mip_level_count(), 3);

        let mut sci =
            DecodedTile::try_new(4, 4, PixelFormat::R32Float, vec![0u8; 4 * 4 * 4]).expect("sci");
        generate_linear_mipmaps(&mut sci);
        assert_eq!(sci.mip_level_count(), 1);

        let mut padded = DecodedTile::try_with_valid_region(
            4,
            4,
            3,
            4,
            0,
            PixelFormat::Rgba8UnormSrgb,
            vec![255u8; 4 * 4 * 4],
        )
        .expect("padded");
        generate_linear_mipmaps(&mut padded);
        assert_eq!(padded.mip_level_count(), 1);

        let mut guttered = DecodedTile::try_with_valid_region(
            6,
            6,
            4,
            4,
            1,
            PixelFormat::Rgba8UnormSrgb,
            vec![255u8; 6 * 6 * 4],
        )
        .expect("guttered");
        generate_linear_mipmaps(&mut guttered);
        assert_eq!(guttered.mip_level_count(), 1);
    }
}
