//! Shared sRGB <-> linear-light lookup tables and CPU linear-light mipmap
//! generation for `Rgba8UnormSrgb` tiles.
//!
//! Precomputing linear-light mipmaps on CPU lets `GpuTile` textures remain
//! strictly `TEXTURE_BINDING | COPY_DST` (no `RENDER_ATTACHMENT` mipgen pass)
//! while pairing with `MipmapFilterMode::Linear` on the globe tile sampler.

use std::sync::OnceLock;

use veldmap_core::tile::{DecodedTile, PixelFormat, max_mip_levels, mip_dimensions};

/// Size of the linear-to-sRGB lookup table (`8 KiB`, fits in L1 cache and
/// round-trips all 256 sRGB code values without `powf` in inner pixel loops).
pub(crate) const LINEAR_TO_SRGB_LUT_LEN: usize = 8192;

pub(crate) fn srgb_to_linear_lut() -> &'static [f32; 256] {
    static LUT: OnceLock<[f32; 256]> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut table = [0.0_f32; 256];
        for (i, slot) in table.iter_mut().enumerate() {
            let s = (i as f32) / 255.0;
            *slot = if s <= 0.04045 {
                s / 12.92
            } else {
                ((s + 0.055) / 1.055).powf(2.4)
            };
        }
        table
    })
}

pub(crate) fn linear_to_srgb_lut() -> &'static [u8; LINEAR_TO_SRGB_LUT_LEN] {
    static LUT: OnceLock<[u8; LINEAR_TO_SRGB_LUT_LEN]> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut table = [0u8; LINEAR_TO_SRGB_LUT_LEN];
        let denom = (LINEAR_TO_SRGB_LUT_LEN - 1) as f32;
        for (i, slot) in table.iter_mut().enumerate() {
            let x = (i as f32) / denom;
            let s = if x <= 0.003_130_8 {
                x * 12.92
            } else {
                1.055 * x.powf(1.0 / 2.4) - 0.055
            };
            *slot = (s * 255.0).round().clamp(0.0, 255.0) as u8;
        }
        table
    })
}

#[inline]
pub(crate) fn linear_to_srgb_u8(linear: f32, lut: &[u8; LINEAR_TO_SRGB_LUT_LEN]) -> u8 {
    let x = linear.clamp(0.0, 1.0);
    if x <= 0.003_130_8 {
        (x * (12.92 * 255.0)).round() as u8
    } else {
        let idx = ((x * ((LINEAR_TO_SRGB_LUT_LEN - 1) as f32)) + 0.5) as usize;
        lut[idx.min(LINEAR_TO_SRGB_LUT_LEN - 1)]
    }
}

/// Downsample an opaque packed `Rgb8` sRGB image 2x in linear light, wrapping
/// horizontally for equirectangular continuity.
pub(crate) fn downsample_rgb8_srgb_wrap_x_2x(
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
            for ch in 0..3 {
                let avg = 0.25
                    * offsets
                        .iter()
                        .map(|&o| to_lin[usize::from(src[o + ch])])
                        .sum::<f32>();
                dst[out + ch] = linear_to_srgb_u8(avg, to_srgb);
            }
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

            let o00 = (sy0 * sw + sx0) * 4;
            let o10 = (sy0 * sw + sx1) * 4;
            let o01 = (sy1 * sw + sx0) * 4;
            let o11 = (sy1 * sw + sx1) * 4;

            let a00 = f32::from(src[o00 + 3]) / 255.0;
            let a10 = f32::from(src[o10 + 3]) / 255.0;
            let a01 = f32::from(src[o01 + 3]) / 255.0;
            let a11 = f32::from(src[o11 + 3]) / 255.0;
            let a_sum = a00 + a10 + a01 + a11;

            let out = (y * dw + x) * 4;
            if a_sum <= 1e-6 {
                dst[out..out + 4].copy_from_slice(&[0, 0, 0, 0]);
                continue;
            }

            // Alpha-weighted linear-light average prevents dark halo bleed from
            // transparent border pixels while reducing to a pure 0.25 average
            // for opaque satellite tiles.
            let inv_a = 1.0 / a_sum;
            for ch in 0..3 {
                let lin = (to_lin[usize::from(src[o00 + ch])] * a00
                    + to_lin[usize::from(src[o10 + ch])] * a10
                    + to_lin[usize::from(src[o01 + ch])] * a01
                    + to_lin[usize::from(src[o11 + ch])] * a11)
                    * inv_a;
                dst[out + ch] = linear_to_srgb_u8(lin, to_srgb);
            }
            dst[out + 3] = ((a_sum * 0.25) * 255.0).round().clamp(0.0, 255.0) as u8;
        }
    }

    dst
}

/// Populate the complete CPU linear-light mipmap chain on `tile` when
/// `tile.format == PixelFormat::Rgba8UnormSrgb`, `tile.gutter == 0`, the valid
/// interior covers the full texture (`valid_w == width && valid_h == height`),
/// and `tile.data` matches `width x height x 4`.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_full_linear_light_mip_chain_and_avoids_dark_srgb_collapse() {
        // 4x4 checkerboard of pure black (0) and pure white (255).
        // Linear average is 0.5 -> sRGB ~188 (whereas naive gamma average is 128).
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
        // 2x2: one bright green opaque pixel + three transparent black pixels.
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

        // Non-power-of-two 5x3 tile produces valid 5x3 -> 2x1 -> 1x1 chain.
        let mut npot =
            DecodedTile::try_new(5, 3, PixelFormat::Rgba8UnormSrgb, vec![200u8; 5 * 3 * 4])
                .expect("valid npot");
        generate_linear_mipmaps(&mut npot);
        assert_eq!(npot.mip_level_count(), 3);

        // Scientific R32Float tiles do not get sRGB mipmaps.
        let mut sci =
            DecodedTile::try_new(4, 4, PixelFormat::R32Float, vec![0u8; 4 * 4 * 4]).expect("sci");
        generate_linear_mipmaps(&mut sci);
        assert_eq!(sci.mip_level_count(), 1);

        // Padded partial-edge and guttered sRGB tiles skip mip generation.
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

    #[test]
    fn srgb_linear_luts_roundtrip_all_code_values() {
        let to_lin = srgb_to_linear_lut();
        let to_srgb = linear_to_srgb_lut();
        for v in 0..=255u8 {
            assert_eq!(linear_to_srgb_u8(to_lin[usize::from(v)], to_srgb), v);
        }
    }
}
