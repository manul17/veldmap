//! Shared L1-cached sRGB <-> linear-light lookup tables and alpha-weighted
//! linear-light block-averaging helpers used by both planetary display mipmaps
//! and analytical raster pyramid resampling.

use std::sync::OnceLock;

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

#[inline]
pub(crate) fn mix_2x2_srgb_rgb(
    src: &[u8],
    offsets: [usize; 4],
    weights: [f32; 4],
    to_lin: &[f32; 256],
    to_srgb: &[u8; LINEAR_TO_SRGB_LUT_LEN],
    out: &mut [u8],
) {
    for ch in 0..3 {
        let lin = to_lin[usize::from(src[offsets[0] + ch])] * weights[0]
            + to_lin[usize::from(src[offsets[1] + ch])] * weights[1]
            + to_lin[usize::from(src[offsets[2] + ch])] * weights[2]
            + to_lin[usize::from(src[offsets[3] + ch])] * weights[3];
        out[ch] = linear_to_srgb_u8(lin, to_srgb);
    }
}

/// Alpha-weighted linear-light sRGB block average across `count <= 9` pixel
/// byte offsets in `src`, writing RGBA8 to `out`.
#[inline]
pub(crate) fn average_srgb_rgba8_block(
    src: &[u8],
    offsets: &[usize],
    to_lin: &[f32; 256],
    to_srgb: &[u8; LINEAR_TO_SRGB_LUT_LEN],
    out: &mut [u8],
) {
    let mut a_sum = 0.0_f64;
    let mut lin_rgb = [0.0_f64; 3];
    for &off in offsets {
        let a = f64::from(src[off + 3]) / 255.0;
        a_sum += a;
        for (ch, acc) in lin_rgb.iter_mut().enumerate() {
            *acc += f64::from(to_lin[usize::from(src[off + ch])]) * a;
        }
    }
    if a_sum <= 1e-9 {
        out[..4].copy_from_slice(&[0, 0, 0, 0]);
        return;
    }
    for (ch, acc) in lin_rgb.iter().enumerate() {
        out[ch] = linear_to_srgb_u8((*acc / a_sum) as f32, to_srgb);
    }
    let count = offsets.len().max(1) as f64;
    out[3] = ((a_sum / count) * 255.0).round().clamp(0.0, 255.0) as u8;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srgb_linear_luts_roundtrip_all_code_values() {
        let to_lin = srgb_to_linear_lut();
        let to_srgb = linear_to_srgb_lut();
        for v in 0..=255u8 {
            assert_eq!(linear_to_srgb_u8(to_lin[usize::from(v)], to_srgb), v);
        }
    }
}
