//! Alpha-weighted linear-light RGBA8 pyramid halving (`halve_rgba8`),
//! fractional window area resampling (`resample_rgba8_window`), and nodata
//! color bleeding (`bleed_alpha`).

use crate::mipmap::{
    average_srgb_rgba8_block, linear_to_srgb_lut, linear_to_srgb_u8, srgb_to_linear_lut,
};

/// Alpha-weighted linear-light 2x2 sRGB block average
/// (`w x h` -> `ceil(w/2) x ceil(h/2)`).
///
/// Keeps fully transparent blocks deterministic `[0, 0, 0, 0]`.
pub fn halve_rgba8(src: &[u8], w: u32, h: u32) -> Vec<u8> {
    if w == 0 || h == 0 {
        return Vec::new();
    }
    debug_assert_eq!(src.len(), (w as usize) * (h as usize) * 4);

    let to_lin = srgb_to_linear_lut();
    let to_srgb = linear_to_srgb_lut();
    let (dw, dh) = (w.div_ceil(2), h.div_ceil(2));
    let mut out = vec![0u8; (dw as usize) * (dh as usize) * 4];
    for dy in 0..dh {
        let y0 = dy * 2;
        let y1 = (y0 + 2).min(h);
        for dx in 0..dw {
            let x0 = dx * 2;
            let x1 = (x0 + 2).min(w);
            let mut offsets = [0usize; 4];
            let mut n = 0usize;
            for y in y0..y1 {
                for x in x0..x1 {
                    offsets[n] = ((y as usize) * (w as usize) + (x as usize)) * 4;
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

/// Fractional sub-rectangle inside a decoded source region in pixel coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Window {
    pub x0: f64,
    pub y0: f64,
    pub x1: f64,
    pub y1: f64,
}

/// Area-weighted resampling of an entire `sw x sh` RGBA8 buffer to `dw x dh`.
pub fn resample_rgba8(src: &[u8], sw: u32, sh: u32, dw: u32, dh: u32) -> Vec<u8> {
    let whole = Window {
        x0: 0.0,
        y0: 0.0,
        x1: f64::from(sw),
        y1: f64::from(sh),
    };
    resample_rgba8_window(src, sw, sh, whole, dw, dh)
}

/// Area-weighted resampling of fractional `window` inside `src` (`sw x sh`) to
/// `dw x dh` RGBA8 output, preserving exact seam alignment across adjacent tiles.
pub fn resample_rgba8_window(
    src: &[u8],
    sw: u32,
    sh: u32,
    window: Window,
    dw: u32,
    dh: u32,
) -> Vec<u8> {
    if sw == 0 || sh == 0 || dw == 0 || dh == 0 {
        return Vec::new();
    }
    debug_assert_eq!(src.len(), (sw as usize) * (sh as usize) * 4);

    if sw == dw
        && sh == dh
        && window.x0 == 0.0
        && window.y0 == 0.0
        && window.x1 == f64::from(sw)
        && window.y1 == f64::from(sh)
    {
        return src.to_vec();
    }

    let to_lin = srgb_to_linear_lut();
    let to_srgb = linear_to_srgb_lut();
    let x_spans = spans(window.x0, window.x1, dw);
    let y_spans = spans(window.y0, window.y1, dh);

    let mut out = vec![0u8; (dw as usize) * (dh as usize) * 4];
    for (dy, &(y0, y1)) in y_spans.iter().enumerate() {
        for (dx, &(x0, x1)) in x_spans.iter().enumerate() {
            let mut rgb = [0f64; 3];
            let mut alpha = 0f64;
            let mut area = 0f64;
            let mut sy = y0.max(0.0).floor() as u32;
            while f64::from(sy) < y1 && sy < sh {
                let wy = overlap(sy, y0, y1);
                let row = (sy as usize) * (sw as usize) * 4;
                let mut sx = x0.max(0.0).floor() as u32;
                while f64::from(sx) < x1 && sx < sw {
                    let wx = overlap(sx, x0, x1);
                    let px = row + (sx as usize) * 4;
                    let w = wx * wy;
                    let aw = (f64::from(src[px + 3]) / 255.0) * w;
                    for c in 0..3 {
                        rgb[c] += f64::from(to_lin[usize::from(src[px + c])]) * aw;
                    }
                    alpha += aw;
                    area += w;
                    sx += 1;
                }
                sy += 1;
            }
            let out_px = (dy * (dw as usize) + dx) * 4;
            for c in 0..3 {
                out[out_px + c] = if alpha <= 1e-9 {
                    0
                } else {
                    linear_to_srgb_u8((rgb[c] / alpha) as f32, to_srgb)
                };
            }
            out[out_px + 3] = if area == 0.0 {
                0
            } else {
                ((alpha / area) * 255.0).round().clamp(0.0, 255.0) as u8
            };
        }
    }
    out
}

fn spans(from: f64, to: f64, dst: u32) -> Vec<(f64, f64)> {
    let scale = (to - from) / f64::from(dst);
    (0..dst)
        .map(|d| {
            let a = from + f64::from(d) * scale;
            let b = from + (f64::from(d) + 1.0) * scale;
            (a, b.min(to))
        })
        .collect()
}

fn overlap(s: u32, a: f64, b: f64) -> f64 {
    let lo = f64::from(s).max(a);
    let hi = f64::from(s + 1).min(b);
    (hi - lo).max(0.0)
}

/// Spread valid RGB colors 2 rings outward into fully transparent (`alpha == 0`)
/// nodata neighbors while keeping alpha at `0`, eliminating dark bilinear edges.
pub fn bleed_alpha(rgba: &mut [u8], width: u32, height: u32) {
    const BLEED_STEPS: u32 = 2;
    let (w, h) = (width as usize, height as usize);
    if w == 0 || h == 0 || rgba.len() < w * h * 4 {
        return;
    }
    let mut coloured: Vec<bool> = (0..w * h).map(|at| rgba[at * 4 + 3] != 0).collect();
    if coloured.iter().all(|has| *has) || coloured.iter().all(|has| !has) {
        return;
    }

    for _ in 0..BLEED_STEPS {
        let known = coloured.clone();
        let mut spread = false;
        for at in 0..w * h {
            if known[at] {
                continue;
            }
            let (x, y) = (at % w, at / w);
            let neighbours = [
                (x > 0).then(|| at - 1),
                (x + 1 < w).then_some(at + 1),
                (y > 0).then(|| at - w),
                (y + 1 < h).then_some(at + w),
            ];
            let Some(from) = neighbours.into_iter().flatten().find(|near| known[*near]) else {
                continue;
            };
            let colour = [rgba[from * 4], rgba[from * 4 + 1], rgba[from * 4 + 2]];
            rgba[at * 4..at * 4 + 3].copy_from_slice(&colour);
            coloured[at] = true;
            spread = true;
        }
        if !spread {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(w: u32, h: u32, value: u8) -> Vec<u8> {
        vec![value; (w * h * 4) as usize]
    }

    #[test]
    fn uniform_stays_uniform_and_fractional_window_seams_align() {
        for (sw, sh, dw, dh) in [(7, 5, 3, 2), (10, 10, 5, 5), (3, 3, 4, 4), (513, 2, 257, 1)] {
            let out = resample_rgba8(&solid(sw, sh, 137), sw, sh, dw, dh);
            assert!(out.iter().all(|&v| v == 137));
        }

        let values: Vec<u8> = (0..9).map(|v| v * 25).collect();
        let mut src = Vec::new();
        for v in &values {
            src.extend_from_slice(&[*v, *v, *v, 255]);
        }
        let whole = resample_rgba8(&src, 9, 1, 6, 1);

        let mut joined = Vec::new();
        for tile in 0..2u32 {
            let (from, to) = (f64::from(tile) * 4.5, f64::from(tile + 1) * 4.5);
            let (sx0, sx1) = (from.floor() as u32, (to.ceil() as u32).min(9));
            let region = &src[(sx0 as usize) * 4..(sx1 as usize) * 4];
            let window = Window {
                x0: from - f64::from(sx0),
                y0: 0.0,
                x1: to - f64::from(sx0),
                y1: 1.0,
            };
            joined.extend(resample_rgba8_window(region, sx1 - sx0, 1, window, 3, 1));
        }
        assert_eq!(joined, whole);
    }

    #[test]
    fn halve_and_bleed_handle_transparency_and_edges() {
        // Checkerboard [0, 255] averages to 0.5 in linear light (~188 sRGB, not 128).
        let checker = [
            0, 0, 0, 255, 255, 255, 255, 255, //
            255, 255, 255, 255, 0, 0, 0, 255,
        ];
        let halved = halve_rgba8(&checker, 2, 2);
        assert!(halved[0].abs_diff(188) <= 2 && halved[3] == 255);
        let resampled = resample_rgba8(&checker, 2, 2, 1, 1);
        assert!(resampled[0].abs_diff(188) <= 2 && resampled[3] == 255);

        // 3x3 odd boundary keeps 2x2, 1x2, 2x1, and 1x1 edge cells exact in linear light.
        let mut src = Vec::new();
        for v in [10u8, 20, 30, 40, 50, 60, 70, 80, 90] {
            src.extend_from_slice(&[v, v, v, 255]);
        }
        let out = halve_rgba8(&src, 3, 3);
        assert_eq!(out.len(), 16);
        assert_eq!(&out[12..16], &[90, 90, 90, 255]);
        assert!(
            out[0] > 30 && out[0] < 40,
            "linear avg > gamma 30: {}",
            out[0]
        );

        let mixed = [
            255, 255, 255, 255, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0,
        ];
        assert_eq!(halve_rgba8(&mixed, 2, 2), vec![255, 255, 255, 128]);
        assert_eq!(resample_rgba8(&mixed, 2, 2, 1, 1), vec![255, 255, 255, 128]);

        let mut bleed_buf = vec![200, 100, 50, 255, 0, 0, 0, 0, 0, 0, 0, 0];
        bleed_alpha(&mut bleed_buf, 3, 1);
        assert_eq!(&bleed_buf[4..8], &[200, 100, 50, 0]);
        assert_eq!(&bleed_buf[8..12], &[200, 100, 50, 0]);
    }
}
