//! Multi-format raster resampling kernels:
//! - Linear-light alpha-weighted RGBA8 halving/window resampling (`halve_rgba8`,
//!   `resample_rgba8_window`) and display-only nodata color bleeding (`bleed_alpha`)
//! - NaN-skipping `R32Float` halving and area-weighted resampling (`halve_f32`,
//!   `resample_f32_window`)
//! - Non-averaging nearest-neighbor `R8Uint`/`R16Uint` halving and window
//!   resampling (`halve_uint_nearest`, `resample_uint_nearest_window`).

use crate::color::{linear_to_srgb_lut, linear_to_srgb_u8, srgb_to_linear_lut};
use crate::mipmap::{
    DownsampleRounding, downsample_r32f_2x, downsample_rgba8_2x, read_f32_ne, write_f32_ne,
};

/// Alpha-weighted linear-light 2x2 sRGB block average
/// (`w x h` -> `ceil(w/2) x ceil(h/2)`).
pub(super) fn halve_rgba8(src: &[u8], w: u32, h: u32) -> Vec<u8> {
    downsample_rgba8_2x(src, w, h, DownsampleRounding::Ceil)
}

/// Fractional sub-rectangle inside a decoded source region in pixel coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Window {
    pub(crate) x0: f64,
    pub(crate) y0: f64,
    pub(crate) x1: f64,
    pub(crate) y1: f64,
}

impl Window {
    #[inline]
    fn is_identity(self, sw: u32, sh: u32, dw: u32, dh: u32) -> bool {
        sw == dw
            && sh == dh
            && self.x0 == 0.0
            && self.y0 == 0.0
            && self.x1 == f64::from(sw)
            && self.y1 == f64::from(sh)
    }
}

/// Area-weighted resampling of fractional `window` inside `src` (`sw x sh`) to
/// `dw x dh` RGBA8 output, preserving exact seam alignment across adjacent tiles.
pub(crate) fn resample_rgba8_window(
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

    if window.is_identity(sw, sh, dw, dh) {
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
pub(super) fn bleed_alpha(rgba: &mut [u8], width: u32, height: u32) {
    const BLEED_STEPS: u32 = 2;
    let (w, h) = (width as usize, height as usize);
    let Some(px_count) = w.checked_mul(h) else {
        return;
    };
    let Some(byte_len) = px_count.checked_mul(4) else {
        return;
    };
    if px_count == 0 || rgba.len() < byte_len {
        return;
    }

    let mut has_opaque = false;
    let mut has_transparent = false;
    for px in rgba[..byte_len].chunks_exact(4) {
        if px[3] != 0 {
            has_opaque = true;
        } else {
            has_transparent = true;
        }
        if has_opaque && has_transparent {
            break;
        }
    }
    if !has_opaque || !has_transparent {
        return;
    }

    let mut known: Vec<bool> = rgba[..byte_len]
        .chunks_exact(4)
        .map(|px| px[3] != 0)
        .collect();
    let mut next = known.clone();

    for step in 0..BLEED_STEPS {
        let mut spread = false;
        for at in 0..px_count {
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
            if step + 1 < BLEED_STEPS {
                next[at] = true;
            }
            spread = true;
        }
        if !spread {
            break;
        }
        if step + 1 < BLEED_STEPS {
            known.copy_from_slice(&next);
        }
    }
}

/// NaN-skipping 2x2 block average for `R32Float` (`w x h` -> `ceil(w/2) x ceil(h/2)`).
pub(super) fn halve_f32(src: &[u8], w: u32, h: u32) -> Vec<u8> {
    downsample_r32f_2x(src, w, h, DownsampleRounding::Ceil)
}

/// Area-weighted NaN-skipping resampling of fractional `window` inside an
/// `R32Float` buffer (`sw x sh`) to `dw x dh`.
pub(super) fn resample_f32_window(
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
    if window.is_identity(sw, sh, dw, dh) {
        return src.to_vec();
    }
    let x_spans = spans(window.x0, window.x1, dw);
    let y_spans = spans(window.y0, window.y1, dh);
    let mut out = vec![0u8; (dw as usize) * (dh as usize) * 4];
    for (dy, &(y0, y1)) in y_spans.iter().enumerate() {
        for (dx, &(x0, x1)) in x_spans.iter().enumerate() {
            let mut sum = 0.0f64;
            let mut weight = 0.0f64;
            let mut sy = y0.max(0.0).floor() as u32;
            while f64::from(sy) < y1 && sy < sh {
                let wy = overlap(sy, y0, y1);
                let row = (sy as usize) * (sw as usize);
                let mut sx = x0.max(0.0).floor() as u32;
                while f64::from(sx) < x1 && sx < sw {
                    let wx = overlap(sx, x0, x1);
                    let v = read_f32_ne(src, row + (sx as usize));
                    if v.is_finite() {
                        let w = wx * wy;
                        sum += f64::from(v) * w;
                        weight += w;
                    }
                    sx += 1;
                }
                sy += 1;
            }
            let cell_area = (x1 - x0) * (y1 - y0);
            let val = if weight > cell_area * 1e-9 {
                (sum / weight) as f32
            } else {
                f32::NAN
            };
            write_f32_ne(&mut out, dy * (dw as usize) + dx, val);
        }
    }
    out
}

/// Non-averaging nearest-neighbor 2x2 decimation for categorical `R8Uint` /
/// `R16Uint` tiles (`bpp` = 1 or 2).
pub(super) fn halve_uint_nearest(src: &[u8], w: u32, h: u32, bpp: usize) -> Vec<u8> {
    if w == 0 || h == 0 || bpp == 0 {
        return Vec::new();
    }
    debug_assert_eq!(src.len(), (w as usize) * (h as usize) * bpp);
    let (dw, dh) = (w.div_ceil(2), h.div_ceil(2));
    let mut out = vec![0u8; (dw as usize) * (dh as usize) * bpp];
    for dy in 0..dh {
        let sy = (dy * 2).min(h - 1) as usize;
        let src_row = sy * (w as usize);
        let dst_row = (dy as usize) * (dw as usize);
        for dx in 0..dw {
            let sx = (dx * 2).min(w - 1) as usize;
            let s_off = (src_row + sx) * bpp;
            let d_off = (dst_row + (dx as usize)) * bpp;
            out[d_off..d_off + bpp].copy_from_slice(&src[s_off..s_off + bpp]);
        }
    }
    out
}

/// Nearest-neighbor resampling of fractional `window` for categorical `R8Uint` / `R16Uint`.
pub(super) fn resample_uint_nearest_window(
    src: &[u8],
    sw: u32,
    sh: u32,
    window: Window,
    dw: u32,
    dh: u32,
    bpp: usize,
) -> Vec<u8> {
    if sw == 0 || sh == 0 || dw == 0 || dh == 0 {
        return Vec::new();
    }
    debug_assert_eq!(src.len(), (sw as usize) * (sh as usize) * bpp);
    if window.is_identity(sw, sh, dw, dh) {
        return src.to_vec();
    }
    let scale_x = (window.x1 - window.x0) / f64::from(dw);
    let scale_y = (window.y1 - window.y0) / f64::from(dh);
    let mut out = vec![0u8; (dw as usize) * (dh as usize) * bpp];
    for dy in 0..dh {
        // Subtract `1e-9` before `floor()` so exact 2x decimation breaks ties
        // toward top-left (`2 * dy`), matching `halve_uint_nearest`.
        let sy_f = window.y0 + (f64::from(dy) + 0.5) * scale_y - 1e-9;
        let sy = (sy_f.floor().clamp(0.0, f64::from(sh - 1))) as usize;
        let src_row = sy * (sw as usize);
        let dst_row = (dy as usize) * (dw as usize);
        for dx in 0..dw {
            let sx_f = window.x0 + (f64::from(dx) + 0.5) * scale_x - 1e-9;
            let sx = (sx_f.floor().clamp(0.0, f64::from(sw - 1))) as usize;
            let s_off = (src_row + sx) * bpp;
            let d_off = (dst_row + (dx as usize)) * bpp;
            out[d_off..d_off + bpp].copy_from_slice(&src[s_off..s_off + bpp]);
        }
    }
    out
}

#[cfg(test)]
#[path = "resample_tests.rs"]
mod tests;
