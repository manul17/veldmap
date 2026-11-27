//! Display and scientific radiometry: converting typed raster samples into
//! display-ready RGBA8, exact `f64` nodata checks, deterministic `[2%, 98%]`
//! percentile stretch, and physical-space histogram statistics.

use super::metadata::{BandStats, HISTOGRAM_BINS, SampleType};

/// Maximum samples retained for deterministic per-file percentile stretch and
/// band statistics.
pub(crate) const STRETCH_SAMPLES: usize = 1 << 20;

/// Return whether `value` (`f64`) is finite and distinct from any declared
/// `nodata` (`f64`), preserving full precision for `i32`/`u32`/`f64` rasters.
pub(crate) fn is_data_f64(value: f64, nodata: Option<f64>) -> bool {
    if !value.is_finite() {
        return false;
    }
    match nodata {
        Some(nd) if nd.is_finite() => value != nd,
        _ => true,
    }
}

/// Decode an IEEE-754 binary16 (`f16`) bit pattern into `f32` without extra
/// crate dependencies.
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exp = u32::from((bits >> 10) & 0x1F);
    let mant = u32::from(bits & 0x03FF);
    let f32_bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            // Subnormal binary16: normalize into f32.
            let mut m = mant;
            let mut e = 127u32 - 14;
            while (m & 0x0400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x03FF;
            sign | (e << 23) | (m << 13)
        }
    } else if exp == 31 {
        // Inf or NaN.
        sign | (0xFF << 23) | (mant << 13)
    } else {
        sign | ((exp + (127 - 15)) << 23) | (mant << 13)
    };
    f32::from_bits(f32_bits)
}

/// Decode one native-endian scalar sample at `sample_index` from raw bytes as
/// `f64` (preserving exact integer precision for 32-bit integers and `f64`).
pub(crate) fn read_native_sample_f64(
    bytes: &[u8],
    sample_type: SampleType,
    sample_index: usize,
) -> Option<f64> {
    let sz = sample_type.byte_size();
    let off = sample_index.checked_mul(sz)?;
    let slice = bytes.get(off..off.checked_add(sz)?)?;
    Some(match sample_type {
        SampleType::U8 => f64::from(slice[0]),
        SampleType::I8 => f64::from(slice[0] as i8),
        SampleType::U16 => f64::from(u16::from_ne_bytes([slice[0], slice[1]])),
        SampleType::I16 => f64::from(i16::from_ne_bytes([slice[0], slice[1]])),
        SampleType::U32 => f64::from(u32::from_ne_bytes([slice[0], slice[1], slice[2], slice[3]])),
        SampleType::I32 => f64::from(i32::from_ne_bytes([slice[0], slice[1], slice[2], slice[3]])),
        SampleType::F16 => f64::from(f16_bits_to_f32(u16::from_ne_bytes([slice[0], slice[1]]))),
        SampleType::F32 => f64::from(f32::from_ne_bytes([slice[0], slice[1], slice[2], slice[3]])),
        SampleType::F64 => f64::from_ne_bytes([
            slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7],
        ]),
    })
}

/// Pixel channel interpretation for display mapping.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PixelLayout {
    pub(crate) channels: usize,
    colors: usize,
    alpha_channel: Option<usize>,
}

impl PixelLayout {
    /// Named color model: 1 = Gray, 2 = GrayA, 3 = RGB, 4 = RGBA.
    pub(crate) fn named(channels: usize) -> Self {
        let ch = channels.max(1);
        Self {
            channels: ch,
            colors: match ch {
                1 | 2 => 1,
                _ => 3,
            },
            alpha_channel: match ch {
                2 => Some(1),
                4 => Some(3),
                _ => None,
            },
        }
    }

    /// Arbitrary multi-band stack without color interpretation (displays band
    /// 0 as grayscale, with an optional explicit alpha channel index).
    pub(crate) fn stack_with_alpha(channels: usize, alpha_channel: Option<usize>) -> Self {
        let ch = channels.max(1);
        Self {
            channels: ch,
            colors: 1,
            alpha_channel: alpha_channel.filter(|&a| a < ch),
        }
    }

    /// Multi-band RGB image with optional explicit alpha channel at any index
    /// (`None` for 4-band RGB+NIR with `ExtraSamples = 0`).
    pub(crate) fn rgb(channels: usize, alpha_channel: Option<usize>) -> Self {
        let ch = channels.max(3);
        Self {
            channels: ch,
            colors: 3,
            alpha_channel: alpha_channel.filter(|&a| a < ch),
        }
    }

    /// Number of color channels mapped to RGB (`1` or `3`).
    pub(crate) const fn colors(&self) -> usize {
        self.colors
    }

    /// Explicit alpha channel index within a pixel, if present.
    pub(crate) const fn alpha_channel(&self) -> Option<usize> {
        self.alpha_channel
    }
}

/// Expand raw 8-bit samples directly into a reusable RGBA8 buffer without
/// stretching or nodata keying.
pub(crate) fn expand_u8_into_rgba(
    samples: &[u8],
    pixel: PixelLayout,
    pixels: usize,
    out: &mut Vec<u8>,
) {
    out.clear();
    out.reserve(pixels.saturating_mul(4));
    for px in samples.chunks_exact(pixel.channels).take(pixels) {
        let rgb = if pixel.colors() == 1 {
            [px[0], px[0], px[0]]
        } else {
            [px[0], px[1], px[2]]
        };
        let alpha = pixel.alpha_channel().map_or(255, |ch| px[ch]);
        out.extend_from_slice(&rgb);
        out.push(alpha);
    }
}

/// Expand raw 8-bit samples directly to RGBA8 without stretching or nodata
/// keying.
fn to_rgba8(samples: &[u8], pixel: PixelLayout, pixels: usize) -> Vec<u8> {
    let mut rgba = Vec::new();
    expand_u8_into_rgba(samples, pixel, pixels, &mut rgba);
    rgba
}

/// File-wide display mapping (`[lo, hi] -> [0, 255]`, optional raw `nodata`, and
/// optional `WhiteIsZero` display inversion applied after nodata checks).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct DisplayMapping {
    stretch: Option<(f64, f64)>,
    nodata: Option<f64>,
    invert: bool,
}

impl DisplayMapping {
    /// Identity 8-bit display mapping with optional raw `nodata` transparency.
    pub(crate) fn identity(nodata: Option<f64>) -> Self {
        Self {
            stretch: None,
            nodata,
            invert: false,
        }
    }

    /// Linear `[lo, hi] -> [0, 255]` display stretch with optional raw `nodata`.
    pub(crate) fn stretched(lo: f64, hi: f64, nodata: Option<f64>) -> Self {
        Self {
            stretch: Some((lo, hi)),
            nodata,
            invert: false,
        }
    }

    /// Configure display-only photometric inversion (`WhiteIsZero`).
    pub(crate) const fn with_invert(mut self, invert: bool) -> Self {
        self.invert = invert;
        self
    }

    /// Optional raw `nodata` value configured on this mapping.
    pub(crate) const fn nodata(&self) -> Option<f64> {
        self.nodata
    }
}

fn expand_flat_bounds(lo: f64, hi: f64) -> (f64, f64) {
    if hi > lo {
        return (lo, hi);
    }
    let half = lo.abs().max(1.0) * 1e-6 * 0.5;
    (lo - half, lo + half)
}

fn sorted_percentile(sorted: &[f64], q: f64) -> f64 {
    sorted[((sorted.len() - 1) as f64 * q).round() as usize]
}

fn summarize_sorted_f64(sorted: &[f64]) -> (f64, f64, f64, f64, [u32; HISTOGRAM_BINS]) {
    let min = sorted[0];
    let max = sorted[sorted.len() - 1];
    let (p2, p98) = expand_flat_bounds(
        sorted_percentile(sorted, 0.02),
        sorted_percentile(sorted, 0.98),
    );
    let mut histogram = [0u32; HISTOGRAM_BINS];
    let span = max - min;
    for &v in sorted {
        let bin = if span > 0.0 && span.is_finite() {
            (((v - min) / span) * (HISTOGRAM_BINS as f64))
                .floor()
                .clamp(0.0, (HISTOGRAM_BINS - 1) as f64) as usize
        } else {
            HISTOGRAM_BINS / 2
        };
        histogram[bin] = histogram[bin].saturating_add(1);
    }
    (min, max, p2, p98, histogram)
}

/// Compute `[2%, 98%]` percentile display bounds in `f64` from `values`
/// (`None` if empty). Expands flat constant fields symmetrically around the
/// constant value so flat valid data renders mid-gray rather than black.
pub(crate) fn percentile_stretch(values: &mut [f32]) -> Option<(f64, f64)> {
    if values.is_empty() {
        return None;
    }
    values.sort_unstable_by(f32::total_cmp);
    let at = |q: f64| f64::from(values[((values.len() - 1) as f64 * q).round() as usize]);
    Some(expand_flat_bounds(at(0.02), at(0.98)))
}

/// Convert native-endian raw sample bytes (`sample_type`, `pixel.channels`
/// channels per pixel) into tightly packed `RGBA8` display bytes using this
/// mapping.
pub(crate) fn raw_bytes_to_rgba(
    mapping: &DisplayMapping,
    raw: &[u8],
    sample_type: SampleType,
    pixel: PixelLayout,
    pixels: usize,
) -> Vec<u8> {
    if sample_type == SampleType::U8
        && mapping.stretch.is_none()
        && mapping.nodata.is_none()
        && !mapping.invert
    {
        return to_rgba8(raw, pixel, pixels);
    }
    let (lo, hi) = mapping.stretch.unwrap_or((0.0, 255.0));
    let span = hi - lo;
    let scale = if span > 0.0 && span.is_finite() {
        255.0 / span
    } else {
        0.0
    };
    let nodata = mapping.nodata;
    let sz = sample_type.byte_size();
    let count = pixels.min(raw.len() / (pixel.channels * sz).max(1));
    let mut rgba = vec![0u8; count * 4];
    for (px, dst) in rgba.chunks_exact_mut(4).enumerate() {
        let base = px * pixel.channels;
        let mut void = true;
        let mut broken = false;
        let mut bytes = [0u8; 3];
        for (c, byte) in bytes.iter_mut().enumerate().take(pixel.colors()) {
            let Some(v) = read_native_sample_f64(raw, sample_type, base + c) else {
                broken = true;
                break;
            };
            broken |= !v.is_finite();
            void &= !is_data_f64(v, nodata);
            let mapped = ((v - lo) * scale).round().clamp(0.0, 255.0) as u8;
            *byte = if mapping.invert { 255 - mapped } else { mapped };
        }
        let alpha = if let Some(a_ch) = pixel.alpha_channel() {
            match read_native_sample_f64(raw, sample_type, base + a_ch) {
                Some(a) if a.is_finite() => normalize_alpha_u8(a, sample_type),
                _ => {
                    broken = true;
                    0
                }
            }
        } else {
            255
        };
        if void || broken {
            continue;
        }
        if pixel.colors() == 1 {
            (bytes[1], bytes[2]) = (bytes[0], bytes[0]);
        }
        dst[..3].copy_from_slice(&bytes);
        dst[3] = alpha;
    }
    rgba
}

/// Normalize a finite numeric alpha sample in `sample_type`'s native range to `0..=255`.
#[inline]
pub(crate) fn normalize_alpha_u8(a: f64, sample_type: SampleType) -> u8 {
    match sample_type {
        SampleType::U8 => a.round().clamp(0.0, 255.0) as u8,
        SampleType::I8 => ((a.clamp(0.0, 127.0) / 127.0) * 255.0).round() as u8,
        SampleType::U16 => ((a.clamp(0.0, 65535.0) as u16) >> 8) as u8,
        SampleType::I16 => ((a.clamp(0.0, 32767.0) as i16) >> 7) as u8,
        SampleType::U32 => ((a.clamp(0.0, f64::from(u32::MAX)) as u32) >> 24) as u8,
        SampleType::I32 => ((a.clamp(0.0, f64::from(i32::MAX)) as i32) >> 23) as u8,
        SampleType::F16 | SampleType::F32 | SampleType::F64 => {
            (a * 255.0).round().clamp(0.0, 255.0) as u8
        }
    }
}

/// Compute [`BandStats`] (min, max, p2, p98, dB stats, and 64-bin histograms)
/// from physical values (`raw * scale + offset`). Non-finite inputs are filtered
/// defensively before sorting.
pub(crate) fn compute_band_stats(values: &mut Vec<f64>) -> Option<BandStats> {
    values.retain(|v| v.is_finite());
    if values.is_empty() {
        return None;
    }
    values.sort_unstable_by(f64::total_cmp);
    let n = values.len();
    let (min, max, p2, p98, histogram) = summarize_sorted_f64(values);

    // `10 * log10(v)` is strictly increasing on positive sorted `values`, so
    // `db_vals` is already sorted without a second O(N log N) sort.
    let db_vals: Vec<f64> = values
        .iter()
        .copied()
        .filter(|&v| v > 0.0)
        .map(|v| 10.0 * v.log10())
        .filter(|v| v.is_finite())
        .collect();
    let (db_min, db_max, db_p2, db_p98, db_histogram) = if db_vals.is_empty() {
        (None, None, None, None, [0u32; HISTOGRAM_BINS])
    } else {
        let (dmin, dmax, dp2, dp98, dhist) = summarize_sorted_f64(&db_vals);
        (Some(dmin), Some(dmax), Some(dp2), Some(dp98), dhist)
    };

    Some(BandStats {
        min,
        max,
        p2,
        p98,
        db_min,
        db_max,
        db_p2,
        db_p98,
        sample_count: n as u64,
        histogram,
        db_histogram,
    })
}

#[cfg(test)]
#[path = "radiometry_tests.rs"]
mod tests;
