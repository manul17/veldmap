//! Display and scientific radiometry: converting typed raster samples into
//! display-ready RGBA8, exact `f64` nodata checks, deterministic `[2%, 98%]`
//! percentile stretch, and physical-space histogram statistics.

use super::metadata::{BandStats, HISTOGRAM_BINS, SampleType};

/// Maximum samples retained for deterministic per-file percentile stretch and
/// band statistics.
pub(crate) const STRETCH_SAMPLES: usize = 1 << 20;

/// Return whether `value` is finite and distinct from any declared `nodata`.
pub(crate) fn is_data(value: f32, nodata: Option<f32>) -> bool {
    value.is_finite() && Some(value) != nodata
}

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
pub(crate) fn f16_bits_to_f32(bits: u16) -> f32 {
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
    alpha: bool,
}

impl PixelLayout {
    /// Named color model: 1 = Gray, 2 = GrayA, 3 = RGB, 4+ = RGBA.
    pub(crate) fn named(channels: usize) -> Self {
        Self {
            channels: channels.max(1),
            colors: match channels {
                1 | 2 => 1,
                _ => 3,
            },
            alpha: matches!(channels, 2 | 4),
        }
    }

    /// Arbitrary multi-band stack without color interpretation (displays band
    /// 0 as grayscale without guessing an alpha channel).
    pub(crate) fn stack(channels: usize) -> Self {
        Self {
            channels: channels.max(1),
            colors: 1,
            alpha: false,
        }
    }

    /// Multi-band RGB image with extra non-alpha data bands (e.g. 4-band RGB+NIR
    /// with `ExtraSamples = 0`): displays channels 0, 1, 2 with opaque alpha.
    pub(crate) fn rgb_opaque(channels: usize) -> Self {
        Self {
            channels: channels.max(3),
            colors: 3,
            alpha: false,
        }
    }

    /// Multi-band RGB image whose final channel is explicit alpha.
    pub(crate) fn rgb_with_alpha(channels: usize) -> Self {
        Self {
            channels: channels.max(4),
            colors: 3,
            alpha: true,
        }
    }

    /// Number of color channels mapped to RGB (`1` or `3`).
    pub(crate) const fn colors(&self) -> usize {
        self.colors
    }

    /// Whether the final channel carries opacity.
    pub(crate) const fn has_alpha(&self) -> bool {
        self.alpha
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
        let (rgb, alpha) = match (pixel.colors(), pixel.has_alpha()) {
            (1, false) => ([px[0], px[0], px[0]], 255),
            (1, true) => ([px[0], px[0], px[0]], px[pixel.channels - 1]),
            (_, false) => ([px[0], px[1], px[2]], 255),
            (_, true) => ([px[0], px[1], px[2]], px[pixel.channels - 1]),
        };
        out.extend_from_slice(&rgb);
        out.push(alpha);
    }
}

/// Expand raw 8-bit samples directly to RGBA8 without stretching or nodata
/// keying.
pub(crate) fn to_rgba8(samples: &[u8], pixel: PixelLayout, pixels: usize) -> Vec<u8> {
    let mut rgba = Vec::new();
    expand_u8_into_rgba(samples, pixel, pixels, &mut rgba);
    rgba
}

/// File-wide display mapping (`[lo, hi] -> [0, 255]` + optional raw `nodata`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct DisplayMapping {
    stretch: Option<(f64, f64)>,
    nodata: Option<f32>,
}

impl DisplayMapping {
    /// Identity 8-bit display mapping with optional raw `nodata` transparency.
    pub(crate) fn identity(nodata: Option<f32>) -> Self {
        Self {
            stretch: None,
            nodata,
        }
    }

    /// Linear `[lo, hi] -> [0, 255]` display stretch with optional raw `nodata`.
    pub(crate) fn stretched(lo: f64, hi: f64, nodata: Option<f32>) -> Self {
        Self {
            stretch: Some((lo, hi)),
            nodata,
        }
    }

    /// Optional raw `nodata` value configured on this mapping.
    pub(crate) const fn nodata(&self) -> Option<f32> {
        self.nodata
    }
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
    let (lo, hi) = (at(0.02), at(0.98));
    if hi > lo {
        return Some((lo, hi));
    }
    let half = lo.abs().max(1.0) * 1e-6 * 0.5;
    Some((lo - half, lo + half))
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
    if sample_type == SampleType::U8 && mapping.stretch.is_none() && mapping.nodata.is_none() {
        return to_rgba8(raw, pixel, pixels);
    }
    let (lo, hi) = mapping.stretch.unwrap_or((0.0, 255.0));
    let span = hi - lo;
    let scale = if span > 0.0 && span.is_finite() {
        255.0 / span
    } else {
        0.0
    };
    let nodata = mapping.nodata.map(f64::from);
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
            *byte = ((v - lo) * scale).round().clamp(0.0, 255.0) as u8;
        }
        let alpha = if pixel.has_alpha() {
            match read_native_sample_f64(raw, sample_type, base + pixel.channels - 1) {
                Some(a) if a.is_finite() => match sample_type {
                    SampleType::U8 => a.round().clamp(0.0, 255.0) as u8,
                    SampleType::I8 => a.max(0.0).round().clamp(0.0, 127.0) as u8 * 2,
                    SampleType::U16 => ((a.clamp(0.0, 65535.0) as u16) >> 8) as u8,
                    SampleType::I16 => ((a.clamp(0.0, 32767.0) as i16) >> 7) as u8,
                    SampleType::U32 => ((a.clamp(0.0, f64::from(u32::MAX)) as u32) >> 24) as u8,
                    SampleType::I32 => ((a.clamp(0.0, f64::from(i32::MAX)) as i32) >> 23) as u8,
                    SampleType::F16 | SampleType::F32 | SampleType::F64 => {
                        (a * 255.0).round().clamp(0.0, 255.0) as u8
                    }
                },
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

/// Compute [`BandStats`] (min, max, p2, p98, dB stats, and 64-bin histograms)
/// from finite physical values (`raw * scale + offset`).
pub(crate) fn compute_band_stats(values: &mut [f64]) -> Option<BandStats> {
    if values.is_empty() {
        return None;
    }
    values.sort_unstable_by(f64::total_cmp);
    let n = values.len();
    let at = |slice: &[f64], q: f64| slice[((slice.len() - 1) as f64 * q).round() as usize];
    let min = values[0];
    let max = values[n - 1];
    let (p2, p98) = {
        let lo = at(values, 0.02);
        let hi = at(values, 0.98);
        if hi > lo {
            (lo, hi)
        } else {
            let half = lo.abs().max(1.0) * 1e-6 * 0.5;
            (lo - half, lo + half)
        }
    };

    let mut histogram = [0u32; HISTOGRAM_BINS];
    let lin_span = max - min;
    for &v in values.iter() {
        let bin = if lin_span > 0.0 && lin_span.is_finite() {
            (((v - min) / lin_span) * (HISTOGRAM_BINS as f64))
                .floor()
                .clamp(0.0, (HISTOGRAM_BINS - 1) as f64) as usize
        } else {
            HISTOGRAM_BINS / 2
        };
        histogram[bin] = histogram[bin].saturating_add(1);
    }

    let mut db_vals: Vec<f64> = values
        .iter()
        .copied()
        .filter(|&v| v > 0.0 && v.is_finite())
        .map(|v| 10.0 * v.log10())
        .filter(|v| v.is_finite())
        .collect();
    let (db_min, db_max, db_p2, db_p98, db_histogram) = if db_vals.is_empty() {
        (None, None, None, None, [0u32; HISTOGRAM_BINS])
    } else {
        db_vals.sort_unstable_by(f64::total_cmp);
        let dmin = db_vals[0];
        let dmax = db_vals[db_vals.len() - 1];
        let dlo = at(&db_vals, 0.02);
        let dhi = at(&db_vals, 0.98);
        let (dp2, dp98) = if dhi > dlo {
            (dlo, dhi)
        } else {
            let half = dlo.abs().max(1.0) * 1e-6 * 0.5;
            (dlo - half, dlo + half)
        };
        let mut dhist = [0u32; HISTOGRAM_BINS];
        let dspan = dmax - dmin;
        for &dv in &db_vals {
            let bin = if dspan > 0.0 && dspan.is_finite() {
                (((dv - dmin) / dspan) * (HISTOGRAM_BINS as f64))
                    .floor()
                    .clamp(0.0, (HISTOGRAM_BINS - 1) as f64) as usize
            } else {
                HISTOGRAM_BINS / 2
            };
            dhist[bin] = dhist[bin].saturating_add(1);
        }
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
mod tests {
    use super::*;

    fn f32_bytes(vals: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(vals.len() * 4);
        for &v in vals {
            out.extend_from_slice(&v.to_ne_bytes());
        }
        out
    }

    #[test]
    fn nodata_and_percentile_stretch_behave_consistently() {
        let mapping = DisplayMapping::stretched(0.0, 100.0, Some(-9999.0));
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -9999.0] {
            assert!(!is_data(bad, Some(-9999.0)));
            let out = raw_bytes_to_rgba(
                &mapping,
                &f32_bytes(&[bad]),
                SampleType::F32,
                PixelLayout::named(1),
                1,
            );
            assert_eq!(out, vec![0, 0, 0, 0]);
        }

        // RGB nodata only clears pixels where every color channel equals nodata.
        let rgb_map = DisplayMapping::stretched(0.0, 255.0, Some(0.0));
        let out = raw_bytes_to_rgba(
            &rgb_map,
            &[0, 0, 0, 0, 0, 1],
            SampleType::U8,
            PixelLayout::named(3),
            2,
        );
        assert_eq!(&out[..4], &[0, 0, 0, 0]);
        assert_eq!(&out[4..], &[0, 0, 1, 255]);

        // Flat field lands mid-scale rather than black (including f32::MAX).
        for val in [0.0f32, 42.0, -273.15, 1.0e7, f32::MAX] {
            let mut vals = vec![val; 10];
            let (lo, hi) = percentile_stretch(&mut vals).expect("non-empty");
            assert!(hi > lo);
            let px = raw_bytes_to_rgba(
                &DisplayMapping::stretched(lo, hi, None),
                &f32_bytes(&[val]),
                SampleType::F32,
                PixelLayout::named(1),
                1,
            );
            assert!((100..=160).contains(&px[0]));
            assert_eq!(px[3], 255);
        }
        assert_eq!(percentile_stretch(&mut []), None);

        // Large finite f32 ranges must not overflow `hi - lo` to infinity, and non-finite alpha clears pixel.
        let wide = DisplayMapping::stretched(-2.0e38, 2.0e38, None);
        let out_wide = raw_bytes_to_rgba(
            &wide,
            &f32_bytes(&[0.0]),
            SampleType::F32,
            PixelLayout::named(1),
            1,
        );
        assert!((120..=135).contains(&out_wide[0]));
        let inf_alpha = raw_bytes_to_rgba(
            &DisplayMapping::stretched(0.0, 1.0, None),
            &f32_bytes(&[0.5, f32::INFINITY]),
            SampleType::F32,
            PixelLayout::named(2),
            1,
        );
        assert_eq!(inf_alpha, vec![0, 0, 0, 0]);
        let nan_alpha = raw_bytes_to_rgba(
            &DisplayMapping::stretched(0.0, 1.0, None),
            &f32_bytes(&[0.5, f32::NAN]),
            SampleType::F32,
            PixelLayout::named(2),
            1,
        );
        assert_eq!(nan_alpha, vec![0, 0, 0, 0]);
    }
}
