//! Display radiometry: converting `u8`/`u16`/`i16`/`f32` raster samples into
//! display-ready RGBA8 bytes with deterministic `[2%, 98%]` percentile stretch
//! and exact raw-value nodata transparency keying.

/// Maximum samples retained for deterministic per-file percentile stretch.
pub const STRETCH_SAMPLES: usize = 1 << 20;

/// Return whether `value` is finite and distinct from any declared `nodata`.
pub fn is_data(value: f32, nodata: Option<f32>) -> bool {
    value.is_finite() && Some(value) != nodata
}

/// Typed slice of decoded raw raster samples.
pub enum Samples<'a> {
    U8(&'a [u8]),
    U16(&'a [u16]),
    I16(&'a [i16]),
    F32(&'a [f32]),
}

impl Samples<'_> {
    /// Total scalar sample count across all channels.
    pub fn len(&self) -> usize {
        match self {
            Self::U8(v) => v.len(),
            Self::U16(v) => v.len(),
            Self::I16(v) => v.len(),
            Self::F32(v) => v.len(),
        }
    }

    /// Return `true` when the slice contains zero samples.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Read scalar sample `i` as `f32`.
    pub fn get(&self, i: usize) -> f32 {
        match self {
            Self::U8(v) => f32::from(v[i]),
            Self::U16(v) => f32::from(v[i]),
            Self::I16(v) => f32::from(v[i]),
            Self::F32(v) => v[i],
        }
    }

    fn alpha(&self, i: usize) -> u8 {
        match self {
            Self::U8(v) => v[i],
            Self::U16(v) => (v[i] >> 8) as u8,
            Self::I16(v) => v[i].clamp(0, 255) as u8,
            Self::F32(v) => (v[i] * 255.0).round().clamp(0.0, 255.0) as u8,
        }
    }
}

/// Pixel channel interpretation for display mapping.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PixelLayout {
    pub channels: usize,
    colors: usize,
    alpha: bool,
}

impl PixelLayout {
    /// Named color model: 1 = Gray, 2 = GrayA, 3 = RGB, 4+ = RGBA.
    pub fn named(channels: usize) -> Self {
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
    pub fn stack(channels: usize) -> Self {
        Self {
            channels: channels.max(1),
            colors: 1,
            alpha: false,
        }
    }

    /// Number of color channels mapped to RGB (`1` or `3`).
    pub const fn colors(&self) -> usize {
        self.colors
    }

    /// Whether the final channel carries opacity.
    pub const fn has_alpha(&self) -> bool {
        self.alpha
    }
}

/// Expand raw 8-bit samples directly to RGBA8 without stretching or nodata
/// keying.
pub fn to_rgba8(samples: &[u8], pixel: PixelLayout, pixels: usize) -> Vec<u8> {
    let mut rgba = Vec::with_capacity(pixels * 4);
    for px in samples.chunks_exact(pixel.channels).take(pixels) {
        let (rgb, alpha) = match (pixel.colors(), pixel.has_alpha()) {
            (1, false) => ([px[0], px[0], px[0]], 255),
            (1, true) => ([px[0], px[0], px[0]], px[pixel.channels - 1]),
            (_, false) => ([px[0], px[1], px[2]], 255),
            (_, true) => ([px[0], px[1], px[2]], px[pixel.channels - 1]),
        };
        rgba.extend_from_slice(&rgb);
        rgba.push(alpha);
    }
    rgba
}

/// File-wide display mapping (`[lo, hi] -> [0, 255]` + optional raw `nodata`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DisplayMapping {
    stretch: Option<(f32, f32)>,
    nodata: Option<f32>,
}

impl DisplayMapping {
    /// Identity 8-bit display mapping with optional raw `nodata` transparency.
    pub fn identity(nodata: Option<f32>) -> Self {
        Self {
            stretch: None,
            nodata,
        }
    }

    /// Linear `[lo, hi] -> [0, 255]` display stretch with optional raw `nodata`.
    pub fn stretched(lo: f32, hi: f32, nodata: Option<f32>) -> Self {
        Self {
            stretch: Some((lo, hi)),
            nodata,
        }
    }

    /// Convert up to `pixels` raw samples into tightly packed RGBA8 bytes.
    pub fn rgba(&self, samples: &Samples<'_>, pixel: PixelLayout, pixels: usize) -> Vec<u8> {
        if let (Samples::U8(v), None, None) = (samples, self.stretch, self.nodata) {
            return to_rgba8(v, pixel, pixels);
        }

        let (lo, hi) = self.stretch.unwrap_or((0.0, 255.0));
        let scale = if hi > lo { 255.0 / (hi - lo) } else { 0.0 };
        let PixelLayout {
            channels,
            colors,
            alpha: has_alpha,
        } = pixel;

        let count = pixels.min(samples.len() / channels);
        let mut rgba = Vec::with_capacity(count * 4);
        for px in 0..count {
            let at = px * channels;
            let mut void = true;
            let mut broken = false;
            let mut bytes = [0u8; 3];
            for (c, byte) in bytes.iter_mut().enumerate().take(colors) {
                let v = samples.get(at + c);
                broken |= !v.is_finite();
                void &= !is_data(v, self.nodata);
                *byte = ((v - lo) * scale).round().clamp(0.0, 255.0) as u8;
            }
            if void || broken {
                rgba.extend_from_slice(&[0, 0, 0, 0]);
                continue;
            }
            if colors == 1 {
                (bytes[1], bytes[2]) = (bytes[0], bytes[0]);
            }
            rgba.extend_from_slice(&bytes);
            rgba.push(if has_alpha {
                samples.alpha(at + channels - 1)
            } else {
                255
            });
        }
        rgba
    }
}

/// Compute `[2%, 98%]` percentile display bounds from `values` (`None` if empty).
/// Expands flat constant fields symmetrically around the constant value so flat
/// valid data renders mid-gray rather than transparent/black.
pub fn percentile_stretch(values: &mut [f32]) -> Option<(f32, f32)> {
    if values.is_empty() {
        return None;
    }
    values.sort_unstable_by(f32::total_cmp);
    let at = |q: f64| values[((values.len() - 1) as f64 * q).round() as usize];
    let (lo, hi) = (at(0.02), at(0.98));
    if hi > lo {
        return Some((lo, hi));
    }
    let half = lo.abs().max(1.0) * 1e-6 * 0.5;
    Some((lo - half, lo + half))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nodata_and_percentile_stretch_behave_consistently() {
        let mapping = DisplayMapping::stretched(0.0, 100.0, Some(-9999.0));
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -9999.0] {
            assert!(!is_data(bad, Some(-9999.0)));
            let out = mapping.rgba(&Samples::F32(&[bad]), PixelLayout::named(1), 1);
            assert_eq!(out, vec![0, 0, 0, 0]);
        }

        // RGB nodata only clears pixels where every color channel equals nodata.
        let rgb_map = DisplayMapping::stretched(0.0, 255.0, Some(0.0));
        let out = rgb_map.rgba(&Samples::U8(&[0, 0, 0, 0, 0, 1]), PixelLayout::named(3), 2);
        assert_eq!(&out[..4], &[0, 0, 0, 0]);
        assert_eq!(&out[4..], &[0, 0, 1, 255]);

        // Flat field lands mid-scale rather than black.
        for val in [0.0f32, 42.0, -273.15, 1.0e7] {
            let mut vals = vec![val; 10];
            let (lo, hi) = percentile_stretch(&mut vals).expect("non-empty");
            assert!(hi > lo);
            let px = DisplayMapping::stretched(lo, hi, None).rgba(
                &Samples::F32(&[val]),
                PixelLayout::named(1),
                1,
            );
            assert!((100..=160).contains(&px[0]));
            assert_eq!(px[3], 255);
        }
        assert_eq!(percentile_stretch(&mut []), None);
    }
}
