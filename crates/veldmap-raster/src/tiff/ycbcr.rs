//! TIFF 6.0 Section 21 `YCbCr -> RGBA8` color conversion (`YCbCrCoefficients`
//! tag 529 and `ReferenceBlackWhite` tag 532).

use crate::pipeline::radiometry::is_data_f64;

pub(super) const TAG_YCBCR_COEFFICIENTS: u16 = 529;
pub(super) const TAG_REFERENCE_BLACK_WHITE: u16 = 532;

const DEFAULT_LUMA: [f64; 3] = [0.299, 0.587, 0.114];
const DEFAULT_REF_BLACK_WHITE: [f64; 6] = [0.0, 255.0, 128.0, 255.0, 128.0, 255.0];

/// TIFF 6.0 Section 21 YCbCr -> RGB conversion parameters (`YCbCrCoefficients`
/// tag 529 defaulting to BT.601 `[0.299, 0.587, 0.114]`, and
/// `ReferenceBlackWhite` tag 532 defaulting to full-range `[0, 255, 128, 255, 128, 255]`).
///
/// Precomputes invariant per-IFD scale factors and reciprocal luma terms on
/// construction so hot per-pixel conversions stay branchless multiply-adds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct YCbCrParams {
    lr: f64,
    lb: f64,
    inv_lg: f64,
    kr: f64,
    kb: f64,
    rb_y: f64,
    scale_y: f64,
    rb_cb: f64,
    scale_cb: f64,
    rb_cr: f64,
    scale_cr: f64,
}

impl YCbCrParams {
    /// Construct validated `YCbCrParams` from optional raw TIFF tag slices
    /// (`YCbCrCoefficients` tag 529 and `ReferenceBlackWhite` tag 532),
    /// falling back to TIFF 6.0 defaults when absent or malformed.
    pub(super) fn from_tags(coeffs: Option<&[f64]>, ref_black_white: Option<&[f64]>) -> Self {
        let mut luma = DEFAULT_LUMA;
        if let Some(coeffs) = coeffs
            && coeffs.len() == 3
            && coeffs.iter().all(|v| v.is_finite() && *v > 0.0 && *v < 1.0)
            && coeffs[1] > 1e-4
        {
            luma = [coeffs[0], coeffs[1], coeffs[2]];
        }
        let mut rbw_out = DEFAULT_REF_BLACK_WHITE;
        if let Some(rbw) = ref_black_white
            && rbw.len() == 6
            && rbw.iter().all(|v| v.is_finite())
            && (rbw[1] - rbw[0]).abs() > 1e-6
            && (rbw[3] - rbw[2]).abs() > 1e-6
            && (rbw[5] - rbw[4]).abs() > 1e-6
        {
            rbw_out = [rbw[0], rbw[1], rbw[2], rbw[3], rbw[4], rbw[5]];
        }
        Self::new(luma, rbw_out)
    }

    fn new(luma: [f64; 3], ref_black_white: [f64; 6]) -> Self {
        let [lr, lg, lb] = luma;
        let [rb_y, rw_y, rb_cb, rw_cb, rb_cr, rw_cr] = ref_black_white;
        let scale_y = if (rw_y - rb_y).abs() > 1e-6 {
            255.0 / (rw_y - rb_y)
        } else {
            1.0
        };
        let scale_cb = if (rw_cb - rb_cb).abs() > 1e-6 {
            127.0 / (rw_cb - rb_cb)
        } else {
            1.0
        };
        let scale_cr = if (rw_cr - rb_cr).abs() > 1e-6 {
            127.0 / (rw_cr - rb_cr)
        } else {
            1.0
        };
        let inv_lg = if lg.abs() > 1e-6 {
            1.0 / lg
        } else {
            1.0 / 0.587
        };
        Self {
            lr,
            lb,
            inv_lg,
            kr: 2.0 - 2.0 * lr,
            kb: 2.0 - 2.0 * lb,
            rb_y,
            scale_y,
            rb_cb,
            scale_cb,
            rb_cr,
            scale_cr,
        }
    }

    /// Convert a single 8-bit `YCbCr` triplet to `[R, G, B]` using TIFF 6.0
    /// Section 21 `YCbCrCoefficients` (tag 529) and `ReferenceBlackWhite` (tag 532).
    #[inline]
    pub(super) fn convert_u8_pixel_to_rgb(&self, y_raw: u8, cb_raw: u8, cr_raw: u8) -> [u8; 3] {
        let y = (f64::from(y_raw) - self.rb_y) * self.scale_y;
        let cb = (f64::from(cb_raw) - self.rb_cb) * self.scale_cb;
        let cr = (f64::from(cr_raw) - self.rb_cr) * self.scale_cr;
        let r = (cr * self.kr + y).clamp(0.0, 255.0).round() as u8;
        let b = (cb * self.kb + y).clamp(0.0, 255.0).round() as u8;
        let g = ((y - self.lb * f64::from(b) - self.lr * f64::from(r)) * self.inv_lg)
            .clamp(0.0, 255.0)
            .round() as u8;
        [r, g, b]
    }

    /// Convert upsampled 8-bit `YCbCr` triplets directly to tightly packed
    /// `RGBA8` without allocating an intermediate `RGB` buffer.
    pub(super) fn convert_u8_to_rgba(
        &self,
        ycbcr: &[u8],
        pixels: usize,
        nodata: Option<f64>,
    ) -> Vec<u8> {
        let count = pixels.min(ycbcr.len() / 3);
        let mut rgba = Vec::with_capacity(count * 4);
        for px in ycbcr.chunks_exact(3).take(count) {
            let [r, g, b] = self.convert_u8_pixel_to_rgb(px[0], px[1], px[2]);
            let void = nodata.is_some()
                && !is_data_f64(f64::from(r), nodata)
                && !is_data_f64(f64::from(g), nodata)
                && !is_data_f64(f64::from(b), nodata);
            if void {
                rgba.extend_from_slice(&[0, 0, 0, 0]);
            } else {
                rgba.extend_from_slice(&[r, g, b, 255]);
            }
        }
        rgba
    }
}
