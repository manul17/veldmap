//! TIFF 6.0 Section 21 `YCbCr -> RGBA8` color conversion (`YCbCrCoefficients`
//! tag 529 and `ReferenceBlackWhite` tag 532).

use tiff::decoder::Decoder;
use tiff::tags::Tag;
use veldmap_core::io::BufferedByteReader;
use veldmap_core::sync::CancelToken;

use crate::RasterError;
use crate::pipeline::radiometry::is_data;

use super::tags::read_tag_numeric_f64_vec_opt;

const TAG_YCBCR_COEFFICIENTS: u16 = 529;
const TAG_REFERENCE_BLACK_WHITE: u16 = 532;

/// TIFF 6.0 Section 21 YCbCr -> RGB conversion parameters (`YCbCrCoefficients`
/// tag 529 defaulting to BT.601 `[0.299, 0.587, 0.114]`, and
/// `ReferenceBlackWhite` tag 532 defaulting to full-range `[0, 255, 128, 255, 128, 255]`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct YCbCrParams {
    pub(super) luma: [f64; 3],
    pub(super) ref_black_white: [f64; 6],
}

impl Default for YCbCrParams {
    fn default() -> Self {
        Self {
            luma: [0.299, 0.587, 0.114],
            ref_black_white: [0.0, 255.0, 128.0, 255.0, 128.0, 255.0],
        }
    }
}

impl YCbCrParams {
    /// Convert upsampled 8-bit `YCbCr` triplets directly to tightly packed
    /// `RGBA8` without allocating an intermediate `RGB` buffer.
    pub(super) fn convert_u8_to_rgba(
        &self,
        ycbcr: &[u8],
        pixels: usize,
        nodata: Option<f32>,
    ) -> Vec<u8> {
        let [lr, lg, lb] = self.luma;
        let [rb_y, rw_y, rb_cb, rw_cb, rb_cr, rw_cr] = self.ref_black_white;
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
        let kr = 2.0 - 2.0 * lr;
        let kb = 2.0 - 2.0 * lb;
        let inv_lg = if lg.abs() > 1e-6 {
            1.0 / lg
        } else {
            1.0 / 0.587
        };

        let count = pixels.min(ycbcr.len() / 3);
        let mut rgba = Vec::with_capacity(count * 4);
        for px in ycbcr.chunks_exact(3).take(count) {
            let y = (f64::from(px[0]) - rb_y) * scale_y;
            let cb = (f64::from(px[1]) - rb_cb) * scale_cb;
            let cr = (f64::from(px[2]) - rb_cr) * scale_cr;
            let r = (cr * kr + y).clamp(0.0, 255.0).round() as u8;
            let b = (cb * kb + y).clamp(0.0, 255.0).round() as u8;
            let g = ((y - lb * f64::from(b) - lr * f64::from(r)) * inv_lg)
                .clamp(0.0, 255.0)
                .round() as u8;
            let void = nodata.is_some()
                && !is_data(f32::from(r), nodata)
                && !is_data(f32::from(g), nodata)
                && !is_data(f32::from(b), nodata);
            if void {
                rgba.extend_from_slice(&[0, 0, 0, 0]);
            } else {
                rgba.extend_from_slice(&[r, g, b, 255]);
            }
        }
        rgba
    }
}

pub(super) fn read_ycbcr_params(
    decoder: &mut Decoder<BufferedByteReader>,
    cancel: &CancelToken,
) -> Result<YCbCrParams, RasterError> {
    let mut params = YCbCrParams::default();
    if let Some(coeffs) =
        read_tag_numeric_f64_vec_opt(decoder, Tag::Unknown(TAG_YCBCR_COEFFICIENTS), cancel)?
        && coeffs.len() == 3
        && coeffs.iter().all(|v| v.is_finite() && *v > 0.0 && *v < 1.0)
        && coeffs[1] > 1e-4
    {
        params.luma = [coeffs[0], coeffs[1], coeffs[2]];
    }
    if let Some(rbw) =
        read_tag_numeric_f64_vec_opt(decoder, Tag::Unknown(TAG_REFERENCE_BLACK_WHITE), cancel)?
        && rbw.len() == 6
        && rbw.iter().all(|v| v.is_finite())
        && (rbw[1] - rbw[0]).abs() > 1e-6
        && (rbw[3] - rbw[2]).abs() > 1e-6
        && (rbw[5] - rbw[4]).abs() > 1e-6
    {
        params.ref_black_white = [rbw[0], rbw[1], rbw[2], rbw[3], rbw[4], rbw[5]];
    }
    Ok(params)
}
