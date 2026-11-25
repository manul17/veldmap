//! Shared PNG and JPEG codec leaf utilities: file-signature probes, hardened
//! `png::Decoder` builders, JPEG progressive/lossless working-memory estimates,
//! and PNG/JPEG pixel-format conversions into packed RGB8/RGBA8.

use std::io::{BufRead, Seek};

const PNG_MAGIC: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
const JPEG_MAGIC: [u8; 3] = [0xFF, 0xD8, 0xFF];

/// Return whether `head` begins with an 8-byte PNG file signature.
pub fn has_png_magic(head: &[u8]) -> bool {
    head.starts_with(&PNG_MAGIC)
}

/// Return whether `head` begins with a JPEG SOI + marker prefix (`FF D8 FF`).
pub fn has_jpeg_magic(head: &[u8]) -> bool {
    head.starts_with(&JPEG_MAGIC)
}

/// Build a hardened [`png::Decoder`] bounded by `max_bytes` with ancillary
/// text and iCCP chunks ignored, without enabling `normalize_to_color8()` so
/// callers can inspect raw `info.color_type` and `info.trns` metadata.
pub(crate) fn build_png_decoder<R: BufRead + Seek>(reader: R, max_bytes: u64) -> png::Decoder<R> {
    let limits = png::Limits {
        bytes: max_bytes.min(usize::MAX as u64) as usize,
    };
    let mut decoder = png::Decoder::new_with_limits(reader, limits);
    // Ignore ancillary text/iCCP chunks so adversarial metadata cannot exhaust
    // decode working memory during header parsing.
    decoder.set_ignore_text_chunk(true);
    decoder.set_ignore_iccp_chunk(true);
    decoder
}

/// Build a hardened [`png::Decoder`] configured with
/// `png::Transformations::normalize_to_color8()`.
pub(crate) fn build_normalized_png_decoder<R: BufRead + Seek>(
    reader: R,
    max_bytes: u64,
) -> png::Decoder<R> {
    let mut decoder = build_png_decoder(reader, max_bytes);
    decoder.set_transformations(png::Transformations::normalize_to_color8());
    decoder
}

/// Return the component channel count for a `jpeg-decoder` pixel format.
pub(crate) const fn jpeg_pixel_format_channels(format: jpeg_decoder::PixelFormat) -> usize {
    match format {
        jpeg_decoder::PixelFormat::L8 | jpeg_decoder::PixelFormat::L16 => 1,
        jpeg_decoder::PixelFormat::RGB24 => 3,
        jpeg_decoder::PixelFormat::CMYK32 => 4,
    }
}

/// Return the raw decoded bytes per pixel for a `jpeg-decoder` pixel format.
pub(crate) const fn jpeg_pixel_format_bpp(format: jpeg_decoder::PixelFormat) -> u32 {
    match format {
        jpeg_decoder::PixelFormat::L8 => 1,
        jpeg_decoder::PixelFormat::L16 => 2,
        jpeg_decoder::PixelFormat::RGB24 => 3,
        jpeg_decoder::PixelFormat::CMYK32 => 4,
    }
}

/// Estimate `jpeg-decoder` internal working memory for progressive and lossless
/// scans that `set_max_decoding_buffer_size` does not bound.
///
/// Progressive JPEGs allocate full-resolution `i16` DCT coefficient planes
/// (`~2 B/px/channel`) before IDCT scaling; lossless JPEGs allocate two
/// full-resolution `u16` buffers (`~4 B/px/channel`) and ignore `scale()`.
pub(crate) fn jpeg_codec_working_bytes(
    width: u32,
    height: u32,
    channels: usize,
    coding_process: jpeg_decoder::CodingProcess,
) -> u64 {
    let full_pixels = u64::from(width).saturating_mul(u64::from(height));
    match coding_process {
        jpeg_decoder::CodingProcess::DctProgressive => full_pixels
            .saturating_mul(channels as u64)
            .saturating_mul(2),
        jpeg_decoder::CodingProcess::Lossless => full_pixels
            .saturating_mul(channels as u64)
            .saturating_mul(4),
        jpeg_decoder::CodingProcess::DctSequential => 0,
    }
}

#[inline]
fn cmyk_pixel_to_rgb(px: &[u8]) -> [u8; 3] {
    let kf = 1.0 - f32::from(px[3]) / 255.0;
    let r = ((255.0 - f32::from(px[0])) * kf).round().clamp(0.0, 255.0) as u8;
    let g = ((255.0 - f32::from(px[1])) * kf).round().clamp(0.0, 255.0) as u8;
    let b = ((255.0 - f32::from(px[2])) * kf).round().clamp(0.0, 255.0) as u8;
    [r, g, b]
}

#[inline]
fn push_rgb_or_rgba<const WITH_ALPHA: bool>(out: &mut Vec<u8>, rgb: [u8; 3]) {
    if WITH_ALPHA {
        out.extend_from_slice(&[rgb[0], rgb[1], rgb[2], 255]);
    } else {
        out.extend_from_slice(&rgb);
    }
}

/// Convert `jpeg-decoder`'s un-inverted standard `[C, M, Y, K]` bytes (`0` = no
/// ink / white, `255` = full ink / black) into tightly packed RGB8 or opaque
/// RGBA8.
fn cmyk_to_packed<const WITH_ALPHA: bool>(cmyk: &[u8], pixel_count: usize) -> Vec<u8> {
    let bpp = if WITH_ALPHA { 4 } else { 3 };
    let mut out = Vec::with_capacity(pixel_count.saturating_mul(bpp));
    for px in cmyk.chunks_exact(4).take(pixel_count) {
        push_rgb_or_rgba::<WITH_ALPHA>(&mut out, cmyk_pixel_to_rgb(px));
    }
    out
}

fn gray_to_packed<const WITH_ALPHA: bool>(pixels: &[u8], pixel_count: usize) -> Vec<u8> {
    let bpp = if WITH_ALPHA { 4 } else { 3 };
    let count = pixel_count.min(pixels.len());
    let mut out = Vec::with_capacity(count.saturating_mul(bpp));
    for &g in pixels.iter().take(count) {
        push_rgb_or_rgba::<WITH_ALPHA>(&mut out, [g, g, g]);
    }
    out
}

fn l16_to_packed<const WITH_ALPHA: bool>(pixels: &[u8], pixel_count: usize) -> Vec<u8> {
    let bpp = if WITH_ALPHA { 4 } else { 3 };
    let scale = l16_normalization_scale(pixels, pixel_count);
    let count = pixel_count.min(pixels.len() / 2);
    let mut out = Vec::with_capacity(count.saturating_mul(bpp));
    for p in pixels.chunks_exact(2).take(count) {
        let v = f64::from(u16::from_ne_bytes([p[0], p[1]]));
        let byte = (v * scale).round().clamp(0.0, 255.0) as u8;
        push_rgb_or_rgba::<WITH_ALPHA>(&mut out, [byte, byte, byte]);
    }
    out
}

/// Expand packed RGB8 pixels into opaque RGBA8 (`alpha = 255`).
pub(crate) fn rgb_to_rgba8(pixels: &[u8], pixel_count: usize) -> Vec<u8> {
    let count = pixel_count.min(pixels.len() / 3);
    let mut out = Vec::with_capacity(count.saturating_mul(4));
    for px in pixels.chunks_exact(3).take(count) {
        out.extend_from_slice(&[px[0], px[1], px[2], 255]);
    }
    out
}

/// Strip the alpha channel from packed RGBA8 pixels into tightly packed RGB8.
pub(crate) fn rgba_to_rgb8(pixels: &[u8], pixel_count: usize) -> Vec<u8> {
    let count = pixel_count.min(pixels.len() / 4);
    let mut out = Vec::with_capacity(count.saturating_mul(3));
    for px in pixels.chunks_exact(4).take(count) {
        out.extend_from_slice(&px[..3]);
    }
    out
}

fn png_u8_to_packed<const WITH_ALPHA: bool>(
    color_type: png::ColorType,
    mut buf: Vec<u8>,
    pixel_count: usize,
) -> Vec<u8> {
    match color_type {
        png::ColorType::Rgba if WITH_ALPHA => buf,
        png::ColorType::Rgba => rgba_to_rgb8(&buf, pixel_count),
        png::ColorType::Rgb if !WITH_ALPHA => buf,
        png::ColorType::Rgb => rgb_to_rgba8(&buf, pixel_count),
        png::ColorType::Grayscale => gray_to_packed::<WITH_ALPHA>(&buf, pixel_count),
        png::ColorType::GrayscaleAlpha => {
            let bpp = if WITH_ALPHA { 4 } else { 3 };
            let mut out = Vec::with_capacity(pixel_count.saturating_mul(bpp));
            for px in buf.chunks_exact(2).take(pixel_count) {
                if WITH_ALPHA {
                    out.extend_from_slice(&[px[0], px[0], px[0], px[1]]);
                } else {
                    out.extend_from_slice(&[px[0], px[0], px[0]]);
                }
            }
            out
        }
        png::ColorType::Indexed => {
            // `normalize_to_color8()` always expands indexed palettes before output.
            buf.clear();
            buf
        }
    }
}

/// Convert 8-bit normalized PNG pixels into tightly packed RGBA8, moving
/// already-RGBA buffers in place.
pub(crate) fn png_u8_to_rgba8(
    color_type: png::ColorType,
    buf: Vec<u8>,
    pixel_count: usize,
) -> Vec<u8> {
    png_u8_to_packed::<true>(color_type, buf, pixel_count)
}

/// Convert 8-bit normalized PNG pixels into tightly packed RGB8, moving
/// already-RGB buffers in place.
pub(crate) fn png_u8_to_rgb8(
    color_type: png::ColorType,
    buf: Vec<u8>,
    pixel_count: usize,
) -> Vec<u8> {
    png_u8_to_packed::<false>(color_type, buf, pixel_count)
}

fn l16_normalization_scale(pixels: &[u8], pixel_count: usize) -> f64 {
    // Lossless JPEG L16 stores unshifted 9..=16-bit samples; scan native-endian
    // pairs without allocating an intermediate `Vec<u16>` buffer.
    let max_val = pixels
        .chunks_exact(2)
        .take(pixel_count)
        .map(|p| u16::from_ne_bytes([p[0], p[1]]))
        .max()
        .unwrap_or(255);
    let hi = if max_val <= 255 {
        255.0_f64
    } else if max_val <= 4095 {
        4095.0_f64
    } else {
        65_535.0_f64
    };
    255.0 / hi
}

/// Convert decoded `jpeg-decoder` pixels into tightly packed RGBA8.
pub(crate) fn decode_jpeg_pixels_to_rgba8(
    format: jpeg_decoder::PixelFormat,
    pixels: &[u8],
    pixel_count: usize,
) -> Vec<u8> {
    match format {
        jpeg_decoder::PixelFormat::L8 => gray_to_packed::<true>(pixels, pixel_count),
        jpeg_decoder::PixelFormat::RGB24 => rgb_to_rgba8(pixels, pixel_count),
        jpeg_decoder::PixelFormat::L16 => l16_to_packed::<true>(pixels, pixel_count),
        jpeg_decoder::PixelFormat::CMYK32 => cmyk_to_packed::<true>(pixels, pixel_count),
    }
}

/// Convert decoded `jpeg-decoder` pixels into tightly packed RGB8, moving
/// already-RGB24 buffers in place.
pub(crate) fn decode_jpeg_pixels_to_rgb8(
    format: jpeg_decoder::PixelFormat,
    pixels: Vec<u8>,
    pixel_count: usize,
) -> Vec<u8> {
    match format {
        jpeg_decoder::PixelFormat::RGB24 => pixels,
        jpeg_decoder::PixelFormat::L8 => gray_to_packed::<false>(&pixels, pixel_count),
        jpeg_decoder::PixelFormat::L16 => l16_to_packed::<false>(&pixels, pixel_count),
        jpeg_decoder::PixelFormat::CMYK32 => cmyk_to_packed::<false>(&pixels, pixel_count),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cmyk_conversions_handle_uninverted_standard_cmyk() {
        // White (0,0,0,0), pure cyan (255,0,0,0), pure black (0,0,0,255), mid-gray (0,0,0,128).
        let input = [0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 128];
        let rgba = cmyk_to_packed::<true>(&input, 4);
        assert_eq!(&rgba[0..4], &[255, 255, 255, 255]);
        assert_eq!(&rgba[4..8], &[0, 255, 255, 255]);
        assert_eq!(&rgba[8..12], &[0, 0, 0, 255]);
        assert_eq!(rgba[12], 127);
        assert_eq!(rgba[15], 255);

        let rgb = cmyk_to_packed::<false>(&input, 4);
        assert_eq!(&rgb[0..3], &[255, 255, 255]);
        assert_eq!(&rgb[3..6], &[0, 255, 255]);
        assert_eq!(&rgb[6..9], &[0, 0, 0]);
        assert_eq!(&rgb[9..12], &[127, 127, 127]);
    }
}
