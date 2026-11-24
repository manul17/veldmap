//! Memory-bounded PNG/JPEG map-tile decoding and equirectangular image
//! decoding built on `image 0.25` with explicit peak-allocation caps.

use std::io::Cursor;

use image::{ColorType, DynamicImage, ImageDecoder, ImageFormat, ImageReader, Limits};
use veldmap_core::tile::{DecodedTile, PixelFormat};

use crate::RasterError;

use super::mipmap::generate_linear_mipmaps;

/// Maximum single map-tile dimension accepted by [`decode_map_tile`].
const MAX_MAP_TILE_DIM: u32 = 2048;

/// Maximum peak allocation budget for decoding a single map tile (`36 MiB`,
/// covering a `2048 x 2048` RGB/PNG decode plus RGBA8 conversion).
const MAX_MAP_TILE_ALLOC_BYTES: u64 = 36 * 1024 * 1024;

/// Decode raw PNG or JPEG map-tile bytes into tightly packed `Rgba8UnormSrgb`
/// pixels under explicit dimension and peak-allocation limits.
pub fn decode_map_tile(bytes: &[u8]) -> Result<DecodedTile, RasterError> {
    decode_rgba8_limited(bytes, MAX_MAP_TILE_DIM, MAX_MAP_TILE_ALLOC_BYTES)
}

/// Decode raw PNG or JPEG map-tile bytes into tightly packed `Rgba8UnormSrgb`
/// pixels and populate a complete CPU linear-light mipmap chain.
pub fn decode_map_tile_with_mipmaps(bytes: &[u8]) -> Result<DecodedTile, RasterError> {
    let mut tile = decode_map_tile(bytes)?;
    generate_linear_mipmaps(&mut tile);
    Ok(tile)
}

fn make_limits(max_dim: u32, max_alloc_bytes: u64) -> Limits {
    let mut limits = Limits::default();
    limits.max_image_width = Some(max_dim);
    limits.max_image_height = Some(max_dim);
    limits.max_alloc = Some(max_alloc_bytes);
    limits
}

fn map_image_err(
    err: image::ImageError,
    width: u32,
    height: u32,
    estimated_bytes: u64,
    limit_bytes: u64,
) -> RasterError {
    match err {
        image::ImageError::Limits(_) => RasterError::LimitExceeded {
            width,
            height,
            estimated_bytes,
            limit_bytes,
        },
        other => RasterError::Decode(other.to_string()),
    }
}

fn checked_peak_bytes(
    width: u32,
    height: u32,
    src_color: ColorType,
    target: ColorType,
    max_dim: u32,
    max_alloc_bytes: u64,
) -> Result<(u64, u64), RasterError> {
    if width == 0 || height == 0 {
        return Err(RasterError::InvalidDimensions { width, height });
    }
    let px_count = u64::from(width)
        .checked_mul(u64::from(height))
        .ok_or(RasterError::InvalidDimensions { width, height })?;
    let src_bytes = px_count
        .checked_mul(u64::from(src_color.bytes_per_pixel()))
        .ok_or(RasterError::InvalidDimensions { width, height })?;
    let dst_bytes = px_count
        .checked_mul(u64::from(target.bytes_per_pixel()))
        .ok_or(RasterError::InvalidDimensions { width, height })?;

    // When `src_color == target`, `.into_rgba8()` / `.into_rgb8()` moves the
    // decoded buffer in place; otherwise reserve both buffers simultaneously.
    let conversion_bytes = if src_color == target { 0 } else { dst_bytes };
    let estimated_peak = src_bytes.saturating_add(conversion_bytes);
    if width > max_dim || height > max_dim || estimated_peak > max_alloc_bytes {
        return Err(RasterError::LimitExceeded {
            width,
            height,
            estimated_bytes: estimated_peak,
            limit_bytes: max_alloc_bytes,
        });
    }
    Ok((conversion_bytes, estimated_peak))
}

fn decode_limited_for_target(
    bytes: &[u8],
    max_dim: u32,
    max_alloc_bytes: u64,
    target: ColorType,
) -> Result<DynamicImage, RasterError> {
    let mut reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|_| RasterError::UnsupportedFormat)?;

    let format = match reader.format() {
        Some(f @ (ImageFormat::Png | ImageFormat::Jpeg)) => f,
        _ => return Err(RasterError::UnsupportedFormat),
    };

    // Apply dimension and allocation limits BEFORE `into_decoder()` so codecs
    // (notably `png::Reader`, whose `set_limits` does not lower internal buffer
    // budgets in `image 0.25`) enforce `max_dim` and `max_alloc_bytes` while
    // parsing headers and metadata chunks.
    reader.limits(make_limits(max_dim, max_alloc_bytes));
    let mut decoder = reader
        .into_decoder()
        .map_err(|err| map_image_err(err, 0, 0, 0, max_alloc_bytes))?;
    let (width, height) = decoder.dimensions();
    let (conversion_bytes, estimated_peak) = checked_peak_bytes(
        width,
        height,
        decoder.color_type(),
        target,
        max_dim,
        max_alloc_bytes,
    )?;

    let final_decoder = if conversion_bytes == 0 {
        decoder
    } else {
        let tight_limits = make_limits(max_dim, max_alloc_bytes - conversion_bytes);
        if format == ImageFormat::Png {
            // Drop the first decoder BEFORE creating the tighter one so two
            // `PngDecoder` instances never overlap in memory, while ensuring
            // `png::Reader`'s internal budget honors `max_alloc - conversion`.
            drop(decoder);
            let mut tight_reader = ImageReader::with_format(Cursor::new(bytes), format);
            tight_reader.limits(tight_limits);
            tight_reader
                .into_decoder()
                .map_err(|err| map_image_err(err, width, height, estimated_peak, max_alloc_bytes))?
        } else {
            decoder.set_limits(tight_limits).map_err(|err| {
                map_image_err(err, width, height, estimated_peak, max_alloc_bytes)
            })?;
            decoder
        }
    };

    DynamicImage::from_decoder(final_decoder)
        .map_err(|err| map_image_err(err, width, height, estimated_peak, max_alloc_bytes))
}

/// Decode a PNG or JPEG image into tightly packed `Rgba8UnormSrgb` pixels
/// while bounding both decoder working set and RGBA8 output within
/// `max_alloc_bytes`.
fn decode_rgba8_limited(
    bytes: &[u8],
    max_dim: u32,
    max_alloc_bytes: u64,
) -> Result<DecodedTile, RasterError> {
    let img =
        decode_limited_for_target(bytes, max_dim, max_alloc_bytes, ColorType::Rgba8)?.into_rgba8();
    let (w, h) = img.dimensions();
    DecodedTile::try_new(w, h, PixelFormat::Rgba8UnormSrgb, img.into_raw()).map_err(|_| {
        RasterError::InvalidDimensions {
            width: w,
            height: h,
        }
    })
}

/// Decode an opaque PNG or JPEG image directly into tightly packed `Rgb8`
/// bytes while bounding peak memory (including any non-RGB8 conversion buffer)
/// within `max_alloc_bytes`.
pub(super) fn decode_rgb8_limited(
    bytes: &[u8],
    max_dim: u32,
    max_alloc_bytes: u64,
) -> Result<(u32, u32, Box<[u8]>), RasterError> {
    let img =
        decode_limited_for_target(bytes, max_dim, max_alloc_bytes, ColorType::Rgb8)?.into_rgb8();
    let (w, h) = img.dimensions();
    if w == 0 || h == 0 {
        return Err(RasterError::InvalidDimensions {
            width: w,
            height: h,
        });
    }
    Ok((w, h, img.into_raw().into_boxed_slice()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Luma, Rgb, Rgba};

    #[test]
    fn decodes_valid_png_and_jpeg_tiles_and_rejects_invalid_inputs() {
        let img: ImageBuffer<Rgba<u8>, Vec<u8>> =
            ImageBuffer::from_fn(4, 4, |x, y| Rgba([x as u8 * 40, y as u8 * 40, 128, 255]));

        let mut png_bytes = Vec::new();
        img.write_to(&mut Cursor::new(&mut png_bytes), ImageFormat::Png)
            .expect("encode png");
        let decoded_png = decode_map_tile(&png_bytes).expect("decode png");
        assert_eq!((decoded_png.width(), decoded_png.height()), (4, 4));
        assert_eq!(decoded_png.format(), PixelFormat::Rgba8UnormSrgb);
        assert_eq!(decoded_png.byte_size(), 4 * 4 * 4);
        assert_eq!(&decoded_png.data()[0..4], &[0, 0, 128, 255]);

        // Grayscale PNG converts cleanly to opaque RGBA8.
        let gray: ImageBuffer<Luma<u8>, Vec<u8>> = ImageBuffer::from_pixel(4, 4, Luma([180]));
        let mut gray_png = Vec::new();
        gray.write_to(&mut Cursor::new(&mut gray_png), ImageFormat::Png)
            .expect("encode gray png");
        let decoded_gray = decode_map_tile(&gray_png).expect("decode gray png");
        assert_eq!(&decoded_gray.data()[0..4], &[180, 180, 180, 255]);

        // Solid-color RGB JPEG preserves color within JPEG quantization tolerance.
        let rgb: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_pixel(8, 8, Rgb([200, 80, 40]));
        let mut jpg_bytes = Vec::new();
        rgb.write_to(&mut Cursor::new(&mut jpg_bytes), ImageFormat::Jpeg)
            .expect("encode jpeg");
        let decoded_jpg = decode_map_tile(&jpg_bytes).expect("decode jpeg");
        assert_eq!((decoded_jpg.width(), decoded_jpg.height()), (8, 8));
        assert_eq!(decoded_jpg.byte_size(), 8 * 8 * 4);
        assert!(decoded_jpg.data()[0].abs_diff(200) <= 6);
        assert!(decoded_jpg.data()[1].abs_diff(80) <= 6);
        assert!(decoded_jpg.data()[2].abs_diff(40) <= 6);
        assert_eq!(decoded_jpg.data()[3], 255);

        // Corrupt / unsupported / truncated payloads fail cleanly.
        assert!(matches!(
            decode_map_tile(b"not an image"),
            Err(RasterError::UnsupportedFormat)
        ));
        assert!(matches!(
            decode_map_tile(&[]),
            Err(RasterError::UnsupportedFormat)
        ));
        assert!(matches!(
            decode_map_tile(b"GIF89a\x01\x00\x01\x00\x00\x00\x00;"),
            Err(RasterError::UnsupportedFormat)
        ));
        assert!(matches!(
            decode_map_tile(&png_bytes[..png_bytes.len() / 2]),
            Err(RasterError::Decode(_))
        ));

        // Explicit dimension and peak-allocation limits are enforced before full decode.
        assert!(matches!(
            decode_rgba8_limited(&png_bytes, 2, MAX_MAP_TILE_ALLOC_BYTES),
            Err(RasterError::LimitExceeded { .. })
        ));
        // 4x4 RGBA8 PNG (64 B) rejected when budget < 64 B.
        assert!(matches!(
            decode_rgba8_limited(&png_bytes, MAX_MAP_TILE_DIM, 63),
            Err(RasterError::LimitExceeded { .. })
        ));
        // 8x8 RGB8 JPEG (192 B src + 256 B dst = 448 B peak) rejected at 400 B.
        assert!(matches!(
            decode_rgba8_limited(&jpg_bytes, MAX_MAP_TILE_DIM, 400),
            Err(RasterError::LimitExceeded { .. })
        ));
        // 4x4 RGBA8 PNG converted to RGB8 (64 B src + 48 B dst = 112 B peak) rejected at 100 B.
        assert!(matches!(
            decode_rgb8_limited(&png_bytes, MAX_MAP_TILE_DIM, 100),
            Err(RasterError::LimitExceeded { .. })
        ));
        // Concise Debug formatting omits raw pixel dumps.
        let dbg = format!("{decoded_png:?}");
        assert!(dbg.contains("byte_size: 64"), "unexpected Debug: {dbg}");
    }
}
