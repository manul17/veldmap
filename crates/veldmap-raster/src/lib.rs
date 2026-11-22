//! Native raster decoding and offline basemap tile generation for VeldMap.
//!
//! Keeps `veldmap-core` permanently free of image/codec dependencies while
//! providing memory-bounded PNG/JPEG map-tile decoding and the virtual offline
//! bundled equirectangular basemap provider (`z = 0..3` Geodetic + `z = 0`
//! Polar Stereographic caps).

use std::io::Cursor;

use image::{ImageFormat, ImageReader, Limits};
use thiserror::Error;
use veldmap_core::tile::{PixelFormat, TileScheme};

pub mod basemap;

pub use basemap::{BUNDLED_MAX_LEVEL, BUNDLED_POLAR_MAX_LEVEL, BUNDLED_TILE_PX, BundledBasemap};

/// Maximum single map-tile dimension accepted by [`decode_map_tile`] and
/// [`BundledBasemap::render_tile_sized`].
pub const MAX_MAP_TILE_DIM: u32 = 2048;

/// Maximum peak allocation budget for decoding a single map tile (`36 MiB`,
/// covering a `2048 x 2048` RGB/PNG decode plus RGBA8 conversion).
pub const MAX_MAP_TILE_ALLOC_BYTES: u64 = 36 * 1024 * 1024;

/// Maximum dimension accepted for a bundled global equirectangular basemap.
pub const MAX_EQUIR_DIM: u32 = 8192;

/// Maximum peak allocation budget for decoding a bundled global equirectangular
/// basemap image (`128 MiB`).
pub const MAX_EQUIR_ALLOC_BYTES: u64 = 128 * 1024 * 1024;

/// Errors returned by raster tile decoders and virtual tile providers.
#[derive(Debug, Error)]
pub enum RasterError {
    #[error("unsupported or unrecognized image format")]
    UnsupportedFormat,
    #[error("image decode failed: {0}")]
    Decode(#[from] image::ImageError),
    #[error("invalid decoded image dimensions {width}x{height}")]
    InvalidDimensions { width: u32, height: u32 },
    #[error("image {width}x{height} exceeds decode limits ({estimated_bytes} B > {limit_bytes} B)")]
    LimitExceeded {
        width: u32,
        height: u32,
        estimated_bytes: u64,
        limit_bytes: u64,
    },
    #[error("tile request out of bounds or above max_level: {scheme:?} z={level} ({x}, {y})")]
    TileOutOfRange {
        scheme: TileScheme,
        level: u8,
        x: u32,
        y: u32,
    },
}

/// CPU-decoded tile pixels ready for GPU texture upload.
#[derive(Clone, Debug)]
pub struct DecodedTile {
    pub width: u32,
    pub height: u32,
    pub format: PixelFormat,
    pub data: Vec<u8>,
}

impl DecodedTile {
    /// Exact unpacked byte size of `data`.
    pub fn byte_size(&self) -> usize {
        self.data.len()
    }
}

/// Decode raw PNG or JPEG map-tile bytes into tightly packed `Rgba8UnormSrgb`
/// pixels under explicit dimension and peak-allocation limits.
pub fn decode_map_tile(bytes: &[u8]) -> Result<DecodedTile, RasterError> {
    decode_rgba8_limited(bytes, MAX_MAP_TILE_DIM, MAX_MAP_TILE_ALLOC_BYTES)
}

fn inspect_and_check_limits(
    bytes: &[u8],
    max_dim: u32,
    max_alloc_bytes: u64,
    output_bpp: u64,
) -> Result<(u32, u32, u64), RasterError> {
    let reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|_| RasterError::UnsupportedFormat)?;

    match reader.format() {
        Some(ImageFormat::Png | ImageFormat::Jpeg) => {}
        _ => return Err(RasterError::UnsupportedFormat),
    }

    let (width, height) = reader.into_dimensions()?;
    if width == 0 || height == 0 {
        return Err(RasterError::InvalidDimensions { width, height });
    }
    let output_bytes = u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|px| px.checked_mul(output_bpp))
        .ok_or(RasterError::InvalidDimensions { width, height })?;
    if width > max_dim || height > max_dim || output_bytes > max_alloc_bytes {
        return Err(RasterError::LimitExceeded {
            width,
            height,
            estimated_bytes: output_bytes,
            limit_bytes: max_alloc_bytes,
        });
    }
    Ok((width, height, output_bytes))
}

fn decode_image_limited(
    bytes: &[u8],
    max_dim: u32,
    decoder_alloc_limit: u64,
) -> Result<image::DynamicImage, RasterError> {
    let mut reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|_| RasterError::UnsupportedFormat)?;

    let mut limits = Limits::default();
    limits.max_image_width = Some(max_dim);
    limits.max_image_height = Some(max_dim);
    limits.max_alloc = Some(decoder_alloc_limit);
    reader.limits(limits);

    Ok(reader.decode()?)
}

/// Decode a PNG or JPEG image into tightly packed `Rgba8UnormSrgb` pixels
/// while bounding both decoder working set and RGBA8 output within
/// `max_alloc_bytes`.
pub(crate) fn decode_rgba8_limited(
    bytes: &[u8],
    max_dim: u32,
    max_alloc_bytes: u64,
) -> Result<DecodedTile, RasterError> {
    let (width, height, rgba_bytes) = inspect_and_check_limits(bytes, max_dim, max_alloc_bytes, 4)?;
    // Reserve space for `.into_rgba8()` conversion when the source is non-RGBA
    // (e.g. RGB8 JPEG), while still allowing an in-place RGBA8 PNG decode.
    let rgb_overhead = (rgba_bytes * 3) / 4;
    if rgba_bytes.saturating_add(rgb_overhead) > max_alloc_bytes {
        return Err(RasterError::LimitExceeded {
            width,
            height,
            estimated_bytes: rgba_bytes.saturating_add(rgb_overhead),
            limit_bytes: max_alloc_bytes,
        });
    }
    let decoder_limit = max_alloc_bytes.saturating_sub(rgba_bytes).max(rgba_bytes);
    let img = decode_image_limited(bytes, max_dim, decoder_limit)?.into_rgba8();
    let (w, h) = img.dimensions();
    if w == 0 || h == 0 {
        return Err(RasterError::InvalidDimensions {
            width: w,
            height: h,
        });
    }
    Ok(DecodedTile {
        width: w,
        height: h,
        format: PixelFormat::Rgba8UnormSrgb,
        data: img.into_raw(),
    })
}

/// Decode an opaque PNG or JPEG image directly into tightly packed `Rgb8`
/// bytes without an intermediate RGBA8 expansion.
pub(crate) fn decode_rgb8_limited(
    bytes: &[u8],
    max_dim: u32,
    max_alloc_bytes: u64,
) -> Result<(u32, u32, Box<[u8]>), RasterError> {
    let (_width, _height, _rgb_bytes) =
        inspect_and_check_limits(bytes, max_dim, max_alloc_bytes, 3)?;
    let img = decode_image_limited(bytes, max_dim, max_alloc_bytes)?.into_rgb8();
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
        assert_eq!((decoded_png.width, decoded_png.height), (4, 4));
        assert_eq!(decoded_png.format, PixelFormat::Rgba8UnormSrgb);
        assert_eq!(decoded_png.byte_size(), 4 * 4 * 4);
        assert_eq!(&decoded_png.data[0..4], &[0, 0, 128, 255]);

        // Grayscale PNG converts cleanly to opaque RGBA8.
        let gray: ImageBuffer<Luma<u8>, Vec<u8>> = ImageBuffer::from_pixel(4, 4, Luma([180]));
        let mut gray_png = Vec::new();
        gray.write_to(&mut Cursor::new(&mut gray_png), ImageFormat::Png)
            .expect("encode gray png");
        let decoded_gray = decode_map_tile(&gray_png).expect("decode gray png");
        assert_eq!(&decoded_gray.data[0..4], &[180, 180, 180, 255]);

        // Solid-color RGB JPEG preserves color within JPEG quantization tolerance.
        let rgb: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_pixel(8, 8, Rgb([200, 80, 40]));
        let mut jpg_bytes = Vec::new();
        rgb.write_to(&mut Cursor::new(&mut jpg_bytes), ImageFormat::Jpeg)
            .expect("encode jpeg");
        let decoded_jpg = decode_map_tile(&jpg_bytes).expect("decode jpeg");
        assert_eq!((decoded_jpg.width, decoded_jpg.height), (8, 8));
        assert_eq!(decoded_jpg.byte_size(), 8 * 8 * 4);
        assert!(decoded_jpg.data[0].abs_diff(200) <= 6);
        assert!(decoded_jpg.data[1].abs_diff(80) <= 6);
        assert!(decoded_jpg.data[2].abs_diff(40) <= 6);
        assert_eq!(decoded_jpg.data[3], 255);

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
        assert!(matches!(
            decode_rgba8_limited(&png_bytes, MAX_MAP_TILE_DIM, 64),
            Err(RasterError::LimitExceeded { .. })
        ));
    }
}
