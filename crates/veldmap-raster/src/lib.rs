//! Native raster decoding and offline basemap tile generation for VeldMap.
//!
//! Keeps `veldmap-core` permanently free of image/codec dependencies while
//! providing memory-bounded PNG/JPEG map-tile decoding and the virtual offline
//! bundled equirectangular basemap provider (`z = 0..3` Geodetic + `z = 0..1`
//! Polar Stereographic caps).

use std::io::Cursor;

use image::{ImageFormat, ImageReader, Limits};
use thiserror::Error;
use veldmap_core::tile::PixelFormat;

pub mod basemap;

pub use basemap::{BUNDLED_MAX_LEVEL, BUNDLED_POLAR_MAX_LEVEL, BUNDLED_TILE_PX, BundledBasemap};

/// Maximum single map-tile dimension accepted by [`decode_map_tile`].
pub const MAX_MAP_TILE_DIM: u32 = 4096;

/// Maximum allocation budget for decoding a single map tile (`64 MiB`).
pub const MAX_MAP_TILE_ALLOC_BYTES: u64 = 64 * 1024 * 1024;

/// Maximum allocation budget for decoding a bundled global equirectangular
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
    #[error("tile request out of bounds or above max_level: level={level}, x={x}, y={y}")]
    TileOutOfRange { level: u8, x: u32, y: u32 },
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
/// pixels under explicit dimension and allocation limits.
pub fn decode_map_tile(bytes: &[u8]) -> Result<DecodedTile, RasterError> {
    decode_rgba8_limited(bytes, MAX_MAP_TILE_DIM, MAX_MAP_TILE_ALLOC_BYTES)
}

/// Decode a PNG or JPEG image with explicit dimension and allocation caps.
pub(crate) fn decode_rgba8_limited(
    bytes: &[u8],
    max_dim: u32,
    max_alloc_bytes: u64,
) -> Result<DecodedTile, RasterError> {
    let mut reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|_| RasterError::UnsupportedFormat)?;

    match reader.format() {
        Some(ImageFormat::Png | ImageFormat::Jpeg) => {}
        _ => return Err(RasterError::UnsupportedFormat),
    }

    let mut limits = Limits::default();
    limits.max_image_width = Some(max_dim);
    limits.max_image_height = Some(max_dim);
    limits.max_alloc = Some(max_alloc_bytes);
    reader.limits(limits);

    let img = reader.decode()?.into_rgba8();
    let (width, height) = img.dimensions();
    if width == 0 || height == 0 {
        return Err(RasterError::InvalidDimensions { width, height });
    }
    Ok(DecodedTile {
        width,
        height,
        format: PixelFormat::Rgba8UnormSrgb,
        data: img.into_raw(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgba};

    #[test]
    fn decodes_valid_png_and_jpeg_tiles_and_rejects_garbage() {
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

        let rgb = image::DynamicImage::ImageRgba8(img).into_rgb8();
        let mut jpg_bytes = Vec::new();
        rgb.write_to(&mut Cursor::new(&mut jpg_bytes), ImageFormat::Jpeg)
            .expect("encode jpeg");
        let decoded_jpg = decode_map_tile(&jpg_bytes).expect("decode jpeg");
        assert_eq!((decoded_jpg.width, decoded_jpg.height), (4, 4));
        assert_eq!(decoded_jpg.byte_size(), 4 * 4 * 4);

        // Corrupt / unsupported bytes must fail cleanly without panicking.
        assert!(decode_map_tile(b"not an image").is_err());
        assert!(decode_map_tile(&[]).is_err());
    }
}
