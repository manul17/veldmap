//! Memory-bounded PNG/JPEG map-tile decoding and equirectangular image
//! decoding built directly on `png` and `jpeg-decoder` with explicit
//! dimension and peak-allocation caps.

use std::io::Cursor;

use veldmap_core::tile::{DecodedTile, PixelFormat};

use crate::RasterError;
use crate::image_codec::{
    build_normalized_png_decoder, decode_jpeg_pixels_to_rgb8, decode_jpeg_pixels_to_rgba8,
    has_jpeg_magic, has_png_magic, jpeg_codec_working_bytes, jpeg_pixel_format_bpp,
    jpeg_pixel_format_channels, png_u8_to_rgb8, png_u8_to_rgba8,
};

use super::mipmap::generate_linear_mipmaps;

/// Maximum single map-tile dimension accepted by [`decode_map_tile`].
const MAX_MAP_TILE_DIM: u32 = 2048;

/// Maximum peak allocation budget for decoding a single map tile (`36 MiB`,
/// covering a `2048 x 2048` RGB/PNG decode plus RGBA8 conversion).
const MAX_MAP_TILE_ALLOC_BYTES: u64 = 36 * 1024 * 1024;

#[derive(Clone, Copy, PartialEq, Eq)]
enum TargetFormat {
    Rgba8,
    Rgb8,
}

impl TargetFormat {
    const fn bytes_per_pixel(self) -> u32 {
        match self {
            Self::Rgba8 => 4,
            Self::Rgb8 => 3,
        }
    }
}

/// Decode raw PNG or JPEG map-tile bytes into tightly packed `Rgba8UnormSrgb`
/// pixels under explicit dimension and peak-allocation limits.
fn decode_map_tile(bytes: &[u8]) -> Result<DecodedTile, RasterError> {
    decode_rgba8_limited(bytes, MAX_MAP_TILE_DIM, MAX_MAP_TILE_ALLOC_BYTES)
}

/// Decode raw PNG or JPEG map-tile bytes into tightly packed `Rgba8UnormSrgb`
/// pixels and populate a complete CPU linear-light mipmap chain.
pub fn decode_map_tile_with_mipmaps(bytes: &[u8]) -> Result<DecodedTile, RasterError> {
    let mut tile = decode_map_tile(bytes)?;
    generate_linear_mipmaps(&mut tile);
    Ok(tile)
}

#[derive(Clone, Copy)]
struct DecodeLimits {
    max_dim: u32,
    max_alloc_bytes: u64,
}

fn checked_peak_bytes(
    width: u32,
    height: u32,
    src_bpp: u32,
    src_matches_target: bool,
    target: TargetFormat,
    codec_working_bytes: u64,
    limits: DecodeLimits,
) -> Result<(u64, u64), RasterError> {
    if width == 0 || height == 0 {
        return Err(RasterError::InvalidDimensions { width, height });
    }
    let px_count = u64::from(width)
        .checked_mul(u64::from(height))
        .ok_or(RasterError::InvalidDimensions { width, height })?;
    let src_bytes = px_count
        .checked_mul(u64::from(src_bpp))
        .ok_or(RasterError::InvalidDimensions { width, height })?;
    let dst_bytes = px_count
        .checked_mul(u64::from(target.bytes_per_pixel()))
        .ok_or(RasterError::InvalidDimensions { width, height })?;

    // Decoder is dropped immediately after producing the raw pixel buffer:
    // phase 1 holds `codec_working_bytes + src_bytes`, and phase 2 holds
    // `src_bytes + conversion_bytes` (0 when moving the buffer in place).
    let conversion_bytes = if src_matches_target { 0 } else { dst_bytes };
    let decode_peak = src_bytes.saturating_add(codec_working_bytes);
    let convert_peak = src_bytes.saturating_add(conversion_bytes);
    let estimated_peak = decode_peak.max(convert_peak);
    if width > limits.max_dim || height > limits.max_dim || estimated_peak > limits.max_alloc_bytes
    {
        return Err(RasterError::LimitExceeded {
            width,
            height,
            estimated_bytes: estimated_peak,
            limit_bytes: limits.max_alloc_bytes,
        });
    }
    Ok((conversion_bytes, estimated_peak))
}

fn map_png_err(
    err: png::DecodingError,
    width: u32,
    height: u32,
    estimated_bytes: u64,
    limit_bytes: u64,
) -> RasterError {
    match err {
        png::DecodingError::LimitsExceeded => RasterError::LimitExceeded {
            width,
            height,
            estimated_bytes,
            limit_bytes,
        },
        other => RasterError::Decode(other.to_string()),
    }
}

fn decode_png_limited(
    bytes: &[u8],
    max_dim: u32,
    max_alloc_bytes: u64,
    target: TargetFormat,
) -> Result<(u32, u32, Vec<u8>), RasterError> {
    let decoder = build_normalized_png_decoder(Cursor::new(bytes), max_alloc_bytes);

    let mut reader = decoder
        .read_info()
        .map_err(|err| map_png_err(err, 0, 0, 0, max_alloc_bytes))?;
    let (width, height) = {
        let info = reader.info();
        (info.width, info.height)
    };
    let (color_type, _) = reader.output_color_type();
    let src_bpp = color_type.samples() as u32;
    let src_matches_target = matches!(
        (color_type, target),
        (png::ColorType::Rgba, TargetFormat::Rgba8) | (png::ColorType::Rgb, TargetFormat::Rgb8)
    );
    let (conversion_bytes, estimated_peak) = checked_peak_bytes(
        width,
        height,
        src_bpp,
        src_matches_target,
        target,
        0,
        DecodeLimits {
            max_dim,
            max_alloc_bytes,
        },
    )?;

    let buf_len = reader
        .output_buffer_size()
        .ok_or(RasterError::InvalidDimensions { width, height })?;
    let frame_peak = (buf_len as u64).saturating_add(conversion_bytes);
    if frame_peak > max_alloc_bytes {
        return Err(RasterError::LimitExceeded {
            width,
            height,
            estimated_bytes: frame_peak,
            limit_bytes: max_alloc_bytes,
        });
    }

    let mut buf = vec![0u8; buf_len];
    let out = reader
        .next_frame(&mut buf)
        .map_err(|err| map_png_err(err, width, height, estimated_peak, max_alloc_bytes))?;
    drop(reader);
    buf.truncate(out.buffer_size());

    let px_count = (width as usize).saturating_mul(height as usize);
    let data = match target {
        TargetFormat::Rgba8 => png_u8_to_rgba8(out.color_type, buf, px_count),
        TargetFormat::Rgb8 => png_u8_to_rgb8(out.color_type, buf, px_count),
    };
    Ok((width, height, data))
}

fn decode_jpeg_limited(
    bytes: &[u8],
    max_dim: u32,
    max_alloc_bytes: u64,
    target: TargetFormat,
) -> Result<(u32, u32, Vec<u8>), RasterError> {
    let mut decoder = jpeg_decoder::Decoder::new(Cursor::new(bytes));
    decoder.set_max_decoding_buffer_size(max_alloc_bytes.min(usize::MAX as u64) as usize);
    decoder
        .read_info()
        .map_err(|err| RasterError::Decode(err.to_string()))?;
    let info = decoder
        .info()
        .ok_or_else(|| RasterError::Decode("JPEG header missing image info".to_string()))?;
    let width = u32::from(info.width);
    let height = u32::from(info.height);
    let pixel_format = info.pixel_format;
    let src_bpp = jpeg_pixel_format_bpp(pixel_format);
    let channels = jpeg_pixel_format_channels(pixel_format);
    let codec_working_bytes =
        jpeg_codec_working_bytes(width, height, channels, info.coding_process);
    let src_matches_target = matches!(
        (pixel_format, target),
        (jpeg_decoder::PixelFormat::RGB24, TargetFormat::Rgb8)
    );
    let (_conversion_bytes, _estimated_peak) = checked_peak_bytes(
        width,
        height,
        src_bpp,
        src_matches_target,
        target,
        codec_working_bytes,
        DecodeLimits {
            max_dim,
            max_alloc_bytes,
        },
    )?;

    decoder.set_max_decoding_buffer_size(
        (max_alloc_bytes.saturating_sub(codec_working_bytes)).min(usize::MAX as u64) as usize,
    );
    let pixels = decoder
        .decode()
        .map_err(|err| RasterError::Decode(err.to_string()))?;
    // Release progressive coefficient / decoder working buffers before
    // allocating the target conversion buffer.
    drop(decoder);
    let px_count = (width as usize).saturating_mul(height as usize);
    let data = match target {
        TargetFormat::Rgba8 => decode_jpeg_pixels_to_rgba8(pixel_format, &pixels, px_count),
        TargetFormat::Rgb8 => decode_jpeg_pixels_to_rgb8(pixel_format, pixels, px_count),
    };
    Ok((width, height, data))
}

fn decode_limited_for_target(
    bytes: &[u8],
    max_dim: u32,
    max_alloc_bytes: u64,
    target: TargetFormat,
) -> Result<(u32, u32, Vec<u8>), RasterError> {
    if has_png_magic(bytes) {
        decode_png_limited(bytes, max_dim, max_alloc_bytes, target)
    } else if has_jpeg_magic(bytes) {
        decode_jpeg_limited(bytes, max_dim, max_alloc_bytes, target)
    } else {
        Err(RasterError::UnsupportedFormat)
    }
}

/// Decode a PNG or JPEG image into tightly packed `Rgba8UnormSrgb` pixels
/// while bounding both decoder working set and RGBA8 output within
/// `max_alloc_bytes`.
fn decode_rgba8_limited(
    bytes: &[u8],
    max_dim: u32,
    max_alloc_bytes: u64,
) -> Result<DecodedTile, RasterError> {
    let (w, h, data) =
        decode_limited_for_target(bytes, max_dim, max_alloc_bytes, TargetFormat::Rgba8)?;
    DecodedTile::try_new(w, h, PixelFormat::Rgba8UnormSrgb, data).map_err(|_| {
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
    let (w, h, data) =
        decode_limited_for_target(bytes, max_dim, max_alloc_bytes, TargetFormat::Rgb8)?;
    let expected = (w as usize)
        .checked_mul(h as usize)
        .and_then(|px| px.checked_mul(3));
    if w == 0 || h == 0 || expected != Some(data.len()) {
        return Err(RasterError::InvalidDimensions {
            width: w,
            height: h,
        });
    }
    Ok((w, h, data.into_boxed_slice()))
}

#[cfg(test)]
#[path = "map_tile_tests.rs"]
mod tests;
