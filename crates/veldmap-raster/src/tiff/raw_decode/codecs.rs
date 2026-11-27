//! Bounded per-chunk TIFF codec decompressors (Deflate, LZW, PackBits, ZSTD,
//! JPEG with `JPEGTables`, and WebP).

use std::borrow::Cow;
use std::io::{Cursor, Read};

use crate::RasterError;
use crate::image_codec::{build_jpeg_decoder, rgb_to_rgba8, rgba_to_rgb8};

use super::super::ifd::ChunkDims;

/// Initial output buffer pre-allocation cap (`256 KiB`) for streaming TIFF
/// chunk decompressors.
const INITIAL_CHUNK_CAP: usize = 256 * 1024;

fn read_bounded(
    reader: impl Read,
    expected_max: usize,
    codec: &str,
) -> Result<Vec<u8>, RasterError> {
    let mut out = Vec::with_capacity(expected_max.min(INITIAL_CHUNK_CAP));
    reader
        .take(expected_max as u64 + 1)
        .read_to_end(&mut out)
        .map_err(|e| RasterError::Decode(format!("TIFF {codec} decode failed: {e}")))?;
    if out.len() > expected_max {
        out.truncate(expected_max);
    }
    Ok(out)
}

pub(super) fn decompress_deflate(
    compressed: &[u8],
    expected_max: usize,
) -> Result<Vec<u8>, RasterError> {
    read_bounded(
        flate2::read::ZlibDecoder::new(compressed),
        expected_max,
        "Deflate",
    )
}

pub(super) fn decompress_lzw(
    compressed: &[u8],
    expected_max: usize,
) -> Result<Vec<u8>, RasterError> {
    let mut decoder = weezl::decode::Decoder::with_tiff_size_switch(weezl::BitOrder::Msb, 8);
    let mut out = Vec::with_capacity(expected_max.min(INITIAL_CHUNK_CAP));
    let mut input = compressed;
    let mut scratch = [0u8; 16 * 1024];
    loop {
        let res = decoder.decode_bytes(input, &mut scratch);
        if res.consumed_out > 0 {
            let remaining = expected_max.saturating_sub(out.len());
            let take = res.consumed_out.min(remaining);
            out.extend_from_slice(&scratch[..take]);
        }
        input = &input[res.consumed_in..];
        match res.status {
            Ok(weezl::LzwStatus::Done) => break,
            Ok(weezl::LzwStatus::Ok) => {
                if out.len() >= expected_max {
                    break;
                }
                if res.consumed_in == 0 && res.consumed_out == 0 {
                    break;
                }
            }
            Ok(weezl::LzwStatus::NoProgress) => break,
            Err(err) => {
                // Some TIFF encoders omit the LZW EOI code once all chunk bytes are emitted.
                if out.len() >= expected_max {
                    break;
                }
                return Err(RasterError::Decode(format!(
                    "TIFF LZW decode failed: {err}"
                )));
            }
        }
    }
    Ok(out)
}

pub(super) fn decompress_packbits(
    compressed: &[u8],
    expected_max: usize,
) -> Result<Vec<u8>, RasterError> {
    let mut out = Vec::with_capacity(expected_max.min(INITIAL_CHUNK_CAP));
    let mut i = 0usize;
    while i < compressed.len() && out.len() < expected_max {
        let header = compressed[i] as i8;
        i += 1;
        if header >= 0 {
            let count = usize::from(header as u8) + 1;
            let end = i.saturating_add(count);
            if end > compressed.len() {
                return Err(RasterError::Decode(
                    "truncated TIFF PackBits literal run".to_owned(),
                ));
            }
            let take = count.min(expected_max - out.len());
            out.extend_from_slice(&compressed[i..i + take]);
            i = end;
        } else if header != -128 {
            let count = usize::from(header.unsigned_abs()) + 1;
            let Some(&val) = compressed.get(i) else {
                return Err(RasterError::Decode(
                    "truncated TIFF PackBits repeat run".to_owned(),
                ));
            };
            i += 1;
            let take = count.min(expected_max - out.len());
            out.resize(out.len() + take, val);
        }
    }
    Ok(out)
}

pub(super) fn decompress_zstd(
    compressed: &[u8],
    expected_max: usize,
) -> Result<Vec<u8>, RasterError> {
    let decoder = zstd::stream::read::Decoder::new(Cursor::new(compressed))
        .map_err(|e| RasterError::Decode(format!("TIFF ZSTD init failed: {e}")))?;
    read_bounded(decoder, expected_max, "ZSTD")
}

fn validate_decoded_chunk_dims(
    codec: &str,
    chunk_index: u32,
    w: u32,
    h: u32,
    expected: ChunkDims,
) -> Result<(), RasterError> {
    let valid_w = w == expected.stored_w || w == expected.data_w;
    let valid_h = h == expected.stored_h || h == expected.data_h;
    if valid_w && valid_h && w >= expected.data_w && h >= expected.data_h {
        Ok(())
    } else {
        Err(RasterError::Decode(format!(
            "TIFF {codec} chunk {chunk_index} decoded {w}x{h}, expected {}x{} or {}x{}",
            expected.stored_w, expected.stored_h, expected.data_w, expected.data_h
        )))
    }
}

fn splice_jpeg_tables<'a>(
    compressed: &'a [u8],
    jpeg_tables: Option<&[u8]>,
) -> Result<Cow<'a, [u8]>, RasterError> {
    let Some(tables) = jpeg_tables else {
        return Ok(Cow::Borrowed(compressed));
    };
    if compressed.len() < 2 || tables.len() < 2 {
        return Err(RasterError::Decode(
            "truncated TIFF JPEG stream or JPEGTables".to_owned(),
        ));
    }
    let table_prefix = tables.strip_suffix(&[0xFF, 0xD9]).unwrap_or(tables);
    let chunk_suffix = compressed.strip_prefix(&[0xFF, 0xD8]).unwrap_or(compressed);
    let mut buf = Vec::with_capacity(table_prefix.len() + chunk_suffix.len());
    buf.extend_from_slice(table_prefix);
    buf.extend_from_slice(chunk_suffix);
    Ok(Cow::Owned(buf))
}

fn validate_jpeg_header<R: Read>(
    decoder: &mut jpeg_decoder::Decoder<R>,
    expected_samples: usize,
    chunk_index: u32,
    expected_dims: ChunkDims,
) -> Result<u32, RasterError> {
    decoder
        .read_info()
        .map_err(|e| RasterError::Decode(format!("TIFF JPEG header decode failed: {e}")))?;
    let info = decoder
        .info()
        .ok_or_else(|| RasterError::Decode("TIFF JPEG decoder missing image info".to_owned()))?;
    let decoded_channels = match info.pixel_format {
        jpeg_decoder::PixelFormat::L8 => 1,
        jpeg_decoder::PixelFormat::RGB24 => 3,
        jpeg_decoder::PixelFormat::CMYK32 => 4,
        jpeg_decoder::PixelFormat::L16 => {
            return Err(RasterError::Decode(
                "16-bit lossless JPEG inside TIFF is not supported".to_owned(),
            ));
        }
    };
    if decoded_channels != expected_samples {
        return Err(RasterError::Decode(format!(
            "TIFF JPEG decoded {decoded_channels} channels, expected {expected_samples}"
        )));
    }
    let w = u32::from(info.width);
    let h = u32::from(info.height);
    validate_decoded_chunk_dims("JPEG", chunk_index, w, h, expected_dims)?;
    Ok(w)
}

pub(super) fn decompress_jpeg(
    compressed: &[u8],
    jpeg_tables: Option<&[u8]>,
    expected_samples: usize,
    max_decoded_bytes: usize,
    chunk_index: u32,
    expected_dims: ChunkDims,
) -> Result<(Vec<u8>, u32), RasterError> {
    let combined = splice_jpeg_tables(compressed, jpeg_tables)?;
    let mut decoder = build_jpeg_decoder(Cursor::new(combined.as_ref()), max_decoded_bytes as u64);

    // Work around `jpeg-decoder 0.3.2` quirks:
    // - Do NOT use `ColorTransform::None` (it outputs row-planar bytes and panics on
    //   subsampled 4:2:0/4:2:2 JPEGs).
    // - `ColorTransform::RGB` for 3-channel streams zips upsampled planes verbatim
    //   with ZERO color conversion — giving exact raw `[R, G, B]` for RGB TIFFs AND
    //   exact upsampled `[Y, Cb, Cr]` for YCbCr TIFFs (so our `YCbCrParams` applies
    //   tags 529 & 532 cleanly).
    // - `ColorTransform::CMYK` for 4-channel streams zips upsampled planes with
    //   `255 - c`; bitwise inverting the output recovers exact raw 4-band samples.
    let invert_cmyk = match expected_samples {
        1 => {
            decoder.set_color_transform(jpeg_decoder::ColorTransform::Grayscale);
            false
        }
        3 => {
            decoder.set_color_transform(jpeg_decoder::ColorTransform::RGB);
            false
        }
        4 => {
            decoder.set_color_transform(jpeg_decoder::ColorTransform::CMYK);
            true
        }
        other => {
            return Err(RasterError::Decode(format!(
                "unsupported TIFF JPEG sample count {other}"
            )));
        }
    };

    // Validate dimensions and channel count BEFORE allocating/decoding pixels.
    let w = validate_jpeg_header(&mut decoder, expected_samples, chunk_index, expected_dims)?;

    let mut pixels = decoder
        .decode()
        .map_err(|e| RasterError::Decode(format!("TIFF JPEG decode failed: {e}")))?;
    if invert_cmyk {
        for b in &mut pixels {
            *b = !*b;
        }
    }
    Ok((pixels, w))
}

pub(super) fn decompress_webp(
    compressed: &[u8],
    expected_samples: usize,
    max_decoded_bytes: usize,
    chunk_index: u32,
    expected_dims: ChunkDims,
) -> Result<(Vec<u8>, u32), RasterError> {
    let mut decoder = image_webp::WebPDecoder::new(Cursor::new(compressed))
        .map_err(|e| RasterError::Decode(format!("TIFF WebP init failed: {e}")))?;
    let (w, h) = decoder.dimensions();
    validate_decoded_chunk_dims("WebP", chunk_index, w, h, expected_dims)?;

    let webp_channels = if decoder.has_alpha() { 4usize } else { 3usize };
    let webp_bytes = (w as usize)
        .checked_mul(h as usize)
        .and_then(|px| px.checked_mul(webp_channels))
        .ok_or_else(|| RasterError::Decode("TIFF WebP dimensions overflow".to_owned()))?;
    if webp_bytes > max_decoded_bytes {
        return Err(RasterError::LimitExceeded {
            width: w,
            height: h,
            estimated_bytes: webp_bytes as u64,
            limit_bytes: max_decoded_bytes as u64,
        });
    }
    let mut buf = vec![0u8; webp_bytes];
    decoder
        .read_image(&mut buf)
        .map_err(|e| RasterError::Decode(format!("TIFF WebP decode failed: {e}")))?;
    if webp_channels == expected_samples {
        return Ok((buf, w));
    }
    // Adapt RGB <-> RGBA if WebP alpha presence differs from TIFF SamplesPerPixel.
    let pixels = (w as usize) * (h as usize);
    if webp_channels == 4 && expected_samples == 3 {
        Ok((rgba_to_rgb8(&buf, pixels), w))
    } else if webp_channels == 3 && expected_samples == 4 {
        Ok((rgb_to_rgba8(&buf, pixels), w))
    } else {
        Err(RasterError::Decode(format!(
            "TIFF WebP decoded {webp_channels} channels, expected {expected_samples}"
        )))
    }
}
