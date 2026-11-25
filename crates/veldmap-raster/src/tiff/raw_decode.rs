//! Direct stateless raw-chunk GeoTIFF decompressor over [`ByteSource`].
//!
//! Bypasses `tiff::decoder::Decoder::read_chunk`, reusing already-locked codecs
//! (`flate2`, `weezl`, `zstd`, `jpeg-decoder`, `image-webp`) with explicit
//! pre-allocation bounds, TIFF endianness handling, `Predictor = 2` & `3`
//! (on full stored tile rows before edge cropping), `PlanarConfiguration = 2`
//! single-plane reads, sparse chunk synthesis, and `WhiteIsZero` inversion.

use std::io::{Cursor, Read};

use veldmap_core::io::ByteSource;
use veldmap_core::sync::CancelToken;

use crate::RasterError;
use crate::pipeline::metadata::SampleType;

use super::ifd::{
    CachedIfd, Photometric, PlanarConfig, TiffByteOrder, TiffCompression, TiffPredictor,
};

/// Result of decoding one stored TIFF chunk (or one plane of a planar chunk).
pub(super) enum RawChunkPayload {
    /// Sparse chunk (`offset == 0 || byte_count == 0`): no bytes stored on disk.
    Sparse,
    /// Tightly packed native-endian raw sample bytes cropped to `(dw, dh)`.
    Bytes {
        data: Vec<u8>,
        /// True when `jpeg-decoder` already converted YCbCr -> RGB8.
        jpeg_already_rgb: bool,
    },
}

/// Decode one TIFF chunk `(ifd, plane, chunk_index)` directly from `byte_source`.
///
/// For `PlanarConfig::Planar`, `plane` selects the exact band plane (`0..spp`)
/// and returns 1 sample/pixel. For `PlanarConfig::Chunky`, `plane` is ignored
/// and all `samples_per_pixel` interleaved samples are returned.
pub(super) fn decode_raw_chunk(
    byte_source: &dyn ByteSource,
    ifd: &CachedIfd,
    byte_order: TiffByteOrder,
    plane: usize,
    chunk_index: u32,
    max_uncompressed_bytes: usize,
    cancel: &CancelToken,
) -> Result<(u32, u32, RawChunkPayload), RasterError> {
    if cancel.is_cancelled() {
        return Err(RasterError::Cancelled);
    }
    let (dw, dh) = ifd.chunk_data_dims(chunk_index).ok_or_else(|| {
        RasterError::Decode(format!(
            "chunk index {chunk_index} out of range for IFD {}",
            ifd.image
        ))
    })?;
    let (stored_w, stored_h) = ifd.chunk_stored_dims(chunk_index).ok_or_else(|| {
        RasterError::Decode(format!(
            "chunk index {chunk_index} out of range for IFD {}",
            ifd.image
        ))
    })?;
    let (offset, byte_count) = ifd.chunk_span(plane, chunk_index)?;
    if offset == 0 || byte_count == 0 {
        return Ok((dw, dh, RawChunkPayload::Sparse));
    }

    let samples_in_chunk: usize = match ifd.planar {
        PlanarConfig::Chunky => usize::from(ifd.samples_per_pixel),
        PlanarConfig::Planar => 1,
    };
    let sample_bytes = ifd.sample_type.byte_size();
    let stored_row_bytes = (stored_w as usize)
        .checked_mul(samples_in_chunk)
        .and_then(|n| n.checked_mul(sample_bytes))
        .ok_or_else(|| RasterError::Decode("TIFF chunk row byte size overflow".to_owned()))?;
    let stored_bytes = stored_row_bytes
        .checked_mul(stored_h as usize)
        .ok_or_else(|| RasterError::Decode("TIFF chunk total byte size overflow".to_owned()))?;
    if stored_bytes > max_uncompressed_bytes {
        return Err(RasterError::LimitExceeded {
            width: stored_w,
            height: stored_h,
            estimated_bytes: stored_bytes as u64,
            limit_bytes: max_uncompressed_bytes as u64,
        });
    }

    // Cap compressed input read size generously relative to decoded size (while
    // still guarding against corrupt multi-gigabyte byte_count tags).
    let max_compressed = max_uncompressed_bytes
        .saturating_mul(2)
        .max(4 * 1024 * 1024) as u64;
    if byte_count > max_compressed {
        return Err(RasterError::LimitExceeded {
            width: stored_w,
            height: stored_h,
            estimated_bytes: byte_count,
            limit_bytes: max_compressed,
        });
    }
    let file_len = byte_source.len();
    if offset
        .checked_add(byte_count)
        .is_none_or(|end| end > file_len)
    {
        return Err(RasterError::Decode(format!(
            "TIFF chunk span {offset}..+{} exceeds file length {file_len}",
            byte_count
        )));
    }

    let mut compressed = vec![0u8; byte_count as usize];
    byte_source.read_exact_at(offset, &mut compressed, cancel)?;
    if cancel.is_cancelled() {
        return Err(RasterError::Cancelled);
    }

    // Decompress into stored chunk buffer.
    let (mut raw, actual_row_w, jpeg_already_rgb) = match ifd.compression {
        TiffCompression::None => (compressed, stored_w, false),
        TiffCompression::Deflate => (
            decompress_deflate(&compressed, stored_bytes)?,
            stored_w,
            false,
        ),
        TiffCompression::Lzw => (decompress_lzw(&compressed, stored_bytes)?, stored_w, false),
        TiffCompression::PackBits => (
            decompress_packbits(&compressed, stored_bytes)?,
            stored_w,
            false,
        ),
        TiffCompression::Zstd => (decompress_zstd(&compressed, stored_bytes)?, stored_w, false),
        TiffCompression::Jpeg => {
            let (decoded, w, h, is_rgb24) = decompress_jpeg(
                &compressed,
                ifd.jpeg_tables.as_deref(),
                samples_in_chunk,
                max_uncompressed_bytes,
            )?;
            validate_decoded_chunk_dims("JPEG", chunk_index, w, h, stored_w, stored_h, dw, dh)?;
            (decoded, w, is_rgb24)
        }
        TiffCompression::WebP => {
            let (decoded, w, h) =
                decompress_webp(&compressed, samples_in_chunk, max_uncompressed_bytes)?;
            validate_decoded_chunk_dims("WebP", chunk_index, w, h, stored_w, stored_h, dw, dh)?;
            (decoded, w, false)
        }
    };

    let actual_row_bytes = (actual_row_w as usize)
        .checked_mul(samples_in_chunk)
        .and_then(|v| v.checked_mul(sample_bytes))
        .ok_or_else(|| RasterError::Decode("TIFF chunk row byte overflow".to_owned()))?;

    // Some writers emit only `dh` rows for bottom-edge tiles (especially under
    // PlanarConfiguration = 2 or stripped edge chunks). Require at least `dh`
    // full decoded rows.
    let min_rows_bytes = actual_row_bytes
        .checked_mul(dh as usize)
        .ok_or_else(|| RasterError::Decode("TIFF chunk height byte overflow".to_owned()))?;
    if raw.len() < min_rows_bytes {
        return Err(RasterError::Decode(format!(
            "decompressed TIFF chunk {} has {} bytes, expected at least {min_rows_bytes} ({actual_row_w}x{dh})",
            chunk_index,
            raw.len()
        )));
    }

    // Apply endianness fix and predictor reversal on every FULL stored row
    // before cropping tiled edge padding.
    let rows_to_process = dh as usize;
    let process_len = rows_to_process * actual_row_bytes;
    raw.truncate(process_len);

    // JPEG & WebP always decode directly into native-endian 8-bit bytes without predictors.
    if !matches!(
        ifd.compression,
        TiffCompression::Jpeg | TiffCompression::WebP
    ) {
        apply_endianness_and_predictor(
            &mut raw,
            actual_row_bytes,
            samples_in_chunk,
            ifd.sample_type,
            byte_order,
            ifd.predictor,
        );
    }

    // Crop right-edge padding on tiled chunks (`dw < actual_row_w`).
    let data_row_bytes = (dw as usize) * samples_in_chunk * sample_bytes;
    let mut cropped = if dw < actual_row_w {
        let mut out = Vec::with_capacity(data_row_bytes * rows_to_process);
        for row in raw.chunks_exact(actual_row_bytes) {
            out.extend_from_slice(&row[..data_row_bytes]);
        }
        out
    } else {
        raw
    };

    if ifd.photometric == Photometric::WhiteIsZero {
        invert_white_is_zero(&mut cropped, ifd.sample_type);
    }

    Ok((
        dw,
        dh,
        RawChunkPayload::Bytes {
            data: cropped,
            jpeg_already_rgb,
        },
    ))
}

fn decompress_deflate(compressed: &[u8], expected_max: usize) -> Result<Vec<u8>, RasterError> {
    let mut decoder = flate2::read::ZlibDecoder::new(compressed).take(expected_max as u64 + 1);
    let mut out = Vec::with_capacity(expected_max.min(256 * 1024));
    decoder
        .read_to_end(&mut out)
        .map_err(|e| RasterError::Decode(format!("TIFF Deflate decode failed: {e}")))?;
    if out.len() > expected_max {
        out.truncate(expected_max);
    }
    Ok(out)
}

fn decompress_lzw(compressed: &[u8], expected_max: usize) -> Result<Vec<u8>, RasterError> {
    let mut decoder = weezl::decode::Decoder::with_tiff_size_switch(weezl::BitOrder::Msb, 8);
    let mut out = Vec::with_capacity(expected_max.min(256 * 1024));
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

fn decompress_packbits(compressed: &[u8], expected_max: usize) -> Result<Vec<u8>, RasterError> {
    let mut out = Vec::with_capacity(expected_max.min(256 * 1024));
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

fn decompress_zstd(compressed: &[u8], expected_max: usize) -> Result<Vec<u8>, RasterError> {
    let decoder = zstd::stream::read::Decoder::new(Cursor::new(compressed))
        .map_err(|e| RasterError::Decode(format!("TIFF ZSTD init failed: {e}")))?;
    let mut out = Vec::with_capacity(expected_max.min(256 * 1024));
    decoder
        .take(expected_max as u64 + 1)
        .read_to_end(&mut out)
        .map_err(|e| RasterError::Decode(format!("TIFF ZSTD decode failed: {e}")))?;
    if out.len() > expected_max {
        out.truncate(expected_max);
    }
    Ok(out)
}

#[expect(clippy::too_many_arguments, reason = "chunk dimension validator")]
fn validate_decoded_chunk_dims(
    codec: &str,
    chunk_index: u32,
    w: u32,
    h: u32,
    stored_w: u32,
    stored_h: u32,
    dw: u32,
    dh: u32,
) -> Result<(), RasterError> {
    let valid_w = w == stored_w || w == dw;
    let valid_h = h == stored_h || h == dh;
    if valid_w && valid_h && w >= dw && h >= dh {
        Ok(())
    } else {
        Err(RasterError::Decode(format!(
            "TIFF {codec} chunk {chunk_index} decoded {w}x{h}, expected {stored_w}x{stored_h} or {dw}x{dh}"
        )))
    }
}

fn decompress_jpeg(
    compressed: &[u8],
    jpeg_tables: Option<&[u8]>,
    expected_samples: usize,
    max_decoded_bytes: usize,
) -> Result<(Vec<u8>, u32, u32, bool), RasterError> {
    let combined: Vec<u8> = if let Some(tables) = jpeg_tables {
        if compressed.len() < 2 || tables.len() < 2 {
            return Err(RasterError::Decode(
                "truncated TIFF JPEG stream or JPEGTables".to_owned(),
            ));
        }
        let table_prefix = if tables.ends_with(&[0xFF, 0xD9]) {
            &tables[..tables.len() - 2]
        } else {
            tables
        };
        let chunk_suffix = if compressed.starts_with(&[0xFF, 0xD8]) {
            &compressed[2..]
        } else {
            compressed
        };
        let mut buf = Vec::with_capacity(table_prefix.len() + chunk_suffix.len());
        buf.extend_from_slice(table_prefix);
        buf.extend_from_slice(chunk_suffix);
        buf
    } else {
        compressed.to_vec()
    };

    let mut decoder = jpeg_decoder::Decoder::new(Cursor::new(combined));
    decoder.set_max_decoding_buffer_size(max_decoded_bytes);
    let pixels = decoder
        .decode()
        .map_err(|e| RasterError::Decode(format!("TIFF JPEG decode failed: {e}")))?;
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
    let is_rgb24 = info.pixel_format == jpeg_decoder::PixelFormat::RGB24;
    Ok((
        pixels,
        u32::from(info.width),
        u32::from(info.height),
        is_rgb24,
    ))
}

fn decompress_webp(
    compressed: &[u8],
    expected_samples: usize,
    max_decoded_bytes: usize,
) -> Result<(Vec<u8>, u32, u32), RasterError> {
    let mut decoder = image_webp::WebPDecoder::new(Cursor::new(compressed))
        .map_err(|e| RasterError::Decode(format!("TIFF WebP init failed: {e}")))?;
    let (w, h) = decoder.dimensions();
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
        return Ok((buf, w, h));
    }
    // Adapt RGB <-> RGBA if WebP alpha presence differs from TIFF SamplesPerPixel.
    let pixels = (w as usize) * (h as usize);
    if webp_channels == 4 && expected_samples == 3 {
        let mut rgb = Vec::with_capacity(pixels * 3);
        for px in buf.chunks_exact(4) {
            rgb.extend_from_slice(&px[..3]);
        }
        Ok((rgb, w, h))
    } else if webp_channels == 3 && expected_samples == 4 {
        let mut rgba = Vec::with_capacity(pixels * 4);
        for px in buf.chunks_exact(3) {
            rgba.extend_from_slice(px);
            rgba.push(255);
        }
        Ok((rgba, w, h))
    } else {
        Err(RasterError::Decode(format!(
            "TIFF WebP decoded {webp_channels} channels, expected {expected_samples}"
        )))
    }
}

fn fix_endianness(buf: &mut [u8], byte_order: TiffByteOrder, sample_bytes: usize) {
    let host_is_le = cfg!(target_endian = "little");
    let file_is_le = byte_order == TiffByteOrder::Little;
    if host_is_le == file_is_le || sample_bytes <= 1 {
        return;
    }
    for chunk in buf.chunks_exact_mut(sample_bytes) {
        chunk.reverse();
    }
}

fn undo_horizontal_predictor_row(row: &mut [u8], samples_per_pixel: usize, sample_bytes: usize) {
    let stride = samples_per_pixel * sample_bytes;
    match sample_bytes {
        1 => {
            for i in stride..row.len() {
                row[i] = row[i].wrapping_add(row[i - stride]);
            }
        }
        2 => {
            for i in (stride..row.len()).step_by(2) {
                let prev = u16::from_ne_bytes([row[i - stride], row[i - stride + 1]]);
                let cur = u16::from_ne_bytes([row[i], row[i + 1]]);
                let sum = cur.wrapping_add(prev).to_ne_bytes();
                row[i..i + 2].copy_from_slice(&sum);
            }
        }
        4 => {
            for i in (stride..row.len()).step_by(4) {
                let prev = u32::from_ne_bytes([
                    row[i - stride],
                    row[i - stride + 1],
                    row[i - stride + 2],
                    row[i - stride + 3],
                ]);
                let cur = u32::from_ne_bytes([row[i], row[i + 1], row[i + 2], row[i + 3]]);
                let sum = cur.wrapping_add(prev).to_ne_bytes();
                row[i..i + 4].copy_from_slice(&sum);
            }
        }
        8 => {
            for i in (stride..row.len()).step_by(8) {
                let mut prev_b = [0u8; 8];
                let mut cur_b = [0u8; 8];
                prev_b.copy_from_slice(&row[i - stride..i - stride + 8]);
                cur_b.copy_from_slice(&row[i..i + 8]);
                let sum = u64::from_ne_bytes(cur_b)
                    .wrapping_add(u64::from_ne_bytes(prev_b))
                    .to_ne_bytes();
                row[i..i + 8].copy_from_slice(&sum);
            }
        }
        _ => {}
    }
}

fn undo_floating_point_predictor_row(
    row: &mut [u8],
    scratch: &mut Vec<u8>,
    samples_per_pixel: usize,
    sample_bytes: usize,
) {
    scratch.clear();
    scratch.extend_from_slice(row);
    for i in samples_per_pixel..scratch.len() {
        scratch[i] = scratch[i].wrapping_add(scratch[i - samples_per_pixel]);
    }
    let count = scratch.len() / sample_bytes;
    match sample_bytes {
        2 => {
            for (i, dst) in row.chunks_exact_mut(2).enumerate() {
                let val = u16::from_be_bytes([scratch[i], scratch[count + i]]);
                dst.copy_from_slice(&val.to_ne_bytes());
            }
        }
        4 => {
            for (i, dst) in row.chunks_exact_mut(4).enumerate() {
                let val = u32::from_be_bytes([
                    scratch[i],
                    scratch[count + i],
                    scratch[count * 2 + i],
                    scratch[count * 3 + i],
                ]);
                dst.copy_from_slice(&val.to_ne_bytes());
            }
        }
        8 => {
            for (i, dst) in row.chunks_exact_mut(8).enumerate() {
                let val = u64::from_be_bytes([
                    scratch[i],
                    scratch[count + i],
                    scratch[count * 2 + i],
                    scratch[count * 3 + i],
                    scratch[count * 4 + i],
                    scratch[count * 5 + i],
                    scratch[count * 6 + i],
                    scratch[count * 7 + i],
                ]);
                dst.copy_from_slice(&val.to_ne_bytes());
            }
        }
        _ => {}
    }
}

fn apply_endianness_and_predictor(
    buf: &mut [u8],
    row_bytes: usize,
    samples_per_pixel: usize,
    sample_type: SampleType,
    byte_order: TiffByteOrder,
    predictor: TiffPredictor,
) {
    let sample_bytes = sample_type.byte_size();
    match predictor {
        TiffPredictor::None => {
            fix_endianness(buf, byte_order, sample_bytes);
        }
        TiffPredictor::Horizontal => {
            fix_endianness(buf, byte_order, sample_bytes);
            for row in buf.chunks_exact_mut(row_bytes) {
                undo_horizontal_predictor_row(row, samples_per_pixel, sample_bytes);
            }
        }
        TiffPredictor::FloatingPoint => {
            let mut scratch = Vec::with_capacity(row_bytes);
            for row in buf.chunks_exact_mut(row_bytes) {
                undo_floating_point_predictor_row(
                    row,
                    &mut scratch,
                    samples_per_pixel,
                    sample_bytes,
                );
            }
        }
    }
}

fn invert_white_is_zero(buf: &mut [u8], sample_type: SampleType) {
    match sample_type {
        SampleType::U8 => {
            for b in buf.iter_mut() {
                *b = !*b;
            }
        }
        SampleType::U16 => {
            for chunk in buf.chunks_exact_mut(2) {
                let v = u16::from_ne_bytes([chunk[0], chunk[1]]);
                chunk.copy_from_slice(&(u16::MAX - v).to_ne_bytes());
            }
        }
        SampleType::U32 => {
            for chunk in buf.chunks_exact_mut(4) {
                let v = u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                chunk.copy_from_slice(&(u32::MAX - v).to_ne_bytes());
            }
        }
        SampleType::F32 => {
            for chunk in buf.chunks_exact_mut(4) {
                let v = f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                chunk.copy_from_slice(&(1.0 - v).to_ne_bytes());
            }
        }
        SampleType::F64 => {
            for chunk in buf.chunks_exact_mut(8) {
                let mut arr = [0u8; 8];
                arr.copy_from_slice(chunk);
                let v = f64::from_ne_bytes(arr);
                chunk.copy_from_slice(&(1.0 - v).to_ne_bytes());
            }
        }
        SampleType::I8 | SampleType::I16 | SampleType::I32 | SampleType::F16 => {}
    }
}
