//! Direct stateless raw-chunk GeoTIFF decompressor over [`ByteSource`].
//!
//! Bypasses `tiff::decoder::Decoder::read_chunk`, reusing already-locked codecs
//! (`flate2`, `weezl`, `zstd`, `jpeg-decoder`, `image-webp`) with explicit
//! pre-allocation bounds, TIFF endianness handling, `Predictor = 2` & `3`
//! (on full stored tile rows before edge cropping), `PlanarConfiguration = 2`
//! single-plane reads, and sparse chunk synthesis.

mod codecs;
mod predictor;

use veldmap_core::io::ByteSource;
use veldmap_core::sync::CancelToken;

use crate::RasterError;

use super::ifd::{CachedIfd, PlanarConfig, TiffByteOrder, TiffCompression};

/// Result of decoding one stored TIFF chunk (or one plane of a planar chunk).
pub(super) enum RawChunkPayload {
    /// Sparse chunk (`offset == 0 || byte_count == 0`): no bytes stored on disk.
    Sparse,
    /// Tightly packed native-endian raw sample bytes cropped to `(dw, dh)`.
    Bytes(Vec<u8>),
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
    let (stored_w, stored_h) = if ifd.tiled {
        (ifd.chunk_w, ifd.chunk_h)
    } else {
        (dw, dh)
    };
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

    // Cap compressed input read size relative to this chunk's `stored_bytes`
    // (never the global 256 MiB chunk cap) to prevent crafted multi-hundred-MiB
    // allocations on tiny tiles.
    let max_compressed = stored_bytes
        .saturating_mul(2)
        .max(4 * 1024 * 1024)
        .min(max_uncompressed_bytes.saturating_mul(2).max(64 * 1024))
        as u64;
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

    decode_compressed_chunk_bytes(
        compressed,
        ifd,
        byte_order,
        chunk_index,
        (stored_w, stored_h, dw, dh),
        samples_in_chunk,
        stored_bytes,
        max_uncompressed_bytes,
    )
}

#[expect(clippy::too_many_arguments, reason = "internal chunk decode pipeline")]
fn decode_compressed_chunk_bytes(
    compressed: Vec<u8>,
    ifd: &CachedIfd,
    byte_order: TiffByteOrder,
    chunk_index: u32,
    dims: (u32, u32, u32, u32),
    samples_in_chunk: usize,
    stored_bytes: usize,
    max_uncompressed_bytes: usize,
) -> Result<(u32, u32, RawChunkPayload), RasterError> {
    let (stored_w, _stored_h, dw, dh) = dims;
    let sample_bytes = ifd.sample_type.byte_size();

    let (mut raw, actual_row_w) = match ifd.compression {
        TiffCompression::None => (compressed, stored_w),
        TiffCompression::Deflate => (
            codecs::decompress_deflate(&compressed, stored_bytes)?,
            stored_w,
        ),
        TiffCompression::Lzw => (codecs::decompress_lzw(&compressed, stored_bytes)?, stored_w),
        TiffCompression::PackBits => (
            codecs::decompress_packbits(&compressed, stored_bytes)?,
            stored_w,
        ),
        TiffCompression::Zstd => (
            codecs::decompress_zstd(&compressed, stored_bytes)?,
            stored_w,
        ),
        TiffCompression::Jpeg => codecs::decompress_jpeg(
            &compressed,
            ifd.jpeg_tables.as_deref(),
            samples_in_chunk,
            stored_bytes,
            chunk_index,
            dims,
        )?,
        TiffCompression::WebP => codecs::decompress_webp(
            &compressed,
            samples_in_chunk,
            max_uncompressed_bytes,
            chunk_index,
            dims,
        )?,
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
    raw.truncate(rows_to_process * actual_row_bytes);

    // JPEG & WebP always decode directly into native-endian 8-bit bytes without predictors.
    if !matches!(
        ifd.compression,
        TiffCompression::Jpeg | TiffCompression::WebP
    ) {
        predictor::apply_endianness_and_predictor(
            &mut raw,
            actual_row_bytes,
            samples_in_chunk,
            ifd.sample_type,
            byte_order,
            ifd.predictor,
        );
    }

    let cropped = crop_right_edge(
        raw,
        actual_row_w,
        actual_row_bytes,
        dw,
        rows_to_process,
        samples_in_chunk * sample_bytes,
    );
    Ok((dw, dh, RawChunkPayload::Bytes(cropped)))
}

fn crop_right_edge(
    raw: Vec<u8>,
    actual_row_w: u32,
    actual_row_bytes: usize,
    dw: u32,
    rows: usize,
    pixel_bytes: usize,
) -> Vec<u8> {
    if dw >= actual_row_w {
        return raw;
    }
    let data_row_bytes = (dw as usize) * pixel_bytes;
    let mut out = Vec::with_capacity(data_row_bytes * rows);
    for row in raw.chunks_exact(actual_row_bytes) {
        out.extend_from_slice(&row[..data_row_bytes]);
    }
    out
}
