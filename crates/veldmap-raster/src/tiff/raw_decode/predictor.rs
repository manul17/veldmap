//! Pure TIFF endianness normalization and `Predictor = 2` (horizontal
//! differencing) / `Predictor = 3` (floating-point byte-lane) reversal.

use crate::pipeline::metadata::SampleType;

use super::super::ifd::{TiffByteOrder, TiffPredictor};

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

/// Normalize sample byte order and undo any TIFF horizontal/floating-point
/// predictor across every full stored row in `buf`.
pub(super) fn apply_endianness_and_predictor(
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
