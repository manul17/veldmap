//! Single-pass Classic TIFF + BigTIFF IFD reader and typed tag decoders built
//! directly over positioned [`veldmap_core::io::ByteSource`].
//!
//! Replaces open-time `tiff::decoder::Decoder` so remote Cloud-Optimized
//! GeoTIFFs read the header prefix once, skip out-of-line tags for non-overview
//! IFDs, enforce `MemoryBudget` limits *before* allocating tag buffers, and
//! coalesce any out-of-prefix tag payloads into sorted reads.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use veldmap_core::io::ByteSource;
use veldmap_core::sync::CancelToken;

use crate::RasterError;

use super::georef::RawGeoTiffTags;
use super::ifd::TiffByteOrder;

/// Initial header prefix window (`256 KiB`, matching `HttpRangeSource`'s head
/// probe) read once on open so typical COG headers, GeoKeys, and tile offset
/// tables resolve in zero extra `read_at`s.
const INITIAL_HEADER_PREFIX_BYTES: usize = 256 * 1024;

/// Maximum number of tag entries admitted in a single IFD directory.
const MAX_IFD_ENTRIES: usize = 4096;

/// Coalescing thresholds for out-of-prefix IFD tag payloads.
const OUT_OF_LINE_COALESCE_GAP: u64 = 16 * 1024;
const OUT_OF_LINE_COALESCE_MAX_LEN: u64 = 2 * 1024 * 1024;

pub(super) const TAG_NEW_SUBFILE_TYPE: u16 = 254;
pub(super) const TAG_IMAGE_WIDTH: u16 = 256;
pub(super) const TAG_IMAGE_LENGTH: u16 = 257;
pub(super) const TAG_BITS_PER_SAMPLE: u16 = 258;
pub(super) const TAG_COMPRESSION: u16 = 259;
pub(super) const TAG_PHOTOMETRIC: u16 = 262;
pub(super) const TAG_STRIP_OFFSETS: u16 = 273;
pub(super) const TAG_SAMPLES_PER_PIXEL: u16 = 277;
pub(super) const TAG_ROWS_PER_STRIP: u16 = 278;
pub(super) const TAG_STRIP_BYTE_COUNTS: u16 = 279;
pub(super) const TAG_PLANAR_CONFIG: u16 = 284;
pub(super) const TAG_PREDICTOR: u16 = 317;
pub(super) const TAG_COLOR_MAP: u16 = 320;
pub(super) const TAG_TILE_WIDTH: u16 = 322;
pub(super) const TAG_TILE_LENGTH: u16 = 323;
pub(super) const TAG_TILE_OFFSETS: u16 = 324;
pub(super) const TAG_TILE_BYTE_COUNTS: u16 = 325;
pub(super) const TAG_EXTRA_SAMPLES: u16 = 338;
pub(super) const TAG_SAMPLE_FORMAT: u16 = 339;
pub(super) const TAG_JPEG_TABLES: u16 = 347;
pub(super) const TAG_YCBCR_SUBSAMPLING: u16 = 530;
pub(super) const TAG_MODEL_PIXEL_SCALE: u16 = 33550;
pub(super) const TAG_MODEL_TIEPOINT: u16 = 33922;
pub(super) const TAG_MODEL_TRANSFORMATION: u16 = 34264;
pub(super) const TAG_GEO_KEY_DIRECTORY: u16 = 34735;
pub(super) const TAG_GEO_DOUBLE_PARAMS: u16 = 34736;
pub(super) const TAG_GEO_ASCII_PARAMS: u16 = 34737;
pub(super) const TAG_GDAL_NODATA: u16 = 42113;

pub(super) fn is_overview(subfile_type: u32) -> bool {
    const REDUCED: u32 = 1;
    const MASK: u32 = 4;
    subfile_type & REDUCED != 0 && subfile_type & MASK == 0
}

const fn field_type_size(field_type: u16) -> Option<u64> {
    match field_type {
        1 | 2 | 6 | 7 => Some(1),              // BYTE, ASCII, SBYTE, UNDEFINED
        3 | 8 => Some(2),                      // SHORT, SSHORT
        4 | 9 | 11 | 13 => Some(4),            // LONG, SLONG, FLOAT, IFD
        5 | 10 | 12 | 16 | 17 | 18 => Some(8), // RATIONAL, SRATIONAL, DOUBLE, LONG8, SLONG8, IFD8
        _ => None,
    }
}

fn read_u16(order: TiffByteOrder, b: &[u8]) -> u16 {
    let arr = [b[0], b[1]];
    match order {
        TiffByteOrder::Little => u16::from_le_bytes(arr),
        TiffByteOrder::Big => u16::from_be_bytes(arr),
    }
}

fn read_i16(order: TiffByteOrder, b: &[u8]) -> i16 {
    let arr = [b[0], b[1]];
    match order {
        TiffByteOrder::Little => i16::from_le_bytes(arr),
        TiffByteOrder::Big => i16::from_be_bytes(arr),
    }
}

fn read_u32(order: TiffByteOrder, b: &[u8]) -> u32 {
    let arr = [b[0], b[1], b[2], b[3]];
    match order {
        TiffByteOrder::Little => u32::from_le_bytes(arr),
        TiffByteOrder::Big => u32::from_be_bytes(arr),
    }
}

fn read_i32(order: TiffByteOrder, b: &[u8]) -> i32 {
    let arr = [b[0], b[1], b[2], b[3]];
    match order {
        TiffByteOrder::Little => i32::from_le_bytes(arr),
        TiffByteOrder::Big => i32::from_be_bytes(arr),
    }
}

fn read_u64(order: TiffByteOrder, b: &[u8]) -> u64 {
    let arr = [b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]];
    match order {
        TiffByteOrder::Little => u64::from_le_bytes(arr),
        TiffByteOrder::Big => u64::from_be_bytes(arr),
    }
}

fn read_i64(order: TiffByteOrder, b: &[u8]) -> i64 {
    let arr = [b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]];
    match order {
        TiffByteOrder::Little => i64::from_le_bytes(arr),
        TiffByteOrder::Big => i64::from_be_bytes(arr),
    }
}

fn read_f32(order: TiffByteOrder, b: &[u8]) -> f32 {
    let arr = [b[0], b[1], b[2], b[3]];
    match order {
        TiffByteOrder::Little => f32::from_le_bytes(arr),
        TiffByteOrder::Big => f32::from_be_bytes(arr),
    }
}

fn read_f64(order: TiffByteOrder, b: &[u8]) -> f64 {
    let arr = [b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]];
    match order {
        TiffByteOrder::Little => f64::from_le_bytes(arr),
        TiffByteOrder::Big => f64::from_be_bytes(arr),
    }
}

/// One parsed Classic or BigTIFF IFD directory entry.
#[derive(Clone, Copy, Debug)]
pub(super) struct IfdEntry {
    pub(super) tag: u16,
    pub(super) field_type: u16,
    pub(super) count: u64,
    inline_bytes: [u8; 8],
    offset: u64,
    byte_len: u64,
    is_inline: bool,
}

impl IfdEntry {
    pub(super) const fn byte_len(&self) -> u64 {
        self.byte_len
    }
}

/// One parsed TIFF IFD directory with its inline entries and lazily fetched
/// coalesced out-of-line tag payloads.
pub(super) struct IfdDirectory {
    pub(super) byte_order: TiffByteOrder,
    pub(super) next_ifd: u64,
    entries: HashMap<u16, IfdEntry>,
    loaded_spans: Vec<(u64, Vec<u8>)>,
}

impl IfdDirectory {
    pub(super) fn entry(&self, tag: u16) -> Option<IfdEntry> {
        self.entries.get(&tag).copied()
    }

    /// Return the exact raw bytes for `entry` from inline storage, the reader's
    /// initial header prefix, or previously coalesced out-of-line buffers.
    fn resolve_bytes<'a>(
        &'a self,
        entry: &'a IfdEntry,
        reader: &'a SinglePassIfdReader,
    ) -> Result<&'a [u8], RasterError> {
        let len = usize::try_from(entry.byte_len).map_err(|_| {
            RasterError::DecoderLimitsExceeded(format!(
                "TIFF tag {} payload {} B overflows usize",
                entry.tag, entry.byte_len
            ))
        })?;
        if entry.is_inline {
            return Ok(&entry.inline_bytes[..len]);
        }
        let end = entry.offset.checked_add(entry.byte_len).ok_or_else(|| {
            RasterError::Decode(format!("TIFF tag {} offset overflow", entry.tag))
        })?;
        if end <= reader.prefix.len() as u64 {
            let s = entry.offset as usize;
            return Ok(&reader.prefix[s..s + len]);
        }
        for (span_off, data) in &self.loaded_spans {
            let span_end = span_off.saturating_add(data.len() as u64);
            if entry.offset >= *span_off && end <= span_end {
                let rel = (entry.offset - *span_off) as usize;
                return Ok(&data[rel..rel + len]);
            }
        }
        Err(RasterError::Decode(format!(
            "TIFF tag {} out-of-line span {}..+{len} was not prefetched",
            entry.tag, entry.offset
        )))
    }

    /// Coalesce and fetch all out-of-line payloads for `tags` that are not
    /// already covered by `reader.prefix`, enforcing `max_payload_bytes` and
    /// `reader`'s cumulative byte budget BEFORE allocation or I/O.
    pub(super) fn prefetch_tags(
        &mut self,
        reader: &mut SinglePassIfdReader,
        tags: &[u16],
        max_payload_bytes: u64,
        cancel: &CancelToken,
    ) -> Result<(), RasterError> {
        let mut needed: Vec<(u64, u64)> = Vec::new();
        let mut total_bytes = 0u64;
        for &tag in tags {
            let Some(entry) = self.entries.get(&tag) else {
                continue;
            };
            if entry.is_inline || entry.byte_len == 0 {
                continue;
            }
            if entry.byte_len > max_payload_bytes {
                return Err(RasterError::DecoderLimitsExceeded(format!(
                    "TIFF tag {tag} payload {} B exceeds limit {max_payload_bytes} B",
                    entry.byte_len
                )));
            }
            total_bytes = total_bytes.saturating_add(entry.byte_len);
            if total_bytes > max_payload_bytes {
                return Err(RasterError::DecoderLimitsExceeded(format!(
                    "TIFF IFD tag payloads {total_bytes} B exceed limit {max_payload_bytes} B"
                )));
            }
            let end = entry
                .offset
                .checked_add(entry.byte_len)
                .ok_or_else(|| RasterError::Decode(format!("TIFF tag {tag} offset overflow")))?;
            if end > reader.file_len {
                return Err(RasterError::Decode(format!(
                    "TIFF tag {tag} span {}..+{} exceeds file length {}",
                    entry.offset, entry.byte_len, reader.file_len
                )));
            }
            if end <= reader.prefix.len() as u64 {
                continue;
            }
            let already_loaded = self.loaded_spans.iter().any(|(off, data)| {
                entry.offset >= *off && end <= off.saturating_add(data.len() as u64)
            });
            if !already_loaded {
                needed.push((entry.offset, entry.byte_len));
            }
        }
        if needed.is_empty() {
            return Ok(());
        }
        needed.sort_unstable_by_key(|&(off, len)| (off, len));

        // Merge nearby spans into minimal contiguous reads without exceeding
        // either `max_payload_bytes` or `reader`'s remaining cumulative budget.
        let mut merged: Vec<(u64, u64)> = Vec::with_capacity(needed.len());
        let mut merged_total = 0u64;
        for (off, len) in needed {
            let next_end = off.saturating_add(len);
            if let Some(last) = merged.last_mut() {
                let last_end = last.0.saturating_add(last.1);
                if next_end <= last_end {
                    continue;
                }
                let new_end = last_end.max(next_end);
                let new_len = new_end.saturating_sub(last.0);
                let added = new_len.saturating_sub(last.1);
                let next_total = merged_total.saturating_add(added);
                if off <= last_end.saturating_add(OUT_OF_LINE_COALESCE_GAP)
                    && new_len <= OUT_OF_LINE_COALESCE_MAX_LEN
                    && next_total <= max_payload_bytes
                    && next_total <= reader.remaining_fetch_budget()
                {
                    last.1 = new_len;
                    merged_total = next_total;
                    continue;
                }
            }
            merged_total = merged_total.saturating_add(len);
            if merged_total > max_payload_bytes {
                return Err(RasterError::DecoderLimitsExceeded(format!(
                    "TIFF IFD tag payloads {merged_total} B exceed limit {max_payload_bytes} B"
                )));
            }
            merged.push((off, len));
        }

        for (off, len) in merged {
            if cancel.is_cancelled() {
                return Err(RasterError::Cancelled);
            }
            reader.charge_fetch(len)?;
            let len_usize = usize::try_from(len).map_err(|_| {
                RasterError::DecoderLimitsExceeded(format!(
                    "TIFF IFD tag span {len} B overflows usize"
                ))
            })?;
            let mut buf = vec![0u8; len_usize];
            reader.source.read_exact_at(off, &mut buf, cancel)?;
            self.loaded_spans.push((off, buf));
        }
        Ok(())
    }

    pub(super) fn read_u16_strict(
        &self,
        reader: &SinglePassIfdReader,
        tag: u16,
    ) -> Result<Option<u16>, RasterError> {
        let Some(vals) = self.read_u16_vec_strict(reader, tag)? else {
            return Ok(None);
        };
        if vals.len() != 1 {
            return Err(RasterError::Decode(format!(
                "TIFF tag {tag} expected 1 unsigned value, got {}",
                vals.len()
            )));
        }
        Ok(Some(vals[0]))
    }

    pub(super) fn read_u32_strict(
        &self,
        reader: &SinglePassIfdReader,
        tag: u16,
    ) -> Result<Option<u32>, RasterError> {
        let Some(vals) = self.read_u64_vec_strict(reader, tag)? else {
            return Ok(None);
        };
        if vals.len() != 1 {
            return Err(RasterError::Decode(format!(
                "TIFF tag {tag} expected 1 unsigned value, got {}",
                vals.len()
            )));
        }
        let v = u32::try_from(vals[0]).map_err(|_| {
            RasterError::Decode(format!("TIFF tag {tag} value {} overflows u32", vals[0]))
        })?;
        Ok(Some(v))
    }

    pub(super) fn read_u32_opt(&self, reader: &SinglePassIfdReader, tag: u16) -> Option<u32> {
        self.read_u32_strict(reader, tag).ok().flatten()
    }

    pub(super) fn read_u16_vec_strict(
        &self,
        reader: &SinglePassIfdReader,
        tag: u16,
    ) -> Result<Option<Vec<u16>>, RasterError> {
        let Some(vals) = self.read_u64_vec_strict(reader, tag)? else {
            return Ok(None);
        };
        let mut out = Vec::with_capacity(vals.len());
        for v in vals {
            out.push(u16::try_from(v).map_err(|_| {
                RasterError::Decode(format!("TIFF tag {tag} value {v} overflows u16"))
            })?);
        }
        Ok(Some(out))
    }

    pub(super) fn read_u16_vec_opt(&self, reader: &SinglePassIfdReader, tag: u16) -> Vec<u16> {
        self.read_u16_vec_strict(reader, tag)
            .ok()
            .flatten()
            .unwrap_or_default()
    }

    pub(super) fn read_u64_vec_strict(
        &self,
        reader: &SinglePassIfdReader,
        tag: u16,
    ) -> Result<Option<Vec<u64>>, RasterError> {
        let Some(entry) = self.entry(tag) else {
            return Ok(None);
        };
        let elem_size = match entry.field_type {
            1 => 1usize,
            3 => 2,
            4 | 13 => 4,
            16 | 18 => 8,
            other => {
                return Err(RasterError::Decode(format!(
                    "TIFF tag {tag} has unexpected non-unsigned field type {other}"
                )));
            }
        };
        let bytes = self.resolve_bytes(&entry, reader)?;
        let raw_count = usize::try_from(entry.count).map_err(|_| {
            RasterError::DecoderLimitsExceeded(format!("TIFF tag {tag} count overflow"))
        })?;
        let count = raw_count.min(bytes.len() / elem_size);
        let mut out = Vec::with_capacity(count);
        let order = self.byte_order;
        match entry.field_type {
            1 => {
                for &b in bytes.iter().take(count) {
                    out.push(u64::from(b));
                }
            }
            3 => {
                for chunk in bytes.chunks_exact(2).take(count) {
                    out.push(u64::from(read_u16(order, chunk)));
                }
            }
            4 | 13 => {
                for chunk in bytes.chunks_exact(4).take(count) {
                    out.push(u64::from(read_u32(order, chunk)));
                }
            }
            16 | 18 => {
                for chunk in bytes.chunks_exact(8).take(count) {
                    out.push(read_u64(order, chunk));
                }
            }
            _ => unreachable!("field_type validated above"),
        }
        Ok(Some(out))
    }

    pub(super) fn read_u8_vec_strict(
        &self,
        reader: &SinglePassIfdReader,
        tag: u16,
    ) -> Result<Option<Vec<u8>>, RasterError> {
        let Some(entry) = self.entry(tag) else {
            return Ok(None);
        };
        if !matches!(entry.field_type, 1 | 7) {
            return Err(RasterError::Decode(format!(
                "TIFF tag {tag} expected BYTE/UNDEFINED, got field type {}",
                entry.field_type
            )));
        }
        let bytes = self.resolve_bytes(&entry, reader)?;
        Ok(Some(bytes.to_vec()))
    }

    pub(super) fn read_numeric_f64_vec_opt(
        &self,
        reader: &SinglePassIfdReader,
        tag: u16,
    ) -> Option<Vec<f64>> {
        let entry = self.entry(tag)?;
        let elem_size = match entry.field_type {
            1 | 6 => 1usize,
            3 | 8 => 2,
            4 | 9 | 11 | 13 => 4,
            5 | 10 | 12 | 16 | 17 | 18 => 8,
            _ => return None,
        };
        let bytes = self.resolve_bytes(&entry, reader).ok()?;
        let count = usize::try_from(entry.count)
            .ok()?
            .min(bytes.len() / elem_size);
        let order = self.byte_order;
        let mut out = Vec::with_capacity(count);
        match entry.field_type {
            1 => out.extend(bytes.iter().take(count).map(|&b| f64::from(b))),
            6 => out.extend(bytes.iter().take(count).map(|&b| f64::from(b as i8))),
            3 => out.extend(
                bytes
                    .chunks_exact(2)
                    .take(count)
                    .map(|c| f64::from(read_u16(order, c))),
            ),
            8 => out.extend(
                bytes
                    .chunks_exact(2)
                    .take(count)
                    .map(|c| f64::from(read_i16(order, c))),
            ),
            4 | 13 => out.extend(
                bytes
                    .chunks_exact(4)
                    .take(count)
                    .map(|c| f64::from(read_u32(order, c))),
            ),
            9 => out.extend(
                bytes
                    .chunks_exact(4)
                    .take(count)
                    .map(|c| f64::from(read_i32(order, c))),
            ),
            16 | 18 => out.extend(
                bytes
                    .chunks_exact(8)
                    .take(count)
                    .map(|c| read_u64(order, c) as f64),
            ),
            17 => out.extend(
                bytes
                    .chunks_exact(8)
                    .take(count)
                    .map(|c| read_i64(order, c) as f64),
            ),
            5 => {
                for c in bytes.chunks_exact(8).take(count) {
                    let n = read_u32(order, &c[0..4]);
                    let d = read_u32(order, &c[4..8]);
                    if d != 0 {
                        out.push(f64::from(n) / f64::from(d));
                    }
                }
            }
            10 => {
                for c in bytes.chunks_exact(8).take(count) {
                    let n = read_i32(order, &c[0..4]);
                    let d = read_i32(order, &c[4..8]);
                    if d != 0 {
                        out.push(f64::from(n) / f64::from(d));
                    }
                }
            }
            11 => out.extend(
                bytes
                    .chunks_exact(4)
                    .take(count)
                    .map(|c| f64::from(read_f32(order, c))),
            ),
            12 => out.extend(
                bytes
                    .chunks_exact(8)
                    .take(count)
                    .map(|c| read_f64(order, c)),
            ),
            _ => return None,
        }
        Some(out)
    }

    pub(super) fn read_f64_vec_opt(&self, reader: &SinglePassIfdReader, tag: u16) -> Vec<f64> {
        let Some(entry) = self.entry(tag) else {
            return Vec::new();
        };
        if !matches!(entry.field_type, 11 | 12) {
            return Vec::new();
        }
        self.read_numeric_f64_vec_opt(reader, tag)
            .unwrap_or_default()
    }

    pub(super) fn read_ascii_bytes_opt(&self, reader: &SinglePassIfdReader, tag: u16) -> Vec<u8> {
        let Some(entry) = self.entry(tag) else {
            return Vec::new();
        };
        if !matches!(entry.field_type, 2 | 1 | 7) {
            return Vec::new();
        }
        let Ok(bytes) = self.resolve_bytes(&entry, reader) else {
            return Vec::new();
        };
        // Match `tiff::Decoder::get_tag_ascii_string`: truncate at the first NUL byte.
        bytes.split(|&b| b == 0).next().unwrap_or(&[]).to_vec()
    }

    pub(super) fn read_ascii_opt(&self, reader: &SinglePassIfdReader, tag: u16) -> String {
        let raw = self.read_ascii_bytes_opt(reader, tag);
        if raw.is_empty() {
            return String::new();
        }
        String::from_utf8_lossy(&raw).into_owned()
    }

    pub(super) fn read_geotiff_tags(&self, reader: &SinglePassIfdReader) -> RawGeoTiffTags {
        RawGeoTiffTags {
            raw_keys: self.read_u16_vec_opt(reader, TAG_GEO_KEY_DIRECTORY),
            double_params: self.read_f64_vec_opt(reader, TAG_GEO_DOUBLE_PARAMS),
            ascii_params: self.read_ascii_bytes_opt(reader, TAG_GEO_ASCII_PARAMS),
            points: self.read_f64_vec_opt(reader, TAG_MODEL_TIEPOINT),
            scale: self.read_f64_vec_opt(reader, TAG_MODEL_PIXEL_SCALE),
            matrix: self.read_f64_vec_opt(reader, TAG_MODEL_TRANSFORMATION),
        }
    }
}

/// Single-pass Classic TIFF + BigTIFF IFD chain reader over [`ByteSource`].
pub(super) struct SinglePassIfdReader {
    source: Arc<dyn ByteSource>,
    file_len: u64,
    prefix: Vec<u8>,
    bytes_fetched: u64,
    max_total_bytes: u64,
    pub(super) byte_order: TiffByteOrder,
    big_tiff: bool,
    pub(super) first_ifd: u64,
    visited: HashSet<u64>,
}

impl SinglePassIfdReader {
    /// Read the initial header prefix once and validate the Classic/BigTIFF
    /// signature and first IFD offset, enforcing a cumulative fetch budget of
    /// `max_total_bytes` across the prefix, all IFD directories, and prefetched
    /// tag payloads.
    pub(super) fn open(
        source: Arc<dyn ByteSource>,
        max_total_bytes: u64,
        cancel: &CancelToken,
    ) -> Result<Self, RasterError> {
        if cancel.is_cancelled() {
            return Err(RasterError::Cancelled);
        }
        let file_len = source.len();
        if file_len < 8 {
            return Err(RasterError::UnsupportedFormat);
        }
        if max_total_bytes < 8 {
            return Err(RasterError::DecoderLimitsExceeded(format!(
                "TIFF IFD fetch budget {max_total_bytes} B is smaller than 8 B header"
            )));
        }
        let prefix_len = file_len
            .min(INITIAL_HEADER_PREFIX_BYTES as u64)
            .min(max_total_bytes) as usize;
        let mut prefix = vec![0u8; prefix_len];
        source.read_exact_at(0, &mut prefix, cancel)?;

        let byte_order = match &prefix[0..2] {
            b"II" => TiffByteOrder::Little,
            b"MM" => TiffByteOrder::Big,
            _ => return Err(RasterError::UnsupportedFormat),
        };
        let version = read_u16(byte_order, &prefix[2..4]);
        let (big_tiff, first_ifd) = match version {
            42 => (false, u64::from(read_u32(byte_order, &prefix[4..8]))),
            43 => {
                if file_len < 16 {
                    return Err(RasterError::Decode("truncated BigTIFF header".to_owned()));
                }
                if prefix.len() < 16 {
                    return Err(RasterError::DecoderLimitsExceeded(format!(
                        "TIFF IFD fetch budget {max_total_bytes} B is smaller than 16 B BigTIFF header"
                    )));
                }
                let offset_size = read_u16(byte_order, &prefix[4..6]);
                let reserved = read_u16(byte_order, &prefix[6..8]);
                if offset_size != 8 || reserved != 0 {
                    return Err(RasterError::Decode(format!(
                        "unsupported BigTIFF header (offset_size={offset_size}, reserved={reserved})"
                    )));
                }
                (true, read_u64(byte_order, &prefix[8..16]))
            }
            _ => return Err(RasterError::UnsupportedFormat),
        };

        Ok(Self {
            source,
            file_len,
            bytes_fetched: prefix.len() as u64,
            max_total_bytes,
            prefix,
            byte_order,
            big_tiff,
            first_ifd,
            visited: HashSet::new(),
        })
    }

    pub(super) const fn remaining_fetch_budget(&self) -> u64 {
        self.max_total_bytes.saturating_sub(self.bytes_fetched)
    }

    fn charge_fetch(&mut self, bytes: u64) -> Result<(), RasterError> {
        let next = self.bytes_fetched.saturating_add(bytes);
        if next > self.max_total_bytes {
            return Err(RasterError::DecoderLimitsExceeded(format!(
                "cumulative TIFF IFD/tag fetch {next} B exceeds limit {} B",
                self.max_total_bytes
            )));
        }
        self.bytes_fetched = next;
        Ok(())
    }

    fn read_bytes_into(
        &mut self,
        offset: u64,
        out: &mut [u8],
        cancel: &CancelToken,
    ) -> Result<(), RasterError> {
        let len = out.len() as u64;
        let end = offset
            .checked_add(len)
            .ok_or_else(|| RasterError::Decode("TIFF IFD offset overflow".to_owned()))?;
        if end > self.file_len {
            return Err(RasterError::Decode(format!(
                "TIFF IFD span {offset}..+{len} exceeds file length {}",
                self.file_len
            )));
        }
        if end <= self.prefix.len() as u64 {
            let s = offset as usize;
            out.copy_from_slice(&self.prefix[s..s + out.len()]);
            return Ok(());
        }
        self.charge_fetch(len)?;
        self.source.read_exact_at(offset, out, cancel)?;
        Ok(())
    }

    /// Read one IFD entry table + `next_ifd` pointer at `ifd_offset` (zero
    /// extra reads when inside the initial header prefix, or at most two
    /// positioned reads—entry count then table—when out-of-prefix).
    pub(super) fn read_directory(
        &mut self,
        ifd_offset: u64,
        cancel: &CancelToken,
    ) -> Result<IfdDirectory, RasterError> {
        if cancel.is_cancelled() {
            return Err(RasterError::Cancelled);
        }
        if ifd_offset == 0 || ifd_offset >= self.file_len {
            return Err(RasterError::Decode(format!(
                "invalid TIFF IFD offset {ifd_offset} (file length {})",
                self.file_len
            )));
        }
        if !self.visited.insert(ifd_offset) {
            return Err(RasterError::Decode(format!(
                "cyclic TIFF IFD chain detected at offset {ifd_offset}"
            )));
        }

        let order = self.byte_order;
        let (count_size, entry_size, ptr_size, inline_cap) = if self.big_tiff {
            (8u64, 20usize, 8usize, 8u64)
        } else {
            (2u64, 12usize, 4usize, 4u64)
        };

        let mut count_buf = [0u8; 8];
        self.read_bytes_into(ifd_offset, &mut count_buf[..count_size as usize], cancel)?;
        let count_u64 = if self.big_tiff {
            read_u64(order, &count_buf[..8])
        } else {
            u64::from(read_u16(order, &count_buf[..2]))
        };
        if count_u64 > MAX_IFD_ENTRIES as u64 {
            return Err(RasterError::DecoderLimitsExceeded(format!(
                "TIFF IFD entry count {count_u64} exceeds limit {MAX_IFD_ENTRIES}"
            )));
        }
        let count = count_u64 as usize;

        let table_bytes = count
            .checked_mul(entry_size)
            .and_then(|n| n.checked_add(ptr_size))
            .ok_or_else(|| RasterError::Decode("TIFF IFD table size overflow".to_owned()))?;
        let table_offset = ifd_offset
            .checked_add(count_size)
            .ok_or_else(|| RasterError::Decode("TIFF IFD offset overflow".to_owned()))?;
        let mut raw_table = vec![0u8; table_bytes];
        self.read_bytes_into(table_offset, &mut raw_table, cancel)?;

        let mut entries = HashMap::with_capacity(count);
        for i in 0..count {
            let e = &raw_table[i * entry_size..(i + 1) * entry_size];
            let tag = read_u16(order, &e[0..2]);
            let field_type = read_u16(order, &e[2..4]);
            let (elem_count, val_field) = if self.big_tiff {
                (read_u64(order, &e[4..12]), &e[12..20])
            } else {
                (u64::from(read_u32(order, &e[4..8])), &e[8..12])
            };
            let Some(type_sz) = field_type_size(field_type) else {
                // Preserve unknown field types with zero inline payload so
                // unread private tags remain harmless while strict tag readers
                // reject corrupt required/layout tags instead of defaulting.
                entries.insert(
                    tag,
                    IfdEntry {
                        tag,
                        field_type,
                        count: 0,
                        inline_bytes: [0u8; 8],
                        offset: 0,
                        byte_len: 0,
                        is_inline: true,
                    },
                );
                continue;
            };
            let Some(byte_len) = elem_count.checked_mul(type_sz) else {
                return Err(RasterError::DecoderLimitsExceeded(format!(
                    "TIFF tag {tag} byte length overflow"
                )));
            };
            let is_inline = byte_len <= inline_cap;
            let mut inline_bytes = [0u8; 8];
            let offset = if is_inline {
                inline_bytes[..val_field.len()].copy_from_slice(val_field);
                0
            } else if self.big_tiff {
                read_u64(order, val_field)
            } else {
                u64::from(read_u32(order, val_field))
            };
            entries.insert(
                tag,
                IfdEntry {
                    tag,
                    field_type,
                    count: elem_count,
                    inline_bytes,
                    offset,
                    byte_len,
                    is_inline,
                },
            );
        }

        let next_slice = &raw_table[count * entry_size..];
        let next_ifd = if self.big_tiff {
            read_u64(order, next_slice)
        } else {
            u64::from(read_u32(order, next_slice))
        };

        Ok(IfdDirectory {
            byte_order: order,
            next_ifd,
            entries,
            loaded_spans: Vec::new(),
        })
    }
}

#[cfg(test)]
#[path = "tags_tests.rs"]
mod tests;
