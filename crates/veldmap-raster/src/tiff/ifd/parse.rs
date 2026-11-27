//! Open-time IFD tag parser over [`super::wire::SinglePassIfdReader`] producing
//! validated, budget-bounded [`CachedIfd`] descriptors.

use std::sync::Arc;

use veldmap_core::budget::MemoryBudget;
use veldmap_core::sync::CancelToken;

use crate::RasterError;
use crate::pipeline::{PixelLayout, SampleType, admit_chunk_working_bytes};

use super::super::ycbcr::YCbCrParams;
use super::tag_ids::{
    TAG_BITS_PER_SAMPLE, TAG_COLOR_MAP, TAG_COMPRESSION, TAG_EXTRA_SAMPLES, TAG_IMAGE_LENGTH,
    TAG_IMAGE_WIDTH, TAG_JPEG_TABLES, TAG_PHOTOMETRIC, TAG_PLANAR_CONFIG, TAG_PREDICTOR,
    TAG_REFERENCE_BLACK_WHITE, TAG_ROWS_PER_STRIP, TAG_SAMPLE_FORMAT, TAG_SAMPLES_PER_PIXEL,
    TAG_STRIP_BYTE_COUNTS, TAG_STRIP_OFFSETS, TAG_TILE_BYTE_COUNTS, TAG_TILE_LENGTH,
    TAG_TILE_OFFSETS, TAG_TILE_WIDTH, TAG_YCBCR_COEFFICIENTS, TAG_YCBCR_SUBSAMPLING,
};
use super::wire::{IfdDirectory, SinglePassIfdReader};
use super::{
    CachedIfd, MAX_SINGLE_CHUNK_BYTES, Photometric, PlanarConfig, TiffCompression, TiffPredictor,
};

pub(in crate::tiff) fn max_cached_ifd_table_bytes(budget: &MemoryBudget, is_remote: bool) -> u64 {
    let base = (budget.decode_working_bytes / 16).max(4 * 1024 * 1024);
    if is_remote {
        base.min(budget.remote_ifd_limit_bytes)
    } else {
        base
    }
}

/// Cumulative memory/fetch accounting context threaded across open-time IFD
/// parsing and overview scanning.
#[derive(Clone, Copy)]
pub(in crate::tiff) struct IfdParseBudget<'a> {
    pub(in crate::tiff) budget: &'a MemoryBudget,
    pub(in crate::tiff) table_bytes_used: u64,
    pub(in crate::tiff) is_remote: bool,
}

impl<'a> IfdParseBudget<'a> {
    pub(in crate::tiff) const fn new(budget: &'a MemoryBudget, is_remote: bool) -> Self {
        Self {
            budget,
            table_bytes_used: 0,
            is_remote,
        }
    }

    pub(in crate::tiff) fn table_cap(&self) -> u64 {
        max_cached_ifd_table_bytes(self.budget, self.is_remote)
    }

    pub(in crate::tiff) fn remaining_table_cap(&self) -> u64 {
        self.table_cap().saturating_sub(self.table_bytes_used)
    }

    pub(in crate::tiff) fn charge_ifd(&mut self, ifd: &CachedIfd) {
        self.table_bytes_used = self.table_bytes_used.saturating_add(ifd.table_bytes());
    }

    pub(in crate::tiff) fn metadata_prefetch_cap(&self, reader: &SinglePassIfdReader) -> u64 {
        self.remaining_table_cap().min(if self.is_remote {
            reader.remaining_fetch_budget()
        } else {
            u64::MAX
        })
    }
}

fn resolve_sample_type(sample_format: u16, bits: u8) -> Result<SampleType, RasterError> {
    match (sample_format, bits) {
        (1, 8) => Ok(SampleType::U8),
        (2, 8) => Ok(SampleType::I8),
        (1, 16) => Ok(SampleType::U16),
        (2, 16) => Ok(SampleType::I16),
        (1, 32) => Ok(SampleType::U32),
        (2, 32) => Ok(SampleType::I32),
        (3, 16) => Ok(SampleType::F16),
        (3, 32) => Ok(SampleType::F32),
        (3, 64) => Ok(SampleType::F64),
        _ => Err(RasterError::Decode(format!(
            "unsupported TIFF sample format ({sample_format}, {bits}-bit)"
        ))),
    }
}

fn parse_color_map(
    dir: &IfdDirectory,
    reader: &SinglePassIfdReader,
    sample_type: SampleType,
) -> Result<Arc<[[u8; 4]]>, RasterError> {
    let expected_entries = match sample_type {
        SampleType::U8 => 256usize,
        SampleType::U16 => 65_536usize,
        other => {
            return Err(RasterError::Decode(format!(
                "unsupported Palette TIFF sample type {other:?} (expected U8 or U16)"
            )));
        }
    };
    let raw = dir
        .read_u16_vec_strict(reader, TAG_COLOR_MAP)?
        .unwrap_or_default();
    if raw.len() != expected_entries * 3 {
        return Err(RasterError::Decode(format!(
            "invalid TIFF ColorMap length {} for {sample_type:?} Palette (expected {})",
            raw.len(),
            expected_entries * 3
        )));
    }
    let n = expected_entries;
    let is_8bit_scaled = raw.iter().all(|&v| v <= 255);
    let to_u8 = |v: u16| -> u8 {
        if is_8bit_scaled {
            v as u8
        } else {
            (v >> 8) as u8
        }
    };
    let mut lut = Vec::with_capacity(n);
    for i in 0..n {
        lut.push([to_u8(raw[i]), to_u8(raw[n + i]), to_u8(raw[2 * n + i]), 255]);
    }
    Ok(Arc::from(lut))
}

fn resolve_pixel_layout(
    photometric: Photometric,
    samples_per_pixel: u16,
    extra_samples: &[u16],
) -> Result<(PixelLayout, u16), RasterError> {
    let spp = usize::from(samples_per_pixel);
    match photometric {
        Photometric::WhiteIsZero => {
            if spp > 1 {
                return Err(RasterError::Decode(
                    "WhiteIsZero with multiple samples is not supported".to_owned(),
                ));
            }
            Ok((PixelLayout::named(1), 1))
        }
        Photometric::BlackIsZero => {
            if spp == 1 {
                Ok((PixelLayout::named(1), 1))
            } else if spp == 2 && extra_samples.first().is_some_and(|&v| v == 1 || v == 2) {
                Ok((PixelLayout::named(2), 1))
            } else {
                let base_count = spp.saturating_sub(extra_samples.len()).max(1);
                let alpha_ch = extra_samples
                    .iter()
                    .position(|&v| v == 1 || v == 2)
                    .map(|idx| base_count + idx)
                    .filter(|&ch| ch < spp);
                let data_bands = if alpha_ch.is_some() {
                    samples_per_pixel.saturating_sub(1).max(1)
                } else {
                    samples_per_pixel
                };
                Ok((PixelLayout::stack_with_alpha(spp, alpha_ch), data_bands))
            }
        }
        Photometric::Rgb => {
            if spp < 3 {
                return Err(RasterError::Decode(format!(
                    "RGB TIFF requires at least 3 samples/pixel, got {spp}"
                )));
            }
            if spp == 3 {
                return Ok((PixelLayout::named(3), 3));
            }
            let alpha_ch = extra_samples
                .iter()
                .position(|&v| v == 1 || v == 2)
                .map(|idx| 3 + idx)
                .filter(|&ch| ch < spp);
            let layout = PixelLayout::rgb(spp, alpha_ch);
            let data_bands = if alpha_ch.is_some() {
                samples_per_pixel.saturating_sub(1).max(3)
            } else {
                samples_per_pixel
            };
            Ok((layout, data_bands))
        }
        Photometric::Palette => {
            if spp == 1 {
                Ok((PixelLayout::named(1), 1))
            } else if spp == 2 && extra_samples.first().is_some_and(|&v| v == 1 || v == 2) {
                Ok((PixelLayout::named(2), 1))
            } else {
                Err(RasterError::Decode(format!(
                    "unsupported Palette TIFF with {spp} samples/pixel"
                )))
            }
        }
        Photometric::YCbCr => {
            if spp != 3 {
                return Err(RasterError::Decode(format!(
                    "YCbCr TIFF requires 3 samples/pixel, got {spp}"
                )));
            }
            Ok((PixelLayout::named(3), 3))
        }
        Photometric::Other(code) => Err(RasterError::Decode(format!(
            "unsupported TIFF PhotometricInterpretation {code}"
        ))),
    }
}

fn read_sample_type(
    dir: &IfdDirectory,
    reader: &SinglePassIfdReader,
) -> Result<SampleType, RasterError> {
    let bps_vec = dir
        .read_u16_vec_strict(reader, TAG_BITS_PER_SAMPLE)?
        .unwrap_or_default();
    let sample_bits_u16 = bps_vec.first().copied().unwrap_or(1);
    if bps_vec.iter().any(|&b| b != sample_bits_u16) {
        return Err(RasterError::Decode(format!(
            "heterogeneous per-band BitsPerSample {bps_vec:?} is not supported"
        )));
    }
    let sample_bits = u8::try_from(sample_bits_u16)
        .map_err(|_| RasterError::Decode(format!("unsupported BitsPerSample {sample_bits_u16}")))?;

    let sample_fmt_vec = dir
        .read_u16_vec_strict(reader, TAG_SAMPLE_FORMAT)?
        .unwrap_or_default();
    let sample_fmt_code = sample_fmt_vec.first().copied().unwrap_or(1);
    if sample_fmt_vec.iter().any(|&f| f != sample_fmt_code) {
        return Err(RasterError::Decode(format!(
            "heterogeneous per-band SampleFormat {sample_fmt_vec:?} is not supported"
        )));
    }
    resolve_sample_type(sample_fmt_code, sample_bits)
}

fn validate_codec_and_layout(
    compression: TiffCompression,
    photometric: Photometric,
    planar: PlanarConfig,
    samples_per_pixel: u16,
    sample_type: SampleType,
) -> Result<(), RasterError> {
    if photometric == Photometric::YCbCr && sample_type != SampleType::U8 {
        return Err(RasterError::Decode(format!(
            "YCbCr TIFF requires 8-bit unsigned samples, got {sample_type:?}"
        )));
    }
    if photometric == Photometric::Palette
        && !matches!(sample_type, SampleType::U8 | SampleType::U16)
    {
        return Err(RasterError::Decode(format!(
            "Palette TIFF requires 8- or 16-bit unsigned indices, got {sample_type:?}"
        )));
    }
    if planar == PlanarConfig::Planar && photometric == Photometric::YCbCr {
        return Err(RasterError::Decode(
            "planar YCbCr TIFF is not supported".to_owned(),
        ));
    }
    if matches!(compression, TiffCompression::Jpeg | TiffCompression::WebP) {
        if sample_type != SampleType::U8 {
            return Err(RasterError::Decode(format!(
                "{compression:?}-compressed TIFF requires 8-bit unsigned samples, got {sample_type:?}"
            )));
        }
        if photometric == Photometric::Palette {
            return Err(RasterError::Decode(format!(
                "{compression:?}-compressed Palette TIFF is not supported"
            )));
        }
    }
    if compression == TiffCompression::Jpeg {
        let samples_in_chunk = match planar {
            PlanarConfig::Chunky => samples_per_pixel,
            PlanarConfig::Planar => 1,
        };
        if !matches!(samples_in_chunk, 1 | 3 | 4) {
            return Err(RasterError::Decode(format!(
                "JPEG-compressed TIFF requires 1, 3, or 4 samples per chunk, got {samples_in_chunk}"
            )));
        }
    }
    if compression == TiffCompression::WebP
        && (planar != PlanarConfig::Chunky
            || !matches!(samples_per_pixel, 3 | 4)
            || photometric == Photometric::YCbCr)
    {
        return Err(RasterError::Decode(format!(
            "WebP-compressed TIFF requires chunky 3- or 4-band RGB/RGBA, got planar={planar:?}, spp={samples_per_pixel}, photometric={photometric:?}"
        )));
    }
    Ok(())
}

struct ChunkTables {
    tiled: bool,
    chunk_w: u32,
    chunk_h: u32,
    chunks_across: u32,
    chunks_per_plane: u32,
    offsets: Vec<u64>,
    byte_counts: Vec<u64>,
}

fn read_chunk_tables(
    dir: &IfdDirectory,
    reader: &SinglePassIfdReader,
    image: usize,
    width: u32,
    height: u32,
    layout: &IfdLayoutTags,
) -> Result<ChunkTables, RasterError> {
    let tile_w = dir.read_u32_opt(reader, TAG_TILE_WIDTH);
    let (tiled, chunk_w, chunk_h, offsets, byte_counts) = if let Some(tw) = tile_w {
        let th = dir
            .read_u32_strict(reader, TAG_TILE_LENGTH)?
            .ok_or_else(|| RasterError::Decode("tiled TIFF missing TileLength".to_owned()))?;
        let offsets = dir
            .read_u64_vec_strict(reader, TAG_TILE_OFFSETS)?
            .ok_or_else(|| RasterError::Decode("tiled TIFF missing TileOffsets".to_owned()))?;
        let byte_counts = dir
            .read_u64_vec_strict(reader, TAG_TILE_BYTE_COUNTS)?
            .ok_or_else(|| RasterError::Decode("tiled TIFF missing TileByteCounts".to_owned()))?;
        (true, tw, th, offsets, byte_counts)
    } else {
        let rows_per_strip = dir
            .read_u32_opt(reader, TAG_ROWS_PER_STRIP)
            .unwrap_or(height)
            .clamp(1, height);
        let offsets = dir
            .read_u64_vec_strict(reader, TAG_STRIP_OFFSETS)?
            .ok_or_else(|| RasterError::Decode("stripped TIFF missing StripOffsets".to_owned()))?;
        let byte_counts = dir
            .read_u64_vec_strict(reader, TAG_STRIP_BYTE_COUNTS)?
            .ok_or_else(|| {
                RasterError::Decode("stripped TIFF missing StripByteCounts".to_owned())
            })?;
        (false, width, rows_per_strip, offsets, byte_counts)
    };

    if chunk_w == 0 || chunk_h == 0 {
        return Err(RasterError::InvalidDimensions {
            width: chunk_w,
            height: chunk_h,
        });
    }

    let chunks_across = width.div_ceil(chunk_w);
    let chunks_down = height.div_ceil(chunk_h);
    let chunks_per_plane = chunks_across
        .checked_mul(chunks_down)
        .ok_or_else(|| RasterError::Decode("TIFF chunk count overflow".to_owned()))?;
    let expected_chunks = match layout.planar {
        PlanarConfig::Chunky => usize::try_from(chunks_per_plane)
            .map_err(|_| RasterError::Decode("TIFF chunk count overflow".to_owned()))?,
        PlanarConfig::Planar => usize::try_from(chunks_per_plane)
            .ok()
            .and_then(|n| n.checked_mul(usize::from(layout.samples_per_pixel)))
            .ok_or_else(|| RasterError::Decode("TIFF planar chunk count overflow".to_owned()))?,
    };
    if offsets.len() < expected_chunks || byte_counts.len() < expected_chunks {
        return Err(RasterError::Decode(format!(
            "TIFF chunk table too short for IFD {image}: expected {expected_chunks} entries, got {} offsets / {} byte counts",
            offsets.len(),
            byte_counts.len()
        )));
    }

    Ok(ChunkTables {
        tiled,
        chunk_w,
        chunk_h,
        chunks_across,
        chunks_per_plane,
        offsets,
        byte_counts,
    })
}

fn read_ycbcr_if_needed(
    dir: &IfdDirectory,
    reader: &SinglePassIfdReader,
    photometric: Photometric,
    compression: TiffCompression,
) -> Result<Option<YCbCrParams>, RasterError> {
    if photometric != Photometric::YCbCr {
        return Ok(None);
    }
    if compression != TiffCompression::Jpeg {
        let sub = dir
            .read_u16_vec_strict(reader, TAG_YCBCR_SUBSAMPLING)?
            .unwrap_or_else(|| vec![2, 2]);
        if sub.as_slice() != [1, 1] {
            return Err(RasterError::Decode(format!(
                "non-JPEG subsampled YCbCr TIFF (YCbCrSubSampling={sub:?}) is not supported"
            )));
        }
    }
    let coeffs = dir.read_numeric_f64_vec_opt(reader, TAG_YCBCR_COEFFICIENTS);
    let rbw = dir.read_numeric_f64_vec_opt(reader, TAG_REFERENCE_BLACK_WHITE);
    Ok(Some(YCbCrParams::from_tags(
        coeffs.as_deref(),
        rbw.as_deref(),
    )))
}

fn read_jpeg_tables_if_needed(
    dir: &IfdDirectory,
    reader: &SinglePassIfdReader,
    compression: TiffCompression,
) -> Result<Option<Arc<[u8]>>, RasterError> {
    if compression != TiffCompression::Jpeg {
        return Ok(None);
    }
    match dir.read_u8_vec_strict(reader, TAG_JPEG_TABLES)? {
        Some(tables) => {
            if tables.len() < 4 {
                return Err(RasterError::Decode(
                    "TIFF JPEGTables tag is shorter than 4 bytes".to_owned(),
                ));
            }
            Ok(Some(Arc::from(tables)))
        }
        None => Ok(None),
    }
}

struct IfdLayoutTags {
    compression: TiffCompression,
    predictor: TiffPredictor,
    planar: PlanarConfig,
    photometric: Photometric,
    samples_per_pixel: u16,
    sample_type: SampleType,
    pixel_layout: PixelLayout,
    data_bands: u16,
}

fn read_ifd_layout_tags(
    dir: &IfdDirectory,
    reader: &SinglePassIfdReader,
) -> Result<IfdLayoutTags, RasterError> {
    let compression_code = dir.read_u16_strict(reader, TAG_COMPRESSION)?.unwrap_or(1);
    let compression = TiffCompression::from_code(compression_code)?;

    let photometric_code = dir.read_u16_strict(reader, TAG_PHOTOMETRIC)?.unwrap_or(1);
    let photometric = Photometric::from_code(photometric_code);

    let samples_per_pixel = dir
        .read_u16_strict(reader, TAG_SAMPLES_PER_PIXEL)?
        .unwrap_or(1);
    if samples_per_pixel == 0 {
        return Err(RasterError::Decode("TIFF SamplesPerPixel is 0".to_owned()));
    }

    let sample_type = read_sample_type(dir, reader)?;

    let planar_code = dir.read_u16_strict(reader, TAG_PLANAR_CONFIG)?.unwrap_or(1);
    let planar = match planar_code {
        1 => PlanarConfig::Chunky,
        2 => PlanarConfig::Planar,
        other => {
            return Err(RasterError::Decode(format!(
                "unsupported TIFF PlanarConfiguration {other}"
            )));
        }
    };

    validate_codec_and_layout(
        compression,
        photometric,
        planar,
        samples_per_pixel,
        sample_type,
    )?;

    let predictor_code = dir.read_u16_strict(reader, TAG_PREDICTOR)?.unwrap_or(1);
    let predictor = TiffPredictor::from_code(predictor_code, sample_type)?;

    let extra_samples = dir
        .read_u16_vec_strict(reader, TAG_EXTRA_SAMPLES)?
        .unwrap_or_default();
    let (pixel_layout, data_bands) =
        resolve_pixel_layout(photometric, samples_per_pixel, &extra_samples)?;

    Ok(IfdLayoutTags {
        compression,
        predictor,
        planar,
        photometric,
        samples_per_pixel,
        sample_type,
        pixel_layout,
        data_bands,
    })
}

fn admit_ifd_budget(
    cached: &CachedIfd,
    parse_budget: IfdParseBudget<'_>,
) -> Result<(), RasterError> {
    let chunk_cap = parse_budget
        .budget
        .decode_working_bytes
        .min(MAX_SINGLE_CHUNK_BYTES);
    admit_chunk_working_bytes(
        cached.chunk_w,
        cached.chunk_h,
        cached.raw_pixel_bytes(),
        4,
        chunk_cap,
    )?;

    let table_cap = parse_budget.table_cap();
    let new_total = parse_budget
        .table_bytes_used
        .saturating_add(cached.table_bytes());
    if new_total > table_cap {
        return Err(RasterError::LimitExceeded {
            width: cached.width,
            height: cached.height,
            estimated_bytes: new_total,
            limit_bytes: table_cap,
        });
    }
    Ok(())
}

/// Small fixed-schema scalar/layout tags capped before prefetching so a
/// crafted TIFF cannot point them at megabytes of out-of-line data.
const MAX_SMALL_SCHEMA_TAG_BYTES: u64 = 64 * 1024;

/// Maximum wire size for TIFF `ColorMap` (tag 320: at most `65536 * 3 * 2 = 384 KiB`).
const MAX_COLOR_MAP_TAG_BYTES: u64 = 512 * 1024;

const SMALL_SCHEMA_TAGS: [u16; 16] = [
    TAG_IMAGE_WIDTH,
    TAG_IMAGE_LENGTH,
    TAG_BITS_PER_SAMPLE,
    TAG_COMPRESSION,
    TAG_PHOTOMETRIC,
    TAG_SAMPLES_PER_PIXEL,
    TAG_ROWS_PER_STRIP,
    TAG_PLANAR_CONFIG,
    TAG_PREDICTOR,
    TAG_TILE_WIDTH,
    TAG_TILE_LENGTH,
    TAG_EXTRA_SAMPLES,
    TAG_SAMPLE_FORMAT,
    TAG_YCBCR_COEFFICIENTS,
    TAG_YCBCR_SUBSAMPLING,
    TAG_REFERENCE_BLACK_WHITE,
];

const TABLE_TAGS: [u16; 6] = [
    TAG_TILE_OFFSETS,
    TAG_TILE_BYTE_COUNTS,
    TAG_STRIP_OFFSETS,
    TAG_STRIP_BYTE_COUNTS,
    TAG_JPEG_TABLES,
    TAG_COLOR_MAP,
];

/// Unified list of all layout + table tags required to build a [`CachedIfd`]
/// (excluding GeoTIFF/GDAL base-only tags) so `prefetch_tags` executes in a
/// single coalesced pass.
const IFD_LAYOUT_AND_TABLE_TAGS: [u16; SMALL_SCHEMA_TAGS.len() + TABLE_TAGS.len()] = {
    let mut combined = [0u16; SMALL_SCHEMA_TAGS.len() + TABLE_TAGS.len()];
    let mut i = 0;
    while i < SMALL_SCHEMA_TAGS.len() {
        combined[i] = SMALL_SCHEMA_TAGS[i];
        i += 1;
    }
    let mut j = 0;
    while j < TABLE_TAGS.len() {
        combined[SMALL_SCHEMA_TAGS.len() + j] = TABLE_TAGS[j];
        j += 1;
    }
    combined
};

fn preflight_and_prefetch_ifd(
    dir: &mut IfdDirectory,
    reader: &mut SinglePassIfdReader,
    parse_budget: IfdParseBudget<'_>,
    cancel: &CancelToken,
) -> Result<(), RasterError> {
    let table_cap = parse_budget.table_cap();
    let remaining_table_cap = parse_budget.remaining_table_cap();

    // Pre-check fixed-schema scalar/layout tags and ColorMap wire size BEFORE prefetching.
    for tag in SMALL_SCHEMA_TAGS {
        if let Some(e) = dir.entry(tag)
            && e.byte_len() > MAX_SMALL_SCHEMA_TAG_BYTES
        {
            return Err(RasterError::DecoderLimitsExceeded(format!(
                "TIFF layout tag {tag} payload {} B exceeds limit {MAX_SMALL_SCHEMA_TAG_BYTES} B",
                e.byte_len()
            )));
        }
    }
    if let Some(e) = dir.entry(TAG_COLOR_MAP)
        && e.byte_len() > MAX_COLOR_MAP_TAG_BYTES
    {
        return Err(RasterError::DecoderLimitsExceeded(format!(
            "TIFF ColorMap payload {} B exceeds limit {MAX_COLOR_MAP_TAG_BYTES} B",
            e.byte_len()
        )));
    }

    // Pre-check estimated table memory BEFORE prefetching large out-of-line offset arrays.
    let mut est_table_bytes = 0u64;
    for tag in TABLE_TAGS {
        if let Some(e) = dir.entry(tag) {
            let scaled = match tag {
                TAG_TILE_OFFSETS
                | TAG_TILE_BYTE_COUNTS
                | TAG_STRIP_OFFSETS
                | TAG_STRIP_BYTE_COUNTS => e.count().saturating_mul(8),
                _ => e.byte_len(),
            };
            est_table_bytes = est_table_bytes.saturating_add(scaled);
        }
    }
    if est_table_bytes > remaining_table_cap {
        let w = dir.read_u32_opt(reader, TAG_IMAGE_WIDTH).unwrap_or(0);
        let h = dir.read_u32_opt(reader, TAG_IMAGE_LENGTH).unwrap_or(0);
        return Err(RasterError::LimitExceeded {
            width: w,
            height: h,
            estimated_bytes: parse_budget
                .table_bytes_used
                .saturating_add(est_table_bytes),
            limit_bytes: table_cap,
        });
    }

    let payload_cap = remaining_table_cap
        .saturating_add(1024 * 1024)
        .min(
            parse_budget
                .budget
                .decode_working_bytes
                .clamp(4 * 1024 * 1024, 128 * 1024 * 1024),
        )
        .min(if parse_budget.is_remote {
            reader.remaining_fetch_budget()
        } else {
            u64::MAX
        });
    dir.prefetch_tags(reader, &IFD_LAYOUT_AND_TABLE_TAGS, payload_cap, cancel)
}

/// Parse and validate all chunk/tag metadata for `dir` while enforcing
/// `MemoryBudget` before allocating out-of-line offset/table arrays.
pub(in crate::tiff) fn parse_cached_ifd(
    dir: &mut IfdDirectory,
    reader: &mut SinglePassIfdReader,
    image: usize,
    parse_budget: IfdParseBudget<'_>,
    cancel: &CancelToken,
) -> Result<CachedIfd, RasterError> {
    preflight_and_prefetch_ifd(dir, reader, parse_budget, cancel)?;

    let width = dir
        .read_u32_strict(reader, TAG_IMAGE_WIDTH)?
        .ok_or_else(|| RasterError::Decode("TIFF missing ImageWidth".to_owned()))?;
    let height = dir
        .read_u32_strict(reader, TAG_IMAGE_LENGTH)?
        .ok_or_else(|| RasterError::Decode("TIFF missing ImageLength".to_owned()))?;
    if width == 0 || height == 0 {
        return Err(RasterError::InvalidDimensions { width, height });
    }

    let layout = read_ifd_layout_tags(dir, reader)?;
    let ycbcr = read_ycbcr_if_needed(dir, reader, layout.photometric, layout.compression)?;
    let color_map = if layout.photometric == Photometric::Palette {
        Some(parse_color_map(dir, reader, layout.sample_type)?)
    } else {
        None
    };
    let tables = read_chunk_tables(dir, reader, image, width, height, &layout)?;
    let jpeg_tables = read_jpeg_tables_if_needed(dir, reader, layout.compression)?;

    let cached = CachedIfd {
        image,
        width,
        height,
        tiled: tables.tiled,
        chunk_w: tables.chunk_w,
        chunk_h: tables.chunk_h,
        chunks_across: tables.chunks_across,
        chunks_per_plane: tables.chunks_per_plane,
        compression: layout.compression,
        predictor: layout.predictor,
        planar: layout.planar,
        photometric: layout.photometric,
        samples_per_pixel: layout.samples_per_pixel,
        sample_type: layout.sample_type,
        pixel_layout: layout.pixel_layout,
        data_bands: layout.data_bands,
        offsets: tables.offsets,
        byte_counts: tables.byte_counts,
        jpeg_tables,
        ycbcr,
        color_map,
    };

    admit_ifd_budget(&cached, parse_budget)?;
    Ok(cached)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pixel_layout_handles_rgbn_and_non_last_alpha_consistently() {
        let (layout, data_bands) = resolve_pixel_layout(Photometric::Rgb, 4, &[0]).expect("rgbn");
        assert_eq!(layout.alpha_channel(), None);
        assert_eq!(layout.colors(), 3);
        assert_eq!(layout.channels, 4);
        assert_eq!(data_bands, 4);

        let (layout5, data_bands5) =
            resolve_pixel_layout(Photometric::Rgb, 5, &[2, 0]).expect("rgb+a+nir");
        assert_eq!(layout5.alpha_channel(), Some(3));
        assert_eq!(data_bands5, 4);

        let (layout_gray3, data_bands_g3) =
            resolve_pixel_layout(Photometric::BlackIsZero, 3, &[2]).expect("gray3+a");
        assert_eq!(layout_gray3.alpha_channel(), Some(2));
        assert_eq!(data_bands_g3, 2);
    }
}
