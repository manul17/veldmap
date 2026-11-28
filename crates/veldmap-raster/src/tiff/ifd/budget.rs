//! Open-time TIFF IFD memory/fetch budget accounting and pre-fetch tag
//! allowlists enforced before allocating out-of-line offset/table arrays.

use veldmap_core::budget::MemoryBudget;
use veldmap_core::sync::CancelToken;

use crate::RasterError;
use crate::pipeline::admit_chunk_working_bytes;

use super::super::tag_ids::{
    TAG_BITS_PER_SAMPLE, TAG_COLOR_MAP, TAG_COMPRESSION, TAG_EXTRA_SAMPLES, TAG_IMAGE_LENGTH,
    TAG_IMAGE_WIDTH, TAG_JPEG_TABLES, TAG_PHOTOMETRIC, TAG_PLANAR_CONFIG, TAG_PREDICTOR,
    TAG_REFERENCE_BLACK_WHITE, TAG_ROWS_PER_STRIP, TAG_SAMPLE_FORMAT, TAG_SAMPLES_PER_PIXEL,
    TAG_STRIP_BYTE_COUNTS, TAG_STRIP_OFFSETS, TAG_TILE_BYTE_COUNTS, TAG_TILE_LENGTH,
    TAG_TILE_OFFSETS, TAG_TILE_WIDTH, TAG_YCBCR_COEFFICIENTS, TAG_YCBCR_SUBSAMPLING,
};
use super::wire::{IfdDirectory, SinglePassIfdReader};
use super::{CachedIfd, MAX_SINGLE_CHUNK_BYTES};

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

fn max_cached_ifd_table_bytes(budget: &MemoryBudget, is_remote: bool) -> u64 {
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
    budget: &'a MemoryBudget,
    table_bytes_used: u64,
    is_remote: bool,
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

pub(super) fn admit_ifd_budget(
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

pub(super) fn preflight_and_prefetch_ifd(
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ifd_table_budget_clamps_remote_limit_safely() {
        let tiny_budget = MemoryBudget::for_host_ram(1024 * 1024 * 1024);
        assert!(max_cached_ifd_table_bytes(&tiny_budget, false) >= 4 * 1024 * 1024);
        assert!(
            max_cached_ifd_table_bytes(&tiny_budget, true) <= tiny_budget.remote_ifd_limit_bytes
        );
    }
}
