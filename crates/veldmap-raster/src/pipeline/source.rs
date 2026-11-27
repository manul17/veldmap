//! Thin format-agnostic analytical [`RasterSource`] seam over concrete
//! decoders ([`crate::TiffSource`], [`crate::PngSource`], [`crate::JpegSource`],
//! and future mission codecs).

use veldmap_core::budget::MemoryBudget;
use veldmap_core::tile::{
    BandKey, DecodedTile, PixelFormat, SourceFingerprint, SourceId, TileStreamSpec,
};

use crate::RasterError;

use super::DecodeContext;
use super::cascade::TileEmit;
use super::metadata::{BandSample, BandStats, RasterDescription};
use super::plan::{DISPLAY_STREAM_SPEC, PyramidMeta};

/// Default maximum gap (`64 KiB`) bridged when coalescing remote byte ranges.
pub const DEFAULT_COALESCE_GAP_BYTES: u64 = 64 * 1024;

/// Default maximum merged span length (`4 MiB`) when coalescing remote byte ranges.
pub const DEFAULT_COALESCE_MAX_SPAN_BYTES: u64 = 4 * 1024 * 1024;

/// Contiguous byte span `(offset, len)` inside a raster's backing
/// [`veldmap_core::io::ByteSource`], used by remote COG batch prefetchers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ByteRange {
    pub offset: u64,
    pub len: u64,
}

/// Sort `ranges` by `offset` and coalesce overlapping or nearby spans whose
/// gap is `<= max_gap` and merged span length stays `<= max_merged_len`.
pub fn coalesce_byte_ranges(ranges: &mut Vec<ByteRange>, max_gap: u64, max_merged_len: u64) {
    ranges.retain(|r| r.len > 0);
    if ranges.len() <= 1 {
        return;
    }
    ranges.sort_unstable_by_key(|r| (r.offset, r.len));
    let mut write = 0usize;
    for read in 1..ranges.len() {
        let cur = ranges[write];
        let next = ranges[read];
        let cur_end = cur.offset.saturating_add(cur.len);
        let next_end = next.offset.saturating_add(next.len);
        let merged_end = cur_end.max(next_end);
        let merged_len = merged_end.saturating_sub(cur.offset);
        if (next.offset <= cur_end.saturating_add(max_gap) && merged_len <= max_merged_len)
            || next_end <= cur_end
        {
            ranges[write].len = merged_len;
        } else {
            write += 1;
            ranges[write] = next;
        }
    }
    ranges.truncate(write + 1);
}

/// Format-agnostic opened analytical raster source capable of describing its
/// pyramid, producing pointwise or sequential-pass tiles across `Display`,
/// `Scientific`, and `Categorical` tracks, and answering zoom-invariant
/// native-resolution pixel queries.
pub trait RasterSource: Send + Sync {
    /// Content fingerprint identifying this source for disk-cache keying.
    fn fingerprint(&self) -> SourceFingerprint;

    /// Deterministic [`SourceId`] derived from [`Self::fingerprint`].
    fn source_id(&self) -> SourceId;

    /// Parsed header metadata, georeference binding info, and pyramid plan.
    fn description(&self) -> &RasterDescription;

    /// Return `true` when this source reads over a high-latency remote
    /// transport (e.g., HTTP Range COGs).
    fn is_remote(&self) -> bool {
        false
    }

    /// Return `true` when `Display` (`Rgba8UnormSrgb`) rendering still requires
    /// computing an initial percentile stretch before tile decode.
    fn needs_display_stretch(&self) -> bool {
        false
    }

    /// Plan coalesced source byte ranges required for pointwise decoding of
    /// `wants` at `level` for stream `spec`.
    fn plan_pointwise_byte_ranges(
        &self,
        spec: TileStreamSpec,
        level: u8,
        wants: &[(u32, u32)],
        budget: &MemoryBudget,
    ) -> Result<Vec<ByteRange>, RasterError> {
        let _ = (spec, level, wants, budget);
        Ok(Vec::new())
    }

    /// Plan coalesced source byte ranges needed to compute the initial display
    /// percentile stretch (empty when already computed or 8-bit identity).
    fn plan_stretch_byte_ranges(&self) -> Result<Vec<ByteRange>, RasterError> {
        Ok(Vec::new())
    }

    /// Plan coalesced source byte ranges needed to compute sampled statistics
    /// for `band` (empty when already cached or unsupported).
    fn plan_band_stats_byte_ranges(&self, band: BandKey) -> Result<Vec<ByteRange>, RasterError> {
        let _ = band;
        Ok(Vec::new())
    }

    /// Default [`TileStreamSpec`] recommended when first opening this raster.
    fn default_stream_spec(&self) -> TileStreamSpec {
        DISPLAY_STREAM_SPEC
    }

    /// Scheduler-facing [`PyramidMeta`] for stream `spec`.
    fn pyramid_meta(&self, spec: TileStreamSpec) -> Result<PyramidMeta, RasterError> {
        if spec == DISPLAY_STREAM_SPEC {
            Ok(PyramidMeta {
                source: self.source_id(),
                spec: DISPLAY_STREAM_SPEC,
                levels: self.description().levels.clone(),
            })
        } else {
            Err(RasterError::Decode(format!(
                "raster source only supports Display RGBA8 stream, not {spec:?}"
            )))
        }
    }

    /// Estimated peak working bytes required by [`Self::produce_pass`] for `spec`.
    fn pass_peak_bytes(&self, spec: TileStreamSpec) -> u64;

    /// Produce pointwise tiles `wants` at `level` for stream `spec`.
    fn produce_direct(
        &self,
        spec: TileStreamSpec,
        level: u8,
        wants: &[(u32, u32)],
        ctx: &DecodeContext<'_>,
        emit: TileEmit<'_>,
    ) -> Result<(), RasterError> {
        let _ = (spec, wants, ctx, emit);
        Err(RasterError::Decode(format!(
            "raster source does not support pointwise random access at level {level}; use sequential pass"
        )))
    }

    /// Run a sequential top-to-bottom pass for stream `spec`, emitting every
    /// pyramid tile via `emit`.
    fn produce_pass(
        &self,
        spec: TileStreamSpec,
        ctx: &DecodeContext<'_>,
        emit: TileEmit<'_>,
    ) -> Result<(), RasterError>;

    /// Produce a single pointwise tile `(level, tx, ty)` for stream `spec` into
    /// a validated [`DecodedTile`].
    fn decode_pointwise_tile(
        &self,
        spec: TileStreamSpec,
        level: u8,
        tx: u32,
        ty: u32,
        ctx: &DecodeContext<'_>,
    ) -> Result<DecodedTile, RasterError> {
        let mut produced: Option<DecodedTile> = None;
        self.produce_direct(
            spec,
            level,
            &[(tx, ty)],
            ctx,
            &mut |_lvl, _x, _y, tw, th, bytes| {
                let tile = DecodedTile::try_new(tw, th, spec.format, bytes)
                    .map_err(|e| RasterError::Decode(e.to_string()))?;
                produced = Some(tile);
                Ok(())
            },
        )?;
        produced.ok_or_else(|| {
            let scheme = self
                .pyramid_meta(spec)
                .map(|m| m.scheme())
                .unwrap_or_else(|_| {
                    let max_level = self.description().levels.len().saturating_sub(1) as u8;
                    veldmap_core::tile::TileScheme::RasterPyramid { max_level }
                });
            RasterError::TileOutOfRange {
                scheme,
                level,
                x: tx,
                y: ty,
            }
        })
    }

    /// Zoom-invariant native-resolution CPU sample readout for `band` and
    /// `format` at `(px, py)`.
    fn sample_band(
        &self,
        band: BandKey,
        format: PixelFormat,
        px: u32,
        py: u32,
        ctx: &DecodeContext<'_>,
    ) -> Result<BandSample, RasterError>;

    /// Compute or return cached sampled physical statistics and histograms for
    /// `band` (`None` if unsupported or all nodata).
    fn band_stats(
        &self,
        band: BandKey,
        ctx: &DecodeContext<'_>,
    ) -> Result<Option<BandStats>, RasterError> {
        let _ = (band, ctx);
        Ok(None)
    }

    /// Optional embedded palette RGBA LUT (e.g., TIFF `ColorMap` tag 320).
    fn color_map(&self) -> Option<std::sync::Arc<[[u8; 4]]>> {
        None
    }
}
