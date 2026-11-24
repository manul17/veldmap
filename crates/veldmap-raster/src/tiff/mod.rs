//! GeoTIFF / BigTIFF / Cloud-Optimized GeoTIFF (COG) reader built over
//! positioned [`veldmap_core::io::ByteSource`] with explicit
//! `tiff::decoder::Limits`, `MAX_IFDS` bounds,
//! `GeoDoubleParamsTag`/`GeoAsciiParamsTag` parsing, ZSTD & `Predictor = 3`
//! (via `tiff`) support, pooled `Decoder` handles, and singleflight chunk
//! caching.

mod chunk_reader;
mod georef;
mod open;
mod stretch;
mod tags;

use veldmap_core::tile::{
    BandKey, PixelFormat, RASTER_TILE_SIZE, SourceFingerprint, SourceId, TileStreamSpec,
};

use crate::RasterError;
use crate::pipeline::driver::{self, Chunked as _, DirectRead};
use crate::pipeline::grid::{Grid, chunk_working_bytes};
use crate::pipeline::metadata::{RasterDescription, SampleType};
use crate::pipeline::plan::PyramidMeta;
use crate::pipeline::radiometry::{DisplayMapping, PixelLayout};
use crate::pipeline::{DecodeContext, RasterSource, TileEmit};

use chunk_reader::{ChunkReadMode, DecoderPool, TiffChunkReader};
use stretch::{StretchCache, StretchRequest};

/// BigTIFF file header signatures (`version = 43`).
const BIG_MAGIC: [&[u8]; 2] = [b"II\x2b\x00", b"MM\x00\x2b"];

/// Classic TIFF file header signatures (`version = 42`).
const CLASSIC_MAGIC: [&[u8]; 2] = [b"II\x2a\x00", b"MM\x00\x2a"];

/// Maximum decoded RGBA8 bytes allowed for a single base-IFD chunk during
/// interactive cursor readouts (`16 MiB`, allowing up to `2048 x 2048` COG
/// chunks and normal strips while refusing giant single-strip TIFFs on
/// `fast_pool`).
const MAX_CURSOR_CHUNK_RGBA_BYTES: u64 = 16 * 1024 * 1024;

/// Maximum number of IFD directories scanned during `open` to guard against
/// malformed or pathological multi-IFD files.
pub const MAX_IFDS: usize = 64;

/// Maximum raster side dimension accepted for untiled/stripped TIFF sources.
///
/// Tiled TIFFs/COGs may exceed this limit provided their chunk grid and at
/// least one pyramid level fit within [`veldmap_core::budget::MemoryBudget`].
pub const MAX_SOURCE_SIDE: u32 = 65_536;

/// Current display decode revision encoded into `TileKey` and `ChunkKey`.
const TIFF_DECODE_REV: u32 = 1;

/// Canonical display stream spec shared across [`RasterSource::pyramid_meta`]
/// and [`TiffChunkReader`].
pub(super) const DISPLAY_STREAM_SPEC: TileStreamSpec = TileStreamSpec::new(
    BandKey::Display,
    PixelFormat::Rgba8UnormSrgb,
    0,
    TIFF_DECODE_REV,
);

/// Return whether `head` begins with a classic TIFF or BigTIFF signature.
pub fn has_tiff_magic(head: &[u8]) -> bool {
    CLASSIC_MAGIC.iter().any(|m| head.starts_with(m))
        || BIG_MAGIC.iter().any(|m| head.starts_with(m))
}

/// Compact sample-layout descriptor shared across TIFF overview scanning,
/// chunk reading, and lazy display-mapping stretch computation.
#[derive(Clone, Copy, Debug)]
pub(in crate::tiff) struct TiffFormat {
    pub(super) pixel_layout: PixelLayout,
    pub(super) sample_bits: u8,
    pub(super) sample_type: SampleType,
    pub(super) depth: u32,
    pub(super) nodata: Option<f32>,
}

/// Thread-safe opened GeoTIFF/BigTIFF/COG source backed by a positioned
/// [`veldmap_core::io::ByteSource`] with pooled `tiff::decoder::Decoder`
/// instances.
pub struct TiffSource {
    decoder_pool: DecoderPool,
    fingerprint: SourceFingerprint,
    source_id: SourceId,
    description: RasterDescription,
    grid: Grid,
    format: TiffFormat,
    ifd_ycbcr: Vec<(usize, tags::YCbCrParams)>,
    stats_ifd: usize,
    stretch: StretchCache,
}

impl TiffSource {
    /// Return the number of compatible reduced-resolution overviews discovered
    /// during [`Self::open`].
    pub fn overview_count(&self) -> usize {
        self.grid.overviews.len()
    }

    /// Return whether the base IFD uses tiled TIFF storage (`TileWidth`).
    pub const fn is_tiled(&self) -> bool {
        self.grid.tiled
    }

    fn display_mapping(&self, ctx: &DecodeContext<'_>) -> Result<DisplayMapping, RasterError> {
        self.stretch.get_or_compute(&StretchRequest {
            decoder_pool: &self.decoder_pool,
            format: self.format,
            stats_ifd: self.stats_ifd,
            budget_tracker: ctx.budget_tracker,
            cancel: ctx.cancel,
        })
    }

    fn pointwise_reader<'a>(
        &'a self,
        ctx: &'a DecodeContext<'a>,
    ) -> Result<TiffChunkReader<'a>, RasterError> {
        let mapping = self.display_mapping(ctx)?;
        Ok(TiffChunkReader::new(
            &self.decoder_pool,
            self.source_id,
            self.format,
            &self.ifd_ycbcr,
            mapping,
            ChunkReadMode::Pointwise {
                chunk_cache: ctx.chunk_cache,
                budget_tracker: ctx.budget_tracker,
            },
        ))
    }
}

impl RasterSource for TiffSource {
    fn fingerprint(&self) -> SourceFingerprint {
        self.fingerprint
    }

    fn source_id(&self) -> SourceId {
        self.source_id
    }

    fn description(&self) -> &RasterDescription {
        &self.description
    }

    fn pyramid_meta(&self) -> PyramidMeta {
        PyramidMeta {
            source: self.source_id,
            spec: DISPLAY_STREAM_SPEC,
            width: self.description.width,
            height: self.description.height,
            tile_size: RASTER_TILE_SIZE,
            levels: self.description.levels.clone(),
        }
    }

    fn pass_peak_bytes(&self) -> u64 {
        self.grid
            .pass_peak((self.description.width, self.description.height))
            .total()
    }

    fn produce_direct(
        &self,
        level: u8,
        wants: &[(u32, u32)],
        ctx: &DecodeContext<'_>,
        emit: TileEmit<'_>,
    ) -> Result<(), RasterError> {
        let mut reader = self.pointwise_reader(ctx)?;
        let req = DirectRead {
            grid: &self.grid,
            base: (self.description.width, self.description.height),
            level,
            wants,
            budget: ctx.budget,
            budget_tracker: ctx.budget_tracker,
            cancel: ctx.cancel,
        };
        driver::direct(&mut reader, &req, emit)
    }

    fn produce_pass(&self, ctx: &DecodeContext<'_>, emit: TileEmit<'_>) -> Result<(), RasterError> {
        let mapping = self.display_mapping(ctx)?;
        let pass_peak = self
            .grid
            .pass_peak((self.description.width, self.description.height));
        pass_peak.admit(ctx.budget.decode_working_bytes)?;
        let _pass_lease = ctx
            .budget_tracker
            .reserve_blocking(pass_peak.total(), ctx.cancel)?;

        // Sequential passes read base-IFD chunks once top-to-bottom without
        // entering `SharedChunkCache` singleflight, avoiding both AB-BA lock
        // inversion against `_pass_lease` and pointwise chunk-cache pollution.
        let mut reader = TiffChunkReader::new(
            &self.decoder_pool,
            self.source_id,
            self.format,
            &self.ifd_ycbcr,
            mapping,
            ChunkReadMode::SequentialPass,
        );
        driver::pass(
            &mut reader,
            &self.grid,
            (self.description.width, self.description.height),
            ctx.cancel,
            emit,
        )
    }

    fn sample_display_rgba(
        &self,
        px: u32,
        py: u32,
        ctx: &DecodeContext<'_>,
    ) -> Result<Option<[u8; 4]>, RasterError> {
        if px >= self.description.width || py >= self.description.height {
            return Ok(None);
        }
        let (cw, ch) = (self.grid.chunk.0.max(1), self.grid.chunk.1.max(1));
        let across = self.description.width.div_ceil(cw);
        let cx = px / cw;
        let cy = py / ch;
        let Some(chunk_index) = cy.checked_mul(across).and_then(|base| base.checked_add(cx)) else {
            return Err(RasterError::Decode("chunk index overflow".to_owned()));
        };
        let lx = px % cw;
        let ly = py % ch;

        // Guard against giant untiled single-strip TIFFs BEFORE computing
        // stretch stats or reading IFD 0 on `fast_pool`.
        let dw = cw.min(self.description.width - cx * cw);
        let dh = ch.min(self.description.height - cy * ch);
        let rgba_bytes = u64::from(dw)
            .saturating_mul(u64::from(dh))
            .saturating_mul(4);
        let rgba_cap =
            (ctx.budget.decoded_chunk_cache_bytes as u64).min(MAX_CURSOR_CHUNK_RGBA_BYTES);
        if rgba_bytes > rgba_cap {
            return Err(RasterError::LimitExceeded {
                width: dw,
                height: dh,
                estimated_bytes: rgba_bytes,
                limit_bytes: rgba_cap,
            });
        }
        let working_bytes = chunk_working_bytes(dw, dh, self.format.depth, true);
        if working_bytes > ctx.budget.decode_working_bytes {
            return Err(RasterError::LimitExceeded {
                width: dw,
                height: dh,
                estimated_bytes: working_bytes,
                limit_bytes: ctx.budget.decode_working_bytes,
            });
        }

        let mut reader = self.pointwise_reader(ctx)?;
        let chunk = reader.chunk(0, chunk_index, ctx.cancel)?;
        if lx >= chunk.width || ly >= chunk.height {
            return Ok(None);
        }
        let idx = (ly as usize)
            .saturating_mul(chunk.width as usize)
            .saturating_add(lx as usize)
            .saturating_mul(4);
        let Some(px_bytes) = chunk.data.get(idx..idx + 4) else {
            return Ok(None);
        };
        let rgba = [px_bytes[0], px_bytes[1], px_bytes[2], px_bytes[3]];
        Ok((rgba[3] > 0).then_some(rgba))
    }
}

#[cfg(test)]
#[path = "tiff_tests.rs"]
mod tests;
