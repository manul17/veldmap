//! GeoTIFF / BigTIFF / Cloud-Optimized GeoTIFF (COG) reader built over
//! positioned [`veldmap_core::io::ByteSource`].
//!
//! Uses `tiff 0.11.3` strictly for IFD/tag parsing on open (caching bounded
//! per-IFD chunk offset/byte-count/table metadata) and executes all chunk reads
//! through the stateless direct raw-chunk decoder (`raw_decode/`), supporting
//! `PlanarConfiguration = 1 & 2`, `Palette` (`ColorMap`), 4-band RGB+NIR
//! (`ExtraSamples = 0`), `YCbCr`, `ZSTD`, `WebP`, `LZW`, `Deflate`, `PackBits`,
//! `JPEG` (`JPEGTables`), `Predictor = 2 & 3`, and typed `Display`, `Scientific`
//! (`R32Float`), and `Categorical` (`R8Uint`/`R16Uint`) streams.

mod chunk_convert;
mod chunk_reader;
mod gdal_meta;
mod georef;
mod ifd;
mod ifd_parse;
mod open;
mod probe;
mod raw_decode;
mod stretch;
mod tags;
mod ycbcr;

use std::sync::Arc;

use veldmap_core::budget::MemoryBudget;
use veldmap_core::tile::{BandKey, PixelFormat, SourceFingerprint, SourceId, TileStreamSpec};

use crate::RasterError;
use crate::pipeline::driver::{self, DirectRead};
use crate::pipeline::grid::Grid;
use crate::pipeline::metadata::{BandSample, BandStats, RasterDescription, SampleType};
use crate::pipeline::plan::{
    DISPLAY_STREAM_SPEC, PyramidMeta, build_chunked_level_table, stream_spec_for_band,
};
use crate::pipeline::radiometry::DisplayMapping;
use crate::pipeline::{DecodeContext, RasterSource, TileEmit};

use chunk_reader::{ChunkReadMode, TiffChunkReader};
use ifd::{CachedIfd, Photometric, RawTiffAccess};
use probe::{ProbeContext, sample_display_rgba, sample_indexed_band};
use stretch::{BandStatsCache, StretchCache, StretchRequest};

/// BigTIFF file header signatures (`version = 43`).
const BIG_MAGIC: [&[u8]; 2] = [b"II\x2b\x00", b"MM\x00\x2b"];

/// Classic TIFF file header signatures (`version = 42`).
const CLASSIC_MAGIC: [&[u8]; 2] = [b"II\x2a\x00", b"MM\x00\x2a"];

/// Maximum number of IFD directories scanned during `open` to guard against
/// malformed or pathological multi-IFD files.
pub const MAX_IFDS: usize = 64;

/// Return whether `head` begins with a classic TIFF or BigTIFF signature.
pub fn has_tiff_magic(head: &[u8]) -> bool {
    CLASSIC_MAGIC.iter().any(|m| head.starts_with(m))
        || BIG_MAGIC.iter().any(|m| head.starts_with(m))
}

/// Thread-safe opened GeoTIFF/BigTIFF/COG source backed by a positioned
/// [`veldmap_core::io::ByteSource`] with cached per-IFD tables and stateless
/// direct raw-chunk decoding.
pub struct TiffSource {
    access: RawTiffAccess,
    fingerprint: SourceFingerprint,
    source_id: SourceId,
    description: RasterDescription,
    grid: Grid,
    ifds: Vec<CachedIfd>,
    nodata_f64: Option<f64>,
    stats_ifd: usize,
    budget_snapshot: MemoryBudget,
    stretch: StretchCache,
    band_stats: BandStatsCache,
}

impl TiffSource {
    /// Return the number of compatible reduced-resolution overviews discovered
    /// during [`Self::open`].
    pub fn overview_count(&self) -> usize {
        self.grid.overviews.len()
    }

    /// Return whether the base IFD uses tiled TIFF storage (`TileWidth`).
    pub fn is_tiled(&self) -> bool {
        self.base_ifd().tiled
    }

    fn base_ifd(&self) -> &CachedIfd {
        &self.ifds[0]
    }

    fn stats_ifd_ref(&self) -> &CachedIfd {
        self.ifds
            .iter()
            .find(|i| i.image == self.stats_ifd)
            .unwrap_or(&self.ifds[0])
    }

    fn band_scale_offset(&self, band: BandKey) -> (f64, f64) {
        self.description
            .band_info(band)
            .map_or((1.0, 0.0), |info| (info.scale, info.offset))
    }

    fn stretch_request<'a>(&'a self, ctx: &'a DecodeContext<'_>) -> StretchRequest<'a> {
        StretchRequest {
            access: &self.access,
            stats_ifd: self.stats_ifd_ref(),
            nodata_f64: self.nodata_f64,
            budget_tracker: ctx.budget_tracker,
            cancel: ctx.cancel,
        }
    }

    fn probe_context(&self) -> ProbeContext<'_> {
        ProbeContext {
            access: &self.access,
            source_id: self.source_id,
            ifds: &self.ifds,
            nodata_f64: self.nodata_f64,
        }
    }

    fn display_mapping(&self, ctx: &DecodeContext<'_>) -> Result<DisplayMapping, RasterError> {
        let req = self.stretch_request(ctx);
        self.stretch.get_or_compute(&req)
    }

    fn validate_stream_spec(&self, spec: TileStreamSpec) -> Result<(), RasterError> {
        if spec.gutter != 0 {
            return Err(RasterError::Decode(format!(
                "TiffSource does not support gutter={} yet",
                spec.gutter
            )));
        }
        let base = self.base_ifd();
        match (spec.band, spec.format) {
            (BandKey::Display, PixelFormat::Rgba8UnormSrgb) => Ok(()),
            (BandKey::Index(b), PixelFormat::R32Float) => {
                if b >= base.samples_per_pixel {
                    return Err(RasterError::Decode(format!(
                        "band {b} out of range ({} samples/pixel)",
                        base.samples_per_pixel
                    )));
                }
                Ok(())
            }
            (BandKey::Index(b), PixelFormat::R8Uint | PixelFormat::R16Uint) => {
                if b >= base.samples_per_pixel {
                    return Err(RasterError::Decode(format!(
                        "band {b} out of range ({} samples/pixel)",
                        base.samples_per_pixel
                    )));
                }
                match (spec.format, base.sample_type) {
                    (PixelFormat::R8Uint, SampleType::U8)
                    | (PixelFormat::R16Uint, SampleType::U8 | SampleType::U16) => Ok(()),
                    _ => Err(RasterError::Decode(format!(
                        "categorical format {:?} requires U8/U16 source samples, got {:?}",
                        spec.format, base.sample_type
                    ))),
                }
            }
            _ => Err(RasterError::Decode(format!(
                "unsupported TIFF stream spec {spec:?}"
            ))),
        }
    }

    fn make_reader<'a>(
        &'a self,
        spec: TileStreamSpec,
        display_mapping: Option<DisplayMapping>,
        mode: ChunkReadMode<'a>,
    ) -> TiffChunkReader<'a> {
        TiffChunkReader::new(
            &self.access,
            self.source_id,
            &self.ifds,
            spec,
            self.nodata_f64,
            display_mapping,
            mode,
        )
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

    fn default_stream_spec(&self) -> TileStreamSpec {
        let base = self.base_ifd();
        if base.photometric == Photometric::Palette {
            let fmt = base
                .sample_type
                .categorical_pixel_format()
                .unwrap_or(PixelFormat::R8Uint);
            return stream_spec_for_band(BandKey::Index(0), fmt);
        }
        // Single-band numeric >8-bit or float rasters default to the Scientific
        // R32Float track for lossless live stretch, colormaps, and histograms.
        if base.data_bands == 1
            && base.alpha_channel().is_none()
            && !matches!(base.sample_type, SampleType::U8)
        {
            return stream_spec_for_band(BandKey::Index(0), PixelFormat::R32Float);
        }
        DISPLAY_STREAM_SPEC
    }

    fn pyramid_meta(&self, spec: TileStreamSpec) -> Result<PyramidMeta, RasterError> {
        self.validate_stream_spec(spec)?;
        let levels = if spec.format == PixelFormat::Rgba8UnormSrgb {
            self.description.levels.clone()
        } else {
            build_chunked_level_table(
                &self.grid,
                self.description.width,
                self.description.height,
                spec.format,
                &self.budget_snapshot,
            )
        };
        Ok(PyramidMeta {
            source: self.source_id,
            spec,
            levels,
        })
    }

    fn pass_peak_bytes(&self, spec: TileStreamSpec) -> u64 {
        self.grid
            .pass_peak(
                (self.description.width, self.description.height),
                spec.format,
            )
            .total()
    }

    fn produce_direct(
        &self,
        spec: TileStreamSpec,
        level: u8,
        wants: &[(u32, u32)],
        ctx: &DecodeContext<'_>,
        emit: TileEmit<'_>,
    ) -> Result<(), RasterError> {
        self.validate_stream_spec(spec)?;
        let mapping = if spec.format == PixelFormat::Rgba8UnormSrgb {
            Some(self.display_mapping(ctx)?)
        } else {
            None
        };
        let mut reader = self.make_reader(
            spec,
            mapping,
            ChunkReadMode::Pointwise {
                chunk_cache: ctx.chunk_cache,
                budget_tracker: ctx.budget_tracker,
            },
        );
        let req = DirectRead {
            grid: &self.grid,
            base: (self.description.width, self.description.height),
            format: spec.format,
            level,
            wants,
            budget: ctx.budget,
            budget_tracker: ctx.budget_tracker,
            cancel: ctx.cancel,
        };
        driver::direct(&mut reader, &req, emit)
    }

    fn produce_pass(
        &self,
        spec: TileStreamSpec,
        ctx: &DecodeContext<'_>,
        emit: TileEmit<'_>,
    ) -> Result<(), RasterError> {
        self.validate_stream_spec(spec)?;
        let mapping = if spec.format == PixelFormat::Rgba8UnormSrgb {
            Some(self.display_mapping(ctx)?)
        } else {
            None
        };
        let pass_peak = self.grid.pass_peak(
            (self.description.width, self.description.height),
            spec.format,
        );
        pass_peak.admit(ctx.budget.decode_working_bytes)?;
        let _pass_lease = ctx
            .budget_tracker
            .reserve_blocking(pass_peak.total(), ctx.cancel)?;

        let mut reader = self.make_reader(spec, mapping, ChunkReadMode::SequentialPass);
        driver::pass(
            &mut reader,
            &self.grid,
            (self.description.width, self.description.height),
            spec.format,
            ctx.cancel,
            emit,
        )
    }

    fn sample_band(
        &self,
        band: BandKey,
        format: PixelFormat,
        px: u32,
        py: u32,
        ctx: &DecodeContext<'_>,
    ) -> Result<BandSample, RasterError> {
        if px >= self.description.width || py >= self.description.height {
            return Ok(BandSample::Unavailable);
        }
        let probe = self.probe_context();
        if band == BandKey::Display && format == PixelFormat::Rgba8UnormSrgb {
            return sample_display_rgba(&probe, px, py, || self.display_mapping(ctx), ctx);
        }
        let spec = stream_spec_for_band(band, format);
        if self.validate_stream_spec(spec).is_err() {
            return Ok(BandSample::Unavailable);
        }
        let BandKey::Index(band_u16) = band else {
            return Ok(BandSample::Unavailable);
        };
        sample_indexed_band(
            &probe,
            spec,
            usize::from(band_u16),
            self.band_scale_offset(band),
            px,
            py,
            ctx,
        )
    }

    fn band_stats(
        &self,
        band: BandKey,
        ctx: &DecodeContext<'_>,
    ) -> Result<Option<BandStats>, RasterError> {
        let BandKey::Index(b) = band else {
            return Ok(None);
        };
        let (scale, offset) = self.band_scale_offset(band);
        let req = self.stretch_request(ctx);
        self.band_stats.get_or_compute(b, scale, offset, &req)
    }

    fn color_map(&self) -> Option<Arc<[[u8; 4]]>> {
        self.base_ifd().color_map.clone()
    }
}

#[cfg(test)]
#[path = "tiff_tests.rs"]
mod tests;
