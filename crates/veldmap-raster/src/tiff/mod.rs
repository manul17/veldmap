//! GeoTIFF / BigTIFF / Cloud-Optimized GeoTIFF (COG) reader built over
//! positioned [`veldmap_core::io::ByteSource`].
//!
//! Uses `tiff 0.11.3` strictly for IFD/tag parsing on open (caching bounded
//! per-IFD chunk offset/byte-count/table metadata) and executes all chunk reads
//! through the stateless direct raw-chunk decoder (`raw_decode.rs`), supporting
//! `PlanarConfiguration = 1 & 2`, `Palette` (`ColorMap`), 4-band RGB+NIR
//! (`ExtraSamples = 0`), `YCbCr`, `ZSTD`, `WebP`, `LZW`, `Deflate`, `PackBits`,
//! `JPEG` (`JPEGTables`), `Predictor = 2 & 3`, and typed `Display`, `Scientific`
//! (`R32Float`), and `Categorical` (`R8Uint`/`R16Uint`) streams.

mod chunk_reader;
mod gdal_meta;
mod georef;
mod ifd;
mod open;
mod raw_decode;
mod stretch;
mod tags;
mod ycbcr;

use std::sync::Arc;

use veldmap_core::budget::MemoryBudget;
use veldmap_core::io::ByteSource;
use veldmap_core::tile::{BandKey, PixelFormat, SourceFingerprint, SourceId, TileStreamSpec};

use crate::RasterError;
use crate::pipeline::driver::{self, Chunked as _, DirectRead};
use crate::pipeline::grid::{Grid, chunk_working_bytes};
use crate::pipeline::metadata::{BandSample, BandStats, RasterDescription, SampleType};
use crate::pipeline::plan::{
    DISPLAY_STREAM_SPEC, PyramidMeta, build_chunked_level_table_for_format, stream_spec_for_band,
};
use crate::pipeline::radiometry::{DisplayMapping, is_data_f64, read_native_sample_f64};
use crate::pipeline::{DecodeContext, RasterSource, TileEmit};

use chunk_reader::{ChunkReadMode, TiffChunkReader};
use ifd::{CachedIfd, Photometric, PlanarConfig, TiffByteOrder};
use raw_decode::{RawChunkPayload, decode_raw_chunk};
use stretch::{BandStatsCache, StretchCache, StretchRequest};

/// BigTIFF file header signatures (`version = 43`).
const BIG_MAGIC: [&[u8]; 2] = [b"II\x2b\x00", b"MM\x00\x2b"];

/// Classic TIFF file header signatures (`version = 42`).
const CLASSIC_MAGIC: [&[u8]; 2] = [b"II\x2a\x00", b"MM\x00\x2a"];

/// Maximum decoded bytes allowed for a single base-IFD chunk during interactive
/// cursor readouts (`16 MiB`).
const MAX_CURSOR_CHUNK_BYTES: u64 = 16 * 1024 * 1024;

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
    byte_source: Arc<dyn ByteSource>,
    byte_order: TiffByteOrder,
    fingerprint: SourceFingerprint,
    source_id: SourceId,
    description: RasterDescription,
    tiled: bool,
    grid: Grid,
    ifds: Vec<CachedIfd>,
    nodata_f64: Option<f64>,
    stats_ifd: usize,
    max_chunk_bytes: usize,
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
    pub const fn is_tiled(&self) -> bool {
        self.tiled
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

    fn stretch_request<'a>(&'a self, ctx: &'a DecodeContext<'_>) -> StretchRequest<'a> {
        StretchRequest {
            byte_source: &self.byte_source,
            byte_order: self.byte_order,
            stats_ifd: self.stats_ifd_ref(),
            nodata_f64: self.nodata_f64,
            max_chunk_bytes: self.max_chunk_bytes,
            budget_tracker: ctx.budget_tracker,
            cancel: ctx.cancel,
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
            &self.byte_source,
            self.byte_order,
            self.source_id,
            &self.ifds,
            spec,
            self.nodata_f64,
            display_mapping,
            self.max_chunk_bytes,
            mode,
        )
    }

    fn base_chunk_loc(&self, px: u32, py: u32) -> Result<(u32, u32, u32, u32, u32), RasterError> {
        let base = self.base_ifd();
        let (cw, ch) = (base.chunk_w.max(1), base.chunk_h.max(1));
        let cx = px / cw;
        let cy = py / ch;
        let chunk_index = cy
            .checked_mul(base.chunks_across)
            .and_then(|row| row.checked_add(cx))
            .ok_or_else(|| RasterError::Decode("chunk index overflow".to_owned()))?;
        let (dw, dh) = base.chunk_data_dims(chunk_index).unwrap_or((
            cw.min(self.description.width.saturating_sub(cx * cw)),
            ch.min(self.description.height.saturating_sub(cy * ch)),
        ));
        Ok((chunk_index, px % cw, py % ch, dw, dh))
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

    fn pyramid_meta_for(&self, spec: TileStreamSpec) -> Result<PyramidMeta, RasterError> {
        self.validate_stream_spec(spec)?;
        let levels = if spec.format == PixelFormat::Rgba8UnormSrgb {
            self.description.levels.clone()
        } else {
            build_chunked_level_table_for_format(
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

    fn pass_peak_bytes(&self) -> u64 {
        self.pass_peak_bytes_for(DISPLAY_STREAM_SPEC)
    }

    fn pass_peak_bytes_for(&self, spec: TileStreamSpec) -> u64 {
        self.grid
            .pass_peak(
                (self.description.width, self.description.height),
                spec.format,
            )
            .total()
    }

    fn produce_direct(
        &self,
        level: u8,
        wants: &[(u32, u32)],
        ctx: &DecodeContext<'_>,
        emit: TileEmit<'_>,
    ) -> Result<(), RasterError> {
        self.produce_direct_stream(DISPLAY_STREAM_SPEC, level, wants, ctx, emit)
    }

    fn produce_direct_stream(
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

    fn produce_pass(&self, ctx: &DecodeContext<'_>, emit: TileEmit<'_>) -> Result<(), RasterError> {
        self.produce_pass_stream(DISPLAY_STREAM_SPEC, ctx, emit)
    }

    fn produce_pass_stream(
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

    fn sample_display_rgba(
        &self,
        px: u32,
        py: u32,
        ctx: &DecodeContext<'_>,
    ) -> Result<Option<[u8; 4]>, RasterError> {
        if px >= self.description.width || py >= self.description.height {
            return Ok(None);
        }
        let base = self.base_ifd();
        let (chunk_index, lx, ly, dw, dh) = self.base_chunk_loc(px, py)?;
        let (stored_w, stored_h) = base.chunk_stored_dims(chunk_index).unwrap_or((dw, dh));
        let rgba_bytes = u64::from(dw)
            .saturating_mul(u64::from(dh))
            .saturating_mul(4);
        let rgba_cap = (ctx.budget.decoded_chunk_cache_bytes as u64).min(MAX_CURSOR_CHUNK_BYTES);
        if rgba_bytes > rgba_cap {
            return Err(RasterError::LimitExceeded {
                width: dw,
                height: dh,
                estimated_bytes: rgba_bytes,
                limit_bytes: rgba_cap,
            });
        }
        let working_bytes = chunk_working_bytes(stored_w, stored_h, base.raw_pixel_bytes(), 4);
        if working_bytes > ctx.budget.decode_working_bytes {
            return Err(RasterError::LimitExceeded {
                width: stored_w,
                height: stored_h,
                estimated_bytes: working_bytes,
                limit_bytes: ctx.budget.decode_working_bytes,
            });
        }

        let mapping = self.display_mapping(ctx)?;
        let mut reader = self.make_reader(
            DISPLAY_STREAM_SPEC,
            Some(mapping),
            ChunkReadMode::Pointwise {
                chunk_cache: ctx.chunk_cache,
                budget_tracker: ctx.budget_tracker,
            },
        );
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
        if band == BandKey::Display || format == PixelFormat::Rgba8UnormSrgb {
            return Ok(match self.sample_display_rgba(px, py, ctx)? {
                Some(rgba) => BandSample::Rgba(rgba),
                None => BandSample::Nodata,
            });
        }
        let BandKey::Index(band_u16) = band else {
            return Ok(BandSample::Unavailable);
        };
        let base = self.base_ifd();
        let band_idx = usize::from(band_u16);
        if band_idx >= usize::from(base.samples_per_pixel) {
            return Ok(BandSample::Unavailable);
        }
        let (chunk_index, lx_u32, ly_u32, dw, dh) = self.base_chunk_loc(px, py)?;
        let (stored_w, stored_h) = base.chunk_stored_dims(chunk_index).unwrap_or((dw, dh));
        let raw_depth = match base.planar {
            PlanarConfig::Planar => base.sample_type.byte_size() as u32,
            PlanarConfig::Chunky => base.raw_pixel_bytes(),
        };
        let chunk_raw_bytes = u64::from(stored_w)
            .saturating_mul(u64::from(stored_h))
            .saturating_mul(u64::from(raw_depth));
        let cap = (ctx.budget.decoded_chunk_cache_bytes as u64)
            .min(ctx.budget.decode_working_bytes)
            .min(MAX_CURSOR_CHUNK_BYTES);
        if chunk_raw_bytes > cap {
            return Err(RasterError::LimitExceeded {
                width: stored_w,
                height: stored_h,
                estimated_bytes: chunk_raw_bytes,
                limit_bytes: cap,
            });
        }

        // Reuse `SharedChunkCache` for sample types lossless in `R32Float` /
        // `R8Uint` / `R16Uint` so hover probes avoid re-decompressing chunks on
        // every cursor move. Fall back to direct raw-chunk `f64` reads for
        // `U32`/`I32`/`F64` whose precision exceeds f32's 24-bit mantissa.
        let needs_raw_f64 = format == PixelFormat::R32Float
            && matches!(
                base.sample_type,
                SampleType::U32 | SampleType::I32 | SampleType::F64
            );
        if !needs_raw_f64
            && matches!(
                format,
                PixelFormat::R32Float | PixelFormat::R8Uint | PixelFormat::R16Uint
            )
        {
            let spec = stream_spec_for_band(band, format);
            let mut reader = self.make_reader(
                spec,
                None,
                ChunkReadMode::Pointwise {
                    chunk_cache: ctx.chunk_cache,
                    budget_tracker: ctx.budget_tracker,
                },
            );
            let chunk = reader.chunk(0, chunk_index, ctx.cancel)?;
            if lx_u32 >= chunk.width || ly_u32 >= chunk.height {
                return Ok(BandSample::Unavailable);
            }
            let pix_idx = (ly_u32 as usize) * (chunk.width as usize) + (lx_u32 as usize);
            return match format {
                PixelFormat::R8Uint => {
                    let Some(&v) = chunk.data.get(pix_idx) else {
                        return Ok(BandSample::Unavailable);
                    };
                    if !is_data_f64(f64::from(v), self.nodata_f64) {
                        Ok(BandSample::Nodata)
                    } else {
                        Ok(BandSample::Categorical(u16::from(v)))
                    }
                }
                PixelFormat::R16Uint => {
                    let off = pix_idx * 2;
                    let Some(slice) = chunk.data.get(off..off + 2) else {
                        return Ok(BandSample::Unavailable);
                    };
                    let v = u16::from_ne_bytes([slice[0], slice[1]]);
                    if !is_data_f64(f64::from(v), self.nodata_f64) {
                        Ok(BandSample::Nodata)
                    } else {
                        Ok(BandSample::Categorical(v))
                    }
                }
                PixelFormat::R32Float => {
                    let off = pix_idx * 4;
                    let Some(slice) = chunk.data.get(off..off + 4) else {
                        return Ok(BandSample::Unavailable);
                    };
                    let raw =
                        f64::from(f32::from_ne_bytes([slice[0], slice[1], slice[2], slice[3]]));
                    if !raw.is_finite() {
                        return Ok(BandSample::Nodata);
                    }
                    let (scale, offset) = self
                        .description
                        .band_info(band)
                        .map_or((1.0, 0.0), |info| (info.scale, info.offset));
                    Ok(BandSample::Scalar {
                        raw,
                        physical: raw.mul_add(scale, offset),
                    })
                }
                PixelFormat::Rgba8UnormSrgb => Ok(BandSample::Unavailable),
            };
        }

        let lx = lx_u32 as usize;
        let ly = ly_u32 as usize;
        let _lease = ctx
            .budget_tracker
            .reserve_blocking(chunk_raw_bytes, ctx.cancel)?;
        let plane = match base.planar {
            PlanarConfig::Planar => band_idx,
            PlanarConfig::Chunky => 0,
        };
        let (cdw, cdh, payload) = decode_raw_chunk(
            self.byte_source.as_ref(),
            base,
            self.byte_order,
            plane,
            chunk_index,
            self.max_chunk_bytes,
            ctx.cancel,
        )?;
        let RawChunkPayload::Bytes(data) = payload else {
            return Ok(BandSample::Nodata);
        };
        if lx >= cdw as usize || ly >= cdh as usize {
            return Ok(BandSample::Unavailable);
        }
        let (stride, ch_off) = match base.planar {
            PlanarConfig::Planar => (1usize, 0usize),
            PlanarConfig::Chunky => (usize::from(base.samples_per_pixel), band_idx),
        };
        let sample_idx = (ly * (cdw as usize) + lx) * stride + ch_off;
        let Some(raw) = read_native_sample_f64(&data, base.sample_type, sample_idx) else {
            return Ok(BandSample::Unavailable);
        };
        if !is_data_f64(raw, self.nodata_f64) {
            return Ok(BandSample::Nodata);
        }
        let (scale, offset) = self
            .description
            .band_info(band)
            .map_or((1.0, 0.0), |info| (info.scale, info.offset));
        Ok(BandSample::Scalar {
            raw,
            physical: raw.mul_add(scale, offset),
        })
    }

    fn band_stats(
        &self,
        band: BandKey,
        ctx: &DecodeContext<'_>,
    ) -> Result<Option<BandStats>, RasterError> {
        let BandKey::Index(b) = band else {
            return Ok(None);
        };
        let (scale, offset) = self
            .description
            .band_info(band)
            .map_or((1.0, 0.0), |info| (info.scale, info.offset));
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
