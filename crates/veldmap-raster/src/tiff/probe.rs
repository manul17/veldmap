//! Zoom-invariant base-IFD cursor pixel probing for GeoTIFF sources.
//!
//! Strictly unidirectional helper module above [`super::ifd`],
//! [`super::chunk_reader`], and [`super::raw_decode`]. Reuses
//! `SharedChunkCache` for sample types lossless in `Rgba8UnormSrgb`,
//! `R32Float`, `R8Uint`, and `R16Uint`, while falling back to direct raw-chunk
//! `f64` reads for `U32`/`I32`/`F64` rasters whose precision exceeds `f32`'s
//! 24-bit mantissa.

use veldmap_core::tile::{PixelFormat, SourceId, TileStreamSpec};

use crate::RasterError;
use crate::pipeline::DecodeContext;
use crate::pipeline::driver::Chunked as _;
use crate::pipeline::grid::chunk_working_bytes;
use crate::pipeline::metadata::{BandSample, SampleType};
use crate::pipeline::plan::DISPLAY_STREAM_SPEC;
use crate::pipeline::radiometry::{DisplayMapping, is_data_f64, read_native_sample_f64};

use super::chunk_reader::{ChunkReadMode, TiffChunkReader};
use super::ifd::{CachedIfd, ChunkDims, ChunkPixelLoc, RawTiffAccess};
use super::raw_decode::{RawChunkPayload, decode_raw_chunk};

/// Maximum decoded bytes allowed for a single base-IFD chunk during interactive
/// cursor readouts (`16 MiB`).
const MAX_CURSOR_CHUNK_BYTES: u64 = 16 * 1024 * 1024;

/// Borrowed base-IFD context passed into cursor probing helpers.
pub(super) struct ProbeContext<'a> {
    pub(super) access: &'a RawTiffAccess,
    pub(super) source_id: SourceId,
    pub(super) ifds: &'a [CachedIfd],
    pub(super) nodata_f64: Option<f64>,
}

impl<'a> ProbeContext<'a> {
    fn base_ifd(&self) -> &'a CachedIfd {
        &self.ifds[0]
    }

    fn make_pointwise_reader(
        &self,
        spec: TileStreamSpec,
        display_mapping: Option<DisplayMapping>,
        ctx: &'a DecodeContext<'a>,
    ) -> TiffChunkReader<'a> {
        TiffChunkReader::new(
            self.access,
            self.source_id,
            self.ifds,
            spec,
            self.nodata_f64,
            display_mapping,
            ChunkReadMode::Pointwise {
                chunk_cache: ctx.chunk_cache,
                budget_tracker: ctx.budget_tracker,
            },
        )
    }
}

fn admit_cursor_chunk_bytes(
    ctx: &DecodeContext<'_>,
    dims: ChunkDims,
    raw_depth: u32,
    out_bpp: u32,
) -> Result<u64, RasterError> {
    let chunk_cap = (ctx.budget.decoded_chunk_cache_bytes as u64).min(MAX_CURSOR_CHUNK_BYTES);
    if out_bpp > 0 {
        let out_bytes = u64::from(dims.data_w)
            .saturating_mul(u64::from(dims.data_h))
            .saturating_mul(u64::from(out_bpp));
        if out_bytes > chunk_cap {
            return Err(RasterError::LimitExceeded {
                width: dims.data_w,
                height: dims.data_h,
                estimated_bytes: out_bytes,
                limit_bytes: chunk_cap,
            });
        }
        let working_bytes = chunk_working_bytes(dims.stored_w, dims.stored_h, raw_depth, out_bpp);
        if working_bytes > ctx.budget.decode_working_bytes {
            return Err(RasterError::LimitExceeded {
                width: dims.stored_w,
                height: dims.stored_h,
                estimated_bytes: working_bytes,
                limit_bytes: ctx.budget.decode_working_bytes,
            });
        }
        Ok(working_bytes)
    } else {
        let raw_bytes = u64::from(dims.stored_w)
            .saturating_mul(u64::from(dims.stored_h))
            .saturating_mul(u64::from(raw_depth));
        let cap = chunk_cap.min(ctx.budget.decode_working_bytes);
        if raw_bytes > cap {
            return Err(RasterError::LimitExceeded {
                width: dims.stored_w,
                height: dims.stored_h,
                estimated_bytes: raw_bytes,
                limit_bytes: cap,
            });
        }
        Ok(raw_bytes)
    }
}

fn scalar_sample(raw: f64, nodata: Option<f64>, scale: f64, offset: f64) -> BandSample {
    if !is_data_f64(raw, nodata) {
        return BandSample::Nodata;
    }
    BandSample::Scalar {
        raw,
        physical: raw.mul_add(scale, offset),
    }
}

/// Sample one rendered display RGBA8 pixel at `(px, py)` from the base IFD.
///
/// Evaluates `get_mapping` lazily only after cursor chunk budget admission
/// succeeds, preventing stretch I/O on probes rejected by `LimitExceeded`.
pub(super) fn sample_display_rgba(
    probe: &ProbeContext<'_>,
    px: u32,
    py: u32,
    get_mapping: impl FnOnce() -> Result<DisplayMapping, RasterError>,
    ctx: &DecodeContext<'_>,
) -> Result<BandSample, RasterError> {
    let base = probe.base_ifd();
    let loc = base.chunk_loc(px, py)?;
    admit_cursor_chunk_bytes(ctx, loc.dims, base.raw_pixel_bytes(), 4)?;
    let mapping = get_mapping()?;

    let mut reader = probe.make_pointwise_reader(DISPLAY_STREAM_SPEC, Some(mapping), ctx);
    let chunk = reader.chunk(0, loc.chunk_index, ctx.cancel)?;
    if loc.local_x >= chunk.width || loc.local_y >= chunk.height {
        return Ok(BandSample::Nodata);
    }
    let idx = (loc.local_y as usize)
        .saturating_mul(chunk.width as usize)
        .saturating_add(loc.local_x as usize)
        .saturating_mul(4);
    let Some(px_bytes) = chunk.data.get(idx..idx + 4) else {
        return Ok(BandSample::Nodata);
    };
    let rgba = [px_bytes[0], px_bytes[1], px_bytes[2], px_bytes[3]];
    Ok(if rgba[3] > 0 {
        BandSample::Rgba(rgba)
    } else {
        BandSample::Nodata
    })
}

/// Sample one scientific or categorical band pixel at `(px, py)` from the base
/// IFD, using `SharedChunkCache` whenever lossless and falling back to direct
/// native `f64` reads for `U32`/`I32`/`F64` scientific rasters.
pub(super) fn sample_indexed_band(
    probe: &ProbeContext<'_>,
    spec: TileStreamSpec,
    band_idx: usize,
    scale_offset: (f64, f64),
    px: u32,
    py: u32,
    ctx: &DecodeContext<'_>,
) -> Result<BandSample, RasterError> {
    let base = probe.base_ifd();
    let loc = base.chunk_loc(px, py)?;
    let needs_raw_f64 = spec.format == PixelFormat::R32Float
        && matches!(
            base.sample_type,
            SampleType::U32 | SampleType::I32 | SampleType::F64
        );
    if needs_raw_f64 {
        sample_raw_f64_band(probe, base, band_idx, scale_offset, loc, ctx)
    } else {
        sample_cached_band(probe, base, spec, scale_offset, loc, ctx)
    }
}

fn sample_cached_band(
    probe: &ProbeContext<'_>,
    base: &CachedIfd,
    spec: TileStreamSpec,
    (scale, offset): (f64, f64),
    loc: ChunkPixelLoc,
    ctx: &DecodeContext<'_>,
) -> Result<BandSample, RasterError> {
    admit_cursor_chunk_bytes(
        ctx,
        loc.dims,
        base.band_raw_pixel_bytes(),
        spec.format.bytes_per_pixel() as u32,
    )?;
    let mut reader = probe.make_pointwise_reader(spec, None, ctx);
    let chunk = reader.chunk(0, loc.chunk_index, ctx.cancel)?;
    if loc.local_x >= chunk.width || loc.local_y >= chunk.height {
        return Ok(BandSample::Unavailable);
    }
    let pix_idx = (loc.local_y as usize) * (chunk.width as usize) + (loc.local_x as usize);
    match spec.format {
        PixelFormat::R8Uint => {
            let Some(&v) = chunk.data.get(pix_idx) else {
                return Ok(BandSample::Unavailable);
            };
            if !is_data_f64(f64::from(v), probe.nodata_f64) {
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
            if !is_data_f64(f64::from(v), probe.nodata_f64) {
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
            let raw = f64::from(f32::from_ne_bytes([slice[0], slice[1], slice[2], slice[3]]));
            // Cached R32Float chunks already map declared nodata to `f32::NAN`,
            // so pass `nodata = None` (checking finiteness only).
            Ok(scalar_sample(raw, None, scale, offset))
        }
        PixelFormat::Rgba8UnormSrgb => Ok(BandSample::Unavailable),
    }
}

fn sample_raw_f64_band(
    probe: &ProbeContext<'_>,
    base: &CachedIfd,
    band_idx: usize,
    (scale, offset): (f64, f64),
    loc: ChunkPixelLoc,
    ctx: &DecodeContext<'_>,
) -> Result<BandSample, RasterError> {
    let chunk_raw_bytes = admit_cursor_chunk_bytes(ctx, loc.dims, base.band_raw_pixel_bytes(), 0)?;
    let _lease = ctx
        .budget_tracker
        .reserve_blocking(chunk_raw_bytes, ctx.cancel)?;
    let (cdw, cdh, payload) = decode_raw_chunk(
        probe.access,
        base,
        base.band_plane(band_idx),
        loc.chunk_index,
        ctx.cancel,
    )?;
    let RawChunkPayload::Bytes(data) = payload else {
        return Ok(BandSample::Nodata);
    };
    let lx = loc.local_x as usize;
    let ly = loc.local_y as usize;
    if lx >= cdw as usize || ly >= cdh as usize {
        return Ok(BandSample::Unavailable);
    }
    let (stride, ch_off) = base.band_stride_offset(band_idx);
    let sample_idx = (ly * (cdw as usize) + lx) * stride + ch_off;
    let Some(raw) = read_native_sample_f64(&data, base.sample_type, sample_idx) else {
        return Ok(BandSample::Unavailable);
    };
    Ok(scalar_sample(raw, probe.nodata_f64, scale, offset))
}
