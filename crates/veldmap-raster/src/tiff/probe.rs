//! Zoom-invariant base-IFD cursor pixel probing for GeoTIFF sources.
//!
//! Strictly unidirectional helper module above [`super::ifd`],
//! [`super::chunk_reader`], and [`super::raw_decode`]. Reuses
//! `SharedChunkCache` for sample types lossless in `Rgba8UnormSrgb`,
//! `R32Float`, `R8Uint`, and `R16Uint`, while falling back to direct raw-chunk
//! `f64` reads for `U32`/`I32`/`F64` rasters whose precision exceeds `f32`'s
//! 24-bit mantissa.

use std::sync::Arc;

use veldmap_core::tile::{PixelFormat, TileStreamSpec};

use crate::RasterError;
use crate::pipeline::{
    BandSample, ChunkKey, Chunked as _, DISPLAY_STREAM_SPEC, DecodeContext, DecodedChunk,
    DisplayMapping, SampleType, chunk_working_bytes, is_data_f64, read_native_sample_f64,
};

use super::chunk_reader::{ChunkReadMode, TiffChunkReader};
use super::ifd::{CachedIfd, ChunkDims, ChunkPixelLoc, TiffSourceView};
use super::raw_decode::{RawChunkPayload, decode_raw_chunk};

/// Maximum decoded bytes allowed for a single base-IFD chunk during interactive
/// cursor readouts (`16 MiB`).
const MAX_CURSOR_CHUNK_BYTES: u64 = 16 * 1024 * 1024;

impl TiffSourceView<'_> {
    /// Fetch base-IFD chunk `chunk_index` for `spec`, staying strictly
    /// cache-only for remote sources and evaluating `get_mapping` lazily only
    /// on local decode paths.
    fn fetch_base_chunk(
        &self,
        spec: TileStreamSpec,
        chunk_index: u32,
        get_mapping: impl FnOnce() -> Result<Option<DisplayMapping>, RasterError>,
        ctx: &DecodeContext<'_>,
    ) -> Result<Option<Arc<DecodedChunk>>, RasterError> {
        if self.access.byte_source.is_remote() {
            let key = ChunkKey {
                source: self.source_id,
                spec,
                image: 0,
                chunk_index,
            };
            return Ok(ctx.chunk_cache.get_cached(&key));
        }
        let display_mapping = get_mapping()?;
        let mut reader = TiffChunkReader::new(
            *self,
            spec,
            display_mapping,
            ChunkReadMode::Pointwise {
                chunk_cache: ctx.chunk_cache,
                budget_tracker: ctx.budget_tracker,
            },
        );
        reader.chunk(0, chunk_index, ctx.cancel).map(Some)
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

fn categorical_sample(value: u16, nodata: Option<f64>) -> BandSample {
    if !is_data_f64(f64::from(value), nodata) {
        BandSample::Nodata
    } else {
        BandSample::Categorical(value)
    }
}

/// Sample one rendered display RGBA8 pixel at `(px, py)` from the base IFD.
///
/// Evaluates `get_mapping` lazily only after cursor chunk budget admission
/// succeeds, preventing stretch I/O on probes rejected by `LimitExceeded`.
pub(super) fn sample_display_rgba(
    view: TiffSourceView<'_>,
    px: u32,
    py: u32,
    get_mapping: impl FnOnce() -> Result<DisplayMapping, RasterError>,
    ctx: &DecodeContext<'_>,
) -> Result<BandSample, RasterError> {
    let base = view.base_ifd();
    let loc = base.chunk_loc(px, py)?;
    admit_cursor_chunk_bytes(ctx, loc.dims, base.raw_pixel_bytes(), 4)?;

    let Some(chunk) = view.fetch_base_chunk(
        DISPLAY_STREAM_SPEC,
        loc.chunk_index,
        || get_mapping().map(Some),
        ctx,
    )?
    else {
        return Ok(BandSample::Unavailable);
    };
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
/// native `f64` reads for local `U32`/`I32`/`F64` scientific rasters.
///
/// Remote sources stay strictly cache-only so interactive hover probes never
/// block the fast decode pool on uncached HTTP Range reads.
pub(super) fn sample_indexed_band(
    view: TiffSourceView<'_>,
    spec: TileStreamSpec,
    band_idx: usize,
    scale_offset: (f64, f64),
    px: u32,
    py: u32,
    ctx: &DecodeContext<'_>,
) -> Result<BandSample, RasterError> {
    let base = view.base_ifd();
    let loc = base.chunk_loc(px, py)?;
    let is_remote = view.access.byte_source.is_remote();
    let needs_raw_f64 = !is_remote
        && spec.format == PixelFormat::R32Float
        && matches!(
            base.sample_type,
            SampleType::U32 | SampleType::I32 | SampleType::F64
        );
    if needs_raw_f64 {
        sample_raw_f64_band(view, base, band_idx, scale_offset, loc, ctx)
    } else {
        sample_cached_band(view, base, spec, scale_offset, loc, ctx)
    }
}

fn sample_cached_band(
    view: TiffSourceView<'_>,
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
    let Some(chunk) = view.fetch_base_chunk(spec, loc.chunk_index, || Ok(None), ctx)? else {
        return Ok(BandSample::Unavailable);
    };
    if loc.local_x >= chunk.width || loc.local_y >= chunk.height {
        return Ok(BandSample::Unavailable);
    }
    let pix_idx = (loc.local_y as usize) * (chunk.width as usize) + (loc.local_x as usize);
    match spec.format {
        PixelFormat::R8Uint => {
            let Some(&v) = chunk.data.get(pix_idx) else {
                return Ok(BandSample::Unavailable);
            };
            Ok(categorical_sample(u16::from(v), view.nodata_f64))
        }
        PixelFormat::R16Uint => {
            let off = pix_idx * 2;
            let Some(slice) = chunk.data.get(off..off + 2) else {
                return Ok(BandSample::Unavailable);
            };
            let v = u16::from_ne_bytes([slice[0], slice[1]]);
            Ok(categorical_sample(v, view.nodata_f64))
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
    view: TiffSourceView<'_>,
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
        view.access,
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
    Ok(scalar_sample(raw, view.nodata_f64, scale, offset))
}
