//! Zoom-invariant base-IFD cursor pixel probing for [`TiffSource`].
//!
//! Reuses `SharedChunkCache` for sample types lossless in `Rgba8UnormSrgb`,
//! `R32Float`, `R8Uint`, and `R16Uint`, while falling back to direct raw-chunk
//! `f64` reads for `U32`/`I32`/`F64` rasters whose precision exceeds `f32`'s
//! 24-bit mantissa.

use veldmap_core::tile::{BandKey, PixelFormat};

use crate::RasterError;
use crate::pipeline::DecodeContext;
use crate::pipeline::driver::Chunked as _;
use crate::pipeline::grid::chunk_working_bytes;
use crate::pipeline::metadata::{BandSample, SampleType};
use crate::pipeline::plan::{DISPLAY_STREAM_SPEC, stream_spec_for_band};
use crate::pipeline::radiometry::{is_data_f64, read_native_sample_f64};

use super::TiffSource;
use super::chunk_reader::ChunkReadMode;
use super::raw_decode::{RawChunkPayload, decode_raw_chunk};

/// Maximum decoded bytes allowed for a single base-IFD chunk during interactive
/// cursor readouts (`16 MiB`).
const MAX_CURSOR_CHUNK_BYTES: u64 = 16 * 1024 * 1024;

impl TiffSource {
    pub(super) fn band_scale_offset(&self, band: BandKey) -> (f64, f64) {
        self.description
            .band_info(band)
            .map_or((1.0, 0.0), |info| (info.scale, info.offset))
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

    fn admit_cached_cursor_chunk(
        ctx: &DecodeContext<'_>,
        dw: u32,
        dh: u32,
        stored_w: u32,
        stored_h: u32,
        raw_depth: u32,
        out_bpp: u32,
    ) -> Result<(), RasterError> {
        let out_bytes = u64::from(dw)
            .saturating_mul(u64::from(dh))
            .saturating_mul(u64::from(out_bpp));
        let out_cap = (ctx.budget.decoded_chunk_cache_bytes as u64).min(MAX_CURSOR_CHUNK_BYTES);
        if out_bytes > out_cap {
            return Err(RasterError::LimitExceeded {
                width: dw,
                height: dh,
                estimated_bytes: out_bytes,
                limit_bytes: out_cap,
            });
        }
        let working_bytes = chunk_working_bytes(stored_w, stored_h, raw_depth, out_bpp);
        if working_bytes > ctx.budget.decode_working_bytes {
            return Err(RasterError::LimitExceeded {
                width: stored_w,
                height: stored_h,
                estimated_bytes: working_bytes,
                limit_bytes: ctx.budget.decode_working_bytes,
            });
        }
        Ok(())
    }

    fn sample_display_rgba(
        &self,
        px: u32,
        py: u32,
        ctx: &DecodeContext<'_>,
    ) -> Result<Option<[u8; 4]>, RasterError> {
        let base = self.base_ifd();
        let (chunk_index, lx, ly, dw, dh) = self.base_chunk_loc(px, py)?;
        let (stored_w, stored_h) = base.chunk_stored_dims(chunk_index).unwrap_or((dw, dh));
        Self::admit_cached_cursor_chunk(
            ctx,
            dw,
            dh,
            stored_w,
            stored_h,
            base.raw_pixel_bytes(),
            4,
        )?;

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

    pub(super) fn sample_band_impl(
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
        if band == BandKey::Display && format == PixelFormat::Rgba8UnormSrgb {
            return Ok(match self.sample_display_rgba(px, py, ctx)? {
                Some(rgba) => BandSample::Rgba(rgba),
                None => BandSample::Nodata,
            });
        }
        let spec = stream_spec_for_band(band, format);
        if self.validate_stream_spec(spec).is_err() {
            return Ok(BandSample::Unavailable);
        }
        let BandKey::Index(band_u16) = band else {
            return Ok(BandSample::Unavailable);
        };
        let base = self.base_ifd();
        let band_idx = usize::from(band_u16);
        let (chunk_index, lx_u32, ly_u32, dw, dh) = self.base_chunk_loc(px, py)?;
        let (stored_w, stored_h) = base.chunk_stored_dims(chunk_index).unwrap_or((dw, dh));
        let raw_depth = base.band_raw_pixel_bytes();

        // Reuse `SharedChunkCache` for sample types lossless in `R32Float` /
        // `R8Uint` / `R16Uint` so hover probes avoid re-decompressing chunks on
        // every cursor move. Fall back to direct raw-chunk `f64` reads for
        // `U32`/`I32`/`F64` whose precision exceeds f32's 24-bit mantissa.
        let needs_raw_f64 = format == PixelFormat::R32Float
            && matches!(
                base.sample_type,
                SampleType::U32 | SampleType::I32 | SampleType::F64
            );
        if !needs_raw_f64 {
            Self::admit_cached_cursor_chunk(
                ctx,
                dw,
                dh,
                stored_w,
                stored_h,
                raw_depth,
                format.bytes_per_pixel() as u32,
            )?;
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
                    let (scale, offset) = self.band_scale_offset(band);
                    Ok(BandSample::Scalar {
                        raw,
                        physical: raw.mul_add(scale, offset),
                    })
                }
                PixelFormat::Rgba8UnormSrgb => Ok(BandSample::Unavailable),
            };
        }

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

        let lx = lx_u32 as usize;
        let ly = ly_u32 as usize;
        let _lease = ctx
            .budget_tracker
            .reserve_blocking(chunk_raw_bytes, ctx.cancel)?;
        let (cdw, cdh, payload) = decode_raw_chunk(
            self.byte_source.as_ref(),
            base,
            self.byte_order,
            base.band_plane(band_idx),
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
        let (stride, ch_off) = base.band_stride_offset(band_idx);
        let sample_idx = (ly * (cdw as usize) + lx) * stride + ch_off;
        let Some(raw) = read_native_sample_f64(&data, base.sample_type, sample_idx) else {
            return Ok(BandSample::Unavailable);
        };
        if !is_data_f64(raw, self.nodata_f64) {
            return Ok(BandSample::Nodata);
        }
        let (scale, offset) = self.band_scale_offset(band);
        Ok(BandSample::Scalar {
            raw,
            physical: raw.mul_add(scale, offset),
        })
    }
}
