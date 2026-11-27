//! Pure-`ByteSource` JPEG [`RasterSource`] adapter (`jpeg-decoder` with
//! `default-features = false`, using DCT `Decoder::scale()` for memory-bounded
//! progressive-pyramid opens).
//!
//! Enforces pre-allocation frame budget checks **before** calling
//! `Decoder::decode()` (accounting for full-resolution coefficient buffers on
//! progressive and lossless JPEGs where `scale()` does not reduce working
//! memory), sets `set_max_decoding_buffer_size`, drops the raw pixel buffer
//! immediately after RGBA8 conversion, and feeds the exact pyramid level grid
//! into [`Cascade`]. Note: EXIF Orientation tags are not auto-rotated because
//! georeferenced world-file coordinates are defined in stored pixel coordinates;
//! `jpeg-decoder`'s built-in `MpscWorker` may use helper threads on large frames
//! even with `rayon` disabled.

use std::sync::Arc;

use veldmap_core::budget::MemoryBudget;
use veldmap_core::io::{BufferedByteReader, ByteSource};
use veldmap_core::sync::CancelToken;
use veldmap_core::tile::{
    BandKey, PixelFormat, SourceFingerprint, SourceId, TileStreamSpec, level_count, level_size,
};

use crate::image_codec::{
    decode_jpeg_pixels_to_rgba8, jpeg_codec_working_bytes, jpeg_pixel_format_channels,
};
use crate::pipeline::{
    BandInfo, BandSample, Cascade, DISPLAY_STREAM_SPEC, DecodeContext, GeoPlacement, LevelPlan,
    RasterDescription, RasterSource, SampleType, ServeMode, TileEmit, Window, bleed_display_emit,
    cascade_bytes, resample_rgba8_window,
};
use crate::{RasterError, map_std_io_error};

use super::world_file_binding_trouble;

/// Pure-`ByteSource` JPEG source implementing [`RasterSource`].
pub struct JpegSource {
    byte_source: Arc<dyn ByteSource>,
    fingerprint: SourceFingerprint,
    source_id: SourceId,
    coding_process: jpeg_decoder::CodingProcess,
    channels: usize,
    pass_level: u8,
    pass_peak_bytes: u64,
    description: RasterDescription,
}

impl JpegSource {
    /// Open and describe a JPEG from `byte_source` with optional externally
    /// loaded `placement` (e.g. from `.jgw` + `.prj` sidecars).
    pub fn open(
        byte_source: Arc<dyn ByteSource>,
        fingerprint: SourceFingerprint,
        placement: Option<GeoPlacement>,
        budget: &MemoryBudget,
        cancel: &CancelToken,
    ) -> Result<Self, RasterError> {
        if cancel.is_cancelled() {
            return Err(RasterError::Cancelled);
        }
        let reader = BufferedByteReader::new(Arc::clone(&byte_source), cancel.clone());
        let mut decoder = jpeg_decoder::Decoder::new(reader);
        decoder.set_max_decoding_buffer_size(
            budget.decode_working_bytes.min(usize::MAX as u64) as usize
        );
        decoder.read_info().map_err(|e| map_jpeg_error(e, cancel))?;
        let info = decoder
            .info()
            .ok_or_else(|| RasterError::Decode("JPEG header missing image info".to_string()))?;
        let width = u32::from(info.width);
        let height = u32::from(info.height);
        if width == 0 || height == 0 {
            return Err(RasterError::InvalidDimensions { width, height });
        }

        let channels = jpeg_pixel_format_channels(info.pixel_format);
        let sample_type = match info.pixel_format {
            jpeg_decoder::PixelFormat::L16 => SampleType::U16,
            jpeg_decoder::PixelFormat::L8
            | jpeg_decoder::PixelFormat::RGB24
            | jpeg_decoder::PixelFormat::CMYK32 => SampleType::U8,
        };
        let coding_process = info.coding_process;

        let (levels, pass_level, pass_peak_bytes) =
            plan_jpeg_levels(width, height, channels, coding_process, budget)?;

        let binding_trouble = world_file_binding_trouble("JPEG", ".jgw/.wld", placement.as_ref());

        let description = RasterDescription {
            width,
            height,
            bands: vec![BandInfo::display(sample_type, channels, false, None)],
            ties: Vec::new(),
            placement,
            geographic_crs: None,
            binding_trouble,
            levels,
        };

        Ok(Self {
            byte_source,
            fingerprint,
            source_id: SourceId::from_fingerprint(&fingerprint),
            coding_process,
            channels,
            pass_level,
            pass_peak_bytes,
            description,
        })
    }
}

fn plan_jpeg_levels(
    width: u32,
    height: u32,
    channels: usize,
    coding_process: jpeg_decoder::CodingProcess,
    budget: &MemoryBudget,
) -> Result<(Vec<LevelPlan>, u8, u64), RasterError> {
    let n_levels = level_count(width, height).min(u32::from(u8::MAX)) as u8;
    let mut levels = Vec::with_capacity(usize::from(n_levels));
    let mut finest_fitting: Option<(u8, u64)> = None;

    for lvl in 0..n_levels {
        let lw = level_size(width, u32::from(lvl));
        let lh = level_size(height, u32::from(lvl));
        let (dw, dh) = match coding_process {
            jpeg_decoder::CodingProcess::Lossless => (width, height),
            _ => dct_decoded_size(width, height, lvl),
        };
        let peak = jpeg_level_peak_bytes(JpegPeakInputs {
            full: (width, height),
            decoded: (dw, dh),
            level: (lw, lh),
            channels,
            coding_process,
        });
        let fits = peak <= budget.decode_working_bytes;
        if fits && finest_fitting.is_none() {
            finest_fitting = Some((lvl, peak));
        }
        levels.push(LevelPlan {
            serve: ServeMode::Pass { from: lvl },
            estimated_peak_bytes: peak,
            fits,
        });
    }

    let Some((pass_level, pass_peak_bytes)) = finest_fitting else {
        let smallest = levels.last().map_or(0, |l| l.estimated_peak_bytes);
        return Err(RasterError::LimitExceeded {
            width,
            height,
            estimated_bytes: smallest,
            limit_bytes: budget.decode_working_bytes,
        });
    };

    // Mark all coarser levels >= pass_level as emitted by the single `pass_level` cascade.
    for row in levels.iter_mut().skip(usize::from(pass_level)) {
        row.serve = ServeMode::Pass { from: pass_level };
    }

    Ok((levels, pass_level, pass_peak_bytes))
}

impl RasterSource for JpegSource {
    fn fingerprint(&self) -> SourceFingerprint {
        self.fingerprint
    }

    fn source_id(&self) -> SourceId {
        self.source_id
    }

    fn description(&self) -> &RasterDescription {
        &self.description
    }

    fn pass_peak_bytes(&self, _spec: TileStreamSpec) -> u64 {
        self.pass_peak_bytes
    }

    fn produce_pass(
        &self,
        spec: TileStreamSpec,
        ctx: &DecodeContext<'_>,
        emit: TileEmit<'_>,
    ) -> Result<(), RasterError> {
        if spec != DISPLAY_STREAM_SPEC {
            return Err(RasterError::Decode(format!(
                "JpegSource only supports Display RGBA8 stream, not {spec:?}"
            )));
        }
        if ctx.cancel.is_cancelled() {
            return Err(RasterError::Cancelled);
        }
        let lw = level_size(self.description.width, u32::from(self.pass_level));
        let lh = level_size(self.description.height, u32::from(self.pass_level));

        let reader = BufferedByteReader::new(Arc::clone(&self.byte_source), ctx.cancel.clone());
        let mut decoder = jpeg_decoder::Decoder::new(reader);
        decoder.set_max_decoding_buffer_size(
            ctx.budget.decode_working_bytes.min(usize::MAX as u64) as usize,
        );

        // `scale()` internally calls `read_info()` and computes the scaled output
        // dimensions BEFORE allocating the decoded frame buffer.
        let (dw, dh) = decoder
            .scale(clamp_u16(lw), clamp_u16(lh))
            .map(|(w, h)| (u32::from(w), u32::from(h)))
            .map_err(|e| map_jpeg_error(e, ctx.cancel))?;

        let peak = jpeg_level_peak_bytes(JpegPeakInputs {
            full: (self.description.width, self.description.height),
            decoded: (dw, dh),
            level: (lw, lh),
            channels: self.channels,
            coding_process: self.coding_process,
        });
        if peak > ctx.budget.decode_working_bytes {
            return Err(RasterError::LimitExceeded {
                width: dw,
                height: dh,
                estimated_bytes: peak,
                limit_bytes: ctx.budget.decode_working_bytes,
            });
        }
        let _lease = ctx.budget_tracker.reserve_blocking(peak, ctx.cancel)?;

        let pixels = decoder
            .decode()
            .map_err(|e| map_jpeg_error(e, ctx.cancel))?;
        if ctx.cancel.is_cancelled() {
            return Err(RasterError::Cancelled);
        }
        let dinfo = decoder
            .info()
            .ok_or_else(|| RasterError::Decode("JPEG header missing after decode".to_string()))?;

        let pixel_count = (dw as usize).saturating_mul(dh as usize);
        let rgba = decode_jpeg_pixels_to_rgba8(dinfo.pixel_format, &pixels, pixel_count);
        // Release raw JPEG buffer before resampling/cascading so we never hold 3 full frames.
        drop(pixels);

        let rgba = if (dw, dh) != (lw, lh) {
            resample_rgba8_window(
                &rgba,
                dw,
                dh,
                Window {
                    x0: 0.0,
                    y0: 0.0,
                    x1: f64::from(dw),
                    y1: f64::from(dh),
                },
                lw,
                lh,
            )
        } else {
            rgba
        };

        let mut cascade = Cascade::new(self.pass_level, lw, lh, PixelFormat::Rgba8UnormSrgb);
        let mut bled_emit = bleed_display_emit(emit);
        cascade.push_rows(&rgba, lh, &mut bled_emit)?;
        cascade.finish(&mut bled_emit)
    }

    fn sample_band(
        &self,
        _band: BandKey,
        _format: PixelFormat,
        _px: u32,
        _py: u32,
        _ctx: &DecodeContext<'_>,
    ) -> Result<BandSample, RasterError> {
        // Full-frame JPEG probing is intentionally disabled on the Fast hover lane;
        // report Unavailable rather than Nodata.
        Ok(BandSample::Unavailable)
    }
}

fn clamp_u16(v: u32) -> u16 {
    v.min(u32::from(u16::MAX)) as u16
}

/// Predict the DCT-scaled dimensions `jpeg_decoder::Decoder::scale` selects
/// (`1/8`, `1/4`, `1/2`, or `1/1` rounded up).
fn dct_decoded_size(width: u32, height: u32, level: u8) -> (u32, u32) {
    let lw = level_size(width, u32::from(level));
    let lh = level_size(height, u32::from(level));
    let scaled =
        |side: u32, eighths: u32| (u64::from(side) * u64::from(eighths)).div_ceil(8) as u32;
    let eighths = [1u32, 2, 4]
        .into_iter()
        .find(|&e| scaled(width, e) >= lw || scaled(height, e) >= lh)
        .unwrap_or(8);
    (scaled(width, eighths), scaled(height, eighths))
}

#[derive(Clone, Copy)]
struct JpegPeakInputs {
    full: (u32, u32),
    decoded: (u32, u32),
    level: (u32, u32),
    channels: usize,
    coding_process: jpeg_decoder::CodingProcess,
}

fn jpeg_level_peak_bytes(inputs: JpegPeakInputs) -> u64 {
    let (full_w, full_h) = inputs.full;
    let (dw, dh) = inputs.decoded;
    let (lw, lh) = inputs.level;
    let codec_working =
        jpeg_codec_working_bytes(full_w, full_h, inputs.channels, inputs.coding_process);
    let raw_decoded = u64::from(dw)
        .saturating_mul(u64::from(dh))
        .saturating_mul(4);
    let rgba = u64::from(dw)
        .saturating_mul(u64::from(dh))
        .saturating_mul(4);
    let resampled = if (dw, dh) != (lw, lh) {
        u64::from(lw)
            .saturating_mul(u64::from(lh))
            .saturating_mul(4)
    } else {
        0
    };
    let max_frame_pair = raw_decoded
        .saturating_add(rgba)
        .max(rgba.saturating_add(resampled));
    codec_working
        .saturating_add(max_frame_pair)
        .saturating_add(cascade_bytes(lw, lh, PixelFormat::Rgba8UnormSrgb))
}

fn map_jpeg_error(err: jpeg_decoder::Error, cancel: &CancelToken) -> RasterError {
    if cancel.is_cancelled() {
        return RasterError::Cancelled;
    }
    match err {
        jpeg_decoder::Error::Io(io_err) => map_std_io_error(io_err, cancel),
        other => RasterError::Decode(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn progressive_and_lossless_peak_accounting_includes_full_resolution_buffers() {
        let seq = jpeg_level_peak_bytes(JpegPeakInputs {
            full: (16_384, 16_384),
            decoded: (2048, 2048),
            level: (2048, 2048),
            channels: 3,
            coding_process: jpeg_decoder::CodingProcess::DctSequential,
        });
        let prog = jpeg_level_peak_bytes(JpegPeakInputs {
            full: (16_384, 16_384),
            decoded: (2048, 2048),
            level: (2048, 2048),
            channels: 3,
            coding_process: jpeg_decoder::CodingProcess::DctProgressive,
        });
        // Full 16384x16384x3x2 = 1.6 GiB coefficient buffer must be charged.
        assert!(prog > seq + 1_500_000_000);
    }
}
