//! Pure-`ByteSource` JPEG [`RasterSource`] adapter (`jpeg-decoder` without
//! rayon threadpools).
//!
//! Enforces pre-allocation frame budget checks via `Decoder::scale()` **before**
//! calling `Decoder::decode()`, drops the raw pixel buffer immediately after
//! RGBA8 conversion, and feeds the exact pyramid level grid into [`Cascade`].

use std::sync::Arc;

use veldmap_core::budget::MemoryBudget;
use veldmap_core::io::{BufferedByteReader, ByteSource};
use veldmap_core::sync::CancelToken;
use veldmap_core::tile::{
    BandKey, PixelFormat, RASTER_TILE_SIZE, SourceFingerprint, SourceId, TileStreamSpec,
    level_count, level_size,
};

use crate::RasterError;
use crate::pipeline::cascade::{self, Cascade, TileEmit};
use crate::pipeline::metadata::{BandInfo, GeoPlacement, RasterDescription, SampleType};
use crate::pipeline::plan::{LevelPlan, PyramidMeta, ServeMode};
use crate::pipeline::radiometry::{DisplayMapping, PixelLayout, Samples};
use crate::pipeline::resample::{Window, bleed_alpha, resample_rgba8_window};
use crate::pipeline::{DecodeContext, RasterSource};

const JPEG_MAGIC: [u8; 3] = [0xFF, 0xD8, 0xFF];
const JPEG_DECODE_REV: u32 = 1;

const DISPLAY_STREAM_SPEC: TileStreamSpec = TileStreamSpec::new(
    BandKey::Display,
    PixelFormat::Rgba8UnormSrgb,
    0,
    JPEG_DECODE_REV,
);

/// Return whether `head` begins with a JPEG SOI + marker prefix (`FF D8 FF`).
pub fn has_jpeg_magic(head: &[u8]) -> bool {
    head.starts_with(&JPEG_MAGIC)
}

/// Pure-`ByteSource` JPEG source implementing [`RasterSource`].
pub struct JpegSource {
    byte_source: Arc<dyn ByteSource>,
    fingerprint: SourceFingerprint,
    source_id: SourceId,
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
        decoder.read_info().map_err(|e| map_jpeg_error(e, cancel))?;
        let info = decoder
            .info()
            .ok_or_else(|| RasterError::Decode("JPEG header missing image info".to_string()))?;
        let width = u32::from(info.width);
        let height = u32::from(info.height);
        if width == 0 || height == 0 {
            return Err(RasterError::InvalidDimensions { width, height });
        }

        let channels = match info.pixel_format {
            jpeg_decoder::PixelFormat::L8 | jpeg_decoder::PixelFormat::L16 => 1,
            jpeg_decoder::PixelFormat::RGB24 => 3,
            jpeg_decoder::PixelFormat::CMYK32 => 4,
        };

        let n_levels = level_count(width, height).min(u32::from(u8::MAX)) as u8;
        let mut levels = Vec::with_capacity(usize::from(n_levels));
        let mut finest_fitting: Option<(u8, u64)> = None;

        for lvl in 0..n_levels {
            let lw = level_size(width, u32::from(lvl));
            let lh = level_size(height, u32::from(lvl));
            let (dw, dh) = dct_decoded_size(width, height, lvl);
            let peak = jpeg_level_peak_bytes(dw, dh, lw, lh);
            let fits = peak <= budget.decode_working_bytes;
            if fits && finest_fitting.is_none() {
                finest_fitting = Some((lvl, peak));
            }
            levels.push(LevelPlan {
                serve: ServeMode::Pass { from: lvl },
                source_pixels: u64::from(dw).saturating_mul(u64::from(dh)),
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

        let binding_trouble = placement.is_none().then(|| {
            "JPEG has no accompanying world file (.jgw/.wld); assign CRS or add a world file"
                .to_string()
        });

        let description = RasterDescription {
            width,
            height,
            bands: vec![BandInfo::display(SampleType::U8, channels, false, None)],
            ties: Vec::new(),
            placement,
            foreign_datum_epsg: None,
            binding_trouble,
            levels,
        };

        Ok(Self {
            byte_source,
            fingerprint,
            source_id: SourceId::from_fingerprint(&fingerprint),
            pass_level,
            pass_peak_bytes,
            description,
        })
    }
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
        self.pass_peak_bytes
    }

    fn produce_direct(
        &self,
        level: u8,
        _wants: &[(u32, u32)],
        _ctx: &DecodeContext<'_>,
        _emit: TileEmit<'_>,
    ) -> Result<(), RasterError> {
        Err(RasterError::Decode(format!(
            "JPEG does not support pointwise random access at level {level}; use sequential pass"
        )))
    }

    fn produce_pass(&self, ctx: &DecodeContext<'_>, emit: TileEmit<'_>) -> Result<(), RasterError> {
        if ctx.cancel.is_cancelled() {
            return Err(RasterError::Cancelled);
        }
        let lw = level_size(self.description.width, u32::from(self.pass_level));
        let lh = level_size(self.description.height, u32::from(self.pass_level));

        let reader = BufferedByteReader::new(Arc::clone(&self.byte_source), ctx.cancel.clone());
        let mut decoder = jpeg_decoder::Decoder::new(reader);
        decoder
            .read_info()
            .map_err(|e| map_jpeg_error(e, ctx.cancel))?;

        // Check scaled output dimensions BEFORE allocating the decoded frame buffer.
        let (dw, dh) = decoder
            .scale(clamp_u16(lw), clamp_u16(lh))
            .map(|(w, h)| (u32::from(w), u32::from(h)))
            .map_err(|e| map_jpeg_error(e, ctx.cancel))?;

        let peak = jpeg_level_peak_bytes(dw, dh, lw, lh);
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
        let mapping = DisplayMapping::identity(None);
        let rgba = match dinfo.pixel_format {
            jpeg_decoder::PixelFormat::L8 => {
                mapping.rgba(&Samples::U8(&pixels), PixelLayout::named(1), pixel_count)
            }
            jpeg_decoder::PixelFormat::RGB24 => {
                mapping.rgba(&Samples::U8(&pixels), PixelLayout::named(3), pixel_count)
            }
            jpeg_decoder::PixelFormat::L16 => {
                let bytes: Vec<u8> = pixels
                    .chunks_exact(2)
                    .map(|p| (u16::from_ne_bytes([p[0], p[1]]) >> 8) as u8)
                    .collect();
                mapping.rgba(&Samples::U8(&bytes), PixelLayout::named(1), pixel_count)
            }
            jpeg_decoder::PixelFormat::CMYK32 => cmyk_to_rgba8(&pixels, pixel_count),
        };
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

        let mut cascade = Cascade::new(self.pass_level, lw, lh);
        let mut bled_emit = |lvl: u8, x: u32, y: u32, tw: u32, th: u32, mut tile_rgba: Vec<u8>| {
            bleed_alpha(&mut tile_rgba, tw, th);
            emit(lvl, x, y, tw, th, tile_rgba)
        };
        cascade.push_rows(&rgba, lh, &mut bled_emit)?;
        cascade.finish(&mut bled_emit)
    }

    fn sample_display_rgba(
        &self,
        _px: u32,
        _py: u32,
        _ctx: &DecodeContext<'_>,
    ) -> Result<Option<[u8; 4]>, RasterError> {
        // Full-frame JPEG decoding is deferred to Bulk passes rather than blocking
        // the Fast hover probe lane on every cursor movement.
        Ok(None)
    }
}

fn cmyk_to_rgba8(cmyk: &[u8], pixel_count: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(pixel_count.saturating_mul(4));
    for px in cmyk.chunks_exact(4).take(pixel_count) {
        // Adobe/JPEG APP14 inverted CMYK convention used by `jpeg-decoder`.
        let k = f32::from(px[3]) / 255.0;
        let r = (f32::from(px[0]) * k).round().clamp(0.0, 255.0) as u8;
        let g = (f32::from(px[1]) * k).round().clamp(0.0, 255.0) as u8;
        let b = (f32::from(px[2]) * k).round().clamp(0.0, 255.0) as u8;
        out.extend_from_slice(&[r, g, b, 255]);
    }
    out
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

fn jpeg_level_peak_bytes(dw: u32, dh: u32, lw: u32, lh: u32) -> u64 {
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
    max_frame_pair.saturating_add(cascade::bytes(lw, lh))
}

fn map_jpeg_error(err: jpeg_decoder::Error, cancel: &CancelToken) -> RasterError {
    if cancel.is_cancelled() {
        return RasterError::Cancelled;
    }
    RasterError::Decode(err.to_string())
}
