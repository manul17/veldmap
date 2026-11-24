//! Pure-`ByteSource` PNG [`RasterSource`] adapter.
//!
//! Non-interlaced PNGs stream row-by-row directly into [`Cascade`] (reusing a
//! single row RGBA8 buffer) so working memory is independent of image height;
//! Adam7 interlaced PNGs enforce an upfront whole-frame budget check against
//! `MemoryBudget::decode_working_bytes` before allocation.

use std::sync::Arc;

use veldmap_core::budget::MemoryBudget;
use veldmap_core::io::{BufferedByteReader, ByteSource};
use veldmap_core::sync::CancelToken;
use veldmap_core::tile::{RASTER_TILE_SIZE, SourceFingerprint, SourceId, level_count};

use crate::pipeline::cascade::{self, Cascade, TileEmit, bleed_display_emit};
use crate::pipeline::metadata::{BandInfo, GeoPlacement, RasterDescription, SampleType};
use crate::pipeline::plan::{
    DISPLAY_STREAM_SPEC, LevelPlan, PyramidMeta, ServeMode, check_max_source_side,
};
use crate::pipeline::radiometry::{DisplayMapping, PixelLayout, Samples, expand_u8_into_rgba};
use crate::pipeline::{DecodeContext, RasterSource};
use crate::{RasterError, map_std_io_error};

use super::world_file_binding_trouble;

const PNG_MAGIC: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];

/// Return whether `head` begins with an 8-byte PNG file signature.
pub fn has_png_magic(head: &[u8]) -> bool {
    head.starts_with(&PNG_MAGIC)
}

fn build_png_decoder(
    reader: BufferedByteReader,
    budget: &MemoryBudget,
) -> png::Decoder<BufferedByteReader> {
    let limits = png::Limits {
        bytes: budget.decode_working_bytes.min(usize::MAX as u64) as usize,
    };
    let mut decoder = png::Decoder::new_with_limits(reader, limits);
    // Ignore ancillary text/iCCP chunks so adversarial metadata cannot exhaust
    // decode working memory during header parsing.
    decoder.set_ignore_text_chunk(true);
    decoder.set_ignore_iccp_chunk(true);
    decoder
}

/// Pure-`ByteSource` PNG source implementing [`RasterSource`].
pub struct PngSource {
    byte_source: Arc<dyn ByteSource>,
    fingerprint: SourceFingerprint,
    source_id: SourceId,
    interlaced: bool,
    pass_peak_bytes: u64,
    description: RasterDescription,
}

impl PngSource {
    /// Open and describe a PNG from `byte_source` with optional externally
    /// loaded `placement` (e.g. from `.pgw` + `.prj` sidecars).
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
        let decoder = build_png_decoder(reader, budget);
        let info_reader = decoder.read_info().map_err(|e| map_png_error(e, cancel))?;
        let info = info_reader.info();
        let (width, height, interlaced) = (info.width, info.height, info.interlaced);
        if width == 0 || height == 0 {
            return Err(RasterError::InvalidDimensions { width, height });
        }
        check_max_source_side(width, height)?;

        let has_alpha = matches!(
            info.color_type,
            png::ColorType::GrayscaleAlpha | png::ColorType::Rgba
        ) || info.trns.is_some();
        let channels = match info.color_type {
            png::ColorType::Indexed => 3 + usize::from(has_alpha),
            png::ColorType::Grayscale => 1 + usize::from(has_alpha),
            png::ColorType::Rgb => 3 + usize::from(has_alpha),
            png::ColorType::GrayscaleAlpha => 2,
            png::ColorType::Rgba => 4,
        };

        let cascade_bytes = cascade::bytes(width, height);
        let row_bytes = u64::from(width).saturating_mul(4);
        let pass_peak_bytes = if interlaced {
            // Adam7 requires a full decoded frame buffer + RGBA conversion + cascade.
            u64::from(width)
                .saturating_mul(u64::from(height))
                .saturating_mul(4)
                .saturating_mul(2)
                .saturating_add(cascade_bytes)
        } else {
            cascade_bytes.saturating_add(row_bytes.saturating_mul(2))
        };

        if pass_peak_bytes > budget.decode_working_bytes {
            return Err(RasterError::LimitExceeded {
                width,
                height,
                estimated_bytes: pass_peak_bytes,
                limit_bytes: budget.decode_working_bytes,
            });
        }

        let n_levels = level_count(width, height);
        let full_pixels = u64::from(width).saturating_mul(u64::from(height));
        let levels: Vec<LevelPlan> = (0..n_levels)
            .map(|_| LevelPlan {
                serve: ServeMode::Pass { from: 0 },
                source_pixels: full_pixels,
                estimated_peak_bytes: pass_peak_bytes,
                fits: true,
            })
            .collect();

        let binding_trouble = world_file_binding_trouble("PNG", ".pgw/.wld", placement.as_ref());

        let description = RasterDescription {
            width,
            height,
            bands: vec![BandInfo::display(SampleType::U8, channels, has_alpha, None)],
            ties: Vec::new(),
            placement,
            foreign_datum_epsg: None,
            geographic_datum: None,
            binding_trouble,
            levels,
        };

        Ok(Self {
            byte_source,
            fingerprint,
            source_id: SourceId::from_fingerprint(&fingerprint),
            interlaced,
            pass_peak_bytes,
            description,
        })
    }
    fn open_normalized_reader(
        &self,
        ctx: &DecodeContext<'_>,
    ) -> Result<(png::Reader<BufferedByteReader>, PixelLayout), RasterError> {
        let reader = BufferedByteReader::new(Arc::clone(&self.byte_source), ctx.cancel.clone());
        let mut decoder = build_png_decoder(reader, ctx.budget);
        decoder.set_transformations(png::Transformations::normalize_to_color8());
        let png_reader = decoder
            .read_info()
            .map_err(|e| map_png_error(e, ctx.cancel))?;
        let layout = PixelLayout::named(png_reader.output_color_type().0.samples());
        Ok((png_reader, layout))
    }
}

impl RasterSource for PngSource {
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
            "PNG does not support pointwise random access at level {level}; use sequential pass"
        )))
    }

    fn produce_pass(&self, ctx: &DecodeContext<'_>, emit: TileEmit<'_>) -> Result<(), RasterError> {
        if ctx.cancel.is_cancelled() {
            return Err(RasterError::Cancelled);
        }

        let (mut png_reader, layout) = self.open_normalized_reader(ctx)?;
        let (w, h, interlaced) = {
            let info = png_reader.info();
            (info.width, info.height, info.interlaced)
        };
        if (w, h) != (self.description.width, self.description.height) {
            return Err(RasterError::Decode(
                "PNG dimensions disagreed with header on decode pass".to_string(),
            ));
        }

        let mapping = DisplayMapping::identity(None);
        let mut cascade = Cascade::new(0, w, h);
        let mut bled_emit = bleed_display_emit(emit);

        if interlaced || self.interlaced {
            let buf_len = png_reader.output_buffer_size().ok_or_else(|| {
                RasterError::Decode("PNG frame size overflowed usize".to_string())
            })?;
            let est = (buf_len as u64)
                .saturating_add(u64::from(w).saturating_mul(u64::from(h)).saturating_mul(4))
                .saturating_add(cascade::bytes(w, h));
            if est > ctx.budget.decode_working_bytes {
                return Err(RasterError::LimitExceeded {
                    width: w,
                    height: h,
                    estimated_bytes: est,
                    limit_bytes: ctx.budget.decode_working_bytes,
                });
            }
            let _lease = ctx.budget_tracker.reserve_blocking(est, ctx.cancel)?;
            let mut buf = vec![0u8; buf_len];
            let out = png_reader
                .next_frame(&mut buf)
                .map_err(|e| map_png_error(e, ctx.cancel))?;
            if ctx.cancel.is_cancelled() {
                return Err(RasterError::Cancelled);
            }
            let channels = out.color_type.samples();
            let rgba = mapping.rgba(
                &Samples::U8(&buf[..out.buffer_size()]),
                PixelLayout::named(channels),
                (w as usize).saturating_mul(h as usize),
            );
            drop(buf);
            cascade.push_rows(&rgba, h, &mut bled_emit)?;
            return cascade.finish(&mut bled_emit);
        }

        let _lease = ctx
            .budget_tracker
            .reserve_blocking(self.pass_peak_bytes, ctx.cancel)?;
        let mut row_rgba = Vec::with_capacity((w as usize).saturating_mul(4));
        let mut row_idx = 0u32;
        while let Some(row) = png_reader
            .next_row()
            .map_err(|e| map_png_error(e, ctx.cancel))?
        {
            if row_idx.is_multiple_of(64) && ctx.cancel.is_cancelled() {
                return Err(RasterError::Cancelled);
            }
            expand_u8_into_rgba(row.data(), layout, w as usize, &mut row_rgba);
            cascade.push_rows(&row_rgba, 1, &mut bled_emit)?;
            row_idx = row_idx.saturating_add(1);
        }
        cascade.finish(&mut bled_emit)
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
        // Only perform on-demand row scans for compact non-interlaced PNGs
        // (<= 2 MP) under an explicit budget lease so hover probes never stall
        // the fast lane on large sequential PNG streams.
        let total_px =
            u64::from(self.description.width).saturating_mul(u64::from(self.description.height));
        if self.interlaced || total_px > 2_000_000 {
            return Ok(None);
        }
        if ctx.cancel.is_cancelled() {
            return Err(RasterError::Cancelled);
        }
        let probe_bytes = u64::from(self.description.width)
            .saturating_mul(8)
            .max(65_536);
        let _lease = ctx
            .budget_tracker
            .reserve_blocking(probe_bytes, ctx.cancel)?;
        let (mut png_reader, layout) = self.open_normalized_reader(ctx)?;
        let mut row_rgba = Vec::new();
        for row_idx in 0..=py {
            if row_idx.is_multiple_of(128) && ctx.cancel.is_cancelled() {
                return Err(RasterError::Cancelled);
            }
            let Some(row) = png_reader
                .next_row()
                .map_err(|e| map_png_error(e, ctx.cancel))?
            else {
                return Ok(None);
            };
            if row_idx == py {
                expand_u8_into_rgba(
                    row.data(),
                    layout,
                    self.description.width as usize,
                    &mut row_rgba,
                );
                let idx = (px as usize).saturating_mul(4);
                let Some(px_bytes) = row_rgba.get(idx..idx + 4) else {
                    return Ok(None);
                };
                let color = [px_bytes[0], px_bytes[1], px_bytes[2], px_bytes[3]];
                return Ok((color[3] > 0).then_some(color));
            }
        }
        Ok(None)
    }
}

fn map_png_error(err: png::DecodingError, cancel: &CancelToken) -> RasterError {
    if cancel.is_cancelled() {
        return RasterError::Cancelled;
    }
    match err {
        png::DecodingError::IoError(io_err) => map_std_io_error(io_err, cancel),
        png::DecodingError::LimitsExceeded => RasterError::DecoderLimitsExceeded(err.to_string()),
        other => RasterError::Decode(other.to_string()),
    }
}
