//! Pure-`ByteSource` PNG [`RasterSource`] adapter.
//!
//! Non-interlaced PNGs stream row-by-row directly into [`Cascade`] so working
//! memory is independent of image height; Adam7 interlaced PNGs enforce an
//! upfront whole-frame budget check against `MemoryBudget::decode_working_bytes`
//! before allocation.

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
use crate::pipeline::resample::bleed_alpha;
use crate::pipeline::{DecodeContext, RasterSource};
use crate::tiff::MAX_SOURCE_SIDE;

const PNG_MAGIC: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
const PNG_DECODE_REV: u32 = 1;

const DISPLAY_STREAM_SPEC: TileStreamSpec = TileStreamSpec::new(
    BandKey::Display,
    PixelFormat::Rgba8UnormSrgb,
    0,
    PNG_DECODE_REV,
);

/// Return whether `head` begins with an 8-byte PNG file signature.
pub fn has_png_magic(head: &[u8]) -> bool {
    head.starts_with(&PNG_MAGIC)
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
        let decoder = png::Decoder::new(reader);
        let info_reader = decoder.read_info().map_err(|e| map_png_error(e, cancel))?;
        let info = info_reader.info();
        let (width, height, interlaced) = (info.width, info.height, info.interlaced);
        if width == 0 || height == 0 {
            return Err(RasterError::InvalidDimensions { width, height });
        }
        if width > MAX_SOURCE_SIDE || height > MAX_SOURCE_SIDE {
            return Err(RasterError::LimitExceeded {
                width,
                height,
                estimated_bytes: u64::from(width)
                    .saturating_mul(u64::from(height))
                    .saturating_mul(4),
                limit_bytes: u64::from(MAX_SOURCE_SIDE)
                    .saturating_mul(u64::from(MAX_SOURCE_SIDE))
                    .saturating_mul(4),
            });
        }

        let has_alpha = matches!(
            info.color_type,
            png::ColorType::GrayscaleAlpha | png::ColorType::Rgba
        ) || info.trns.is_some();
        let channels = match info.color_type {
            png::ColorType::Indexed => 3 + usize::from(has_alpha),
            _ => info.color_type.samples(),
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
        let levels: Vec<LevelPlan> = (0..n_levels)
            .map(|lvl| {
                let lw = level_size(width, lvl);
                let lh = level_size(height, lvl);
                LevelPlan {
                    serve: ServeMode::Pass { from: 0 },
                    source_pixels: u64::from(lw).saturating_mul(u64::from(lh)),
                    estimated_peak_bytes: pass_peak_bytes,
                    fits: true,
                }
            })
            .collect();

        let binding_trouble = placement.is_none().then(|| {
            "PNG has no accompanying world file (.pgw/.wld); assign CRS or add a world file"
                .to_string()
        });

        let description = RasterDescription {
            width,
            height,
            bands: vec![BandInfo::display(SampleType::U8, channels, has_alpha, None)],
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
            interlaced,
            pass_peak_bytes,
            description,
        })
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
        let _lease = ctx
            .budget_tracker
            .reserve_blocking(self.pass_peak_bytes, ctx.cancel)?;

        let reader = BufferedByteReader::new(Arc::clone(&self.byte_source), ctx.cancel.clone());
        let mut decoder = png::Decoder::new(reader);
        decoder.set_transformations(png::Transformations::normalize_to_color8());
        let mut png_reader = decoder
            .read_info()
            .map_err(|e| map_png_error(e, ctx.cancel))?;

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
        let mut bled_emit = |lvl: u8, x: u32, y: u32, tw: u32, th: u32, mut rgba: Vec<u8>| {
            bleed_alpha(&mut rgba, tw, th);
            emit(lvl, x, y, tw, th, rgba)
        };

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

        let channels = png_reader.output_color_type().0.samples();
        let layout = PixelLayout::named(channels);
        let mut row_idx = 0u32;
        while let Some(row) = png_reader
            .next_row()
            .map_err(|e| map_png_error(e, ctx.cancel))?
        {
            if row_idx.is_multiple_of(64) && ctx.cancel.is_cancelled() {
                return Err(RasterError::Cancelled);
            }
            let rgba = mapping.rgba(&Samples::U8(row.data()), layout, w as usize);
            cascade.push_rows(&rgba, 1, &mut bled_emit)?;
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
        // Only perform on-demand row scans for reasonably sized non-interlaced PNGs
        // to avoid stalling the fast probe lane on giant full-image streams.
        let total_px =
            u64::from(self.description.width).saturating_mul(u64::from(self.description.height));
        if self.interlaced || total_px > 16_000_000 {
            return Ok(None);
        }
        if ctx.cancel.is_cancelled() {
            return Err(RasterError::Cancelled);
        }
        let reader = BufferedByteReader::new(Arc::clone(&self.byte_source), ctx.cancel.clone());
        let mut decoder = png::Decoder::new(reader);
        decoder.set_transformations(png::Transformations::normalize_to_color8());
        let mut png_reader = decoder
            .read_info()
            .map_err(|e| map_png_error(e, ctx.cancel))?;
        let channels = png_reader.output_color_type().0.samples();
        let layout = PixelLayout::named(channels);
        let mapping = DisplayMapping::identity(None);
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
                let rgba = mapping.rgba(
                    &Samples::U8(row.data()),
                    layout,
                    self.description.width as usize,
                );
                let idx = (px as usize).saturating_mul(4);
                let Some(px_bytes) = rgba.get(idx..idx + 4) else {
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
        png::DecodingError::IoError(io_err) => RasterError::Decode(io_err.to_string()),
        png::DecodingError::LimitsExceeded => RasterError::DecoderLimitsExceeded(err.to_string()),
        other => RasterError::Decode(other.to_string()),
    }
}
