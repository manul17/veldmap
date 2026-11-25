//! Thin format-agnostic analytical [`RasterSource`] seam over concrete
//! decoders ([`crate::TiffSource`], [`crate::PngSource`], [`crate::JpegSource`],
//! and future mission codecs).

use veldmap_core::tile::{DecodedTile, PixelFormat, SourceFingerprint, SourceId};

use crate::RasterError;

use super::DecodeContext;
use super::cascade::TileEmit;
use super::metadata::RasterDescription;
use super::plan::{DISPLAY_STREAM_SPEC, PyramidMeta};

/// Format-agnostic opened analytical raster source capable of describing its
/// pyramid, producing pointwise or sequential-pass tiles, and answering
/// zoom-invariant native-resolution display pixel queries.
pub trait RasterSource: Send + Sync {
    /// Content fingerprint identifying this source for disk-cache keying.
    fn fingerprint(&self) -> SourceFingerprint;

    /// Deterministic [`SourceId`] derived from [`Self::fingerprint`].
    fn source_id(&self) -> SourceId;

    /// Parsed header metadata, georeference binding info, and pyramid plan.
    fn description(&self) -> &RasterDescription;

    /// Scheduler-facing [`PyramidMeta`] for the display band.
    fn pyramid_meta(&self) -> PyramidMeta {
        let desc = self.description();
        PyramidMeta {
            source: self.source_id(),
            spec: DISPLAY_STREAM_SPEC,
            levels: desc.levels.clone(),
        }
    }

    /// Estimated peak working bytes required by [`Self::produce_pass`].
    fn pass_peak_bytes(&self) -> u64;

    /// Produce pointwise tiles `wants` at `level`, deduplicating source chunk
    /// reads via `ctx.chunk_cache` and bounding active decode memory via
    /// `ctx.budget_tracker`.
    fn produce_direct(
        &self,
        level: u8,
        _wants: &[(u32, u32)],
        _ctx: &DecodeContext<'_>,
        _emit: TileEmit<'_>,
    ) -> Result<(), RasterError> {
        Err(RasterError::Decode(format!(
            "raster source does not support pointwise random access at level {level}; use sequential pass"
        )))
    }

    /// Run a sequential top-to-bottom pass over the base image, emitting every
    /// pyramid tile via `emit`.
    fn produce_pass(&self, ctx: &DecodeContext<'_>, emit: TileEmit<'_>) -> Result<(), RasterError>;

    /// Produce a single pointwise display tile `(level, tx, ty)` into a
    /// validated [`DecodedTile`].
    fn decode_pointwise_tile(
        &self,
        level: u8,
        tx: u32,
        ty: u32,
        ctx: &DecodeContext<'_>,
    ) -> Result<DecodedTile, RasterError> {
        let mut produced: Option<DecodedTile> = None;
        self.produce_direct(
            level,
            &[(tx, ty)],
            ctx,
            &mut |_lvl, _x, _y, tw, th, rgba| {
                let tile = DecodedTile::try_new(tw, th, PixelFormat::Rgba8UnormSrgb, rgba)
                    .map_err(|e| RasterError::Decode(e.to_string()))?;
                produced = Some(tile);
                Ok(())
            },
        )?;
        produced.ok_or_else(|| RasterError::TileOutOfRange {
            scheme: self.pyramid_meta().scheme(),
            level,
            x: tx,
            y: ty,
        })
    }

    /// Zoom-invariant CPU display-RGBA readout at native base pixel `(px, py)`.
    ///
    /// Returns `Ok(None)` if `(px, py)` lies outside the raster bounds or is
    /// transparent/nodata (`alpha == 0`).
    fn sample_display_rgba(
        &self,
        px: u32,
        py: u32,
        ctx: &DecodeContext<'_>,
    ) -> Result<Option<[u8; 4]>, RasterError>;
}
