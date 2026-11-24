//! Thin format-agnostic analytical [`RasterSource`] seam over concrete
//! decoders (such as [`crate::TiffSource`], future PNG/JPEG world-file
//! adapters, and v1.1 mission codecs).

use veldmap_core::geo::{CrsSpec, DatumWarning, Frame, Tie, TieGrid, known_datum_from_code};
use veldmap_core::tile::{DecodedTile, PixelFormat, SourceFingerprint, SourceId};

use crate::RasterError;

use super::DecodeContext;
use super::cascade::TileEmit;
use super::metadata::RasterDescription;
use super::plan::PyramidMeta;

fn geographic_frame_from_grid(grid: TieGrid, foreign_datum_epsg: Option<u16>) -> Frame {
    let Some(code) = foreign_datum_epsg else {
        return Frame::from_grid(grid);
    };
    if let Some((datum, warning)) = known_datum_from_code(u32::from(code)) {
        return Frame::from_grid_with_datum(grid, datum, format!("EPSG:{code}"))
            .with_datum_warning(warning);
    }
    if let Ok(CrsSpec::GeographicProj(proj)) = CrsSpec::from_epsg(u32::from(code)) {
        return Frame::from_grid_with_proj(grid, proj);
    }
    tracing::warn!(
        foreign_datum_epsg = code,
        "Geographic raster uses foreign datum EPSG:{code} without a bundled shift; binding unshifted on WGS84"
    );
    Frame::from_grid(grid).with_datum_warning(Some(DatumWarning::UnshiftedForeignDatum {
        label: format!("EPSG:{code}"),
    }))
}

/// Derive a [`veldmap_core::geo::Frame`] georeference binding from a parsed
/// [`RasterDescription`].
///
/// Prefers a dense tie-point lattice (`> 4` ties), then an explicit custom CRS
/// or EPSG placement (`GeoPlacement`), then a 4-corner geographic tie-point grid.
pub fn frame_from_description(desc: &RasterDescription) -> Result<Frame, RasterError> {
    frame_from_description_with_override(desc, None)
}

/// Derive a [`veldmap_core::geo::Frame`] from `desc`, optionally overriding
/// the CRS applied to `desc.placement`'s affine transform (or assigning a CRS
/// to a world-file raster that shipped without a `.prj` sidecar).
pub fn frame_from_description_with_override(
    desc: &RasterDescription,
    crs_override: Option<&CrsSpec>,
) -> Result<Frame, RasterError> {
    if desc.width == 0 || desc.height == 0 {
        return Err(RasterError::InvalidDimensions {
            width: desc.width,
            height: desc.height,
        });
    }

    if let Some(crs) = crs_override
        && let Some(ref placement) = desc.placement
    {
        return Frame::from_crs_pixel_affine(crs, placement.affine, desc.width, desc.height)
            .ok_or_else(|| {
                RasterError::Binding(format!(
                    "CRS override '{}' produced no valid ground extent for this raster's affine transform",
                    crs.label()
                ))
            });
    }

    let w = f64::from(desc.width);
    let h = f64::from(desc.height);

    let ties: Vec<Tie> = desc
        .ties
        .iter()
        .map(|t| Tie {
            fx: t.px / w,
            fy: t.py / h,
            lat: t.lat,
            lon: t.lon,
        })
        .collect();
    let grid = TieGrid::new(&ties);

    if let Some(ref g) = grid
        && g.is_dense()
    {
        return Ok(geographic_frame_from_grid(
            g.clone(),
            desc.foreign_datum_epsg,
        ));
    }

    if let Some(ref placement) = desc.placement {
        if let Some(ref custom) = placement.custom_crs
            && let Some(frame) =
                Frame::from_crs_pixel_affine(custom, placement.affine, desc.width, desc.height)
        {
            return Ok(frame.with_datum_warning(placement.datum_warning.clone()));
        }
        if let Some(epsg) = placement.epsg {
            match CrsSpec::from_epsg(epsg) {
                Ok(crs) => {
                    if let Some(frame) = Frame::from_crs_pixel_affine(
                        &crs,
                        placement.affine,
                        desc.width,
                        desc.height,
                    ) {
                        return Ok(frame);
                    }
                    if let Some(g) = grid {
                        return Ok(geographic_frame_from_grid(g, desc.foreign_datum_epsg));
                    }
                    return Err(veldmap_core::geo::BindingError::InvalidExtent(epsg).into());
                }
                Err(_) => {
                    if let Some(g) = grid {
                        return Ok(geographic_frame_from_grid(g, desc.foreign_datum_epsg));
                    }
                    return Err(veldmap_core::geo::BindingError::UnsupportedEpsg(epsg).into());
                }
            }
        }
    }

    if let Some(g) = grid {
        return Ok(geographic_frame_from_grid(g, desc.foreign_datum_epsg));
    }

    Err(RasterError::Binding(
        desc.binding_trouble
            .clone()
            .unwrap_or_else(|| "raster carries no supported georeference binding".to_owned()),
    ))
}

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
    fn pyramid_meta(&self) -> PyramidMeta;

    /// Estimated peak working bytes required by [`Self::produce_pass`].
    fn pass_peak_bytes(&self) -> u64;

    /// Produce pointwise tiles `wants` at `level`, deduplicating source chunk
    /// reads via `ctx.chunk_cache` and bounding active decode memory via
    /// `ctx.budget_tracker`.
    fn produce_direct(
        &self,
        level: u8,
        wants: &[(u32, u32)],
        ctx: &DecodeContext<'_>,
        emit: TileEmit<'_>,
    ) -> Result<(), RasterError>;

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
