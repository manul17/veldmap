//! Format-agnostic georeference [`Frame`] resolution from [`RasterDescription`]
//! with optional user CRS overrides.

use veldmap_core::geo::{CrsSpec, Datum, DatumWarning, Frame, Tie, TieGrid, known_datum_from_code};

use crate::RasterError;

use super::metadata::RasterDescription;

fn geographic_frame_from_grid(
    grid: TieGrid,
    foreign_datum_epsg: Option<u16>,
    geographic_datum: Option<&(Datum, Option<DatumWarning>)>,
) -> Frame {
    if let Some((datum, warning)) = geographic_datum {
        return Frame::from_grid_with_datum(grid, *datum, "User-defined geographic CRS")
            .with_datum_warning(warning.clone());
    }
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
/// the CRS applied to `desc.placement`'s affine transform (or geographic tie
/// grid / world-file raster without a `.prj` sidecar).
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
    let bind_grid = |g: TieGrid| {
        geographic_frame_from_grid(g, desc.foreign_datum_epsg, desc.geographic_datum.as_ref())
    };

    if let Some(crs) = crs_override {
        if let Some(ref placement) = desc.placement {
            return Frame::from_crs_pixel_affine(crs, placement.affine, desc.width, desc.height)
                .ok_or_else(|| {
                    RasterError::Binding(format!(
                        "CRS override '{}' produced no valid ground extent for this raster's affine transform",
                        crs.label()
                    ))
                });
        }
        if let Some(g) = grid {
            return match crs {
                CrsSpec::Geographic {
                    datum,
                    label,
                    warning,
                } => Ok(Frame::from_grid_with_datum(g, *datum, label.clone())
                    .with_datum_warning(warning.clone())),
                CrsSpec::GeographicProj(proj) => Ok(Frame::from_grid_with_proj(g, proj.clone())),
                CrsSpec::Projected(_) => Err(RasterError::Binding(format!(
                    "cannot apply projected CRS '{}' to a geographic tie-grid raster without a projected affine transform",
                    crs.label()
                ))),
            };
        }
    }

    if let Some(ref g) = grid
        && g.is_dense()
    {
        return Ok(bind_grid(g.clone()));
    }

    let mut placement_err: Option<RasterError> = None;
    if let Some(ref placement) = desc.placement {
        if let Some(ref custom) = placement.custom_crs {
            if let Some(frame) =
                Frame::from_crs_pixel_affine(custom, placement.affine, desc.width, desc.height)
            {
                return Ok(frame.with_datum_warning(placement.datum_warning.clone()));
            }
            placement_err = Some(RasterError::Binding(format!(
                "CRS '{}' produced no valid ground extent for this raster's affine transform",
                custom.label()
            )));
        } else if let Some(epsg) = placement.epsg {
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
                    placement_err =
                        Some(veldmap_core::geo::BindingError::InvalidExtent(epsg).into());
                }
                Err(_) => {
                    placement_err =
                        Some(veldmap_core::geo::BindingError::UnsupportedEpsg(epsg).into());
                }
            }
        }
    }

    if let Some(g) = grid {
        return Ok(bind_grid(g));
    }

    if let Some(err) = placement_err {
        return Err(err);
    }

    Err(RasterError::Binding(
        desc.binding_trouble
            .clone()
            .unwrap_or_else(|| "raster carries no supported georeference binding".to_owned()),
    ))
}
