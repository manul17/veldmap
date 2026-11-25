//! Format-agnostic georeference [`Frame`] resolution from [`RasterDescription`]
//! with optional user CRS overrides.

use veldmap_core::geo::{CrsSpec, Frame, Tie, TieGrid};

use crate::RasterError;

use super::metadata::{PlacementCrs, RasterDescription};

fn geographic_frame_from_grid(grid: TieGrid, geographic_crs: Option<&CrsSpec>) -> Frame {
    match geographic_crs {
        Some(CrsSpec::Geographic {
            datum,
            label,
            warning,
        }) => Frame::from_grid_with_datum(grid, *datum, label.clone())
            .with_datum_warning(warning.clone()),
        Some(CrsSpec::GeographicProj(proj)) => Frame::from_grid_with_proj(grid, proj.clone()),
        Some(CrsSpec::Projected(_)) | None => Frame::from_grid(grid),
    }
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
    let bind_grid = |g: TieGrid| geographic_frame_from_grid(g, desc.geographic_crs.as_ref());

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
                CrsSpec::Geographic { .. } | CrsSpec::GeographicProj(_) => {
                    Ok(geographic_frame_from_grid(g, Some(crs)))
                }
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
        match placement.crs {
            Some(PlacementCrs::Resolved {
                ref crs,
                ref warning,
            }) => {
                if let Some(frame) =
                    Frame::from_crs_pixel_affine(crs, placement.affine, desc.width, desc.height)
                {
                    return Ok(frame.with_datum_warning(warning.clone()));
                }
                placement_err = Some(RasterError::Binding(format!(
                    "CRS '{}' produced no valid ground extent for this raster's affine transform",
                    crs.label()
                )));
            }
            Some(PlacementCrs::Epsg(epsg)) => match CrsSpec::from_epsg(epsg) {
                Ok(crs) => {
                    if let Some(frame) = Frame::from_crs_pixel_affine(
                        &crs,
                        placement.affine,
                        desc.width,
                        desc.height,
                    ) {
                        return Ok(frame);
                    }
                    placement_err = Some(RasterError::Binding(format!(
                        "EPSG:{epsg} georeference transform has no valid ground extent"
                    )));
                }
                Err(_) => {
                    placement_err =
                        Some(RasterError::Binding(format!("unsupported CRS EPSG:{epsg}")));
                }
            },
            None => {}
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
