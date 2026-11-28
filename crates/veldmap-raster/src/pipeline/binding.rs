//! Format-agnostic georeference [`Frame`] resolution from [`RasterDescription`]
//! with optional user CRS overrides.

use veldmap_core::geo::{CrsSpec, Frame, Tie, TieGrid};

use crate::RasterError;

use super::metadata::{GeoPlacement, PlacementCrs, RasterDescription};

/// Derive a [`veldmap_core::geo::Frame`] georeference binding from a parsed
/// [`RasterDescription`].
///
/// Prefers a dense tie-point lattice (`> 4` ties), then an explicit custom CRS
/// or EPSG placement (`GeoPlacement`), then a 4-corner geographic tie-point grid.
pub fn frame_from_description(desc: &RasterDescription) -> Result<Frame, RasterError> {
    frame_from_description_with_override(desc, None)
}

fn bind_with_override(
    desc: &RasterDescription,
    crs: &CrsSpec,
    grid: Option<&TieGrid>,
) -> Option<Result<Frame, RasterError>> {
    if let Some(ref placement) = desc.placement {
        return Some(
            Frame::from_crs_pixel_affine(crs, placement.affine, desc.width, desc.height)
                .ok_or_else(|| {
                    RasterError::Binding(format!(
                        "CRS override '{}' produced no valid ground extent for this raster's affine transform",
                        crs.label()
                    ))
                }),
        );
    }
    grid.map(|g| {
        if crs.is_geographic() {
            Ok(Frame::from_grid_with_crs(g.clone(), Some(crs)))
        } else {
            Err(RasterError::Binding(format!(
                "cannot apply projected CRS '{}' to a geographic tie-grid raster without a projected affine transform",
                crs.label()
            )))
        }
    })
}

fn bind_placement(
    placement: &GeoPlacement,
    width: u32,
    height: u32,
) -> Option<Result<Frame, RasterError>> {
    match placement.crs {
        Some(PlacementCrs::Resolved {
            ref crs,
            ref warning,
        }) => Some(
            Frame::from_crs_pixel_affine(crs, placement.affine, width, height)
                .map(|frame| frame.with_datum_warning(warning.clone()))
                .ok_or_else(|| {
                    RasterError::Binding(format!(
                        "CRS '{}' produced no valid ground extent for this raster's affine transform",
                        crs.label()
                    ))
                }),
        ),
        Some(PlacementCrs::Epsg(epsg)) => Some(match CrsSpec::from_epsg(epsg) {
            Ok(crs) => {
                Frame::from_crs_pixel_affine(&crs, placement.affine, width, height).ok_or_else(
                    || {
                        RasterError::Binding(format!(
                            "EPSG:{epsg} georeference transform has no valid ground extent"
                        ))
                    },
                )
            }
            Err(_) => Err(RasterError::Binding(format!(
                "unsupported CRS EPSG:{epsg}"
            ))),
        }),
        None => None,
    }
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
    let bind_grid = |g: TieGrid| Frame::from_grid_with_crs(g, desc.geographic_crs.as_ref());

    if let Some(crs) = crs_override
        && let Some(result) = bind_with_override(desc, crs, grid.as_ref())
    {
        return result;
    }

    if let Some(ref g) = grid
        && g.is_dense()
    {
        return Ok(bind_grid(g.clone()));
    }

    let mut placement_err: Option<RasterError> = None;
    if let Some(ref placement) = desc.placement
        && let Some(res) = bind_placement(placement, desc.width, desc.height)
    {
        match res {
            Ok(frame) => return Ok(frame),
            Err(err) => placement_err = Some(err),
        }
    }

    if let Some(g) = grid {
        return Ok(bind_grid(g));
    }

    if let Some(err) = placement_err {
        return Err(err);
    }

    Err(RasterError::Binding(
        desc.binding_trouble.clone().unwrap_or_else(|| {
            "raster carries no supported georeference binding (requires embedded GeoTIFF tags/tie points or a world file + .prj sidecar)"
                .to_owned()
        }),
    ))
}
