//! Pure GeoTIFF georeference parsing: `GeoKeyDirectoryTag` (`34735`–`34737`),
//! pixel-to-model affine & tie-point synthesis, and user-defined CRS/datum
//! resolution from plain pre-extracted TIFF tag payloads.

mod crs;
mod geokeys;

use veldmap_core::geo::CrsSpec;

use crate::pipeline::metadata::{GeoPlacement, GeoTie, PlacementCrs, is_valid_lat_lon};

use crs::{build_user_defined_projected_crs, resolve_geographic_crs};
use geokeys::{
    GEOKEY_UNDEFINED, GEOKEY_USER_DEFINED, GeoKeyParams, MODEL_TYPE_GEOGRAPHIC,
    MODEL_TYPE_PROJECTED, parse_geokeys,
};

/// Plain pre-extracted GeoTIFF tag payloads consumed by [`parse_georef`],
/// keeping `georef/` completely free of `tiff::decoder::Decoder`.
pub(super) struct RawGeoTiffTags {
    pub(super) raw_keys: Vec<u16>,
    pub(super) double_params: Vec<f64>,
    /// Raw NUL-truncated bytes of `GeoAsciiParamsTag` (`34737`), preserved as
    /// bytes until `GeoKeyDirectoryTag` byte-offset slicing completes so
    /// non-UTF-8 Latin-1 bytes never shift subsequent key offsets.
    pub(super) ascii_params: Vec<u8>,
    pub(super) points: Vec<f64>,
    pub(super) scale: Vec<f64>,
    pub(super) matrix: Vec<f64>,
}

pub(super) struct ParsedGeoref {
    pub(super) ties: Vec<GeoTie>,
    pub(super) placement: Option<GeoPlacement>,
    /// Embedded pixel affine from `ModelTiepoint`/`ModelPixelScale`/`ModelTransformation`
    /// even when no CRS GeoKeys were present (enabling `.prj`-only sidecars).
    pub(super) embedded_affine: Option<[f64; 6]>,
    pub(super) geographic_crs: Option<CrsSpec>,
    pub(super) trouble: Option<String>,
}

pub(super) fn parse_georef(tags: &RawGeoTiffTags, width: u32, height: u32) -> ParsedGeoref {
    let geo_params = parse_geokeys(&tags.raw_keys, &tags.double_params, &tags.ascii_params);
    let points = &tags.points;
    let scale = &tags.scale;
    let matrix = &tags.matrix;

    let placement = geo_placement(&geo_params, points, scale, matrix, width, height);
    let ties = geo_ties(&geo_params, points, scale, matrix, width, height);
    let embedded_affine = if placement.is_none() && points.len() <= 6 {
        affine_from_tags(
            points,
            scale,
            matrix,
            geo_params.raster_type.half_pixel_shift(),
        )
    } else {
        None
    };
    let geographic_crs = if !ties.is_empty() {
        resolve_geographic_crs(&geo_params)
    } else {
        None
    };

    let carries = !points.is_empty() || !scale.is_empty() || !matrix.is_empty();
    let taken = !ties.is_empty() || placement.is_some();
    let trouble = (carries && !taken).then(|| {
        let named = |opt: Option<u16>| opt.map_or_else(|| "unnamed".to_string(), |v| v.to_string());
        format!(
            "raster carries georeference tags, but could not be bound: model {}, projected {}, tie points {}",
            named(geo_params.model_type),
            named(geo_params.projected_crs),
            points.len() / 6
        )
    });

    ParsedGeoref {
        ties,
        placement,
        embedded_affine,
        geographic_crs,
        trouble,
    }
}

fn usable_step(scale: &[f64]) -> bool {
    matches!(scale.get(..2), Some([x, y]) if x.is_finite() && y.is_finite() && *x != 0.0 && *y != 0.0)
}

fn affine_from_matrix(matrix: &[f64], half: f64) -> Option<[f64; 6]> {
    if matrix.len() != 16 {
        return None;
    }
    let a = matrix.get(..8)?;
    if !a.iter().all(|v| v.is_finite()) || a[0] * a[5] - a[1] * a[4] == 0.0 {
        return None;
    }
    Some([
        a[0],
        a[1],
        a[3] - (a[0] + a[1]) * half,
        a[4],
        a[5],
        a[7] - (a[4] + a[5]) * half,
    ])
}

fn affine_from_tags(points: &[f64], scale: &[f64], matrix: &[f64], half: f64) -> Option<[f64; 6]> {
    // Prefer `ModelTransformationTag` when it carries non-zero rotation/shear
    // (which `ModelTiepointTag + ModelPixelScaleTag` cannot represent).
    if let Some(mat_affine) = affine_from_matrix(matrix, half)
        && (mat_affine[1] != 0.0 || mat_affine[3] != 0.0)
    {
        return Some(mat_affine);
    }
    if let (Some(tie), true) = (points.get(..6), usable_step(scale)) {
        let affine = [
            scale[0],
            0.0,
            tie[3] - (tie[0] + half) * scale[0],
            0.0,
            -scale[1],
            tie[4] + (tie[1] + half) * scale[1],
        ];
        if affine.iter().all(|v| v.is_finite()) {
            return Some(affine);
        }
    }
    affine_from_matrix(matrix, half)
}

fn geo_placement(
    geo_params: &GeoKeyParams,
    points: &[f64],
    scale: &[f64],
    matrix: &[f64],
    width: u32,
    height: u32,
) -> Option<GeoPlacement> {
    if geo_params.model_type != Some(MODEL_TYPE_PROJECTED) || points.len() > 6 {
        return None;
    }
    let crs = match geo_params.projected_crs {
        Some(GEOKEY_USER_DEFINED | GEOKEY_UNDEFINED) | None => {
            let (crs, warning) = build_user_defined_projected_crs(geo_params)?;
            PlacementCrs::Resolved { crs, warning }
        }
        Some(code) => PlacementCrs::Epsg(u32::from(code)),
    };

    let half = geo_params.raster_type.half_pixel_shift();
    let affine = affine_from_tags(points, scale, matrix, half)?;

    let (right, bottom) = (f64::from(width), f64::from(height));
    let corner = [
        affine[0] * right + affine[1] * bottom + affine[2],
        affine[3] * right + affine[4] * bottom + affine[5],
    ];
    if !affine.iter().chain(corner.iter()).all(|v| v.is_finite()) {
        return None;
    }

    Some(GeoPlacement {
        crs: Some(crs),
        affine,
    })
}

fn snap_deg_bound(v: f64, limit: f64) -> f64 {
    const EPS_DEG: f64 = 1e-6;
    if v > limit && v <= limit + EPS_DEG {
        limit
    } else if v < -limit && v >= -limit - EPS_DEG {
        -limit
    } else {
        v
    }
}

fn synthesize_affine_ties(a: [f64; 6], width: u32, height: u32, half: f64) -> Vec<GeoTie> {
    let (right, bottom) = (f64::from(width), f64::from(height));
    // For `PixelIsPoint`, sample ties at the outer pixel centers (matching
    // the explicit multi-tiepoint branch) so pole-touching global rasters
    // (`lat = ±90°`) and `[0°, 360°]` rasters stay within valid geodetic
    // bounds while `TieGrid` linearly extrapolates outer half-pixel edges.
    let (px0, px1) = if half > 0.0 && width > 1 {
        (half, right - half)
    } else {
        (0.0, right)
    };
    let (py0, py1) = if half > 0.0 && height > 1 {
        (half, bottom - half)
    } else {
        (0.0, bottom)
    };

    let dlon_x = (a[0] * (px1 - px0)).abs();
    let dlon_y = (a[1] * (py1 - py0)).abs();
    // Guard against corrupt TIFFs before subdividing; valid longitudes are
    // within `[-360°, +360°]` (`span <= 720°`).
    if !dlon_x.is_finite() || !dlon_y.is_finite() || dlon_x > 720.0 || dlon_y > 720.0 {
        return Vec::new();
    }
    // Subdivide any axis whose longitude span exceeds `170°` so adjacent
    // lattice samples stay `< 180°` apart and `TieGrid::unwind` never wraps
    // wide (`180° < Δlon < 359°`) geographic rasters backwards.
    let x_steps = if dlon_x > 170.0 {
        (dlon_x / 90.0).ceil().max(2.0) as usize
    } else {
        1
    };
    let y_steps = if dlon_y > 170.0 {
        (dlon_y / 90.0).ceil().max(2.0) as usize
    } else {
        1
    };

    let mut ties = Vec::with_capacity((x_steps + 1) * (y_steps + 1));
    for iy in 0..=y_steps {
        let py = if iy == y_steps {
            py1
        } else {
            py0 + (py1 - py0) * (iy as f64 / y_steps as f64)
        };
        for ix in 0..=x_steps {
            let px = if ix == x_steps {
                px1
            } else {
                px0 + (px1 - px0) * (ix as f64 / x_steps as f64)
            };
            // Snap tiny IEEE-754 rounding noise near ±90° / ±360° without masking
            // genuine out-of-range coordinates (e.g. UTM metres mis-tagged as geo).
            let lon = snap_deg_bound(a[0] * px + a[1] * py + a[2], 360.0);
            let lat = snap_deg_bound(a[3] * px + a[4] * py + a[5], 90.0);
            ties.push(GeoTie { px, py, lon, lat });
        }
    }
    ties
}

fn geo_ties(
    geo_params: &GeoKeyParams,
    points: &[f64],
    scale: &[f64],
    matrix: &[f64],
    width: u32,
    height: u32,
) -> Vec<GeoTie> {
    if geo_params.model_type != Some(MODEL_TYPE_GEOGRAPHIC) {
        return Vec::new();
    }
    let half = geo_params.raster_type.half_pixel_shift();
    let finite_ties = |ties: Vec<GeoTie>| -> Vec<GeoTie> {
        let valid = ties
            .iter()
            .all(|t| t.px.is_finite() && t.py.is_finite() && is_valid_lat_lon(t.lat, t.lon));
        if valid { ties } else { Vec::new() }
    };

    if points.len() > 6 {
        let ties: Vec<GeoTie> = points
            .chunks_exact(6)
            .map(|tie| GeoTie {
                px: tie[0] + half,
                py: tie[1] + half,
                lon: tie[3],
                lat: tie[4],
            })
            .collect();
        return finite_ties(ties);
    }

    if let Some(a) = affine_from_tags(points, scale, matrix, half) {
        return finite_ties(synthesize_affine_ties(a, width, height, half));
    }

    Vec::new()
}

#[cfg(test)]
#[path = "georef_tests.rs"]
mod tests;
