//! External georeferencing sidecars: ESRI world files (`.tfw`, `.jgw`, `.pgw`,
//! `.wld`, etc.) and `.prj` WKT1/WKT2 files.
//!
//! Decoders themselves remain pure [`veldmap_core::io::ByteSource`] consumers;
//! callers load optional sidecars at the filesystem boundary and inject the
//! resulting [`SidecarGeoref`] / [`GeoPlacement`] into
//! [`crate::TiffSource::open_with_sidecar`], [`crate::PngSource::open`], or
//! [`crate::JpegSource::open`].

use std::path::{Path, PathBuf};

use veldmap_core::geo::{CrsSpec, trim_crs_text};

use crate::RasterError;
use crate::pipeline::{GeoPlacement, PlacementCrs};

/// Maximum sidecar file size (1 MiB) to guard against reading huge accidental files.
const MAX_SIDECAR_BYTES: u64 = 1024 * 1024;

/// Format consistent user guidance when a raster has an affine/world-file
/// transform but lacks an accompanying `.prj` CRS sidecar.
pub(crate) fn missing_prj_trouble(subject: &str) -> String {
    format!("{subject} but no accompanying .prj CRS sidecar; assign a CRS to place it")
}

/// Parsed external georeferencing sidecars accompanying a local raster file.
///
/// Either or both fields may be present: a PNG/JPEG typically carries a world
/// file + `.prj`, whereas a TIFF with embedded `ModelTiepointTag` +
/// `ModelPixelScaleTag` may carry only an external `.prj` sidecar.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SidecarGeoref {
    pub affine: Option<[f64; 6]>,
    pub crs: Option<CrsSpec>,
}

impl SidecarGeoref {
    /// Convert into a [`GeoPlacement`] when an external world-file affine is
    /// present.
    pub fn into_placement(self) -> Option<GeoPlacement> {
        let affine = self.affine?;
        Some(GeoPlacement {
            crs: self
                .crs
                .map(|crs| PlacementCrs::Resolved { crs, warning: None }),
            affine,
        })
    }

    /// Return whether neither a world file nor a `.prj` sidecar was found.
    pub fn is_empty(&self) -> bool {
        self.affine.is_none() && self.crs.is_none()
    }
}

/// Parse a 6-line ESRI world file (`A, D, B, E, C, F`) into VeldMap's
/// top-left-corner pixel affine `[a0, a1, a2, a3, a4, a5]`:
///
/// `x = a0 * i + a1 * j + a2`
/// `y = a3 * i + a4 * j + a5`
///
/// Per the ESRI world-file specification, `(C, F)` is the coordinate of the
/// **center** of pixel `(0, 0)`, so the top-left corner is shifted by
/// `(-0.5 * (A + B), -0.5 * (D + E))`.
pub fn parse_world_file(text: &str) -> Result<[f64; 6], RasterError> {
    let text = text.trim_start_matches('\u{FEFF}');
    let mut vals = Vec::with_capacity(6);
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let v: f64 = trimmed.parse().map_err(|_| {
            RasterError::Binding(format!("invalid numeric entry '{trimmed}' in world file"))
        })?;
        if !v.is_finite() {
            return Err(RasterError::Binding(
                "world file contains non-finite value".to_string(),
            ));
        }
        vals.push(v);
        if vals.len() == 6 {
            break;
        }
    }
    if vals.len() != 6 {
        return Err(RasterError::Binding(format!(
            "world file must contain 6 numbers, found {}",
            vals.len()
        )));
    }
    let [a, d, b, e, c, f] = [vals[0], vals[1], vals[2], vals[3], vals[4], vals[5]];
    let det = a * e - b * d;
    if !det.is_finite() || det.abs() < 1e-24 {
        return Err(RasterError::Binding(
            "world file affine transform is singular".to_string(),
        ));
    }
    let x0 = c - 0.5 * (a + b);
    let y0 = f - 0.5 * (d + e);
    if !(x0.is_finite() && y0.is_finite()) {
        return Err(RasterError::Binding(
            "world file origin overflowed finite range".to_string(),
        ));
    }
    Ok([a, b, x0, d, e, y0])
}

/// Parse a `.prj` sidecar containing WKT1/WKT2 or a PROJ/EPSG string into a
/// [`CrsSpec`].
pub fn parse_prj_text(text: &str) -> Result<CrsSpec, RasterError> {
    let trimmed = trim_crs_text(text);
    if trimmed.is_empty() {
        return Err(RasterError::Binding(
            ".prj sidecar file is empty".to_string(),
        ));
    }
    CrsSpec::parse_text(trimmed)
        .map(|(crs, _)| crs)
        .map_err(|err| RasterError::Binding(format!("could not parse .prj CRS: {err}")))
}

/// Return ordered candidate paths for an ESRI world file accompanying
/// `raster_path` (e.g., `photo.jpg` -> `photo.jgw`, `photo.jpgw`, `photo.wld`
/// plus uppercase variants on case-sensitive filesystems).
fn world_file_candidates(raster_path: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Some(ext) = raster_path.extension().and_then(|e| e.to_str()) else {
        return out;
    };
    let lower = ext.to_ascii_lowercase();
    let mut exts: Vec<String> = Vec::new();

    // Standard ESRI 3-letter rule: first + last + 'w' (tif->tfw, jpg->jgw, png->pgw, jpeg->jgw).
    if lower == "jpeg" {
        exts.push("jgw".to_string());
        exts.push("jpegw".to_string());
    } else if lower == "tiff" {
        exts.push("tfw".to_string());
        exts.push("tiffw".to_string());
    } else if lower.len() >= 2 {
        let bytes = lower.as_bytes();
        let short = format!("{}{}w", bytes[0] as char, bytes[bytes.len() - 1] as char);
        exts.push(short);
        exts.push(format!("{lower}w"));
    }
    exts.push("wld".to_string());

    for e in exts {
        out.push(raster_path.with_extension(&e));
        let upper = e.to_ascii_uppercase();
        if upper != e {
            out.push(raster_path.with_extension(upper));
        }
    }
    out
}

/// Return ordered candidate paths for a `.prj` sidecar accompanying `raster_path`.
///
/// Raster-specific `<filename>.prj` / `<filename>.PRJ` take precedence over
/// stem-shared `<stem>.prj` / `<stem>.PRJ` so sibling shapefiles in the same
/// folder never shadow a raster's own `.tif.prj`.
fn prj_file_candidates(raster_path: &Path) -> Vec<PathBuf> {
    let mut out = Vec::with_capacity(4);
    if let Some(name) = raster_path.file_name().and_then(|n| n.to_str()) {
        out.push(raster_path.with_file_name(format!("{name}.prj")));
        out.push(raster_path.with_file_name(format!("{name}.PRJ")));
    }
    out.push(raster_path.with_extension("prj"));
    out.push(raster_path.with_extension("PRJ"));
    out
}

/// Discover and parse any accompanying world file and/or `.prj` sidecar for
/// `raster_path` using a caller-supplied text loader `read_candidate`.
///
/// Keeping filesystem reads injected at the caller seam leaves `veldmap-raster`
/// completely free of `std::fs` calls. `read_candidate(path, MAX_SIDECAR_BYTES)`
/// returns `Ok(None)` when `path` does not exist or is not a regular file,
/// `Ok(Some(text))` when read, or `Err(msg)` on size-limit/read errors.
pub(crate) fn load_sidecar_georef_with<F>(
    raster_path: &Path,
    mut read_candidate: F,
) -> Result<SidecarGeoref, RasterError>
where
    F: FnMut(&Path, u64) -> Result<Option<String>, String>,
{
    let affine =
        match read_first_candidate(&world_file_candidates(raster_path), &mut read_candidate)? {
            Some(world_text) => Some(parse_world_file(&world_text)?),
            None => None,
        };
    let crs = match read_first_candidate(&prj_file_candidates(raster_path), &mut read_candidate)? {
        Some(prj_text) => Some(parse_prj_text(&prj_text)?),
        None => None,
    };
    Ok(SidecarGeoref { affine, crs })
}

fn read_first_candidate<F>(
    candidates: &[PathBuf],
    read_candidate: &mut F,
) -> Result<Option<String>, RasterError>
where
    F: FnMut(&Path, u64) -> Result<Option<String>, String>,
{
    for path in candidates {
        match read_candidate(path, MAX_SIDECAR_BYTES) {
            Ok(Some(text)) => return Ok(Some(text)),
            Ok(None) => continue,
            Err(msg) => return Err(RasterError::Binding(msg)),
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_world_file_with_bom_and_shifts_pixel_center_to_top_left_corner() {
        let tfw = "\u{FEFF}10.0\n0.0\n0.0\n-10.0\n500005.0\n5249995.0\n";
        let affine = parse_world_file(tfw).expect("valid tfw with BOM");
        assert!((affine[0] - 10.0).abs() < 1e-12);
        assert!((affine[1] - 0.0).abs() < 1e-12);
        assert!((affine[2] - 500_000.0).abs() < 1e-12);
        assert!((affine[3] - 0.0).abs() < 1e-12);
        assert!((affine[4] + 10.0).abs() < 1e-12);
        assert!((affine[5] - 5_250_000.0).abs() < 1e-12);
    }

    #[test]
    fn parses_prj_wkt_with_bom_and_prioritizes_raster_specific_prj() {
        let wkt = "\u{FEFF}GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\",6378137,298.257223563]],PRIMEM[\"Greenwich\",0],UNIT[\"degree\",0.0174532925199433]]";
        let crs = parse_prj_text(wkt).expect("valid WGS84 prj with BOM");
        assert!(crs.datum_warning().is_none());

        let prj_cands = prj_file_candidates(Path::new("/tmp/scene.tif"));
        assert_eq!(prj_cands[0], PathBuf::from("/tmp/scene.tif.prj"));
        assert_eq!(prj_cands[1], PathBuf::from("/tmp/scene.tif.PRJ"));
        assert_eq!(prj_cands[2], PathBuf::from("/tmp/scene.prj"));

        let jpg_cands = world_file_candidates(Path::new("/tmp/drone.jpg"));
        assert!(jpg_cands.contains(&PathBuf::from("/tmp/drone.jgw")));
        assert!(jpg_cands.contains(&PathBuf::from("/tmp/drone.wld")));

        let png_cands = world_file_candidates(Path::new("/tmp/ortho.png"));
        assert!(png_cands.contains(&PathBuf::from("/tmp/ortho.pgw")));
    }
}
