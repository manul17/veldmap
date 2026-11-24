//! Format-dispatching local raster opener seam (`open_local_raster`).
//!
//! Dispatches GeoTIFF/BigTIFF/COG, PNG, and JPEG by file magic header, loads
//! optional external world-file (`.tfw`/`.pgw`/`.jgw`/`.wld`) and `.prj`
//! sidecars, and applies optional CRS overrides cleanly off the UI thread.

use std::path::Path;
use std::sync::Arc;

use veldmap_core::budget::MemoryBudget;
use veldmap_core::geo::Frame;
use veldmap_core::io::ByteSource;
use veldmap_core::sync::CancelToken;
use veldmap_raster::{
    JpegSource, PngSource, RasterSource, TiffSource, frame_from_description_with_override,
    has_jpeg_magic, has_png_magic, has_tiff_magic, load_sidecar_placement,
};
use veldmap_storage::{LocalFileSource, LocalFingerprintCache};

use crate::session::SavedCrsOverride;

/// Open a local raster file (GeoTIFF/BigTIFF/COG, PNG, or JPEG + optional
/// world-file/`.prj` sidecars), compute its sampled fingerprint, decode
/// metadata, and bind its geodetic [`Frame`].
pub(super) fn open_local_raster(
    path: &Path,
    fingerprint_cache: &LocalFingerprintCache,
    budget: &MemoryBudget,
    bypass_cache: bool,
    crs_override: Option<&SavedCrsOverride>,
    cancel: &CancelToken,
) -> Result<(Arc<dyn RasterSource>, Frame), String> {
    let fingerprint = fingerprint_cache
        .get_or_compute_with_bypass(path, bypass_cache, cancel)
        .map_err(|e| format!("failed to fingerprint '{}': {e}", path.display()))?;
    let file_source: Arc<dyn ByteSource> = Arc::new(
        LocalFileSource::open(path)
            .map_err(|e| format!("failed to open '{}': {e}", path.display()))?,
    );

    let mut head = [0u8; 16];
    let n = file_source
        .read_at(0, &mut head, cancel)
        .map_err(|e| format!("failed to read header of '{}': {e}", path.display()))?;
    let head = &head[..n];

    let sidecar = load_sidecar_placement(path)
        .map_err(|e| format!("failed to load sidecar for '{}': {e}", path.display()))?;

    let source: Arc<dyn RasterSource> = if has_tiff_magic(head) {
        Arc::new(
            TiffSource::open_with_placement(file_source, fingerprint, sidecar, budget, cancel)
                .map_err(|e| format!("failed to read GeoTIFF '{}': {e}", path.display()))?,
        )
    } else if has_png_magic(head) {
        Arc::new(
            PngSource::open(file_source, fingerprint, sidecar, budget, cancel)
                .map_err(|e| format!("failed to read PNG '{}': {e}", path.display()))?,
        )
    } else if has_jpeg_magic(head) {
        Arc::new(
            JpegSource::open(file_source, fingerprint, sidecar, budget, cancel)
                .map_err(|e| format!("failed to read JPEG '{}': {e}", path.display()))?,
        )
    } else {
        return Err(format!(
            "'{}': unsupported file format (expected GeoTIFF/COG, PNG, JPEG, or GeoJSON)",
            path.display()
        ));
    };

    let resolved_override = match crs_override {
        Some(saved) => Some(
            saved
                .to_crs_spec()
                .map_err(|e| format!("invalid CRS override for '{}': {e}", path.display()))?,
        ),
        None => None,
    };

    let frame =
        frame_from_description_with_override(source.description(), resolved_override.as_ref())
            .map_err(|e| format!("'{}': {e}", path.display()))?;
    Ok((source, frame))
}
