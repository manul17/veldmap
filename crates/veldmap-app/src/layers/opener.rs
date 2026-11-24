//! Format-dispatching local raster opener seam (`open_local_raster`).
//!
//! Isolates file fingerprinting, `ByteSource` construction, codec selection, and
//! georeference `Frame` binding from [`super::io_worker::OverlayIoWorker`] so
//! additional local formats (e.g. PNG/JPEG + world-file/`.prj` sidecars) plug
//! in cleanly without coupling I/O worker lifecycle to individual decoders.

use std::path::Path;
use std::sync::Arc;

use veldmap_core::budget::MemoryBudget;
use veldmap_core::geo::Frame;
use veldmap_core::io::ByteSource;
use veldmap_core::sync::CancelToken;
use veldmap_raster::{RasterSource, TiffSource, frame_from_description};
use veldmap_storage::{LocalFileSource, LocalFingerprintCache};

/// Open a local raster file, compute its sampled fingerprint, decode metadata,
/// and bind its geodetic [`Frame`].
pub(super) fn open_local_raster(
    path: &Path,
    fingerprint_cache: &LocalFingerprintCache,
    budget: &MemoryBudget,
    bypass_cache: bool,
    cancel: &CancelToken,
) -> Result<(Arc<dyn RasterSource>, Frame), String> {
    let fingerprint = fingerprint_cache
        .get_or_compute_with_bypass(path, bypass_cache, cancel)
        .map_err(|e| format!("failed to fingerprint '{}': {e}", path.display()))?;
    let file_source: Arc<dyn ByteSource> = Arc::new(
        LocalFileSource::open(path)
            .map_err(|e| format!("failed to open '{}': {e}", path.display()))?,
    );
    let tiff = TiffSource::open(file_source, fingerprint, budget, cancel)
        .map_err(|e| format!("failed to read GeoTIFF '{}': {e}", path.display()))?;
    let frame = frame_from_description(tiff.description())
        .map_err(|e| format!("'{}': {e}", path.display()))?;
    Ok((Arc::new(tiff), frame))
}
