//! Format-dispatching local raster opener seam (`open_local_raster`).
//!
//! Dispatches GeoTIFF/BigTIFF/COG, PNG, and JPEG by file magic header, opens
//! and fingerprints the file handle once (closing TOCTOU windows), loads
//! optional external world-file (`.tfw`/`.pgw`/`.jgw`/`.wld`) and `.prj`
//! sidecars via TOCTOU-safe `veldmap-storage` reads, and applies optional CRS
//! overrides cleanly off the UI thread.

use std::path::Path;
use std::sync::Arc;

use veldmap_core::budget::MemoryBudget;
use veldmap_core::geo::Frame;
use veldmap_core::io::ByteSource;
use veldmap_core::sync::CancelToken;
use veldmap_raster::{
    JpegSource, PngSource, RasterDescription, RasterError, RasterSource, TiffSource,
    frame_from_description_with_override, has_jpeg_magic, has_png_magic, has_tiff_magic,
    load_sidecar_georef_with,
};
use veldmap_storage::{LocalFingerprintCache, StorageError, read_regular_file_utf8_limited};

use crate::session::SavedCrsOverride;

/// Structured open failure distinguishing missing/invalid georeferencing (where
/// offering `"Assign CRS…"` in the UI makes sense) from I/O or format errors.
#[derive(Clone, Debug)]
pub(super) struct OpenRasterError {
    pub(super) message: String,
    pub(super) can_assign_crs: bool,
}

impl OpenRasterError {
    fn other(message: String) -> Self {
        Self {
            message,
            can_assign_crs: false,
        }
    }

    fn binding(message: String) -> Self {
        Self {
            message,
            can_assign_crs: true,
        }
    }
}

/// Resolve an optional [`SavedCrsOverride`] and bind `desc` into a [`Frame`].
pub(super) fn bind_frame_with_saved_override(
    desc: &RasterDescription,
    crs_override: Option<&SavedCrsOverride>,
) -> Result<Frame, String> {
    let resolved = match crs_override {
        Some(saved) => Some(saved.to_crs_spec()?),
        None => None,
    };
    frame_from_description_with_override(desc, resolved.as_ref()).map_err(|e| e.to_string())
}

fn read_sidecar_candidate(path: &Path, max_bytes: u64) -> Result<Option<String>, String> {
    match read_regular_file_utf8_limited(path, max_bytes) {
        Ok(text) => Ok(Some(text)),
        Err(StorageError::Io(err)) if err.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(StorageError::InvalidPath(_)) => Ok(None),
        Err(StorageError::FileTooLarge { limit_bytes, .. }) => Err(format!(
            "sidecar '{}' exceeds {limit_bytes} byte size limit",
            path.display()
        )),
        Err(err) => Err(format!(
            "failed to read sidecar '{}': {err}",
            path.display()
        )),
    }
}

/// Open a local raster file (GeoTIFF/BigTIFF/COG, PNG, or JPEG + optional
/// world-file/`.prj` sidecars), compute its sampled fingerprint on the same
/// opened handle, decode metadata, and bind its geodetic [`Frame`].
pub(super) fn open_local_raster(
    path: &Path,
    fingerprint_cache: &LocalFingerprintCache,
    budget: &MemoryBudget,
    bypass_cache: bool,
    crs_override: Option<&SavedCrsOverride>,
    cancel: &CancelToken,
) -> Result<(Arc<dyn RasterSource>, Frame), OpenRasterError> {
    let (local_source, fingerprint) = fingerprint_cache
        .open_and_fingerprint(path, bypass_cache, cancel)
        .map_err(|e| {
            OpenRasterError::other(format!(
                "failed to open/fingerprint '{}': {e}",
                path.display()
            ))
        })?;
    let file_source: Arc<dyn ByteSource> = Arc::new(local_source);

    let mut head = [0u8; 16];
    let n = file_source.read_at(0, &mut head, cancel).map_err(|e| {
        OpenRasterError::other(format!(
            "failed to read header of '{}': {e}",
            path.display()
        ))
    })?;
    let head = &head[..n];

    let sidecar_res = load_sidecar_georef_with(path, read_sidecar_candidate);

    let (source, sidecar_err): (Arc<dyn RasterSource>, Option<RasterError>) =
        if has_tiff_magic(head) {
            // For GeoTIFFs, do not let a corrupt/unrelated sibling `.prj`/`.wld`
            // block opening a valid self-contained GeoTIFF; only surface the
            // sidecar error if embedded georeferencing is missing.
            let (sidecar, err) = match sidecar_res {
                Ok(sc) if !sc.is_empty() => (Some(sc), None),
                Ok(_) => (None, None),
                Err(e) => (None, Some(e)),
            };
            let src =
                TiffSource::open_with_sidecar(file_source, fingerprint, sidecar, budget, cancel)
                    .map_err(|e| {
                        OpenRasterError::other(format!(
                            "failed to read GeoTIFF '{}': {e}",
                            path.display()
                        ))
                    })?;
            (Arc::new(src), err)
        } else if has_png_magic(head) {
            let sidecar = sidecar_res
                .map_err(|e| {
                    OpenRasterError::binding(format!(
                        "failed to load sidecar for '{}': {e}",
                        path.display()
                    ))
                })?
                .into_placement();
            let src = PngSource::open(file_source, fingerprint, sidecar, budget, cancel).map_err(
                |e| OpenRasterError::other(format!("failed to read PNG '{}': {e}", path.display())),
            )?;
            (Arc::new(src), None)
        } else if has_jpeg_magic(head) {
            let sidecar = sidecar_res
                .map_err(|e| {
                    OpenRasterError::binding(format!(
                        "failed to load sidecar for '{}': {e}",
                        path.display()
                    ))
                })?
                .into_placement();
            let src = JpegSource::open(file_source, fingerprint, sidecar, budget, cancel).map_err(
                |e| {
                    OpenRasterError::other(format!("failed to read JPEG '{}': {e}", path.display()))
                },
            )?;
            (Arc::new(src), None)
        } else {
            return Err(OpenRasterError::other(format!(
                "'{}': unsupported file format (expected GeoTIFF/COG, PNG, JPEG, or GeoJSON)",
                path.display()
            )));
        };

    let resolved_override = match crs_override {
        Some(saved) => Some(saved.to_crs_spec().map_err(|e| {
            OpenRasterError::binding(format!(
                "invalid CRS override for '{}': {e}",
                path.display()
            ))
        })?),
        None => None,
    };

    let frame =
        frame_from_description_with_override(source.description(), resolved_override.as_ref())
            .map_err(|e| {
                let can_assign_crs = matches!(e, RasterError::Binding(_))
                    && (source.description().placement.is_some()
                        || !source.description().ties.is_empty());
                let msg = if let Some(sc_err) = sidecar_err.as_ref() {
                    format!("'{}': {e} (sidecar error: {sc_err})", path.display())
                } else {
                    format!("'{}': {e}", path.display())
                };
                OpenRasterError {
                    message: msg,
                    can_assign_crs,
                }
            })?;
    if let Some(sc_err) = sidecar_err {
        tracing::warn!(
            path = %path.display(),
            error = %sc_err,
            "Ignored invalid sibling sidecar because GeoTIFF carries embedded georeferencing"
        );
    }
    Ok((source, frame))
}
