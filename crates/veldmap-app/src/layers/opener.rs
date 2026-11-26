//! Local and remote Cloud-Optimized GeoTIFF raster opener seams
//! (`open_local_raster`, `open_remote_cog_source`).
//!
//! Opens and fingerprints the local file handle once via `veldmap-storage`
//! (closing TOCTOU windows), delegates format sniffing and sidecar precedence
//! to `veldmap_raster::open_raster` with a TOCTOU-safe sidecar reader, and
//! binds the resulting geodetic `Frame` with optional CRS overrides off the UI
//! thread.

use std::path::Path;
use std::sync::Arc;

use veldmap_core::budget::MemoryBudget;
use veldmap_core::geo::Frame;
use veldmap_core::io::ByteSource;
use veldmap_core::sync::CancelToken;
use veldmap_net::HttpRangeSource;
use veldmap_raster::{
    OpenedRaster, RasterDescription, RasterError, RasterSource, TiffSource,
    frame_from_description_with_override, open_raster,
};
use veldmap_storage::{LocalFingerprintCache, StorageError, read_regular_file_utf8_limited};

use super::CrsOverride;

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

/// Resolve an optional [`CrsOverride`] and bind `desc` into a [`Frame`].
pub(super) fn bind_frame_with_override(
    desc: &RasterDescription,
    crs_override: Option<&CrsOverride>,
) -> Result<Frame, String> {
    let resolved = match crs_override {
        Some(ov) => Some(ov.to_crs_spec()?),
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
    crs_override: Option<&CrsOverride>,
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

    let mut read_sidecar = read_sidecar_candidate;
    let OpenedRaster {
        source,
        ignored_sidecar_error: sidecar_err,
    } = open_raster(
        file_source,
        fingerprint,
        Some((path, &mut read_sidecar)),
        budget,
        cancel,
    )
    .map_err(|err| match err {
        RasterError::UnsupportedFormat => OpenRasterError::other(format!(
            "'{}': unsupported file format (expected GeoTIFF/COG, PNG, JPEG, or GeoJSON)",
            path.display()
        )),
        RasterError::Binding(msg) => OpenRasterError::binding(format!(
            "failed to load sidecar for '{}': {msg}",
            path.display()
        )),
        other => OpenRasterError::other(format!(
            "failed to read raster '{}': {other}",
            path.display()
        )),
    })?;

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
                let can_assign_crs =
                    matches!(e, RasterError::Binding(_)) && source.description().can_override_crs();
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

/// Open a remote Cloud-Optimized GeoTIFF over an already-probed
/// [`HttpRangeSource`], applying optional STAC `raster:bands` `(scale, offset)`
/// metadata when the TIFF header lacks embedded GDAL scale/offset tags.
pub(super) fn open_remote_cog_source(
    range_source: HttpRangeSource,
    scale_offset: Option<(f64, f64)>,
    budget: &MemoryBudget,
    cancel: &CancelToken,
) -> Result<(Arc<dyn RasterSource>, Frame), OpenRasterError> {
    let fingerprint = range_source.fingerprint();
    let byte_source: Arc<dyn ByteSource> = Arc::new(range_source);
    let mut tiff =
        TiffSource::open(byte_source, fingerprint, budget, cancel).map_err(|err| match err {
            RasterError::UnsupportedFormat => OpenRasterError::other(
                "remote asset is not a valid GeoTIFF / Cloud-Optimized GeoTIFF".to_owned(),
            ),
            RasterError::RemotePassUnsupported(reason) => OpenRasterError::other(format!(
                "remote GeoTIFF is not cloud-optimized (requires tiled pointwise access): {reason}"
            )),
            other => OpenRasterError::other(format!("failed to read remote COG: {other}")),
        })?;
    if let Some((scale, offset)) = scale_offset {
        tiff.apply_external_scale_offset(scale, offset);
    }
    let source: Arc<dyn RasterSource> = Arc::new(tiff);
    let frame = frame_from_description_with_override(source.description(), None)
        .map_err(|e| OpenRasterError::other(format!("remote COG georeference error: {e}")))?;
    Ok((source, frame))
}
