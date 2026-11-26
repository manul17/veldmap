//! Format-dispatching raster opener (`open_raster`) classifying GeoTIFF/COG,
//! PNG, and JPEG by magic bytes and applying format-specific sidecar
//! precedence while keeping `veldmap-raster` completely free of `std::fs`.

use std::path::Path;
use std::sync::Arc;

use veldmap_core::budget::MemoryBudget;
use veldmap_core::io::ByteSource;
use veldmap_core::sync::CancelToken;
use veldmap_core::tile::SourceFingerprint;

use crate::RasterError;
use crate::adapters::{JpegSource, PngSource};
use crate::image_codec::{has_jpeg_magic, has_png_magic};
use crate::pipeline::RasterSource;
use crate::sidecar::{SidecarGeoref, load_sidecar_georef_with};
use crate::tiff::{TiffSource, has_tiff_magic};

/// Callback signature for caller-injected sidecar text reads, keeping
/// `veldmap-raster` free of filesystem dependencies.
pub type SidecarReader<'a> = &'a mut dyn FnMut(&Path, u64) -> Result<Option<String>, String>;

/// Result of opening a raster source via [`open_raster`].
pub struct OpenedRaster {
    pub source: Arc<dyn RasterSource>,
    /// Non-fatal sidecar load/parse error ignored because a self-describing
    /// format (GeoTIFF) opened cleanly. Callers should surface this if
    /// embedded georeferencing is absent when binding a `Frame`, or log it as
    /// a warning when embedded georeferencing succeeds.
    pub ignored_sidecar_error: Option<RasterError>,
}

#[derive(Clone, Copy)]
enum RasterFormat {
    Tiff,
    Png,
    Jpeg,
}

fn classify_raster_format(head: &[u8]) -> Option<RasterFormat> {
    if has_tiff_magic(head) {
        Some(RasterFormat::Tiff)
    } else if has_png_magic(head) {
        Some(RasterFormat::Png)
    } else if has_jpeg_magic(head) {
        Some(RasterFormat::Jpeg)
    } else {
        None
    }
}

/// Sniff `source`'s file signature (GeoTIFF/BigTIFF/COG, PNG, or JPEG),
/// optionally probe world-file/`.prj` sidecars via an injected callback, and
/// open the matching [`RasterSource`].
///
/// Sidecars are probed only after magic-byte classification succeeds so
/// unsupported files never trigger wasted I/O. For GeoTIFFs, a corrupt sibling
/// sidecar does not block opening a valid self-contained GeoTIFF; instead, the
/// error is returned in [`OpenedRaster::ignored_sidecar_error`]. For PNG and
/// JPEG, any sidecar read/parse failure is returned directly as
/// [`RasterError::Binding`].
pub fn open_raster(
    source: Arc<dyn ByteSource>,
    fingerprint: SourceFingerprint,
    sidecars: Option<(&Path, SidecarReader<'_>)>,
    budget: &MemoryBudget,
    cancel: &CancelToken,
) -> Result<OpenedRaster, RasterError> {
    let mut head = [0u8; 16];
    let to_read = source.len().min(head.len() as u64) as usize;
    source.read_exact_at(0, &mut head[..to_read], cancel)?;
    let format = classify_raster_format(&head[..to_read]).ok_or(RasterError::UnsupportedFormat)?;

    let sidecar_res: Option<Result<SidecarGeoref, RasterError>> =
        sidecars.map(|(path, read_candidate)| load_sidecar_georef_with(path, read_candidate));

    match format {
        RasterFormat::Tiff => {
            let (sidecar, ignored_sidecar_error) = match sidecar_res {
                Some(Ok(sc)) if !sc.is_empty() => (Some(sc), None),
                Some(Ok(_)) | None => (None, None),
                Some(Err(err)) => (None, Some(err)),
            };
            let src = TiffSource::open_with_sidecar(source, fingerprint, sidecar, budget, cancel)?;
            Ok(OpenedRaster {
                source: Arc::new(src),
                ignored_sidecar_error,
            })
        }
        RasterFormat::Png => {
            if source.is_remote() {
                return Err(RasterError::RemotePassUnsupported(
                    "PNG requires whole-file decoding and cannot be streamed over HTTP Range"
                        .to_owned(),
                ));
            }
            let placement = sidecar_res
                .transpose()?
                .and_then(SidecarGeoref::into_placement);
            let src = PngSource::open(source, fingerprint, placement, budget, cancel)?;
            Ok(OpenedRaster {
                source: Arc::new(src),
                ignored_sidecar_error: None,
            })
        }
        RasterFormat::Jpeg => {
            if source.is_remote() {
                return Err(RasterError::RemotePassUnsupported(
                    "JPEG requires whole-file decoding and cannot be streamed over HTTP Range"
                        .to_owned(),
                ));
            }
            let placement = sidecar_res
                .transpose()?
                .and_then(SidecarGeoref::into_placement);
            let src = JpegSource::open(source, fingerprint, placement, budget, cancel)?;
            Ok(OpenedRaster {
                source: Arc::new(src),
                ignored_sidecar_error: None,
            })
        }
    }
}

#[cfg(test)]
#[path = "open_tests.rs"]
mod tests;
