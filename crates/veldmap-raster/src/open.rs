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

/// Optional sidecar georeferencing and external band scaling applied when
/// opening a raster via [`open_raster_with_options`].
#[derive(Clone, Copy, Debug, Default)]
pub struct OpenRasterOptions<'a> {
    /// Optional pre-parsed world-file / `.prj` sidecar georeferencing.
    pub sidecars: Option<&'a SidecarGeoref>,
    /// Optional external per-band `(scale, offset)` override (e.g., from STAC
    /// `raster:bands`), applied to indexed bands of GeoTIFF sources whose
    /// embedded metadata left scaling at identity `(1.0, 0.0)`.
    pub external_scale_offset: Option<(Option<f64>, Option<f64>)>,
}

/// Result of opening a raster source via [`open_raster`] or
/// [`open_raster_with_options`].
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

fn sniff_raster_format(
    source: &dyn ByteSource,
    cancel: &CancelToken,
) -> Result<RasterFormat, RasterError> {
    let mut head = [0u8; 16];
    let to_read = source.len().min(head.len() as u64) as usize;
    source.read_exact_at(0, &mut head[..to_read], cancel)?;
    classify_raster_format(&head[..to_read]).ok_or(RasterError::UnsupportedFormat)
}

fn open_classified_raster(
    source: Arc<dyn ByteSource>,
    fingerprint: SourceFingerprint,
    format: RasterFormat,
    options: OpenRasterOptions<'_>,
    budget: &MemoryBudget,
    cancel: &CancelToken,
) -> Result<OpenedRaster, RasterError> {
    match format {
        RasterFormat::Tiff => {
            let sidecar = options.sidecars.filter(|sc| !sc.is_empty()).cloned();
            let mut src =
                TiffSource::open_with_sidecar(source, fingerprint, sidecar, budget, cancel)?;
            if let Some((scale, offset)) = options.external_scale_offset
                && (scale.is_some() || offset.is_some())
            {
                src.apply_external_scale_offset(scale.unwrap_or(1.0), offset.unwrap_or(0.0));
            }
            Ok(OpenedRaster {
                source: Arc::new(src),
                ignored_sidecar_error: None,
            })
        }
        RasterFormat::Png => {
            let placement = options
                .sidecars
                .cloned()
                .and_then(SidecarGeoref::into_placement);
            let src = PngSource::open(source, fingerprint, placement, budget, cancel)?;
            Ok(OpenedRaster {
                source: Arc::new(src),
                ignored_sidecar_error: None,
            })
        }
        RasterFormat::Jpeg => {
            let placement = options
                .sidecars
                .cloned()
                .and_then(SidecarGeoref::into_placement);
            let src = JpegSource::open(source, fingerprint, placement, budget, cancel)?;
            Ok(OpenedRaster {
                source: Arc::new(src),
                ignored_sidecar_error: None,
            })
        }
    }
}

/// Sniff `source`'s file signature (GeoTIFF/BigTIFF/COG, PNG, or JPEG) and
/// open the matching [`RasterSource`] with caller-supplied
/// [`OpenRasterOptions`].
pub fn open_raster_with_options(
    source: Arc<dyn ByteSource>,
    fingerprint: SourceFingerprint,
    options: OpenRasterOptions<'_>,
    budget: &MemoryBudget,
    cancel: &CancelToken,
) -> Result<OpenedRaster, RasterError> {
    let format = sniff_raster_format(source.as_ref(), cancel)?;
    open_classified_raster(source, fingerprint, format, options, budget, cancel)
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
    let format = sniff_raster_format(source.as_ref(), cancel)?;
    let sidecar_res: Option<Result<SidecarGeoref, RasterError>> =
        sidecars.map(|(path, read_candidate)| load_sidecar_georef_with(path, read_candidate));

    let (parsed_sidecar, ignored_sidecar_error) = match (format, sidecar_res) {
        (RasterFormat::Tiff, Some(Ok(sc))) if !sc.is_empty() => (Some(sc), None),
        (RasterFormat::Tiff, Some(Ok(_)) | None) => (None, None),
        (RasterFormat::Tiff, Some(Err(err))) => (None, Some(err)),
        (RasterFormat::Png | RasterFormat::Jpeg, res) => (res.transpose()?, None),
    };

    let mut opened = open_classified_raster(
        source,
        fingerprint,
        format,
        OpenRasterOptions {
            sidecars: parsed_sidecar.as_ref(),
            external_scale_offset: None,
        },
        budget,
        cancel,
    )?;
    opened.ignored_sidecar_error = ignored_sidecar_error;
    Ok(opened)
}

#[cfg(test)]
#[path = "open_tests.rs"]
mod tests;
