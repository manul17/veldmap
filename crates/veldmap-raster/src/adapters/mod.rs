//! Full-image pure-`ByteSource` raster adapters (PNG and JPEG) for
//! world-file / `.prj` georeferenced orthophotos and imagery.

mod jpeg;
mod png;

use veldmap_core::io::IoError;
use veldmap_core::sync::CancelToken;
use veldmap_core::tile::{BandKey, PixelFormat, TileStreamSpec};

use crate::RasterError;
use crate::pipeline::metadata::GeoPlacement;

pub use jpeg::{JpegSource, has_jpeg_magic};
pub use png::{PngSource, has_png_magic};

const ADAPTER_DECODE_REV: u32 = 1;

pub(super) const DISPLAY_STREAM_SPEC: TileStreamSpec = TileStreamSpec::new(
    BandKey::Display,
    PixelFormat::Rgba8UnormSrgb,
    0,
    ADAPTER_DECODE_REV,
);

/// Map a `std::io::Error` produced by a decoder reading from
/// [`veldmap_core::io::BufferedByteReader`] back into [`RasterError::Cancelled`]
/// or [`RasterError::Io`], preserving structured core I/O errors.
pub(crate) fn map_std_io_error(io_err: std::io::Error, cancel: &CancelToken) -> RasterError {
    if cancel.is_cancelled() {
        return RasterError::Cancelled;
    }
    match io_err.downcast::<IoError>() {
        Ok(core_err) => RasterError::from(core_err),
        Err(io_err) => RasterError::from(IoError::Io(io_err)),
    }
}

pub(super) fn world_file_binding_trouble(
    format_name: &str,
    ext_hint: &str,
    placement: Option<&GeoPlacement>,
) -> Option<String> {
    match placement {
        None => Some(format!(
            "{format_name} has no accompanying world file ({ext_hint}); add a world file + .prj or assign a CRS"
        )),
        Some(p) if p.epsg.is_none() && p.custom_crs.is_none() => Some(format!(
            "{format_name} has a world file ({ext_hint}) but no accompanying .prj CRS sidecar; assign a CRS to place it"
        )),
        Some(_) => None,
    }
}
