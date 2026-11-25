//! Full-image pure-`ByteSource` raster adapters (PNG and JPEG) for
//! world-file / `.prj` georeferenced orthophotos and imagery.

mod jpeg;
mod png;

use crate::pipeline::metadata::GeoPlacement;

pub use jpeg::{JpegSource, has_jpeg_magic};
pub(crate) use jpeg::{decode_jpeg_pixels_to_rgb8, decode_jpeg_pixels_to_rgba8};
pub use png::{PngSource, has_png_magic};

pub(super) fn world_file_binding_trouble(
    format_name: &str,
    ext_hint: &str,
    placement: Option<&GeoPlacement>,
) -> Option<String> {
    match placement {
        None => Some(format!(
            "{format_name} has no accompanying world file ({ext_hint}); add a world file + .prj or assign a CRS"
        )),
        Some(p) if p.crs.is_none() => Some(crate::sidecar::missing_prj_trouble(&format!(
            "{format_name} has a world file ({ext_hint})"
        ))),
        Some(_) => None,
    }
}
