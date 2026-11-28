//! Full-image pure-`ByteSource` raster adapters (PNG and JPEG) for
//! world-file / `.prj` georeferenced orthophotos and imagery.

mod jpeg;
mod png;

use crate::pipeline::GeoPlacement;

pub use jpeg::JpegSource;
pub use png::PngSource;

pub(super) fn world_file_binding_trouble(
    format_name: &str,
    ext_hint: &str,
    placement: Option<&GeoPlacement>,
) -> Option<String> {
    match placement {
        None => Some(format!(
            "{format_name} has no accompanying world file ({ext_hint}); add a world file + .prj to place it on the globe"
        )),
        Some(p) if p.crs.is_none() => Some(crate::sidecar::missing_prj_trouble(&format!(
            "{format_name} has a world file ({ext_hint})"
        ))),
        Some(_) => None,
    }
}
