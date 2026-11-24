//! Full-image pure-`ByteSource` raster adapters (PNG and JPEG) for
//! world-file / `.prj` georeferenced orthophotos and imagery.

mod jpeg;
mod png;

pub use jpeg::{JpegSource, has_jpeg_magic};
pub use png::{PngSource, has_png_magic};
