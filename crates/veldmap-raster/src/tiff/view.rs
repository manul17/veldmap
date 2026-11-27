//! Stateless runtime access parameters and borrowed source views shared across
//! TIFF chunk decoding, display stretching, and cursor pixel probing.

use std::sync::Arc;

use veldmap_core::io::ByteSource;
use veldmap_core::tile::SourceId;

use crate::RasterError;

use super::ifd::{CachedIfd, TiffByteOrder};

/// Shared stateless raw-chunk I/O access parameters borrowed by chunk readers,
/// stretch samplers, and cursor probes.
pub(super) struct RawTiffAccess {
    pub(super) byte_source: Arc<dyn ByteSource>,
    pub(super) byte_order: TiffByteOrder,
    pub(super) max_chunk_bytes: usize,
}

/// Borrowed immutable view of an opened [`super::TiffSource`]'s raw I/O
/// access, cached IFD table, and nodata metadata.
#[derive(Clone, Copy)]
pub(super) struct TiffSourceView<'a> {
    pub(super) access: &'a RawTiffAccess,
    pub(super) source_id: SourceId,
    pub(super) ifds: &'a [CachedIfd],
    pub(super) nodata_f64: Option<f64>,
    pub(super) per_band_nodata: &'a [Option<f64>],
}

impl<'a> TiffSourceView<'a> {
    pub(super) fn base_ifd(&self) -> &'a CachedIfd {
        &self.ifds[0]
    }

    /// Resolve declared nodata for 0-based band `band_index`.
    ///
    /// `per_band_nodata` is initialized from file-wide `GDAL_NODATA` at open
    /// time and overlaid with per-band `GDAL_METADATA` / external overrides, so
    /// indexed bands never inherit display-only nodata derived from color bands.
    pub(super) fn nodata_for_band(&self, band_index: usize) -> Option<f64> {
        self.per_band_nodata.get(band_index).copied().flatten()
    }
}

/// Derive a uniform display nodata value across the first `colors` bands when
/// every display color band declares the same nodata sentinel (treating two
/// `NaN` values as equal).
pub(super) fn uniform_display_nodata(
    per_band_nodata: &[Option<f64>],
    colors: usize,
) -> Option<f64> {
    if colors == 0 || per_band_nodata.len() < colors {
        return None;
    }
    let first = per_band_nodata[0]?;
    for &slot in &per_band_nodata[1..colors] {
        let val = slot?;
        let same = (first.is_nan() && val.is_nan()) || first == val;
        if !same {
            return None;
        }
    }
    Some(first)
}

/// Look up cached IFD metadata for directory index `image`.
pub(super) fn find_ifd(ifds: &[CachedIfd], image: usize) -> Result<&CachedIfd, RasterError> {
    ifds.iter()
        .find(|i| i.image == image)
        .ok_or_else(|| RasterError::Decode(format!("TIFF IFD {image} is not cached")))
}
