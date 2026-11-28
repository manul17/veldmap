//! Stateless runtime access parameters, borrowed source views, and
//! stream/display policy helpers shared across TIFF chunk decoding, display
//! stretching, remote COG range planning, and cursor pixel probing.

use std::sync::Arc;

use veldmap_core::io::ByteSource;
use veldmap_core::tile::{BandKey, PixelFormat, SourceId, TileStreamSpec};

use crate::RasterError;
use crate::pipeline::{DisplayMapping, SampleType};

use super::ifd::{CachedIfd, Photometric, PlanarConfig, TiffByteOrder};

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

/// Build the default unstretched [`DisplayMapping`] for `ifd`, inverting
/// photometric `WhiteIsZero` samples after raw nodata checks.
pub(super) fn identity_display_mapping(ifd: &CachedIfd, nodata: Option<f64>) -> DisplayMapping {
    DisplayMapping::identity(nodata).with_invert(ifd.photometric == Photometric::WhiteIsZero)
}

/// Return whether Display RGBA8 rendering requires percentile stretching
/// (true for >8-bit / non-U8 non-palette rasters).
pub(super) const fn needs_display_stretch(ifd: &CachedIfd) -> bool {
    !matches!(ifd.sample_type, SampleType::U8) && !matches!(ifd.photometric, Photometric::Palette)
}

/// TIFF planes sampled for display percentile stretching (`&[0]` for
/// chunky or 1-color planar, `&[0, 1, 2]` for 3-color planar).
pub(super) const fn stretch_planes(ifd: &CachedIfd) -> &'static [usize] {
    match (ifd.planar, ifd.pixel_layout.colors()) {
        (PlanarConfig::Planar, 3) => &[0, 1, 2],
        _ => &[0],
    }
}

/// Resolve `band` into a validated `0..samples_per_pixel` band index for `ifd`.
pub(super) fn resolve_band_index(ifd: &CachedIfd, band: BandKey) -> Result<usize, RasterError> {
    let idx = match band {
        BandKey::Display => 0usize,
        BandKey::Index(i) => usize::from(i),
        BandKey::NamedHash(_) => {
            return Err(RasterError::Decode(
                "named band hash is not supported by TiffSource".to_owned(),
            ));
        }
    };
    if idx >= usize::from(ifd.samples_per_pixel) {
        return Err(RasterError::Decode(format!(
            "band index {idx} out of range for TIFF with {} samples/pixel",
            ifd.samples_per_pixel
        )));
    }
    Ok(idx)
}

/// Return the planar (`PlanarConfiguration = 2`) source planes required
/// for non-palette Display RGBA8 rendering, avoiding heap allocation.
pub(super) const fn planar_display_planes(ifd: &CachedIfd) -> ([usize; 4], usize) {
    match (ifd.pixel_layout.colors(), ifd.alpha_channel()) {
        (1, None) => ([0, 0, 0, 0], 1),
        (1, Some(a)) => ([0, a, 0, 0], 2),
        (_, None) => ([0, 1, 2, 0], 3),
        (_, Some(a)) => ([0, 1, 2, a], 4),
    }
}

/// Resolve the TIFF planes read for `spec`, shared across
/// [`super::chunk_reader::TiffChunkReader`] and remote COG byte-range
/// planners so plane selection never diverges.
pub(super) fn planes_for_stream(
    ifd: &CachedIfd,
    spec: TileStreamSpec,
) -> Result<Vec<usize>, RasterError> {
    match spec.format {
        PixelFormat::Rgba8UnormSrgb => {
            if ifd.planar == PlanarConfig::Chunky {
                return Ok(vec![0]);
            }
            if ifd.photometric == Photometric::Palette {
                return Ok(match ifd.alpha_channel() {
                    Some(a) => vec![0, a],
                    None => vec![0],
                });
            }
            let (planes, len) = planar_display_planes(ifd);
            Ok(planes[..len].to_vec())
        }
        PixelFormat::R32Float | PixelFormat::R8Uint | PixelFormat::R16Uint => {
            let band_idx = resolve_band_index(ifd, spec.band)?;
            Ok(vec![ifd.band_plane(band_idx)])
        }
    }
}
