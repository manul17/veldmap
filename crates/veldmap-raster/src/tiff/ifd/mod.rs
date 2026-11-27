//! Runtime cached per-IFD TIFF metadata (`TileOffsets`/`StripOffsets`,
//! `TileByteCounts`/`StripByteCounts`, `JPEGTables`, `ColorMap`, layout, and
//! compression/predictor descriptors) and raw-access value types.
//!
//! Open-time parsing lives in [`parse`] over [`wire::SinglePassIfdReader`],
//! allowing the direct raw-chunk decoder (`raw_decode`) to execute lock-free
//! positioned reads over [`ByteSource`].

pub(super) mod budget;
pub(super) mod parse;
mod tag_decode;
pub(super) mod tag_ids;
pub(super) mod wire;

use std::sync::Arc;

use veldmap_core::io::ByteSource;
use veldmap_core::tile::{BandKey, PixelFormat, SourceId, TileStreamSpec};

use crate::RasterError;
use crate::pipeline::{PixelLayout, SampleType};

use super::ycbcr::YCbCrParams;

pub(super) use wire::TiffByteOrder;

/// Maximum uncompressed byte size admitted for any single TIFF strip/tile,
/// matching [`crate::tiff::TiffSource`]'s per-chunk decode cap.
pub(super) const MAX_SINGLE_CHUNK_BYTES: u64 = 256 * 1024 * 1024;

/// Return whether `NewSubfileType` marks a reduced-resolution overview (and
/// not a transparency mask).
pub(in crate::tiff) fn is_overview(subfile_type: u32) -> bool {
    const REDUCED: u32 = 1;
    const MASK: u32 = 4;
    subfile_type & REDUCED != 0 && subfile_type & MASK == 0
}

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

/// Stored (on-disk padded) and valid (cropped) dimensions of a single TIFF chunk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ChunkDims {
    pub(super) stored_w: u32,
    pub(super) stored_h: u32,
    pub(super) data_w: u32,
    pub(super) data_h: u32,
}

/// Chunk addressing coordinates for a base-IFD pixel `(px, py)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ChunkPixelLoc {
    pub(super) chunk_index: u32,
    pub(super) local_x: u32,
    pub(super) local_y: u32,
    pub(super) dims: ChunkDims,
}

/// Validated non-sparse TIFF chunk byte span and uncompressed storage size.
#[derive(Clone, Copy, Debug)]
pub(super) struct ValidatedChunkSpan {
    pub(super) offset: u64,
    pub(super) byte_count: u64,
    pub(super) stored_bytes: usize,
}

/// Supported TIFF compression codecs for direct raw-chunk decoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TiffCompression {
    None,
    Lzw,
    Jpeg,
    Deflate,
    PackBits,
    Zstd,
    WebP,
}

impl TiffCompression {
    pub(super) fn from_code(code: u16) -> Result<Self, RasterError> {
        match code {
            1 => Ok(Self::None),
            5 => Ok(Self::Lzw),
            7 => Ok(Self::Jpeg),
            8 | 0x80B2 => Ok(Self::Deflate),
            0x8005 => Ok(Self::PackBits),
            0xC350 => Ok(Self::Zstd),
            0xC351 => Ok(Self::WebP),
            other => Err(RasterError::Decode(format!(
                "unsupported TIFF compression code {other}"
            ))),
        }
    }
}

/// TIFF horizontal / floating-point predictor mode (Tag `317`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TiffPredictor {
    None,
    Horizontal,
    FloatingPoint,
}

impl TiffPredictor {
    pub(super) fn from_code(code: u16, sample_type: SampleType) -> Result<Self, RasterError> {
        match code {
            1 => Ok(Self::None),
            2 => Ok(Self::Horizontal),
            3 => match sample_type {
                SampleType::F16 | SampleType::F32 | SampleType::F64 => Ok(Self::FloatingPoint),
                _ => Err(RasterError::Decode(format!(
                    "floating-point predictor (3) is incompatible with {sample_type:?}"
                ))),
            },
            other => Err(RasterError::Decode(format!(
                "unsupported TIFF predictor code {other}"
            ))),
        }
    }
}

/// TIFF sample interleaving (`PlanarConfiguration`, Tag `284`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PlanarConfig {
    Chunky,
    Planar,
}

/// TIFF photometric interpretation (Tag `262`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Photometric {
    WhiteIsZero,
    BlackIsZero,
    Rgb,
    Palette,
    YCbCr,
    Other(u16),
}

impl Photometric {
    pub(super) fn from_code(code: u16) -> Self {
        match code {
            0 => Self::WhiteIsZero,
            1 => Self::BlackIsZero,
            2 => Self::Rgb,
            3 => Self::Palette,
            6 => Self::YCbCr,
            other => Self::Other(other),
        }
    }
}

/// Cached immutable metadata for one TIFF IFD (base image or overview).
#[derive(Clone, Debug)]
pub(super) struct CachedIfd {
    pub(super) image: usize,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) tiled: bool,
    pub(super) chunk_w: u32,
    pub(super) chunk_h: u32,
    pub(super) chunks_across: u32,
    pub(super) chunks_per_plane: u32,
    pub(super) compression: TiffCompression,
    pub(super) predictor: TiffPredictor,
    pub(super) planar: PlanarConfig,
    pub(super) photometric: Photometric,
    pub(super) samples_per_pixel: u16,
    pub(super) sample_type: SampleType,
    pub(super) pixel_layout: PixelLayout,
    pub(super) data_bands: u16,
    pub(super) offsets: Vec<u64>,
    pub(super) byte_counts: Vec<u64>,
    pub(super) jpeg_tables: Option<Arc<[u8]>>,
    pub(super) ycbcr: Option<YCbCrParams>,
    pub(super) color_map: Option<Arc<[[u8; 4]]>>,
}

impl CachedIfd {
    /// Approximate heap bytes retained by this cached IFD's offset/count/table arrays.
    pub(super) fn table_bytes(&self) -> u64 {
        let offsets_bytes = (self.offsets.len() as u64).saturating_mul(8);
        let counts_bytes = (self.byte_counts.len() as u64).saturating_mul(8);
        let jpeg_bytes = self.jpeg_tables.as_ref().map_or(0, |t| t.len() as u64);
        let cmap_bytes = self
            .color_map
            .as_ref()
            .map_or(0, |c| (c.len() as u64).saturating_mul(4));
        offsets_bytes
            .saturating_add(counts_bytes)
            .saturating_add(jpeg_bytes)
            .saturating_add(cmap_bytes)
    }

    /// Valid (cropped) and stored (on-disk) dimensions for `chunk_index` in a plane.
    pub(super) fn chunk_dims(&self, chunk_index: u32) -> Option<ChunkDims> {
        if chunk_index >= self.chunks_per_plane || self.chunks_across == 0 {
            return None;
        }
        let cx = chunk_index % self.chunks_across;
        let cy = chunk_index / self.chunks_across;
        let x0 = cx.checked_mul(self.chunk_w)?;
        let y0 = cy.checked_mul(self.chunk_h)?;
        if x0 >= self.width || y0 >= self.height {
            return None;
        }
        let data_w = self.chunk_w.min(self.width - x0);
        let data_h = self.chunk_h.min(self.height - y0);
        if data_w == 0 || data_h == 0 {
            return None;
        }
        let (stored_w, stored_h) = if self.tiled {
            (self.chunk_w, self.chunk_h)
        } else {
            (data_w, data_h)
        };
        Some(ChunkDims {
            stored_w,
            stored_h,
            data_w,
            data_h,
        })
    }

    /// Resolve the chunk index, local pixel offset, and chunk dimensions for
    /// image pixel `(px, py)`.
    pub(super) fn chunk_loc(&self, px: u32, py: u32) -> Result<ChunkPixelLoc, RasterError> {
        if px >= self.width || py >= self.height {
            return Err(RasterError::Decode(format!(
                "pixel ({px}, {py}) out of bounds for IFD {} ({}x{})",
                self.image, self.width, self.height
            )));
        }
        let (cw, ch) = (self.chunk_w.max(1), self.chunk_h.max(1));
        let cx = px / cw;
        let cy = py / ch;
        let chunk_index = cy
            .checked_mul(self.chunks_across)
            .and_then(|row| row.checked_add(cx))
            .ok_or_else(|| RasterError::Decode("chunk index overflow".to_owned()))?;
        let dims = self.chunk_dims(chunk_index).ok_or_else(|| {
            RasterError::Decode(format!(
                "chunk index {chunk_index} out of range for IFD {}",
                self.image
            ))
        })?;
        Ok(ChunkPixelLoc {
            chunk_index,
            local_x: px % cw,
            local_y: py % ch,
            dims,
        })
    }

    /// Number of interleaved samples stored per pixel within a single chunk
    /// (`samples_per_pixel` for `Chunky`, `1` for `Planar`).
    pub(super) const fn samples_in_chunk(&self) -> usize {
        match self.planar {
            PlanarConfig::Chunky => self.samples_per_pixel as usize,
            PlanarConfig::Planar => 1,
        }
    }

    /// Raw bytes per pixel across all channels (`samples_per_pixel * sample_bytes`).
    pub(super) const fn raw_pixel_bytes(&self) -> u32 {
        (self.samples_per_pixel as u32) * (self.sample_type.byte_size() as u32)
    }

    /// Raw bytes per pixel decoded when reading a single band (`sample_bytes`
    /// for `PlanarConfig::Planar`, full `raw_pixel_bytes` for `Chunky`).
    pub(super) const fn band_raw_pixel_bytes(&self) -> u32 {
        match self.planar {
            PlanarConfig::Planar => self.sample_type.byte_size() as u32,
            PlanarConfig::Chunky => self.raw_pixel_bytes(),
        }
    }

    /// TIFF plane index holding `band` (`band` for `Planar`, `0` for `Chunky`).
    pub(super) const fn band_plane(&self, band: usize) -> usize {
        match self.planar {
            PlanarConfig::Planar => band,
            PlanarConfig::Chunky => 0,
        }
    }

    /// Interleaved `(sample_stride, channel_offset)` within a decoded chunk for `band`.
    pub(super) const fn band_stride_offset(&self, band: usize) -> (usize, usize) {
        match self.planar {
            PlanarConfig::Planar => (1, 0),
            PlanarConfig::Chunky => (self.samples_per_pixel as usize, band),
        }
    }

    /// Return `Some(YCbCrParams)` when reading single RGB band `band < 3` from
    /// an 8-bit `YCbCr` IFD.
    pub(super) const fn ycbcr_band_params(&self, band: usize) -> Option<YCbCrParams> {
        if matches!(self.sample_type, SampleType::U8) && band < 3 {
            self.ycbcr
        } else {
            None
        }
    }

    /// Explicit alpha channel index within a pixel, if present.
    pub(super) const fn alpha_channel(&self) -> Option<usize> {
        self.pixel_layout.alpha_channel()
    }

    /// Return whether Display RGBA8 rendering requires percentile stretching
    /// (true for >8-bit / non-U8 non-palette rasters).
    pub(super) const fn needs_display_stretch(&self) -> bool {
        !matches!(self.sample_type, SampleType::U8)
            && !matches!(self.photometric, Photometric::Palette)
    }

    /// TIFF planes sampled for display percentile stretching (`&[0]` for
    /// chunky or 1-color planar, `&[0, 1, 2]` for 3-color planar).
    pub(super) const fn stretch_planes(&self) -> &'static [usize] {
        match (self.planar, self.pixel_layout.colors()) {
            (PlanarConfig::Planar, 3) => &[0, 1, 2],
            _ => &[0],
        }
    }

    /// Resolve `band` into a validated `0..samples_per_pixel` band index.
    pub(super) fn resolve_band_index(&self, band: BandKey) -> Result<usize, RasterError> {
        let idx = match band {
            BandKey::Display => 0usize,
            BandKey::Index(i) => usize::from(i),
            BandKey::NamedHash(_) => {
                return Err(RasterError::Decode(
                    "named band hash is not supported by TiffSource".to_owned(),
                ));
            }
        };
        if idx >= usize::from(self.samples_per_pixel) {
            return Err(RasterError::Decode(format!(
                "band index {idx} out of range for TIFF with {} samples/pixel",
                self.samples_per_pixel
            )));
        }
        Ok(idx)
    }

    /// Resolve the TIFF planes read for `spec`, shared across
    /// [`super::chunk_reader::TiffChunkReader`] and remote COG byte-range
    /// planners so plane selection never diverges.
    pub(super) fn planes_for_stream(
        &self,
        spec: TileStreamSpec,
    ) -> Result<Vec<usize>, RasterError> {
        match spec.format {
            PixelFormat::Rgba8UnormSrgb => {
                if self.planar == PlanarConfig::Chunky {
                    return Ok(vec![0]);
                }
                if self.photometric == Photometric::Palette {
                    return Ok(match self.alpha_channel() {
                        Some(a) => vec![0, a],
                        None => vec![0],
                    });
                }
                let (planes, len) = self.planar_display_planes();
                Ok(planes[..len].to_vec())
            }
            PixelFormat::R32Float | PixelFormat::R8Uint | PixelFormat::R16Uint => {
                let band_idx = self.resolve_band_index(spec.band)?;
                Ok(vec![self.band_plane(band_idx)])
            }
        }
    }

    /// Return the planar (`PlanarConfiguration = 2`) source planes required
    /// for non-palette Display RGBA8 rendering, avoiding heap allocation.
    pub(super) const fn planar_display_planes(&self) -> ([usize; 4], usize) {
        match (self.pixel_layout.colors(), self.alpha_channel()) {
            (1, None) => ([0, 0, 0, 0], 1),
            (1, Some(a)) => ([0, a, 0, 0], 2),
            (_, None) => ([0, 1, 2, 0], 3),
            (_, Some(a)) => ([0, 1, 2, a], 4),
        }
    }

    /// Validate a chunk's uncompressed size, compression-bomb cap, and backing
    /// file bounds, returning `Ok(None)` for sparse (`offset == 0 || len == 0`)
    /// chunks. Shared between remote range planning and raw-chunk decoding.
    pub(super) fn validated_chunk_span(
        &self,
        plane: usize,
        chunk_index: u32,
        dims: ChunkDims,
        max_uncompressed_bytes: usize,
        file_len: u64,
    ) -> Result<Option<ValidatedChunkSpan>, RasterError> {
        let (offset, byte_count) = self.chunk_span(plane, chunk_index)?;
        if offset == 0 || byte_count == 0 {
            return Ok(None);
        }
        let samples_in_chunk = self.samples_in_chunk();
        let sample_bytes = self.sample_type.byte_size();
        let stored_row_bytes = (dims.stored_w as usize)
            .checked_mul(samples_in_chunk)
            .and_then(|n| n.checked_mul(sample_bytes))
            .ok_or_else(|| RasterError::Decode("TIFF chunk row byte size overflow".to_owned()))?;
        let stored_bytes = stored_row_bytes
            .checked_mul(dims.stored_h as usize)
            .ok_or_else(|| RasterError::Decode("TIFF chunk total byte size overflow".to_owned()))?;
        if stored_bytes > max_uncompressed_bytes {
            return Err(RasterError::LimitExceeded {
                width: dims.stored_w,
                height: dims.stored_h,
                estimated_bytes: stored_bytes as u64,
                limit_bytes: max_uncompressed_bytes as u64,
            });
        }
        // Cap compressed input read size relative to this chunk's `stored_bytes`
        // (never the global 256 MiB chunk cap) to prevent crafted multi-hundred-MiB
        // allocations on tiny tiles.
        let max_compressed = stored_bytes
            .saturating_mul(2)
            .max(4 * 1024 * 1024)
            .min(max_uncompressed_bytes.saturating_mul(2).max(64 * 1024))
            as u64;
        if byte_count > max_compressed {
            return Err(RasterError::LimitExceeded {
                width: dims.stored_w,
                height: dims.stored_h,
                estimated_bytes: byte_count,
                limit_bytes: max_compressed,
            });
        }
        if offset
            .checked_add(byte_count)
            .is_none_or(|end| end > file_len)
        {
            return Err(RasterError::Decode(format!(
                "TIFF chunk span {offset}..+{byte_count} exceeds file length {file_len}"
            )));
        }
        Ok(Some(ValidatedChunkSpan {
            offset,
            byte_count,
            stored_bytes,
        }))
    }

    /// Look up `(offset, byte_count)` for `(plane, chunk_index)`.
    pub(super) fn chunk_span(
        &self,
        plane: usize,
        chunk_index: u32,
    ) -> Result<(u64, u64), RasterError> {
        let flat = match self.planar {
            PlanarConfig::Chunky => chunk_index as usize,
            PlanarConfig::Planar => plane
                .checked_mul(self.chunks_per_plane as usize)
                .and_then(|base| base.checked_add(chunk_index as usize))
                .ok_or_else(|| RasterError::Decode("planar chunk index overflow".to_owned()))?,
        };
        let offset = *self.offsets.get(flat).ok_or_else(|| {
            RasterError::Decode(format!(
                "TIFF chunk offset {flat} out of range ({} entries)",
                self.offsets.len()
            ))
        })?;
        let count = *self.byte_counts.get(flat).ok_or_else(|| {
            RasterError::Decode(format!(
                "TIFF chunk byte count {flat} out of range ({} entries)",
                self.byte_counts.len()
            ))
        })?;
        Ok((offset, count))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overview_bit_filters_reduced_resolution_and_excludes_masks() {
        assert!(!is_overview(0));
        assert!(is_overview(1));
        assert!(!is_overview(5)); // mask bit set
    }
}
