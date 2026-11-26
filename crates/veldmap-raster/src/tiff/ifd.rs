//! Runtime cached per-IFD TIFF metadata (`TileOffsets`/`StripOffsets`,
//! `TileByteCounts`/`StripByteCounts`, `JPEGTables`, `ColorMap`, layout, and
//! compression/predictor descriptors) and raw-access value types.
//!
//! Completely free of `tiff::decoder::Decoder`: open-time parsing lives in
//! [`super::ifd_parse`], allowing the direct raw-chunk decoder (`raw_decode`)
//! to execute lock-free positioned reads over [`ByteSource`].

use std::sync::Arc;

use veldmap_core::io::ByteSource;
use veldmap_core::tile::{BandKey, PixelFormat, TileStreamSpec};

use crate::RasterError;
use crate::pipeline::metadata::SampleType;
use crate::pipeline::radiometry::PixelLayout;

use super::ycbcr::YCbCrParams;

/// Maximum uncompressed byte size admitted for any single TIFF strip/tile,
/// matching [`crate::tiff::TiffSource`]'s per-chunk decode cap.
pub(super) const MAX_SINGLE_CHUNK_BYTES: u64 = 256 * 1024 * 1024;

/// TIFF file byte order from header bytes `0..2`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TiffByteOrder {
    Little,
    Big,
}

/// Shared stateless raw-chunk I/O access parameters borrowed by chunk readers,
/// stretch samplers, and cursor probes.
pub(super) struct RawTiffAccess {
    pub(super) byte_source: Arc<dyn ByteSource>,
    pub(super) byte_order: TiffByteOrder,
    pub(super) max_chunk_bytes: usize,
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
                Ok(match (self.pixel_layout.colors(), self.alpha_channel()) {
                    (1, None) => vec![0],
                    (1, Some(a)) => vec![0, a],
                    (_, None) => vec![0, 1, 2],
                    (_, Some(a)) => vec![0, 1, 2, a],
                })
            }
            PixelFormat::R32Float | PixelFormat::R8Uint | PixelFormat::R16Uint => {
                let band_idx = match spec.band {
                    BandKey::Display => 0usize,
                    BandKey::Index(i) => usize::from(i),
                    BandKey::NamedHash(_) => {
                        return Err(RasterError::Decode(
                            "named band hash is not supported by TiffSource".to_owned(),
                        ));
                    }
                };
                if band_idx >= usize::from(self.samples_per_pixel) {
                    return Err(RasterError::Decode(format!(
                        "band index {band_idx} out of range for TIFF with {} samples/pixel",
                        self.samples_per_pixel
                    )));
                }
                Ok(vec![self.band_plane(band_idx)])
            }
        }
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
