//! Stateless direct raw-chunk TIFF reader bridging [`ByteSource`] +
//! [`CachedIfd`] to [`Chunked`] and [`SharedChunkCache`] across `Display`,
//! `Scientific` (`R32Float`), and `Categorical` (`R8Uint`/`R16Uint`) streams.

use std::sync::Arc;

use veldmap_core::budget::DecodeBudgetTracker;
use veldmap_core::io::ByteSource;
use veldmap_core::sync::CancelToken;
use veldmap_core::tile::{BandKey, PixelFormat, SourceId, TileStreamSpec};

use crate::RasterError;
use crate::pipeline::chunk_cache::{ChunkKey, DecodedChunk, SharedChunkCache};
use crate::pipeline::driver::Chunked;
use crate::pipeline::grid;
use crate::pipeline::metadata::SampleType;
use crate::pipeline::radiometry::{
    DisplayMapping, is_data_f64, raw_bytes_to_rgba, read_native_sample_f64,
};

use super::ifd::{CachedIfd, Photometric, PlanarConfig, TiffByteOrder};
use super::raw_decode::{RawChunkPayload, decode_raw_chunk};

/// Explicitly models the two chunk-reading modes of [`TiffChunkReader`].
pub(super) enum ChunkReadMode<'a> {
    /// Pointwise random access: deduplicates through `chunk_cache` and reserves
    /// each fresh chunk on `budget_tracker`.
    Pointwise {
        chunk_cache: &'a SharedChunkCache,
        budget_tracker: &'a DecodeBudgetTracker,
    },
    /// Sequential full-IFD pass: bypasses `SharedChunkCache` and per-chunk
    /// budget reservation because the caller already holds the full pass lease.
    SequentialPass,
}

/// Stateless raw-chunk reader producing typed [`DecodedChunk`]s for a specific
/// [`TileStreamSpec`].
pub(super) struct TiffChunkReader<'a> {
    byte_source: &'a Arc<dyn ByteSource>,
    byte_order: TiffByteOrder,
    source_id: SourceId,
    ifds: &'a [CachedIfd],
    spec: TileStreamSpec,
    nodata_f64: Option<f64>,
    display_mapping: Option<DisplayMapping>,
    max_chunk_bytes: usize,
    mode: ChunkReadMode<'a>,
}

impl<'a> TiffChunkReader<'a> {
    #[expect(clippy::too_many_arguments, reason = "internal reader constructor")]
    pub(super) fn new(
        byte_source: &'a Arc<dyn ByteSource>,
        byte_order: TiffByteOrder,
        source_id: SourceId,
        ifds: &'a [CachedIfd],
        spec: TileStreamSpec,
        nodata_f64: Option<f64>,
        display_mapping: Option<DisplayMapping>,
        max_chunk_bytes: usize,
        mode: ChunkReadMode<'a>,
    ) -> Self {
        Self {
            byte_source,
            byte_order,
            source_id,
            ifds,
            spec,
            nodata_f64,
            display_mapping,
            max_chunk_bytes,
            mode,
        }
    }

    fn ifd_for(&self, image: usize) -> Result<&'a CachedIfd, RasterError> {
        self.ifds
            .iter()
            .find(|i| i.image == image)
            .ok_or_else(|| RasterError::Decode(format!("TIFF IFD {image} is not cached")))
    }

    fn decode_chunk_uncached(
        &self,
        image: usize,
        index: u32,
        budget_tracker: Option<&DecodeBudgetTracker>,
        cancel: &CancelToken,
    ) -> Result<DecodedChunk, RasterError> {
        let ifd = self.ifd_for(image)?;
        let (dw, dh) = ifd.chunk_data_dims(index).ok_or_else(|| {
            RasterError::Decode(format!(
                "chunk index {index} out of range for IFD {image} ({} chunks)",
                ifd.chunks_per_plane
            ))
        })?;
        let out_bpp = self.spec.format.bytes_per_pixel() as u32;
        let raw_depth = match (self.spec.format, ifd.planar) {
            (PixelFormat::Rgba8UnormSrgb, _) => ifd.raw_pixel_bytes(),
            (_, PlanarConfig::Planar) => ifd.sample_type.byte_size() as u32,
            (_, PlanarConfig::Chunky) => ifd.raw_pixel_bytes(),
        };
        let est_bytes = grid::chunk_working_bytes(dw, dh, raw_depth, out_bpp);
        let _lease = match budget_tracker {
            Some(tracker) => Some(tracker.reserve_blocking(est_bytes, cancel)?),
            None => None,
        };

        let data = match self.spec.format {
            PixelFormat::Rgba8UnormSrgb => self.decode_display_chunk(ifd, index, dw, dh, cancel)?,
            PixelFormat::R32Float => self.decode_scientific_chunk(ifd, index, dw, dh, cancel)?,
            PixelFormat::R8Uint | PixelFormat::R16Uint => {
                self.decode_categorical_chunk(ifd, index, dw, dh, cancel)?
            }
        };

        Ok(DecodedChunk {
            data,
            width: dw,
            height: dh,
            format: self.spec.format,
        })
    }

    fn band_index(&self, ifd: &CachedIfd) -> Result<usize, RasterError> {
        let idx = match self.spec.band {
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

    fn decode_display_chunk(
        &self,
        ifd: &CachedIfd,
        index: u32,
        dw: u32,
        dh: u32,
        cancel: &CancelToken,
    ) -> Result<Vec<u8>, RasterError> {
        let pixels = (dw as usize) * (dh as usize);
        let mapping = self
            .display_mapping
            .unwrap_or_else(|| DisplayMapping::identity(self.nodata_f64.map(|v| v as f32)));

        // Palette TIFF: expand ColorMap indices to RGBA8.
        if ifd.photometric == Photometric::Palette {
            let color_map = ifd
                .color_map
                .as_deref()
                .ok_or_else(|| RasterError::Decode("Palette TIFF missing ColorMap".to_owned()))?;
            let (_, _, payload) = decode_raw_chunk(
                self.byte_source.as_ref(),
                ifd,
                self.byte_order,
                0,
                index,
                self.max_chunk_bytes,
                cancel,
            )?;
            let RawChunkPayload::Bytes { data, .. } = payload else {
                return Ok(vec![0u8; pixels * 4]);
            };
            let alpha_plane = if ifd.planar == PlanarConfig::Planar {
                if let Some(a_ch) = ifd.alpha_channel {
                    match decode_raw_chunk(
                        self.byte_source.as_ref(),
                        ifd,
                        self.byte_order,
                        a_ch,
                        index,
                        self.max_chunk_bytes,
                        cancel,
                    )?
                    .2
                    {
                        RawChunkPayload::Bytes { data, .. } => Some(data),
                        RawChunkPayload::Sparse => {
                            // Sparse alpha plane means alpha == 0 across the chunk.
                            return Ok(vec![0u8; pixels * 4]);
                        }
                    }
                } else {
                    None
                }
            } else {
                None
            };
            let palette_alpha = match (ifd.planar, ifd.alpha_channel, alpha_plane.as_deref()) {
                (PlanarConfig::Planar, Some(_), Some(ap)) => PaletteAlpha::Planar(ap),
                (PlanarConfig::Chunky, Some(a_ch), _) => PaletteAlpha::Chunky(a_ch),
                _ => PaletteAlpha::None,
            };
            return Ok(expand_palette_to_rgba(
                &data,
                palette_alpha,
                ifd,
                color_map,
                self.nodata_f64,
                pixels,
            ));
        }

        // PlanarConfiguration = 2 display read: read needed planes and interleave.
        if ifd.planar == PlanarConfig::Planar {
            let need_planes: Vec<usize> = match (ifd.pixel_layout.colors(), ifd.alpha_channel) {
                (1, None) => vec![0],
                (1, Some(a)) => vec![0, a],
                (_, None) => vec![0, 1, 2],
                (_, Some(a)) => vec![0, 1, 2, a],
            };
            let sample_bytes = ifd.sample_type.byte_size();
            let interleaved_spp = need_planes.len();
            let mut interleaved = vec![0u8; pixels * interleaved_spp * sample_bytes];
            let mut any_non_sparse = false;
            let mut jpeg_already_rgb = false;

            for (slot, &plane) in need_planes.iter().enumerate() {
                let (_, _, payload) = decode_raw_chunk(
                    self.byte_source.as_ref(),
                    ifd,
                    self.byte_order,
                    plane,
                    index,
                    self.max_chunk_bytes,
                    cancel,
                )?;
                if let RawChunkPayload::Bytes {
                    data,
                    jpeg_already_rgb: jrgb,
                } = payload
                {
                    any_non_sparse = true;
                    jpeg_already_rgb |= jrgb;
                    for px in 0..pixels {
                        let s = px * sample_bytes;
                        let d = (px * interleaved_spp + slot) * sample_bytes;
                        if s + sample_bytes <= data.len() && d + sample_bytes <= interleaved.len() {
                            interleaved[d..d + sample_bytes]
                                .copy_from_slice(&data[s..s + sample_bytes]);
                        }
                    }
                }
            }
            if !any_non_sparse {
                return Ok(vec![0u8; pixels * 4]);
            }
            if let Some(params) = ifd.ycbcr
                && !jpeg_already_rgb
                && ifd.sample_type == SampleType::U8
            {
                return Ok(params.convert_u8_to_rgba(&interleaved, pixels, mapping.nodata()));
            }
            let packed_layout = match (ifd.pixel_layout.colors(), ifd.alpha_channel.is_some()) {
                (1, false) => crate::pipeline::radiometry::PixelLayout::named(1),
                (1, true) => crate::pipeline::radiometry::PixelLayout::named(2),
                (_, false) => crate::pipeline::radiometry::PixelLayout::named(3),
                (_, true) => crate::pipeline::radiometry::PixelLayout::named(4),
            };
            return Ok(raw_bytes_to_rgba(
                &mapping,
                &interleaved,
                ifd.sample_type,
                packed_layout,
                pixels,
            ));
        }

        // Chunky display read.
        let (_, _, payload) = decode_raw_chunk(
            self.byte_source.as_ref(),
            ifd,
            self.byte_order,
            0,
            index,
            self.max_chunk_bytes,
            cancel,
        )?;
        let RawChunkPayload::Bytes {
            data,
            jpeg_already_rgb,
        } = payload
        else {
            return Ok(vec![0u8; pixels * 4]);
        };

        if let Some(params) = ifd.ycbcr
            && !jpeg_already_rgb
            && ifd.sample_type == SampleType::U8
        {
            return Ok(params.convert_u8_to_rgba(&data, pixels, mapping.nodata()));
        }

        Ok(raw_bytes_to_rgba(
            &mapping,
            &data,
            ifd.sample_type,
            ifd.pixel_layout,
            pixels,
        ))
    }

    fn decode_scientific_chunk(
        &self,
        ifd: &CachedIfd,
        index: u32,
        dw: u32,
        dh: u32,
        cancel: &CancelToken,
    ) -> Result<Vec<u8>, RasterError> {
        let band = self.band_index(ifd)?;
        let pixels = (dw as usize) * (dh as usize);
        let plane = match ifd.planar {
            PlanarConfig::Planar => band,
            PlanarConfig::Chunky => 0,
        };
        let (_, _, payload) = decode_raw_chunk(
            self.byte_source.as_ref(),
            ifd,
            self.byte_order,
            plane,
            index,
            self.max_chunk_bytes,
            cancel,
        )?;
        let RawChunkPayload::Bytes { data, .. } = payload else {
            // Sparse chunk -> all NaN.
            let nan = f32::NAN.to_ne_bytes();
            let mut out = Vec::with_capacity(pixels * 4);
            for _ in 0..pixels {
                out.extend_from_slice(&nan);
            }
            return Ok(out);
        };

        let (stride, ch_offset) = match ifd.planar {
            PlanarConfig::Planar => (1usize, 0usize),
            PlanarConfig::Chunky => (usize::from(ifd.samples_per_pixel), band),
        };
        let mut out = Vec::with_capacity(pixels * 4);
        for px in 0..pixels {
            let sample_idx = px * stride + ch_offset;
            let raw_f64 = read_native_sample_f64(&data, ifd.sample_type, sample_idx);
            let out_val = match raw_f64 {
                Some(v) if is_data_f64(v, self.nodata_f64) => {
                    let f = v as f32;
                    if f.is_finite() { f } else { f32::NAN }
                }
                _ => f32::NAN,
            };
            out.extend_from_slice(&out_val.to_ne_bytes());
        }
        Ok(out)
    }

    fn decode_categorical_chunk(
        &self,
        ifd: &CachedIfd,
        index: u32,
        dw: u32,
        dh: u32,
        cancel: &CancelToken,
    ) -> Result<Vec<u8>, RasterError> {
        let band = self.band_index(ifd)?;
        let pixels = (dw as usize) * (dh as usize);
        let bpp = self.spec.format.bytes_per_pixel();
        let plane = match ifd.planar {
            PlanarConfig::Planar => band,
            PlanarConfig::Chunky => 0,
        };
        let (_, _, payload) = decode_raw_chunk(
            self.byte_source.as_ref(),
            ifd,
            self.byte_order,
            plane,
            index,
            self.max_chunk_bytes,
            cancel,
        )?;
        let RawChunkPayload::Bytes { mut data, .. } = payload else {
            // Sparse categorical chunk -> nodata value (if integer) or 0.
            let fill_u16 = self
                .nodata_f64
                .filter(|v| v.is_finite() && *v >= 0.0 && *v <= f64::from(u16::MAX))
                .map_or(0u16, |v| v.round() as u16);
            return Ok(match self.spec.format {
                PixelFormat::R8Uint => vec![fill_u16.min(255) as u8; pixels],
                PixelFormat::R16Uint => {
                    let b = fill_u16.to_ne_bytes();
                    let mut out = Vec::with_capacity(pixels * 2);
                    for _ in 0..pixels {
                        out.extend_from_slice(&b);
                    }
                    out
                }
                _ => vec![0u8; pixels * bpp],
            });
        };

        let (stride, ch_offset) = match ifd.planar {
            PlanarConfig::Planar => (1usize, 0usize),
            PlanarConfig::Chunky => (usize::from(ifd.samples_per_pixel), band),
        };
        match (self.spec.format, ifd.sample_type) {
            (PixelFormat::R8Uint, SampleType::U8) if stride == 1 => {
                data.truncate(pixels);
                Ok(data)
            }
            (PixelFormat::R8Uint, SampleType::U8) => {
                let mut out = Vec::with_capacity(pixels);
                for px in 0..pixels {
                    out.push(data[px * stride + ch_offset]);
                }
                Ok(out)
            }
            (PixelFormat::R16Uint, SampleType::U16) if stride == 1 => {
                data.truncate(pixels * 2);
                Ok(data)
            }
            (PixelFormat::R16Uint, SampleType::U16) => {
                let mut out = Vec::with_capacity(pixels * 2);
                for px in 0..pixels {
                    let off = (px * stride + ch_offset) * 2;
                    out.extend_from_slice(&data[off..off + 2]);
                }
                Ok(out)
            }
            (PixelFormat::R16Uint, SampleType::U8) => {
                let mut out = Vec::with_capacity(pixels * 2);
                for px in 0..pixels {
                    let val = u16::from(data[px * stride + ch_offset]);
                    out.extend_from_slice(&val.to_ne_bytes());
                }
                Ok(out)
            }
            (fmt, st) => Err(RasterError::Decode(format!(
                "categorical format {fmt:?} is incompatible with source sample type {st:?}"
            ))),
        }
    }
}

enum PaletteAlpha<'a> {
    None,
    Chunky(usize),
    Planar(&'a [u8]),
}

fn expand_palette_to_rgba(
    data: &[u8],
    alpha_mode: PaletteAlpha<'_>,
    ifd: &CachedIfd,
    color_map: &[[u8; 4]],
    nodata_f64: Option<f64>,
    pixels: usize,
) -> Vec<u8> {
    let mut rgba = vec![0u8; pixels * 4];
    let chunky_spp = match ifd.planar {
        PlanarConfig::Chunky => usize::from(ifd.samples_per_pixel),
        PlanarConfig::Planar => 1,
    };
    for (px, dst) in rgba.chunks_exact_mut(4).enumerate() {
        let Some(raw_val) = read_native_sample_f64(data, ifd.sample_type, px * chunky_spp) else {
            continue;
        };
        if !is_data_f64(raw_val, nodata_f64) {
            continue;
        }
        let idx = raw_val.round().clamp(0.0, (color_map.len() - 1) as f64) as usize;
        let mut color = color_map[idx];
        let alpha_val = match alpha_mode {
            PaletteAlpha::None => None,
            PaletteAlpha::Planar(ap) => read_native_sample_f64(ap, ifd.sample_type, px),
            PaletteAlpha::Chunky(a_ch) => {
                read_native_sample_f64(data, ifd.sample_type, px * chunky_spp + a_ch)
            }
        };
        if let Some(a) = alpha_val {
            color[3] = match ifd.sample_type {
                SampleType::U16 => ((a.clamp(0.0, 65535.0) as u16) >> 8) as u8,
                _ => a.round().clamp(0.0, 255.0) as u8,
            };
        }
        dst.copy_from_slice(&color);
    }
    rgba
}

impl Chunked for TiffChunkReader<'_> {
    fn chunk(
        &mut self,
        image: usize,
        index: u32,
        cancel: &CancelToken,
    ) -> Result<Arc<DecodedChunk>, RasterError> {
        match self.mode {
            ChunkReadMode::SequentialPass => self
                .decode_chunk_uncached(image, index, None, cancel)
                .map(Arc::new),
            ChunkReadMode::Pointwise {
                chunk_cache,
                budget_tracker,
            } => {
                let key = ChunkKey {
                    source: self.source_id,
                    spec: self.spec,
                    image,
                    chunk_index: index,
                };
                chunk_cache.get_or_decode(key, cancel, || {
                    self.decode_chunk_uncached(image, index, Some(budget_tracker), cancel)
                })
            }
        }
    }
}
