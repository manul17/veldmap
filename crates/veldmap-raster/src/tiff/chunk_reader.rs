//! Stateless direct raw-chunk TIFF reader bridging [`TiffSourceView`] to
//! [`Chunked`] and [`SharedChunkCache`] across `Display`, `Scientific`
//! (`R32Float`), and `Categorical` (`R8Uint`/`R16Uint`) streams.
//!
//! Pure pixel-format conversion lives in [`super::chunk_convert`].

use std::sync::Arc;

use veldmap_core::budget::DecodeBudgetTracker;
use veldmap_core::sync::CancelToken;
use veldmap_core::tile::{PixelFormat, TileStreamSpec};

use crate::RasterError;
use crate::pipeline::{
    ChunkKey, Chunked, DecodedChunk, DisplayMapping, SharedChunkCache, alloc_empty_surface,
    chunk_working_bytes,
};

use super::chunk_convert::{
    CategoricalFormat, convert_categorical_chunk, convert_chunky_display,
    convert_interleaved_planar_display, convert_scientific_chunk, expand_palette_to_rgba,
    palette_alpha_mode, scatter_plane_into_interleaved, sparse_categorical_chunk,
};
use super::ifd::{CachedIfd, Photometric, PlanarConfig, TiffSourceView, find_ifd};
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
    view: TiffSourceView<'a>,
    spec: TileStreamSpec,
    display_mapping: Option<DisplayMapping>,
    mode: ChunkReadMode<'a>,
}

impl<'a> TiffChunkReader<'a> {
    pub(super) fn new(
        view: TiffSourceView<'a>,
        spec: TileStreamSpec,
        display_mapping: Option<DisplayMapping>,
        mode: ChunkReadMode<'a>,
    ) -> Self {
        Self {
            view,
            spec,
            display_mapping,
            mode,
        }
    }

    fn decode_plane(
        &self,
        ifd: &CachedIfd,
        plane: usize,
        index: u32,
        cancel: &CancelToken,
    ) -> Result<RawChunkPayload, RasterError> {
        Ok(decode_raw_chunk(self.view.access, ifd, plane, index, cancel)?.2)
    }

    fn decode_chunk_uncached(
        &self,
        image: usize,
        index: u32,
        budget_tracker: Option<&DecodeBudgetTracker>,
        cancel: &CancelToken,
    ) -> Result<DecodedChunk, RasterError> {
        let ifd = find_ifd(self.view.ifds, image)?;
        let dims = ifd.chunk_dims(index).ok_or_else(|| {
            RasterError::Decode(format!(
                "chunk index {index} out of range for IFD {image} ({} chunks)",
                ifd.chunks_per_plane
            ))
        })?;
        let out_bpp = self.spec.format.bytes_per_pixel() as u32;
        let raw_depth = if self.spec.format == PixelFormat::Rgba8UnormSrgb {
            ifd.raw_pixel_bytes()
        } else {
            ifd.band_raw_pixel_bytes()
        };
        let est_bytes = chunk_working_bytes(dims.stored_w, dims.stored_h, raw_depth, out_bpp);
        let _lease = match budget_tracker {
            Some(tracker) => Some(tracker.reserve_blocking(est_bytes, cancel)?),
            None => None,
        };

        let pixels = (dims.data_w as usize) * (dims.data_h as usize);
        let data = match self.spec.format {
            PixelFormat::Rgba8UnormSrgb => self.decode_display_chunk(ifd, index, pixels, cancel)?,
            PixelFormat::R32Float => self.decode_scientific_chunk(ifd, index, pixels, cancel)?,
            PixelFormat::R8Uint => {
                self.decode_categorical_chunk(ifd, index, pixels, CategoricalFormat::U8, cancel)?
            }
            PixelFormat::R16Uint => {
                self.decode_categorical_chunk(ifd, index, pixels, CategoricalFormat::U16, cancel)?
            }
        };

        Ok(DecodedChunk {
            data,
            width: dims.data_w,
            height: dims.data_h,
            format: self.spec.format,
        })
    }

    fn decode_display_chunk(
        &self,
        ifd: &CachedIfd,
        index: u32,
        pixels: usize,
        cancel: &CancelToken,
    ) -> Result<Vec<u8>, RasterError> {
        let mapping = self.display_mapping.unwrap_or_else(|| {
            DisplayMapping::identity(self.view.nodata_f64)
                .with_invert(ifd.photometric == Photometric::WhiteIsZero)
        });

        if ifd.photometric == Photometric::Palette {
            return self.decode_palette_display_chunk(ifd, index, pixels, cancel);
        }
        if ifd.planar == PlanarConfig::Planar {
            let (plane_buf, plane_len) = ifd.planar_display_planes();
            let need_planes = &plane_buf[..plane_len];
            let sample_bytes = ifd.sample_type.byte_size();
            let interleaved_spp = need_planes.len();
            let mut interleaved: Option<Vec<u8>> = None;
            for (slot, &plane) in need_planes.iter().enumerate() {
                let RawChunkPayload::Bytes(data) = self.decode_plane(ifd, plane, index, cancel)?
                else {
                    continue;
                };
                let dst = interleaved
                    .get_or_insert_with(|| vec![0u8; pixels * interleaved_spp * sample_bytes]);
                scatter_plane_into_interleaved(
                    dst,
                    &data,
                    slot,
                    interleaved_spp,
                    sample_bytes,
                    pixels,
                );
            }
            let Some(interleaved) = interleaved else {
                return Ok(alloc_empty_surface(pixels, PixelFormat::Rgba8UnormSrgb));
            };
            return Ok(convert_interleaved_planar_display(
                &interleaved,
                ifd,
                pixels,
                mapping,
            ));
        }

        let RawChunkPayload::Bytes(data) = self.decode_plane(ifd, 0, index, cancel)? else {
            return Ok(alloc_empty_surface(pixels, PixelFormat::Rgba8UnormSrgb));
        };
        Ok(convert_chunky_display(&data, ifd, pixels, mapping))
    }

    fn decode_palette_display_chunk(
        &self,
        ifd: &CachedIfd,
        index: u32,
        pixels: usize,
        cancel: &CancelToken,
    ) -> Result<Vec<u8>, RasterError> {
        let color_map = ifd
            .color_map
            .as_deref()
            .ok_or_else(|| RasterError::Decode("Palette TIFF missing ColorMap".to_owned()))?;
        let RawChunkPayload::Bytes(data) = self.decode_plane(ifd, 0, index, cancel)? else {
            return Ok(alloc_empty_surface(pixels, PixelFormat::Rgba8UnormSrgb));
        };
        let alpha_plane = if ifd.planar == PlanarConfig::Planar {
            if let Some(a_ch) = ifd.alpha_channel() {
                match self.decode_plane(ifd, a_ch, index, cancel)? {
                    RawChunkPayload::Bytes(data) => Some(data),
                    RawChunkPayload::Sparse => {
                        // Sparse alpha plane means alpha == 0 across the chunk.
                        return Ok(alloc_empty_surface(pixels, PixelFormat::Rgba8UnormSrgb));
                    }
                }
            } else {
                None
            }
        } else {
            None
        };
        let alpha_mode = palette_alpha_mode(ifd, alpha_plane.as_deref());
        Ok(expand_palette_to_rgba(
            &data,
            alpha_mode,
            ifd,
            color_map,
            self.view.nodata_for_band(0),
            pixels,
        ))
    }

    fn decode_scientific_chunk(
        &self,
        ifd: &CachedIfd,
        index: u32,
        pixels: usize,
        cancel: &CancelToken,
    ) -> Result<Vec<u8>, RasterError> {
        let band = ifd.resolve_band_index(self.spec.band)?;
        let RawChunkPayload::Bytes(data) =
            self.decode_plane(ifd, ifd.band_plane(band), index, cancel)?
        else {
            return Ok(alloc_empty_surface(pixels, PixelFormat::R32Float));
        };
        Ok(convert_scientific_chunk(
            &data,
            ifd,
            band,
            pixels,
            self.view.nodata_for_band(band),
        ))
    }

    fn decode_categorical_chunk(
        &self,
        ifd: &CachedIfd,
        index: u32,
        pixels: usize,
        format: CategoricalFormat,
        cancel: &CancelToken,
    ) -> Result<Vec<u8>, RasterError> {
        let band = ifd.resolve_band_index(self.spec.band)?;
        let RawChunkPayload::Bytes(data) =
            self.decode_plane(ifd, ifd.band_plane(band), index, cancel)?
        else {
            return Ok(sparse_categorical_chunk(
                format,
                pixels,
                self.view.nodata_for_band(band),
            ));
        };
        convert_categorical_chunk(data, ifd, band, pixels, format)
    }
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
                let key = ChunkKey::from_spec(self.view.source_id, self.spec, image, index);
                chunk_cache.get_or_decode(key, cancel, || {
                    self.decode_chunk_uncached(image, index, Some(budget_tracker), cancel)
                })
            }
        }
    }
}
