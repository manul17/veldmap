//! Cached per-IFD TIFF metadata (`TileOffsets`/`StripOffsets`,
//! `TileByteCounts`/`StripByteCounts`, `JPEGTables`, `ColorMap`, layout, and
//! compression/predictor descriptors) bounded by [`MemoryBudget`].
//!
//! Parsing and caching IFD tables once during `TiffSource::open` allows the
//! direct raw-chunk decoder (`raw_decode.rs`) to execute lock-free positioned
//! reads over `ByteSource` without stateful `tiff::decoder::Decoder` pooling,
//! while also resolving `tiff 0.11.3`'s `RGBPalette`, 4-band RGB+NIR, and
//! tiled `PlanarConfiguration = 2` blockers.

use std::sync::Arc;

use tiff::decoder::Decoder;
use tiff::tags::Tag;
use veldmap_core::budget::MemoryBudget;
use veldmap_core::io::BufferedByteReader;
use veldmap_core::sync::CancelToken;

use crate::RasterError;
use crate::pipeline::grid;
use crate::pipeline::metadata::SampleType;
use crate::pipeline::peak::DecodePeak;
use crate::pipeline::radiometry::PixelLayout;

use super::tags::{
    map_tiff_error, opt_tag, opt_tag_strict, read_tag_u8_vec_opt, read_tag_u16_vec_opt,
    read_tag_u64_vec_opt,
};
use super::ycbcr::{YCbCrParams, read_ycbcr_params};

/// Maximum uncompressed byte size admitted for any single TIFF strip/tile,
/// matching [`crate::tiff::TiffSource`]'s per-chunk decode cap.
pub(super) const MAX_SINGLE_CHUNK_BYTES: u64 = 256 * 1024 * 1024;

/// TIFF file byte order from header bytes `0..2`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TiffByteOrder {
    Little,
    Big,
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
    fn from_code(code: u16) -> Self {
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
    pub(super) sample_bits: u8,
    pub(super) sample_type: SampleType,
    pub(super) pixel_layout: PixelLayout,
    pub(super) alpha_channel: Option<usize>,
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

    /// Valid (cropped) pixel dimensions `(dw, dh)` for chunk `chunk_index` in a plane.
    pub(super) fn chunk_data_dims(&self, chunk_index: u32) -> Option<(u32, u32)> {
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
        let dw = self.chunk_w.min(self.width - x0);
        let dh = self.chunk_h.min(self.height - y0);
        (dw > 0 && dh > 0).then_some((dw, dh))
    }

    /// Stored (padded for tiled, exact for stripped) dimensions `(sw, sh)` on disk.
    pub(super) fn chunk_stored_dims(&self, chunk_index: u32) -> Option<(u32, u32)> {
        let (dw, dh) = self.chunk_data_dims(chunk_index)?;
        if self.tiled {
            Some((self.chunk_w, self.chunk_h))
        } else {
            Some((dw, dh))
        }
    }

    /// Raw bytes per pixel across all channels (or 1 plane when planar single-band).
    pub(super) const fn raw_pixel_bytes(&self) -> u32 {
        (self.samples_per_pixel as u32) * (self.sample_type.byte_size() as u32)
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

fn resolve_sample_type(sample_format: u16, bits: u8) -> Result<SampleType, RasterError> {
    match (sample_format, bits) {
        (1, 8) => Ok(SampleType::U8),
        (2, 8) => Ok(SampleType::I8),
        (1, 16) => Ok(SampleType::U16),
        (2, 16) => Ok(SampleType::I16),
        (1, 32) => Ok(SampleType::U32),
        (2, 32) => Ok(SampleType::I32),
        (3, 16) => Ok(SampleType::F16),
        (3, 32) => Ok(SampleType::F32),
        (3, 64) => Ok(SampleType::F64),
        _ => Err(RasterError::Decode(format!(
            "unsupported TIFF sample format ({sample_format}, {bits}-bit)"
        ))),
    }
}

fn parse_color_map(
    decoder: &mut Decoder<BufferedByteReader>,
    sample_bits: u8,
    cancel: &CancelToken,
) -> Result<Arc<[[u8; 4]]>, RasterError> {
    let raw = read_tag_u16_vec_opt(decoder, Tag::ColorMap, cancel)?;
    let expected_entries = match sample_bits {
        8 => 256usize,
        16 => 65_536usize,
        other => {
            return Err(RasterError::Decode(format!(
                "unsupported Palette TIFF BitsPerSample {other} (expected 8 or 16)"
            )));
        }
    };
    if raw.len() < expected_entries * 3 || !raw.len().is_multiple_of(3) {
        return Err(RasterError::Decode(format!(
            "invalid TIFF ColorMap length {} for {sample_bits}-bit Palette",
            raw.len()
        )));
    }
    let n = raw.len() / 3;
    // Some legacy writers store 8-bit values [0..=255] directly in u16 ColorMap slots.
    let is_8bit_scaled = raw.iter().all(|&v| v <= 255);
    let to_u8 = |v: u16| -> u8 {
        if is_8bit_scaled {
            v as u8
        } else {
            (v >> 8) as u8
        }
    };
    let mut lut = Vec::with_capacity(n);
    for i in 0..n {
        lut.push([to_u8(raw[i]), to_u8(raw[n + i]), to_u8(raw[2 * n + i]), 255]);
    }
    Ok(Arc::from(lut))
}

fn resolve_pixel_layout(
    photometric: Photometric,
    samples_per_pixel: u16,
    extra_samples: &[u16],
) -> Result<(PixelLayout, Option<usize>, u16), RasterError> {
    let spp = usize::from(samples_per_pixel);
    match photometric {
        Photometric::WhiteIsZero => {
            if spp > 1 {
                return Err(RasterError::Decode(
                    "WhiteIsZero with multiple samples is not supported".to_owned(),
                ));
            }
            Ok((PixelLayout::named(1), None, 1))
        }
        Photometric::BlackIsZero => {
            if spp == 1 {
                Ok((PixelLayout::named(1), None, 1))
            } else if spp == 2 && extra_samples.first().is_some_and(|&v| v == 1 || v == 2) {
                Ok((PixelLayout::named(2), Some(1), 1))
            } else {
                // Multi-band stack (e.g., Sentinel-2 / hyperspectral): check if
                // an explicit alpha ExtraSamples entry exists.
                let base_count = spp.saturating_sub(extra_samples.len()).max(1);
                let alpha_ch = extra_samples
                    .iter()
                    .position(|&v| v == 1 || v == 2)
                    .map(|idx| base_count + idx)
                    .filter(|&ch| ch < spp);
                let data_bands = if alpha_ch.is_some() {
                    samples_per_pixel.saturating_sub(1).max(1)
                } else {
                    samples_per_pixel
                };
                Ok((PixelLayout::stack(spp), alpha_ch, data_bands))
            }
        }
        Photometric::Rgb => {
            if spp < 3 {
                return Err(RasterError::Decode(format!(
                    "RGB TIFF requires at least 3 samples/pixel, got {spp}"
                )));
            }
            if spp == 3 {
                return Ok((PixelLayout::named(3), None, 3));
            }
            // 4+ band RGB (e.g., RGBA vs 4-band RGB+NIR with ExtraSamples = 0):
            // Only ExtraSamples == 1 (Associated) or 2 (Unassociated) is alpha!
            // ExtraSamples == 0 (Unspecified) or missing ExtraSamples keeps all
            // bands as opaque data bands so NIR is never mistaken for alpha.
            let alpha_ch = extra_samples
                .iter()
                .position(|&v| v == 1 || v == 2)
                .map(|idx| 3 + idx)
                .filter(|&ch| ch < spp);
            let layout = match alpha_ch {
                Some(ch) if ch == spp - 1 => PixelLayout::rgb_with_alpha(spp),
                _ => PixelLayout::rgb_opaque(spp),
            };
            let data_bands = if alpha_ch.is_some() {
                samples_per_pixel.saturating_sub(1).max(3)
            } else {
                samples_per_pixel
            };
            Ok((layout, alpha_ch, data_bands))
        }
        Photometric::Palette => {
            if spp == 1 {
                Ok((PixelLayout::named(1), None, 1))
            } else if spp == 2 && extra_samples.first().is_some_and(|&v| v == 1 || v == 2) {
                Ok((PixelLayout::named(2), Some(1), 1))
            } else {
                Err(RasterError::Decode(format!(
                    "unsupported Palette TIFF with {spp} samples/pixel"
                )))
            }
        }
        Photometric::YCbCr => {
            if spp != 3 {
                return Err(RasterError::Decode(format!(
                    "YCbCr TIFF requires 3 samples/pixel, got {spp}"
                )));
            }
            Ok((PixelLayout::named(3), None, 3))
        }
        Photometric::Other(code) => Err(RasterError::Decode(format!(
            "unsupported TIFF PhotometricInterpretation {code}"
        ))),
    }
}

/// Maximum total cached IFD table bytes allowed across a single TIFF file.
pub(super) fn max_cached_ifd_table_bytes(budget: &MemoryBudget) -> u64 {
    (budget.decode_working_bytes / 16).max(4 * 1024 * 1024)
}

/// Parse and validate all chunk/tag metadata for the currently aimed IFD.
pub(super) fn parse_cached_ifd(
    decoder: &mut Decoder<BufferedByteReader>,
    image: usize,
    budget: &MemoryBudget,
    table_bytes_used: u64,
    cancel: &CancelToken,
) -> Result<CachedIfd, RasterError> {
    let (width, height) = decoder
        .dimensions()
        .map_err(|e| map_tiff_error(e, cancel))?;
    if width == 0 || height == 0 {
        return Err(RasterError::InvalidDimensions { width, height });
    }

    let compression_code =
        opt_tag_strict(decoder.get_tag_unsigned::<u16>(Tag::Compression), cancel)?.unwrap_or(1);
    let compression = TiffCompression::from_code(compression_code)?;

    let photometric_code = opt_tag_strict(
        decoder.get_tag_unsigned::<u16>(Tag::PhotometricInterpretation),
        cancel,
    )?
    .unwrap_or(1);
    let photometric = Photometric::from_code(photometric_code);

    let samples_per_pixel = opt_tag_strict(
        decoder.get_tag_unsigned::<u16>(Tag::SamplesPerPixel),
        cancel,
    )?
    .unwrap_or(1);
    if samples_per_pixel == 0 {
        return Err(RasterError::Decode("TIFF SamplesPerPixel is 0".to_owned()));
    }

    let bps_vec = read_tag_u16_vec_opt(decoder, Tag::BitsPerSample, cancel)?;
    let sample_bits_u16 = bps_vec.first().copied().unwrap_or(8);
    if bps_vec.iter().any(|&b| b != sample_bits_u16) {
        return Err(RasterError::Decode(format!(
            "heterogeneous per-band BitsPerSample {bps_vec:?} is not supported"
        )));
    }
    let sample_bits = u8::try_from(sample_bits_u16)
        .map_err(|_| RasterError::Decode(format!("unsupported BitsPerSample {sample_bits_u16}")))?;

    let sample_fmt_vec = read_tag_u16_vec_opt(decoder, Tag::SampleFormat, cancel)?;
    let sample_fmt_code = sample_fmt_vec.first().copied().unwrap_or(1);
    if sample_fmt_vec.iter().any(|&f| f != sample_fmt_code) {
        return Err(RasterError::Decode(format!(
            "heterogeneous per-band SampleFormat {sample_fmt_vec:?} is not supported"
        )));
    }
    let sample_type = resolve_sample_type(sample_fmt_code, sample_bits)?;

    if photometric == Photometric::YCbCr && sample_type != SampleType::U8 {
        return Err(RasterError::Decode(format!(
            "YCbCr TIFF requires 8-bit unsigned samples, got {sample_type:?}"
        )));
    }
    if photometric == Photometric::Palette
        && !matches!(sample_type, SampleType::U8 | SampleType::U16)
    {
        return Err(RasterError::Decode(format!(
            "Palette TIFF requires 8- or 16-bit unsigned indices, got {sample_type:?}"
        )));
    }

    let planar_code = opt_tag_strict(
        decoder.get_tag_unsigned::<u16>(Tag::PlanarConfiguration),
        cancel,
    )?
    .unwrap_or(1);
    let planar = match planar_code {
        1 => PlanarConfig::Chunky,
        2 => PlanarConfig::Planar,
        other => {
            return Err(RasterError::Decode(format!(
                "unsupported TIFF PlanarConfiguration {other}"
            )));
        }
    };
    if planar == PlanarConfig::Planar
        && matches!(compression, TiffCompression::WebP)
        && samples_per_pixel > 1
    {
        return Err(RasterError::Decode(
            "planar WebP TIFF is not supported".to_owned(),
        ));
    }

    let predictor_code =
        opt_tag_strict(decoder.get_tag_unsigned::<u16>(Tag::Predictor), cancel)?.unwrap_or(1);
    let predictor = TiffPredictor::from_code(predictor_code, sample_type)?;

    let extra_samples = read_tag_u16_vec_opt(decoder, Tag::ExtraSamples, cancel)?;
    let (pixel_layout, alpha_channel, data_bands) =
        resolve_pixel_layout(photometric, samples_per_pixel, &extra_samples)?;

    let ycbcr = if photometric == Photometric::YCbCr {
        if compression != TiffCompression::Jpeg {
            let sub = read_tag_u16_vec_opt(decoder, Tag::ChromaSubsampling, cancel)?;
            if !sub.is_empty() && sub.as_slice() != [1, 1] {
                return Err(RasterError::Decode(format!(
                    "non-JPEG subsampled YCbCr TIFF (YCbCrSubSampling={sub:?}) is not supported"
                )));
            }
        }
        Some(read_ycbcr_params(decoder, cancel)?)
    } else {
        None
    };

    let color_map = if photometric == Photometric::Palette {
        Some(parse_color_map(decoder, sample_bits, cancel)?)
    } else {
        None
    };

    // Determine tiled vs stripped layout and read offset/byte-count tables.
    let tile_w = opt_tag(decoder.get_tag_unsigned::<u32>(Tag::TileWidth), cancel)?;
    let (tiled, chunk_w, chunk_h, offsets, byte_counts) = if let Some(tw) = tile_w {
        let th = opt_tag_strict(decoder.get_tag_unsigned::<u32>(Tag::TileLength), cancel)?
            .ok_or_else(|| RasterError::Decode("tiled TIFF missing TileLength".to_owned()))?;
        let offsets = read_tag_u64_vec_opt(decoder, Tag::TileOffsets, cancel)?
            .ok_or_else(|| RasterError::Decode("tiled TIFF missing TileOffsets".to_owned()))?;
        let byte_counts = read_tag_u64_vec_opt(decoder, Tag::TileByteCounts, cancel)?
            .ok_or_else(|| RasterError::Decode("tiled TIFF missing TileByteCounts".to_owned()))?;
        (true, tw, th, offsets, byte_counts)
    } else {
        let rows_per_strip = opt_tag(decoder.get_tag_unsigned::<u32>(Tag::RowsPerStrip), cancel)?
            .unwrap_or(height)
            .clamp(1, height);
        let offsets = read_tag_u64_vec_opt(decoder, Tag::StripOffsets, cancel)?
            .ok_or_else(|| RasterError::Decode("stripped TIFF missing StripOffsets".to_owned()))?;
        let byte_counts =
            read_tag_u64_vec_opt(decoder, Tag::StripByteCounts, cancel)?.ok_or_else(|| {
                RasterError::Decode("stripped TIFF missing StripByteCounts".to_owned())
            })?;
        (false, width, rows_per_strip, offsets, byte_counts)
    };

    if chunk_w == 0 || chunk_h == 0 {
        return Err(RasterError::InvalidDimensions {
            width: chunk_w,
            height: chunk_h,
        });
    }

    let chunks_across = width.div_ceil(chunk_w);
    let chunks_down = height.div_ceil(chunk_h);
    let chunks_per_plane = chunks_across
        .checked_mul(chunks_down)
        .ok_or_else(|| RasterError::Decode("TIFF chunk count overflow".to_owned()))?;
    let expected_chunks = match planar {
        PlanarConfig::Chunky => usize::try_from(chunks_per_plane)
            .map_err(|_| RasterError::Decode("TIFF chunk count overflow".to_owned()))?,
        PlanarConfig::Planar => usize::try_from(chunks_per_plane)
            .ok()
            .and_then(|n| n.checked_mul(usize::from(samples_per_pixel)))
            .ok_or_else(|| RasterError::Decode("TIFF planar chunk count overflow".to_owned()))?,
    };
    if offsets.len() < expected_chunks || byte_counts.len() < expected_chunks {
        return Err(RasterError::Decode(format!(
            "TIFF chunk table too short for IFD {image}: expected {expected_chunks} entries, got {} offsets / {} byte counts",
            offsets.len(),
            byte_counts.len()
        )));
    }

    let jpeg_tables = if compression == TiffCompression::Jpeg {
        match read_tag_u8_vec_opt(decoder, Tag::JPEGTables, cancel)? {
            Some(tables) => {
                if tables.len() < 4 {
                    return Err(RasterError::Decode(
                        "TIFF JPEGTables tag is shorter than 4 bytes".to_owned(),
                    ));
                }
                Some(Arc::from(tables))
            }
            None => None,
        }
    } else {
        None
    };

    let depth = (samples_per_pixel as u32) * (sample_type.byte_size() as u32);
    let chunk_cap = budget.decode_working_bytes.min(MAX_SINGLE_CHUNK_BYTES);
    DecodePeak::new()
        .with(
            "fresh chunk",
            grid::chunk_working_bytes(chunk_w, chunk_h, depth, 4),
        )
        .admit(chunk_cap)?;

    let cached = CachedIfd {
        image,
        width,
        height,
        tiled,
        chunk_w,
        chunk_h,
        chunks_across,
        chunks_per_plane,
        compression,
        predictor,
        planar,
        photometric,
        samples_per_pixel,
        sample_bits,
        sample_type,
        pixel_layout,
        alpha_channel,
        data_bands,
        offsets,
        byte_counts,
        jpeg_tables,
        ycbcr,
        color_map,
    };

    let table_cap = max_cached_ifd_table_bytes(budget);
    let new_total = table_bytes_used.saturating_add(cached.table_bytes());
    if new_total > table_cap {
        return Err(RasterError::LimitExceeded {
            width,
            height,
            estimated_bytes: new_total,
            limit_bytes: table_cap,
        });
    }

    Ok(cached)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_type_resolution_covers_all_scientific_and_display_types() {
        assert_eq!(resolve_sample_type(1, 8).unwrap(), SampleType::U8);
        assert_eq!(resolve_sample_type(2, 8).unwrap(), SampleType::I8);
        assert_eq!(resolve_sample_type(1, 16).unwrap(), SampleType::U16);
        assert_eq!(resolve_sample_type(2, 16).unwrap(), SampleType::I16);
        assert_eq!(resolve_sample_type(1, 32).unwrap(), SampleType::U32);
        assert_eq!(resolve_sample_type(2, 32).unwrap(), SampleType::I32);
        assert_eq!(resolve_sample_type(3, 16).unwrap(), SampleType::F16);
        assert_eq!(resolve_sample_type(3, 32).unwrap(), SampleType::F32);
        assert_eq!(resolve_sample_type(3, 64).unwrap(), SampleType::F64);
        assert!(resolve_sample_type(1, 12).is_err());
    }

    #[test]
    fn rgb_extra_samples_zero_preserves_4band_rgbn_without_alpha() {
        // 4-band RGB+NIR with ExtraSamples = [0] (Unspecified): NOT alpha!
        let (layout, alpha, data_bands) =
            resolve_pixel_layout(Photometric::Rgb, 4, &[0]).expect("rgbn");
        assert!(!layout.has_alpha());
        assert_eq!(layout.colors(), 3);
        assert_eq!(layout.channels, 4);
        assert_eq!(alpha, None);
        assert_eq!(data_bands, 4);

        // 4-band RGBA with ExtraSamples = [2] (Unassociated Alpha): IS alpha!
        let (layout_a, alpha_a, data_bands_a) =
            resolve_pixel_layout(Photometric::Rgb, 4, &[2]).expect("rgba");
        assert!(layout_a.has_alpha());
        assert_eq!(alpha_a, Some(3));
        assert_eq!(data_bands_a, 3);
    }
}
