//! Header/IFD discovery, decoder limits, and overview scanning for
//! [`super::TiffSource::open`].

use std::sync::Arc;

use tiff::decoder::{Decoder, Limits};
use tiff::tags::Tag;
use veldmap_core::budget::MemoryBudget;
use veldmap_core::io::{BufferedByteReader, ByteSource};
use veldmap_core::sync::CancelToken;
use veldmap_core::tile::{SourceFingerprint, SourceId};

use crate::RasterError;
use crate::pipeline::grid::{Grid, Overview};
use crate::pipeline::metadata::{BandInfo, RasterDescription};
use crate::pipeline::plan::build_chunked_level_table;
use crate::pipeline::radiometry::DisplayMapping;

use super::chunk_reader::{DecoderPool, open_decoder};
use super::georef::{ParsedGeoref, georef};
use super::stretch::StretchCache;
use super::tags::{
    chunk_grid, depth_of_layout, ensure_readable, is_overview, map_tiff_error, opt_tag,
    parse_gdal_nodata,
};
use super::{MAX_IFDS, MAX_SOURCE_SIDE, TiffFormat, TiffSource};

fn build_tiff_limits(budget: &MemoryBudget) -> Limits {
    let mut limits = Limits::default();
    limits.decoding_buffer_size = budget.decode_working_bytes.min(256 * 1024 * 1024) as usize;
    limits.intermediate_buffer_size = budget
        .decode_working_bytes
        .clamp(4 * 1024 * 1024, 128 * 1024 * 1024) as usize;
    limits
}

fn scan_overviews(
    decoder: &mut Decoder<BufferedByteReader>,
    format: TiffFormat,
    budget: &MemoryBudget,
    cancel: &CancelToken,
) -> Result<(Vec<Overview>, usize, bool), RasterError> {
    let mut overviews = Vec::new();
    let mut ifd_index = 0usize;
    let mut decoder_clean = true;
    while ifd_index + 1 < MAX_IFDS && decoder.more_images() {
        if cancel.is_cancelled() {
            return Err(RasterError::Cancelled);
        }
        if let Err(err) = decoder.next_image() {
            if cancel.is_cancelled() {
                return Err(RasterError::Cancelled);
            }
            tracing::debug!(error = %err, "Stopped scanning TIFF overviews");
            decoder_clean = false;
            break;
        }
        ifd_index += 1;
        let subfile = match opt_tag(decoder.get_tag_unsigned::<u32>(Tag::NewSubfileType), cancel) {
            Ok(v) => v.unwrap_or(0),
            Err(RasterError::Cancelled) => return Err(RasterError::Cancelled),
            Err(_) => continue,
        };
        if !is_overview(subfile) {
            continue;
        }
        match ensure_readable(decoder, cancel) {
            Ok((o_layout, o_bits, o_type, o_ycbcr)) => {
                if o_layout != format.pixel_layout
                    || o_bits != format.sample_bits
                    || o_type != format.sample_type
                    || o_ycbcr != format.ycbcr
                {
                    continue;
                }
            }
            Err(RasterError::Cancelled) => return Err(RasterError::Cancelled),
            Err(_) => continue,
        }
        let Ok((ow, oh)) = decoder.dimensions() else {
            if cancel.is_cancelled() {
                return Err(RasterError::Cancelled);
            }
            continue;
        };
        if ow == 0 || oh == 0 {
            continue;
        }
        // `chunk_grid` checks `ensure_chunky` once per overview.
        let ochunk = match chunk_grid(decoder, budget, format.depth, cancel) {
            Ok(c) => c,
            Err(RasterError::Cancelled) => return Err(RasterError::Cancelled),
            Err(_) => continue,
        };
        overviews.push(Overview {
            image: ifd_index,
            width: ow,
            height: oh,
            chunk: ochunk,
        });
    }
    Ok((overviews, ifd_index, decoder_clean))
}

impl TiffSource {
    /// Open and describe a GeoTIFF/BigTIFF/COG from `byte_source` using
    /// `fingerprint`, `budget`, and `cancel` without reading base-level pixels.
    pub fn open(
        byte_source: Arc<dyn ByteSource>,
        fingerprint: SourceFingerprint,
        budget: &MemoryBudget,
        cancel: &CancelToken,
    ) -> Result<Self, RasterError> {
        Self::open_with_placement(byte_source, fingerprint, None, budget, cancel)
    }

    /// Open a GeoTIFF/BigTIFF/COG, falling back to an injected external
    /// `sidecar_placement` (e.g. from `.tfw` + `.prj` sidecars) when the TIFF
    /// itself carries no embedded georeferencing.
    pub fn open_with_placement(
        byte_source: Arc<dyn ByteSource>,
        fingerprint: SourceFingerprint,
        sidecar_placement: Option<crate::pipeline::metadata::GeoPlacement>,
        budget: &MemoryBudget,
        cancel: &CancelToken,
    ) -> Result<Self, RasterError> {
        if cancel.is_cancelled() {
            return Err(RasterError::Cancelled);
        }
        let limits = build_tiff_limits(budget);
        let mut decoder = open_decoder(&byte_source, &limits, cancel)?;

        let (width, height) = decoder
            .dimensions()
            .map_err(|e| map_tiff_error(e, cancel))?;
        if width == 0 || height == 0 {
            return Err(RasterError::InvalidDimensions { width, height });
        }

        let (pixel_layout, sample_bits, sample_type, ycbcr) =
            ensure_readable(&mut decoder, cancel)?;
        let tiled = opt_tag(decoder.get_tag_unsigned::<u32>(Tag::TileWidth), cancel)?.is_some();
        if !tiled && (width > MAX_SOURCE_SIDE || height > MAX_SOURCE_SIDE) {
            return Err(RasterError::LimitExceeded {
                width,
                height,
                estimated_bytes: u64::from(width)
                    .saturating_mul(u64::from(height))
                    .saturating_mul(4),
                limit_bytes: u64::from(MAX_SOURCE_SIDE)
                    .saturating_mul(u64::from(MAX_SOURCE_SIDE))
                    .saturating_mul(4),
            });
        }
        let depth = depth_of_layout(pixel_layout, sample_bits);
        let chunk = chunk_grid(&mut decoder, budget, depth, cancel)?;

        let nodata_f64 = parse_gdal_nodata(&mut decoder, cancel)?;
        let nodata = nodata_f64.and_then(|v| {
            if !v.is_finite() || v < f64::from(f32::MIN) || v > f64::from(f32::MAX) {
                None
            } else {
                Some(v as f32)
            }
        });

        let ParsedGeoref {
            ties,
            mut placement,
            foreign_datum_epsg,
            mut trouble,
        } = georef(&mut decoder, width, height, cancel)?;

        if placement.is_none() && ties.is_empty() && sidecar_placement.is_some() {
            placement = sidecar_placement;
            trouble = None;
        }
        let binding_trouble = trouble;

        let format = TiffFormat {
            pixel_layout,
            sample_bits,
            sample_type,
            depth,
            nodata,
            ycbcr,
        };
        let (overviews, ifd_index, decoder_clean) =
            scan_overviews(&mut decoder, format, budget, cancel)?;

        let grid = Grid {
            tiled,
            chunk,
            overviews,
            depth,
        };
        let levels = build_chunked_level_table(&grid, width, height, budget);
        if levels.is_empty() || levels.iter().all(|row| !row.fits) {
            let max_peak = levels.last().map_or(0, |row| row.estimated_peak_bytes);
            return Err(RasterError::LimitExceeded {
                width,
                height,
                estimated_bytes: max_peak,
                limit_bytes: budget.decode_working_bytes,
            });
        }

        let stats_ifd = grid
            .overviews
            .iter()
            .min_by_key(|o| o.width)
            .map_or(0, |o| o.image);

        let description = RasterDescription {
            width,
            height,
            bands: vec![BandInfo::display(
                format.sample_type,
                format.pixel_layout.channels,
                format.pixel_layout.has_alpha(),
                nodata_f64,
            )],
            ties,
            placement,
            foreign_datum_epsg,
            binding_trouble,
            levels,
        };

        let initial_mapping =
            (format.sample_bits <= 8).then(|| DisplayMapping::identity(format.nodata));
        let seed_decoder = decoder_clean.then_some((decoder, ifd_index));

        Ok(Self {
            decoder_pool: DecoderPool::new(byte_source, limits, seed_decoder),
            fingerprint,
            source_id: SourceId::from_fingerprint(&fingerprint),
            description,
            grid,
            format,
            stats_ifd,
            stretch: StretchCache::new(initial_mapping),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiff_limits_clamp_intermediate_buffer_to_safe_bounds() {
        let tiny_budget = MemoryBudget::for_host_ram(1024 * 1024 * 1024);
        let limits = build_tiff_limits(&tiny_budget);
        assert!(limits.intermediate_buffer_size >= 4 * 1024 * 1024);
        assert!(limits.intermediate_buffer_size <= 128 * 1024 * 1024);
    }
}
