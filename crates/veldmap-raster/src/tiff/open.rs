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
use crate::pipeline::metadata::{BandInfo, GeoPlacement, RasterDescription};
use crate::pipeline::plan::{build_chunked_level_table, check_max_source_side};
use crate::pipeline::radiometry::DisplayMapping;

use super::chunk_reader::{DecoderPool, open_decoder};
use super::georef::{ParsedGeoref, georef};
use super::stretch::StretchCache;
use super::tags::{
    chunk_grid, depth_of_layout, ensure_readable, is_overview, map_tiff_error, opt_tag,
    parse_gdal_nodata,
};
use super::ycbcr::YCbCrParams;
use super::{MAX_IFDS, TiffFormat, TiffSource};

fn build_tiff_limits(budget: &MemoryBudget) -> Limits {
    let mut limits = Limits::default();
    limits.decoding_buffer_size = budget.decode_working_bytes.min(256 * 1024 * 1024) as usize;
    limits.intermediate_buffer_size = budget
        .decode_working_bytes
        .clamp(4 * 1024 * 1024, 128 * 1024 * 1024) as usize;
    limits
}

struct ScannedOverviews {
    overviews: Vec<Overview>,
    overview_ycbcr: Vec<(usize, YCbCrParams)>,
    ifd_index: usize,
    decoder_clean: bool,
}

fn scan_overviews(
    decoder: &mut Decoder<BufferedByteReader>,
    format: TiffFormat,
    budget: &MemoryBudget,
    cancel: &CancelToken,
) -> Result<ScannedOverviews, RasterError> {
    let mut overviews = Vec::new();
    let mut overview_ycbcr = Vec::new();
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
        let o_ycbcr = match ensure_readable(decoder, cancel) {
            Ok((o_layout, o_bits, o_type, o_ycbcr)) => {
                if o_layout != format.pixel_layout
                    || o_bits != format.sample_bits
                    || o_type != format.sample_type
                {
                    continue;
                }
                o_ycbcr
            }
            Err(RasterError::Cancelled) => return Err(RasterError::Cancelled),
            Err(_) => continue,
        };
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
        if let Some(params) = o_ycbcr {
            overview_ycbcr.push((ifd_index, params));
        }
        overviews.push(Overview {
            image: ifd_index,
            width: ow,
            height: oh,
            chunk: ochunk,
        });
    }
    Ok(ScannedOverviews {
        overviews,
        overview_ycbcr,
        ifd_index,
        decoder_clean,
    })
}

fn merge_georef_with_sidecar(
    placement: Option<GeoPlacement>,
    ties_empty: bool,
    embedded_affine: Option<[f64; 6]>,
    trouble: Option<String>,
    sidecar: Option<crate::sidecar::SidecarGeoref>,
) -> (Option<GeoPlacement>, Option<String>) {
    if placement.is_some() || !ties_empty {
        return (placement, trouble);
    }
    let Some(sc) = sidecar else {
        let fallback_placement = embedded_affine.map(|affine| GeoPlacement {
            epsg: None,
            custom_crs: None,
            datum_warning: None,
            affine,
            geo_params: None,
        });
        return (fallback_placement, trouble);
    };
    let Some(affine) = sc.affine.or(embedded_affine) else {
        return (None, trouble);
    };
    let has_crs = sc.crs.is_some();
    let merged_placement = Some(GeoPlacement {
        epsg: None,
        custom_crs: sc.crs,
        datum_warning: None,
        affine,
        geo_params: None,
    });
    let merged_trouble = if has_crs {
        None
    } else {
        trouble.or_else(|| {
            Some(
                "TIFF has pixel affine / world file (.tfw/.wld) but no accompanying .prj CRS sidecar; assign a CRS to place it"
                    .to_string(),
            )
        })
    };
    (merged_placement, merged_trouble)
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
        Self::open_with_sidecar(byte_source, fingerprint, None, budget, cancel)
    }

    /// Open a GeoTIFF/BigTIFF/COG with optional external [`crate::SidecarGeoref`]
    /// (`.tfw`/`.wld` and/or `.prj` sidecar).
    ///
    /// Embedded GeoTIFF georeferencing always takes precedence; when the TIFF
    /// carries embedded pixel affine tags without a CRS, an external `.prj`
    /// sidecar can supply the missing CRS without requiring a `.tfw`.
    pub fn open_with_sidecar(
        byte_source: Arc<dyn ByteSource>,
        fingerprint: SourceFingerprint,
        sidecar: Option<crate::sidecar::SidecarGeoref>,
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
        if !tiled {
            check_max_source_side(width, height)?;
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
            placement,
            embedded_affine,
            foreign_datum_epsg,
            geographic_datum,
            trouble,
        } = georef(&mut decoder, width, height, cancel)?;

        let (placement, binding_trouble) = merge_georef_with_sidecar(
            placement,
            ties.is_empty(),
            embedded_affine,
            trouble,
            sidecar,
        );

        let format = TiffFormat {
            pixel_layout,
            sample_bits,
            sample_type,
            depth,
            nodata,
        };
        let ScannedOverviews {
            overviews,
            overview_ycbcr: mut ifd_ycbcr,
            ifd_index,
            decoder_clean,
        } = scan_overviews(&mut decoder, format, budget, cancel)?;
        if let Some(base_ycbcr) = ycbcr {
            ifd_ycbcr.push((0, base_ycbcr));
        }

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
            geographic_datum,
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
            ifd_ycbcr,
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

    #[test]
    fn merge_georef_with_sidecar_preserves_trouble_when_sidecar_lacks_crs() {
        let affine = [1.0, 0.0, 100.0, 0.0, -1.0, 200.0];
        let sidecar_no_crs = crate::sidecar::SidecarGeoref {
            affine: Some(affine),
            crs: None,
        };
        let (placement, trouble) = merge_georef_with_sidecar(
            None,
            true,
            None,
            Some("embedded CRS error".to_owned()),
            Some(sidecar_no_crs.clone()),
        );
        assert_eq!(placement.as_ref().map(|p| p.affine), Some(affine));
        assert_eq!(trouble.as_deref(), Some("embedded CRS error"));

        // Without prior embedded trouble, missing `.prj` still reports guidance.
        let (_, fallback_trouble) =
            merge_georef_with_sidecar(None, true, None, None, Some(sidecar_no_crs));
        assert!(
            fallback_trouble
                .as_deref()
                .is_some_and(|msg| msg.contains(".prj"))
        );

        // When sidecar supplies a CRS, trouble is cleared.
        let sidecar_with_crs = crate::sidecar::SidecarGeoref {
            affine: Some(affine),
            crs: Some(veldmap_core::geo::CrsSpec::from_epsg(4326).expect("EPSG:4326")),
        };
        let (placement_ok, trouble_ok) = merge_georef_with_sidecar(
            None,
            true,
            None,
            Some("embedded CRS error".to_owned()),
            Some(sidecar_with_crs),
        );
        assert!(
            placement_ok
                .as_ref()
                .and_then(|p| p.custom_crs.as_ref())
                .is_some()
        );
        assert!(trouble_ok.is_none());
    }
}
