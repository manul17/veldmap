//! Header/IFD discovery, cached chunk-table validation, `GDAL_METADATA`
//! parsing, and overview scanning for [`super::TiffSource::open`].

use std::sync::Arc;

use tiff::decoder::{Decoder, Limits};
use tiff::tags::Tag;
use veldmap_core::budget::MemoryBudget;
use veldmap_core::io::{BufferedByteReader, ByteSource};
use veldmap_core::sync::CancelToken;
use veldmap_core::tile::{PixelFormat, SourceFingerprint, SourceId};

use crate::RasterError;
use crate::pipeline::grid::{Grid, Overview};
use crate::pipeline::metadata::{
    BandInfo, GeoPlacement, PlacementCrs, RasterDescription, SampleType,
};
use crate::pipeline::plan::{build_chunked_level_table, check_max_source_side};
use crate::pipeline::radiometry::DisplayMapping;

use super::gdal_meta::{GdalMetadata, TAG_GDAL_METADATA, parse_gdal_metadata, parse_gdal_nodata};
use super::georef::{ParsedGeoref, parse_georef};
use super::ifd::{CachedIfd, MAX_SINGLE_CHUNK_BYTES, Photometric, RawTiffAccess, TiffByteOrder};
use super::ifd_parse::parse_cached_ifd;
use super::stretch::{BandStatsCache, StretchCache};
use super::tags::{is_overview, map_tiff_error, opt_tag, read_geotiff_tags, read_tag_ascii_opt};
use super::{MAX_IFDS, TiffSource};

fn build_tiff_limits(budget: &MemoryBudget) -> Limits {
    let mut limits = Limits::default();
    // Enforced by `tiff 0.11.3` during open-time IFD tag value reads
    // (TileOffsets, ColorMap, JPEGTables, ASCII tags).
    limits.decoding_buffer_size = budget.decode_working_bytes.min(MAX_SINGLE_CHUNK_BYTES) as usize;
    limits.intermediate_buffer_size = budget
        .decode_working_bytes
        .clamp(4 * 1024 * 1024, 128 * 1024 * 1024) as usize;
    limits
}

fn open_tag_decoder(
    byte_source: &Arc<dyn ByteSource>,
    limits: &Limits,
    cancel: &CancelToken,
) -> Result<(TiffByteOrder, Decoder<BufferedByteReader>), RasterError> {
    use std::io::{Read, Seek, SeekFrom};

    let mut reader = BufferedByteReader::new(Arc::clone(byte_source), cancel.clone());
    let mut magic = [0u8; 2];
    reader
        .read_exact(&mut magic)
        .map_err(|e| crate::map_std_io_error(e, cancel))?;
    let byte_order = match &magic {
        b"II" => TiffByteOrder::Little,
        b"MM" => TiffByteOrder::Big,
        _ => return Err(RasterError::UnsupportedFormat),
    };
    reader
        .seek(SeekFrom::Start(0))
        .map_err(|e| crate::map_std_io_error(e, cancel))?;
    let decoder = Decoder::new(reader)
        .map_err(|e| map_tiff_error(e, cancel))?
        .with_limits(limits.clone());
    Ok((byte_order, decoder))
}

fn scan_overviews(
    decoder: &mut Decoder<BufferedByteReader>,
    base_ifd: &CachedIfd,
    budget: &MemoryBudget,
    mut table_bytes_used: u64,
    cancel: &CancelToken,
) -> Result<(Vec<Overview>, Vec<CachedIfd>), RasterError> {
    let mut overviews = Vec::new();
    let mut cached_overviews = Vec::new();
    let mut ifd_index = 0usize;
    while ifd_index + 1 < MAX_IFDS && decoder.more_images() {
        if cancel.is_cancelled() {
            return Err(RasterError::Cancelled);
        }
        if let Err(err) = decoder.next_image() {
            if cancel.is_cancelled() {
                return Err(RasterError::Cancelled);
            }
            tracing::debug!(error = %err, "Stopped scanning TIFF overviews");
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
        let overview_ifd =
            match parse_cached_ifd(decoder, ifd_index, budget, table_bytes_used, cancel) {
                Ok(ifd) => ifd,
                Err(RasterError::Cancelled) => return Err(RasterError::Cancelled),
                Err(_) => continue,
            };
        // Overviews must match base sample count, bits, sample type, and photometric
        // compatibility (allowing YCbCr overviews on RGB base images).
        let photo_compat = overview_ifd.photometric == base_ifd.photometric
            || (base_ifd.photometric == Photometric::Rgb
                && overview_ifd.photometric == Photometric::YCbCr);
        if !photo_compat
            || overview_ifd.samples_per_pixel != base_ifd.samples_per_pixel
            || overview_ifd.sample_type != base_ifd.sample_type
            || overview_ifd.alpha_channel() != base_ifd.alpha_channel()
        {
            continue;
        }
        table_bytes_used = table_bytes_used.saturating_add(overview_ifd.table_bytes());
        overviews.push(Overview {
            image: ifd_index,
            width: overview_ifd.width,
            height: overview_ifd.height,
            chunk: (overview_ifd.chunk_w, overview_ifd.chunk_h),
        });
        cached_overviews.push(overview_ifd);
    }
    Ok((overviews, cached_overviews))
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
        let fallback_placement = embedded_affine.map(|affine| GeoPlacement { crs: None, affine });
        return (fallback_placement, trouble);
    };
    let Some(affine) = sc.affine.or(embedded_affine) else {
        return (None, trouble);
    };
    let has_crs = sc.crs.is_some();
    let merged_placement = Some(GeoPlacement {
        crs: sc
            .crs
            .map(|crs| PlacementCrs::Resolved { crs, warning: None }),
        affine,
    });
    let merged_trouble = if has_crs {
        None
    } else {
        trouble.or_else(|| {
            Some(crate::sidecar::missing_prj_trouble(
                "TIFF has pixel affine / world file (.tfw/.wld)",
            ))
        })
    };
    (merged_placement, merged_trouble)
}

fn build_band_infos(
    base_ifd: &CachedIfd,
    nodata_f64: Option<f64>,
    gdal_meta: &GdalMetadata,
) -> Vec<BandInfo> {
    let mut bands = Vec::with_capacity(1 + usize::from(base_ifd.data_bands));
    bands.push(BandInfo::display(
        base_ifd.sample_type,
        usize::from(base_ifd.samples_per_pixel),
        base_ifd.alpha_channel().is_some(),
        nodata_f64,
    ));

    let is_palette = base_ifd.photometric == Photometric::Palette;
    for b in 0..base_ifd.samples_per_pixel {
        if base_ifd.alpha_channel() == Some(usize::from(b)) {
            continue;
        }
        let gmeta = gdal_meta.for_band(b);
        let pixel_format = if is_palette {
            base_ifd
                .sample_type
                .categorical_pixel_format()
                .unwrap_or(PixelFormat::R8Uint)
        } else {
            PixelFormat::R32Float
        };
        let default_name = if base_ifd.photometric == Photometric::Rgb && b < 4 {
            match b {
                0 => Some("Red".to_owned()),
                1 => Some("Green".to_owned()),
                2 => Some("Blue".to_owned()),
                3 => Some("NIR / Band 4".to_owned()),
                _ => None,
            }
        } else if is_palette {
            Some("Class Index".to_owned())
        } else {
            None
        };
        bands.push(BandInfo::band(
            b,
            gmeta.description.or(default_name),
            gmeta.units,
            pixel_format,
            base_ifd.sample_type,
            gmeta.scale.unwrap_or(1.0),
            gmeta.offset.unwrap_or(0.0),
            nodata_f64,
        ));
    }
    bands
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
        let (byte_order, mut decoder) = open_tag_decoder(&byte_source, &limits, cancel)?;

        let base_ifd = parse_cached_ifd(&mut decoder, 0, budget, 0, cancel)?;
        if !base_ifd.tiled {
            check_max_source_side(base_ifd.width, base_ifd.height)?;
        }

        let nodata_raw = read_tag_ascii_opt(&mut decoder, Tag::GdalNodata, cancel)?;
        let nodata_f64 = parse_gdal_nodata(&nodata_raw);
        let gdal_xml = read_tag_ascii_opt(&mut decoder, Tag::Unknown(TAG_GDAL_METADATA), cancel)?;
        let gdal_meta = if gdal_xml.is_empty() {
            GdalMetadata::default()
        } else {
            parse_gdal_metadata(&gdal_xml)
        };

        let raw_geotiff_tags = read_geotiff_tags(&mut decoder, cancel)?;
        let ParsedGeoref {
            ties,
            placement,
            embedded_affine,
            geographic_crs,
            trouble,
        } = parse_georef(&raw_geotiff_tags, base_ifd.width, base_ifd.height);

        let (placement, binding_trouble) = merge_georef_with_sidecar(
            placement,
            ties.is_empty(),
            embedded_affine,
            trouble,
            sidecar,
        );

        let table_bytes_used = base_ifd.table_bytes();
        let (overviews, cached_overviews) =
            scan_overviews(&mut decoder, &base_ifd, budget, table_bytes_used, cancel)?;

        // Drop `tiff::decoder::Decoder` immediately after open — all subsequent
        // chunk reads execute directly via `raw_decode` over `byte_source`.
        drop(decoder);

        let width = base_ifd.width;
        let height = base_ifd.height;
        let depth = base_ifd.raw_pixel_bytes();
        let grid = Grid {
            chunk: (base_ifd.chunk_w, base_ifd.chunk_h),
            overviews,
            depth,
        };
        let levels = build_chunked_level_table(
            &grid,
            width,
            height,
            veldmap_core::tile::PixelFormat::Rgba8UnormSrgb,
            budget,
        );
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

        let bands = build_band_infos(&base_ifd, nodata_f64, &gdal_meta);
        // Only unsigned 8-bit rasters use identity [0, 255] display mapping; signed
        // `I8` (-128..=127) and >8-bit / float types compute a percentile stretch.
        let initial_mapping = (base_ifd.sample_type == SampleType::U8).then(|| {
            DisplayMapping::identity(nodata_f64)
                .with_invert(base_ifd.photometric == Photometric::WhiteIsZero)
        });

        let mut ifds = Vec::with_capacity(1 + cached_overviews.len());
        ifds.push(base_ifd);
        ifds.extend(cached_overviews);

        let description = RasterDescription {
            width,
            height,
            bands,
            ties,
            placement,
            geographic_crs,
            binding_trouble,
            levels,
        };

        let access = RawTiffAccess {
            byte_source,
            byte_order,
            max_chunk_bytes: budget.decode_working_bytes.min(MAX_SINGLE_CHUNK_BYTES) as usize,
        };

        Ok(Self {
            access,
            fingerprint,
            source_id: SourceId::from_fingerprint(&fingerprint),
            description,
            grid,
            ifds,
            nodata_f64,
            stats_ifd,
            budget_snapshot: *budget,
            stretch: StretchCache::new(initial_mapping),
            band_stats: BandStatsCache::new(),
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

        let (_, fallback_trouble) =
            merge_georef_with_sidecar(None, true, None, None, Some(sidecar_no_crs));
        assert!(
            fallback_trouble
                .as_deref()
                .is_some_and(|msg| msg.contains(".prj"))
        );

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
        assert!(matches!(
            placement_ok.as_ref().and_then(|p| p.crs.as_ref()),
            Some(PlacementCrs::Resolved { .. })
        ));
        assert!(trouble_ok.is_none());
    }
}
