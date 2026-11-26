//! Single-pass header/IFD discovery, cached chunk-table validation,
//! `GDAL_METADATA` parsing, remote COG enforcement, and overview scanning for
//! [`super::TiffSource::open`].

use std::sync::Arc;

use veldmap_core::budget::MemoryBudget;
use veldmap_core::io::ByteSource;
use veldmap_core::sync::CancelToken;
use veldmap_core::tile::{PixelFormat, SourceFingerprint, SourceId};

use crate::RasterError;
use crate::pipeline::grid::{Grid, Overview};
use crate::pipeline::metadata::{
    BandInfo, GeoPlacement, PlacementCrs, RasterDescription, SampleType,
};
use crate::pipeline::plan::{ServeMode, build_chunked_level_table, check_max_source_side};
use crate::pipeline::radiometry::DisplayMapping;

use super::gdal_meta::{GdalMetadata, TAG_GDAL_METADATA, parse_gdal_metadata, parse_gdal_nodata};
use super::georef::{ParsedGeoref, parse_georef};
use super::ifd::{CachedIfd, MAX_SINGLE_CHUNK_BYTES, Photometric, RawTiffAccess};
use super::ifd_parse::{max_cached_ifd_table_bytes, parse_cached_ifd};
use super::stretch::{BandStatsCache, StretchCache};
use super::tags::{
    SinglePassIfdReader, TAG_GDAL_NODATA, TAG_GEO_ASCII_PARAMS, TAG_GEO_DOUBLE_PARAMS,
    TAG_GEO_KEY_DIRECTORY, TAG_MODEL_PIXEL_SCALE, TAG_MODEL_TIEPOINT, TAG_MODEL_TRANSFORMATION,
    TAG_NEW_SUBFILE_TYPE, is_overview,
};
use super::{MAX_IFDS, TiffSource};

const BASE_METADATA_TAGS: &[u16] = &[
    TAG_GDAL_NODATA,
    TAG_GDAL_METADATA,
    TAG_GEO_KEY_DIRECTORY,
    TAG_GEO_DOUBLE_PARAMS,
    TAG_GEO_ASCII_PARAMS,
    TAG_MODEL_TIEPOINT,
    TAG_MODEL_PIXEL_SCALE,
    TAG_MODEL_TRANSFORMATION,
];

fn scan_overviews(
    reader: &mut SinglePassIfdReader,
    mut next_ifd: u64,
    base_ifd: &CachedIfd,
    budget: &MemoryBudget,
    mut table_bytes_used: u64,
    is_remote: bool,
    cancel: &CancelToken,
) -> Result<(Vec<Overview>, Vec<CachedIfd>), RasterError> {
    let mut overviews = Vec::new();
    let mut cached_overviews = Vec::new();
    let mut ifd_index = 0usize;

    while ifd_index + 1 < MAX_IFDS && next_ifd != 0 {
        if cancel.is_cancelled() {
            return Err(RasterError::Cancelled);
        }
        let mut dir = match reader.read_directory(next_ifd, cancel) {
            Ok(dir) => dir,
            Err(
                err @ (RasterError::Cancelled
                | RasterError::Io(_)
                | RasterError::DecoderLimitsExceeded(_)),
            ) => return Err(err),
            Err(err) => {
                tracing::debug!(error = %err, "Stopped scanning TIFF overviews");
                break;
            }
        };
        next_ifd = dir.next_ifd;
        ifd_index += 1;

        // `NewSubfileType` is inline; skip non-overview IFDs (e.g. transparency
        // masks) before fetching any out-of-line offset/table tags.
        let subfile = dir.read_u32_opt(reader, TAG_NEW_SUBFILE_TYPE).unwrap_or(0);
        if !is_overview(subfile) {
            continue;
        }
        let overview_ifd = match parse_cached_ifd(
            &mut dir,
            reader,
            ifd_index,
            budget,
            table_bytes_used,
            is_remote,
            cancel,
        ) {
            Ok(ifd) => ifd,
            Err(
                err @ (RasterError::Cancelled
                | RasterError::Io(_)
                | RasterError::DecoderLimitsExceeded(_)),
            ) => return Err(err),
            Err(RasterError::LimitExceeded { .. }) => break,
            Err(_) => continue,
        };
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
        let is_remote = byte_source.is_remote();
        let max_fetch_bytes = if is_remote {
            budget.remote_ifd_limit_bytes
        } else {
            u64::MAX
        };
        let mut reader =
            SinglePassIfdReader::open(Arc::clone(&byte_source), max_fetch_bytes, cancel)?;
        let byte_order = reader.byte_order;
        let first_ifd = reader.first_ifd;

        let mut base_dir = reader.read_directory(first_ifd, cancel)?;
        let next_ifd = base_dir.next_ifd;

        let base_ifd =
            parse_cached_ifd(&mut base_dir, &mut reader, 0, budget, 0, is_remote, cancel)?;
        if !base_ifd.tiled {
            if is_remote {
                return Err(RasterError::RemotePassUnsupported(
                    "remote GeoTIFF is stripped rather than tiled (not a Cloud-Optimized GeoTIFF)"
                        .to_owned(),
                ));
            }
            check_max_source_side(base_ifd.width, base_ifd.height)?;
        }

        let meta_cap = max_cached_ifd_table_bytes(budget, is_remote)
            .saturating_sub(base_ifd.table_bytes())
            .min(if is_remote {
                reader.remaining_fetch_budget()
            } else {
                u64::MAX
            });
        base_dir.prefetch_tags(&mut reader, BASE_METADATA_TAGS, meta_cap, cancel)?;

        let nodata_raw = base_dir.read_ascii_opt(&reader, TAG_GDAL_NODATA);
        let nodata_f64 = parse_gdal_nodata(&nodata_raw);
        let gdal_xml = base_dir.read_ascii_opt(&reader, TAG_GDAL_METADATA);
        let gdal_meta = if gdal_xml.is_empty() {
            GdalMetadata::default()
        } else {
            parse_gdal_metadata(&gdal_xml)
        };

        let raw_geotiff_tags = base_dir.read_geotiff_tags(&reader);
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
        let (overviews, cached_overviews) = scan_overviews(
            &mut reader,
            next_ifd,
            &base_ifd,
            budget,
            table_bytes_used,
            is_remote,
            cancel,
        )?;
        drop(reader);

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
            PixelFormat::Rgba8UnormSrgb,
            budget,
            !is_remote,
        );
        if is_remote
            && (levels.is_empty()
                || levels
                    .iter()
                    .any(|row| !matches!(row.serve, ServeMode::Pointwise) || !row.fits))
        {
            return Err(RasterError::RemotePassUnsupported(
                "remote GeoTIFF cannot be streamed pointwise within memory budget (missing coarse overviews or unsupported tile geometry)"
                    .to_owned(),
            ));
        }
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
        let initial_mapping = (base_ifd.sample_type == SampleType::U8
            || base_ifd.photometric == Photometric::Palette)
            .then(|| {
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
    fn ifd_table_budget_clamps_remote_limit_safely() {
        let tiny_budget = MemoryBudget::for_host_ram(1024 * 1024 * 1024);
        assert!(max_cached_ifd_table_bytes(&tiny_budget, false) >= 4 * 1024 * 1024);
        assert!(
            max_cached_ifd_table_bytes(&tiny_budget, true) <= tiny_budget.remote_ifd_limit_bytes
        );
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
