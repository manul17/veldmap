//! Single-pass header/IFD discovery, cached chunk-table validation,
//! `GDAL_METADATA` parsing, remote COG enforcement, and overview scanning for
//! [`super::TiffSource::open`].

use std::sync::Arc;

use veldmap_core::budget::MemoryBudget;
use veldmap_core::geo::CrsSpec;
use veldmap_core::io::ByteSource;
use veldmap_core::sync::CancelToken;
use veldmap_core::tile::{BandKey, PixelFormat, SourceFingerprint, SourceId};

use crate::RasterError;
use crate::pipeline::{
    BandInfo, DisplayMapping, ExternalBandMeta, GeoPlacement, GeoTie, Grid, LevelPlan, Overview,
    PlacementCrs, RasterDescription, ServeMode, build_chunked_level_table, check_max_source_side,
};

use super::gdal_meta::{GdalMetadata, parse_gdal_metadata, parse_gdal_nodata};
use super::georef::{ParsedGeoref, RawGeoTiffTags, parse_georef};
use super::ifd::{
    CachedIfd, IfdDirectory, IfdParseBudget, MAX_SINGLE_CHUNK_BYTES, Photometric,
    SinglePassIfdReader, is_overview, parse_cached_ifd,
};
use super::stretch::{BandStatsCache, StretchCache};
use super::tag_ids::{
    TAG_DATETIME, TAG_GDAL_METADATA, TAG_GDAL_NODATA, TAG_GEO_ASCII_PARAMS, TAG_GEO_DOUBLE_PARAMS,
    TAG_GEO_KEY_DIRECTORY, TAG_MODEL_PIXEL_SCALE, TAG_MODEL_TIEPOINT, TAG_MODEL_TRANSFORMATION,
    TAG_NEW_SUBFILE_TYPE,
};
use super::view::{
    RawTiffAccess, identity_display_mapping, needs_display_stretch, uniform_display_nodata,
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

fn read_geotiff_tags(dir: &IfdDirectory, reader: &SinglePassIfdReader) -> RawGeoTiffTags {
    RawGeoTiffTags {
        raw_keys: dir.read_u16_vec_opt(reader, TAG_GEO_KEY_DIRECTORY),
        double_params: dir.read_f64_vec_opt(reader, TAG_GEO_DOUBLE_PARAMS),
        ascii_params: dir.read_ascii_bytes_opt(reader, TAG_GEO_ASCII_PARAMS),
        points: dir.read_f64_vec_opt(reader, TAG_MODEL_TIEPOINT),
        scale: dir.read_f64_vec_opt(reader, TAG_MODEL_PIXEL_SCALE),
        matrix: dir.read_f64_vec_opt(reader, TAG_MODEL_TRANSFORMATION),
    }
}

struct BaseMetadata {
    nodata_f64: Option<f64>,
    gdal_meta: GdalMetadata,
    ties: Vec<GeoTie>,
    placement: Option<GeoPlacement>,
    geographic_crs: Option<CrsSpec>,
    binding_trouble: Option<String>,
}

fn read_base_metadata_and_georef(
    base_dir: &mut IfdDirectory,
    reader: &mut SinglePassIfdReader,
    base_ifd: &CachedIfd,
    parse_budget: IfdParseBudget<'_>,
    sidecar: Option<crate::sidecar::SidecarGeoref>,
    cancel: &CancelToken,
) -> Result<BaseMetadata, RasterError> {
    let meta_cap = parse_budget.metadata_prefetch_cap(reader);
    base_dir.prefetch_tags(reader, BASE_METADATA_TAGS, meta_cap, cancel)?;

    let nodata_raw = base_dir.read_ascii_opt(reader, TAG_GDAL_NODATA);
    let nodata_f64 = parse_gdal_nodata(&nodata_raw);
    let gdal_xml = base_dir.read_ascii_opt(reader, TAG_GDAL_METADATA);
    let gdal_meta = if gdal_xml.is_empty() {
        GdalMetadata::default()
    } else {
        parse_gdal_metadata(&gdal_xml)
    };

    let raw_geotiff_tags = read_geotiff_tags(base_dir, reader);
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

    Ok(BaseMetadata {
        nodata_f64,
        gdal_meta,
        ties,
        placement,
        geographic_crs,
        binding_trouble,
    })
}

fn plan_and_validate_levels(
    grid: &Grid,
    width: u32,
    height: u32,
    budget: &MemoryBudget,
    is_remote: bool,
) -> Result<Vec<LevelPlan>, RasterError> {
    let levels = build_chunked_level_table(
        grid,
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
    Ok(levels)
}

fn is_fatal_overview_err(err: &RasterError) -> bool {
    matches!(
        err,
        RasterError::Cancelled | RasterError::Io(_) | RasterError::DecoderLimitsExceeded(_)
    )
}

fn is_compatible_overview(base_ifd: &CachedIfd, overview_ifd: &CachedIfd) -> bool {
    let photo_compat = overview_ifd.photometric == base_ifd.photometric
        || (base_ifd.photometric == Photometric::Rgb
            && overview_ifd.photometric == Photometric::YCbCr);
    photo_compat
        && overview_ifd.samples_per_pixel == base_ifd.samples_per_pixel
        && overview_ifd.sample_type == base_ifd.sample_type
        && overview_ifd.alpha_channel() == base_ifd.alpha_channel()
}

fn scan_overviews(
    reader: &mut SinglePassIfdReader,
    mut next_ifd: u64,
    base_ifd: &CachedIfd,
    mut parse_budget: IfdParseBudget<'_>,
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
            Err(err) if is_fatal_overview_err(&err) => return Err(err),
            Err(err) => {
                tracing::debug!(error = %err, "Stopped scanning TIFF overviews");
                break;
            }
        };
        next_ifd = dir.next_ifd();
        ifd_index += 1;

        // `NewSubfileType` is inline; skip non-overview IFDs (e.g. transparency
        // masks) before fetching any out-of-line offset/table tags.
        let subfile = dir.read_u32_opt(reader, TAG_NEW_SUBFILE_TYPE).unwrap_or(0);
        if !is_overview(subfile) {
            continue;
        }
        let overview_ifd = match parse_cached_ifd(&mut dir, reader, ifd_index, parse_budget, cancel)
        {
            Ok(ifd) => ifd,
            Err(err) if is_fatal_overview_err(&err) => return Err(err),
            Err(RasterError::LimitExceeded { .. }) => break,
            Err(_) => continue,
        };
        if !is_compatible_overview(base_ifd, &overview_ifd) {
            continue;
        }
        parse_budget.charge_ifd(&overview_ifd);
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
        bands.push(BandInfo {
            band_key: BandKey::Index(b),
            index: b,
            name: gmeta.description.or(default_name),
            units: gmeta.units,
            pixel_format,
            sample_type: base_ifd.sample_type,
            scale: gmeta.scale.unwrap_or(1.0),
            offset: gmeta.offset.unwrap_or(0.0),
            has_embedded_scale_offset: gmeta.scale.is_some() || gmeta.offset.is_some(),
            nodata: gmeta.nodata.or(nodata_f64),
            channels: 1,
            has_alpha: false,
        });
    }
    bands
}

struct BandsAndStretch {
    bands: Vec<BandInfo>,
    per_band_nodata: Vec<Option<f64>>,
    display_nodata: Option<f64>,
    initial_mapping: Option<DisplayMapping>,
}

fn build_bands_and_stretch(
    base_ifd: &CachedIfd,
    file_nodata: Option<f64>,
    gdal_meta: &GdalMetadata,
    external_bands: &[ExternalBandMeta],
) -> BandsAndStretch {
    let mut bands = build_band_infos(base_ifd, file_nodata, gdal_meta);
    if !external_bands.is_empty() {
        for band in &mut bands {
            let BandKey::Index(b) = band.band_key else {
                continue;
            };
            let Some(ext) = external_bands.get(usize::from(b)) else {
                continue;
            };
            if !band.has_embedded_scale_offset && (ext.scale.is_some() || ext.offset.is_some()) {
                let scale = ext.scale.unwrap_or(1.0);
                let offset = ext.offset.unwrap_or(0.0);
                if scale.is_finite() && scale != 0.0 && offset.is_finite() {
                    band.scale = scale;
                    band.offset = offset;
                }
            }
            if band.nodata.is_none()
                && let Some(nd) = ext.nodata
                && (nd.is_finite() || nd.is_nan())
            {
                band.nodata = Some(nd);
            }
            if band.units.is_none()
                && let Some(unit) = ext.unit.as_deref().map(str::trim).filter(|s| !s.is_empty())
            {
                band.units = Some(unit.to_owned());
            }
        }
    }

    let mut per_band_nodata = vec![file_nodata; usize::from(base_ifd.samples_per_pixel)];
    for info in &bands {
        if let BandKey::Index(b) = info.band_key
            && let Some(slot) = per_band_nodata.get_mut(usize::from(b))
        {
            *slot = info.nodata;
        }
    }
    let display_nodata = file_nodata.or_else(|| {
        uniform_display_nodata(&per_band_nodata, base_ifd.pixel_layout.colors().max(1))
    });
    if let Some(nd) = display_nodata
        && let Some(display) = bands.iter_mut().find(|b| b.band_key == BandKey::Display)
        && display.nodata.is_none()
    {
        display.nodata = Some(nd);
    }
    let initial_mapping = (!needs_display_stretch(base_ifd))
        .then(|| identity_display_mapping(base_ifd, display_nodata));
    BandsAndStretch {
        bands,
        per_band_nodata,
        display_nodata,
        initial_mapping,
    }
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
        Self::open_with_options(byte_source, fingerprint, None, &[], budget, cancel)
    }

    /// Open a GeoTIFF/BigTIFF/COG with optional sidecar georeferencing and
    /// external per-band radiometric metadata merged before nodata and stretch
    /// initialization.
    pub(crate) fn open_with_options(
        byte_source: Arc<dyn ByteSource>,
        fingerprint: SourceFingerprint,
        sidecar: Option<crate::sidecar::SidecarGeoref>,
        external_bands: &[ExternalBandMeta],
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
        let next_ifd = base_dir.next_ifd();

        let mut parse_budget = IfdParseBudget::new(budget, is_remote);
        let base_ifd = parse_cached_ifd(&mut base_dir, &mut reader, 0, parse_budget, cancel)?;
        if !base_ifd.tiled {
            if is_remote {
                return Err(RasterError::RemotePassUnsupported(
                    "remote GeoTIFF is stripped rather than tiled (not a Cloud-Optimized GeoTIFF)"
                        .to_owned(),
                ));
            }
            check_max_source_side(base_ifd.width, base_ifd.height)?;
        }
        parse_budget.charge_ifd(&base_ifd);

        let meta = read_base_metadata_and_georef(
            &mut base_dir,
            &mut reader,
            &base_ifd,
            parse_budget,
            sidecar,
            cancel,
        )?;
        let (overviews, cached_overviews) =
            scan_overviews(&mut reader, next_ifd, &base_ifd, parse_budget, cancel)?;
        drop(reader);

        let (width, height) = (base_ifd.width, base_ifd.height);
        let grid = Grid {
            chunk: (base_ifd.chunk_w, base_ifd.chunk_h),
            overviews,
            depth: base_ifd.raw_pixel_bytes(),
        };
        let levels = plan_and_validate_levels(&grid, width, height, budget, is_remote)?;
        let stats_ifd = grid
            .overviews
            .iter()
            .min_by_key(|o| o.width)
            .map_or(0, |o| o.image);

        let BandsAndStretch {
            bands,
            per_band_nodata,
            display_nodata,
            initial_mapping,
        } = build_bands_and_stretch(&base_ifd, meta.nodata_f64, &meta.gdal_meta, external_bands);

        let mut ifds = Vec::with_capacity(1 + cached_overviews.len());
        ifds.push(base_ifd);
        ifds.extend(cached_overviews);

        let description = RasterDescription {
            width,
            height,
            bands,
            ties: meta.ties,
            placement: meta.placement,
            geographic_crs: meta.geographic_crs,
            binding_trouble: meta.binding_trouble,
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
            nodata_f64: display_nodata,
            per_band_nodata,
            stats_ifd,
            budget_snapshot: *budget,
            stretch: StretchCache::new(initial_mapping),
            band_stats: BandStatsCache::new(),
        })
    }
}

/// Read the base IFD's TIFF `DateTime` ASCII tag (`306`) without decoding
/// image tiles or scanning overviews, returning the trimmed non-empty tag
/// value if present (e.g. `"2024:06:15 10:30:00"`).
pub fn read_tiff_datetime(
    byte_source: Arc<dyn ByteSource>,
    cancel: &CancelToken,
) -> Option<String> {
    const INITIAL_PREFIX_BYTES: usize = 4096;
    const MAX_IFD_PROBE_BYTES: u64 = 128 * 1024;
    const MAX_DATETIME_TAG_BYTES: u64 = 1024;
    let mut reader = SinglePassIfdReader::open_with_prefix(
        byte_source,
        INITIAL_PREFIX_BYTES,
        MAX_IFD_PROBE_BYTES,
        cancel,
    )
    .ok()?;
    let first_ifd = reader.first_ifd;
    let mut base_dir = reader.read_directory(first_ifd, cancel).ok()?;
    let tag_cap = reader.remaining_fetch_budget().min(MAX_DATETIME_TAG_BYTES);
    base_dir
        .prefetch_tags(&mut reader, &[TAG_DATETIME], tag_cap, cancel)
        .ok()?;
    let raw = base_dir.read_ascii_opt(&reader, TAG_DATETIME);
    let trimmed = raw.trim().trim_end_matches('\0').trim();
    (!trimmed.is_empty()).then(|| trimmed.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

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
            crs: Some(CrsSpec::from_epsg(4326).expect("EPSG:4326")),
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

    struct VecByteSource(Vec<u8>);

    impl ByteSource for VecByteSource {
        fn len(&self) -> u64 {
            self.0.len() as u64
        }

        fn read_at(
            &self,
            offset: u64,
            buf: &mut [u8],
            _cancel: &CancelToken,
        ) -> Result<usize, veldmap_core::io::IoError> {
            let start = usize::try_from(offset).unwrap_or(self.0.len());
            if start >= self.0.len() {
                return Ok(0);
            }
            let n = buf.len().min(self.0.len() - start);
            buf[..n].copy_from_slice(&self.0[start..start + n]);
            Ok(n)
        }
    }

    fn build_eof_ifd_classic_tiff(pad_bytes: usize, big_endian: bool, dt: Option<&str>) -> Vec<u8> {
        let mut buf = Vec::new();
        let write_u16 = |b: &mut Vec<u8>, v: u16| {
            b.extend_from_slice(&if big_endian {
                v.to_be_bytes()
            } else {
                v.to_le_bytes()
            });
        };
        let write_u32 = |b: &mut Vec<u8>, v: u32| {
            b.extend_from_slice(&if big_endian {
                v.to_be_bytes()
            } else {
                v.to_le_bytes()
            });
        };
        buf.extend_from_slice(if big_endian { b"MM" } else { b"II" });
        write_u16(&mut buf, 42);
        let ifd_offset = (8 + pad_bytes) as u32;
        write_u32(&mut buf, ifd_offset);
        buf.resize(8 + pad_bytes, 0);

        if let Some(dt_str) = dt {
            let mut ascii = dt_str.as_bytes().to_vec();
            ascii.push(0);
            let value_offset = ifd_offset + 2 + 12 + 4;
            write_u16(&mut buf, 1);
            write_u16(&mut buf, TAG_DATETIME);
            write_u16(&mut buf, 2); // ASCII
            write_u32(&mut buf, ascii.len() as u32);
            if ascii.len() <= 4 {
                let mut inline = [0u8; 4];
                inline[..ascii.len()].copy_from_slice(&ascii);
                buf.extend_from_slice(&inline);
                write_u32(&mut buf, 0);
            } else {
                write_u32(&mut buf, value_offset);
                write_u32(&mut buf, 0);
                buf.extend_from_slice(&ascii);
            }
        } else {
            write_u16(&mut buf, 0);
            write_u32(&mut buf, 0);
        }
        buf
    }

    fn build_eof_ifd_bigtiff(pad_bytes: usize, dt: &str) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"II");
        buf.extend_from_slice(&43u16.to_le_bytes());
        buf.extend_from_slice(&8u16.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes());
        let ifd_offset = (16 + pad_bytes) as u64;
        buf.extend_from_slice(&ifd_offset.to_le_bytes());
        buf.resize(16 + pad_bytes, 0);

        let mut ascii = dt.as_bytes().to_vec();
        ascii.push(0);
        let value_offset = ifd_offset + 8 + 20 + 8;
        buf.extend_from_slice(&1u64.to_le_bytes());
        buf.extend_from_slice(&TAG_DATETIME.to_le_bytes());
        buf.extend_from_slice(&2u16.to_le_bytes()); // ASCII
        buf.extend_from_slice(&(ascii.len() as u64).to_le_bytes());
        buf.extend_from_slice(&value_offset.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&ascii);
        buf
    }

    #[test]
    fn read_tiff_datetime_handles_eof_ifd_beyond_64k_classic_bigtiff_and_big_endian() {
        let cancel = CancelToken::new();

        let classic_le = build_eof_ifd_classic_tiff(128 * 1024, false, Some("2025:07:14 08:15:00"));
        assert_eq!(
            read_tiff_datetime(Arc::new(VecByteSource(classic_le)), &cancel).as_deref(),
            Some("2025:07:14 08:15:00")
        );

        let classic_be = build_eof_ifd_classic_tiff(96 * 1024, true, Some("2024:11:03 19:45:10"));
        assert_eq!(
            read_tiff_datetime(Arc::new(VecByteSource(classic_be)), &cancel).as_deref(),
            Some("2024:11:03 19:45:10")
        );

        let bigtiff = build_eof_ifd_bigtiff(128 * 1024, "2026:01:22 12:00:00");
        assert_eq!(
            read_tiff_datetime(Arc::new(VecByteSource(bigtiff)), &cancel).as_deref(),
            Some("2026:01:22 12:00:00")
        );

        let missing = build_eof_ifd_classic_tiff(96 * 1024, false, None);
        assert_eq!(
            read_tiff_datetime(Arc::new(VecByteSource(missing)), &cancel),
            None
        );

        let empty_dt = build_eof_ifd_classic_tiff(64, false, Some("   "));
        assert_eq!(
            read_tiff_datetime(Arc::new(VecByteSource(empty_dt)), &cancel),
            None
        );
    }
}
