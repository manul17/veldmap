use std::io::Cursor;
use std::sync::Arc;

use ::tiff::encoder::TiffEncoder;
use ::tiff::tags::Tag;
use veldmap_core::budget::{DecodeBudgetTracker, MemoryBudget};
use veldmap_core::io::ByteSource;
use veldmap_core::io::testing::{CountingByteSource, MemoryByteSource};
use veldmap_core::sync::CancelToken;
use veldmap_core::tile::{RASTER_TILE_SIZE, tile_extent};
use veldmap_raster::{
    DecodeContext, MAX_IFDS, MAX_SOURCE_SIDE, RasterError, RasterSource, SharedChunkCache,
    TiffSource, frame_from_description, has_tiff_magic,
};

use crate::common::{build_stripped_tiff, build_tiled_cog, fake_fingerprint, rgb_pattern};

const WINDOW: u64 = 64 * 1024;

fn overlaps(a: (u64, u64), b: (u64, u64)) -> bool {
    a.0 < b.0 + b.1 && b.0 < a.0 + a.1
}

fn overlap_bytes(a: (u64, u64), b: (u64, u64)) -> u64 {
    let from = a.0.max(b.0);
    let to = (a.0 + a.1).min(b.0 + b.1);
    to.saturating_sub(from)
}

fn near_head_or_ifd(span: (u64, u64), ifds: &[u64]) -> bool {
    overlaps(span, (0, WINDOW))
        || ifds
            .iter()
            .any(|&at| overlaps(span, (at.saturating_sub(WINDOW), 2 * WINDOW)))
}

#[test]
fn describing_reads_only_header_and_ifds_and_pointwise_reuses_pooled_decoder() {
    let levels = 2;
    let (file, base_tiles, ifds) =
        build_tiled_cog(2 * RASTER_TILE_SIZE, 2 * RASTER_TILE_SIZE, levels);
    assert!(file.len() as u64 > 8 * WINDOW);

    let fp = fake_fingerprint(&file);
    let counting = Arc::new(CountingByteSource::new(Arc::new(
        MemoryByteSource::from_vec(file),
    )));
    let budget = MemoryBudget::default();
    let cancel = CancelToken::new();

    let src = TiffSource::open(
        Arc::clone(&counting) as Arc<dyn ByteSource>,
        fp,
        &budget,
        &cancel,
    )
    .expect("open COG");
    assert_eq!(src.overview_count(), levels as usize);
    assert!(src.is_tiled());

    let describe_stats = counting.snapshot();
    assert!(
        describe_stats.read_calls <= 1 + 3 * (levels as u64 + 1),
        "describe issued {} read_at calls (expected <= {})",
        describe_stats.read_calls,
        1 + 3 * (levels as u64 + 1)
    );
    let base_total: u64 = base_tiles.iter().map(|t| t.1).sum();
    let over_base: u64 = describe_stats
        .spans
        .iter()
        .map(|s| {
            base_tiles
                .iter()
                .map(|t| overlap_bytes((s.offset, s.len as u64), *t))
                .sum::<u64>()
        })
        .sum();
    assert!(
        over_base * 4 < base_total,
        "describe read {over_base} B of {base_total} B base pixels"
    );
    for span in &describe_stats.spans {
        assert!(
            near_head_or_ifd((span.offset, span.len as u64), &ifds),
            "describe read far from header/IFDs: {span:?}"
        );
    }

    // Pointwise read of tile (0, 1, 1) touches only chunk (1, 1) + IFDs and decodes expected pixels.
    counting.reset();
    let chunk_cache = SharedChunkCache::new(budget.decoded_chunk_cache_bytes);
    let tracker = DecodeBudgetTracker::new(budget.decode_working_bytes);
    let ctx = DecodeContext {
        chunk_cache: &chunk_cache,
        budget_tracker: &tracker,
        budget: &budget,
        cancel: &cancel,
    };
    let tile = src
        .decode_pointwise_tile(0, 1, 1, &ctx)
        .expect("decode tile (1,1)");
    assert_eq!(
        (tile.width(), tile.height()),
        (RASTER_TILE_SIZE, RASTER_TILE_SIZE)
    );
    // Pixel (512, 512) in `rgb_pattern` has RGB = (0, 0, (512*512/97) & 0xFF = 142).
    assert_eq!(&tile.data()[..4], &[0, 0, 142, 255]);

    let own = base_tiles[3];
    let read_stats = counting.snapshot();
    assert!(
        read_stats
            .spans
            .iter()
            .any(|s| overlaps((s.offset, s.len as u64), own)),
        "expected read overlapping tile (1, 1)"
    );
    for span in &read_stats.spans {
        let range = (span.offset, span.len as u64);
        assert!(
            overlaps(range, own) || near_head_or_ifd(range, &ifds),
            "tile (1,1) read unrelated range {range:?} (own={own:?})"
        );
    }

    // A second pointwise miss on the SAME IFD reuses the pooled Decoder without
    // re-reading offset 0 (TIFF header) or re-walking IFDs.
    counting.reset();
    let tile2 = src
        .decode_pointwise_tile(0, 0, 0, &ctx)
        .expect("decode tile (0,0)");
    assert_eq!(
        (tile2.width(), tile2.height()),
        (RASTER_TILE_SIZE, RASTER_TILE_SIZE)
    );
    let second_stats = counting.snapshot();
    assert!(
        !second_stats.spans.iter().any(|s| s.offset == 0),
        "pooled Decoder must not re-read TIFF header on same-IFD miss: {:?}",
        second_stats.spans
    );
    // Direct path also tracks `region + tile` working bytes on `DecodeBudgetTracker`.
    assert!(tracker.peak_bytes() > 0);
    assert_eq!(tracker.active_bytes(), 0);
}

#[test]
fn direct_overview_matches_sequential_pass_on_halved_cog() {
    let (file, _, _) = build_tiled_cog(2 * RASTER_TILE_SIZE, 2 * RASTER_TILE_SIZE, 1);
    let fp = fake_fingerprint(&file);
    let budget = MemoryBudget::default();
    let cancel = CancelToken::new();
    let src = TiffSource::open(
        Arc::new(MemoryByteSource::from_vec(file)),
        fp,
        &budget,
        &cancel,
    )
    .expect("open COG");

    let chunk_cache = SharedChunkCache::new(budget.decoded_chunk_cache_bytes);
    let tracker = DecodeBudgetTracker::new(budget.decode_working_bytes);
    let ctx = DecodeContext {
        chunk_cache: &chunk_cache,
        budget_tracker: &tracker,
        budget: &budget,
        cancel: &cancel,
    };
    let direct_tile = src
        .decode_pointwise_tile(1, 0, 0, &ctx)
        .expect("direct overview tile");

    let mut pass_lvl1 = None;
    src.produce_pass(&ctx, &mut |lvl, tx, ty, _w, _h, rgba| {
        if (lvl, tx, ty) == (1, 0, 0) {
            pass_lvl1 = Some(rgba);
        }
        Ok(())
    })
    .expect("sequential pass");
    assert_eq!(
        direct_tile.data(),
        pass_lvl1.expect("pass produced lvl 1").as_slice()
    );

    // Stripped TIFF sequential pass emits every pyramid level and reads file once.
    let stripped = build_stripped_tiff(2 * RASTER_TILE_SIZE, 2 * RASTER_TILE_SIZE, 64);
    let s_len = stripped.len() as u64;
    let s_fp = fake_fingerprint(&stripped);
    let s_counting = Arc::new(CountingByteSource::new(Arc::new(
        MemoryByteSource::from_vec(stripped),
    )));
    let s_src = TiffSource::open(
        Arc::clone(&s_counting) as Arc<dyn ByteSource>,
        s_fp,
        &budget,
        &cancel,
    )
    .expect("open stripped");
    s_counting.reset();
    let mut emitted = Vec::new();
    s_src
        .produce_pass(&ctx, &mut |lvl, tx, ty, w, h, rgba| {
            assert_eq!(rgba.len(), (w * h * 4) as usize);
            emitted.push((lvl, tx, ty));
            Ok(())
        })
        .expect("stripped pass");
    assert_eq!(emitted.len(), 5);
    assert!(
        s_counting.snapshot().bytes_read <= s_len + 2 * WINDOW,
        "sequential pass read {} B for {} B file",
        s_counting.snapshot().bytes_read,
        s_len
    );

    // Non-power-of-two (700x600) raster: direct level-1 output matches sequential pass edge-for-edge.
    let npot = build_stripped_tiff(700, 600, 600);
    let npot_src = TiffSource::open(
        Arc::new(MemoryByteSource::from_vec(npot.clone())),
        fake_fingerprint(&npot),
        &budget,
        &cancel,
    )
    .expect("open NPOT");
    let npot_direct = npot_src
        .decode_pointwise_tile(1, 0, 0, &ctx)
        .expect("direct NPOT lvl 1");
    let mut npot_pass = None;
    npot_src
        .produce_pass(&ctx, &mut |lvl, tx, ty, w, h, rgba| {
            if (lvl, tx, ty) == (1, 0, 0) {
                npot_pass = Some((w, h, rgba));
            }
            Ok(())
        })
        .expect("pass NPOT");
    let (pw, ph, p_rgba) = npot_pass.expect("NPOT pass emitted level 1");
    assert_eq!((pw, ph), (350, 300));
    assert_eq!(npot_direct.data(), p_rgba.as_slice());
}

#[test]
fn decodes_bigtiff_with_geodouble_geoascii_pixel_is_point_and_model_transformation() {
    let budget = MemoryBudget::default();
    let cancel = CancelToken::new();
    let chunk_cache = SharedChunkCache::new(budget.decoded_chunk_cache_bytes);
    let tracker = DecodeBudgetTracker::new(budget.decode_working_bytes);
    let ctx = DecodeContext {
        chunk_cache: &chunk_cache,
        budget_tracker: &tracker,
        budget: &budget,
        cancel: &cancel,
    };

    // BigTIFF + user-defined TM CRS using GeoDoubleParamsTag + GeoAsciiParamsTag +
    // PixelIsPoint + ModelTransformationTag.
    let mut big_buf = Cursor::new(Vec::new());
    {
        let mut enc = TiffEncoder::new_big(&mut big_buf).expect("bigtiff encoder");
        let mut dir = enc.image_directory().expect("bigtiff dir");
        dir.write_tag(Tag::ImageWidth, 8u32).expect("tag");
        dir.write_tag(Tag::ImageLength, 8u32).expect("tag");
        dir.write_tag(Tag::BitsPerSample, &[8u16, 8, 8][..])
            .expect("tag");
        dir.write_tag(Tag::Compression, 1u16).expect("tag");
        dir.write_tag(Tag::PhotometricInterpretation, 2u16)
            .expect("tag");
        dir.write_tag(Tag::SamplesPerPixel, 3u16).expect("tag");
        dir.write_tag(Tag::PlanarConfiguration, 1u16).expect("tag");
        dir.write_tag(Tag::RowsPerStrip, 8u32).expect("tag");
        let data = rgb_pattern(8, 8);
        let at = dir.write_data(&data[..]).expect("write");
        dir.write_tag(Tag::StripOffsets, &[at][..]).expect("tag");
        dir.write_tag(Tag::StripByteCounts, &[data.len() as u64][..])
            .expect("tag");
        let ascii = "VeldMap BigTIFF TM|";
        let geokeys: [u16; 44] = [
            1,
            1,
            0,
            10, // header: 10 keys
            1024,
            0,
            1,
            1, // GTModelTypeGeoKey = Projected
            1025,
            0,
            1,
            2, // GTRasterTypeGeoKey = PixelIsPoint
            2048,
            0,
            1,
            4326, // GeographicTypeGeoKey = WGS 84
            3072,
            0,
            1,
            32767, // ProjectedCSTypeGeoKey = User-Defined
            3073,
            34737,
            ascii.len() as u16,
            0, // PCSCitationGeoKey in GeoAsciiParamsTag
            3075,
            0,
            1,
            1, // ProjCoordTransGeoKey = Transverse Mercator
            3080,
            34736,
            1,
            0, // ProjNatOriginLongGeoKey = 39.0
            3081,
            34736,
            1,
            1, // ProjNatOriginLatGeoKey = 55.75
            3082,
            34736,
            1,
            2, // ProjFalseEastingGeoKey = 500_000.0
            3092,
            34736,
            1,
            3, // ProjScaleAtNatOriginGeoKey = 0.9996
        ];
        dir.write_tag(Tag::GeoKeyDirectoryTag, &geokeys[..])
            .expect("tag");
        dir.write_tag(
            Tag::GeoDoubleParamsTag,
            &[39.0_f64, 55.75, 500_000.0, 0.9996][..],
        )
        .expect("tag");
        dir.write_tag(Tag::GeoAsciiParamsTag, ascii).expect("tag");
        let matrix: [f64; 16] = [
            10.0, 0.0, 0.0, 500_000.0, 0.0, -10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        dir.write_tag(Tag::ModelTransformationTag, &matrix[..])
            .expect("tag");
        dir.finish().expect("finish");
    }
    let big_bytes = big_buf.into_inner();
    assert!(has_tiff_magic(&big_bytes));
    let big_src = TiffSource::open(
        Arc::new(MemoryByteSource::from_vec(big_bytes.clone())),
        fake_fingerprint(&big_bytes),
        &budget,
        &cancel,
    )
    .expect("open BigTIFF");
    let placement = big_src
        .description()
        .placement
        .as_ref()
        .expect("projected placement");
    assert!((placement.affine[2] - 499_995.0).abs() < 1e-9);
    assert!((placement.affine[5] - 5.0).abs() < 1e-9);

    let frame = frame_from_description(big_src.description()).expect("BigTIFF user-defined frame");
    assert_eq!(frame.crs_label(), "VeldMap BigTIFF TM");
    // Pixel (0, 0) center sits at exact natural origin (500_000, 0) -> (55.75, 39.0).
    let (lat, lon) = frame.geodetic(0.5 / 8.0, 0.5 / 8.0);
    assert!((lat - 55.75).abs() < 1e-9, "got lat={lat}");
    assert!((lon - 39.0).abs() < 1e-9, "got lon={lon}");

    let tile = big_src
        .decode_pointwise_tile(0, 0, 0, &ctx)
        .expect("decode BigTIFF tile");
    assert_eq!((tile.width(), tile.height()), (8, 8));
}

#[test]
fn decodes_zstd_predictor3_f32_with_gdal_nodata() {
    let budget = MemoryBudget::default();
    let cancel = CancelToken::new();
    let chunk_cache = SharedChunkCache::new(budget.decoded_chunk_cache_bytes);
    let tracker = DecodeBudgetTracker::new(budget.decode_working_bytes);
    let ctx = DecodeContext {
        chunk_cache: &chunk_cache,
        budget_tracker: &tracker,
        budget: &budget,
        cancel: &cancel,
    };

    let mut f32_buf = Cursor::new(Vec::new());
    let (w, h) = (4u32, 4u32);
    let floats: Vec<f32> = vec![
        -9999.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0,
        140.0, 150.0,
    ];
    let mut predictor3_bytes = Vec::with_capacity((w * h * 4) as usize);
    for row in floats.chunks_exact(w as usize) {
        let mut shuffled = vec![0u8; (w as usize) * 4];
        for (i, &val) in row.iter().enumerate() {
            let be = val.to_be_bytes();
            shuffled[i] = be[0];
            shuffled[w as usize + i] = be[1];
            shuffled[(w as usize) * 2 + i] = be[2];
            shuffled[(w as usize) * 3 + i] = be[3];
        }
        for i in (1..shuffled.len()).rev() {
            shuffled[i] = shuffled[i].wrapping_sub(shuffled[i - 1]);
        }
        predictor3_bytes.extend_from_slice(&shuffled);
    }
    let zstd_payload = zstd::bulk::compress(&predictor3_bytes, 3).expect("zstd compress");
    {
        let mut enc = TiffEncoder::new(&mut f32_buf).expect("enc");
        let mut dir = enc.image_directory().expect("dir");
        dir.write_tag(Tag::ImageWidth, w).expect("tag");
        dir.write_tag(Tag::ImageLength, h).expect("tag");
        dir.write_tag(Tag::BitsPerSample, 32u16).expect("tag");
        dir.write_tag(Tag::SampleFormat, 3u16).expect("tag");
        dir.write_tag(Tag::Compression, 50_000u16).expect("tag");
        dir.write_tag(Tag::Predictor, 3u16).expect("tag");
        dir.write_tag(Tag::PhotometricInterpretation, 1u16)
            .expect("tag");
        dir.write_tag(Tag::SamplesPerPixel, 1u16).expect("tag");
        dir.write_tag(Tag::PlanarConfiguration, 1u16).expect("tag");
        dir.write_tag(Tag::RowsPerStrip, h).expect("tag");
        dir.write_tag(Tag::GdalNodata, "-9999").expect("tag");
        let at = dir.write_data(&zstd_payload[..]).expect("write");
        dir.write_tag(Tag::StripOffsets, &[at as u32][..])
            .expect("tag");
        dir.write_tag(Tag::StripByteCounts, &[zstd_payload.len() as u32][..])
            .expect("tag");
        dir.finish().expect("finish");
    }
    let f32_bytes = f32_buf.into_inner();
    let f32_src = TiffSource::open(
        Arc::new(MemoryByteSource::from_vec(f32_bytes.clone())),
        fake_fingerprint(&f32_bytes),
        &budget,
        &cancel,
    )
    .expect("open ZSTD Predictor=3 Float32 TIFF");
    let f32_tile = f32_src
        .decode_pointwise_tile(0, 0, 0, &ctx)
        .expect("decode ZSTD Predictor=3 Float32");
    assert_eq!(f32_tile.data()[3], 0, "nodata -9999 must be transparent");
    assert_eq!(f32_tile.data()[7], 255, "valid 10.0 pixel must be opaque");
    assert!(f32_tile.data()[4] < f32_tile.data()[8]);
    assert!(f32_tile.data()[8] < f32_tile.data()[15 * 4]);
    assert!(
        f32_tile.data()[15 * 4] >= 245,
        "max 150.0 should stretch near white"
    );
}

#[test]
fn decodes_geographic_pulkovo42_i16_with_gdal_nodata() {
    use veldmap_core::geo::{CrsSpec, PULKOVO_1942};

    let budget = MemoryBudget::default();
    let cancel = CancelToken::new();
    let chunk_cache = SharedChunkCache::new(budget.decoded_chunk_cache_bytes);
    let tracker = DecodeBudgetTracker::new(budget.decode_working_bytes);
    let ctx = DecodeContext {
        chunk_cache: &chunk_cache,
        budget_tracker: &tracker,
        budget: &budget,
        cancel: &cancel,
    };

    let mut i16_buf = Cursor::new(Vec::new());
    {
        let mut enc = TiffEncoder::new(&mut i16_buf).expect("enc");
        let mut dir = enc.image_directory().expect("dir");
        dir.write_tag(Tag::ImageWidth, 4u32).expect("tag");
        dir.write_tag(Tag::ImageLength, 2u32).expect("tag");
        dir.write_tag(Tag::BitsPerSample, 16u16).expect("tag");
        dir.write_tag(Tag::SampleFormat, 2u16).expect("tag");
        dir.write_tag(Tag::Compression, 1u16).expect("tag");
        dir.write_tag(Tag::PhotometricInterpretation, 1u16)
            .expect("tag");
        dir.write_tag(Tag::SamplesPerPixel, 1u16).expect("tag");
        dir.write_tag(Tag::PlanarConfiguration, 1u16).expect("tag");
        dir.write_tag(Tag::RowsPerStrip, 2u32).expect("tag");
        dir.write_tag(Tag::GdalNodata, "-32768").expect("tag");
        let geokeys: [u16; 12] = [1, 1, 0, 2, 1024, 0, 1, 2, 2048, 0, 1, 4284];
        dir.write_tag(Tag::GeoKeyDirectoryTag, &geokeys[..])
            .expect("tag");
        dir.write_tag(Tag::ModelPixelScaleTag, &[0.5_f64, 0.25, 0.0][..])
            .expect("tag");
        dir.write_tag(
            Tag::ModelTiepointTag,
            &[0.0_f64, 0.0, 0.0, 10.0, 50.0, 0.0][..],
        )
        .expect("tag");
        let i16_vals: [i16; 8] = [-32768, -100, 0, 100, 200, 300, 400, 500];
        let raw_bytes: Vec<u8> = i16_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let at = dir.write_data(&raw_bytes[..]).expect("write");
        dir.write_tag(Tag::StripOffsets, &[at as u32][..])
            .expect("tag");
        dir.write_tag(Tag::StripByteCounts, &[raw_bytes.len() as u32][..])
            .expect("tag");
        dir.finish().expect("finish");
    }
    let i16_bytes = i16_buf.into_inner();
    let i16_src = TiffSource::open(
        Arc::new(MemoryByteSource::from_vec(i16_bytes.clone())),
        fake_fingerprint(&i16_bytes),
        &budget,
        &cancel,
    )
    .expect("open geographic i16 TIFF");
    assert!(matches!(
        i16_src.description().geographic_crs,
        Some(CrsSpec::Geographic { datum, .. }) if datum == PULKOVO_1942
    ));
    assert_eq!(i16_src.description().ties.len(), 4);
    assert!((i16_src.description().ties[3].lon - 12.0).abs() < 1e-9);
    assert!((i16_src.description().ties[3].lat - 49.5).abs() < 1e-9);
    let i16_tile = i16_src
        .decode_pointwise_tile(0, 0, 0, &ctx)
        .expect("decode i16");
    assert_eq!(i16_tile.data()[3], 0, "i16 nodata must be transparent");
    assert_eq!(i16_tile.data()[7], 255);
    assert!(i16_tile.data()[4] < i16_tile.data()[12]);
    assert!(i16_tile.data()[12] < i16_tile.data()[7 * 4]);
}

#[test]
fn decodes_gray_alpha_preserving_transparency() {
    let budget = MemoryBudget::default();
    let cancel = CancelToken::new();
    let chunk_cache = SharedChunkCache::new(budget.decoded_chunk_cache_bytes);
    let tracker = DecodeBudgetTracker::new(budget.decode_working_bytes);
    let ctx = DecodeContext {
        chunk_cache: &chunk_cache,
        budget_tracker: &tracker,
        budget: &budget,
        cancel: &cancel,
    };

    let mut ga_buf = Cursor::new(Vec::new());
    {
        let mut enc = TiffEncoder::new(&mut ga_buf).expect("enc");
        let mut dir = enc.image_directory().expect("dir");
        dir.write_tag(Tag::ImageWidth, 2u32).expect("tag");
        dir.write_tag(Tag::ImageLength, 1u32).expect("tag");
        dir.write_tag(Tag::BitsPerSample, &[8u16, 8][..])
            .expect("tag");
        dir.write_tag(Tag::Compression, 1u16).expect("tag");
        dir.write_tag(Tag::PhotometricInterpretation, 1u16)
            .expect("tag");
        dir.write_tag(Tag::SamplesPerPixel, 2u16).expect("tag");
        dir.write_tag(Tag::ExtraSamples, &[2u16][..]).expect("tag");
        dir.write_tag(Tag::PlanarConfiguration, 1u16).expect("tag");
        dir.write_tag(Tag::RowsPerStrip, 1u32).expect("tag");
        let at = dir.write_data(&[200u8, 0, 120, 255][..]).expect("write");
        dir.write_tag(Tag::StripOffsets, &[at as u32][..])
            .expect("tag");
        dir.write_tag(Tag::StripByteCounts, &[4u32][..])
            .expect("tag");
        dir.finish().expect("finish");
    }
    let ga_bytes = ga_buf.into_inner();
    let ga_src = TiffSource::open(
        Arc::new(MemoryByteSource::from_vec(ga_bytes.clone())),
        fake_fingerprint(&ga_bytes),
        &budget,
        &cancel,
    )
    .expect("open Gray+Alpha");
    assert!(ga_src.description().bands[0].has_alpha);
    let ga_tile = ga_src
        .decode_pointwise_tile(0, 0, 0, &ctx)
        .expect("decode Gray+Alpha");
    assert_eq!(
        ga_tile.data()[3],
        0,
        "transparent Gray+Alpha pixel keeps alpha 0"
    );
    assert_eq!(&ga_tile.data()[4..], &[120, 120, 120, 255]);
}

#[test]
fn rejects_planar_configuration_2_at_open() {
    let budget = MemoryBudget::default();
    let cancel = CancelToken::new();

    let mut planar_buf = Cursor::new(Vec::new());
    {
        let mut enc = TiffEncoder::new(&mut planar_buf).expect("enc");
        let mut dir = enc.image_directory().expect("dir");
        dir.write_tag(Tag::ImageWidth, 2u32).expect("tag");
        dir.write_tag(Tag::ImageLength, 2u32).expect("tag");
        dir.write_tag(Tag::BitsPerSample, &[8u16, 8, 8][..])
            .expect("tag");
        dir.write_tag(Tag::Compression, 1u16).expect("tag");
        dir.write_tag(Tag::PhotometricInterpretation, 2u16)
            .expect("tag");
        dir.write_tag(Tag::SamplesPerPixel, 3u16).expect("tag");
        dir.write_tag(Tag::PlanarConfiguration, 2u16).expect("tag");
        dir.write_tag(Tag::RowsPerStrip, 2u32).expect("tag");
        let at = dir.write_data(&[0u8; 12][..]).expect("write");
        dir.write_tag(Tag::StripOffsets, &[at as u32][..])
            .expect("tag");
        dir.write_tag(Tag::StripByteCounts, &[12u32][..])
            .expect("tag");
        dir.finish().expect("finish");
    }
    let planar_bytes = planar_buf.into_inner();
    assert!(
        TiffSource::open(
            Arc::new(MemoryByteSource::from_vec(planar_bytes.clone())),
            fake_fingerprint(&planar_bytes),
            &budget,
            &cancel,
        )
        .is_err()
    );
}

#[test]
fn caps_overview_scanning_at_max_ifds() {
    let budget = MemoryBudget::default();
    let cancel = CancelToken::new();

    let mut many_buf = Cursor::new(Vec::new());
    {
        let mut enc = TiffEncoder::new(&mut many_buf).expect("enc");
        for idx in 0..70u32 {
            let mut dir = enc.image_directory().expect("dir");
            if idx > 0 {
                dir.write_tag(Tag::NewSubfileType, 1u32).expect("subfile");
            }
            dir.write_tag(Tag::ImageWidth, 4u32).expect("w");
            dir.write_tag(Tag::ImageLength, 4u32).expect("h");
            dir.write_tag(Tag::BitsPerSample, 8u16).expect("bps");
            dir.write_tag(Tag::Compression, 1u16).expect("comp");
            dir.write_tag(Tag::PhotometricInterpretation, 1u16)
                .expect("photo");
            dir.write_tag(Tag::SamplesPerPixel, 1u16).expect("spp");
            dir.write_tag(Tag::PlanarConfiguration, 1u16)
                .expect("planar");
            dir.write_tag(Tag::RowsPerStrip, 4u32).expect("rows");
            let at = dir.write_data(&[idx as u8; 16][..]).expect("write");
            dir.write_tag(Tag::StripOffsets, &[at as u32][..])
                .expect("offs");
            dir.write_tag(Tag::StripByteCounts, &[16u32][..])
                .expect("cnts");
            dir.finish().expect("finish");
        }
    }
    let many_bytes = many_buf.into_inner();
    let many_src = TiffSource::open(
        Arc::new(MemoryByteSource::from_vec(many_bytes.clone())),
        fake_fingerprint(&many_bytes),
        &budget,
        &cancel,
    )
    .expect("open 70-IFD TIFF");
    assert_eq!(many_src.overview_count(), MAX_IFDS - 1);
}

#[test]
fn tiled_tiff_above_max_source_side_opens_and_decodes_while_untiled_is_rejected() {
    let budget = MemoryBudget::default();
    let cancel = CancelToken::new();

    // 1. Untiled stripped TIFF exceeding `MAX_SOURCE_SIDE` must still be rejected.
    let mut untiled_buf = Cursor::new(Vec::new());
    {
        let mut enc = TiffEncoder::new(&mut untiled_buf).expect("enc");
        let mut dir = enc.image_directory().expect("dir");
        dir.write_tag(Tag::ImageWidth, MAX_SOURCE_SIDE + 1)
            .expect("w");
        dir.write_tag(Tag::ImageLength, 16u32).expect("h");
        dir.write_tag(Tag::BitsPerSample, 8u16).expect("bps");
        dir.write_tag(Tag::Compression, 1u16).expect("comp");
        dir.write_tag(Tag::PhotometricInterpretation, 1u16)
            .expect("photo");
        dir.write_tag(Tag::SamplesPerPixel, 1u16).expect("spp");
        dir.write_tag(Tag::PlanarConfiguration, 1u16)
            .expect("planar");
        dir.write_tag(Tag::RowsPerStrip, 16u32).expect("rows");
        let at = dir.write_data(&[0u8; 16][..]).expect("write");
        dir.write_tag(Tag::StripOffsets, &[at as u32][..])
            .expect("offs");
        dir.write_tag(Tag::StripByteCounts, &[16u32][..])
            .expect("cnts");
        dir.finish().expect("finish");
    }
    let untiled_bytes = untiled_buf.into_inner();
    let untiled_res = TiffSource::open(
        Arc::new(MemoryByteSource::from_vec(untiled_bytes.clone())),
        fake_fingerprint(&untiled_bytes),
        &budget,
        &cancel,
    );
    assert!(matches!(
        untiled_res,
        Err(RasterError::LimitExceeded { .. })
    ));

    // 2. Tiled BigTIFF/COG exceeding `MAX_SOURCE_SIDE` (70_000 x 1_024) opens
    //    and decodes pointwise cleanly using a shared 512x512 tile buffer.
    let wide = MAX_SOURCE_SIDE + 4_464; // 70_000 px
    let height = 1_024u32;
    let across = wide.div_ceil(RASTER_TILE_SIZE);
    let down = height.div_ceil(RASTER_TILE_SIZE);
    let tile_count = (across * down) as usize;

    let mut tiled_buf = Cursor::new(Vec::new());
    {
        let mut enc = TiffEncoder::new_big(&mut tiled_buf).expect("bigtiff enc");
        let mut dir = enc.image_directory().expect("dir");
        dir.write_tag(Tag::ImageWidth, wide).expect("w");
        dir.write_tag(Tag::ImageLength, height).expect("h");
        dir.write_tag(Tag::BitsPerSample, 8u16).expect("bps");
        dir.write_tag(Tag::Compression, 1u16).expect("comp");
        dir.write_tag(Tag::PhotometricInterpretation, 1u16)
            .expect("photo");
        dir.write_tag(Tag::SamplesPerPixel, 1u16).expect("spp");
        dir.write_tag(Tag::PlanarConfiguration, 1u16)
            .expect("planar");
        dir.write_tag(Tag::TileWidth, RASTER_TILE_SIZE).expect("tw");
        dir.write_tag(Tag::TileLength, RASTER_TILE_SIZE)
            .expect("th");
        let chunk = vec![173u8; (RASTER_TILE_SIZE * RASTER_TILE_SIZE) as usize];
        let at = dir.write_data(&chunk[..]).expect("write shared tile");
        let offsets = vec![at; tile_count];
        let counts = vec![chunk.len() as u64; tile_count];
        dir.write_tag(Tag::TileOffsets, &offsets[..]).expect("offs");
        dir.write_tag(Tag::TileByteCounts, &counts[..])
            .expect("cnts");
        dir.finish().expect("finish");
    }
    let tiled_bytes = tiled_buf.into_inner();
    let tiled_src = TiffSource::open(
        Arc::new(MemoryByteSource::from_vec(tiled_bytes.clone())),
        fake_fingerprint(&tiled_bytes),
        &budget,
        &cancel,
    )
    .expect("wide tiled BigTIFF opens");
    assert_eq!(tiled_src.description().width, wide);

    let chunk_cache = SharedChunkCache::new(budget.decoded_chunk_cache_bytes);
    let tracker = DecodeBudgetTracker::new(budget.decode_working_bytes);
    let ctx = DecodeContext {
        chunk_cache: &chunk_cache,
        budget_tracker: &tracker,
        budget: &budget,
        cancel: &cancel,
    };
    let tile = tiled_src
        .decode_pointwise_tile(0, across - 1, 0, &ctx)
        .expect("decode far-right edge tile");
    assert_eq!(
        (tile.width(), tile.height()),
        (tile_extent(across - 1, wide), RASTER_TILE_SIZE)
    );
    assert_eq!(&tile.data()[..4], &[173, 173, 173, 255]);
}

#[test]
fn waiting_for_display_mapping_wakes_immediately_on_caller_cancel() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::thread;
    use std::time::Duration;
    use veldmap_core::io::IoError;

    struct BlockingTailSource {
        inner: MemoryByteSource,
        threshold: u64,
        armed: AtomicBool,
        hit: AtomicBool,
        release: CancelToken,
    }

    impl ByteSource for BlockingTailSource {
        fn len(&self) -> u64 {
            self.inner.len()
        }

        fn read_at(
            &self,
            offset: u64,
            buf: &mut [u8],
            cancel: &CancelToken,
        ) -> Result<usize, IoError> {
            if self.armed.load(Ordering::Acquire)
                && offset.saturating_add(buf.len() as u64) > self.threshold
            {
                self.hit.store(true, Ordering::Release);
                while !self.release.is_cancelled() && !cancel.is_cancelled() {
                    thread::sleep(Duration::from_millis(5));
                }
            }
            self.inner.read_at(offset, buf, cancel)
        }
    }

    let mut buf = Cursor::new(Vec::new());
    let tile_offset: u64;
    {
        let mut enc = TiffEncoder::new(&mut buf).expect("enc");
        let mut dir = enc.image_directory().expect("dir");
        dir.write_tag(Tag::ImageWidth, 256u32).expect("w");
        dir.write_tag(Tag::ImageLength, 256u32).expect("h");
        dir.write_tag(Tag::BitsPerSample, 16u16).expect("bps");
        dir.write_tag(Tag::Compression, 1u16).expect("comp");
        dir.write_tag(Tag::PhotometricInterpretation, 1u16)
            .expect("photo");
        dir.write_tag(Tag::SamplesPerPixel, 1u16).expect("spp");
        dir.write_tag(Tag::PlanarConfiguration, 1u16)
            .expect("planar");
        dir.write_tag(Tag::TileWidth, 256u32).expect("tw");
        dir.write_tag(Tag::TileLength, 256u32).expect("th");
        let samples: Vec<u16> = (0..65_536u32).map(|v| (v % 4096) as u16).collect();
        tile_offset = dir.write_data(&samples[..]).expect("write");
        dir.write_tag(Tag::TileOffsets, &[tile_offset as u32][..])
            .expect("offs");
        dir.write_tag(Tag::TileByteCounts, &[(65_536u32 * 2)][..])
            .expect("cnts");
        dir.finish().expect("finish");
    }
    let raw = buf.into_inner();
    let fp = fake_fingerprint(&raw);
    let release = CancelToken::new();
    let gate = Arc::new(BlockingTailSource {
        inner: MemoryByteSource::from_vec(raw),
        threshold: 64 * 1024,
        armed: AtomicBool::new(false),
        hit: AtomicBool::new(false),
        release: release.clone(),
    });
    let budget = MemoryBudget::default();
    let src = Arc::new(
        TiffSource::open(
            Arc::clone(&gate) as Arc<dyn ByteSource>,
            fp,
            &budget,
            &CancelToken::new(),
        )
        .expect("open 16-bit"),
    );
    gate.armed.store(true, Ordering::Release);

    let chunk_cache = Arc::new(SharedChunkCache::new(budget.decoded_chunk_cache_bytes));
    let tracker = Arc::new(DecodeBudgetTracker::new(budget.decode_working_bytes));

    let leader = {
        let src = Arc::clone(&src);
        let cache = Arc::clone(&chunk_cache);
        let tracker = Arc::clone(&tracker);
        let b = budget;
        thread::spawn(move || {
            let cancel = CancelToken::new();
            let ctx = DecodeContext {
                chunk_cache: &cache,
                budget_tracker: &tracker,
                budget: &b,
                cancel: &cancel,
            };
            src.decode_pointwise_tile(0, 0, 0, &ctx)
        })
    };

    while !gate.hit.load(Ordering::Acquire) {
        thread::sleep(Duration::from_millis(2));
    }

    let waiter_cancel = CancelToken::new();
    let waiter = {
        let src = Arc::clone(&src);
        let cache = Arc::clone(&chunk_cache);
        let tracker = Arc::clone(&tracker);
        let c = waiter_cancel.clone();
        let b = budget;
        thread::spawn(move || {
            let ctx = DecodeContext {
                chunk_cache: &cache,
                budget_tracker: &tracker,
                budget: &b,
                cancel: &c,
            };
            src.decode_pointwise_tile(0, 0, 0, &ctx)
        })
    };
    thread::sleep(Duration::from_millis(15));
    waiter_cancel.cancel();
    let waiter_res = waiter.join().expect("join waiter");
    assert!(matches!(waiter_res, Err(RasterError::Cancelled)));

    release.cancel();
    assert!(leader.join().expect("join leader").is_ok());
}

#[test]
fn pixel_is_area_vs_pixel_is_point_shifts_warp_mesh_and_readout_by_half_pixel() {
    use veldmap_core::geo::geodesy::{Geodetic, WGS84};
    use veldmap_core::geo::projection::Projection;
    use veldmap_core::tile::{MIN_WARP_SEGMENTS, build_warp_patch};

    let build_geotiff = |raster_type_key: u16| -> Vec<u8> {
        let mut buf = Cursor::new(Vec::new());
        {
            let mut enc = TiffEncoder::new(&mut buf).expect("tiff encoder");
            let mut dir = enc.image_directory().expect("dir");
            let (w, h) = (64u32, 64u32);
            dir.write_tag(Tag::ImageWidth, w).expect("tag");
            dir.write_tag(Tag::ImageLength, h).expect("tag");
            dir.write_tag(Tag::BitsPerSample, &[8u16, 8, 8][..])
                .expect("tag");
            dir.write_tag(Tag::Compression, 1u16).expect("tag");
            dir.write_tag(Tag::PhotometricInterpretation, 2u16)
                .expect("tag");
            dir.write_tag(Tag::SamplesPerPixel, 3u16).expect("tag");
            dir.write_tag(Tag::PlanarConfiguration, 1u16).expect("tag");
            dir.write_tag(Tag::RowsPerStrip, h).expect("tag");
            let data = rgb_pattern(w, h);
            let at = dir.write_data(&data[..]).expect("write");
            dir.write_tag(Tag::StripOffsets, &[at as u32][..])
                .expect("tag");
            dir.write_tag(Tag::StripByteCounts, &[data.len() as u32][..])
                .expect("tag");
            let geokeys: [u16; 16] = [
                1,
                1,
                0,
                3, // header: 3 keys
                1024,
                0,
                1,
                1, // GTModelTypeGeoKey = Projected
                1025,
                0,
                1,
                raster_type_key, // GTRasterTypeGeoKey
                3072,
                0,
                1,
                32637, // ProjectedCSTypeGeoKey = EPSG:32637
            ];
            dir.write_tag(Tag::GeoKeyDirectoryTag, &geokeys[..])
                .expect("tag");
            // Pixel scale = 10 m/px; tiepoint pixel (0, 0) at (550_000, 5_250_000).
            dir.write_tag(Tag::ModelPixelScaleTag, &[10.0_f64, 10.0, 0.0][..])
                .expect("tag");
            dir.write_tag(
                Tag::ModelTiepointTag,
                &[0.0_f64, 0.0, 0.0, 550_000.0, 5_250_000.0, 0.0][..],
            )
            .expect("tag");
            dir.finish().expect("finish");
        }
        buf.into_inner()
    };

    let budget = MemoryBudget::default();
    let cancel = CancelToken::new();
    let chunk_cache = SharedChunkCache::new(budget.decoded_chunk_cache_bytes);
    let tracker = DecodeBudgetTracker::new(budget.decode_working_bytes);
    let ctx = DecodeContext {
        chunk_cache: &chunk_cache,
        budget_tracker: &tracker,
        budget: &budget,
        cancel: &cancel,
    };

    let bytes_area = build_geotiff(1);
    let bytes_point = build_geotiff(2);
    let src_area: Arc<dyn RasterSource> = Arc::new(
        TiffSource::open(
            Arc::new(MemoryByteSource::new(Arc::from(bytes_area.as_slice()))),
            fake_fingerprint(&bytes_area),
            &budget,
            &cancel,
        )
        .expect("open PixelIsArea"),
    );
    let src_point: Arc<dyn RasterSource> = Arc::new(
        TiffSource::open(
            Arc::new(MemoryByteSource::new(Arc::from(bytes_point.as_slice()))),
            fake_fingerprint(&bytes_point),
            &budget,
            &cancel,
        )
        .expect("open PixelIsPoint"),
    );

    let frame_area = frame_from_description(src_area.description()).expect("area frame");
    let frame_point = frame_from_description(src_point.description()).expect("point frame");

    let patch_area = build_warp_patch(
        &frame_area,
        64,
        64,
        [0.0, 0.0, 64.0, 64.0],
        MIN_WARP_SEGMENTS,
    );
    let patch_point = build_warp_patch(
        &frame_point,
        64,
        64,
        [0.0, 0.0, 64.0, 64.0],
        MIN_WARP_SEGMENTS,
    );

    let utm37 = Projection::from_epsg(32637).expect("UTM 37N");
    let geo_area = WGS84.from_ecef(patch_area.evaluate_ecef(0.0, 0.0).expect("area ul"));
    let geo_point = WGS84.from_ecef(patch_point.evaluate_ecef(0.0, 0.0).expect("point ul"));
    let (ea, na) = utm37.from_geodetic(geo_area.lat_deg, geo_area.lon_deg);
    let (ep, np) = utm37.from_geodetic(geo_point.lat_deg, geo_point.lon_deg);

    // PixelIsArea UL corner lands at (550_000, 5_250_000).
    assert!((ea - 550_000.0).abs() < 0.002, "area easting={ea}");
    assert!((na - 5_250_000.0).abs() < 0.002, "area northing={na}");
    // PixelIsPoint shifts UL corner by exact (-0.5 px, +0.5 px) = (-5.0 m, +5.0 m).
    assert!((ep - 549_995.0).abs() < 0.002, "point easting={ep}");
    assert!((np - 5_250_005.0).abs() < 0.002, "point northing={np}");
    let (pt_lat, pt_lon) = utm37.to_geodetic(549_995.0, 5_250_005.0);
    let exact_pt_ul = WGS84.to_ecef(Geodetic::surface(pt_lat, pt_lon));
    let drawn_pt_ul = patch_point.evaluate_ecef(0.0, 0.0).expect("point ul");
    assert!(exact_pt_ul.distance(drawn_pt_ul) < 0.002);

    // Zoom-invariant native base-IFD display readout matches synthetic RGB pattern.
    let (px, py) = (17u32, 23u32);
    let expected = [
        (px * 7 + py * 3) as u8,
        (px ^ py) as u8,
        ((px * py) / 97) as u8,
        255,
    ];
    assert_eq!(
        src_area
            .sample_display_rgba(px, py, &ctx)
            .expect("sample area"),
        Some(expected)
    );
    assert_eq!(
        src_point
            .sample_display_rgba(px, py, &ctx)
            .expect("sample point"),
        Some(expected)
    );
    assert_eq!(
        src_area
            .sample_display_rgba(64, 0, &ctx)
            .expect("out of bounds"),
        None
    );

    // Multi-chunk tiled COG readout: sample pixels in non-zero (cx, cy) tiles
    // and verify the large-chunk guard rejects oversized single-chunk readouts.
    let (cog_bytes, _, _) = build_tiled_cog(700, 600, 1);
    let cog_src = TiffSource::open(
        Arc::new(MemoryByteSource::new(Arc::from(cog_bytes.as_slice()))),
        fake_fingerprint(&cog_bytes),
        &budget,
        &cancel,
    )
    .expect("open multi-chunk cog");
    let (cpx, cpy) = (610u32, 540u32);
    let cog_expected = [
        (cpx * 7 + cpy * 3) as u8,
        (cpx ^ cpy) as u8,
        ((cpx * cpy) / 97) as u8,
        255,
    ];
    assert_eq!(
        cog_src
            .sample_display_rgba(cpx, cpy, &ctx)
            .expect("sample multi-chunk cog"),
        Some(cog_expected)
    );

    // Tight chunk cache budget triggers LimitExceeded before decoding on fast_pool.
    let tiny_budget = MemoryBudget {
        decoded_chunk_cache_bytes: 16 * 1024,
        ..budget
    };
    let tiny_ctx = DecodeContext {
        chunk_cache: &chunk_cache,
        budget_tracker: &tracker,
        budget: &tiny_budget,
        cancel: &cancel,
    };
    assert!(matches!(
        cog_src.sample_display_rgba(cpx, cpy, &tiny_ctx),
        Err(RasterError::LimitExceeded { .. })
    ));
}
