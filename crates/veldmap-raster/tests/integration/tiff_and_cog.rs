//! Integration tests for COG I/O amplification, pooled decoder reuse,
//! overview-vs-pass equivalence, TIFF compression/sample formats (ZSTD +
//! Predictor=3 Float32, i16 + GDAL nodata, Gray+Alpha, PlanarConfiguration=2),
//! and IFD/dimension limit guards.

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
    TiffSource,
};

use crate::common::{
    TiffIfdHeader, build_stripped_tiff, build_tiled_cog, fake_fingerprint, write_ifd_header,
    write_single_strip_u32,
};

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
        write_ifd_header(
            &mut dir,
            &TiffIfdHeader {
                bits_per_sample: &[32],
                compression: 50_000,
                sample_format: Some(3),
                ..TiffIfdHeader::gray8(w, h)
            },
        );
        dir.write_tag(Tag::Predictor, 3u16).expect("tag");
        dir.write_tag(Tag::GdalNodata, "-9999").expect("tag");
        write_single_strip_u32(&mut dir, h, &zstd_payload);
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
        write_ifd_header(
            &mut dir,
            &TiffIfdHeader {
                bits_per_sample: &[16],
                sample_format: Some(2),
                ..TiffIfdHeader::gray8(4, 2)
            },
        );
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
        write_single_strip_u32(&mut dir, 2, &raw_bytes);
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
        write_ifd_header(
            &mut dir,
            &TiffIfdHeader {
                bits_per_sample: &[8, 8],
                samples_per_pixel: 2,
                ..TiffIfdHeader::gray8(2, 1)
            },
        );
        dir.write_tag(Tag::ExtraSamples, &[2u16][..]).expect("tag");
        write_single_strip_u32(&mut dir, 1, &[200u8, 0, 120, 255]);
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
        write_ifd_header(
            &mut dir,
            &TiffIfdHeader {
                planar_configuration: 2,
                ..TiffIfdHeader::rgb8(2, 2)
            },
        );
        write_single_strip_u32(&mut dir, 2, &[0u8; 12]);
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
            write_ifd_header(
                &mut dir,
                &TiffIfdHeader {
                    is_overview: idx > 0,
                    ..TiffIfdHeader::gray8(4, 4)
                },
            );
            write_single_strip_u32(&mut dir, 4, &[idx as u8; 16]);
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
        write_ifd_header(&mut dir, &TiffIfdHeader::gray8(MAX_SOURCE_SIDE + 1, 16));
        write_single_strip_u32(&mut dir, 16, &[0u8; 16]);
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
        write_ifd_header(&mut dir, &TiffIfdHeader::gray8(wide, height));
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
