//! Integration tests for COG I/O amplification, stateless cached-IFD chunk
//! reading, overview-vs-pass equivalence, TIFF compression/sample formats
//! (ZSTD + Predictor=3 Float32, JPEG, i16/i8 + GDAL nodata, WhiteIsZero,
//! Gray+Alpha, PlanarConfiguration=2, Palette, sparse chunks), and IFD/dimension
//! limit guards.

use std::io::Cursor;
use std::sync::Arc;

use ::tiff::encoder::TiffEncoder;
use ::tiff::tags::Tag;
use veldmap_core::budget::{DecodeBudgetTracker, MemoryBudget};
use veldmap_core::io::ByteSource;
use veldmap_core::io::testing::{CountingByteSource, MemoryByteSource};
use veldmap_core::sync::CancelToken;
use veldmap_core::tile::{BandKey, PixelFormat, RASTER_TILE_SIZE, tile_extent};
use veldmap_raster::{
    BandSample, DecodeContext, MAX_IFDS, MAX_SOURCE_SIDE, RasterError, RasterSource,
    SharedChunkCache, TiffSource, generate_scientific_mipmaps, stream_spec_for_band,
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
fn describing_reads_only_header_and_ifds_and_pointwise_avoids_rereading_ifds() {
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
fn rejects_malformed_planar_configuration_2_with_missing_plane_offsets() {
    let budget = MemoryBudget::default();
    let cancel = CancelToken::new();

    // PlanarConfiguration = 2 with 3 samples/pixel requires 3 strip offsets, not 1.
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

#[test]
fn decodes_planar2_npot_tiled_rgb_and_single_band_scientific_and_palette_and_rgbn() {
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

    // 1. Tiled PlanarConfiguration = 2 with NPOT dimensions (20x18, tile 16x16):
    //    2x2 = 4 tiles per plane * 3 planes = 12 tile offsets.
    let (pw, ph) = (20u32, 18u32);
    let (tw, th) = (16u32, 16u32);
    let mut planar_buf = Cursor::new(Vec::new());
    {
        let mut enc = TiffEncoder::new(&mut planar_buf).expect("enc");
        let mut dir = enc.image_directory().expect("dir");
        write_ifd_header(
            &mut dir,
            &TiffIfdHeader {
                planar_configuration: 2,
                ..TiffIfdHeader::rgb8(pw, ph)
            },
        );
        dir.write_tag(Tag::TileWidth, tw).expect("tw");
        dir.write_tag(Tag::TileLength, th).expect("th");
        let plane_vals = [10u8, 120u8, 240u8];
        let mut offsets = Vec::with_capacity(12);
        let mut counts = Vec::with_capacity(12);
        for &val in &plane_vals {
            for tile_idx in 0..4u32 {
                // Bottom-edge tiles intentionally truncated to only `dh = 2` stored rows
                // (testing NPOT bottom-edge planar tolerance!).
                let rows = if tile_idx >= 2 { 2usize } else { th as usize };
                let payload = vec![val; (tw as usize) * rows];
                let off = dir.write_data(&payload[..]).expect("tile data") as u32;
                offsets.push(off);
                counts.push(payload.len() as u32);
            }
        }
        dir.write_tag(Tag::TileOffsets, &offsets[..]).expect("offs");
        dir.write_tag(Tag::TileByteCounts, &counts[..])
            .expect("cnts");
        dir.finish().expect("finish");
    }
    let planar_bytes = planar_buf.into_inner();
    let planar_src = TiffSource::open(
        Arc::new(MemoryByteSource::from_vec(planar_bytes.clone())),
        fake_fingerprint(&planar_bytes),
        &budget,
        &cancel,
    )
    .expect("open tiled Planar-2 NPOT TIFF");
    let display_tile = planar_src
        .decode_pointwise_tile(0, 0, 0, &ctx)
        .expect("decode Planar-2 Display");
    assert_eq!((display_tile.width(), display_tile.height()), (pw, ph));
    // Verify bottom-right NPOT corner pixel (19, 17).
    let last_px = ((17 * pw + 19) * 4) as usize;
    assert_eq!(
        &display_tile.data()[last_px..last_px + 4],
        &[10, 120, 240, 255]
    );

    // Single-band Scientific R32Float read of plane 2 (value 240.0).
    let sci_spec = stream_spec_for_band(BandKey::Index(2), PixelFormat::R32Float);
    let mut sci_tile = planar_src
        .decode_pointwise_tile_stream(sci_spec, 0, 0, 0, &ctx)
        .expect("decode Planar-2 Band 2 R32Float");
    generate_scientific_mipmaps(&mut sci_tile);
    assert!(sci_tile.mip_level_count() > 1);
    let f0 = f32::from_ne_bytes(sci_tile.data()[0..4].try_into().expect("4B"));
    assert_eq!(f0, 240.0);

    // 2. Palette TIFF (Photometric = 3, ColorMap tag 320):
    //    Expands via ColorMap on Display AND exposes raw R8Uint class indices on Categorical.
    let mut pal_buf = Cursor::new(Vec::new());
    {
        let mut enc = TiffEncoder::new(&mut pal_buf).expect("enc");
        let mut dir = enc.image_directory().expect("dir");
        write_ifd_header(
            &mut dir,
            &TiffIfdHeader {
                photometric: 3,
                ..TiffIfdHeader::gray8(4, 2)
            },
        );
        let mut cmap = vec![0u16; 256 * 3];
        // Class 1 -> pure red (255, 0, 0), Class 2 -> pure green (0, 255, 0).
        cmap[1] = 65535;
        cmap[256 + 2] = 65535;
        dir.write_tag(Tag::ColorMap, &cmap[..]).expect("cmap");
        dir.write_tag(Tag::GdalNodata, "0").expect("nodata");
        write_single_strip_u32(&mut dir, 2, &[0, 1, 2, 1, 2, 1, 0, 2]);
        dir.finish().expect("finish");
    }
    let pal_bytes = pal_buf.into_inner();
    let pal_src = TiffSource::open(
        Arc::new(MemoryByteSource::from_vec(pal_bytes.clone())),
        fake_fingerprint(&pal_bytes),
        &budget,
        &cancel,
    )
    .expect("open Palette TIFF");
    assert_eq!(pal_src.default_stream_spec().format, PixelFormat::R8Uint);
    assert!(pal_src.color_map().is_some());
    let pal_disp = pal_src
        .decode_pointwise_tile(0, 0, 0, &ctx)
        .expect("decode Palette Display");
    assert_eq!(
        pal_disp.data()[3],
        0,
        "nodata 0 must stay transparent (alpha=0)"
    );
    assert_eq!(&pal_disp.data()[4..8], &[255, 0, 0, 255]); // class 1 red
    assert_eq!(&pal_disp.data()[8..12], &[0, 255, 0, 255]); // class 2 green

    let pal_cat = pal_src
        .decode_pointwise_tile_stream(pal_src.default_stream_spec(), 0, 0, 0, &ctx)
        .expect("decode Palette Categorical");
    assert_eq!(pal_cat.format(), PixelFormat::R8Uint);
    assert_eq!(pal_cat.data(), &[0, 1, 2, 1, 2, 1, 0, 2]);
    assert_eq!(
        pal_src
            .sample_band(BandKey::Index(0), PixelFormat::R8Uint, 2, 0, &ctx)
            .expect("sample palette"),
        BandSample::Categorical(2)
    );

    // 3. 4-band RGB+NIR with ExtraSamples = [0] (Unspecified) + GDAL_METADATA scale/offset:
    //    NIR band (band 3) must NOT be treated as alpha, Display RGB stays opaque,
    //    and band 3 decodes cleanly as R32Float with physical stats and readout.
    let mut rgbn_buf = Cursor::new(Vec::new());
    {
        let mut enc = TiffEncoder::new(&mut rgbn_buf).expect("enc");
        let mut dir = enc.image_directory().expect("dir");
        write_ifd_header(
            &mut dir,
            &TiffIfdHeader {
                photometric: 2,
                samples_per_pixel: 4,
                bits_per_sample: &[16, 16, 16, 16],
                sample_format: Some(1),
                ..TiffIfdHeader::gray8(2, 2)
            },
        );
        dir.write_tag(Tag::ExtraSamples, &[0u16][..])
            .expect("extra");
        dir.write_tag(
            Tag::Unknown(42112),
            r#"<GDALMetadata><Item name="SCALE" sample="3">0.0001</Item><Item name="OFFSET" sample="3">-0.1</Item><Item name="UNITTYPE" sample="3">reflectance</Item></GDALMetadata>"#,
        )
        .expect("gdal meta");
        // Band 3 (NIR) is 0 on pixel 0 — if wrongly treated as alpha, pixel 0 would vanish!
        let pixels_u16: [u16; 16] = [
            1000, 2000, 3000, 0, // px 0: NIR = 0
            4000, 5000, 6000, 5000, // px 1: NIR = 5000 -> phys = 0.4
            7000, 8000, 9000, 8000, // px 2: NIR = 8000 -> phys = 0.7
            10000, 11000, 12000, 10000, // px 3: NIR = 10000 -> phys = 0.9
        ];
        let raw: Vec<u8> = pixels_u16.iter().flat_map(|v| v.to_le_bytes()).collect();
        write_single_strip_u32(&mut dir, 2, &raw);
        dir.finish().expect("finish");
    }
    let rgbn_bytes = rgbn_buf.into_inner();
    let rgbn_src = TiffSource::open(
        Arc::new(MemoryByteSource::from_vec(rgbn_bytes.clone())),
        fake_fingerprint(&rgbn_bytes),
        &budget,
        &cancel,
    )
    .expect("open 4-band RGBN TIFF");
    assert!(!rgbn_src.description().bands[0].has_alpha);
    assert_eq!(rgbn_src.description().indexed_bands().count(), 4);
    let rgbn_disp = rgbn_src
        .decode_pointwise_tile(0, 0, 0, &ctx)
        .expect("decode RGBN Display");
    assert_eq!(
        rgbn_disp.data()[3],
        255,
        "ExtraSamples=0 NIR=0 must stay opaque on Display"
    );
    let nir_spec = stream_spec_for_band(BandKey::Index(3), PixelFormat::R32Float);
    let nir_tile = rgbn_src
        .decode_pointwise_tile_stream(nir_spec, 0, 0, 0, &ctx)
        .expect("decode NIR R32Float");
    let nir_vals: Vec<f32> = nir_tile
        .data()
        .chunks_exact(4)
        .map(|b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    assert_eq!(nir_vals, vec![0.0, 5000.0, 8000.0, 10000.0]);
    let nir_sample = rgbn_src
        .sample_band(BandKey::Index(3), PixelFormat::R32Float, 1, 0, &ctx)
        .expect("sample NIR");
    match nir_sample {
        BandSample::Scalar { raw, physical } => {
            assert_eq!(raw, 5000.0);
            assert!((physical - 0.4).abs() < 1e-9);
        }
        other => panic!("expected Scalar, got {other:?}"),
    }
    let stats = rgbn_src
        .band_stats(BandKey::Index(3), &ctx)
        .expect("band stats")
        .expect("non-empty stats");
    assert!((stats.min - (-0.1)).abs() < 1e-9);
    assert!((stats.max - 0.9).abs() < 1e-9);
    assert_eq!(stats.sample_count, 4);
}

#[test]
fn direct_raw_decoder_handles_deflate_lzw_packbits_predictor2_big_endian_and_scientific_pass() {
    use std::io::Write;
    use veldmap_raster::TileEmit;

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

    // 1. Deflate + Horizontal Predictor (Predictor = 2) 16-bit single-band TIFF.
    let src_u16: [u16; 8] = [100, 150, 225, 400, 1000, 900, 850, 1200];
    let mut diff_u16 = Vec::with_capacity(16);
    for row in src_u16.chunks_exact(4) {
        let mut prev = 0u16;
        for &v in row {
            let d = v.wrapping_sub(prev);
            prev = v;
            diff_u16.extend_from_slice(&d.to_le_bytes());
        }
    }
    let mut zenc = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
    zenc.write_all(&diff_u16).expect("zlib write");
    let deflated = zenc.finish().expect("zlib finish");
    let mut def_buf = Cursor::new(Vec::new());
    {
        let mut enc = TiffEncoder::new(&mut def_buf).expect("enc");
        let mut dir = enc.image_directory().expect("dir");
        write_ifd_header(
            &mut dir,
            &TiffIfdHeader {
                bits_per_sample: &[16],
                sample_format: Some(1),
                compression: 8,
                ..TiffIfdHeader::gray8(4, 2)
            },
        );
        dir.write_tag(Tag::Predictor, 2u16).expect("predictor");
        write_single_strip_u32(&mut dir, 2, &deflated);
        dir.finish().expect("finish");
    }
    let def_bytes = def_buf.into_inner();
    let def_src = TiffSource::open(
        Arc::new(MemoryByteSource::from_vec(def_bytes.clone())),
        fake_fingerprint(&def_bytes),
        &budget,
        &cancel,
    )
    .expect("open Deflate+Predictor2 u16 TIFF");
    let def_tile = def_src
        .decode_pointwise_tile_stream(
            stream_spec_for_band(BandKey::Index(0), PixelFormat::R32Float),
            0,
            0,
            0,
            &ctx,
        )
        .expect("decode Deflate+Predictor2");
    let decoded_f32: Vec<f32> = def_tile
        .data()
        .chunks_exact(4)
        .map(|b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let expected_f32: Vec<f32> = src_u16.iter().map(|&v| f32::from(v)).collect();
    assert_eq!(decoded_f32, expected_f32);

    // 2. LZW (compression = 5) + PackBits (compression = 32773) 8-bit grayscale TIFFs.
    let raw_lzw = [10u8, 20, 30, 40, 40, 40, 50, 60];
    let mut lzw_enc = weezl::encode::Encoder::with_tiff_size_switch(weezl::BitOrder::Msb, 8);
    let lzw_payload = lzw_enc.encode(&raw_lzw).expect("lzw encode");
    let mut lzw_buf = Cursor::new(Vec::new());
    {
        let mut enc = TiffEncoder::new(&mut lzw_buf).expect("enc");
        let mut dir = enc.image_directory().expect("dir");
        write_ifd_header(
            &mut dir,
            &TiffIfdHeader {
                compression: 5,
                ..TiffIfdHeader::gray8(4, 2)
            },
        );
        write_single_strip_u32(&mut dir, 2, &lzw_payload);
        dir.finish().expect("finish");
    }
    let lzw_bytes = lzw_buf.into_inner();
    let lzw_src = TiffSource::open(
        Arc::new(MemoryByteSource::from_vec(lzw_bytes.clone())),
        fake_fingerprint(&lzw_bytes),
        &budget,
        &cancel,
    )
    .expect("open LZW TIFF");
    let lzw_tile = lzw_src
        .decode_pointwise_tile_stream(
            stream_spec_for_band(BandKey::Index(0), PixelFormat::R8Uint),
            0,
            0,
            0,
            &ctx,
        )
        .expect("decode LZW");
    assert_eq!(lzw_tile.data(), &raw_lzw);

    // PackBits: literal run of 3 bytes (header=2: 10, 20, 30) + repeat 40 x 3 (header=-2 = 0xFE, 40)
    //           + literal run of 2 bytes (header=1: 50, 60).
    let packbits_payload = [2u8, 10, 20, 30, 0xFE, 40, 1, 50, 60];
    let mut pb_buf = Cursor::new(Vec::new());
    {
        let mut enc = TiffEncoder::new(&mut pb_buf).expect("enc");
        let mut dir = enc.image_directory().expect("dir");
        write_ifd_header(
            &mut dir,
            &TiffIfdHeader {
                compression: 32773,
                ..TiffIfdHeader::gray8(4, 2)
            },
        );
        write_single_strip_u32(&mut dir, 2, &packbits_payload);
        dir.finish().expect("finish");
    }
    let pb_bytes = pb_buf.into_inner();
    let pb_src = TiffSource::open(
        Arc::new(MemoryByteSource::from_vec(pb_bytes.clone())),
        fake_fingerprint(&pb_bytes),
        &budget,
        &cancel,
    )
    .expect("open PackBits TIFF");
    let pb_tile = pb_src
        .decode_pointwise_tile_stream(
            stream_spec_for_band(BandKey::Index(0), PixelFormat::R8Uint),
            0,
            0,
            0,
            &ctx,
        )
        .expect("decode PackBits");
    assert_eq!(pb_tile.data(), &raw_lzw);

    // 3. Big-Endian ("MM") multi-byte Float32 TIFF + `produce_pass_stream`
    //    verifying exact `f32::NAN` preservation (confirming `bleed_alpha` never runs on Scientific).
    let be_vals: [f32; 4] = [-9999.0, 12.5, 42.25, f32::NAN];
    let mut be_raw = Vec::with_capacity(16);
    for v in be_vals {
        be_raw.extend_from_slice(&v.to_be_bytes());
    }
    // Hand-build a minimal Big-Endian classic TIFF ("MM", version 42) with 1 IFD & 10 tags.
    let mut be_tiff = Vec::new();
    be_tiff.extend_from_slice(b"MM");
    be_tiff.extend_from_slice(&42u16.to_be_bytes());
    be_tiff.extend_from_slice(&8u32.to_be_bytes()); // IFD at offset 8
    let num_entries: u16 = 10;
    be_tiff.extend_from_slice(&num_entries.to_be_bytes());
    let data_offset: u32 = 8 + 2 + u32::from(num_entries) * 12 + 4;
    let nodata_str = b"-9999\0";
    let nodata_offset = data_offset + be_raw.len() as u32;
    let push_be_short = |buf: &mut Vec<u8>, tag: u16, val: u16| {
        buf.extend_from_slice(&tag.to_be_bytes());
        buf.extend_from_slice(&3u16.to_be_bytes()); // SHORT
        buf.extend_from_slice(&1u32.to_be_bytes());
        buf.extend_from_slice(&val.to_be_bytes());
        buf.extend_from_slice(&0u16.to_be_bytes());
    };
    let push_be_long = |buf: &mut Vec<u8>, tag: u16, val: u32| {
        buf.extend_from_slice(&tag.to_be_bytes());
        buf.extend_from_slice(&4u16.to_be_bytes()); // LONG
        buf.extend_from_slice(&1u32.to_be_bytes());
        buf.extend_from_slice(&val.to_be_bytes());
    };
    push_be_long(&mut be_tiff, 256, 2); // ImageWidth
    push_be_long(&mut be_tiff, 257, 2); // ImageLength
    push_be_short(&mut be_tiff, 258, 32); // BitsPerSample
    push_be_short(&mut be_tiff, 259, 1); // Compression = None
    push_be_short(&mut be_tiff, 262, 1); // Photometric = BlackIsZero
    push_be_long(&mut be_tiff, 273, data_offset); // StripOffsets
    push_be_long(&mut be_tiff, 278, 2); // RowsPerStrip
    push_be_long(&mut be_tiff, 279, be_raw.len() as u32); // StripByteCounts
    push_be_short(&mut be_tiff, 339, 3); // SampleFormat = IEEEFP
    // Tag 42113 (GDAL_NODATA), ASCII, len = 6
    be_tiff.extend_from_slice(&42113u16.to_be_bytes());
    be_tiff.extend_from_slice(&2u16.to_be_bytes());
    be_tiff.extend_from_slice(&(nodata_str.len() as u32).to_be_bytes());
    be_tiff.extend_from_slice(&nodata_offset.to_be_bytes());
    be_tiff.extend_from_slice(&0u32.to_be_bytes()); // next IFD = 0
    be_tiff.extend_from_slice(&be_raw);
    be_tiff.extend_from_slice(nodata_str);

    let be_src = TiffSource::open(
        Arc::new(MemoryByteSource::from_vec(be_tiff.clone())),
        fake_fingerprint(&be_tiff),
        &budget,
        &cancel,
    )
    .expect("open Big-Endian Float32 TIFF");
    let mut emitted = Vec::new();
    let mut emit_fn = |level: u8, x: u32, y: u32, w: u32, h: u32, bytes: Vec<u8>| {
        emitted.push((level, x, y, w, h, bytes));
        Ok(())
    };
    be_src
        .produce_pass_stream(
            stream_spec_for_band(BandKey::Index(0), PixelFormat::R32Float),
            &ctx,
            &mut emit_fn as TileEmit<'_>,
        )
        .expect("produce_pass_stream R32Float");
    assert_eq!(emitted.len(), 1);
    assert_eq!((emitted[0].3, emitted[0].4), (2, 2));
    let pass_vals: Vec<f32> = emitted[0]
        .5
        .chunks_exact(4)
        .map(|b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    assert!(pass_vals[0].is_nan(), "nodata -9999 must become NaN");
    assert_eq!(pass_vals[1], 12.5);
    assert_eq!(pass_vals[2], 42.25);
    assert!(pass_vals[3].is_nan(), "source NaN must remain NaN");
}

#[test]
fn direct_raw_decoder_handles_jpeg_ycbcr_white_is_zero_i8_sparse_and_compressed_cap() {
    let budget = MemoryBudget::default();
    let cancel = CancelToken::new();
    let chunk_cache = SharedChunkCache::new(budget.decoded_chunk_cache_bytes);
    let tracker = DecodeBudgetTracker::new(budget.decode_working_bytes);
    let ctx = DecodeContext {
        budget: &budget,
        chunk_cache: &chunk_cache,
        budget_tracker: &tracker,
        cancel: &cancel,
    };

    // 1. TIFF Compression = 7 (JPEG) with Photometric = YCbCr (6) & custom YCbCrCoefficients (529):
    // Encode a solid red 8x8 JPEG via `image` crate (which writes standard YCbCr JFIF).
    let mut jpeg_bytes = Vec::new();
    let rgb_img = image::RgbImage::from_pixel(8, 8, image::Rgb([220, 30, 40]));
    image::DynamicImage::ImageRgb8(rgb_img)
        .write_to(&mut Cursor::new(&mut jpeg_bytes), image::ImageFormat::Jpeg)
        .expect("encode jpeg");
    let mut ycbcr_tif = Vec::new();
    {
        let mut enc = TiffEncoder::new(Cursor::new(&mut ycbcr_tif)).expect("enc");
        let mut dir = enc.image_directory().expect("dir");
        write_ifd_header(
            &mut dir,
            &TiffIfdHeader {
                width: 8,
                height: 8,
                samples_per_pixel: 3,
                bits_per_sample: &[8, 8, 8],
                photometric: 6, // YCbCr
                compression: 7, // JPEG
                planar_configuration: 1,
                sample_format: None,
                is_overview: false,
            },
        );
        // Explicitly set BT.709 YCbCrCoefficients (tag 529) as rationals to verify tags are applied.
        dir.write_tag(
            Tag::Unknown(529),
            &[
                ::tiff::encoder::Rational { n: 2126, d: 10_000 },
                ::tiff::encoder::Rational { n: 7152, d: 10_000 },
                ::tiff::encoder::Rational { n: 722, d: 10_000 },
            ][..],
        )
        .expect("coeffs");
        write_single_strip_u32(&mut dir, 8, &jpeg_bytes);
        dir.finish().expect("finish");
    }
    let ycbcr_src = TiffSource::open(
        Arc::new(MemoryByteSource::from_vec(ycbcr_tif.clone())),
        fake_fingerprint(&ycbcr_tif),
        &budget,
        &cancel,
    )
    .expect("open JPEG YCbCr TIFF");
    let display_px = ycbcr_src
        .sample_display_rgba(4, 4, &ctx)
        .expect("sample ycbcr")
        .expect("opaque");
    assert!(
        display_px[0] > 180 && display_px[1] < 80 && display_px[2] < 80 && display_px[3] == 255,
        "JPEG YCbCr red should decode cleanly, got {display_px:?}"
    );
    // Scientific band 0 on YCbCr IFD must also return converted Red, not raw Y luma!
    let sci_red = ycbcr_src
        .sample_band(BandKey::Index(0), PixelFormat::R32Float, 4, 4, &ctx)
        .expect("sample sci ycbcr");
    match sci_red {
        BandSample::Scalar { raw, .. } => {
            assert!(raw > 180.0, "expected converted Red > 180, got {raw}");
        }
        other => panic!("expected Scalar, got {other:?}"),
    }

    // 2. WhiteIsZero with GDAL_NODATA = "0": Display inverts valid pixels AFTER nodata check,
    //    while Scientific readout preserves exact raw uninverted values.
    let mut wiz_tif = Vec::new();
    {
        let mut enc = TiffEncoder::new(Cursor::new(&mut wiz_tif)).expect("enc");
        let mut dir = enc.image_directory().expect("dir");
        write_ifd_header(
            &mut dir,
            &TiffIfdHeader {
                width: 4,
                height: 1,
                samples_per_pixel: 1,
                bits_per_sample: &[8],
                photometric: 0, // WhiteIsZero
                compression: 1,
                planar_configuration: 1,
                sample_format: Some(1),
                is_overview: false,
            },
        );
        dir.write_tag(Tag::GdalNodata, "0").expect("nodata");
        write_single_strip_u32(&mut dir, 1, &[0u8, 10, 200, 255]);
        dir.finish().expect("finish");
    }
    let wiz_src = TiffSource::open(
        Arc::new(MemoryByteSource::from_vec(wiz_tif.clone())),
        fake_fingerprint(&wiz_tif),
        &budget,
        &cancel,
    )
    .expect("open WhiteIsZero");
    assert_eq!(
        wiz_src.sample_display_rgba(0, 0, &ctx).unwrap(),
        None,
        "nodata 0 stays transparent"
    );
    assert_eq!(
        wiz_src.sample_display_rgba(1, 0, &ctx).unwrap(),
        Some([245, 245, 245, 255])
    );
    assert_eq!(
        wiz_src.sample_display_rgba(3, 0, &ctx).unwrap(),
        Some([0, 0, 0, 255])
    );
    assert_eq!(
        wiz_src
            .sample_band(BandKey::Index(0), PixelFormat::R32Float, 0, 0, &ctx)
            .unwrap(),
        BandSample::Nodata
    );
    assert_eq!(
        wiz_src
            .sample_band(BandKey::Index(0), PixelFormat::R32Float, 1, 0, &ctx)
            .unwrap(),
        BandSample::Scalar {
            raw: 10.0,
            physical: 10.0
        }
    );

    // 3. Signed 8-bit (`SampleType::I8`) stretches negative values on Display instead of crushing to 0.
    let mut i8_tif = Vec::new();
    {
        let mut enc = TiffEncoder::new(Cursor::new(&mut i8_tif)).expect("enc");
        let mut dir = enc.image_directory().expect("dir");
        write_ifd_header(
            &mut dir,
            &TiffIfdHeader {
                width: 4,
                height: 1,
                samples_per_pixel: 1,
                bits_per_sample: &[8],
                photometric: 1,
                compression: 1,
                planar_configuration: 1,
                sample_format: Some(2), // signed int
                is_overview: false,
            },
        );
        let raw_i8: [u8; 4] = [(-100i8) as u8, (-50i8) as u8, 0u8, 100u8];
        write_single_strip_u32(&mut dir, 1, &raw_i8);
        dir.finish().expect("finish");
    }
    let i8_src = TiffSource::open(
        Arc::new(MemoryByteSource::from_vec(i8_tif.clone())),
        fake_fingerprint(&i8_tif),
        &budget,
        &cancel,
    )
    .expect("open I8 TIFF");
    let px_neg50 = i8_src
        .sample_display_rgba(1, 0, &ctx)
        .unwrap()
        .expect("opaque");
    assert!(
        px_neg50[0] > 40 && px_neg50[0] < 90,
        "I8 value -50 over [-100, 100] should stretch to ~64, got {px_neg50:?}"
    );

    // 4. Sparse chunk (`offset = 0, byte_count = 0`) returns transparent Display and Nodata Scientific.
    let mut sparse_tif = Vec::new();
    {
        let mut enc = TiffEncoder::new(Cursor::new(&mut sparse_tif)).expect("enc");
        let mut dir = enc.image_directory().expect("dir");
        write_ifd_header(
            &mut dir,
            &TiffIfdHeader {
                width: 4,
                height: 4,
                samples_per_pixel: 1,
                bits_per_sample: &[32],
                photometric: 1,
                compression: 1,
                planar_configuration: 1,
                sample_format: Some(3),
                is_overview: false,
            },
        );
        dir.write_tag(Tag::RowsPerStrip, 4u32).expect("rps");
        dir.write_tag(Tag::StripOffsets, &[0u32][..]).expect("off");
        dir.write_tag(Tag::StripByteCounts, &[0u32][..])
            .expect("bc");
        dir.finish().expect("finish");
    }
    let sparse_src = TiffSource::open(
        Arc::new(MemoryByteSource::from_vec(sparse_tif.clone())),
        fake_fingerprint(&sparse_tif),
        &budget,
        &cancel,
    )
    .expect("open sparse TIFF");
    assert_eq!(sparse_src.sample_display_rgba(1, 1, &ctx).unwrap(), None);
    assert_eq!(
        sparse_src
            .sample_band(BandKey::Index(0), PixelFormat::R32Float, 1, 1, &ctx)
            .unwrap(),
        BandSample::Nodata
    );

    // 5. Crafted oversized StripByteCounts on tiny 4x4 tile rejected by per-chunk `max_compressed` cap.
    let mut bomb_tif = Vec::new();
    {
        let mut enc = TiffEncoder::new(Cursor::new(&mut bomb_tif)).expect("enc");
        let mut dir = enc.image_directory().expect("dir");
        write_ifd_header(
            &mut dir,
            &TiffIfdHeader {
                width: 4,
                height: 4,
                samples_per_pixel: 1,
                bits_per_sample: &[8],
                photometric: 1,
                compression: 8, // Deflate
                planar_configuration: 1,
                sample_format: Some(1),
                is_overview: false,
            },
        );
        dir.write_tag(Tag::RowsPerStrip, 4u32).expect("rps");
        dir.write_tag(Tag::StripOffsets, &[8u32][..]).expect("off");
        dir.write_tag(Tag::StripByteCounts, &[32u32 * 1024 * 1024][..])
            .expect("bc");
        dir.finish().expect("finish");
    }
    let bomb_src = TiffSource::open(
        Arc::new(MemoryByteSource::from_vec(bomb_tif.clone())),
        fake_fingerprint(&bomb_tif),
        &budget,
        &cancel,
    )
    .expect("open bomb TIFF");
    assert!(matches!(
        bomb_src.sample_display_rgba(0, 0, &ctx),
        Err(RasterError::LimitExceeded { .. })
    ));
}
