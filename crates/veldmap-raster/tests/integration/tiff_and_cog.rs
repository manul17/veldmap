//! Integration tests for COG I/O amplification, stateless cached-IFD chunk
//! reading, overview-vs-pass equivalence, IFD/dimension limit guards, and
//! remote HTTP-Range COG enforcement.

use std::io::Cursor;
use std::sync::Arc;

use ::tiff::encoder::TiffEncoder;
use ::tiff::tags::Tag;
use veldmap_core::budget::MemoryBudget;
use veldmap_core::io::ByteSource;
use veldmap_core::io::testing::{CountingByteSource, MemoryByteSource};
use veldmap_core::sync::CancelToken;
use veldmap_core::tile::{BandKey, PixelFormat, RASTER_TILE_SIZE, SourceFingerprint, tile_extent};
use veldmap_raster::{
    BandSample, DISPLAY_STREAM_SPEC, MAX_IFDS, MAX_SOURCE_SIDE, RasterError, RasterSource,
    TiffSource, open_raster, stream_spec_for_band,
};

use crate::common::{
    DecodeHarness, TiffIfdHeader, build_stripped_tiff, build_tiled_cog, fake_fingerprint,
    open_mem_tiff, write_ifd_header, write_single_strip_u32,
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
    let harness = DecodeHarness::new();

    let src = TiffSource::open(
        Arc::clone(&counting) as Arc<dyn ByteSource>,
        fp,
        &harness.budget,
        &harness.cancel,
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

    // Pointwise read of tile (0, 1, 1) touches only chunk (1, 1) and decodes expected pixels.
    counting.reset();
    let ctx = harness.ctx();
    let tile = src
        .decode_pointwise_tile(DISPLAY_STREAM_SPEC, 0, 1, 1, &ctx)
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

    // A second pointwise miss on the SAME IFD reuses the cached IFD tables without
    // re-reading offset 0 (TIFF header) or re-walking IFDs.
    counting.reset();
    let tile2 = src
        .decode_pointwise_tile(DISPLAY_STREAM_SPEC, 0, 0, 0, &ctx)
        .expect("decode tile (0,0)");
    assert_eq!(
        (tile2.width(), tile2.height()),
        (RASTER_TILE_SIZE, RASTER_TILE_SIZE)
    );
    let second_stats = counting.snapshot();
    assert!(
        !second_stats.spans.iter().any(|s| s.offset == 0),
        "cached IFD decoder must not re-read TIFF header on same-IFD miss: {:?}",
        second_stats.spans
    );
    // Direct path also tracks `region + tile` working bytes on `DecodeBudgetTracker`.
    assert!(harness.tracker.peak_bytes() > 0);
    assert_eq!(harness.tracker.active_bytes(), 0);
}

#[test]
fn direct_overview_matches_sequential_pass_on_halved_cog() {
    let (file, _, _) = build_tiled_cog(2 * RASTER_TILE_SIZE, 2 * RASTER_TILE_SIZE, 1);
    let harness = DecodeHarness::new();
    let src = open_mem_tiff(file, &harness.budget, &harness.cancel).expect("open COG");

    let ctx = harness.ctx();
    let direct_tile = src
        .decode_pointwise_tile(DISPLAY_STREAM_SPEC, 1, 0, 0, &ctx)
        .expect("direct overview tile");

    let mut pass_lvl1 = None;
    src.produce_pass(
        DISPLAY_STREAM_SPEC,
        &ctx,
        &mut |lvl, tx, ty, _w, _h, rgba| {
            if (lvl, tx, ty) == (1, 0, 0) {
                pass_lvl1 = Some(rgba);
            }
            Ok(())
        },
    )
    .expect("sequential pass");
    assert_eq!(
        direct_tile.data(),
        pass_lvl1.expect("pass produced lvl 1").as_slice()
    );
}

#[test]
fn stripped_tiff_pass_emits_all_levels_in_single_read() {
    let stripped = build_stripped_tiff(2 * RASTER_TILE_SIZE, 2 * RASTER_TILE_SIZE, 64);
    let s_len = stripped.len() as u64;
    let s_fp = fake_fingerprint(&stripped);
    let s_counting = Arc::new(CountingByteSource::new(Arc::new(
        MemoryByteSource::from_vec(stripped),
    )));
    let harness = DecodeHarness::new();
    let s_src = TiffSource::open(
        Arc::clone(&s_counting) as Arc<dyn ByteSource>,
        s_fp,
        &harness.budget,
        &harness.cancel,
    )
    .expect("open stripped");
    s_counting.reset();

    let ctx = harness.ctx();
    let mut emitted = Vec::new();
    s_src
        .produce_pass(DISPLAY_STREAM_SPEC, &ctx, &mut |lvl, tx, ty, w, h, rgba| {
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
}

#[test]
fn npot_direct_level1_matches_sequential_pass() {
    let harness = DecodeHarness::new();
    let npot = build_stripped_tiff(700, 600, 600);
    let npot_src = open_mem_tiff(npot, &harness.budget, &harness.cancel).expect("open NPOT");

    let ctx = harness.ctx();
    let npot_direct = npot_src
        .decode_pointwise_tile(DISPLAY_STREAM_SPEC, 1, 0, 0, &ctx)
        .expect("direct NPOT lvl 1");
    let mut npot_pass = None;
    npot_src
        .produce_pass(DISPLAY_STREAM_SPEC, &ctx, &mut |lvl, tx, ty, w, h, rgba| {
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
    let many_src = open_mem_tiff(many_buf.into_inner(), &budget, &cancel).expect("open 70-IFD");
    assert_eq!(many_src.overview_count(), MAX_IFDS - 1);
}

#[test]
fn tiled_tiff_above_max_source_side_opens_and_decodes_while_untiled_is_rejected() {
    let harness = DecodeHarness::new();

    // 1. Untiled stripped TIFF exceeding `MAX_SOURCE_SIDE` must still be rejected.
    let mut untiled_buf = Cursor::new(Vec::new());
    {
        let mut enc = TiffEncoder::new(&mut untiled_buf).expect("enc");
        let mut dir = enc.image_directory().expect("dir");
        write_ifd_header(&mut dir, &TiffIfdHeader::gray8(MAX_SOURCE_SIDE + 1, 16));
        write_single_strip_u32(&mut dir, 16, &[0u8; 16]);
        dir.finish().expect("finish");
    }
    let untiled_res = open_mem_tiff(untiled_buf.into_inner(), &harness.budget, &harness.cancel);
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
    let tiled_src = open_mem_tiff(tiled_buf.into_inner(), &harness.budget, &harness.cancel)
        .expect("wide tiled BigTIFF opens");
    assert_eq!(tiled_src.description().width, wide);

    let ctx = harness.ctx();
    let tile = tiled_src
        .decode_pointwise_tile(DISPLAY_STREAM_SPEC, 0, across - 1, 0, &ctx)
        .expect("decode far-right edge tile");
    assert_eq!(
        (tile.width(), tile.height()),
        (tile_extent(across - 1, wide), RASTER_TILE_SIZE)
    );
    assert_eq!(&tile.data()[..4], &[173, 173, 173, 255]);
}

#[test]
fn remote_stripped_tiff_is_rejected_as_terminal() {
    let budget = MemoryBudget::default();
    let cancel = CancelToken::new();

    let mut stripped_buf = Cursor::new(Vec::new());
    {
        let mut enc = TiffEncoder::new(&mut stripped_buf).expect("enc");
        let mut dir = enc.image_directory().expect("dir");
        write_ifd_header(&mut dir, &TiffIfdHeader::gray8(16, 16));
        write_single_strip_u32(&mut dir, 16, &[42u8; 256]);
        dir.finish().expect("finish");
    }
    let stripped_remote = Arc::new(CountingByteSource::new(Arc::new(
        MemoryByteSource::from_vec(stripped_buf.into_inner()).with_remote(true),
    )));
    let err = match open_raster(
        stripped_remote,
        SourceFingerprint::from_digest([9; 32]),
        None,
        &budget,
        &cancel,
    ) {
        Ok(_) => panic!("remote stripped TIFF must fail"),
        Err(e) => e,
    };
    assert!(matches!(err, RasterError::RemotePassUnsupported(_)));
    assert!(err.is_terminal());
}

#[test]
fn remote_cog_opens_in_one_read_plans_ranges_and_keeps_hover_cache_only() {
    let harness = DecodeHarness::new();

    let mut cog_buf = Cursor::new(Vec::new());
    {
        let mut enc = TiffEncoder::new(&mut cog_buf).expect("enc");
        let mut dir = enc.image_directory().expect("dir");
        write_ifd_header(
            &mut dir,
            &TiffIfdHeader {
                width: 64,
                height: 64,
                samples_per_pixel: 1,
                bits_per_sample: &[64],
                photometric: 1,
                compression: 1,
                planar_configuration: 1,
                sample_format: Some(3), // IEEEFP f64
                is_overview: false,
            },
        );
        dir.write_tag(Tag::TileWidth, 64u32).expect("tw");
        dir.write_tag(Tag::TileLength, 64u32).expect("th");
        let vals = vec![12.5f64; 64 * 64];
        let raw: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let off = dir.write_data(&raw[..]).expect("tile") as u32;
        dir.write_tag(Tag::TileOffsets, &[off][..]).expect("offs");
        dir.write_tag(Tag::TileByteCounts, &[raw.len() as u32][..])
            .expect("cnts");
        dir.finish().expect("finish");
    }
    let cog_remote = Arc::new(CountingByteSource::new(Arc::new(
        MemoryByteSource::from_vec(cog_buf.into_inner()).with_remote(true),
    )));
    let cog_src = TiffSource::open(
        Arc::clone(&cog_remote) as Arc<dyn ByteSource>,
        SourceFingerprint::from_digest([10; 32]),
        &harness.budget,
        &harness.cancel,
    )
    .expect("remote tiled COG opens");
    assert!(cog_src.is_remote());
    assert_eq!(
        cog_remote.read_calls(),
        1,
        "single-pass IFD reader must open compact COG in exactly 1 prefix read"
    );

    let ctx = harness.ctx();

    // Remote sample_band on uncached F64 COG must NOT hit ByteSource (0 extra reads!).
    let hover_miss = cog_src
        .sample_band(BandKey::Index(0), PixelFormat::R32Float, 10, 10, &ctx)
        .expect("cache-only hover");
    assert_eq!(hover_miss, BandSample::Unavailable);
    assert_eq!(
        cog_remote.read_calls(),
        1,
        "remote hover must never trigger network reads on cache miss"
    );

    // Sync byte-range planners return the exact tile span for pointwise and band stats.
    let sci_spec = stream_spec_for_band(BandKey::Index(0), PixelFormat::R32Float);
    let ranges = cog_src
        .plan_pointwise_byte_ranges(sci_spec, 0, &[(0, 0)], &harness.budget)
        .expect("plan pointwise ranges");
    assert_eq!(ranges.len(), 1);
    assert_eq!(ranges[0].len, (64 * 64 * 8) as u64);

    let stats_ranges = cog_src
        .plan_band_stats_byte_ranges(BandKey::Index(0))
        .expect("plan stats ranges");
    assert_eq!(stats_ranges.len(), 1);

    // Decoding populates SharedChunkCache so cache-only hover hits with zero extra reads.
    let _tile = cog_src
        .decode_pointwise_tile(sci_spec, 0, 0, 0, &ctx)
        .expect("decode tile");
    let reads_after_decode = cog_remote.read_calls();
    let hover_hit = cog_src
        .sample_band(BandKey::Index(0), PixelFormat::R32Float, 10, 10, &ctx)
        .expect("cached hover");
    assert!(matches!(hover_hit, BandSample::Scalar { raw, .. } if (raw - 12.5).abs() < 1e-6));
    assert_eq!(cog_remote.read_calls(), reads_after_decode);

    // Remote produce_pass is rejected as terminal.
    let mut noop = |_l: u8, _x: u32, _y: u32, _w: u32, _h: u32, _b: Vec<u8>| Ok(());
    let pass_err = cog_src
        .produce_pass(sci_spec, &ctx, &mut noop)
        .expect_err("remote pass forbidden");
    assert!(matches!(pass_err, RasterError::RemotePassUnsupported(_)));
}
