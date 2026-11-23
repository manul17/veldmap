use std::io::Cursor;
use std::sync::Arc;

use ::tiff::encoder::{TiffEncoder, colortype};
use ::tiff::tags::Tag;
use veldmap_core::budget::{DecodeBudgetTracker, MemoryBudget};
use veldmap_core::io::{CancelToken, CountingByteSource, MemoryByteSource, SourceFingerprint};
use veldmap_raster::pyramid::{self, RASTER_TILE_SIZE};
use veldmap_raster::resample::halve_rgba8;
use veldmap_raster::{DecodeContext, SharedChunkCache, TiffSource};

const WINDOW: u64 = 64 * 1024;

fn fake_fingerprint(bytes: &[u8]) -> SourceFingerprint {
    let mut digest = [0u8; 32];
    for (i, b) in bytes.iter().take(32).enumerate() {
        digest[i] = *b;
    }
    digest[31] ^= (bytes.len() & 0xFF) as u8;
    SourceFingerprint::from_digest(digest)
}

fn rgb_pattern(width: u32, height: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            out.push((x * 7 + y * 3) as u8);
            out.push((x ^ y) as u8);
            out.push(((x * y) / 97) as u8);
        }
    }
    out
}

fn rgb_to_rgba(rgb: &[u8]) -> Vec<u8> {
    rgb.chunks_exact(3)
        .flat_map(|px| [px[0], px[1], px[2], 255])
        .collect()
}

fn rgba_to_rgb(rgba: &[u8]) -> Vec<u8> {
    rgba.chunks_exact(4)
        .flat_map(|px| [px[0], px[1], px[2]])
        .collect()
}

fn tile_bytes(rgb: &[u8], width: u32, height: u32, tx: u32, ty: u32) -> Vec<u8> {
    let mut out = vec![0u8; (RASTER_TILE_SIZE * RASTER_TILE_SIZE * 3) as usize];
    for row in 0..RASTER_TILE_SIZE {
        let y = ty * RASTER_TILE_SIZE + row;
        if y >= height {
            break;
        }
        let x0 = tx * RASTER_TILE_SIZE;
        let x1 = (x0 + RASTER_TILE_SIZE).min(width);
        let from = ((y * width + x0) * 3) as usize;
        let len = ((x1 - x0) * 3) as usize;
        let to = (row * RASTER_TILE_SIZE * 3) as usize;
        out[to..to + len].copy_from_slice(&rgb[from..from + len]);
    }
    out
}

/// Build an uncompressed tiled TIFF with `levels` 2x2-halved overview IFDs,
/// returning `(bytes, base_tile_byte_ranges, ifd_offsets)`.
fn build_tiled_cog(width: u32, height: u32, levels: u32) -> (Vec<u8>, Vec<(u64, u64)>, Vec<u64>) {
    let mut file = Cursor::new(Vec::new());
    let mut base_tiles = Vec::new();
    let mut ifds = Vec::new();
    {
        let mut encoder = TiffEncoder::new(&mut file).expect("tiff encoder");
        let mut rgb = rgb_pattern(width, height);
        let (mut w, mut h) = (width, height);

        for level in 0..=levels {
            let mut dir = encoder.image_directory().expect("image_directory");
            if level > 0 {
                dir.write_tag(Tag::NewSubfileType, 1u32).expect("subfile");
            }
            dir.write_tag(Tag::ImageWidth, w).expect("width");
            dir.write_tag(Tag::ImageLength, h).expect("height");
            dir.write_tag(Tag::BitsPerSample, &[8u16, 8, 8][..])
                .expect("bps");
            dir.write_tag(Tag::Compression, 1u16).expect("compression");
            dir.write_tag(Tag::PhotometricInterpretation, 2u16)
                .expect("photometric");
            dir.write_tag(Tag::SamplesPerPixel, 3u16).expect("spp");
            dir.write_tag(Tag::PlanarConfiguration, 1u16)
                .expect("planar");
            dir.write_tag(Tag::TileWidth, RASTER_TILE_SIZE).expect("tw");
            dir.write_tag(Tag::TileLength, RASTER_TILE_SIZE)
                .expect("th");

            let (across, down) = (w.div_ceil(RASTER_TILE_SIZE), h.div_ceil(RASTER_TILE_SIZE));
            let mut offsets = Vec::new();
            let mut counts = Vec::new();
            for ty in 0..down {
                for tx in 0..across {
                    let tile = tile_bytes(&rgb, w, h, tx, ty);
                    let at = dir.write_data(&tile[..]).expect("write tile");
                    offsets.push(at as u32);
                    counts.push(tile.len() as u32);
                    if level == 0 {
                        base_tiles.push((at, tile.len() as u64));
                    }
                }
            }
            dir.write_tag(Tag::TileOffsets, &offsets[..]).expect("offs");
            dir.write_tag(Tag::TileByteCounts, &counts[..])
                .expect("cnts");
            let finish = dir.finish_with_offsets().expect("finish ifd");
            ifds.push(u64::from(finish.offset));

            rgb = rgba_to_rgb(&halve_rgba8(&rgb_to_rgba(&rgb), w, h));
            w = pyramid::level_size(w, 1);
            h = pyramid::level_size(h, 1);
        }
    }
    (file.into_inner(), base_tiles, ifds)
}

fn build_stripped_tiff(width: u32, height: u32, rows_per_strip: u32) -> Vec<u8> {
    let mut file = Cursor::new(Vec::new());
    {
        let mut encoder = TiffEncoder::new(&mut file).expect("tiff encoder");
        let mut image = encoder
            .new_image::<colortype::RGB8>(width, height)
            .expect("new_image");
        image.rows_per_strip(rows_per_strip).expect("rows");
        image
            .write_data(&rgb_pattern(width, height))
            .expect("write stripped");
    }
    file.into_inner()
}

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
fn describing_reads_only_header_and_ifds_and_pointwise_reads_own_chunk() {
    let levels = 2;
    let (file, base_tiles, ifds) =
        build_tiled_cog(2 * RASTER_TILE_SIZE, 2 * RASTER_TILE_SIZE, levels);
    assert!(file.len() as u64 > 8 * WINDOW);

    let fp = fake_fingerprint(&file);
    let counting = Arc::new(CountingByteSource::new(Arc::new(
        MemoryByteSource::from_vec(file),
    )));
    let budget = MemoryBudget::detect();
    let cancel = CancelToken::new();

    let src = TiffSource::open(
        Arc::clone(&counting) as Arc<dyn veldmap_core::io::ByteSource>,
        fp,
        &budget,
        &cancel,
    )
    .expect("open COG");
    assert_eq!(src.grid().overviews.len(), levels as usize);

    let describe_stats = counting.snapshot();
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

    // Pointwise read of tile (0, 1, 1) touches only chunk (1, 1) + IFDs.
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
}

#[test]
fn direct_overview_matches_sequential_pass_on_halved_cog() {
    let (file, _, _) = build_tiled_cog(2 * RASTER_TILE_SIZE, 2 * RASTER_TILE_SIZE, 1);
    let fp = fake_fingerprint(&file);
    let budget = MemoryBudget::detect();
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

    // Stripped TIFF sequential pass emits every pyramid level.
    let stripped = build_stripped_tiff(2 * RASTER_TILE_SIZE, 2 * RASTER_TILE_SIZE, 64);
    let s_fp = fake_fingerprint(&stripped);
    let s_src = TiffSource::open(
        Arc::new(MemoryByteSource::from_vec(stripped)),
        s_fp,
        &budget,
        &cancel,
    )
    .expect("open stripped");
    let mut emitted = Vec::new();
    s_src
        .produce_pass(&ctx, &mut |lvl, tx, ty, _, _, _| {
            emitted.push((lvl, tx, ty));
            Ok(())
        })
        .expect("stripped pass");
    assert_eq!(emitted.len(), 5);
}

#[test]
fn decodes_bigtiff_zstd_predictor3_f32_and_geotiff_params() {
    let budget = MemoryBudget::detect();
    let cancel = CancelToken::new();
    let chunk_cache = SharedChunkCache::new(budget.decoded_chunk_cache_bytes);
    let tracker = DecodeBudgetTracker::new(budget.decode_working_bytes);
    let ctx = DecodeContext {
        chunk_cache: &chunk_cache,
        budget_tracker: &tracker,
        budget: &budget,
        cancel: &cancel,
    };

    // 1. BigTIFF + GeoDoubleParamsTag + GeoAsciiParamsTag + PixelIsPoint + ModelTransformationTag.
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
        // GeoKeyDirectory: ModelType=1 (Projected), RasterType=2 (PixelIsPoint),
        // ProjectedCSType=32637, ProjNatOriginLat (3081) -> DoubleParams[0],
        // GTCitationGeoKey (1026) -> AsciiParams[0..7].
        let geokeys: [u16; 24] = [
            1, 1, 0, 5, // header: 5 keys
            1024, 0, 1, 1, // GTModelTypeGeoKey = Projected
            1025, 0, 1, 2, // GTRasterTypeGeoKey = PixelIsPoint
            1026, 34737, 7, 0, // GTCitationGeoKey in GeoAsciiParamsTag
            3072, 0, 1, 32637, // ProjectedCSTypeGeoKey = EPSG:32637
            3081, 34736, 1, 0, // ProjNatOriginLatGeoKey in GeoDoubleParamsTag
        ];
        dir.write_tag(Tag::GeoKeyDirectoryTag, &geokeys[..])
            .expect("tag");
        dir.write_tag(Tag::GeoDoubleParamsTag, &[55.75_f64][..])
            .expect("tag");
        dir.write_tag(Tag::GeoAsciiParamsTag, "VeldMap|")
            .expect("tag");
        let matrix: [f64; 16] = [
            10.0,
            0.0,
            0.0,
            500_000.0, // x = 10*i + 500000
            0.0,
            -10.0,
            0.0,
            6_000_000.0, // y = -10*j + 6000000
            0.0,
            0.0,
            0.0,
            0.0, //
            0.0,
            0.0,
            0.0,
            1.0,
        ];
        dir.write_tag(Tag::ModelTransformationTag, &matrix[..])
            .expect("tag");
        dir.finish().expect("finish");
    }
    let big_bytes = big_buf.into_inner();
    assert!(veldmap_raster::tiff::has_tiff_magic(&big_bytes));
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
    assert_eq!(placement.epsg, Some(32637));
    assert_eq!(placement.geo_params.get_f64(3081), Some(55.75));
    assert_eq!(placement.geo_params.get_ascii(1026), Some("VeldMap"));
    // PixelIsPoint shifts affine origin by half a pixel: 500000 - 5 = 499995, 6000000 + 5 = 6000005.
    assert!((placement.affine[2] - 499_995.0).abs() < 1e-9);
    assert!((placement.affine[5] - 6_000_005.0).abs() < 1e-9);
    let tile = big_src
        .decode_pointwise_tile(0, 0, 0, &ctx)
        .expect("decode BigTIFF tile");
    assert_eq!((tile.width(), tile.height()), (8, 8));

    // 2. Float32 + Predictor = 3 (FloatingPoint) + ZSTD compression + GDAL_NODATA.
    let mut f32_buf = Cursor::new(Vec::new());
    let (w, h) = (4u32, 4u32);
    let floats: Vec<f32> = vec![
        -9999.0, 10.0, 20.0, 30.0, // row 0 (col 0 = nodata)
        40.0, 50.0, 60.0, 70.0, // row 1
        80.0, 90.0, 100.0, 110.0, // row 2
        120.0, 130.0, 140.0, 150.0, // row 3
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
        dir.write_tag(Tag::SampleFormat, 3u16).expect("tag"); // IEEEFP
        dir.write_tag(Tag::Compression, 50_000u16).expect("tag"); // ZSTD
        dir.write_tag(Tag::Predictor, 3u16).expect("tag"); // FloatingPoint
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
    assert!(
        f32_tile.data()[15 * 4] >= 245,
        "max 150.0 should stretch near white"
    );

    // 3. Geographic (GTModelTypeGeoKey=2) with PixelIsArea + 16-bit signed (i16) samples.
    let mut i16_buf = Cursor::new(Vec::new());
    {
        let mut enc = TiffEncoder::new(&mut i16_buf).expect("enc");
        let mut dir = enc.image_directory().expect("dir");
        dir.write_tag(Tag::ImageWidth, 4u32).expect("tag");
        dir.write_tag(Tag::ImageLength, 2u32).expect("tag");
        dir.write_tag(Tag::BitsPerSample, 16u16).expect("tag");
        dir.write_tag(Tag::SampleFormat, 2u16).expect("tag"); // Int16
        dir.write_tag(Tag::Compression, 1u16).expect("tag");
        dir.write_tag(Tag::PhotometricInterpretation, 1u16)
            .expect("tag");
        dir.write_tag(Tag::SamplesPerPixel, 1u16).expect("tag");
        dir.write_tag(Tag::PlanarConfiguration, 1u16).expect("tag");
        dir.write_tag(Tag::RowsPerStrip, 2u32).expect("tag");
        dir.write_tag(Tag::GdalNodata, "-32768").expect("tag");
        let geokeys: [u16; 8] = [1, 1, 0, 1, 1024, 0, 1, 2]; // Geographic
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
        let raw_bytes: Vec<u8> = i16_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
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
    assert_eq!(i16_src.description().ties.len(), 4);
    assert!((i16_src.description().ties[3].lon - 12.0).abs() < 1e-9);
    assert!((i16_src.description().ties[3].lat - 49.5).abs() < 1e-9);
    let i16_tile = i16_src
        .decode_pointwise_tile(0, 0, 0, &ctx)
        .expect("decode i16");
    assert_eq!(i16_tile.data()[3], 0, "i16 nodata must be transparent");
    assert_eq!(i16_tile.data()[7], 255);

    // 4. PlanarConfiguration = 2 is rejected cleanly at open time.
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

    // 5. MAX_IFDS cap bounds scanning of multi-IFD TIFFs (> 64 IFDs).
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
    assert_eq!(
        many_src.grid().overviews.len(),
        veldmap_raster::tiff::MAX_IFDS - 1
    );
}
