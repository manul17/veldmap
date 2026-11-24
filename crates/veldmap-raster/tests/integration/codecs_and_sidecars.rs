//! Integration tests for Slice 3A-2 drone RGB GeoTIFF codecs (YCbCr +
//! `YCbCrCoefficients`/`ReferenceBlackWhite`, WebP `0xC351`), user-defined
//! GeoKeys (`3072 == 32767`, `3076` feet, `3081` origin lat, `2062` TOWGS84),
//! foreign-datum geographic GeoTIFF datum shifts, PNG/JPEG adapters, and
//! world-file / `.prj` sidecars.

use std::io::Cursor;
use std::sync::Arc;

use veldmap_core::budget::{DecodeBudgetTracker, MemoryBudget};
use veldmap_core::geo::{PULKOVO_1942, US_SURVEY_FOOT_M};
use veldmap_core::io::ByteSource;
use veldmap_core::io::testing::MemoryByteSource;
use veldmap_core::sync::CancelToken;
use veldmap_core::tile::SourceFingerprint;
use veldmap_raster::{
    DecodeContext, GeoPlacement, JpegSource, PngSource, RasterError, RasterSource,
    SharedChunkCache, TiffSource, frame_from_description, has_jpeg_magic, has_png_magic,
    parse_prj_text, parse_world_file,
};

/// Build a minimal classic little-endian single-strip TIFF with explicit IFD
/// entries (sorted by tag id).
fn build_raw_tiff(entries: &mut [(u16, u16, u32, Vec<u8>)], pixel_payload: &[u8]) -> Vec<u8> {
    entries.sort_by_key(|(tag, _, _, _)| *tag);
    // Header: 8 bytes (II, 42, ifd_offset=8)
    let mut out = vec![b'I', b'I', 42, 0, 8, 0, 0, 0];
    let n_entries = entries.len() as u16;
    out.extend_from_slice(&n_entries.to_le_bytes());

    let ifd_size = 2 + entries.len() * 12 + 4;
    let mut extra_data_offset = (8 + ifd_size) as u32;
    let mut extra_blobs: Vec<u8> = Vec::new();

    // Reserve pixel_payload right after extra blobs.
    let mut inline_or_deferred: Vec<(u16, u16, u32, [u8; 4], bool)> = Vec::new();
    for (tag, ty, count, raw) in entries.iter() {
        if *tag == 273 {
            // StripOffsets placeholder (patched once extra_blobs finishes)
            inline_or_deferred.push((*tag, *ty, *count, [0; 4], true));
            continue;
        }
        if raw.len() <= 4 {
            let mut val = [0u8; 4];
            val[..raw.len()].copy_from_slice(raw);
            inline_or_deferred.push((*tag, *ty, *count, val, false));
        } else {
            let off = extra_data_offset + extra_blobs.len() as u32;
            extra_blobs.extend_from_slice(raw);
            if extra_blobs.len() % 2 == 1 {
                extra_blobs.push(0);
            }
            inline_or_deferred.push((*tag, *ty, *count, off.to_le_bytes(), false));
        }
    }

    let pixel_offset = extra_data_offset + extra_blobs.len() as u32;
    let _ = &mut extra_data_offset;
    for (tag, ty, count, mut val, is_strip_off) in inline_or_deferred {
        if is_strip_off {
            val = pixel_offset.to_le_bytes();
        }
        out.extend_from_slice(&tag.to_le_bytes());
        out.extend_from_slice(&ty.to_le_bytes());
        out.extend_from_slice(&count.to_le_bytes());
        out.extend_from_slice(&val);
    }
    // Next IFD = 0
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&extra_blobs);
    out.extend_from_slice(pixel_payload);
    out
}

fn tag_u16(tag: u16, v: u16) -> (u16, u16, u32, Vec<u8>) {
    (tag, 3, 1, v.to_le_bytes().to_vec())
}

fn tag_u16_vec(tag: u16, vals: &[u16]) -> (u16, u16, u32, Vec<u8>) {
    let mut b = Vec::with_capacity(vals.len() * 2);
    for v in vals {
        b.extend_from_slice(&v.to_le_bytes());
    }
    (tag, 3, vals.len() as u32, b)
}

fn tag_u32(tag: u16, v: u32) -> (u16, u16, u32, Vec<u8>) {
    (tag, 4, 1, v.to_le_bytes().to_vec())
}

fn tag_f64_vec(tag: u16, vals: &[f64]) -> (u16, u16, u32, Vec<u8>) {
    let mut b = Vec::with_capacity(vals.len() * 8);
    for v in vals {
        b.extend_from_slice(&v.to_le_bytes());
    }
    (tag, 12, vals.len() as u32, b)
}

fn tag_rational_vec(tag: u16, vals: &[(u32, u32)]) -> (u16, u16, u32, Vec<u8>) {
    let mut b = Vec::with_capacity(vals.len() * 8);
    for &(n, d) in vals {
        b.extend_from_slice(&n.to_le_bytes());
        b.extend_from_slice(&d.to_le_bytes());
    }
    (tag, 5, vals.len() as u32, b)
}

#[test]
fn decodes_ycbcr_444_tiff_with_bt601_and_reference_black_white() {
    // 4x4 uncompressed 4:4:4 YCbCr TIFF:
    // Pixel (0,0): Y=255, Cb=128, Cr=128 -> pure white (255, 255, 255)
    // Pixel (1,0): Y=76,  Cb=85,  Cr=255 -> saturated red (~254, 0, 0)
    // Remaining pixels: neutral mid-gray Y=128, Cb=128, Cr=128
    let mut payload = vec![128u8; 4 * 4 * 3];
    payload[0..3].copy_from_slice(&[255, 128, 128]);
    payload[3..6].copy_from_slice(&[76, 85, 255]);

    let mut entries = vec![
        tag_u32(256, 4),                                                 // ImageWidth
        tag_u32(257, 4),                                                 // ImageLength
        tag_u16_vec(258, &[8, 8, 8]),                                    // BitsPerSample
        tag_u16(259, 1),                                                 // Compression = None
        tag_u16(262, 6),                                                 // Photometric = YCbCr
        tag_u32(273, 0),                                                 // StripOffsets (patched)
        tag_u16(277, 3),                                                 // SamplesPerPixel
        tag_u32(278, 4),                                                 // RowsPerStrip
        tag_u32(279, payload.len() as u32),                              // StripByteCounts
        tag_u16(284, 1), // PlanarConfiguration = Chunky
        tag_rational_vec(529, &[(299, 1000), (587, 1000), (114, 1000)]), // YCbCrCoefficients
        tag_u16_vec(530, &[1, 1]), // YCbCrSubSampling = 4:4:4
        tag_rational_vec(
            532,
            &[(0, 1), (255, 1), (128, 1), (255, 1), (128, 1), (255, 1)],
        ), // ReferenceBlackWhite
    ];
    let bytes = build_raw_tiff(&mut entries, &payload);

    let budget = MemoryBudget::default();
    let cancel = CancelToken::new();
    let src: Arc<dyn ByteSource> = Arc::new(MemoryByteSource::new(bytes.into()));
    let tiff = TiffSource::open(
        src,
        SourceFingerprint::from_digest([1; 32]),
        &budget,
        &cancel,
    )
    .expect("YCbCr 4:4:4 TIFF opens");

    let chunk_cache = SharedChunkCache::new(budget.decoded_chunk_cache_bytes);
    let tracker = DecodeBudgetTracker::new(budget.decode_working_bytes);
    let ctx = DecodeContext {
        chunk_cache: &chunk_cache,
        budget_tracker: &tracker,
        budget: &budget,
        cancel: &cancel,
    };

    let white = tiff
        .sample_display_rgba(0, 0, &ctx)
        .expect("probe")
        .expect("white");
    assert_eq!(white, [255, 255, 255, 255]);

    let red = tiff
        .sample_display_rgba(1, 0, &ctx)
        .expect("probe")
        .expect("red");
    assert!(
        red[0] >= 250 && red[1] <= 5 && red[2] <= 5 && red[3] == 255,
        "got {red:?}"
    );

    let gray = tiff
        .sample_display_rgba(2, 2, &ctx)
        .expect("probe")
        .expect("gray");
    assert_eq!(gray, [128, 128, 128, 255]);
}

#[test]
fn decodes_webp_compressed_0xc351_rgb_tiff() {
    // Encode a 4x4 RGB lossless WebP payload using `image-webp`.
    let mut rgb = vec![0u8; 4 * 4 * 3];
    for i in 0..16 {
        rgb[i * 3] = (i * 15) as u8;
        rgb[i * 3 + 1] = 200;
        rgb[i * 3 + 2] = 50;
    }
    let mut webp_bytes = Vec::new();
    let encoder = image_webp::WebPEncoder::new(&mut webp_bytes);
    encoder
        .encode(&rgb, 4, 4, image_webp::ColorType::Rgb8)
        .expect("encode lossless webp");

    let mut entries = vec![
        tag_u32(256, 4),
        tag_u32(257, 4),
        tag_u16_vec(258, &[8, 8, 8]),
        tag_u16(259, 0xC351), // Compression = WebP (50001)
        tag_u16(262, 2),      // Photometric = RGB
        tag_u32(273, 0),
        tag_u16(277, 3),
        tag_u32(278, 4),
        tag_u32(279, webp_bytes.len() as u32),
        tag_u16(284, 1),
    ];
    let tiff_bytes = build_raw_tiff(&mut entries, &webp_bytes);

    let budget = MemoryBudget::default();
    let cancel = CancelToken::new();
    let src: Arc<dyn ByteSource> = Arc::new(MemoryByteSource::new(tiff_bytes.into()));
    let tiff = TiffSource::open(
        src,
        SourceFingerprint::from_digest([2; 32]),
        &budget,
        &cancel,
    )
    .expect("WebP TIFF opens");

    let chunk_cache = SharedChunkCache::new(budget.decoded_chunk_cache_bytes);
    let tracker = DecodeBudgetTracker::new(budget.decode_working_bytes);
    let ctx = DecodeContext {
        chunk_cache: &chunk_cache,
        budget_tracker: &tracker,
        budget: &budget,
        cancel: &cancel,
    };

    let px = tiff
        .sample_display_rgba(3, 1, &ctx)
        .expect("probe")
        .expect("pixel");
    // Index 7 -> R = 7*15 = 105, G = 200, B = 50
    assert_eq!(px, [105, 200, 50, 255]);
}

#[test]
fn geotiff_user_defined_32767_tm_feet_origin_lat_and_foreign_datum_shift() {
    // 1) User-defined projected CRS (3072 = 32767), CT_TransverseMercator (3075 = 1),
    //    US Survey Feet (3076 = 9003), origin_lat = 38.0 (3081), lon_0 = -75.0 (3080),
    //    false_easting = 656166.6667 ftUS (~200 km), scale = 0.9999 (3092).
    let fe_ft = 200_000.0 / US_SURVEY_FOOT_M;
    let doubles = [
        -75.0_f64,  // idx 0: lon_0
        38.0_f64,   // idx 1: lat_0
        fe_ft,      // idx 2: false easting (in ftUS!)
        0.0_f64,    // idx 3: false northing
        0.9999_f64, // idx 4: scale
    ];
    let geokeys: Vec<u16> = vec![
        1, 1, 0, 10, // header: 10 keys
        1024, 0, 1, 1, // ModelType = Projected
        1025, 0, 1, 1, // RasterType = PixelIsArea
        2048, 0, 1, 4326, // Geographic = WGS84
        3072, 0, 1, 32767, // Projected = User-Defined
        3075, 0, 1, 1, // CoordTrans = Transverse Mercator
        3076, 0, 1, 9003, // LinearUnits = US Survey Foot
        3080, 34736, 1, 0, // NatOriginLong
        3081, 34736, 1, 1, // NatOriginLat
        3082, 34736, 1, 2, // FalseEasting
        3092, 34736, 1, 4, // ScaleAtNatOrigin
    ];
    let payload = vec![200u8; 4 * 4 * 3];
    let mut entries = vec![
        tag_u32(256, 4),
        tag_u32(257, 4),
        tag_u16_vec(258, &[8, 8, 8]),
        tag_u16(259, 1),
        tag_u16(262, 2),
        tag_u32(273, 0),
        tag_u16(277, 3),
        tag_u32(278, 4),
        tag_u32(279, payload.len() as u32),
        tag_u16(284, 1),
        tag_f64_vec(33550, &[10.0, 10.0, 0.0]), // 10 ftUS/px
        tag_f64_vec(33922, &[0.0, 0.0, 0.0, fe_ft, 20.0, 0.0]), // UL at (fe_ft, 20 ftUS)
        tag_u16_vec(34735, &geokeys),
        tag_f64_vec(34736, &doubles),
    ];
    let tiff_bytes = build_raw_tiff(&mut entries, &payload);
    let budget = MemoryBudget::default();
    let cancel = CancelToken::new();
    let src: Arc<dyn ByteSource> = Arc::new(MemoryByteSource::new(tiff_bytes.into()));
    let tiff = TiffSource::open(
        src,
        SourceFingerprint::from_digest([3; 32]),
        &budget,
        &cancel,
    )
    .expect("user-defined GeoTIFF opens");
    let frame = frame_from_description(tiff.description()).expect("user-defined frame binds");
    let (lat, lon) = frame.geodetic(0.0, 0.5);
    assert!((lat - 38.0).abs() < 1e-4, "expected lat ~38.0, got {lat}");
    assert!((lon + 75.0).abs() < 1e-4, "expected lon ~-75.0, got {lon}");
    // Pixel size must reflect US Survey Feet (~3.048 m/px, not 10 m/px!).
    let (px_x, px_y) = frame.pixel_size_m(4, 4).expect("pixel size");
    assert!((px_x - 10.0 * US_SURVEY_FOOT_M).abs() < 0.05, "got {px_x}");
    assert!((px_y - 10.0 * US_SURVEY_FOOT_M).abs() < 0.05, "got {px_y}");

    // 2) Foreign-datum geographic GeoTIFF (EPSG:4284 Pulkovo 1942) shifts to WGS84.
    let geokeys_p42: Vec<u16> = vec![
        1, 1, 0, 3, 1024, 0, 1, 2, // Geographic
        1025, 0, 1, 1, // PixelIsArea
        2048, 0, 1, 4284, // Pulkovo 1942
    ];
    let mut entries_p42 = vec![
        tag_u32(256, 4),
        tag_u32(257, 4),
        tag_u16_vec(258, &[8, 8, 8]),
        tag_u16(259, 1),
        tag_u16(262, 2),
        tag_u32(273, 0),
        tag_u16(277, 3),
        tag_u32(278, 4),
        tag_u32(279, payload.len() as u32),
        tag_u16(284, 1),
        tag_f64_vec(33550, &[0.01, 0.01, 0.0]),
        tag_f64_vec(33922, &[0.0, 0.0, 0.0, 39.70, 47.25, 0.0]),
        tag_u16_vec(34735, &geokeys_p42),
    ];
    let p42_bytes = build_raw_tiff(&mut entries_p42, &payload);
    let p42_src: Arc<dyn ByteSource> = Arc::new(MemoryByteSource::new(p42_bytes.into()));
    let p42_tiff = TiffSource::open(
        p42_src,
        SourceFingerprint::from_digest([4; 32]),
        &budget,
        &cancel,
    )
    .expect("Pulkovo 42 geographic GeoTIFF opens");
    let p42_frame = frame_from_description(p42_tiff.description()).expect("P42 frame binds");
    let got = p42_frame.geodetic(0.5, 0.5);
    let expected = PULKOVO_1942.to_wgs84(47.23, 39.72);
    assert!((got.0 - expected.0).abs() < 1e-6);
    assert!((got.1 - expected.1).abs() < 1e-6);
}

#[test]
fn png_and_jpeg_adapters_decode_world_file_and_enforce_memory_budget() {
    let wkt_utm37 = r#"PROJCS["WGS 84 / UTM zone 37N",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",39],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1]]"#;
    let world_text = "1.0\n0.0\n0.0\n-1.0\n550000.5\n5230009.5\n";
    let placement = GeoPlacement {
        epsg: None,
        custom_crs: Some(parse_prj_text(wkt_utm37).expect("parse prj")),
        datum_warning: None,
        affine: parse_world_file(world_text).expect("parse world file"),
        geo_params: None,
    };

    // Encode a synthetic 16x16 RGB PNG.
    let img = image::RgbImage::from_fn(16, 16, |x, y| {
        image::Rgb([(x * 16) as u8, (y * 16) as u8, 128])
    });
    let mut png_bytes = Vec::new();
    img.write_to(&mut Cursor::new(&mut png_bytes), image::ImageFormat::Png)
        .expect("encode png");
    assert!(has_png_magic(&png_bytes));

    let budget = MemoryBudget::default();
    let cancel = CancelToken::new();
    let png_src = PngSource::open(
        Arc::new(MemoryByteSource::new(png_bytes.into())),
        SourceFingerprint::from_digest([5; 32]),
        Some(placement.clone()),
        &budget,
        &cancel,
    )
    .expect("PngSource opens");
    let png_frame = frame_from_description(png_src.description()).expect("PNG frame binds");
    assert!(png_frame.measurable());

    let chunk_cache = SharedChunkCache::new(budget.decoded_chunk_cache_bytes);
    let tracker = DecodeBudgetTracker::new(budget.decode_working_bytes);
    let ctx = DecodeContext {
        chunk_cache: &chunk_cache,
        budget_tracker: &tracker,
        budget: &budget,
        cancel: &cancel,
    };
    let mut emitted_png = 0usize;
    png_src
        .produce_pass(&ctx, &mut |_lvl, _x, _y, tw, th, rgba| {
            assert_eq!(rgba.len(), (tw * th * 4) as usize);
            emitted_png += 1;
            Ok(())
        })
        .expect("PNG pass succeeds");
    assert_eq!(emitted_png, 1);
    let probe = png_src
        .sample_display_rgba(2, 3, &ctx)
        .expect("png probe")
        .expect("pixel");
    assert_eq!(probe, [32, 48, 128, 255]);

    // Encode a synthetic 16x16 RGB JPEG and verify decode + tight-budget guard.
    let mut jpg_bytes = Vec::new();
    img.write_to(&mut Cursor::new(&mut jpg_bytes), image::ImageFormat::Jpeg)
        .expect("encode jpeg");
    assert!(has_jpeg_magic(&jpg_bytes));

    let jpg_src = JpegSource::open(
        Arc::new(MemoryByteSource::new(jpg_bytes.clone().into())),
        SourceFingerprint::from_digest([6; 32]),
        Some(placement.clone()),
        &budget,
        &cancel,
    )
    .expect("JpegSource opens");
    let jpg_frame = frame_from_description(jpg_src.description()).expect("JPEG frame binds");
    assert!(jpg_frame.measurable());
    let mut emitted_jpg = 0usize;
    jpg_src
        .produce_pass(&ctx, &mut |_lvl, _x, _y, tw, th, rgba| {
            assert_eq!(rgba.len(), (tw * th * 4) as usize);
            emitted_jpg += 1;
            Ok(())
        })
        .expect("JPEG pass succeeds");
    assert_eq!(emitted_jpg, 1);

    // Tiny working budget must reject JPEG upfront BEFORE allocating the full frame.
    let mut tiny_budget = budget;
    tiny_budget.decode_working_bytes = 64;
    let err = JpegSource::open(
        Arc::new(MemoryByteSource::new(jpg_bytes.into())),
        SourceFingerprint::from_digest([7; 32]),
        Some(placement),
        &tiny_budget,
        &cancel,
    );
    assert!(matches!(err, Err(RasterError::LimitExceeded { .. })));
}

#[test]
fn user_defined_geotiff_handles_datum_ellipsoid_polar_stere_sphere_and_warnings() {
    use veldmap_core::geo::{DatumWarning, KRASOVSKY};

    let budget = MemoryBudget::default();
    let cancel = CancelToken::new();
    let payload = vec![180u8; 4 * 4 * 3];

    // 1) GeoKey 2048=4284 (Pulkovo 1942) + GeoKey 2062 (TOWGS84) without 2056:
    //    must preserve Krasovsky ellipsoid (semi_major 6378245) rather than WGS84!
    let doubles_p42 = [
        39.0_f64,  // 0: lon_0
        0.0_f64,   // 1: lat_0
        500_000.0, // 2: false_easting
        1.0,       // 3: scale
        23.57, -140.95, -79.8, 0.0, -0.35, -0.79, -0.22, // 4..11: TOWGS84
    ];
    let geokeys_p42: Vec<u16> = vec![
        1, 1, 0, 10, 1024, 0, 1, 1, 1025, 0, 1, 1, 2048, 0, 1,
        4284, // Pulkovo 1942 (Krasovsky ellipsoid)
        2062, 34736, 7, 4, // GeogTOWGS84GeoKey
        3072, 0, 1, 32767, 3075, 0, 1, 1, // CT_TransverseMercator
        3080, 34736, 1, 0, 3081, 34736, 1, 1, 3082, 34736, 1, 2, 3092, 34736, 1, 3,
    ];
    let mut entries_p42 = vec![
        tag_u32(256, 4),
        tag_u32(257, 4),
        tag_u16_vec(258, &[8, 8, 8]),
        tag_u16(259, 1),
        tag_u16(262, 2),
        tag_u32(273, 0),
        tag_u16(277, 3),
        tag_u32(278, 4),
        tag_u32(279, payload.len() as u32),
        tag_u16(284, 1),
        tag_f64_vec(33550, &[10.0, 10.0, 0.0]),
        tag_f64_vec(33922, &[0.0, 0.0, 0.0, 500_000.0, 5_230_000.0, 0.0]),
        tag_u16_vec(34735, &geokeys_p42),
        tag_f64_vec(34736, &doubles_p42),
    ];
    let tiff_p42 = TiffSource::open(
        Arc::new(MemoryByteSource::new(
            build_raw_tiff(&mut entries_p42, &payload).into(),
        )),
        SourceFingerprint::from_digest([10; 32]),
        &budget,
        &cancel,
    )
    .expect("Pulkovo 42 + TOWGS84 user-defined TIFF opens");
    let placement_p42 = tiff_p42
        .description()
        .placement
        .as_ref()
        .expect("geo placement");
    match placement_p42.custom_crs.as_ref() {
        Some(veldmap_core::geo::CrsSpec::Projected(veldmap_core::geo::Projection::Transverse(
            sys,
        ))) => {
            assert!(
                (sys.datum.ellipsoid.semi_major_m - KRASOVSKY.semi_major_m).abs() < 1e-3,
                "expected Krasovsky ellipsoid, got {:?}",
                sys.datum.ellipsoid
            );
        }
        other => panic!("expected Transverse Mercator with Krasovsky ellipsoid, got {other:?}"),
    }

    // 2) CT_PolarStereographic (3075=15) using StraightVertPoleLong (3095) and
    //    southern latitude of true scale (-71.0): must project onto southern pole.
    let doubles_ps = [
        -71.0_f64, // 0: NatOriginLat (lat_ts = -71S)
        0.0_f64,   // 1: StraightVertPoleLong (lon_0 = 0)
    ];
    let geokeys_ps: Vec<u16> = vec![
        1, 1, 0, 7, 1024, 0, 1, 1, 1025, 0, 1, 1, 2048, 0, 1, 4326, 3072, 0, 1, 32767, 3075, 0, 1,
        15, // CT_PolarStereographic
        3081, 34736, 1, 0, // NatOriginLat (-71)
        3095, 34736, 1, 1, // StraightVertPoleLong (0)
    ];
    let mut entries_ps = vec![
        tag_u32(256, 4),
        tag_u32(257, 4),
        tag_u16_vec(258, &[8, 8, 8]),
        tag_u16(259, 1),
        tag_u16(262, 2),
        tag_u32(273, 0),
        tag_u16(277, 3),
        tag_u32(278, 4),
        tag_u32(279, payload.len() as u32),
        tag_u16(284, 1),
        tag_f64_vec(33550, &[1000.0, 1000.0, 0.0]),
        tag_f64_vec(33922, &[0.0, 0.0, 0.0, 0.0, 1_000_000.0, 0.0]),
        tag_u16_vec(34735, &geokeys_ps),
        tag_f64_vec(34736, &doubles_ps),
    ];
    let tiff_ps = TiffSource::open(
        Arc::new(MemoryByteSource::new(
            build_raw_tiff(&mut entries_ps, &payload).into(),
        )),
        SourceFingerprint::from_digest([11; 32]),
        &budget,
        &cancel,
    )
    .expect("Polar Stereographic user-defined TIFF opens");
    let frame_ps =
        frame_from_description(tiff_ps.description()).expect("Polar Stereographic binds");
    let (lat_ps, _lon_ps) = frame_ps.geodetic(0.5, 0.5);
    assert!(
        lat_ps < -75.0 && lat_ps > -85.0,
        "Antarctic polar stereographic (x=0, y=1000 km) should lie around -81S, got {lat_ps}"
    );

    // 3) Spherical ellipsoid (a == b = 6378137, flattening == 0) in user-defined Mercator:
    //    must emit +b= instead of +rf=inf and bind cleanly!
    let doubles_sph = [6_378_137.0_f64, 6_378_137.0_f64];
    let geokeys_sph: Vec<u16> = vec![
        1, 1, 0, 7, 1024, 0, 1, 1, 1025, 0, 1, 1, 2057, 34736, 1, 0, // SemiMajorAxis
        2058, 34736, 1, 1, // SemiMinorAxis (sphere!)
        3072, 0, 1, 32767, 3075, 0, 1, 7, // CT_Mercator
        3076, 0, 1, 9001,
    ];
    let mut entries_sph = vec![
        tag_u32(256, 4),
        tag_u32(257, 4),
        tag_u16_vec(258, &[8, 8, 8]),
        tag_u16(259, 1),
        tag_u16(262, 2),
        tag_u32(273, 0),
        tag_u16(277, 3),
        tag_u32(278, 4),
        tag_u32(279, payload.len() as u32),
        tag_u16(284, 1),
        tag_f64_vec(33550, &[100.0, 100.0, 0.0]),
        tag_f64_vec(33922, &[0.0, 0.0, 0.0, 0.0, 1000.0, 0.0]),
        tag_u16_vec(34735, &geokeys_sph),
        tag_f64_vec(34736, &doubles_sph),
    ];
    let tiff_sph = TiffSource::open(
        Arc::new(MemoryByteSource::new(
            build_raw_tiff(&mut entries_sph, &payload).into(),
        )),
        SourceFingerprint::from_digest([12; 32]),
        &budget,
        &cancel,
    )
    .expect("Spherical user-defined Mercator TIFF opens");
    let frame_sph =
        frame_from_description(tiff_sph.description()).expect("Spherical Mercator binds");
    let (lat_sph, lon_sph) = frame_sph.geodetic(0.5, 0.5);
    assert!(lat_sph.abs() < 0.1 && lon_sph.abs() < 0.1);

    // 4) User-defined TM on NAD83 (GeoKey 2048 = 4269) must attach DatumWarning::GridShiftRecommended.
    let doubles_nad = [-75.0_f64, 38.0_f64, 200_000.0_f64, 0.9999_f64];
    let geokeys_nad: Vec<u16> = vec![
        1, 1, 0, 9, 1024, 0, 1, 1, 1025, 0, 1, 1, 2048, 0, 1, 4269, // NAD83
        3072, 0, 1, 32767, 3075, 0, 1, 1, // CT_TransverseMercator
        3080, 34736, 1, 0, 3081, 34736, 1, 1, 3082, 34736, 1, 2, 3092, 34736, 1, 3,
    ];
    let mut entries_nad = vec![
        tag_u32(256, 4),
        tag_u32(257, 4),
        tag_u16_vec(258, &[8, 8, 8]),
        tag_u16(259, 1),
        tag_u16(262, 2),
        tag_u32(273, 0),
        tag_u16(277, 3),
        tag_u32(278, 4),
        tag_u32(279, payload.len() as u32),
        tag_u16(284, 1),
        tag_f64_vec(33550, &[10.0, 10.0, 0.0]),
        tag_f64_vec(33922, &[0.0, 0.0, 0.0, 200_000.0, 100.0, 0.0]),
        tag_u16_vec(34735, &geokeys_nad),
        tag_f64_vec(34736, &doubles_nad),
    ];
    let tiff_nad = TiffSource::open(
        Arc::new(MemoryByteSource::new(
            build_raw_tiff(&mut entries_nad, &payload).into(),
        )),
        SourceFingerprint::from_digest([13; 32]),
        &budget,
        &cancel,
    )
    .expect("NAD83 user-defined TIFF opens");
    let frame_nad = frame_from_description(tiff_nad.description()).expect("NAD83 frame binds");
    assert!(
        matches!(
            frame_nad.datum_warning(),
            Some(DatumWarning::GridShiftRecommended { .. })
        ),
        "expected GridShiftRecommended warning for NAD83 user-defined CRS, got {:?}",
        frame_nad.datum_warning()
    );
}
