//! Integration tests for drone RGB GeoTIFF codecs (YCbCr +
//! `YCbCrCoefficients`/`ReferenceBlackWhite`, WebP `0xC351`), PNG/JPEG
//! adapters, and world-file / `.prj` sidecars.

use std::io::Cursor;
use std::sync::Arc;

use veldmap_core::budget::{DecodeBudgetTracker, MemoryBudget};
use veldmap_core::io::ByteSource;
use veldmap_core::io::testing::MemoryByteSource;
use veldmap_core::sync::CancelToken;
use veldmap_core::tile::SourceFingerprint;
use veldmap_raster::{
    DecodeContext, GeoPlacement, JpegSource, PlacementCrs, PngSource, RasterError, RasterSource,
    SharedChunkCache, TiffSource, frame_from_description, has_jpeg_magic, has_png_magic,
    parse_prj_text, parse_world_file,
};

use crate::common::{build_raw_tiff, tag_rational_vec, tag_u16, tag_u16_vec, tag_u32};

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
fn png_and_jpeg_adapters_decode_world_file_and_enforce_memory_budget() {
    let wkt_utm37 = r#"PROJCS["WGS 84 / UTM zone 37N",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",39],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1]]"#;
    let world_text = "1.0\n0.0\n0.0\n-1.0\n550000.5\n5230009.5\n";
    let placement = GeoPlacement {
        crs: Some(PlacementCrs::Resolved {
            crs: parse_prj_text(wkt_utm37).expect("parse prj"),
            warning: None,
        }),
        affine: parse_world_file(world_text).expect("parse world file"),
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
