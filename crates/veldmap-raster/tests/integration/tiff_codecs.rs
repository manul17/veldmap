//! Integration tests for TIFF compression codecs, horizontal/floating-point
//! predictors, big-endian decoding, JPEG YCbCr custom coefficients, and
//! per-chunk compressed-size caps.

use std::io::{Cursor, Write};

use ::tiff::encoder::TiffEncoder;
use ::tiff::tags::Tag;
use veldmap_core::tile::{BandKey, PixelFormat};
use veldmap_raster::{
    BandSample, DISPLAY_STREAM_SPEC, RasterError, RasterSource, TileEmit, stream_spec_for_band,
};

use crate::common::{
    DecodeHarness, TiffIfdHeader, open_mem_tiff, sample_display_rgba, write_ifd_header,
    write_single_strip_u32,
};

#[test]
fn decodes_zstd_predictor3_f32_with_gdal_nodata() {
    let harness = DecodeHarness::new();
    let ctx = harness.ctx();

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
    let f32_src = open_mem_tiff(f32_buf.into_inner(), &harness.budget, &harness.cancel)
        .expect("open ZSTD Predictor=3 Float32 TIFF");
    let f32_tile = f32_src
        .decode_pointwise_tile(DISPLAY_STREAM_SPEC, 0, 0, 0, &ctx)
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
fn decodes_deflate_with_horizontal_predictor2_u16() {
    let harness = DecodeHarness::new();
    let ctx = harness.ctx();

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
    let def_src = open_mem_tiff(def_buf.into_inner(), &harness.budget, &harness.cancel)
        .expect("open Deflate+Predictor2 u16 TIFF");
    let def_tile = def_src
        .decode_pointwise_tile(
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
}

#[test]
fn decodes_lzw_and_packbits_u8() {
    let harness = DecodeHarness::new();
    let ctx = harness.ctx();

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
    let lzw_src =
        open_mem_tiff(lzw_buf.into_inner(), &harness.budget, &harness.cancel).expect("open LZW");
    let lzw_tile = lzw_src
        .decode_pointwise_tile(
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
    let pb_src = open_mem_tiff(pb_buf.into_inner(), &harness.budget, &harness.cancel)
        .expect("open PackBits TIFF");
    let pb_tile = pb_src
        .decode_pointwise_tile(
            stream_spec_for_band(BandKey::Index(0), PixelFormat::R8Uint),
            0,
            0,
            0,
            &ctx,
        )
        .expect("decode PackBits");
    assert_eq!(pb_tile.data(), &raw_lzw);
}

#[test]
fn decodes_big_endian_f32_pass_preserving_nan() {
    let harness = DecodeHarness::new();
    let ctx = harness.ctx();

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

    let be_src = open_mem_tiff(be_tiff, &harness.budget, &harness.cancel)
        .expect("open Big-Endian Float32 TIFF");
    let mut emitted = Vec::new();
    let mut emit_fn = |level: u8, x: u32, y: u32, w: u32, h: u32, bytes: Vec<u8>| {
        emitted.push((level, x, y, w, h, bytes));
        Ok(())
    };
    be_src
        .produce_pass(
            stream_spec_for_band(BandKey::Index(0), PixelFormat::R32Float),
            &ctx,
            &mut emit_fn as TileEmit<'_>,
        )
        .expect("produce_pass R32Float");
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
fn decodes_jpeg_ycbcr_with_custom_coefficients() {
    let harness = DecodeHarness::new();
    let ctx = harness.ctx();

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
    let ycbcr_src =
        open_mem_tiff(ycbcr_tif, &harness.budget, &harness.cancel).expect("open JPEG YCbCr TIFF");
    let display_px = sample_display_rgba(&ycbcr_src, 4, 4, &ctx)
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
}

#[test]
fn rejects_crafted_oversized_compressed_strip_byte_count() {
    let harness = DecodeHarness::new();
    let ctx = harness.ctx();

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
    let bomb_src =
        open_mem_tiff(bomb_tif, &harness.budget, &harness.cancel).expect("open bomb TIFF");
    assert!(matches!(
        sample_display_rgba(&bomb_src, 0, 0, &ctx),
        Err(RasterError::LimitExceeded { .. })
    ));
}
