//! Integration tests for TIFF pixel layouts, sample types, photometric
//! interpretations, planar configurations, palette LUTs, RGBN bands with
//! GDAL scale/offset metadata, and sparse chunks.

use std::io::Cursor;

use ::tiff::encoder::TiffEncoder;
use ::tiff::tags::Tag;
use veldmap_core::budget::MemoryBudget;
use veldmap_core::geo::{CrsSpec, PULKOVO_1942};
use veldmap_core::sync::CancelToken;
use veldmap_core::tile::{BandKey, PixelFormat};
use veldmap_raster::{
    BandSample, DISPLAY_STREAM_SPEC, RasterSource, generate_tile_mipmaps, stream_spec_for_band,
};

use crate::common::{
    DecodeHarness, TiffIfdHeader, open_mem_tiff, sample_display_rgba, write_ifd_header,
    write_single_strip_u32,
};

#[test]
fn decodes_geographic_pulkovo42_i16_with_gdal_nodata() {
    let harness = DecodeHarness::new();
    let ctx = harness.ctx();

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
    let i16_src = open_mem_tiff(i16_buf.into_inner(), &harness.budget, &harness.cancel)
        .expect("open geographic i16 TIFF");
    assert!(matches!(
        i16_src.description().geographic_crs,
        Some(CrsSpec::Geographic { datum, .. }) if datum == PULKOVO_1942
    ));
    assert_eq!(i16_src.description().ties.len(), 4);
    assert!((i16_src.description().ties[3].lon - 12.0).abs() < 1e-9);
    assert!((i16_src.description().ties[3].lat - 49.5).abs() < 1e-9);
    let i16_tile = i16_src
        .decode_pointwise_tile(DISPLAY_STREAM_SPEC, 0, 0, 0, &ctx)
        .expect("decode i16");
    assert_eq!(i16_tile.data()[3], 0, "i16 nodata must be transparent");
    assert_eq!(i16_tile.data()[7], 255);
    assert!(i16_tile.data()[4] < i16_tile.data()[12]);
    assert!(i16_tile.data()[12] < i16_tile.data()[7 * 4]);
}

#[test]
fn decodes_gray_alpha_preserving_transparency() {
    let harness = DecodeHarness::new();
    let ctx = harness.ctx();

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
    let ga_src = open_mem_tiff(ga_buf.into_inner(), &harness.budget, &harness.cancel)
        .expect("open Gray+Alpha");
    assert!(ga_src.description().bands[0].has_alpha);
    let ga_tile = ga_src
        .decode_pointwise_tile(DISPLAY_STREAM_SPEC, 0, 0, 0, &ctx)
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
    assert!(open_mem_tiff(planar_buf.into_inner(), &budget, &cancel).is_err());
}

#[test]
fn decodes_planar2_npot_tiled_rgb_and_single_band_scientific() {
    let harness = DecodeHarness::new();
    let ctx = harness.ctx();

    // Tiled PlanarConfiguration = 2 with NPOT dimensions (20x18, tile 16x16):
    // 2x2 = 4 tiles per plane * 3 planes = 12 tile offsets.
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
    let planar_src = open_mem_tiff(planar_buf.into_inner(), &harness.budget, &harness.cancel)
        .expect("open tiled Planar-2 NPOT TIFF");
    let display_tile = planar_src
        .decode_pointwise_tile(DISPLAY_STREAM_SPEC, 0, 0, 0, &ctx)
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
        .decode_pointwise_tile(sci_spec, 0, 0, 0, &ctx)
        .expect("decode Planar-2 Band 2 R32Float");
    generate_tile_mipmaps(&mut sci_tile);
    assert!(sci_tile.mip_level_count() > 1);
    let f0 = f32::from_ne_bytes(sci_tile.data()[0..4].try_into().expect("4B"));
    assert_eq!(f0, 240.0);
}

#[test]
fn decodes_palette_tiff_display_and_categorical_indices() {
    let harness = DecodeHarness::new();
    let ctx = harness.ctx();

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
    let pal_src = open_mem_tiff(pal_buf.into_inner(), &harness.budget, &harness.cancel)
        .expect("open Palette TIFF");
    assert_eq!(pal_src.default_stream_spec().format, PixelFormat::R8Uint);
    assert!(pal_src.color_map().is_some());
    let pal_disp = pal_src
        .decode_pointwise_tile(DISPLAY_STREAM_SPEC, 0, 0, 0, &ctx)
        .expect("decode Palette Display");
    assert_eq!(
        pal_disp.data()[3],
        0,
        "nodata 0 must stay transparent (alpha=0)"
    );
    assert_eq!(&pal_disp.data()[4..8], &[255, 0, 0, 255]); // class 1 red
    assert_eq!(&pal_disp.data()[8..12], &[0, 255, 0, 255]); // class 2 green

    let pal_cat = pal_src
        .decode_pointwise_tile(pal_src.default_stream_spec(), 0, 0, 0, &ctx)
        .expect("decode Palette Categorical");
    assert_eq!(pal_cat.format(), PixelFormat::R8Uint);
    assert_eq!(pal_cat.data(), &[0, 1, 2, 1, 2, 1, 0, 2]);
    assert_eq!(
        pal_src
            .sample_band(BandKey::Index(0), PixelFormat::R8Uint, 2, 0, &ctx)
            .expect("sample palette"),
        BandSample::Categorical(2)
    );
}

#[test]
fn decodes_4band_rgbn_with_unspecified_extra_samples_and_gdal_scale_offset() {
    let harness = DecodeHarness::new();
    let ctx = harness.ctx();

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
    let rgbn_src = open_mem_tiff(rgbn_buf.into_inner(), &harness.budget, &harness.cancel)
        .expect("open 4-band RGBN TIFF");
    assert!(!rgbn_src.description().bands[0].has_alpha);
    assert_eq!(rgbn_src.description().indexed_bands().count(), 4);
    let rgbn_disp = rgbn_src
        .decode_pointwise_tile(DISPLAY_STREAM_SPEC, 0, 0, 0, &ctx)
        .expect("decode RGBN Display");
    assert_eq!(
        rgbn_disp.data()[3],
        255,
        "ExtraSamples=0 NIR=0 must stay opaque on Display"
    );
    let nir_spec = stream_spec_for_band(BandKey::Index(3), PixelFormat::R32Float);
    let nir_tile = rgbn_src
        .decode_pointwise_tile(nir_spec, 0, 0, 0, &ctx)
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
fn decodes_white_is_zero_with_nodata_inversion_and_raw_scientific() {
    let harness = DecodeHarness::new();
    let ctx = harness.ctx();

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
    let wiz_src =
        open_mem_tiff(wiz_tif, &harness.budget, &harness.cancel).expect("open WhiteIsZero");
    assert_eq!(
        sample_display_rgba(&wiz_src, 0, 0, &ctx).expect("nodata"),
        None,
        "nodata 0 stays transparent"
    );
    assert_eq!(
        sample_display_rgba(&wiz_src, 1, 0, &ctx).expect("px1"),
        Some([245, 245, 245, 255])
    );
    assert_eq!(
        sample_display_rgba(&wiz_src, 3, 0, &ctx).expect("px3"),
        Some([0, 0, 0, 255])
    );
    assert_eq!(
        wiz_src
            .sample_band(BandKey::Index(0), PixelFormat::R32Float, 0, 0, &ctx)
            .expect("sci nodata"),
        BandSample::Nodata
    );
    assert_eq!(
        wiz_src
            .sample_band(BandKey::Index(0), PixelFormat::R32Float, 1, 0, &ctx)
            .expect("sci px1"),
        BandSample::Scalar {
            raw: 10.0,
            physical: 10.0
        }
    );
}

#[test]
fn decodes_signed_i8_stretching_negative_values() {
    let harness = DecodeHarness::new();
    let ctx = harness.ctx();

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
    let i8_src = open_mem_tiff(i8_tif, &harness.budget, &harness.cancel).expect("open I8 TIFF");
    let px_neg50 = sample_display_rgba(&i8_src, 1, 0, &ctx)
        .expect("i8 sample")
        .expect("opaque");
    assert!(
        px_neg50[0] > 40 && px_neg50[0] < 90,
        "I8 value -50 over [-100, 100] should stretch to ~64, got {px_neg50:?}"
    );
}

#[test]
fn decodes_sparse_chunk_as_transparent_display_and_nodata_scientific() {
    let harness = DecodeHarness::new();
    let ctx = harness.ctx();

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
    let sparse_src =
        open_mem_tiff(sparse_tif, &harness.budget, &harness.cancel).expect("open sparse TIFF");
    assert_eq!(
        sample_display_rgba(&sparse_src, 1, 1, &ctx).expect("sparse display"),
        None
    );
    assert_eq!(
        sparse_src
            .sample_band(BandKey::Index(0), PixelFormat::R32Float, 1, 1, &ctx)
            .expect("sparse sci"),
        BandSample::Nodata
    );
}
