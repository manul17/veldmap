use std::io::Cursor;

use veldmap_core::io::IoError;

use super::*;

/// `ByteSource` returning at most `max_chunk` bytes per `read_at` call to
/// verify short-read tolerance during header sniffing and decoding.
struct ShortReadSource {
    bytes: Vec<u8>,
    max_chunk: usize,
}

impl ByteSource for ShortReadSource {
    fn len(&self) -> u64 {
        self.bytes.len() as u64
    }

    fn read_at(&self, offset: u64, buf: &mut [u8], cancel: &CancelToken) -> Result<usize, IoError> {
        if cancel.is_cancelled() {
            return Err(IoError::Cancelled);
        }
        let Ok(off) = usize::try_from(offset) else {
            return Ok(0);
        };
        if off >= self.bytes.len() || buf.is_empty() {
            return Ok(0);
        }
        let avail = self.bytes.len() - off;
        let n = buf.len().min(avail).min(self.max_chunk);
        buf[..n].copy_from_slice(&self.bytes[off..off + n]);
        Ok(n)
    }
}

fn tiny_png() -> Vec<u8> {
    let mut out = Vec::new();
    let mut enc = png::Encoder::new(&mut out, 2, 2);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    let mut w = enc.write_header().expect("png header");
    w.write_image_data(&[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120])
        .expect("png data");
    drop(w);
    out
}

fn tiny_tiff() -> Vec<u8> {
    let mut cur = Cursor::new(Vec::new());
    let mut enc = ::tiff::encoder::TiffEncoder::new(&mut cur).expect("tiff encoder");
    enc.write_image::<::tiff::encoder::colortype::RGB8>(
        2,
        2,
        &[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
    )
    .expect("tiff write");
    cur.into_inner()
}

#[test]
fn open_raster_skips_sidecars_for_unsupported_format() {
    let src: Arc<dyn ByteSource> = Arc::new(ShortReadSource {
        bytes: b"not-a-raster-file".to_vec(),
        max_chunk: 2,
    });
    let mut probed = false;
    let mut reader = |_p: &Path, _max: u64| {
        probed = true;
        Ok(None)
    };
    let err = match open_raster(
        src,
        SourceFingerprint::from_digest([1; 32]),
        Some((Path::new("bad.bin"), &mut reader)),
        &MemoryBudget::default(),
        &CancelToken::new(),
    ) {
        Ok(_) => panic!("unsupported format must fail"),
        Err(err) => err,
    };
    assert!(matches!(err, RasterError::UnsupportedFormat));
    assert!(!probed, "sidecars must not be probed on unsupported format");
}

#[test]
fn open_raster_handles_short_reads_and_sidecar_precedence() {
    let budget = MemoryBudget::default();
    let cancel = CancelToken::new();

    // PNG tolerates short reads and returns sidecar errors directly as Binding.
    let png_src: Arc<dyn ByteSource> = Arc::new(ShortReadSource {
        bytes: tiny_png(),
        max_chunk: 2,
    });
    let opened_png = open_raster(
        Arc::clone(&png_src),
        SourceFingerprint::from_digest([2; 32]),
        None,
        &budget,
        &cancel,
    )
    .expect("short-read PNG opens");
    assert_eq!(opened_png.source.description().width, 2);
    assert!(opened_png.ignored_sidecar_error.is_none());

    let mut corrupt_sidecar = |p: &Path, _max: u64| {
        if p.extension().and_then(|e| e.to_str()) == Some("pgw") {
            Ok(Some("not-six-numbers".to_owned()))
        } else {
            Ok(None)
        }
    };
    let png_err = match open_raster(
        png_src,
        SourceFingerprint::from_digest([2; 32]),
        Some((Path::new("scene.png"), &mut corrupt_sidecar)),
        &budget,
        &cancel,
    ) {
        Ok(_) => panic!("corrupt PNG sidecar must fail"),
        Err(err) => err,
    };
    assert!(matches!(png_err, RasterError::Binding(_)));

    // TIFF tolerates short reads and records corrupt sidecar in ignored_sidecar_error.
    let tiff_src: Arc<dyn ByteSource> = Arc::new(ShortReadSource {
        bytes: tiny_tiff(),
        max_chunk: 2,
    });
    let mut corrupt_tfw = |p: &Path, _max: u64| {
        if p.extension().and_then(|e| e.to_str()) == Some("tfw") {
            Ok(Some("bad-world-file".to_owned()))
        } else {
            Ok(None)
        }
    };
    let opened_tiff = open_raster(
        tiff_src,
        SourceFingerprint::from_digest([3; 32]),
        Some((Path::new("scene.tif"), &mut corrupt_tfw)),
        &budget,
        &cancel,
    )
    .expect("TIFF opens despite corrupt sidecar");
    assert_eq!(opened_tiff.source.description().width, 2);
    assert!(matches!(
        opened_tiff.ignored_sidecar_error,
        Some(RasterError::Binding(_))
    ));
}

#[test]
fn open_raster_with_options_applies_external_scale_offset_and_adapters_reject_remote() {
    use veldmap_core::io::testing::MemoryByteSource;

    let budget = MemoryBudget::default();
    let cancel = CancelToken::new();

    // External per-band metadata applied via OpenRasterOptions onto indexed bands.
    let external = [
        ExternalBandMeta {
            scale: Some(0.0001),
            offset: Some(-0.1),
            nodata: Some(0.0),
            unit: Some("reflectance".to_owned()),
        },
        ExternalBandMeta {
            scale: Some(0.0002),
            offset: Some(-0.05),
            nodata: Some(65535.0),
            unit: Some("reflectance".to_owned()),
        },
        ExternalBandMeta {
            scale: Some(0.0003),
            offset: Some(0.0),
            nodata: None,
            unit: Some("dn".to_owned()),
        },
    ];
    let opened = open_raster_with_options(
        Arc::new(MemoryByteSource::from_vec(tiny_tiff())),
        SourceFingerprint::from_digest([4; 32]),
        OpenRasterOptions {
            sidecars: None,
            external_bands: &external,
        },
        &budget,
        &cancel,
    )
    .expect("open TIFF with external per-band metadata");
    let indexed: Vec<_> = opened.source.description().indexed_bands().collect();
    assert_eq!(indexed.len(), 3);
    assert!((indexed[0].scale - 0.0001).abs() < 1e-12);
    assert!((indexed[0].offset - (-0.1)).abs() < 1e-12);
    assert_eq!(indexed[0].nodata, Some(0.0));
    assert_eq!(indexed[0].units.as_deref(), Some("reflectance"));
    assert!(!indexed[0].has_embedded_scale_offset);
    assert!((indexed[1].scale - 0.0002).abs() < 1e-12);
    assert!((indexed[1].offset - (-0.05)).abs() < 1e-12);
    assert_eq!(indexed[1].nodata, Some(65535.0));
    assert!((indexed[2].scale - 0.0003).abs() < 1e-12);
    assert_eq!(indexed[2].nodata, None);
    assert_eq!(indexed[2].units.as_deref(), Some("dn"));

    // PngSource and JpegSource self-enforce RemotePassUnsupported even when opened directly.
    let remote_png: Arc<dyn ByteSource> =
        Arc::new(MemoryByteSource::from_vec(tiny_png()).with_remote(true));
    assert!(matches!(
        PngSource::open(
            remote_png,
            SourceFingerprint::from_digest([5; 32]),
            None,
            &budget,
            &cancel,
        ),
        Err(RasterError::RemotePassUnsupported(_))
    ));

    let mut jpg_bytes = Vec::new();
    image::RgbImage::from_pixel(2, 2, image::Rgb([10, 20, 30]))
        .write_to(&mut Cursor::new(&mut jpg_bytes), image::ImageFormat::Jpeg)
        .expect("jpeg");
    let remote_jpg: Arc<dyn ByteSource> =
        Arc::new(MemoryByteSource::from_vec(jpg_bytes).with_remote(true));
    assert!(matches!(
        JpegSource::open(
            remote_jpg,
            SourceFingerprint::from_digest([6; 32]),
            None,
            &budget,
            &cancel,
        ),
        Err(RasterError::RemotePassUnsupported(_))
    ));
}

#[test]
fn per_band_embedded_scale_offset_precedence_and_nodata_propagation() {
    use ::tiff::encoder::TiffEncoder;
    use ::tiff::tags::Tag;
    use veldmap_core::io::testing::MemoryByteSource;
    use veldmap_core::tile::{BandKey, PixelFormat};

    // 4x2 2-band U16 GeoTIFF:
    // Band 0 embeds explicit SCALE=1.0, OFFSET=0.0, NODATA=65535 in GDAL_METADATA.
    // Band 1 embeds neither scale/offset nor nodata, so external_bands[1] applies.
    // Pixels (x=0..4, y=0..2):
    //   x=0: b0=65535 (nodata), b1=0 (valid 0.0 reflectance after external scale/offset!)
    //   x=1: b0=0 (valid 0!),   b1=9999 (external nodata!)
    //   x=2: b0=100,            b1=5000 (physical 5000*0.0001 - 0.1 = 0.4)
    //   x=3: b0=200,            b1=2000 (physical 2000*0.0001 - 0.1 = 0.1)
    let mut raw_u16 = Vec::with_capacity(4 * 2 * 2);
    for _y in 0..2 {
        raw_u16.extend_from_slice(&[65535u16, 0, 0, 9999, 100, 5000, 200, 2000]);
    }
    let gdal_xml = "<GDALMetadata>\
        <Item name=\"SCALE\" sample=\"0\">1.0</Item>\
        <Item name=\"OFFSET\" sample=\"0\">0.0</Item>\
        <Item name=\"NODATA\" sample=\"0\">65535</Item>\
    </GDALMetadata>";

    let mut cur = Cursor::new(Vec::new());
    {
        let mut enc = TiffEncoder::new(&mut cur).expect("tiff encoder");
        let mut dir = enc.image_directory().expect("ifd");
        dir.write_tag(Tag::ImageWidth, 4u32).expect("w");
        dir.write_tag(Tag::ImageLength, 2u32).expect("h");
        dir.write_tag(Tag::BitsPerSample, &[16u16, 16][..])
            .expect("bps");
        dir.write_tag(Tag::Compression, 1u16).expect("comp");
        dir.write_tag(Tag::PhotometricInterpretation, 1u16)
            .expect("photo");
        dir.write_tag(Tag::SamplesPerPixel, 2u16).expect("spp");
        dir.write_tag(Tag::PlanarConfiguration, 1u16)
            .expect("planar");
        dir.write_tag(Tag::RowsPerStrip, 2u32).expect("rps");
        let mut bytes = Vec::with_capacity(raw_u16.len() * 2);
        for v in &raw_u16 {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let off = dir.write_data(&bytes[..]).expect("strip data");
        dir.write_tag(Tag::StripOffsets, &[off as u32][..])
            .expect("offsets");
        dir.write_tag(Tag::StripByteCounts, &[bytes.len() as u32][..])
            .expect("counts");
        dir.write_tag(Tag::Unknown(42112), gdal_xml)
            .expect("gdal metadata");
        dir.finish().expect("finish");
    }

    let external = [
        ExternalBandMeta {
            scale: Some(0.0001),
            offset: Some(-0.1),
            nodata: Some(0.0),
            unit: Some("should-apply-unit-only".to_owned()),
        },
        ExternalBandMeta {
            scale: Some(0.0001),
            offset: Some(-0.1),
            nodata: Some(9999.0),
            unit: Some("reflectance".to_owned()),
        },
    ];
    let budget = MemoryBudget::default();
    let cancel = CancelToken::new();
    let opened = open_raster_with_options(
        Arc::new(MemoryByteSource::from_vec(cur.into_inner())),
        SourceFingerprint::from_digest([7; 32]),
        OpenRasterOptions {
            sidecars: None,
            external_bands: &external,
        },
        &budget,
        &cancel,
    )
    .expect("open 2-band tiff");

    let bands: Vec<_> = opened.source.description().indexed_bands().collect();
    assert_eq!(bands.len(), 2);
    // Band 0 embedded (1.0, 0.0) & nodata=65535 MUST NOT be overridden by external_bands[0].
    assert!(bands[0].has_embedded_scale_offset);
    assert!((bands[0].scale - 1.0).abs() < 1e-12);
    assert!(bands[0].offset.abs() < 1e-12);
    assert_eq!(bands[0].nodata, Some(65535.0));
    assert_eq!(bands[0].units.as_deref(), Some("should-apply-unit-only"));

    // Band 1 had no embedded metadata, so external_bands[1] applies cleanly.
    assert!(!bands[1].has_embedded_scale_offset);
    assert!((bands[1].scale - 0.0001).abs() < 1e-12);
    assert!((bands[1].offset - (-0.1)).abs() < 1e-12);
    assert_eq!(bands[1].nodata, Some(9999.0));
    assert_eq!(bands[1].units.as_deref(), Some("reflectance"));

    // Native f64 sample_band respects per-band nodata & physical scaling while preserving valid 0.
    let chunk_cache = crate::SharedChunkCache::new(budget.decoded_chunk_cache_bytes);
    let budget_tracker =
        veldmap_core::budget::DecodeBudgetTracker::new(budget.decode_working_bytes);
    let ctx = crate::DecodeContext {
        chunk_cache: &chunk_cache,
        budget_tracker: &budget_tracker,
        budget: &budget,
        cancel: &cancel,
    };

    let s0_nodata = opened
        .source
        .sample_band(BandKey::Index(0), PixelFormat::R32Float, 0, 0, &ctx)
        .expect("sample b0 nodata");
    assert_eq!(
        s0_nodata,
        crate::BandSample::Nodata,
        "band 0 x=0 (65535) must be nodata"
    );
    let s0_zero = opened
        .source
        .sample_band(BandKey::Index(0), PixelFormat::R32Float, 1, 0, &ctx)
        .expect("sample b0 zero");
    assert_eq!(
        s0_zero,
        crate::BandSample::Scalar {
            raw: 0.0,
            physical: 0.0,
        }
    );

    let s1_nodata = opened
        .source
        .sample_band(BandKey::Index(1), PixelFormat::R32Float, 1, 0, &ctx)
        .expect("sample b1 nodata");
    assert_eq!(
        s1_nodata,
        crate::BandSample::Nodata,
        "band 1 x=1 (9999) must be external nodata"
    );
    let s1_valid = opened
        .source
        .sample_band(BandKey::Index(1), PixelFormat::R32Float, 2, 0, &ctx)
        .expect("sample b1 valid");
    match s1_valid {
        crate::BandSample::Scalar { raw, physical } => {
            assert_eq!(raw, 5000.0);
            assert!((physical - 0.4).abs() < 1e-9);
        }
        other => panic!("expected Scalar for band 1 x=2, got {other:?}"),
    }

    // Pointwise R32Float decode maps per-band nodata to NaN and preserves valid 0.0.
    let spec1 = crate::stream_spec_for_band(BandKey::Index(1), PixelFormat::R32Float);
    let tile1 = opened
        .source
        .decode_pointwise_tile(spec1, 0, 0, 0, &ctx)
        .expect("decode b1 tile");
    let floats1: Vec<f32> = tile1
        .data()
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(floats1[0], 0.0, "band 1 x=0 valid 0.0 must not be NaN");
    assert!(
        floats1[1].is_nan(),
        "band 1 x=1 external nodata=9999 must decode to NaN"
    );
    assert_eq!(floats1[2], 5000.0);

    // band_stats excludes per-band nodata cleanly and reports physical bounds.
    let stats0 = opened
        .source
        .band_stats(BandKey::Index(0), &ctx)
        .expect("stats0")
        .expect("stats0 present");
    assert_eq!(stats0.min, 0.0);
    assert_eq!(stats0.max, 200.0);
    let stats1 = opened
        .source
        .band_stats(BandKey::Index(1), &ctx)
        .expect("stats1")
        .expect("stats1 present");
    assert!((stats1.min - (-0.1)).abs() < 1e-9);
    assert!((stats1.max - 0.4).abs() < 1e-9);
}

#[test]
fn uniform_external_display_nodata_reaches_display_without_polluting_extra_bands() {
    use std::io::Cursor;
    use tiff::encoder::TiffEncoder;
    use tiff::tags::Tag;
    use veldmap_core::io::testing::MemoryByteSource;
    use veldmap_core::tile::{BandKey, PixelFormat, SourceFingerprint};

    let mut cur = Cursor::new(Vec::new());
    {
        let mut enc = TiffEncoder::new(&mut cur).expect("encoder");
        let mut dir = enc.image_directory().expect("dir");
        dir.write_tag(Tag::ImageWidth, 4u32).unwrap();
        dir.write_tag(Tag::ImageLength, 4u32).unwrap();
        dir.write_tag(Tag::BitsPerSample, &[16u16, 16, 16, 16][..])
            .unwrap();
        dir.write_tag(Tag::Compression, 1u16).unwrap();
        dir.write_tag(Tag::PhotometricInterpretation, 2u16).unwrap();
        dir.write_tag(Tag::SamplesPerPixel, 4u16).unwrap();
        dir.write_tag(Tag::ExtraSamples, &[0u16][..]).unwrap();
        dir.write_tag(Tag::PlanarConfiguration, 1u16).unwrap();
        dir.write_tag(Tag::RowsPerStrip, 4u32).unwrap();
        let data = [0u8; 4 * 4 * 4 * 2];
        let off = dir.write_data(&data[..]).unwrap();
        dir.write_tag(Tag::StripOffsets, &[off as u32][..]).unwrap();
        dir.write_tag(Tag::StripByteCounts, &[data.len() as u32][..])
            .unwrap();
        dir.finish().unwrap();
    }

    // RGB color bands declare nodata = 0; NIR (band 3) intentionally has no nodata.
    let ext = [
        ExternalBandMeta {
            nodata: Some(0.0),
            ..Default::default()
        },
        ExternalBandMeta {
            nodata: Some(0.0),
            ..Default::default()
        },
        ExternalBandMeta {
            nodata: Some(0.0),
            ..Default::default()
        },
        ExternalBandMeta {
            nodata: None,
            ..Default::default()
        },
    ];
    let budget = MemoryBudget::default();
    let cancel = CancelToken::new();
    let opened = open_raster_with_options(
        Arc::new(MemoryByteSource::from_vec(cur.into_inner())),
        SourceFingerprint::from_digest([9; 32]),
        OpenRasterOptions {
            sidecars: None,
            external_bands: &ext,
        },
        &budget,
        &cancel,
    )
    .expect("open rgbnir");
    let desc = opened.source.description();
    assert_eq!(
        desc.band_info(BandKey::Display).and_then(|b| b.nodata),
        Some(0.0),
        "uniform RGB external nodata must propagate to BandKey::Display"
    );
    assert_eq!(
        desc.band_info(BandKey::Index(3)).and_then(|b| b.nodata),
        None,
        "display nodata must not leak into NIR band 3"
    );

    let chunk_cache = crate::SharedChunkCache::new(budget.decoded_chunk_cache_bytes);
    let budget_tracker =
        veldmap_core::budget::DecodeBudgetTracker::new(budget.decode_working_bytes);
    let ctx = crate::DecodeContext {
        chunk_cache: &chunk_cache,
        budget_tracker: &budget_tracker,
        budget: &budget,
        cancel: &cancel,
    };
    // Band 3 (NIR) with value 0 must remain valid scalar 0.0, not nodata.
    let s3 = opened
        .source
        .sample_band(BandKey::Index(3), PixelFormat::R32Float, 0, 0, &ctx)
        .expect("sample nir");
    assert!(
        matches!(s3, crate::BandSample::Scalar { raw, .. } if raw == 0.0),
        "NIR sample 0 must not inherit RGB display nodata, got {s3:?}"
    );
    // Display decode treats [0, 0, 0] as transparent nodata.
    let rgba = opened
        .source
        .sample_band(BandKey::Display, PixelFormat::Rgba8UnormSrgb, 0, 0, &ctx)
        .expect("sample display");
    assert_eq!(rgba, crate::BandSample::Nodata);
}
