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

    // External scale/offset applied via OpenRasterOptions onto indexed bands.
    let opened = open_raster_with_options(
        Arc::new(MemoryByteSource::from_vec(tiny_tiff())),
        SourceFingerprint::from_digest([4; 32]),
        OpenRasterOptions {
            sidecars: None,
            external_scale_offset: Some((Some(0.0001), Some(-0.1))),
        },
        &budget,
        &cancel,
    )
    .expect("open TIFF with external scale/offset");
    let indexed: Vec<_> = opened.source.description().indexed_bands().collect();
    assert!(!indexed.is_empty());
    for band in indexed {
        assert!((band.scale - 0.0001).abs() < 1e-12);
        assert!((band.offset - (-0.1)).abs() < 1e-12);
    }

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
