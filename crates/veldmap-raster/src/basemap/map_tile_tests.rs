use image::{ImageBuffer, ImageFormat, Luma, LumaA, Rgb, Rgba};

use super::*;

#[test]
fn decodes_valid_png_and_jpeg_tiles_and_rejects_invalid_inputs() {
    let img: ImageBuffer<Rgba<u8>, Vec<u8>> =
        ImageBuffer::from_fn(4, 4, |x, y| Rgba([x as u8 * 40, y as u8 * 40, 128, 255]));

    let mut png_bytes = Vec::new();
    img.write_to(&mut Cursor::new(&mut png_bytes), ImageFormat::Png)
        .expect("encode png");
    let decoded_png = decode_map_tile(&png_bytes).expect("decode png");
    assert_eq!((decoded_png.width(), decoded_png.height()), (4, 4));
    assert_eq!(decoded_png.format(), PixelFormat::Rgba8UnormSrgb);
    assert_eq!(decoded_png.byte_size(), 4 * 4 * 4);
    assert_eq!(&decoded_png.data()[0..4], &[0, 0, 128, 255]);

    // Grayscale PNG converts cleanly to opaque RGBA8 and RGB8.
    let gray: ImageBuffer<Luma<u8>, Vec<u8>> = ImageBuffer::from_pixel(4, 4, Luma([180]));
    let mut gray_png = Vec::new();
    gray.write_to(&mut Cursor::new(&mut gray_png), ImageFormat::Png)
        .expect("encode gray png");
    let decoded_gray = decode_map_tile(&gray_png).expect("decode gray png");
    assert_eq!(&decoded_gray.data()[0..4], &[180, 180, 180, 255]);
    let (_, _, gray_rgb) =
        decode_rgb8_limited(&gray_png, MAX_MAP_TILE_DIM, MAX_MAP_TILE_ALLOC_BYTES)
            .expect("decode gray png to rgb8");
    assert_eq!(&gray_rgb[0..3], &[180, 180, 180]);

    // GrayscaleAlpha PNG preserves alpha in RGBA8 and drops alpha in RGB8.
    let gray_a: ImageBuffer<LumaA<u8>, Vec<u8>> = ImageBuffer::from_pixel(2, 2, LumaA([90, 128]));
    let mut gray_a_png = Vec::new();
    gray_a
        .write_to(&mut Cursor::new(&mut gray_a_png), ImageFormat::Png)
        .expect("encode gray+alpha png");
    let decoded_ga = decode_map_tile(&gray_a_png).expect("decode gray+alpha");
    assert_eq!(&decoded_ga.data()[0..4], &[90, 90, 90, 128]);
    let (_, _, ga_rgb) =
        decode_rgb8_limited(&gray_a_png, MAX_MAP_TILE_DIM, MAX_MAP_TILE_ALLOC_BYTES)
            .expect("decode gray+alpha to rgb8");
    assert_eq!(&ga_rgb[0..3], &[90, 90, 90]);

    // Solid-color RGB JPEG preserves color within JPEG quantization tolerance.
    let rgb: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_pixel(8, 8, Rgb([200, 80, 40]));
    let mut jpg_bytes = Vec::new();
    rgb.write_to(&mut Cursor::new(&mut jpg_bytes), ImageFormat::Jpeg)
        .expect("encode jpeg");
    let decoded_jpg = decode_map_tile(&jpg_bytes).expect("decode jpeg");
    assert_eq!((decoded_jpg.width(), decoded_jpg.height()), (8, 8));
    assert_eq!(decoded_jpg.byte_size(), 8 * 8 * 4);
    assert!(decoded_jpg.data()[0].abs_diff(200) <= 6);
    assert!(decoded_jpg.data()[1].abs_diff(80) <= 6);
    assert!(decoded_jpg.data()[2].abs_diff(40) <= 6);
    assert_eq!(decoded_jpg.data()[3], 255);

    // Corrupt / unsupported / truncated payloads fail cleanly.
    assert!(matches!(
        decode_map_tile(b"not an image"),
        Err(RasterError::UnsupportedFormat)
    ));
    assert!(matches!(
        decode_map_tile(&[]),
        Err(RasterError::UnsupportedFormat)
    ));
    assert!(matches!(
        decode_map_tile(b"GIF89a\x01\x00\x01\x00\x00\x00\x00;"),
        Err(RasterError::UnsupportedFormat)
    ));
    assert!(matches!(
        decode_map_tile(&png_bytes[..png_bytes.len() / 2]),
        Err(RasterError::Decode(_))
    ));

    // Explicit dimension and peak-allocation limits are enforced before full decode.
    assert!(matches!(
        decode_rgba8_limited(&png_bytes, 2, MAX_MAP_TILE_ALLOC_BYTES),
        Err(RasterError::LimitExceeded { .. })
    ));
    // 4x4 RGBA8 PNG (64 B) rejected when budget < 64 B.
    assert!(matches!(
        decode_rgba8_limited(&png_bytes, MAX_MAP_TILE_DIM, 63),
        Err(RasterError::LimitExceeded { .. })
    ));
    // 8x8 RGB8 JPEG (192 B src + 256 B dst = 448 B peak) rejected at 400 B.
    assert!(matches!(
        decode_rgba8_limited(&jpg_bytes, MAX_MAP_TILE_DIM, 400),
        Err(RasterError::LimitExceeded { .. })
    ));
    // 4x4 RGBA8 PNG converted to RGB8 (64 B src + 48 B dst = 112 B peak) rejected at 100 B.
    assert!(matches!(
        decode_rgb8_limited(&png_bytes, MAX_MAP_TILE_DIM, 100),
        Err(RasterError::LimitExceeded { .. })
    ));
    // Concise Debug formatting omits raw pixel dumps.
    let dbg = format!("{decoded_png:?}");
    assert!(dbg.contains("byte_size: 64"), "unexpected Debug: {dbg}");
}

#[test]
fn decodes_indexed_trns_16bit_and_adam7_pngs() {
    // 1. Indexed palette PNG with tRNS transparency chunk.
    let mut pal_bytes = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut pal_bytes, 2, 1);
        enc.set_color(png::ColorType::Indexed);
        enc.set_depth(png::BitDepth::Eight);
        enc.set_palette(vec![10, 20, 30, 200, 100, 50]);
        enc.set_trns(vec![0, 255]);
        let mut writer = enc.write_header().expect("write indexed header");
        writer.write_image_data(&[0, 1]).expect("write indexed");
    }
    let decoded_pal = decode_map_tile(&pal_bytes).expect("decode indexed+tRNS");
    assert_eq!(decoded_pal.data(), &[10, 20, 30, 0, 200, 100, 50, 255]);

    // 2. 16-bit RGB PNG normalized to 8-bit RGBA8.
    let mut rgb16_bytes = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut rgb16_bytes, 1, 1);
        enc.set_color(png::ColorType::Rgb);
        enc.set_depth(png::BitDepth::Sixteen);
        let mut writer = enc.write_header().expect("write 16-bit header");
        writer
            .write_image_data(&[0x80, 0x00, 0xFF, 0xFF, 0x00, 0x00])
            .expect("write 16-bit");
    }
    let decoded_16 = decode_map_tile(&rgb16_bytes).expect("decode 16-bit png");
    assert_eq!(decoded_16.data(), &[0x80, 0xFF, 0x00, 255]);

    // 3. 1-bit grayscale PNG normalized to 8-bit RGBA8.
    let mut g1_bytes = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut g1_bytes, 2, 1);
        enc.set_color(png::ColorType::Grayscale);
        enc.set_depth(png::BitDepth::One);
        let mut writer = enc.write_header().expect("write 1-bit header");
        writer
            .write_image_data(&[0b1000_0000])
            .expect("write 1-bit");
    }
    let decoded_1 = decode_map_tile(&g1_bytes).expect("decode 1-bit png");
    assert_eq!(decoded_1.data(), &[255, 255, 255, 255, 0, 0, 0, 255]);

    // 4. Multi-pass Adam7 interlaced 2x2 RGB PNG (passes 1, 6, and 7).
    let mut adam7_info = png::Info::with_size(2, 2);
    adam7_info.color_type = png::ColorType::Rgb;
    adam7_info.bit_depth = png::BitDepth::Eight;
    adam7_info.interlaced = true;
    let adam7_scanlines: [u8; 15] = [
        0, 10, 20, 30, // pass 1: (0, 0)
        0, 40, 50, 60, // pass 6: (1, 0)
        0, 70, 80, 90, 100, 110, 120, // pass 7: (0, 1), (1, 1)
    ];
    let mut zlib = vec![0x78, 0x01, 0x01];
    let len = adam7_scanlines.len() as u16;
    zlib.extend_from_slice(&len.to_le_bytes());
    zlib.extend_from_slice(&(!len).to_le_bytes());
    zlib.extend_from_slice(&adam7_scanlines);
    let (mut s1, mut s2) = (1u32, 0u32);
    for &b in &adam7_scanlines {
        s1 = (s1 + u32::from(b)) % 65_521;
        s2 = (s2 + s1) % 65_521;
    }
    zlib.extend_from_slice(&((s2 << 16) | s1).to_be_bytes());
    let mut adam7_bytes = Vec::new();
    {
        let enc = png::Encoder::with_info(&mut adam7_bytes, adam7_info).expect("adam7 enc");
        let mut writer = enc.write_header().expect("adam7 header");
        writer
            .write_chunk(png::chunk::IDAT, &zlib)
            .expect("write adam7 IDAT");
        writer.finish().expect("finish adam7");
    }
    let decoded_adam7 = decode_map_tile(&adam7_bytes).expect("decode adam7 png");
    assert_eq!((decoded_adam7.width(), decoded_adam7.height()), (2, 2));
    assert_eq!(
        decoded_adam7.data(),
        &[
            10, 20, 30, 255, // (0, 0)
            40, 50, 60, 255, // (1, 0)
            70, 80, 90, 255, // (0, 1)
            100, 110, 120, 255, // (1, 1)
        ]
    );
}

#[test]
fn jpeg_handles_grayscale_rgb8_in_place_truncation_and_progressive_budget() {
    // 1. Grayscale (L8) JPEG decodes to opaque RGBA8 and RGB8.
    let gray_jpg_img: ImageBuffer<Luma<u8>, Vec<u8>> = ImageBuffer::from_pixel(8, 8, Luma([140]));
    let mut gray_jpg = Vec::new();
    gray_jpg_img
        .write_to(&mut Cursor::new(&mut gray_jpg), ImageFormat::Jpeg)
        .expect("encode grayscale jpeg");
    let decoded_gray = decode_map_tile(&gray_jpg).expect("decode grayscale jpeg");
    assert_eq!((decoded_gray.width(), decoded_gray.height()), (8, 8));
    assert!(decoded_gray.data()[0].abs_diff(140) <= 6);
    assert_eq!(decoded_gray.data()[0], decoded_gray.data()[1]);
    assert_eq!(decoded_gray.data()[1], decoded_gray.data()[2]);
    assert_eq!(decoded_gray.data()[3], 255);

    let (gw, gh, gray_rgb) =
        decode_rgb8_limited(&gray_jpg, MAX_MAP_TILE_DIM, MAX_MAP_TILE_ALLOC_BYTES)
            .expect("decode grayscale jpeg to rgb8");
    assert_eq!((gw, gh, gray_rgb.len()), (8, 8, 8 * 8 * 3));
    assert!(gray_rgb[0].abs_diff(140) <= 6);
    assert_eq!(gray_rgb[0], gray_rgb[1]);
    assert_eq!(gray_rgb[1], gray_rgb[2]);

    // 2. RGB JPEG decodes directly into RGB8 via the in-place RGB24 path
    //    (8x8 RGB24 = 192 B peak; no extra conversion buffer needed).
    let rgb_img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_pixel(8, 8, Rgb([60, 160, 220]));
    let mut rgb_jpg = Vec::new();
    rgb_img
        .write_to(&mut Cursor::new(&mut rgb_jpg), ImageFormat::Jpeg)
        .expect("encode rgb jpeg");
    let (rw, rh, rgb_out) =
        decode_rgb8_limited(&rgb_jpg, MAX_MAP_TILE_DIM, 192).expect("in-place rgb8 decode");
    assert_eq!((rw, rh, rgb_out.len()), (8, 8, 192));
    assert!(rgb_out[0].abs_diff(60) <= 6);
    assert!(rgb_out[1].abs_diff(160) <= 6);
    assert!(rgb_out[2].abs_diff(220) <= 6);

    // 3. Truncated JPEG fails cleanly with Decode error.
    assert!(matches!(
        decode_map_tile(&rgb_jpg[..rgb_jpg.len() / 2]),
        Err(RasterError::Decode(_))
    ));

    // 4. Progressive (SOF2) JPEG accounts for full-res i16 DCT coefficient
    //    working memory before decode (8x8 RGB: 192 B src + 384 B coeffs = 576 B
    //    > 448 B which would fit sequential 192 B src + 256 B RGBA dst).
    let mut prog_jpg = rgb_jpg.clone();
    let sof0_pos = prog_jpg
        .windows(2)
        .position(|w| w == [0xFF, 0xC0])
        .expect("SOF0 marker in baseline JPEG");
    prog_jpg[sof0_pos + 1] = 0xC2; // SOF2 (progressive DCT)
    assert!(matches!(
        decode_rgba8_limited(&prog_jpg, MAX_MAP_TILE_DIM, 448),
        Err(RasterError::LimitExceeded {
            width: 8,
            height: 8,
            estimated_bytes: 576,
            limit_bytes: 448,
        })
    ));
}
