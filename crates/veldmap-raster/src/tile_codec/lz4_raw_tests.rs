use super::super::MAX_CACHED_TILE_DECODED_BYTES;
use super::*;

#[test]
fn lz4_raw_round_trips_scientific_and_categorical_formats() {
    // R32Float including NaN bit patterns and smooth gradients (high shuffle compression).
    let (w, h) = (32u32, 24u32);
    let mut f32_bytes = Vec::with_capacity((w * h * 4) as usize);
    for i in 0..(w * h) {
        let v = if i % 7 == 0 {
            f32::NAN
        } else {
            100.0 + (i as f32) * 0.25
        };
        f32_bytes.extend_from_slice(&v.to_ne_bytes());
    }
    let tile = DecodedTile::try_new(w, h, PixelFormat::R32Float, f32_bytes.clone()).expect("tile");
    let blob = encode_lz4_raw_tile(&tile, MAX_CACHED_TILE_DECODED_BYTES).expect("encode f32");
    assert!(
        blob.len() < f32_bytes.len(),
        "byte-shuffled LZ4 should compress smooth f32 data"
    );
    let decoded = decode_lz4_raw_tile(&blob, PixelFormat::R32Float, MAX_CACHED_TILE_DECODED_BYTES)
        .expect("decode f32");
    assert_eq!(decoded.width(), w);
    assert_eq!(decoded.height(), h);
    assert_eq!(decoded.valid_w(), w);
    assert_eq!(decoded.valid_h(), h);
    assert_eq!(decoded.data(), f32_bytes.as_slice());

    // R8Uint and R16Uint with gutter / partial valid region.
    let u8_tile = DecodedTile::try_with_valid_region(
        18,
        18,
        16,
        16,
        1,
        PixelFormat::R8Uint,
        vec![7u8; 18 * 18],
    )
    .expect("u8 tile");
    let u8_blob = encode_lz4_raw_tile(&u8_tile, MAX_CACHED_TILE_DECODED_BYTES).expect("encode u8");
    let u8_dec = decode_lz4_raw_tile(&u8_blob, PixelFormat::R8Uint, MAX_CACHED_TILE_DECODED_BYTES)
        .expect("decode u8");
    assert_eq!(
        (u8_dec.valid_w(), u8_dec.valid_h(), u8_dec.gutter()),
        (16, 16, 1)
    );
    assert_eq!(u8_dec.data(), u8_tile.data());

    let mut u16_bytes = Vec::with_capacity(16 * 16 * 2);
    for i in 0u16..(16 * 16) {
        u16_bytes.extend_from_slice(&(i * 3).to_ne_bytes());
    }
    let u16_tile =
        DecodedTile::try_new(16, 16, PixelFormat::R16Uint, u16_bytes.clone()).expect("u16");
    let u16_blob =
        encode_lz4_raw_tile(&u16_tile, MAX_CACHED_TILE_DECODED_BYTES).expect("encode u16");
    let u16_dec = decode_lz4_raw_tile(
        &u16_blob,
        PixelFormat::R16Uint,
        MAX_CACHED_TILE_DECODED_BYTES,
    )
    .expect("decode u16");
    assert_eq!(u16_dec.data(), u16_bytes.as_slice());
}

#[test]
fn lz4_raw_enforces_header_bounds_format_and_corruption_guards() {
    let rgba_tile =
        DecodedTile::try_new(4, 4, PixelFormat::Rgba8UnormSrgb, vec![0u8; 64]).expect("rgba");
    assert!(matches!(
        encode_lz4_raw_tile(&rgba_tile, MAX_CACHED_TILE_DECODED_BYTES),
        Err(RasterError::Encode(_))
    ));

    let small = DecodedTile::try_new(8, 8, PixelFormat::R32Float, vec![0u8; 256]).expect("ok");
    assert!(matches!(
        encode_lz4_raw_tile(&small, 128),
        Err(RasterError::LimitExceeded { .. })
    ));
    let blob = encode_lz4_raw_tile(&small, MAX_CACHED_TILE_DECODED_BYTES).expect("encode");

    // Format mismatch rejected.
    assert!(
        decode_lz4_raw_tile(&blob, PixelFormat::R8Uint, MAX_CACHED_TILE_DECODED_BYTES).is_err()
    );

    // Oversized crafted header rejected BEFORE allocation.
    let mut bomb = Vec::with_capacity(HEADER_LEN + 4);
    bomb.extend_from_slice(&MAGIC);
    bomb.push(VERSION);
    bomb.push(1); // R32Float
    bomb.push(0);
    bomb.push(FLAG_BYTE_SHUFFLED);
    bomb.extend_from_slice(&10_000u32.to_le_bytes());
    bomb.extend_from_slice(&10_000u32.to_le_bytes());
    bomb.extend_from_slice(&10_000u32.to_le_bytes());
    bomb.extend_from_slice(&10_000u32.to_le_bytes());
    bomb.extend_from_slice(&[0u8; 4]);
    assert!(matches!(
        decode_lz4_raw_tile(&bomb, PixelFormat::R32Float, MAX_CACHED_TILE_DECODED_BYTES),
        Err(RasterError::LimitExceeded { .. })
    ));

    // Truncated payload rejected.
    assert!(
        decode_lz4_raw_tile(
            &blob[..HEADER_LEN],
            PixelFormat::R32Float,
            MAX_CACHED_TILE_DECODED_BYTES
        )
        .is_err()
    );

    // Bad magic, bad version, unknown flags, and overflowed valid_w + 2*gutter rejected.
    let mut bad_magic = blob.clone();
    bad_magic[0] = b'X';
    assert!(
        decode_lz4_raw_tile(
            &bad_magic,
            PixelFormat::R32Float,
            MAX_CACHED_TILE_DECODED_BYTES
        )
        .is_err()
    );

    let mut bad_ver = blob.clone();
    bad_ver[4] = VERSION + 1;
    assert!(
        decode_lz4_raw_tile(
            &bad_ver,
            PixelFormat::R32Float,
            MAX_CACHED_TILE_DECODED_BYTES
        )
        .is_err()
    );

    let mut bad_flags = blob.clone();
    bad_flags[7] = 0x80;
    assert!(
        decode_lz4_raw_tile(
            &bad_flags,
            PixelFormat::R32Float,
            MAX_CACHED_TILE_DECODED_BYTES
        )
        .is_err()
    );

    let mut bad_geom = blob.clone();
    bad_geom[6] = 1; // gutter = 1 while valid_w == width == 8
    assert!(matches!(
        decode_lz4_raw_tile(
            &bad_geom,
            PixelFormat::R32Float,
            MAX_CACHED_TILE_DECODED_BYTES
        ),
        Err(RasterError::InvalidDimensions { .. })
    ));

    // Unshuffled (flags = 0) multi-byte payload also decodes cleanly.
    let raw_f32: Vec<u8> = [1.5f32, -2.25f32, f32::NAN, 42.0f32]
        .into_iter()
        .flat_map(f32::to_ne_bytes)
        .collect();
    let comp = lz4_flex::block::compress(&raw_f32);
    let mut unshuffled = Vec::with_capacity(HEADER_LEN + comp.len());
    unshuffled.extend_from_slice(&MAGIC);
    unshuffled.push(VERSION);
    unshuffled.push(1);
    unshuffled.push(0);
    unshuffled.push(0); // flags = 0 (no shuffle)
    unshuffled.extend_from_slice(&2u32.to_le_bytes());
    unshuffled.extend_from_slice(&2u32.to_le_bytes());
    unshuffled.extend_from_slice(&2u32.to_le_bytes());
    unshuffled.extend_from_slice(&2u32.to_le_bytes());
    unshuffled.extend_from_slice(&comp);
    let dec = decode_lz4_raw_tile(
        &unshuffled,
        PixelFormat::R32Float,
        MAX_CACHED_TILE_DECODED_BYTES,
    )
    .expect("unshuffled decode");
    assert_eq!(dec.data(), raw_f32.as_slice());
}
