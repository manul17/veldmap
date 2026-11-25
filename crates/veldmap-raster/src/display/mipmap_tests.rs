use super::*;

#[test]
fn generates_full_linear_light_mip_chain_and_avoids_dark_srgb_collapse() {
    let mut data = Vec::with_capacity(4 * 4 * 4);
    for y in 0..4 {
        for x in 0..4 {
            let v = if (x + y) % 2 == 0 { 0u8 } else { 255u8 };
            data.extend_from_slice(&[v, v, v, 255]);
        }
    }
    let mut tile =
        DecodedTile::try_new(4, 4, PixelFormat::Rgba8UnormSrgb, data).expect("valid tile");
    generate_linear_mipmaps(&mut tile);

    assert_eq!(tile.mip_level_count(), 3);
    assert_eq!(tile.mip_dims(1), (2, 2));
    assert_eq!(tile.mip_dims(2), (1, 1));

    let mip1 = tile.mip_data(1).expect("mip 1");
    assert!(
        mip1[0].abs_diff(188) <= 2 && mip1[3] == 255,
        "expected linear-light ~188 sRGB, got {:?}",
        &mip1[0..4]
    );
    let mip2 = tile.mip_data(2).expect("mip 2");
    assert!(
        mip2[0].abs_diff(188) <= 2 && mip2[3] == 255,
        "expected linear-light ~188 sRGB at 1x1, got {:?}",
        &mip2[0..4]
    );
}

#[test]
fn transparent_neighbors_do_not_darken_opaque_rgb_and_non_srgb_is_untouched() {
    let data = vec![
        0, 255, 0, 255, 0, 0, 0, 0, //
        0, 0, 0, 0, 0, 0, 0, 0,
    ];
    let mut tile =
        DecodedTile::try_new(2, 2, PixelFormat::Rgba8UnormSrgb, data).expect("valid tile");
    generate_linear_mipmaps(&mut tile);
    let mip1 = tile.mip_data(1).expect("mip 1");
    assert_eq!(mip1[0], 0);
    assert_eq!(mip1[1], 255, "RGB must stay pure green without black halo");
    assert_eq!(mip1[2], 0);
    assert!(mip1[3].abs_diff(64) <= 1, "alpha={}", mip1[3]);

    let mut npot = DecodedTile::try_new(5, 3, PixelFormat::Rgba8UnormSrgb, vec![200u8; 5 * 3 * 4])
        .expect("valid npot");
    generate_linear_mipmaps(&mut npot);
    assert_eq!(npot.mip_level_count(), 3);

    let mut sci =
        DecodedTile::try_new(4, 4, PixelFormat::R32Float, vec![0u8; 4 * 4 * 4]).expect("sci");
    generate_linear_mipmaps(&mut sci);
    assert_eq!(sci.mip_level_count(), 1);

    let mut padded = DecodedTile::try_with_valid_region(
        4,
        4,
        3,
        4,
        0,
        PixelFormat::Rgba8UnormSrgb,
        vec![255u8; 4 * 4 * 4],
    )
    .expect("padded");
    generate_linear_mipmaps(&mut padded);
    assert_eq!(padded.mip_level_count(), 1);

    let mut guttered = DecodedTile::try_with_valid_region(
        6,
        6,
        4,
        4,
        1,
        PixelFormat::Rgba8UnormSrgb,
        vec![255u8; 6 * 6 * 4],
    )
    .expect("guttered");
    generate_linear_mipmaps(&mut guttered);
    assert_eq!(guttered.mip_level_count(), 1);
}

#[test]
fn scientific_mipmaps_skip_nan_and_inf_handle_npot_and_categorical_stays_1mip() {
    let f32_bytes =
        |vals: &[f32]| -> Vec<u8> { vals.iter().flat_map(|v| v.to_ne_bytes()).collect() };
    let read_f32 = |bytes: &[u8], idx: usize| -> f32 {
        let o = idx * 4;
        f32::from_ne_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]])
    };

    // 4x2 tile: left 2x2 has [10, NaN, 30, +Inf] -> 20; right 2x2 is all NaN/-Inf -> NaN.
    let data = f32_bytes(&[
        10.0,
        f32::NAN,
        f32::NAN,
        f32::NEG_INFINITY,
        30.0,
        f32::INFINITY,
        f32::NAN,
        f32::NAN,
    ]);
    let mut tile = DecodedTile::try_new(4, 2, PixelFormat::R32Float, data).expect("sci");
    generate_tile_mipmaps(&mut tile);
    assert_eq!(tile.mip_level_count(), 3); // 4x2 -> 2x1 -> 1x1
    let mip1 = tile.mip_data(1).expect("mip1");
    assert!((read_f32(mip1, 0) - 20.0).abs() < 1e-5);
    assert!(read_f32(mip1, 1).is_nan());
    let mip2 = tile.mip_data(2).expect("mip2");
    assert!((read_f32(mip2, 0) - 20.0).abs() < 1e-5);

    // NPOT 5x3 folds trailing column/row into last texel cleanly.
    let mut npot =
        DecodedTile::try_new(5, 3, PixelFormat::R32Float, f32_bytes(&[4.0; 15])).expect("npot");
    generate_scientific_mipmaps(&mut npot);
    assert_eq!(npot.mip_level_count(), 3);
    assert!((read_f32(npot.mip_data(2).expect("mip2"), 0) - 4.0).abs() < 1e-5);

    // Guttered R32Float stays 1 mip.
    let mut guttered = DecodedTile::try_with_valid_region(
        6,
        6,
        4,
        4,
        1,
        PixelFormat::R32Float,
        f32_bytes(&[1.0; 36]),
    )
    .expect("guttered");
    generate_scientific_mipmaps(&mut guttered);
    assert_eq!(guttered.mip_level_count(), 1);

    // Categorical tiles stay strictly 1 mip.
    let mut u8_tile = DecodedTile::try_new(4, 4, PixelFormat::R8Uint, vec![3u8; 16]).expect("u8");
    generate_tile_mipmaps(&mut u8_tile);
    assert_eq!(u8_tile.mip_level_count(), 1);
    let mut u16_tile =
        DecodedTile::try_new(4, 4, PixelFormat::R16Uint, vec![0u8; 32]).expect("u16");
    generate_tile_mipmaps(&mut u16_tile);
    assert_eq!(u16_tile.mip_level_count(), 1);
}
