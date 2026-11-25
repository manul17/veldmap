use super::*;

#[test]
fn mip_dimensions_and_validation_handle_pow2_and_non_pow2() {
    assert_eq!(max_mip_levels(0, 0), 0);
    assert_eq!(max_mip_levels(0, 10), 0);
    assert_eq!(max_mip_levels(10, 0), 0);
    assert_eq!(mip_dimensions(0, 0, 0), (0, 0));
    assert_eq!(mip_dimensions(0, 10, 0), (0, 0));
    assert_eq!(mip_dimensions(10, 0, 2), (0, 0));
    assert_eq!(max_mip_levels(1, 1), 1);
    assert_eq!(max_mip_levels(256, 256), 9);
    assert_eq!(max_mip_levels(5, 3), 3);

    assert!(matches!(
        DecodedTile::try_new(0, 4, PixelFormat::Rgba8UnormSrgb, Vec::new()),
        Err(DecodedTileError::ZeroDimension { .. })
    ));
    assert!(matches!(
        DecodedTile::try_new(4, 0, PixelFormat::Rgba8UnormSrgb, Vec::new()),
        Err(DecodedTileError::ZeroDimension { .. })
    ));

    assert_eq!(mip_dimensions(5, 3, 0), (5, 3));
    assert_eq!(mip_dimensions(5, 3, 1), (2, 1));
    assert_eq!(mip_dimensions(5, 3, 2), (1, 1));
    assert_eq!(
        mip_chain_byte_size(4, 4, PixelFormat::Rgba8UnormSrgb, 3),
        64 + 16 + 4
    );

    let mut valid = DecodedTile::try_with_mips(
        4,
        4,
        PixelFormat::Rgba8UnormSrgb,
        vec![0u8; 4 * 4 * 4],
        vec![vec![0u8; 2 * 2 * 4], vec![0u8; 4]],
    )
    .expect("valid mipmapped tile");
    assert_eq!(valid.mip_level_count(), 3);
    assert_eq!(valid.byte_size(), 64 + 16 + 4);
    assert!(format!("{valid:?}").contains("mip_levels: 3"));

    // Too many mip levels rejected without mutating existing mips.
    assert!(matches!(
        valid.try_set_mips(vec![vec![0u8; 16], vec![0u8; 4], vec![0u8; 4]]),
        Err(DecodedTileError::TooManyMipLevels { .. })
    ));
    assert_eq!(valid.mip_level_count(), 3);

    // Mismatched mip byte length rejected.
    assert!(matches!(
        valid.try_set_mips(vec![vec![0u8; 15], vec![0u8; 4]]),
        Err(DecodedTileError::MipByteLengthMismatch { level: 1, .. })
    ));

    // Guttered DEM tile (6x6 storing 4x4 interior with gutter = 1) constructs cleanly.
    let mut dem = DecodedTile::try_with_valid_region(
        6,
        6,
        4,
        4,
        1,
        PixelFormat::R32Float,
        vec![0u8; 6 * 6 * 4],
    )
    .expect("valid guttered DEM");

    // Guttered tile rejects extra mip levels.
    assert!(matches!(
        dem.try_set_mips(vec![vec![0u8; 3 * 3 * 4]]),
        Err(DecodedTileError::MipsUnsupportedForPaddedOrGuttered { .. })
    ));

    // Overflowing valid_w + 2*gutter > width is rejected.
    assert!(matches!(
        DecodedTile::try_with_valid_region(
            6,
            6,
            5,
            4,
            1,
            PixelFormat::R32Float,
            vec![0u8; 6 * 6 * 4],
        ),
        Err(DecodedTileError::InvalidValidRegion { .. })
    ));
}
