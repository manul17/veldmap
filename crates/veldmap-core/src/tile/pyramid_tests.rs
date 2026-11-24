use super::*;
use crate::tile::{BandKey, PixelFormat, SourceId, TileScheme};

fn carrier_key(level: u8, x: u32, y: u32, gutter: u8) -> TileKey {
    TileKey {
        source: SourceId::new(1),
        band: BandKey::Display,
        scheme: TileScheme::RasterPyramid { max_level: 5 },
        format: PixelFormat::Rgba8UnormSrgb,
        gutter,
        decode_rev: 1,
        level,
        x,
        y,
    }
}

#[test]
fn level_size_and_count_match_ceil_halving() {
    for side in [1u32, 2, 3, 511, 512, 513, 1000, 10980] {
        for level in 0..8 {
            assert_eq!(level_size(side, level), side.div_ceil(1 << level));
        }
    }
    assert_eq!(level_count(RASTER_TILE_SIZE, RASTER_TILE_SIZE), 1);
    assert_eq!(level_count(RASTER_TILE_SIZE + 1, 1), 2);
    assert_eq!(level_count(0, 100), 0);
    assert_eq!(level_count(10980, 10980), 6);
}

#[test]
fn grid_and_extent_cover_side_exactly() {
    for side in [
        1u32,
        RASTER_TILE_SIZE - 1,
        RASTER_TILE_SIZE,
        RASTER_TILE_SIZE + 1,
        10980,
    ] {
        let tiles = grid_dim(side);
        let covered: u32 = (0..tiles).map(|i| tile_extent(i, side)).sum();
        assert_eq!(covered, side);
        for i in 0..tiles.saturating_sub(1) {
            assert_eq!(tile_extent(i, side), RASTER_TILE_SIZE);
        }
        assert_eq!(tile_extent(tiles, side), 0);
    }
}

#[test]
fn cell_uv_rect_handles_exact_parent_clipped_and_guttered_carriers() {
    let full_512 = CarrierLayout {
        width: 512,
        height: 512,
        valid_w: 512,
        valid_h: 512,
        gutter: 0,
    };
    // Exact tile uses full [0, 1]^2 -> offset (0, 0), scale (1, 1).
    let cell = cell_image_rect(1, 0, 0, 1024, 1024);
    assert_eq!(cell, [512.0, 0.0, 1024.0, 512.0]);
    assert_eq!(
        cell_uv_rect(cell, &carrier_key(0, 1, 0, 0), full_512),
        [0.0, 0.0, 1.0, 1.0]
    );

    // Cell (0; 1, 1) sits in the bottom-right quadrant of parent (1; 0, 0).
    let cell_br = cell_image_rect(1, 1, 0, 2048, 2048);
    assert_eq!(
        cell_uv_rect(cell_br, &carrier_key(1, 0, 0, 0), full_512),
        [0.5, 0.5, 0.5, 0.5]
    );

    // Clipped 188x40 edge tile maps [0, 1]^2 cleanly.
    let edge = cell_image_rect(1, 0, 0, 700, 40);
    assert_eq!(edge, [512.0, 0.0, 700.0, 40.0]);
    assert_eq!(
        cell_uv_rect(
            edge,
            &carrier_key(0, 1, 0, 0),
            CarrierLayout {
                width: 188,
                height: 40,
                valid_w: 188,
                valid_h: 40,
                gutter: 0,
            },
        ),
        [0.0, 0.0, 1.0, 1.0]
    );

    // Guttered 514x514 tile (valid 512x512, gutter=1) maps into offset 1/514, scale 512/514.
    let uv_gutter = cell_uv_rect(
        cell,
        &carrier_key(0, 1, 0, 1),
        CarrierLayout {
            width: 514,
            height: 514,
            valid_w: 512,
            valid_h: 512,
            gutter: 1,
        },
    );
    assert!((uv_gutter[0] - (1.0 / 514.0)).abs() < 1e-6);
    assert!((uv_gutter[2] - (512.0 / 514.0)).abs() < 1e-6);
}
