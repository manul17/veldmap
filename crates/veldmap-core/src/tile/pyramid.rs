//! Raster pyramid arithmetic (`level_size`, `level_count`, `grid_dim`,
//! `tile_extent`, `cell_image_rect`) and carrier-relative UV projection
//! (`CarrierLayout`, `cell_uv_rect`) honoring clipped edge tiles and border
//! gutters.

use super::key::TileKey;

/// Canonical raster pyramid tile side in pixels (`512`).
pub const RASTER_TILE_SIZE: u32 = 512;

/// Side dimension in pixels at pyramid `level` (`0` = finest native, rounded
/// up at each halving step so `level_size(side, k) == side.div_ceil(2^k)`).
pub fn level_size(side: u32, level: u32) -> u32 {
    let mut size = side;
    for _ in 0..level {
        if size <= 1 {
            break;
        }
        size = size.div_ceil(2);
    }
    size
}

/// Number of pyramid levels required until the longest side fits inside a
/// single [`RASTER_TILE_SIZE`] tile (`0` for empty rasters).
pub fn level_count(width: u32, height: u32) -> u32 {
    if width == 0 || height == 0 {
        return 0;
    }
    let mut side = width.max(height);
    let mut levels = 1;
    while side > RASTER_TILE_SIZE {
        side = side.div_ceil(2);
        levels += 1;
    }
    levels
}

/// Number of tiles along one axis of length `side_px`.
pub fn grid_dim(side_px: u32) -> u32 {
    side_px.div_ceil(RASTER_TILE_SIZE)
}

/// Actual valid pixel extent of tile `index` along a side of length `side_px`.
/// Interior tiles return `RASTER_TILE_SIZE`, edge tiles return the remainder,
/// and out-of-range indices return `0`.
pub fn tile_extent(index: u32, side_px: u32) -> u32 {
    let start = index.saturating_mul(RASTER_TILE_SIZE);
    side_px
        .min(start.saturating_add(RASTER_TILE_SIZE))
        .saturating_sub(start)
}

/// Side length of one pixel at `level` expressed in base-level pixels (`2^level`).
pub(super) fn level_px(level: u32) -> f64 {
    if level >= 63 {
        (1u64 << 62) as f64
    } else {
        (1u64 << level) as f64
    }
}

/// Bounding rectangle `[x0, y0, x1, y1]` of pyramid cell `(level, x, y)` in
/// base-level pixel coordinates, clipped to `(width, height)`.
pub fn cell_image_rect(x: u32, y: u32, level: u32, width: u32, height: u32) -> [f64; 4] {
    let px = level_px(level);
    let lw = level_size(width, level);
    let lh = level_size(height, level);
    let x0 = (f64::from(x) * f64::from(RASTER_TILE_SIZE) * px).min(f64::from(width));
    let y0 = (f64::from(y) * f64::from(RASTER_TILE_SIZE) * px).min(f64::from(height));
    let x1 = (x0 + f64::from(tile_extent(x, lw)) * px).min(f64::from(width));
    let y1 = (y0 + f64::from(tile_extent(y, lh)) * px).min(f64::from(height));
    [x0, y0, x1, y1]
}

/// Allocated texture dimensions, valid interior extent, and border gutter of a
/// carrier tile texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CarrierLayout {
    pub width: u32,
    pub height: u32,
    pub valid_w: u32,
    pub valid_h: u32,
    pub gutter: u8,
}

/// Compute the normalized UV sub-rectangle `[u_off, v_off, u_scale, v_scale]`
/// inside a carrier tile covering `cell` (`[x0, y0, x1, y1]` from
/// [`cell_image_rect`]), honoring the carrier texture's [`CarrierLayout`].
///
/// Works uniformly for exact tiles, coarser ancestor carriers, clipped
/// non-power-of-two edge tiles (`188x40`), and `gutter = 1` DEM tiles (`514x514`).
pub fn cell_uv_rect(cell: [f64; 4], carrier: &TileKey, layout: CarrierLayout) -> [f32; 4] {
    if layout.width == 0 || layout.height == 0 || layout.valid_w == 0 || layout.valid_h == 0 {
        return [0.0, 0.0, 0.0, 0.0];
    }
    let px = level_px(u32::from(carrier.level));
    let origin_x = f64::from(carrier.x) * f64::from(RASTER_TILE_SIZE);
    let origin_y = f64::from(carrier.y) * f64::from(RASTER_TILE_SIZE);
    let gutter = f64::from(layout.gutter);

    let map_u = |ix: f64| -> f32 {
        let local = (ix / px - origin_x).clamp(0.0, f64::from(layout.valid_w));
        ((gutter + local) / f64::from(layout.width)).clamp(0.0, 1.0) as f32
    };
    let map_v = |iy: f64| -> f32 {
        let local = (iy / px - origin_y).clamp(0.0, f64::from(layout.valid_h));
        ((gutter + local) / f64::from(layout.height)).clamp(0.0, 1.0) as f32
    };
    let u0 = map_u(cell[0]);
    let v0 = map_v(cell[1]);
    let u1 = map_u(cell[2]);
    let v1 = map_v(cell[3]);
    [u0, v0, (u1 - u0).max(0.0), (v1 - v0).max(0.0)]
}

#[cfg(test)]
mod tests {
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
}
