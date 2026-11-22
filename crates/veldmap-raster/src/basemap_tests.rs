use std::path::PathBuf;

use super::*;

fn synthetic_quadrant_basemap() -> BundledBasemap {
    // Synthetic 128x64 equirectangular image with distinct N/S & E/W quadrants
    // plus white South Pole and cyan North Pole so flips/rotations fail loudly:
    //   lat > +80: cyan   (40, 220, 230)
    //   lat < -80: white  (240, 240, 240)
    //   NW (lat >= 0, lon < 0):  red    (210, 50, 40)
    //   NE (lat >= 0, lon >= 0): green  (40, 190, 60)
    //   SW (lat < 0, lon < 0):   blue   (40, 70, 210)
    //   SE (lat < 0, lon >= 0):  yellow (220, 200, 40)
    let (w, h) = (128u32, 64u32);
    let mut rgba = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        let lat = 90.0 - 180.0 * (f64::from(y) + 0.5) / f64::from(h);
        for x in 0..w {
            let lon = -180.0 + 360.0 * (f64::from(x) + 0.5) / f64::from(w);
            let color = if lat > 80.0 {
                [40, 220, 230, 255]
            } else if lat < -80.0 {
                [240, 240, 240, 255]
            } else {
                match (lat >= 0.0, lon >= 0.0) {
                    (true, false) => [210, 50, 40, 255],
                    (true, true) => [40, 190, 60, 255],
                    (false, false) => [40, 70, 210, 255],
                    (false, true) => [220, 200, 40, 255],
                }
            };
            let off = ((y * w + x) * 4) as usize;
            rgba[off..off + 4].copy_from_slice(&color);
        }
    }
    BundledBasemap::from_rgba(w, h, rgba).expect("build synthetic basemap")
}

#[test]
fn renders_geodetic_web_mercator_and_polar_tiles_with_orientation_and_bounds_checks() {
    let basemap = synthetic_quadrant_basemap();

    // 1. Geodetic z = 1: NW (x=0,y=0) = red, NE (x=2,y=0) = green, SE (x=2,y=1) = yellow.
    let nw = basemap
        .render_tile_sized(
            QuadTileId {
                scheme: PlanetaryScheme::Geodetic,
                level: 1,
                x: 0,
                y: 0,
            },
            16,
        )
        .expect("geodetic NW");
    let mid = (8 * 16 + 8) * 4;
    assert!(
        nw.data()[mid] > 180 && nw.data()[mid + 1] < 80,
        "NW={:?}",
        &nw.data()[mid..mid + 4]
    );

    let ne = basemap
        .render_tile_sized(
            QuadTileId {
                scheme: PlanetaryScheme::Geodetic,
                level: 1,
                x: 2,
                y: 0,
            },
            16,
        )
        .expect("geodetic NE");
    assert!(
        ne.data()[mid + 1] > 160 && ne.data()[mid] < 80,
        "NE={:?}",
        &ne.data()[mid..mid + 4]
    );

    // 2. WebMercator z = 1: NW (0,0) = red, NE (1,0) = green, SW (0,1) = blue, SE (1,1) = yellow.
    let wm_nw = basemap
        .render_tile_sized(
            QuadTileId {
                scheme: PlanetaryScheme::WebMercator,
                level: 1,
                x: 0,
                y: 0,
            },
            16,
        )
        .expect("web mercator NW");
    assert!(wm_nw.data()[mid] > 180 && wm_nw.data()[mid + 1] < 80);

    let wm_se = basemap
        .render_tile_sized(
            QuadTileId {
                scheme: PlanetaryScheme::WebMercator,
                level: 1,
                x: 1,
                y: 1,
            },
            16,
        )
        .expect("web mercator SE");
    assert!(wm_se.data()[mid] > 180 && wm_se.data()[mid + 1] > 160 && wm_se.data()[mid + 2] < 80);

    // 3. Both North (cyan) and South (white) PolarCap tiles render cleanly.
    let north_cap = basemap
        .render_tile_sized(
            QuadTileId {
                scheme: PlanetaryScheme::PolarCap { north: true },
                level: 0,
                x: 0,
                y: 0,
            },
            16,
        )
        .expect("north polar cap");
    assert!(
        north_cap.data()[mid] < 80
            && north_cap.data()[mid + 1] > 190
            && north_cap.data()[mid + 2] > 200
    );

    let south_cap = basemap
        .render_tile_sized(
            QuadTileId {
                scheme: PlanetaryScheme::PolarCap { north: false },
                level: 0,
                x: 0,
                y: 0,
            },
            16,
        )
        .expect("south polar cap");
    assert!(
        south_cap.data()[mid] > 220
            && south_cap.data()[mid + 1] > 220
            && south_cap.data()[mid + 2] > 220
    );

    // 4. Out-of-range levels, OOB x/y, and invalid tile_px fail cleanly.
    for bad in [
        QuadTileId {
            scheme: PlanetaryScheme::Geodetic,
            level: BUNDLED_MAX_LEVEL + 1,
            x: 0,
            y: 0,
        },
        QuadTileId {
            scheme: PlanetaryScheme::PolarCap { north: true },
            level: BUNDLED_POLAR_MAX_LEVEL + 1,
            x: 0,
            y: 0,
        },
        QuadTileId {
            scheme: PlanetaryScheme::WebMercator,
            level: 1,
            x: 2,
            y: 0,
        },
    ] {
        assert!(matches!(
            basemap.render_tile(bad),
            Err(RasterError::TileOutOfRange { .. })
        ));
    }
    assert!(matches!(
        basemap.render_tile_sized(
            QuadTileId {
                scheme: PlanetaryScheme::Geodetic,
                level: 0,
                x: 0,
                y: 0,
            },
            0,
        ),
        Err(RasterError::InvalidDimensions { .. })
    ));
    assert!(matches!(
        basemap.render_tile_sized(
            QuadTileId {
                scheme: PlanetaryScheme::Geodetic,
                level: 0,
                x: 0,
                y: 0,
            },
            MAX_MAP_TILE_DIM + 1,
        ),
        Err(RasterError::InvalidDimensions { .. })
    ));
}

#[test]
fn bundled_blue_marble_asset_decodes_and_preserves_geographic_landmarks() {
    let asset_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets/basemap/blue_marble_ng.jpg");
    let bytes = std::fs::read(&asset_path).expect("read bundled blue marble asset");
    let basemap = BundledBasemap::from_encoded_bytes(&bytes).expect("decode blue marble");
    assert_eq!(basemap.dimensions(), (3600, 1800));
    assert!(basemap.levels.len() >= 4, "expected base + 3 mip levels");

    let to_lin = srgb_to_linear_lut();
    let to_srgb = linear_to_srgb_lut();
    // Verify exact sRGB -> linear -> sRGB LUT round-trip across all 256 values.
    for v in 0..=255u8 {
        assert_eq!(linear_to_srgb_u8(to_lin[usize::from(v)], to_srgb), v);
    }

    // South pole (Antarctica) must be bright white/ice.
    let sp = basemap.levels[0].sample_bilinear_srgb_with_luts(-89.0, 0.0, to_lin, to_srgb);
    assert!(
        sp[0] > 200 && sp[1] > 200 && sp[2] > 200,
        "south pole={sp:?}"
    );

    // Sahara desert (23 N, 13 E) must be bright tan (R > G > B).
    let sahara = basemap.levels[0].sample_bilinear_srgb_with_luts(23.0, 13.0, to_lin, to_srgb);
    assert!(
        sahara[0] > 150 && sahara[0] > sahara[2].saturating_add(40),
        "sahara={sahara:?}"
    );

    // Central Pacific (0 N, -150 E) must be deep ocean blue (B > R).
    let pacific = basemap.levels[0].sample_bilinear_srgb_with_luts(0.0, -150.0, to_lin, to_srgb);
    assert!(
        pacific[2] > pacific[0].saturating_add(20) && pacific[0] < 40,
        "pacific={pacific:?}"
    );
}
