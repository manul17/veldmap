use super::*;

fn basemap_from_rgba(
    width: u32,
    height: u32,
    rgba: Vec<u8>,
) -> Result<BundledBasemap, RasterError> {
    let px_count = (width as usize)
        .checked_mul(height as usize)
        .ok_or(RasterError::InvalidDimensions { width, height })?;
    let expected = px_count
        .checked_mul(4)
        .ok_or(RasterError::InvalidDimensions { width, height })?;
    if width == 0 || height == 0 || rgba.len() != expected {
        return Err(RasterError::InvalidDimensions { width, height });
    }
    let mut rgb = Vec::with_capacity(px_count * 3);
    for px in rgba.chunks_exact(4) {
        rgb.extend_from_slice(&px[0..3]);
    }
    Ok(BundledBasemap::from_rgb_box(
        width,
        height,
        rgb.into_boxed_slice(),
        BundledBasemapConfig::default(),
    ))
}

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
    basemap_from_rgba(w, h, rgba).expect("build synthetic basemap")
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
    let cfg = basemap.config();
    for bad in [
        QuadTileId {
            scheme: PlanetaryScheme::Geodetic,
            level: cfg.max_level + 1,
            x: 0,
            y: 0,
        },
        QuadTileId {
            scheme: PlanetaryScheme::PolarCap { north: true },
            level: cfg.polar_max_level + 1,
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
            MAX_BASEMAP_TILE_DIM + 1,
        ),
        Err(RasterError::InvalidDimensions { .. })
    ));
}
