use super::*;

#[test]
fn mercator_y_bounds_and_lat_conversion_cover_web_mercator_extent() {
    let (n0, s0) = web_mercator_y_bounds(0, 1);
    assert!((n0 - std::f64::consts::PI).abs() < 1e-12);
    assert!((s0 + std::f64::consts::PI).abs() < 1e-12);
    assert!((web_mercator_lat_deg(n0) - WEB_MERCATOR_LIMIT_DEG).abs() < 1e-6);
    assert!((web_mercator_lat_deg(s0) + WEB_MERCATOR_LIMIT_DEG).abs() < 1e-6);
    assert!(web_mercator_lat_deg(0.0).abs() < 1e-12);

    let (n1, s1) = web_mercator_y_bounds(0, 2);
    assert!((n1 - std::f64::consts::PI).abs() < 1e-12);
    assert!(s1.abs() < 1e-12);
}

#[test]
fn geodetic_tile_bounds_handles_polar_clamping_cap_exclusion_and_out_of_range() {
    let geo0 = QuadTileId {
        scheme: PlanetaryScheme::Geodetic,
        level: 0,
        x: 0,
        y: 0,
    };
    let b = geodetic_tile_bounds(geo0).expect("geodetic z=0 valid");
    assert_eq!(b.lon_west_deg, -180.0);
    assert_eq!(b.lon_east_deg, 0.0);
    assert_eq!(b.raw_lat_north_deg, 90.0);
    assert_eq!(b.raw_lat_south_deg, -90.0);
    assert!((b.lat_north_deg - WEB_MERCATOR_LIMIT_DEG).abs() < 1e-12);
    assert!((b.lat_south_deg + WEB_MERCATOR_LIMIT_DEG).abs() < 1e-12);

    let inside_cap = QuadTileId {
        scheme: PlanetaryScheme::Geodetic,
        level: 6,
        x: 0,
        y: 0,
    };
    assert!(geodetic_tile_bounds(inside_cap).is_none());

    assert!(
        geodetic_tile_bounds(QuadTileId {
            scheme: PlanetaryScheme::Geodetic,
            level: 0,
            x: 2,
            y: 0,
        })
        .is_none()
    );
    assert!(
        geodetic_tile_bounds(QuadTileId {
            scheme: PlanetaryScheme::PolarCap { north: true },
            level: 0,
            x: 0,
            y: 0,
        })
        .is_none()
    );
    assert!(
        geodetic_tile_bounds(QuadTileId {
            scheme: PlanetaryScheme::WebMercator,
            level: 0,
            x: 0,
            y: 0,
        })
        .is_none()
    );
}

#[test]
fn sample_tile_surface_clamped_geodetic_v_tex_maps_unclamped_texture_range() {
    let geo0 = QuadTileId {
        scheme: PlanetaryScheme::Geodetic,
        level: 0,
        x: 0,
        y: 0,
    };
    let nw = sample_tile_surface(geo0, 0.0, 0.0).expect("north-west corner");
    let se = sample_tile_surface(geo0, 1.0, 1.0).expect("south-east corner");
    assert!((nw.lat_deg - WEB_MERCATOR_LIMIT_DEG).abs() < 1e-12);
    assert!((se.lat_deg + WEB_MERCATOR_LIMIT_DEG).abs() < 1e-12);
    assert_eq!(
        (nw.lon_deg, se.lon_deg, nw.u_tex, se.u_tex),
        (-180.0, 0.0, 0.0, 1.0)
    );
    let expected_v0 = (90.0 - WEB_MERCATOR_LIMIT_DEG) / 180.0;
    assert!((nw.v_tex - expected_v0).abs() < 1e-12);
    assert!((se.v_tex - (1.0 - expected_v0)).abs() < 1e-12);
}

#[test]
fn polar_r_cap_constant_and_surface_bounds_guards_are_exact() {
    let expected = ((90.0 - WEB_MERCATOR_LIMIT_DEG) * 0.5).to_radians().tan();
    assert!((POLAR_R_CAP - expected).abs() < 1e-16);

    // Out-of-bounds WebMercator and PolarCap tile coordinates return None.
    assert!(
        sample_tile_surface(
            QuadTileId {
                scheme: PlanetaryScheme::WebMercator,
                level: 1,
                x: 2,
                y: 0,
            },
            0.5,
            0.5,
        )
        .is_none()
    );
    assert!(
        sample_tile_surface(
            QuadTileId {
                scheme: PlanetaryScheme::PolarCap { north: true },
                level: 0,
                x: 1,
                y: 0,
            },
            0.5,
            0.5,
        )
        .is_none()
    );
}
