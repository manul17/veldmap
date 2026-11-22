use super::*;

#[test]
fn web_mercator_world_is_square() {
    let map = Projection::WebMercator;
    let side = 20_037_508.34;

    let (x, y) = map.from_geodetic(0.0, 0.0);
    assert!(x.abs() < 1e-6 && y.abs() < 1e-6);

    let (x, _) = map.from_geodetic(0.0, 180.0);
    assert!((x - side).abs() < 0.5, "longitude edge: {x}");

    let (_, y) = map.from_geodetic(WEB_MERCATOR_LIMIT_DEG, 0.0);
    assert!((y - side).abs() < 0.5, "latitude edge: {y}");
}

#[test]
fn web_mercator_reference_places_match_tables() {
    let map = Projection::WebMercator;
    for (lat, lon, ex, ey) in [
        (47.2357, 39.7015, 4_419_550.76, 5_980_631.53),
        (-33.8688, 151.2093, 16_832_542.28, -4_011_198.65),
    ] {
        let (x, y) = map.from_geodetic(lat, lon);
        assert!(
            (x - ex).abs() < 0.5 && (y - ey).abs() < 0.5,
            "{lat} {lon} -> {x} {y}"
        );
    }
}

#[test]
fn web_mercator_roundtrip() {
    let map = Projection::WebMercator;
    for lat in [-84.0, -45.0, -0.5, 0.0, 12.25, 60.0, 84.9] {
        for lon in [-179.5, -90.0, 0.0, 33.3, 179.5] {
            let (x, y) = map.from_geodetic(lat, lon);
            let (back_lat, back_lon) = map.to_geodetic(x, y);
            assert!(
                (back_lat - lat).abs() < 1e-9 && (back_lon - lon).abs() < 1e-9,
                "{lat} {lon} -> {x} {y} -> {back_lat} {back_lon}"
            );
        }
    }
}

#[test]
fn web_mercator_clamps_beyond_domain() {
    let map = Projection::WebMercator;
    let (_, edge) = map.from_geodetic(WEB_MERCATOR_LIMIT_DEG, 0.0);
    let (_, beyond) = map.from_geodetic(89.9, 0.0);
    assert_eq!(edge, beyond);
    assert!(beyond.is_finite());
}

#[test]
fn web_mercator_recognized_by_epsg() {
    assert_eq!(Projection::from_epsg(3857), Some(Projection::WebMercator));
    assert!(matches!(
        Projection::from_epsg(32637),
        Some(Projection::Transverse(_))
    ));
    assert_eq!(Projection::from_epsg(3035), None);
}

fn ground_m(first: (f64, f64), second: (f64, f64)) -> f64 {
    let lat = ((first.0 + second.0) / 2.0).to_radians();
    let per_lat = 111_132.92 - 559.82 * (2.0 * lat).cos();
    let per_lon = 111_412.84 * lat.cos() - 93.5 * (3.0 * lat).cos();
    ((first.0 - second.0) * per_lat).hypot((first.1 - second.1) * per_lon)
}

#[test]
fn the_local_grid_matches_an_outside_converter() {
    let checks = [
        (1, 500_000.0, 1_330_000.0, 47.932_102_711, 38.383_287_711),
        (2, 500_000.0, 2_330_000.0, 47.932_149_061, 41.383_334_064),
        (3, 500_000.0, 3_330_000.0, 47.932_194_230, 44.383_385_063),
        (2, 500_000.0, 2_350_000.0, 47.930_905_649, 41.650_973_905),
    ];
    for (zone, north, east, lat, lon) in checks {
        let system = System::msk61(zone).expect("MSK-61 zone");
        let got = system.to_geodetic(east, north);
        let off = ground_m(got, (lat, lon));
        assert!(
            off < 0.002,
            "zone {zone}: {got:?} vs ({lat}, {lon}) — {off} m"
        );
    }
}

#[test]
fn the_datum_shift_is_worth_a_hundred_metres() {
    let rostov = (47.23, 39.72);
    let moved = PULKOVO_1942.to_wgs84(rostov.0, rostov.1);
    let shift = ground_m(rostov, moved);
    assert!((100.0..130.0).contains(&shift), "{shift} m");
    assert_eq!(WGS84_DATUM.to_wgs84(rostov.0, rostov.1), rostov);
}

#[test]
fn the_helmert_and_its_inverse_are_exact() {
    let points = [
        [3_000_000.0, 2_000_000.0, 5_200_000.0],
        [-1_500_000.0, 5_800_000.0, 2_000_000.0],
        [4_000_000.0, -3_000_000.0, -3_500_000.0],
    ];
    let turned = Datum {
        rotate_sec: [-1.5, -0.35, -0.79],
        ..PULKOVO_1942
    };
    for datum in [PULKOVO_1942, turned] {
        for point in points {
            let back = datum.helmert_back(datum.helmert(point));
            for axis in 0..3 {
                let off = (back[axis] - point[axis]).abs();
                assert!(off < 1e-6, "{:?}, axis {axis}: {off} m", datum.rotate_sec);
            }
        }
    }
}

#[test]
fn the_datum_shift_and_its_reverse_converge_where_it_is_fitted() {
    for (lat, lon) in [
        (47.23, 39.72),
        (55.75, 37.62),
        (66.53, 66.60),
        (43.1, 131.9),
    ] {
        let (there_lat, there_lon) = PULKOVO_1942.to_wgs84(lat, lon);
        let back = PULKOVO_1942.from_wgs84(there_lat, there_lon);
        assert!(ground_m(back, (lat, lon)) < 1e-3);
    }
}

#[test]
fn epsg_names_a_system_only_inside_its_bands() {
    assert_eq!(System::from_epsg(32601), Some(System::utm(1, false)));
    assert_eq!(System::from_epsg(32660), Some(System::utm(60, false)));
    assert_eq!(System::from_epsg(32701), Some(System::utm(1, true)));
    assert_eq!(System::from_epsg(32760), Some(System::utm(60, true)));
    assert_eq!(System::from_epsg(28401), Some(System::gauss_kruger(1)));
    assert_eq!(System::from_epsg(28408), Some(System::gauss_kruger(8)));
    assert_eq!(System::from_epsg(28432), Some(System::gauss_kruger(32)));
    assert_eq!(System::from_epsg(6_336_101), System::msk61(1));
    assert_eq!(System::from_epsg(6_336_102), System::msk61(2));
    assert_eq!(System::from_epsg(6_336_103), System::msk61(3));

    assert_eq!(System::from_epsg(32600), None);
    assert_eq!(System::from_epsg(32700), None);
    assert_eq!(System::from_epsg(32661), None);
    assert_eq!(System::from_epsg(32761), None);
    assert_eq!(System::from_epsg(28400), None);
    assert_eq!(System::from_epsg(28433), None);
    assert_eq!(System::from_epsg(6_336_100), None);
    assert_eq!(System::from_epsg(6_336_104), None);
    assert_eq!(System::from_epsg(4326), None);
    assert_eq!(System::from_epsg(3031), None);
}

#[test]
fn each_family_keeps_its_own_numbers() {
    let utm = System::from_epsg(32638).expect("UTM zone 38N");
    assert!((utm.central_meridian_deg - 45.0).abs() < 1e-12);
    assert_eq!(utm.scale, 0.9996);
    assert_eq!(utm.datum, WGS84_DATUM);

    let gk = System::gauss_kruger(8);
    assert!((gk.central_meridian_deg - 45.0).abs() < 1e-12);
    assert_eq!(gk.scale, 1.0);
    assert_eq!(gk.false_easting_m, 8_500_000.0);
    assert_eq!(gk.datum.ellipsoid, KRASOVSKY);

    let msk = System::msk61(2).expect("MSK-61 zone 2");
    assert!((msk.central_meridian_deg - (40.0 + 59.0 / 60.0)).abs() < 1e-10);
    assert_eq!(msk.false_easting_m, 2_300_000.0);
    assert_eq!(msk.false_northing_m, -4_811_057.628);
}

#[test]
fn roundtrip_converges_across_zone() {
    let cases = [
        (System::utm(40, false), [-79.0, -8.0, 0.0, 45.0, 83.5], 1e-4),
        (System::utm(40, true), [-79.0, -8.0, 0.0, 45.0, 83.5], 1e-4),
        (
            System::gauss_kruger(8),
            [42.0, 50.0, 55.0, 65.0, 75.0],
            1e-3,
        ),
        (
            System::msk61(2).expect("MSK-61 zone 2"),
            [46.0, 47.0, 48.0, 49.0, 50.0],
            1e-3,
        ),
    ];
    for (system, latitudes, tolerance_m) in cases {
        let middle = system.central_meridian_deg;
        for lat in latitudes {
            for offset in [-2.5, 0.0, 1.7, 2.9] {
                let lon = middle + offset;
                let (e, n) = system.from_geodetic(lat, lon);
                let back = system.to_geodetic(e, n);
                let off = ground_m(back, (lat, lon));
                assert!(off < tolerance_m, "{system:?}: {back:?} — {off} m");
            }
        }
    }
}

#[test]
fn a_longitude_across_the_seam_lands_in_its_zone() {
    let zone = System::utm(1, false);
    assert!((zone.central_meridian_deg + 177.0).abs() < 1e-12);

    let (east, north) = zone.from_geodetic(10.0, 179.0);
    assert!((0.0..500_000.0).contains(&east));
    assert!((1_000_000.0..1_200_000.0).contains(&north));

    let (lat, lon) = zone.to_geodetic(east, north);
    assert!((lat - 10.0).abs() < 1e-9);
    assert!((lon + 181.0).abs() < 1e-9);
}

#[test]
fn a_system_beyond_the_seam_keeps_its_branch() {
    let far = System::gauss_kruger(32);
    assert!((far.central_meridian_deg - 189.0).abs() < 1e-12);

    let (lat, lon) = far.to_geodetic(far.false_easting_m, 5_000_000.0);
    assert!(lon > 180.0);
    assert!((lon - 189.0).abs() < 0.01);
    assert!((44.0..46.0).contains(&lat));
}

#[test]
fn a_shifted_datum_on_the_same_ellipsoid_still_moves() {
    let nudged = Datum {
        shift_m: [100.0, 0.0, 0.0],
        ..WGS84_DATUM
    };
    let here = (0.0, 90.0);
    let moved = nudged.to_wgs84(here.0, here.1);
    assert!((ground_m(moved, here) - 100.0).abs() < 0.5);

    let back = nudged.from_wgs84(moved.0, moved.1);
    assert!(ground_m(back, here) < 1e-3);
}

#[test]
fn the_axis_gives_a_pole_and_not_a_nan() {
    let polar = KRASOVSKY.semi_major_m * (1.0 - KRASOVSKY.flattening);
    let (north, lon) = KRASOVSKY.earth_centred_to_geodetic([0.0, 0.0, polar]);
    assert!((north - 90.0).abs() < 1e-9);
    assert_eq!(lon, 0.0);

    let (south, _) = KRASOVSKY.earth_centred_to_geodetic([0.0, 0.0, -polar]);
    assert!((south + 90.0).abs() < 1e-9);
}

#[test]
fn axis_anchors_are_exact() {
    let north = System::utm(31, false);
    let (e, n) = north.from_geodetic(0.0, 3.0);
    assert!((e - 500_000.0).abs() < 1e-6 && n.abs() < 1e-6);

    let south = System::utm(31, true);
    let (_, n) = south.from_geodetic(0.0, 3.0);
    assert!((n - 10_000_000.0).abs() < 1e-6);
}

#[test]
fn hemispheres_mirror() {
    let zone = System::utm(40, false);
    let (e1, n1) = zone.from_geodetic(30.0, 58.0);
    let (e2, n2) = zone.from_geodetic(-30.0, 58.0);
    assert!((e1 - e2).abs() < 1e-6);
    assert!((n1 + n2).abs() < 1e-6);
}

#[test]
fn near_pole_geodetic_and_datum_roundtrips_avoid_singularity() {
    for lat_deg in [85.0, 89.5, 89.999, -89.999, 90.0, -90.0] {
        let ecef = WGS84.to_earth_centred(lat_deg, 37.0);
        let (back_lat, _) = WGS84.earth_centred_to_geodetic(ecef);
        assert!(
            (back_lat - lat_deg).abs() < 1e-12,
            "near-pole ECEF roundtrip failed at {lat_deg}: got {back_lat}"
        );

        let shifted = PULKOVO_1942.to_wgs84(lat_deg, 37.0);
        let back = PULKOVO_1942.from_wgs84(shifted.0, shifted.1);
        assert!(back.0.is_finite() && back.1.is_finite());
        let err_m = (back.0 - lat_deg).abs().to_radians() * WGS84.semi_major_m;
        assert!(err_m < 1e-2, "near-pole lat {lat_deg} drifted by {err_m} m");
    }
}

#[test]
fn sentinel_tile_corner_lands_where_the_tile_is() {
    let zone = System::utm(40, false);
    let (lat, lon) = zone.to_geodetic(600_000.0, 7_800_000.0);
    assert!((69.9..70.5).contains(&lat));
    assert!((59.0..60.3).contains(&lon));

    for step in 0..9 {
        let northing = 500_000.0 + f64::from(step) * 1_000_000.0;
        let (lat, lon) = zone.to_geodetic(600_000.0, northing);
        let (e, n) = zone.from_geodetic(lat, lon);
        assert!((e - 600_000.0).abs() < 1e-6);
        assert!((n - northing).abs() < 1e-6);
    }
}
