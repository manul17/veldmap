use super::*;
use crate::geo::datum::{INTERNATIONAL_FOOT_M, KRASOVSKY, US_SURVEY_FOOT_M};
use crate::geo::ground_m;

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
fn web_mercator_and_expanded_epsg_recognized() {
    assert_eq!(Projection::from_epsg(3857), Some(Projection::WebMercator));
    assert!(matches!(
        Projection::from_epsg(32637),
        Some(Projection::Transverse { .. })
    ));
    assert!(matches!(
        Projection::from_epsg(3035),
        Some(Projection::Proj4(_))
    ));
    assert_eq!(Projection::from_epsg(999_999), None);

    // In-house GK and MSK-61 labels are recognized without requiring a manual name.
    assert!(
        Projection::from_epsg(28407)
            .expect("GK 7")
            .label()
            .contains("28407")
    );
    assert!(
        Projection::from_epsg(6_336_101)
            .expect("MSK-61 z1")
            .label()
            .contains("MSK-61")
    );
}

#[test]
fn system_origin_latitude_and_linear_units_round_trip() {
    // Non-zero origin latitude (e.g., British National Grid-style 49 deg N) + US Survey Feet
    let sys = System {
        datum: WGS84_DATUM,
        central_meridian_deg: -75.0,
        origin_lat_deg: 38.0,
        scale: 0.9999,
        false_easting_m: 200_000.0,
        false_northing_m: 0.0,
        unit_to_metres: US_SURVEY_FOOT_M,
    };
    let (e_origin, n_origin) = sys.from_geodetic(38.0, -75.0);
    assert!((e_origin * US_SURVEY_FOOT_M - 200_000.0).abs() < 1e-6);
    assert!(n_origin.abs() < 1e-6);

    for (lat, lon) in [(39.5, -74.2), (37.1, -76.3), (40.7, -74.0)] {
        let (e, n) = sys.from_geodetic(lat, lon);
        let back = sys.to_geodetic(e, n);
        assert!(
            ground_m(back, (lat, lon)) < 1e-4,
            "round-trip error too high: {back:?} vs ({lat}, {lon})"
        );
    }

    // International Foot check
    let sys_ft = System {
        unit_to_metres: INTERNATIONAL_FOOT_M,
        ..sys
    };
    let (e_ft, n_ft) = sys_ft.from_geodetic(39.5, -74.2);
    let back_ft = sys_ft.to_geodetic(e_ft, n_ft);
    assert!(ground_m(back_ft, (39.5, -74.2)) < 1e-4);
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
fn epsg_names_a_system_only_inside_its_bands() {
    assert_eq!(System::from_epsg(32601), Some(System::utm(1, false)));
    assert_eq!(System::from_epsg(32660), Some(System::utm(60, false)));
    assert_eq!(System::from_epsg(32701), Some(System::utm(1, true)));
    assert_eq!(System::from_epsg(32760), Some(System::utm(60, true)));
    assert_eq!(System::from_epsg(28401), None);
    assert_eq!(System::from_epsg(28402), Some(System::gauss_kruger(2)));
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
