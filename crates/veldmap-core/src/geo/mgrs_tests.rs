use super::*;

fn band_middle_deg(band: usize) -> f64 {
    let (south_edge, north_edge) = band_bounds_deg(band);
    (south_edge + north_edge) * 0.5
}

#[test]
fn every_band_fits_the_row_window_from_both_sides() {
    for (band, &letter) in BANDS.iter().enumerate() {
        let (south, north) = band_bounds_deg(band);
        let target = band_middle_deg(band).abs() * 110_946.0;

        for (edge, side) in [(south, "south"), (north, "north")] {
            let margin = ROW_CYCLE_M / 2.0 - (edge.abs() * 110_946.0 - target).abs();
            assert!(
                margin > 250_000.0,
                "band {} {} margin {:.0} m",
                letter as char,
                side,
                margin
            );
        }
    }
}

#[test]
fn the_x_band_is_twelve_degrees_wide() {
    assert_eq!(band_middle_deg(BANDS.len() - 1), 78.0);
    assert_eq!(band_middle_deg(0), -76.0);
}

#[test]
fn garbage_codes_are_refused() {
    assert!(parse_to_wgs84("40WIC").is_err());
    assert!(parse_to_wgs84("00WFC").is_err());
    assert!(parse_to_wgs84("40ZFC").is_err());
    // Non-ASCII 5-byte and multi-byte UTF-8 inputs must error, never panic on char boundaries.
    assert!(parse_to_wgs84("a\u{00E9}bc").is_err());
    assert!(parse_to_wgs84("37UD\u{0419}").is_err());
    assert!(parse_to_wgs84("37UD\u{00E9}").is_err());
    // Out-of-band row letter wrapping into the wrong hemisphere, unequal digit halves,
    // nonexistent Svalbard zones (32X/34X/36X), and out-of-zone high-latitude columns.
    assert!(parse_to_wgs84("37MAA").is_err());
    assert!(parse_to_wgs84("37UDQ 12 3456").is_err());
    assert!(matches!(
        parse_to_wgs84("32XAA"),
        Err(MgrsError::InvalidZoneBand {
            zone: 32,
            band: 'X'
        })
    ));
    assert!(parse_to_wgs84("34XAA").is_err());
    assert!(parse_to_wgs84("36XAA").is_err());
    assert!(matches!(
        parse_to_wgs84("37XAA"),
        Err(MgrsError::InvalidColumn { col: 'A', zone: 37 })
    ));
}

#[test]
fn parse_mgrs_square_and_full_grid_to_wgs84() {
    // 31N AA 66021 00000 is the exact equator / prime meridian (0 N, 0 E) to ~1 m.
    let (lat0, lon0, radius0) = parse_to_wgs84("31N AA 66021 00000").unwrap();
    assert!(lat0.abs() < 1e-4, "lat0={lat0}");
    assert!(lon0.abs() < 1e-4, "lon0={lon0}");
    assert!(radius0 <= 0.003);

    // Sentinel-2 reference granule T40WFC center (650 km E, 7750 km N in UTM 40N).
    let (lat, lon, radius) = parse_to_wgs84("T40WFC").unwrap();
    assert!((lat - 69.8117).abs() < 0.01, "lat={lat}");
    assert!((lon - 60.9036).abs() < 0.01, "lon={lon}");
    assert!(radius > 0.5);

    // 37U DQ 50000 50000: UTM 37N E=450,000.5 m, N=5,450,000.5 m.
    let (lat_m, lon_m, radius_m) = parse_to_wgs84("37U DQ 50000 50000").unwrap();
    assert!((lat_m - 49.2015).abs() < 0.001, "lat={lat_m}");
    assert!((lon_m - 38.3132).abs() < 0.001, "lon={lon_m}");
    assert!(radius_m < 0.01);

    // Single-digit zone (Hawaii 4Q FJ 1234 5678: UTM 4N E=612,345 m, N=2,356,785 m).
    let (lat_h, lon_h, _) = parse_to_wgs84("4Q FJ 1234 5678").unwrap();
    assert!((lat_h - 21.309_478).abs() < 1e-4, "lat={lat_h}");
    assert!((lon_h + 157.916_819).abs() < 1e-4, "lon={lon_h}");

    // Norway widened 32V (Bergen ~60.391 N, 5.322 E -> 32V KN 973 010).
    let (lat_no, lon_no, _) = parse_to_wgs84("32V KN 973 010").unwrap();
    assert!((lat_no - 60.395).abs() < 0.01, "lat_no={lat_no}");
    assert!((lon_no - 5.322).abs() < 0.01, "lon_no={lon_no}");
}

#[test]
fn partial_boundary_mgrs_squares_stay_inside_their_zone() {
    // Standard 5-char `31N AA` is clipped on the west by the 0 E meridian of Zone 31.
    let (lat, lon, _) = parse_to_wgs84("31N AA").unwrap();
    assert!((0.0..=8.0).contains(&lat), "lat={lat} outside band N");
    assert!((0.0..=6.0).contains(&lon), "lon={lon} outside zone 31");

    // Standard 5-char `40W FC` stays inside Zone 40 (54..60 E), whereas Sentinel-2
    // granule `T40WFC` keeps its full 100 km UTM square center (~60.90 E).
    let (lat_m, lon_m, _) = parse_to_wgs84("40W FC").unwrap();
    assert!((64.0..=72.0).contains(&lat_m), "lat_m={lat_m}");
    assert!((54.0..=60.0).contains(&lon_m), "lon_m={lon_m}");
}

#[test]
fn format_wgs84_to_mgrs_round_trips_globally_including_norway_svalbard_and_poles() {
    use crate::geo::ground_m;

    // Equator / prime meridian exact match.
    assert_eq!(
        format_wgs84_to_mgrs(0.0, 0.0, 5).as_deref(),
        Some("31N AA 66021 00000")
    );
    assert_eq!(format_wgs84_to_mgrs(0.0, 0.0, 0).as_deref(), Some("31N AA"));

    // Norway widened zone 32V (Bergen, 5.322 E lies inside 3..12 E -> 32V)
    // and narrowed zone 31V (1.5 E -> 31V).
    let bergen = format_wgs84_to_mgrs(60.3913, 5.3221, 5).expect("Bergen MGRS");
    assert!(bergen.starts_with("32V "), "expected 32V, got {bergen}");
    let north_sea = format_wgs84_to_mgrs(60.0, 1.5, 5).expect("31V MGRS");
    assert!(
        north_sea.starts_with("31V "),
        "expected 31V, got {north_sea}"
    );

    // Svalbard widened odd zones 31X, 33X, 35X, 37X.
    for (lon, expected_prefix) in [
        (4.0, "31X "),
        (15.6, "33X "),
        (25.0, "35X "),
        (38.0, "37X "),
    ] {
        let code = format_wgs84_to_mgrs(78.2232, lon, 5).expect("Svalbard MGRS");
        assert!(
            code.starts_with(expected_prefix),
            "lon={lon} expected {expected_prefix}, got {code}"
        );
        let (rlat, rlon, _) = parse_to_wgs84(&code).expect("parse Svalbard MGRS");
        let err_m = ground_m((78.2232, lon), (rlat, rlon));
        assert!(err_m < 2.0, "Svalbard round-trip err={err_m} m for {code}");
    }

    // Broad global round-trip including exact boundaries, sub-ULP negative
    // latitudes, antimeridian, and high-convergence Norway/Svalbard edges.
    let samples = [
        (47.2357, 39.7015),
        (-33.8688, 151.2093),
        (-54.8019, -68.3030),
        (21.3095, -157.9168),
        (0.0001, -179.999),
        (-0.0001, 179.999),
        (-80.0, -180.0),
        (-8e-15, 0.0),
        (-1e-15, 0.0),
        (-0.0, 0.0),
        (84.0, 15.0),
        (83.9999, 9.0001), // Svalbard 33X high-convergence west/north corner
        (60.0, 3.0001),    // Norway 32V west/south boundary
        (60.3913, 5.3221),
    ];
    for (lat, lon) in samples {
        for prec in 1_u8..=5 {
            let formatted = format_wgs84_to_mgrs(lat, lon, prec).expect("valid UTM band");
            let (rlat, rlon, _) = parse_to_wgs84(&formatted).unwrap_or_else(|e| {
                panic!("round-trip parse failed for {formatted} ({lat},{lon},p={prec}): {e:?}")
            });
            let tol_m = 10_f64.powi(5 - i32::from(prec)) * 1.1;
            let err_m = ground_m((lat, lon), (rlat, rlon));
            assert!(
                err_m <= tol_m,
                "lat={lat}, lon={lon}, prec={prec}, code={formatted}: err={err_m} m > {tol_m} m"
            );
        }
    }

    // Sub-ULP negative latitude stays consistently in southern band `M` with row `V`.
    let tiny_south = format_wgs84_to_mgrs(-8e-15, 0.0, 5).expect("sub-ULP south");
    assert_eq!(tiny_south, "31M AV 66021 99999");

    // Longitude +180 wraps cleanly into Zone 01; precision > 5 clamps to 5.
    let anti = format_wgs84_to_mgrs(10.0, 180.0, 4).expect("+180 lon");
    assert!(anti.starts_with("01P "), "got {anti}");
    assert_eq!(
        format_wgs84_to_mgrs(0.0, 0.0, 9),
        format_wgs84_to_mgrs(0.0, 0.0, 5)
    );

    // Polar and non-finite inputs return None.
    assert!(format_wgs84_to_mgrs(84.5, 10.0, 4).is_none());
    assert!(format_wgs84_to_mgrs(-80.5, 10.0, 4).is_none());
    assert!(format_wgs84_to_mgrs(f64::NAN, 0.0, 4).is_none());
}
