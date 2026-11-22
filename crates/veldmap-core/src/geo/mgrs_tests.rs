use super::*;

#[test]
fn every_band_fits_the_row_window_from_both_sides() {
    for (band, &letter) in BANDS.iter().enumerate() {
        let height = if band == BANDS.len() - 1 {
            LAST_BAND_DEG
        } else {
            BAND_DEG
        };
        let south = -80.0 + band as f64 * BAND_DEG;
        let target = band_middle_deg(band).abs() * 110_946.0;

        for (edge, side) in [(south, "south"), (south + height, "north")] {
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
fn tile_is_read_from_product_name() {
    assert_eq!(
        tile_of("S2C_MSIL2A_20260812T081601_N0512_R121_T40WFC_20260812T121315.SAFE"),
        Some("40WFC")
    );
    assert_eq!(
        tile_of("S1C_IW_GRDH_1SDV_20260810T031955_011C6E_EDC8.SAFE"),
        None
    );
}

#[test]
fn short_and_odd_segments_are_refused_not_sliced() {
    assert_eq!(tile_of("LC09_L2SP_02_T1"), None);
    assert_eq!(tile_of("A_T_B"), None);
    assert_eq!(tile_of("T"), None);
    assert_eq!(tile_of("T40W\u{00E9}C"), None);
    assert_eq!(tile_of("T40wfc"), None);
}

#[test]
fn reference_granule_frame_matches_its_xml() {
    let frame = frame("40WFC").unwrap();
    assert_eq!((frame.zone, frame.south), (40, false));
    assert_eq!((frame.x0, frame.y1), (600_000.0, 7_800_000.0));
    assert_eq!(
        (frame.x1 - frame.x0, frame.y1 - frame.y0),
        (109_800.0, 109_800.0)
    );
}

#[test]
fn odd_zone_rows_start_at_a() {
    let frame = frame("31NAA").unwrap();
    assert_eq!((frame.x0, frame.y1), (100_000.0, 100_000.0));
    assert!(!frame.south);
}

#[test]
fn southern_band_stays_under_false_northing() {
    let frame = frame("33HVB").unwrap();
    assert!(frame.south);
    assert!(frame.y1 < 10_000_000.0 && frame.y0 > 5_000_000.0);
}

#[test]
fn garbage_codes_are_refused() {
    assert!(frame("40WIC").is_err());
    assert!(frame("00WFC").is_err());
    assert!(frame("4WFC").is_err());
    assert!(frame("40ZFC").is_err());
    // Non-ASCII 5-byte and multi-byte UTF-8 inputs must error, never panic on char boundaries.
    assert!(frame("a\u{00E9}bc").is_err());
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
