use super::*;

#[test]
fn parses_decimal_degrees() {
    let loc = parse_location("47.2357, 39.7015").unwrap();
    assert_eq!(loc.kind, LocationKind::DecimalDegrees);
    assert!((loc.lat_deg - 47.2357).abs() < 1e-9);
    assert!((loc.lon_deg - 39.7015).abs() < 1e-9);

    let loc2 = parse_location("-33.8688 151.2093").unwrap();
    assert!((loc2.lat_deg + 33.8688).abs() < 1e-9);
    assert!((loc2.lon_deg - 151.2093).abs() < 1e-9);

    let loc3 = parse_location("33.8688S, 151.2093E").unwrap();
    assert!((loc3.lat_deg + 33.8688).abs() < 1e-9);
    assert!((loc3.lon_deg - 151.2093).abs() < 1e-9);

    // Space-separated suffix and prefix hemispheres, plus lon-first order.
    let loc4 = parse_location("33.8688S 151.2093E").unwrap();
    assert!((loc4.lat_deg + 33.8688).abs() < 1e-9);
    assert!((loc4.lon_deg - 151.2093).abs() < 1e-9);

    let loc5 = parse_location("N47.2357 E39.7015").unwrap();
    assert!((loc5.lat_deg - 47.2357).abs() < 1e-9);
    assert!((loc5.lon_deg - 39.7015).abs() < 1e-9);

    let loc6 = parse_location("39.7015E 47.2357N").unwrap();
    assert!((loc6.lat_deg - 47.2357).abs() < 1e-9);
    assert!((loc6.lon_deg - 39.7015).abs() < 1e-9);
}

#[test]
fn parses_dms_formats_in_both_hemispheres() {
    let loc = parse_location("47°14'08.5\"N 39°42'05.4\"E").unwrap();
    assert_eq!(loc.kind, LocationKind::DegreesMinutesSeconds);
    assert!((loc.lat_deg - (47.0 + 14.0 / 60.0 + 8.5 / 3600.0)).abs() < 1e-7);
    assert!((loc.lon_deg - (39.0 + 42.0 / 60.0 + 5.4 / 3600.0)).abs() < 1e-7);

    let loc2 = parse_location("47d 14m 8.5s N, 39d 42m 5.4s E").unwrap();
    assert!((loc2.lat_deg - loc.lat_deg).abs() < 1e-7);
    assert!((loc2.lon_deg - loc.lon_deg).abs() < 1e-7);

    // Space-separated `d/m/s` without comma.
    let loc_space = parse_location("47d 14m 8.5s N 39d 42m 5.4s E").unwrap();
    assert!((loc_space.lat_deg - loc.lat_deg).abs() < 1e-7);
    assert!((loc_space.lon_deg - loc.lon_deg).abs() < 1e-7);

    // Southern hemisphere DMS and DM with `S` must produce negative latitude.
    let sydney_lat = -(33.0 + 52.0 / 60.0 + 7.7 / 3600.0);
    let sydney_lon = 151.0 + 12.0 / 60.0 + 33.5 / 3600.0;
    for input in [
        "33°52'07.7\"S, 151°12'33.5\"E",
        "33°52'07.7\"S 151°12'33.5\"E",
        "33d 52m 7.7s S, 151d 12m 33.5s E",
        "33d 52m 7.7s S 151d 12m 33.5s E",
        "33d52m7.7sS 151d12m33.5sE",
        "S 33°52'07.7\" E 151°12'33.5\"",
        "-33° 52' 07.7\", 151° 12' 33.5\"",
    ] {
        let parsed = parse_location(input).unwrap_or_else(|e| panic!("{input}: {e}"));
        assert_eq!(parsed.kind, LocationKind::DegreesMinutesSeconds, "{input}");
        assert!(
            (parsed.lat_deg - sydney_lat).abs() < 1e-7,
            "{input} -> lat {}",
            parsed.lat_deg
        );
        assert!(
            (parsed.lon_deg - sydney_lon).abs() < 1e-7,
            "{input} -> lon {}",
            parsed.lon_deg
        );
    }

    let dm_south = parse_location("33° 52' S, 151° 12' E").unwrap();
    assert!((dm_south.lat_deg + (33.0 + 52.0 / 60.0)).abs() < 1e-7);
    assert!((dm_south.lon_deg - (151.0 + 12.0 / 60.0)).abs() < 1e-7);

    // Negative zero degrees must preserve negative sign on minutes/seconds.
    let neg_zero = parse_location("-0° 30' 00\", 10° 00' 00\"").unwrap();
    assert!((neg_zero.lat_deg + 0.5).abs() < 1e-9);
    assert!((neg_zero.lon_deg - 10.0).abs() < 1e-9);

    // Degrees + seconds with omitted minutes must treat `10s` as seconds, not minutes.
    let deg_sec = parse_location("45d 10s, 10 E").unwrap();
    assert_eq!(deg_sec.kind, LocationKind::DegreesMinutesSeconds);
    assert!((deg_sec.lat_deg - (45.0 + 10.0 / 3600.0)).abs() < 1e-9);
    assert!((deg_sec.lon_deg - 10.0).abs() < 1e-9);

    // Colon-separated DMS with semicolon delimiter and exact ±90/±180 boundaries.
    let colon = parse_location("47:14:08.5N; 39:42:05.4E").unwrap();
    assert!((colon.lat_deg - loc.lat_deg).abs() < 1e-7);
    assert!((colon.lon_deg - loc.lon_deg).abs() < 1e-7);
    let bounds = parse_location("-90.0, 180.0").unwrap();
    assert_eq!((bounds.lat_deg, bounds.lon_deg), (-90.0, 180.0));
}

#[test]
fn parses_mgrs_codes() {
    let loc = parse_location("T40WFC").unwrap();
    assert_eq!(loc.kind, LocationKind::Mgrs);
    assert!((69.0..70.5).contains(&loc.lat_deg));
    assert!((59.0..62.0).contains(&loc.lon_deg));
}

#[test]
fn rejects_invalid_inputs() {
    assert_eq!(parse_location("   "), Err(ParseLocationError::Empty));
    assert!(matches!(
        parse_location("95.0, 10.0"),
        Err(ParseLocationError::LatitudeOutOfRange(_))
    ));
    assert!(matches!(
        parse_location("10.0, 190.0"),
        Err(ParseLocationError::LongitudeOutOfRange(_))
    ));
    for bad in [
        "not a coordinate",
        "-33.8688S, 151.2093E",
        "N 45 S, 10 E",
        "N 45S, 10 E",
        "N 45° 10' 5\" S, 10 E",
        "47° 60' 0\", 10°",
        "47° 10' 60\", 10°",
        "47m 14d, 10 E",
        "47d 14d, 10 E",
        "NaN, 10.0",
        "inf, 10.0",
        "37MAA",
        "32XAA",
        "37UD\u{0419}",
        "37UD\u{00E9}",
    ] {
        assert!(
            matches!(
                parse_location(bad),
                Err(ParseLocationError::Unrecognized(_))
            ),
            "expected Unrecognized for '{bad}'"
        );
    }
}
