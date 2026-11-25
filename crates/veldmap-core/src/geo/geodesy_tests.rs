use super::*;

#[test]
fn a_half_turn_stays_where_it_is() {
    assert_eq!(unwind(0.0, 180.0), 180.0);
    assert_eq!(unwind(0.0, -180.0), -180.0);
    assert_eq!(unwind(180.0, -180.0), 180.0);
    assert_eq!(unwind(-180.0, 180.0), -180.0);
}

#[test]
fn a_longitude_lands_within_a_half_turn_of_its_base() {
    for base in [-179.0, 0.0, 37.0, 179.0, 900.0] {
        let dense = [336.457_855_398_182_46, -157.459_371_577_220_8, 0.5];
        for lon in [
            -1081.0, -359.0, -181.0, -180.0, -1.0, 0.0, 180.0, 181.0, 359.0, 1081.0,
        ]
        .into_iter()
        .chain(dense)
        {
            let near = unwind(base, lon);
            assert!(
                (near - base).abs() <= 180.0,
                "base {base} and lon {lon} produced {near}"
            );
            let turns = (near - lon) / TURN_DEG;
            assert!(
                (turns - turns.round()).abs() < 1.0e-9,
                "shift {} is not a multiple of a full turn",
                near - lon
            );
        }
    }
}

#[test]
fn an_endless_longitude_does_not_hang_the_unwinding() {
    assert!(unwind(0.0, f64::INFINITY).is_nan());
    assert!(unwind(0.0, f64::NEG_INFINITY).is_nan());
    assert!(unwind(0.0, f64::NAN).is_nan());
    assert!(unwind(f64::INFINITY, 0.0).is_infinite());
    assert_eq!(unwind(0.0, 1.0e300), 0.0);
    assert_eq!(unwind(0.0, f64::MAX), 0.0);
}

#[test]
fn surface_roundtrip() {
    for &(lat, lon) in &[(0.0, 0.0), (45.0, 60.0), (-33.9, 151.2), (78.0, -179.5)] {
        let (back_lat, back_lon) = surface_at(world(Geodetic::surface(lat, lon)));
        assert!((back_lat - lat).abs() < 1e-12, "{lat} -> {back_lat}");
        assert!((back_lon - lon).abs() < 1e-12, "{lon} -> {back_lon}");
    }
}

#[test]
fn fukushima_3d_roundtrip_across_altitudes_and_poles() {
    let ellipsoids = [
        WGS84,
        Ellipsoid {
            semi_major_m: 6_378_245.0,
            flattening: 1.0 / 298.3,
        },
    ];
    for ellipsoid in ellipsoids {
        for lat_deg in [
            -90.0,
            -89.999,
            -85.051_129,
            -45.0,
            -0.001,
            0.0,
            0.001,
            45.0,
            85.051_129,
            89.999,
            90.0,
        ] {
            for lon_deg in [-179.5, -45.0, 0.0, 39.7015, 179.9] {
                for height_m in [-500.0, 0.0, 2.0, 10_000.0, 1_000_000.0, 80_000_000.0] {
                    let input = Geodetic {
                        lat_deg,
                        lon_deg,
                        height_m,
                    };
                    let ecef = ellipsoid.to_ecef(input);
                    let back = ellipsoid.from_ecef(ecef);
                    assert!(
                        (back.lat_deg - lat_deg).abs() < 1e-11,
                        "lat mismatch at {input:?}: got {back:?}"
                    );
                    if lat_deg.abs() < 90.0 {
                        assert!(
                            (unwind(lon_deg, back.lon_deg) - lon_deg).abs() < 1e-11,
                            "lon mismatch at {input:?}: got {back:?}"
                        );
                    } else {
                        assert!(back.lon_deg.is_finite());
                    }
                    let h_tol = 1e-6_f64.max(height_m.abs() * 1e-12);
                    assert!(
                        (back.height_m - height_m).abs() < h_tol,
                        "height mismatch at {input:?}: got {back:?} (tol {h_tol})"
                    );
                }
            }
        }
    }
}

#[test]
fn flattening_is_real_in_metres() {
    let point = world(Geodetic::surface(45.0, 0.0));
    assert!(
        point.z < SEMI_MAJOR_M * 45.0_f64.to_radians().sin() - 1_000.0,
        "z = {}",
        point.z
    );
    assert!((world(Geodetic::surface(0.0, 0.0)).x - SEMI_MAJOR_M).abs() < 1e-9);
    let polar = world(Geodetic::surface(90.0, 0.0)).z;
    assert!(
        ((SEMI_MAJOR_M - polar) - 21_384.7).abs() < 1.0,
        "polar compression ~21.4 km: {polar}"
    );
}

#[test]
fn between_follows_the_arc_not_the_degrees() {
    let (ne, nw) = ((73.395_172, 2.707_061), (74.523_872, -10.408_985));
    let arc = between(ne, nw, 0.5);
    let flat = ((ne.0 + nw.0) / 2.0, (ne.1 + nw.1) / 2.0);
    let apart = separation(arc, flat).to_radians() * SEMI_MAJOR_M;
    assert!(
        (12_000.0..14_000.0).contains(&apart),
        "diverged by {apart:.0} m"
    );
    assert!(arc.0 > flat.0, "{} vs {}", arc.0, flat.0);

    assert_eq!(between(ne, nw, 0.0), ne);
    assert_eq!(between(ne, nw, 1.0), nw);
}

#[test]
fn between_crosses_the_antimeridian_the_short_way() {
    let (lat, lon) = between((10.0, 179.0), (10.0, -179.0), 0.5);
    assert!((lon - 180.0).abs() < 1e-9, "{lon}");
    assert!(lat > 10.0, "northern hemisphere arc bows poleward: {lat}");
    assert!((separation((10.0, 179.0), (10.0, -179.0)) - 1.9696).abs() < 1e-3);
}

#[test]
fn surface_ecef_and_normal_matches_world_and_unit() {
    for &(lat, lon) in &[
        (0.0, 0.0),
        (47.2357, 39.7015),
        (-85.051_129, -179.5),
        (90.0, 0.0),
    ] {
        let (ecef, normal) = WGS84.surface_ecef_and_normal(lat, lon);
        assert!(ecef.distance(world(Geodetic::surface(lat, lon))) < 1e-9);
        assert!(normal.distance(unit(lat, lon)) < 1e-15);
    }
}

#[test]
fn ray_hits_and_misses_in_metres() {
    let eye = DVec3::new(2.0 * SEMI_MAJOR_M, 0.0, 0.0);
    let hit = intersect_at(eye, DVec3::NEG_X, 0.0).expect("ray toward center");
    let (lat, lon) = surface_at(hit);
    assert!(lat.abs() < 1e-12 && lon.abs() < 1e-12);

    let raised = intersect_at(eye, DVec3::NEG_X, 1_000.0).expect("ray toward raised surface");
    let lifted = raised.x - hit.x;
    assert!((lifted - 1_000.0).abs() < 1e-6, "lift measured {lifted} m");

    // 2 m close-up ray intersection preserves sub-millimetre precision.
    let low_eye = world(Geodetic {
        lat_deg: 47.2357,
        lon_deg: 39.7015,
        height_m: 2.0,
    });
    let nadir = -unit(47.2357, 39.7015);
    let low_hit = intersect_at(low_eye, nadir, 0.0).expect("2 m nadir hit");
    let expected = world(Geodetic::surface(47.2357, 39.7015));
    assert!(
        low_hit.distance(expected) < 1e-6,
        "2 m nadir hit drifted by {} m",
        low_hit.distance(expected)
    );

    assert!(intersect_at(eye, DVec3::Y, 0.0).is_none());
    assert!(intersect_at(eye, DVec3::X, 0.0).is_none());
    assert!(intersect_at(eye, DVec3::ZERO, 0.0).is_none());
}

#[test]
fn sub_metre_separation_and_between_preserve_precision() {
    let p0 = (47.2357, 39.7015);
    let dlat = 0.01 / (SEMI_MAJOR_M * std::f64::consts::PI / 180.0);
    let p1 = (p0.0 + dlat, p0.1);
    let sep_m = separation(p0, p1).to_radians() * SEMI_MAJOR_M;
    assert!((sep_m - 0.01).abs() < 1e-6, "sep_m={sep_m}");

    let mid = between(p0, p1, 0.5);
    assert!((mid.0 - (p0.0 + 0.5 * dlat)).abs() < 1e-12);
    assert!((mid.1 - p0.1).abs() < 1e-12);
}

#[test]
fn surface_at_exact_poles_is_finite() {
    for pole in [90.0, -90.0] {
        let (lat, lon) = surface_at(world(Geodetic::surface(pole, 0.0)));
        assert!((lat - pole).abs() < 1e-12);
        assert!(lon.is_finite());
    }
}
