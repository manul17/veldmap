use glam::DVec3;

use super::*;
use crate::geo::geodesy::{Geodetic, WGS84};
use crate::geo::ground_m;

#[test]
fn helmert_coordinate_frame_and_position_vector_conventions_agree() {
    let pv = Datum::from_towgs84(KRASOVSKY, [23.57, -140.95, -79.8, 0.0, 0.35, 0.79, -0.22]);
    assert_eq!(pv.convention, HelmertConvention::PositionVector);
    assert_eq!(PULKOVO_1942.convention, HelmertConvention::CoordinateFrame);

    for (lat, lon) in [(47.23, 39.72), (55.75, 37.62), (66.53, 66.60)] {
        let cf_wgs = PULKOVO_1942.to_wgs84(lat, lon);
        let pv_wgs = pv.to_wgs84(lat, lon);
        assert!(
            ground_m(cf_wgs, pv_wgs) < 1e-6,
            "conventions diverged: {cf_wgs:?} vs {pv_wgs:?}"
        );
        let back_pv = pv.from_wgs84(pv_wgs.0, pv_wgs.1);
        assert!(ground_m(back_pv, (lat, lon)) < 1e-3);
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
        DVec3::new(3_000_000.0, 2_000_000.0, 5_200_000.0),
        DVec3::new(-1_500_000.0, 5_800_000.0, 2_000_000.0),
        DVec3::new(4_000_000.0, -3_000_000.0, -3_500_000.0),
    ];
    let turned = Datum {
        rotate_sec: [-1.5, -0.35, -0.79],
        ..PULKOVO_1942
    };
    for datum in [PULKOVO_1942, turned] {
        for point in points {
            let back = datum.helmert_back(datum.helmert(point));
            let off = back.distance(point);
            assert!(off < 1e-6, "{:?}: {off} m", datum.rotate_sec);
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
    let polar = KRASOVSKY.semi_minor_m();
    let north = KRASOVSKY.from_ecef(DVec3::new(0.0, 0.0, polar));
    assert!((north.lat_deg - 90.0).abs() < 1e-12);
    assert_eq!(north.lon_deg, 0.0);
    assert!(north.height_m.abs() < 1e-9);

    let south = KRASOVSKY.from_ecef(DVec3::new(0.0, 0.0, -polar));
    assert!((south.lat_deg + 90.0).abs() < 1e-12);
    assert!(south.height_m.abs() < 1e-9);
}

#[test]
fn near_pole_geodetic_and_datum_roundtrips_avoid_singularity() {
    for lat_deg in [85.0, 89.5, 89.999, -89.999, 90.0, -90.0] {
        let ecef = WGS84.to_ecef(Geodetic::surface(lat_deg, 37.0));
        let back = WGS84.from_ecef(ecef);
        assert!(
            (back.lat_deg - lat_deg).abs() < 1e-12,
            "near-pole ECEF roundtrip failed at {lat_deg}: got {}",
            back.lat_deg
        );

        let shifted = PULKOVO_1942.to_wgs84(lat_deg, 37.0);
        let back = PULKOVO_1942.from_wgs84(shifted.0, shifted.1);
        assert!(back.0.is_finite() && back.1.is_finite());
        let err_m = (back.0 - lat_deg).abs().to_radians() * WGS84.semi_major_m;
        assert!(err_m < 1e-2, "near-pole lat {lat_deg} drifted by {err_m} m");
    }
}

#[test]
fn known_geographic_epsg_keeps_4xxx_crs_codes_distinct_from_6xxx_datum_codes() {
    assert!(known_geographic_epsg(4326).is_some());
    assert!(known_geographic_epsg(4284).is_some());
    assert!(known_geographic_epsg(6326).is_none());
    assert!(known_geographic_epsg(6284).is_none());
    assert!(known_datum_from_code(6326).is_some());
}

#[test]
fn to_towgs84_and_ellipsoid_from_epsg_preserve_conventions() {
    // Pulkovo 1942 uses CoordinateFrame; exporting to PositionVector negates rotations.
    assert_eq!(PULKOVO_1942.convention, HelmertConvention::CoordinateFrame);
    let pv_params = PULKOVO_1942.to_towgs84();
    assert_eq!(pv_params, [23.57, -140.95, -79.8, -0.0, 0.35, 0.79, -0.22]);
    let roundtrip = Datum::from_towgs84(PULKOVO_1942.ellipsoid, pv_params);
    assert_eq!(roundtrip.to_towgs84(), pv_params);
    assert_eq!(ellipsoid_from_epsg(7001), Some(AIRY_1830));
}
