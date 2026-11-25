use super::*;

fn ring(points: &[(f64, f64)]) -> Ring {
    Ring {
        points: points
            .iter()
            .map(|&(lat, lon)| GeoPoint { lat, lon })
            .collect(),
    }
}

fn polygon(points: &[(f64, f64)]) -> Polygon {
    Polygon::new(ring(points))
}

#[test]
fn a_ring_wider_than_the_earth_is_not_traced_forever() {
    let endless = traced(&[(0.0, 0.0), (1.0e9, 0.0), (0.0, 1.0)]);
    assert!(endless.len() < 5000, "vertices {}", endless.len());

    let nowhere = traced(&[(0.0, 0.0), (f64::NAN, 0.0), (0.0, 1.0)]);
    assert!(nowhere.len() < 20, "vertices {}", nowhere.len());
}

#[test]
fn covers_a_plain_square() {
    let poly = polygon(&[(10.0, 10.0), (10.0, 20.0), (20.0, 20.0), (20.0, 10.0)]);
    assert!(covers_polygon(&poly, 15.0, 15.0));
    assert!(!covers_polygon(&poly, 25.0, 15.0));
    assert!(!covers_polygon(&poly, 15.0, 25.0));
}

#[test]
fn covers_across_the_seam() {
    let poly = polygon(&[(10.0, 175.0), (10.0, -175.0), (20.0, -175.0), (20.0, 175.0)]);
    assert!(covers_polygon(&poly, 15.0, 180.0));
    assert!(covers_polygon(&poly, 15.0, -178.0));
    assert!(!covers_polygon(&poly, 15.0, 170.0));
    assert!(!covers_polygon(&poly, 15.0, -170.0));
}

#[test]
fn frame_coords_stays_on_the_seam() {
    let pts = [(10.0, 175.0), (10.0, -175.0), (20.0, -175.0), (20.0, 175.0)];
    let f = frame_coords(pts.iter().copied()).expect("coords have geometry");
    assert!((f.lat - 15.0).abs() < 0.2);
    assert!(f.lon.abs() > 179.0);
    assert!(f.radius_deg > 5.0 && f.radius_deg < 12.0);
}

#[test]
fn frame_coords_of_nothing_is_none() {
    assert!(frame_coords(std::iter::empty()).is_none());
}

#[test]
fn covers_a_ring_around_the_pole() {
    let around: Vec<(f64, f64)> = (0..12)
        .map(|step| (80.0, -180.0 + f64::from(step) * 30.0))
        .collect();
    let poly = polygon(&around);

    assert!(covers_polygon(&poly, 89.0, 0.0));
    assert!(covers_polygon(&poly, 89.0, 150.0));
    assert!(covers_polygon(&poly, 85.0, -60.0));
    assert!(!covers_polygon(&poly, 70.0, 0.0));
    assert!(!covers_polygon(&poly, -85.0, 0.0));
}

#[test]
fn covers_agrees_with_the_drawn_box() {
    let poly = polygon(&[(72.0, 10.0), (72.0, 30.0), (73.0, 30.0), (73.0, 10.0)]);

    assert!(covers_polygon(&poly, 72.5, 20.0));
    assert!(covers_polygon(&poly, 72.1, 20.0));
    assert!(!covers_polygon(&poly, 73.1, 20.0));
    assert!(!covers_polygon(&poly, 71.9, 20.0));
    assert!(!covers_polygon(&poly, 73.01, 20.0));
}

#[test]
fn covers_a_whole_earth_band() {
    let poly = polygon(&[
        (-70.0125, -180.0125),
        (70.0125, -180.0125),
        (70.0125, 179.9875),
        (-70.0125, 179.9875),
    ]);

    assert!(covers_polygon(&poly, 0.0, 0.0));
    assert!(covers_polygon(&poly, 45.0, 30.0));
    assert!(covers_polygon(&poly, -69.0, 150.0));
    assert!(!covers_polygon(&poly, 80.0, 0.0));
    assert!(!covers_polygon(&poly, -80.0, 50.0));
}

#[test]
fn covers_a_band_written_with_extra_vertices() {
    let poly = polygon(&[
        (-70.0125, -180.0125),
        (70.0125, -180.0125),
        (70.0125, -90.0),
        (70.0125, 0.0),
        (70.0125, 90.0),
        (70.0125, 179.9875),
        (-70.0125, 179.9875),
        (-70.0125, 90.0),
        (-70.0125, 0.0),
        (-70.0125, -90.0),
    ]);

    assert!(covers_polygon(&poly, 0.0, 0.0));
    assert!(covers_polygon(&poly, 45.0, 30.0));
    assert!(!covers_polygon(&poly, 80.0, 0.0));
}

#[test]
fn frame_coords_of_a_polar_ring_sits_on_the_pole() {
    let around: Vec<(f64, f64)> = (0..12)
        .map(|step| (80.0, -180.0 + f64::from(step) * 30.0))
        .collect();
    let f = frame_coords(around.iter().copied()).expect("ring has geometry");
    assert!(f.lat > 89.9);
    assert!((f.radius_deg - 10.0).abs() < 1e-6);
}

#[test]
fn covers_a_concave_clockwise_ring_whose_centroid_is_outside() {
    // Clockwise C-shaped ring around (10, 10); centroid falls in the notch at ~(10, 10).
    let c_shape = polygon(&[
        (0.0, 0.0),
        (20.0, 0.0),
        (20.0, 20.0),
        (16.0, 20.0),
        (16.0, 4.0),
        (4.0, 4.0),
        (4.0, 20.0),
        (0.0, 20.0),
    ]);
    assert!(covers_polygon(&c_shape, 10.0, 2.0));
    assert!(covers_polygon(&c_shape, 18.0, 15.0));
    assert!(!covers_polygon(&c_shape, 10.0, 12.0)); // inside notch
    assert!(!covers_polygon(&c_shape, -10.0, -178.0)); // antipode
}

#[test]
fn tilted_polar_ring_rejects_exterior_points_within_its_latitude_span() {
    let tilted = polygon(&[(80.0, 0.0), (80.0, 120.0), (88.0, -120.0)]);
    assert!(covers_polygon(&tilted, 89.0, 0.0));
    assert!(!covers_polygon(&tilted, 82.0, -120.0));
    assert!(!covers_polygon(&tilted, -85.0, 0.0));

    // GeoJSON-style closed ring (first == last) must behave identically.
    let closed = polygon(&[(80.0, 0.0), (80.0, 120.0), (88.0, -120.0), (80.0, 0.0)]);
    assert!(covers_polygon(&closed, 89.0, 0.0));
    assert!(!covers_polygon(&closed, 82.0, -120.0));
}

#[test]
fn an_edge_goes_the_way_it_is_named() {
    let (west, east) = ((60.0, 10.0), (60.0, 30.0));
    assert_eq!(edge_point(west, east, 0.5), (60.0, 20.0));
    assert!(geodesy::between(west, east, 0.5).0 > 60.0);

    let (south, north) = ((50.0, 10.0), (70.0, 10.0));
    assert_eq!(edge_point(south, north, 0.5), (60.0, 10.0));

    let (from, to) = ((73.395_172, 2.707_061), (74.523_872, -10.408_985));
    assert_eq!(edge_point(from, to, 0.5), geodesy::between(from, to, 0.5));
}

#[test]
fn every_kind_of_edge_lands_on_its_ends() {
    let edges = [
        ((60.0, 10.0), (60.0, 30.0)),
        ((50.0, 10.0), (70.0, 10.0)),
        ((73.4, 2.7), (74.5, -10.4)),
        ((10.0, 179.0), (10.0, -179.0)),
        ((-78.676_191, 12.0), (1.323_561, 12.0)),
        ((43.0, -78.676_191), (43.0, 1.323_561)),
    ];
    for (from, to) in edges {
        assert_eq!(edge_point(from, to, 0.0), from);
        let end = edge_point(from, to, 1.0);
        let apart = (end.0 - to.0).abs() + (geodesy::unwind(to.1, end.1) - to.1).abs();
        assert!(
            apart < 1e-9,
            "edge end {from:?} -> {to:?} off by {apart} deg"
        );
    }

    let (west, east) = ((80.0, -180.0125), (80.0, 179.9875));
    assert_eq!(edge_point(west, east, 1.0).1, 179.9875);
}

#[test]
fn a_full_circle_stays_a_circle_and_a_seam_takes_the_short_way() {
    let (west, east) = ((80.0, -180.0125), (80.0, 179.9875));
    assert!((sweep(west, east) - 360.0).abs() < 0.05);
    assert!(edge_point(west, east, 0.5).1.abs() < 0.05);

    let (from, to) = ((10.0, 170.0), (10.0, -170.0));
    assert_eq!(sweep(from, to), 20.0);
    assert_eq!(edge_point(from, to, 0.5), (10.0, 180.0));
}

#[test]
fn an_edge_is_measured_by_the_way_it_goes() {
    assert!((edge_span((70.0, 10.0), (70.0, 30.0)) - 6.8404).abs() < 1e-4);
    assert_eq!(edge_span((50.0, 10.0), (70.0, 10.0)), 20.0);
    assert!((edge_span((73.395_172, 2.707_061), (74.523_872, -10.408_985)) - 3.7868).abs() < 1e-4);
}

#[test]
fn trace_edge_handles_non_positive_and_non_finite_limits_without_panicking() {
    for bad in [
        0.0,
        -1.0,
        -1e-9,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        1e308,
    ] {
        let mut emitted = Vec::new();
        trace_edge((10.0, 20.0), (15.0, 25.0), bad, |p| emitted.push(p));
        assert_eq!(emitted, vec![(10.0, 20.0)], "max_edge_deg={bad}");
    }
}

#[test]
fn dedup_geo_ring_removes_consecutive_wrap_around_and_non_finite_points() {
    let pts = [
        GeoPoint::new(0.0, 0.0),
        GeoPoint::new(0.0, 0.0),
        GeoPoint::new(f64::NAN, 1.0),
        GeoPoint::new(1.0, 0.0),
        GeoPoint::new(1.0, 1.0),
        GeoPoint::new(1.0, 1.0),
        GeoPoint::new(0.0, 0.0),
    ];
    assert_eq!(
        dedup_geo_ring(&pts),
        vec![
            GeoPoint::new(0.0, 0.0),
            GeoPoint::new(1.0, 0.0),
            GeoPoint::new(1.0, 1.0),
        ]
    );
    assert_eq!(
        dedup_geo_ring(&[GeoPoint::new(1.0, 2.0), GeoPoint::new(1.0, 2.0)]),
        vec![GeoPoint::new(1.0, 2.0)]
    );
    assert!(dedup_geo_ring(&[]).is_empty());
}

#[test]
fn covers_polygons_checks_disjoint_polygons_and_respects_holes() {
    let donut = Polygon::with_holes(
        ring(&[(0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0)]),
        vec![ring(&[(3.0, 3.0), (3.0, 7.0), (7.0, 7.0), (7.0, 3.0)])],
    );
    let island = Polygon::new(ring(&[
        (20.0, 20.0),
        (20.0, 25.0),
        (25.0, 25.0),
        (25.0, 20.0),
    ]));
    let polygons = [donut, island];
    assert!(covers_polygons(&polygons, 1.0, 1.0));
    assert!(!covers_polygons(&polygons, 5.0, 5.0)); // inside hole
    assert!(covers_polygons(&polygons, 22.0, 22.0)); // second polygon
    assert!(!covers_polygons(&polygons, 15.0, 15.0)); // between polygons
    assert!(!covers_polygons(&[], 1.0, 1.0));
}

#[test]
fn band_seam_cuts_identifies_whole_earth_bands_and_rejects_local_boxes() {
    let band = [
        (-70.0125, -180.0125),
        (70.0125, -180.0125),
        (70.0125, 179.9875),
        (-70.0125, 179.9875),
    ];
    assert_eq!(band_seam_cuts(&band), vec![0, 2]);

    let local = [(10.0, 10.0), (20.0, 10.0), (20.0, 20.0), (10.0, 20.0)];
    assert!(band_seam_cuts(&local).is_empty());
}
