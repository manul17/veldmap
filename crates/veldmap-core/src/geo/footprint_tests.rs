use super::*;

fn ring(points: &[(f64, f64)]) -> Ring {
    Ring {
        points: points
            .iter()
            .map(|&(lat, lon)| GeoPoint { lat, lon })
            .collect(),
    }
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
    let rings = [ring(&[
        (10.0, 10.0),
        (10.0, 20.0),
        (20.0, 20.0),
        (20.0, 10.0),
    ])];
    assert!(covers(&rings, 15.0, 15.0));
    assert!(!covers(&rings, 25.0, 15.0));
    assert!(!covers(&rings, 15.0, 25.0));
}

#[test]
fn covers_across_the_seam() {
    let rings = [ring(&[
        (10.0, 175.0),
        (10.0, -175.0),
        (20.0, -175.0),
        (20.0, 175.0),
    ])];
    assert!(covers(&rings, 15.0, 180.0));
    assert!(covers(&rings, 15.0, -178.0));
    assert!(!covers(&rings, 15.0, 170.0));
    assert!(!covers(&rings, 15.0, -170.0));
}

#[test]
fn frame_stays_on_the_seam() {
    let rings = [ring(&[
        (10.0, 175.0),
        (10.0, -175.0),
        (20.0, -175.0),
        (20.0, 175.0),
    ])];
    let f = frame(&rings).expect("ring has geometry");
    assert!((f.lat - 15.0).abs() < 0.2);
    assert!(f.lon.abs() > 179.0);
    assert!(f.radius_deg > 5.0 && f.radius_deg < 12.0);
}

#[test]
fn frame_of_nothing_is_none() {
    assert!(frame(&[]).is_none());
    assert!(frame(&[ring(&[])]).is_none());
}

#[test]
fn covers_a_ring_around_the_pole() {
    let around: Vec<(f64, f64)> = (0..12)
        .map(|step| (80.0, -180.0 + f64::from(step) * 30.0))
        .collect();
    let rings = [ring(&around)];

    assert!(covers(&rings, 89.0, 0.0));
    assert!(covers(&rings, 89.0, 150.0));
    assert!(covers(&rings, 85.0, -60.0));
    assert!(!covers(&rings, 70.0, 0.0));
    assert!(!covers(&rings, -85.0, 0.0));
}

#[test]
fn covers_agrees_with_the_drawn_box() {
    let rings = [ring(&[
        (72.0, 10.0),
        (72.0, 30.0),
        (73.0, 30.0),
        (73.0, 10.0),
    ])];

    assert!(covers(&rings, 72.5, 20.0));
    assert!(covers(&rings, 72.1, 20.0));
    assert!(!covers(&rings, 73.1, 20.0));
    assert!(!covers(&rings, 71.9, 20.0));
    assert!(!covers(&rings, 73.01, 20.0));
}

#[test]
fn covers_a_whole_earth_band() {
    let rings = [ring(&[
        (-70.0125, -180.0125),
        (70.0125, -180.0125),
        (70.0125, 179.9875),
        (-70.0125, 179.9875),
    ])];

    assert!(covers(&rings, 0.0, 0.0));
    assert!(covers(&rings, 45.0, 30.0));
    assert!(covers(&rings, -69.0, 150.0));
    assert!(!covers(&rings, 80.0, 0.0));
    assert!(!covers(&rings, -80.0, 50.0));
}

#[test]
fn covers_a_band_written_with_extra_vertices() {
    let rings = [ring(&[
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
    ])];

    assert!(covers(&rings, 0.0, 0.0));
    assert!(covers(&rings, 45.0, 30.0));
    assert!(!covers(&rings, 80.0, 0.0));
}

#[test]
fn frame_of_a_polar_ring_sits_on_the_pole() {
    let around: Vec<(f64, f64)> = (0..12)
        .map(|step| (80.0, -180.0 + f64::from(step) * 30.0))
        .collect();
    let f = frame(&[ring(&around)]).expect("ring has geometry");
    assert!(f.lat > 89.9);
    assert!((f.radius_deg - 10.0).abs() < 1e-6);
}

#[test]
fn covers_a_concave_clockwise_ring_whose_centroid_is_outside() {
    // Clockwise C-shaped ring around (10, 10); centroid falls in the notch at ~(10, 10).
    let c_shape = [ring(&[
        (0.0, 0.0),
        (20.0, 0.0),
        (20.0, 20.0),
        (16.0, 20.0),
        (16.0, 4.0),
        (4.0, 4.0),
        (4.0, 20.0),
        (0.0, 20.0),
    ])];
    assert!(covers(&c_shape, 10.0, 2.0));
    assert!(covers(&c_shape, 18.0, 15.0));
    assert!(!covers(&c_shape, 10.0, 12.0)); // inside notch
    assert!(!covers(&c_shape, -10.0, -178.0)); // antipode
}

#[test]
fn tilted_polar_ring_rejects_exterior_points_within_its_latitude_span() {
    let tilted = [ring(&[(80.0, 0.0), (80.0, 120.0), (88.0, -120.0)])];
    assert!(covers(&tilted, 89.0, 0.0));
    assert!(!covers(&tilted, 82.0, -120.0));
    assert!(!covers(&tilted, -85.0, 0.0));

    // GeoJSON-style closed ring (first == last) must behave identically.
    let closed = [ring(&[
        (80.0, 0.0),
        (80.0, 120.0),
        (88.0, -120.0),
        (80.0, 0.0),
    ])];
    assert!(covers(&closed, 89.0, 0.0));
    assert!(!covers(&closed, 82.0, -120.0));
}
