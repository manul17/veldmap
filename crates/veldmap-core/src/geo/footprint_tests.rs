use super::*;

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
fn frame_coords_of_a_polar_ring_sits_on_the_pole() {
    let around: Vec<(f64, f64)> = (0..12)
        .map(|step| (80.0, -180.0 + f64::from(step) * 30.0))
        .collect();
    let f = frame_coords(around.iter().copied()).expect("ring has geometry");
    assert!(f.lat > 89.9);
    assert!((f.radius_deg - 10.0).abs() < 1e-6);
}
