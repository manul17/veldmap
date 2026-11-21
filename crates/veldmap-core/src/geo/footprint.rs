//! Scene footprint framing and spherical point-in-polygon hit testing.
//!
//! Uses unit-vector azimuth winding along the exact edge tracing rules in
//! [`super::geodesy`] so clicking on a footprint agrees with how it is drawn,
//! including across the antimeridian, around the poles, and for global bands.

use glam::DVec3;

use super::geodesy;

/// A latitude/longitude coordinate pair in WGS84 degrees.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeoPoint {
    pub lat: f64,
    pub lon: f64,
}

impl GeoPoint {
    pub const fn new(lat: f64, lon: f64) -> Self {
        Self { lat, lon }
    }
}

/// A closed ring of geographic vertices.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Ring {
    pub points: Vec<GeoPoint>,
}

/// Spherical bounding circle around a set of footprint rings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frame {
    pub lat: f64,
    pub lon: f64,
    /// Angular radius in degrees of arc (`1 deg ~ 111 km`).
    pub radius_deg: f64,
}

/// Compute the spherical framing circle enclosing `rings`, or `None` if empty.
pub fn frame(rings: &[Ring]) -> Option<Frame> {
    let first = rings.iter().flat_map(|ring| &ring.points).next()?;
    let (lat, lon) = middle(points(rings)).unwrap_or((first.lat, first.lon));
    let radius_deg = points(rings)
        .map(|point| geodesy::separation((lat, lon), point))
        .fold(0.0, f64::max);

    Some(Frame {
        lat,
        lon: wrap(lon),
        radius_deg,
    })
}

/// Return whether any ring in `rings` covers `(lat, lon)`.
pub fn covers(rings: &[Ring], lat: f64, lon: f64) -> bool {
    rings.iter().any(|ring| inside(&angles(ring), lat, lon))
}

fn points(rings: &[Ring]) -> impl Iterator<Item = (f64, f64)> + '_ {
    rings
        .iter()
        .flat_map(|ring| &ring.points)
        .map(|point| (point.lat, point.lon))
}

fn angles(ring: &Ring) -> Vec<(f64, f64)> {
    ring.points.iter().map(|p| (p.lat, p.lon)).collect()
}

fn traced(ring: &[(f64, f64)]) -> Vec<(f64, f64)> {
    const MAX_EDGE_DEG: f64 = 0.25;
    const MOST: f64 = geodesy::TURN_DEG / MAX_EDGE_DEG;

    let mut out = Vec::with_capacity(ring.len());
    for (at, &from) in ring.iter().enumerate() {
        let to = ring[(at + 1) % ring.len()];
        let raw = (geodesy::edge_span(from, to) / MAX_EDGE_DEG).ceil();
        let steps = if raw.is_finite() {
            raw.clamp(1.0, MOST) as u32
        } else {
            1
        };
        for step in 0..steps {
            out.push(geodesy::edge_point(
                from,
                to,
                f64::from(step) / f64::from(steps),
            ));
        }
    }
    out
}

fn inside(ring: &[(f64, f64)], lat: f64, lon: f64) -> bool {
    if ring.len() < 3 {
        return false;
    }
    let traced = traced(ring);
    let turned = turn(&traced, lat, lon);
    if turned.abs() > 180.0 {
        let Some((centre_lat, centre_lon)) = middle(ring.iter().copied()) else {
            return true;
        };
        let centre_turn = turn(&traced, centre_lat, centre_lon);
        if centre_turn.abs() > 180.0 {
            return turned.signum() == centre_turn.signum();
        }
        // Concave ring whose spherical centroid falls outside: reject the
        // antipodal image by requiring the same hemisphere as the centroid.
        return geodesy::unit(lat, lon).dot(geodesy::unit(centre_lat, centre_lon)) > 0.0;
    }

    geodesy::encircles(&traced) && {
        let (low, high) = ring.iter().fold((f64::MAX, f64::MIN), |(low, high), p| {
            (low.min(p.0), high.max(p.0))
        });
        (low..=high).contains(&lat)
    }
}

fn turn(ring: &[(f64, f64)], lat: f64, lon: f64) -> f64 {
    let mut turned = 0.0;
    for (index, from) in ring.iter().enumerate() {
        let to = ring[(index + 1) % ring.len()];
        turned += wrap(bearing(lat, lon, to.0, to.1) - bearing(lat, lon, from.0, from.1));
    }
    turned
}

fn middle(points: impl Iterator<Item = (f64, f64)>) -> Option<(f64, f64)> {
    let sum: DVec3 = points.map(|(lat, lon)| geodesy::unit(lat, lon)).sum();
    (sum.length() > 1e-9).then(|| geodesy::angles(sum.normalize()))
}

fn bearing(from_lat: f64, from_lon: f64, to_lat: f64, to_lon: f64) -> f64 {
    let (from, to) = (from_lat.to_radians(), to_lat.to_radians());
    let delta = (to_lon - from_lon).to_radians();
    let east = delta.sin() * to.cos();
    let north = from.cos() * to.sin() - from.sin() * to.cos() * delta.cos();
    east.atan2(north).to_degrees()
}

fn wrap(degrees: f64) -> f64 {
    geodesy::unwind(0.0, degrees)
}

#[cfg(test)]
mod tests {
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
}
