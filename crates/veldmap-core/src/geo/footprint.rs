//! Spherical scene footprint framing (`FootprintBounds`, `frame_visit`).

use glam::DVec3;

use super::geodesy;

/// Spherical bounding circle around a set of geographic coordinates or rings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FootprintBounds {
    pub lat: f64,
    pub lon: f64,
    /// Angular radius in degrees of arc (`1 deg ~ 111 km`).
    pub radius_deg: f64,
}

/// Compute the spherical framing circle enclosing all `(lat, lon)` degree
/// coordinates emitted by `visit` in two streaming passes with zero heap
/// allocation, or `None` if empty.
pub(in crate::geo) fn frame_visit(
    mut visit: impl FnMut(&mut dyn FnMut(f64, f64)),
) -> Option<FootprintBounds> {
    let mut first: Option<(f64, f64)> = None;
    let mut sum = DVec3::ZERO;
    visit(&mut |lat, lon| {
        if first.is_none() {
            first = Some((lat, lon));
        }
        sum += geodesy::unit(lat, lon);
    });
    let first = first?;
    let (lat, lon) = if sum.length() > 1e-9 {
        geodesy::angles(sum.normalize())
    } else {
        first
    };
    let mut radius_deg = 0.0_f64;
    visit(&mut |p_lat, p_lon| {
        radius_deg = radius_deg.max(geodesy::separation((lat, lon), (p_lat, p_lon)));
    });

    Some(FootprintBounds {
        lat,
        lon: geodesy::unwind(0.0, lon),
        radius_deg,
    })
}

/// Compute the spherical framing circle enclosing an iterator of `(lat, lon)`
/// degree coordinates, or `None` if empty.
#[cfg(test)]
fn frame_coords(coords: impl Iterator<Item = (f64, f64)> + Clone) -> Option<FootprintBounds> {
    frame_visit(|emit| {
        for (lat, lon) in coords.clone() {
            emit(lat, lon);
        }
    })
}

#[cfg(test)]
mod tests {
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
}
