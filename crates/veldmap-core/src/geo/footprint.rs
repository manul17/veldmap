//! Spherical scene footprint framing (`FootprintBounds`, `frame_coords`).

use super::geodesy;

/// Spherical bounding circle around a set of geographic coordinates or rings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FootprintBounds {
    pub lat: f64,
    pub lon: f64,
    /// Angular radius in degrees of arc (`1 deg ~ 111 km`).
    pub radius_deg: f64,
}

/// Compute the spherical framing circle enclosing an iterator of `(lat, lon)`
/// degree coordinates, or `None` if empty.
pub(crate) fn frame_coords(
    coords: impl Iterator<Item = (f64, f64)> + Clone,
) -> Option<FootprintBounds> {
    let first = coords.clone().next()?;
    let (lat, lon) = geodesy::spherical_centroid(coords.clone()).unwrap_or(first);
    let radius_deg = coords
        .map(|point| geodesy::separation((lat, lon), point))
        .fold(0.0, f64::max);

    Some(FootprintBounds {
        lat,
        lon: geodesy::unwind(0.0, lon),
        radius_deg,
    })
}

#[cfg(test)]
#[path = "footprint_tests.rs"]
mod tests;
