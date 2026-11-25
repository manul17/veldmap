//! Chord and quad sagitta helpers for curved WGS84 surface meshes.
//!
//! Used for geodesic footprint densification, planetary quadtree subdivision,
//! and overlay warp-mesh culling/subdivision to bound how far flat chords dip
//! below the curved ellipsoid.

use glam::DVec3;

use super::geodesy::{FLATTENING, SEMI_MAJOR_M};

/// Maximum WGS84 radius-of-curvature ratio relative to `SEMI_MAJOR_M`
/// (`a / (b^2/a) = 1 / (1 - f)`), keeping spherical chord-sag bounds strictly
/// conservative at polar latitudes.
const POLAR_CURVATURE_SCALE: f64 = 1.0 / (1.0 - FLATTENING);

/// How far the midpoint of a spherical chord spanning `arc_deg` degrees dips
/// below the surface, in metres.
fn sag_m(arc_deg: f64) -> f64 {
    let quarter = arc_deg.to_radians() * 0.25;
    let s = quarter.sin();
    2.0 * SEMI_MAJOR_M * s * s
}

/// Conservative WGS84 oblate chord sagitta in metres for a chord spanning
/// `arc_deg` degrees.
pub(crate) fn oblate_sag_m(arc_deg: f64) -> f64 {
    sag_m(arc_deg) * POLAR_CURVATURE_SCALE
}

/// Conservative WGS84 subcell bulge margin (polar curvature + 15% trapezoid
/// slack) so a bounding sphere built from probe points strictly encloses the
/// curved ellipsoid surface across subcell diagonals up to `max_subcell_arc_deg`.
pub(crate) fn subcell_bulge_m(max_subcell_arc_deg: f64) -> f64 {
    oblate_sag_m((max_subcell_arc_deg * 1.15).min(180.0))
}

/// Estimate maximum subcell diagonal arc in degrees across a `side x side`
/// row-major grid of unit normals via squared chord length
/// (`|a - b|^2 = 4 sin^2(theta / 2)`, strictly monotonic on `[0, pi]`),
/// requiring only one `asin` instead of per-diagonal `atan2` calls.
pub(crate) fn max_subcell_arc_deg(sample_units: &[DVec3], side: usize) -> f64 {
    if side < 2 || sample_units.len() < side * side {
        return 0.0;
    }
    let mut max_chord_sq = 0.0_f64;
    for row in 0..side - 1 {
        for col in 0..side - 1 {
            let a = sample_units[row * side + col];
            let b = sample_units[(row + 1) * side + (col + 1)];
            let c = sample_units[row * side + (col + 1)];
            let d = sample_units[(row + 1) * side + col];
            max_chord_sq = max_chord_sq
                .max((a - b).length_squared())
                .max((c - d).length_squared());
        }
    }
    2.0 * (0.25 * max_chord_sq)
        .clamp(0.0, 1.0)
        .sqrt()
        .asin()
        .to_degrees()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sag_bounds_are_conservative_and_match_ellipsoid_geometry() {
        // 1 degree equatorial arc sags ~243.0 m on WGS84 (R * (1 - cos(0.5 deg))).
        assert!(
            (sag_m(1.0) - 243.0).abs() < 0.2,
            "sag_m(1.0)={}",
            sag_m(1.0)
        );
        assert!(oblate_sag_m(1.0) > sag_m(1.0));
        assert!(subcell_bulge_m(1.0) > oblate_sag_m(1.0));
    }
}
