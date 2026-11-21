//! Chord and quad sagitta helpers for curved WGS84 surface meshes.
//!
//! Used by tile skirts, surface outlines, and raster warp-mesh subdivision to
//! bound how far flat chords dip below the curved ellipsoid surface.

use super::geodesy::SEMI_MAJOR_M;

/// Maximum chord sagitta tolerance used for surface outline densification
/// (metres). Keeps outline chords visually tight against the ellipsoid.
pub const DEFAULT_SAG_TOLERANCE_M: f64 = 70.0;

/// How far the midpoint of a spherical chord spanning `arc_deg` degrees dips
/// below the surface, in metres.
pub fn sag_m(arc_deg: f64) -> f64 {
    let quarter = arc_deg.to_radians() * 0.25;
    let s = quarter.sin();
    2.0 * SEMI_MAJOR_M * s * s
}

/// Maximum chord arc in degrees whose sagitta stays within `tolerance_m`
/// metres. The inverse of [`sag_m`].
pub fn max_chord_deg(tolerance_m: f64) -> f64 {
    let half_sag = (tolerance_m.max(0.0) / (2.0 * SEMI_MAJOR_M)).clamp(0.0, 1.0);
    4.0 * half_sag.sqrt().asin().to_degrees()
}

/// Maximum quad side in degrees whose triangle diagonal stays within
/// `tolerance_m` metres of sagitta (`max_chord_deg / sqrt(2)`).
pub fn max_quad_side_deg(tolerance_m: f64) -> f64 {
    max_chord_deg(tolerance_m) / std::f64::consts::SQRT_2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limit_chord_sags_exactly_to_tolerance() {
        for tolerance_m in [0.001, 1.0, 70.0, 10_000.0] {
            let sag = sag_m(max_chord_deg(tolerance_m));
            assert!(
                (sag - tolerance_m).abs() < 1e-9,
                "limit chord sag {sag} m != {tolerance_m} m"
            );
        }
    }

    #[test]
    fn quad_diagonal_stays_within_tolerance() {
        let tol = 70.0;
        let diagonal = max_quad_side_deg(tol) * std::f64::consts::SQRT_2;
        assert!((sag_m(diagonal) - tol).abs() < 1e-6);
    }
}
