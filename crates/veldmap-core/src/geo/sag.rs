//! Chord and quad sagitta helpers for curved WGS84 surface meshes.
//!
//! Used by tile skirts (Slice 2), surface outlines, and raster warp-mesh
//! subdivision (Slice 3A-1) to bound how far flat chords dip below the curved
//! ellipsoid surface.

use super::geodesy::SEMI_MAJOR_M;

/// Maximum chord sagitta tolerance used for surface outline densification
/// (metres). Keeps outline chords visually tight against the ellipsoid.
pub const DEFAULT_SAG_TOLERANCE_M: f64 = 70.0;

/// How far the midpoint of a spherical chord spanning `arc_deg` degrees dips
/// below the surface, in metres.
pub fn sag_m(arc_deg: f64) -> f64 {
    SEMI_MAJOR_M * (1.0 - (arc_deg.to_radians() * 0.5).cos())
}

/// Maximum chord arc in degrees whose sagitta stays within `tolerance_m`
/// metres. The inverse of [`sag_m`].
pub fn max_chord_deg(tolerance_m: f64) -> f64 {
    let sag = (tolerance_m.max(0.0)) / SEMI_MAJOR_M;
    2.0 * (1.0 - sag).clamp(-1.0, 1.0).acos().to_degrees()
}

/// Maximum quad side in degrees whose triangle diagonal stays within
/// `tolerance_m` metres of sagitta (`max_chord_deg / sqrt(2)`).
pub fn max_quad_side_deg(tolerance_m: f64) -> f64 {
    max_chord_deg(tolerance_m) / std::f64::consts::SQRT_2
}

/// Required radial skirt or lift in metres for a chord of `chord_deg` degrees
/// given a base height and clearance margin.
pub fn lift_m(chord_deg: f64, base_lift_m: f64, clearance_m: f64) -> f64 {
    base_lift_m.max(sag_m(chord_deg) + clearance_m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limit_chord_sags_exactly_to_tolerance() {
        let tolerance_m = 70.0;
        let sag = sag_m(max_chord_deg(tolerance_m));
        assert!(
            (sag - tolerance_m).abs() < 1e-6,
            "limit chord sag {sag} m != {tolerance_m} m"
        );
    }

    #[test]
    fn finely_divided_quad_stays_within_base_lift() {
        let base_lift = 80.0;
        let clearance = 10.0;
        let tol = base_lift - clearance;
        let diagonal = max_quad_side_deg(tol) * std::f64::consts::SQRT_2;
        assert!(diagonal <= max_chord_deg(tol) + 1e-9);
        for chord in [max_chord_deg(tol), diagonal, max_quad_side_deg(tol), 0.0] {
            assert!((lift_m(chord, base_lift, clearance) - base_lift).abs() < 1e-6);
        }
    }

    #[test]
    fn coarse_chord_rises_by_its_own_sag_plus_clearance() {
        let base_lift = 80.0;
        let clearance = 10.0;
        let coarse = 8.0;
        let raised = lift_m(coarse, base_lift, clearance);
        assert!(raised > base_lift);
        assert!((raised - sag_m(coarse) - clearance).abs() < 1e-9);
    }
}
