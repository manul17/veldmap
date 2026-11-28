//! High-precision `f64` bounding-ball frustum, near-plane, and ellipsoidal
//! horizon culling in ECEF metres.
//!
//! Performs conservative one-sided rejection against the four side planes and
//! Reverse-Z near plane of the `DMat4` view-projection matrix, plus the exact
//! WGS84 ellipsoidal tangent horizon plane, without any transcendental calls
//! per tested ball.

use glam::{DMat4, DVec3};

use super::geodesy;

/// Bounding sphere in `f64` ECEF metres.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ball {
    pub centre: DVec3,
    pub radius: f64,
}

impl Ball {
    /// Construct a bounding sphere centered at the AABB midpoint of `points`
    /// (in ECEF metres).
    pub(crate) fn over(points: &[DVec3]) -> Self {
        if points.is_empty() {
            return Self {
                centre: DVec3::ZERO,
                radius: 0.0,
            };
        }
        let mut low = DVec3::splat(f64::MAX);
        let mut high = DVec3::splat(f64::MIN);
        for &point in points {
            low = low.min(point);
            high = high.max(point);
        }
        let centre = (low + high) * 0.5;
        let radius = points
            .iter()
            .map(|&point| centre.distance(point))
            .fold(0.0, f64::max);
        Self { centre, radius }
    }

    /// Expand the bounding sphere radius by `by_m` metres.
    pub(crate) fn widened(self, by_m: f64) -> Self {
        Self {
            radius: self.radius + by_m,
            ..self
        }
    }
}

/// Conservative frustum + Reverse-Z near-plane + ellipsoidal-horizon culling
/// frame in `f64` ECEF metres (exact for surface geometry at `height_m <= 0`).
#[derive(Clone, Copy, Debug)]
pub struct CullFrustum {
    inward: [DVec3; 4],
    eye: DVec3,
    forward: DVec3,
    near: f64,
    towards: DVec3,
    horizon: f64,
}

impl CullFrustum {
    /// Extract the four side frustum planes, the Reverse-Z near plane, and the
    /// exact WGS84 ellipsoidal tangent horizon plane from a rotation-only
    /// `DMat4` view-projection matrix and `f64` eye in ECEF metres.
    pub(in crate::geo) fn new(view_proj: &DMat4, eye: DVec3) -> Self {
        let (row0, row1, row2, row3) = (
            view_proj.row(0),
            view_proj.row(1),
            view_proj.row(2),
            view_proj.row(3),
        );
        let inward = [row3 + row0, row3 - row0, row3 + row1, row3 - row1]
            .map(|plane| plane.truncate().normalize());

        // Reverse-Z infinite perspective: `clip.z = row2.w = z_near` and
        // `clip.w = row3.xyz · (p - eye)`, so `z_ndc <= 1.0` (`w > 0`) iff
        // `forward · (p - eye) >= near`.
        let row3_xyz = row3.truncate();
        let row3_len = row3_xyz.length();
        let (forward, near) = if row3_len > 0.0 {
            (row3_xyz / row3_len, row2.w / row3_len)
        } else {
            (DVec3::ZERO, 0.0)
        };

        // For WGS84 ellipsoid `p^T M p = a^2` with `M = diag(1, 1, 1/(1-f)^2)`,
        // a surface point `p` is front-facing iff `(M p) · (eye - p) >= 0`,
        // equivalently `p · (M eye) >= a^2`.
        let polar = 1.0 - geodesy::FLATTENING;
        let a_eye = DVec3::new(eye.x, eye.y, eye.z / (polar * polar));
        let len = a_eye.length();
        let a_sq = geodesy::SEMI_MAJOR_M * geodesy::SEMI_MAJOR_M;
        let (towards, horizon) = if len > 0.0 && len.is_finite() {
            (a_eye / len, a_sq / len)
        } else {
            (DVec3::ZERO, f64::NEG_INFINITY)
        };
        Self {
            inward,
            eye,
            forward,
            near,
            towards,
            horizon,
        }
    }

    /// Signed distance in metres of `point` above (`>= 0`) or below (`< 0`)
    /// the exact WGS84 ellipsoidal tangent horizon plane.
    pub(crate) fn horizon_margin(&self, point: DVec3) -> f64 {
        point.dot(self.towards) - self.horizon
    }

    /// Return `false` if `ball` is guaranteed invisible, or `true` if it may be
    /// visible.
    pub fn admits(&self, ball: &Ball) -> bool {
        if ball.centre.dot(self.towards) + ball.radius < self.horizon {
            return false;
        }
        let from_eye = ball.centre - self.eye;
        if self.forward.dot(from_eye) + ball.radius < self.near {
            return false;
        }
        self.inward
            .iter()
            .all(|inward| inward.dot(from_eye) + ball.radius >= 0.0)
    }
}

#[cfg(test)]
#[path = "cull_tests.rs"]
mod tests;
