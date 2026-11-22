//! High-precision `f64` bounding-ball frustum and horizon culling.
//!
//! Performs conservative one-sided rejection against the four side planes of
//! the Reverse-Z `DMat4` view-projection matrix and the exact WGS84
//! ellipsoidal tangent horizon plane without any transcendental calls per
//! tested ball.

use glam::{DMat4, DVec3};

use super::geodesy;

/// Bounding sphere in `f64` ECEF world units (`SEMI_MAJOR_M = 1.0`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ball {
    pub centre: DVec3,
    pub radius: f64,
}

impl Ball {
    /// Construct a bounding sphere centered at the AABB midpoint of `points`.
    pub fn over(points: &[DVec3]) -> Self {
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

    /// Expand the bounding sphere radius by `by` world units.
    pub fn widened(self, by: f64) -> Self {
        Self {
            radius: self.radius + by,
            ..self
        }
    }
}

/// Conservative frustum + ellipsoidal-horizon culling frame in `f64`
/// (exact for surface geometry at `height_m <= 0`).
#[derive(Clone, Copy, Debug)]
pub struct Frame {
    inward: [DVec3; 4],
    eye: DVec3,
    towards: DVec3,
    horizon: f64,
}

impl Frame {
    /// Extract the four side frustum planes and the exact WGS84 ellipsoidal
    /// tangent horizon plane from a rotation-only `DMat4` view-projection
    /// matrix and `f64` eye.
    pub fn new(view_proj: &DMat4, eye: DVec3) -> Self {
        let (row0, row1, row3) = (view_proj.row(0), view_proj.row(1), view_proj.row(3));
        let inward = [row3 + row0, row3 - row0, row3 + row1, row3 - row1]
            .map(|plane| plane.truncate().normalize());
        // For WGS84 ellipsoid `p^T A p = 1` with `A = diag(1, 1, 1/b^2)`, a
        // surface point `p` is front-facing iff `(A p) · (eye - p) >= 0`,
        // equivalently `p · (A eye) >= 1`.
        let polar = 1.0 - geodesy::FLATTENING;
        let a_eye = DVec3::new(eye.x, eye.y, eye.z / (polar * polar));
        let len = a_eye.length();
        Self {
            inward,
            eye,
            towards: a_eye / len,
            horizon: 1.0 / len,
        }
    }

    /// Return `false` if `ball` is guaranteed invisible, or `true` if it may be
    /// visible.
    pub fn admits(&self, ball: &Ball) -> bool {
        let from_eye = ball.centre - self.eye;
        if ball.centre.dot(self.towards) + ball.radius < self.horizon {
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
