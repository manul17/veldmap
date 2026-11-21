//! High-precision `f64` bounding-ball frustum and horizon culling.
//!
//! Performs conservative one-sided rejection against the four side planes of
//! the Reverse-Z `DMat4` view-projection matrix and the conservative polar
//! horizon plane without any transcendental calls per tested ball.

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

/// Conservative frustum + polar-horizon culling frame in `f64`.
#[derive(Clone, Copy, Debug)]
pub struct Frame {
    inward: [DVec3; 4],
    eye: DVec3,
    towards: DVec3,
    horizon: f64,
}

impl Frame {
    /// Extract the four side frustum planes and conservative polar horizon
    /// from a rotation-only `DMat4` view-projection matrix and `f64` eye.
    pub fn new(view_proj: &DMat4, eye: DVec3) -> Self {
        let (row0, row1, row3) = (view_proj.row(0), view_proj.row(1), view_proj.row(3));
        let inward = [row3 + row0, row3 - row0, row3 + row1, row3 - row1]
            .map(|plane| plane.truncate().normalize());
        let towards = eye.normalize();
        let polar = 1.0 - geodesy::FLATTENING;
        Self {
            inward,
            eye,
            towards,
            horizon: polar * polar / eye.length(),
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
mod tests {
    use super::*;
    use crate::geo::camera::{self, Camera};
    use crate::geo::geodesy::Geodetic;

    fn at(lat: f64, lon: f64, height_m: f64) -> DVec3 {
        geodesy::world(Geodetic {
            lat_deg: lat,
            lon_deg: lon,
            height_m,
        })
    }

    fn looking(lat: f64, lon: f64, radius_deg: f64) -> (DMat4, DVec3) {
        let mut cam = Camera::default();
        cam.focus(lat, lon, radius_deg);
        cam.advance(10.0);
        (cam.view_projection(1.5), cam.eye_ecef())
    }

    #[test]
    fn a_point_inside_the_frame_clears_every_plane() {
        let (view_proj, eye) = looking(30.0, 20.0, 20.0);
        let frame = Frame::new(&view_proj, eye);
        let (mut inside, mut outside) = (0, 0);
        for lat in (-80..=80).step_by(5) {
            for lon in (-180..180).step_by(5) {
                let point = at(f64::from(lat), f64::from(lon), 0.0);
                let clip = camera::project(&view_proj, point, eye);
                if clip.w <= 0.0 {
                    continue;
                }
                let (x, y) = (clip.x / clip.w, clip.y / clip.w);
                let framed = (-1.0..=1.0).contains(&x) && (-1.0..=1.0).contains(&y);
                let admitted = frame
                    .inward
                    .iter()
                    .all(|inward| inward.dot(point - frame.eye) >= 0.0);
                assert_eq!(framed, admitted, "point {lat},{lon}");
                if framed {
                    inside += 1;
                } else {
                    outside += 1;
                }
            }
        }
        assert!(inside > 20 && outside > 200);
    }

    #[test]
    fn the_far_side_of_the_earth_is_refused_and_the_near_side_is_not() {
        let (view_proj, eye) = looking(0.0, 0.0, 25.0);
        let frame = Frame::new(&view_proj, eye);
        let under = Ball {
            centre: at(0.0, 0.0, 0.0),
            radius: 0.0,
        };
        let behind = Ball {
            centre: at(0.0, 180.0, 0.0),
            radius: 0.0,
        };
        assert!(frame.admits(&under));
        assert!(!frame.admits(&behind));
        assert!(frame.admits(&Ball {
            centre: DVec3::ZERO,
            radius: 1.01
        }));
    }

    #[test]
    fn the_horizon_is_measured_by_the_smallest_radius_the_earth_has() {
        let (view_proj, eye) = looking(0.0, 0.0, 70.0);
        let frame = Frame::new(&view_proj, eye);
        let polar = at(90.0, 0.0, 0.0).z;
        let away = DVec3::Z;
        let (towards, distance) = (eye.normalize(), eye.length());
        let along = polar * polar / distance;
        let across = polar * (1.0 - (polar / distance).powi(2)).sqrt();
        let side = towards.cross(away).cross(towards).normalize();
        let grazing = towards * along + side * across;
        let touching = Ball {
            centre: grazing,
            radius: 0.0,
        };
        assert!(frame.admits(&touching));

        let squashed = polar * polar - at(0.0, 0.0, 0.0).x.powi(2);
        let beyond = towards * (along + squashed * 0.2 / distance) + side * across;
        assert!(!frame.admits(&Ball {
            centre: beyond,
            radius: 0.0
        }));
    }

    #[test]
    fn a_ball_that_swallows_the_eye_is_admitted() {
        let (view_proj, eye) = looking(45.0, 10.0, 0.01);
        let frame = Frame::new(&view_proj, eye);
        let swallowing = Ball {
            centre: DVec3::ZERO,
            radius: 2.0,
        };
        assert!(frame.admits(&swallowing));
    }

    #[test]
    fn a_ball_reaching_into_the_frame_is_admitted_though_its_centre_is_out() {
        let (view_proj, eye) = looking(30.0, 20.0, 20.0);
        let frame = Frame::new(&view_proj, eye);
        let outside = at(60.0, 20.0, 0.0);
        let deficit = frame
            .inward
            .iter()
            .map(|inward| inward.dot(outside - frame.eye))
            .fold(f64::MAX, f64::min);
        assert!(deficit < 0.0);
        assert!(!frame.admits(&Ball {
            centre: outside,
            radius: 0.0
        }));
        assert!(frame.admits(&Ball {
            centre: outside,
            radius: -deficit * 1.01
        }));
        assert!(!frame.admits(&Ball {
            centre: outside,
            radius: -deficit * 0.99
        }));
    }

    #[test]
    fn a_ball_covers_every_point_it_was_built_on() {
        for (lat, lon, span) in [(0.0, 0.0, 60.0), (45.0, 10.0, 2.0), (80.0, 100.0, 0.05)] {
            let points: Vec<DVec3> = [4usize, 0, 1, 2, 3, 5, 6, 7, 8]
                .iter()
                .map(|at_index| {
                    let (row, col) = (at_index / 3, at_index % 3);
                    at(
                        lat + span * f64::from(row as u32) / 2.0,
                        lon + span * f64::from(col as u32) / 2.0,
                        0.0,
                    )
                })
                .collect();
            let ball = Ball::over(&points);
            for &point in &points {
                assert!(ball.centre.distance(point) <= ball.radius + 1e-12);
            }
        }
    }
}
