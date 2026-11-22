use glam::{DMat3, DVec4};

use super::*;
use crate::geo::geodesy::Geodetic;

fn at(lat: f64, lon: f64, height_m: f64) -> DVec3 {
    geodesy::world(Geodetic {
        lat_deg: lat,
        lon_deg: lon,
        height_m,
    })
}

/// Self-contained right-handed infinite Reverse-Z view-projection matrix so
/// `cull` unit tests stay strictly decoupled from `camera`.
fn view_proj_from(lat: f64, lon: f64, height_m: f64, aspect: f64) -> (DMat4, DVec3) {
    let eye = at(lat, lon, height_m);
    let forward = -eye.normalize();
    let side = forward.cross(DVec3::Z).normalize();
    let up = side.cross(forward);
    let rot = DMat3::from_cols(side, up, -forward).transpose();
    let view = DMat4::from_cols(
        rot.x_axis.extend(0.0),
        rot.y_axis.extend(0.0),
        rot.z_axis.extend(0.0),
        DVec3::ZERO.extend(1.0),
    );
    let f = 1.0 / (25.0_f64.to_radians()).tan();
    let near = height_m * 0.5;
    let proj = DMat4::from_cols(
        DVec4::new(f / aspect, 0.0, 0.0, 0.0),
        DVec4::new(0.0, f, 0.0, 0.0),
        DVec4::new(0.0, 0.0, 0.0, -1.0),
        DVec4::new(0.0, 0.0, near, 0.0),
    );
    (proj * view, eye)
}

#[test]
fn a_point_inside_the_frame_clears_every_plane() {
    let (view_proj, eye) = view_proj_from(30.0, 20.0, 5_000_000.0, 1.5);
    let frame = Frame::new(&view_proj, eye);
    let (mut inside, mut outside) = (0, 0);
    for lat in (-80..=80).step_by(5) {
        for lon in (-180..180).step_by(5) {
            let point = at(f64::from(lat), f64::from(lon), 0.0);
            let clip = view_proj * (point - eye).extend(1.0);
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
fn the_horizon_matches_the_oblate_wgs84_ellipsoid_tangent_plane() {
    for (lat_cam, height_m) in [
        (0.0, 14_000_000.0),
        (45.0, 14_000_000.0),
        (45.0, 80_000_000.0),
    ] {
        let (vp, eye) = view_proj_from(lat_cam, 0.0, height_m, 1.5);
        let frame = Frame::new(&vp, eye);

        for lat in (-89..=89).step_by(2) {
            for lon in [0.0, 90.0, 180.0, -90.0] {
                let p = at(f64::from(lat), lon, 0.0);
                let n = geodesy::unit(f64::from(lat), lon);
                let facing = n.dot(eye - p);
                let clears_horizon = p.dot(frame.towards) >= frame.horizon;
                if facing > 1e-3 {
                    assert!(
                        clears_horizon,
                        "visible surface point ({lat}, {lon}) at cam ({lat_cam}, {height_m} m) was falsely culled"
                    );
                } else if facing < -1e-3 {
                    assert!(
                        !clears_horizon,
                        "back-side surface point ({lat}, {lon}) at cam ({lat_cam}, {height_m} m) was not culled"
                    );
                }
            }
        }
    }
}

#[test]
fn a_ball_reaching_into_the_frame_is_admitted_though_its_centre_is_out() {
    let (view_proj, eye) = view_proj_from(30.0, 20.0, 5_000_000.0, 1.5);
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
    let empty = Ball::over(&[]);
    assert_eq!(empty.centre, DVec3::ZERO);
    assert_eq!(empty.radius, 0.0);
}

#[test]
fn reverse_z_near_plane_culls_balls_behind_near_and_admits_straddling_balls() {
    let height_m = 100.0;
    let (view_proj, eye) = view_proj_from(45.0, 20.0, height_m, 1.5);
    let frame = Frame::new(&view_proj, eye);
    let forward = -eye.normalize();
    let near = height_m * 0.5;

    // Ball strictly between eye and near plane (behind near plane).
    let behind_near = Ball {
        centre: eye + forward * (near * 0.4),
        radius: near * 0.2,
    };
    assert!(
        !frame.admits(&behind_near),
        "ball strictly behind near plane should be culled"
    );

    // Ball behind the camera eye.
    let behind_eye = Ball {
        centre: eye - forward * 50.0,
        radius: 10.0,
    };
    assert!(
        !frame.admits(&behind_eye),
        "ball behind camera eye should be culled"
    );

    // Ball straddling the near plane (extends past near along forward).
    let straddling = Ball {
        centre: eye + forward * (near * 0.8),
        radius: near * 0.3,
    };
    assert!(
        frame.admits(&straddling),
        "ball straddling near plane must be admitted"
    );
}

#[test]
fn degenerate_zero_eye_produces_no_nan_in_horizon_plane() {
    let (view_proj, _) = view_proj_from(0.0, 0.0, 1000.0, 1.5);
    let frame = Frame::new(&view_proj, DVec3::ZERO);
    assert!(frame.towards.is_finite());
    assert!(!frame.horizon_margin(at(0.0, 0.0, 0.0)).is_nan());
}
