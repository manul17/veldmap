//! Cross-module integration tests exercising the public `veldmap-core::geo` API.

use glam::DVec3;
use veldmap_core::geo::{Ball, Camera, Geodetic, MIN_HEIGHT_M, SEMI_MAJOR_M, WGS84};

fn surface_world(lat: f64, lon: f64) -> DVec3 {
    WGS84.to_ecef(Geodetic::surface(lat, lon))
}

#[test]
fn camera_frustum_refuses_far_side_and_admits_near_side() {
    let mut cam = Camera::default();
    cam.focus(0.0, 0.0, 25.0);
    cam.advance(10.0);
    let frame = cam.frustum(1.5);

    let under = Ball {
        centre: surface_world(0.0, 0.0),
        radius: 0.0,
    };
    let behind = Ball {
        centre: surface_world(0.0, 180.0),
        radius: 0.0,
    };
    assert!(frame.admits(&under));
    assert!(!frame.admits(&behind));
    assert!(frame.admits(&Ball {
        centre: DVec3::ZERO,
        radius: SEMI_MAJOR_M * 1.01,
    }));
}

#[test]
fn camera_frustum_admits_ball_swallowing_the_eye() {
    let mut cam = Camera::default();
    cam.focus(45.0, 10.0, 0.01);
    cam.advance(10.0);
    let frame = cam.frustum(1.5);
    assert!(frame.admits(&Ball {
        centre: DVec3::ZERO,
        radius: SEMI_MAJOR_M * 2.0,
    }));
}

#[test]
fn camera_frustum_culling_is_accurate_at_2m_altitude() {
    let cam = Camera::at(45.0, 39.0, MIN_HEIGHT_M);
    let frame = cam.frustum(1.5);
    let directly_below = Ball {
        centre: surface_world(45.0, 39.0),
        radius: 0.5,
    };
    let far_away = Ball {
        centre: surface_world(45.01, 39.0),
        radius: 0.5,
    };
    assert!(frame.admits(&directly_below));
    assert!(!frame.admits(&far_away));
}

#[test]
fn oblate_horizon_plane_separates_near_limb_points_at_mid_latitude() {
    // At orbital altitudes the full WGS84 limb sits inside the 50 deg FOV,
    // so `CullFrustum::admits` exercises the oblate horizon boundary directly.
    for height_m in [14_000_000.0, 30_000_000.0, 80_000_000.0] {
        let cam = Camera::at(45.0, 39.7, height_m);
        let frustum = cam.frustum(16.0 / 9.0);
        let point_ball = |lat: f64, lon: f64| Ball {
            centre: surface_world(lat, lon),
            radius: 0.0,
        };
        assert!(frustum.admits(&point_ball(45.0, 39.7)));

        let mut lo = 0.0_f64;
        let mut hi = 180.0_f64;
        for _ in 0..48 {
            let mid = 0.5 * (lo + hi);
            if frustum.admits(&point_ball(45.0, 39.7 + mid)) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        assert!(lo > 0.0 && hi < 180.0);
        assert!(frustum.admits(&point_ball(45.0, 39.7 + (lo - 0.05).max(0.0))));
        assert!(!frustum.admits(&point_ball(45.0, 39.7 + hi + 0.05)));
    }
}
