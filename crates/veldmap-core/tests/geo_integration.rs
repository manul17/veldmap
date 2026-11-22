//! Cross-module integration tests exercising the public `veldmap-core::geo` API.

use glam::DVec3;
use veldmap_core::geo::camera::{Camera, MIN_HEIGHT_M};
use veldmap_core::geo::cull::Ball;
use veldmap_core::geo::footprint::{self, GeoPoint, Ring};
use veldmap_core::geo::geodesy::{self, Geodetic, SEMI_MAJOR_M};

fn surface_world(lat: f64, lon: f64) -> DVec3 {
    geodesy::world(Geodetic::surface(lat, lon))
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
fn footprint_covers_geodesic_ring_cleanly() {
    let coords = [(72.0, 10.0), (72.0, 30.0), (73.0, 30.0), (73.0, 10.0)];
    let points: Vec<GeoPoint> = coords
        .iter()
        .map(|&(lat, lon)| GeoPoint::new(lat, lon))
        .collect();
    let rings = [Ring { points }];

    assert!(footprint::covers(&rings, 72.5, 20.0));
    assert!(!footprint::covers(&rings, 73.1, 20.0));
}
