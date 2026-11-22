use super::*;

fn at(lat_deg: f64, lon_deg: f64, height_m: f64) -> Camera {
    Camera::at(lat_deg, lon_deg, height_m)
}

#[test]
fn the_frame_shape_reaches_the_matrix_unchanged() {
    let camera = at(45.0, 60.0, 10_000.0);
    let narrow = camera.view_projection(1.0 / 200.0);

    let point = geodesy::world(Geodetic::surface(45.0, 60.1));
    let clip = project(&narrow, point, camera.eye_ecef());
    let wide = project(
        &camera.view_projection(16.0 / 9.0),
        point,
        camera.eye_ecef(),
    );

    let ratio = (clip.x / clip.w) / (wide.x / wide.w);
    let expected = (16.0 / 9.0) / (1.0 / 200.0);
    assert!(
        (ratio - expected).abs() / expected < 1e-9,
        "narrow aspect ratio {ratio} vs expected {expected}"
    );
}

#[test]
fn a_probe_names_the_place_that_is_drawn_there() {
    for height_m in [MIN_HEIGHT_M, 100.0, 10_000.0, 155_000.0, 1_000_000.0] {
        let camera = at(45.0, 60.0, height_m);
        let aspect = 16.0 / 9.0;
        let view_proj = camera.view_projection(aspect);
        for (x, y) in [(0.9, 0.1), (0.1, 0.9), (0.85, 0.85), (0.15, 0.2)] {
            let (lat_deg, lon_deg) = camera.probe(x, y, aspect).expect("ray hits Earth");
            let drawn = geodesy::world(Geodetic::surface(lat_deg, lon_deg));
            let clip = project(&view_proj, drawn, camera.eye_ecef());
            let back = ((clip.x / clip.w + 1.0) * 0.5, (1.0 - clip.y / clip.w) * 0.5);
            let residual = (back.0 - x).hypot(back.1 - y);
            assert!(
                residual < 1e-6,
                "at {height_m} m ({x}, {y}) projected back to {back:?} (err {residual:.3e})"
            );
        }
    }
}

#[test]
fn reverse_z_depth_is_monotone_and_in_unit_range_at_2m_and_orbit() {
    for height_m in [MIN_HEIGHT_M, 100.0, 100_000.0, 14_000_000.0] {
        let camera = at(45.0, 60.0, height_m);
        let vp = camera.view_projection(16.0 / 9.0);
        let eye = camera.eye_ecef();

        let nadir = geodesy::world(Geodetic::surface(45.0, 60.0));
        let offset_deg = (camera.visible_deg() * 0.5).min(10.0);
        let off_center = geodesy::world(Geodetic::surface(45.0 + offset_deg, 60.0));

        let clip_near = project(&vp, nadir, eye);
        let clip_far = project(&vp, off_center, eye);
        let z_near = clip_near.z / clip_near.w;
        let z_far = clip_far.z / clip_far.w;

        assert!(
            z_near > 0.0 && z_near < 1.0,
            "height {height_m}: z_near={z_near}"
        );
        assert!(
            z_far > 0.0 && z_far < 1.0,
            "height {height_m}: z_far={z_far}"
        );
        assert!(
            z_near > z_far,
            "Reverse-Z expects nearer surface to have larger depth: {z_near} vs {z_far}"
        );
    }
}

#[test]
fn height_inverts_visible_arc() {
    let brim = 90.0 - FOV_Y_DEG * 0.5;
    for radius_deg in [0.001, 0.5, 2.0, 10.0, 30.0, 55.0, brim] {
        let camera = at(0.0, 0.0, height_for(radius_deg));
        assert!(
            (camera.visible_deg() - radius_deg).abs() < 1e-6,
            "radius {radius_deg} deg read back as {} deg",
            camera.visible_deg()
        );
    }
}

#[test]
fn center_gsd_matches_probe_spacing_and_floor_target() {
    let height_px = 1000.0;
    let floor = at(0.0, 0.0, MIN_HEIGHT_M);
    // At 2 m altitude, 50 deg FOV, 1000 rows -> ~1.865 mm/px (< 2 cm drone imagery).
    let floor_gsd = floor.metres_per_pixel(height_px);
    assert!(
        (floor_gsd - 0.001_865_2).abs() < 1e-6,
        "unexpected floor GSD: {floor_gsd} m/px"
    );

    // Verify GSD agrees with adjacent-pixel ray probe spacing at center.
    for height_m in [MIN_HEIGHT_M, 1_000.0, 100_000.0] {
        let camera = at(0.0, 0.0, height_m);
        let p0 = camera.probe(0.5, 0.5, 1.0).expect("center hit");
        let p1 = camera
            .probe(0.5, 0.5 + 1.0 / height_px, 1.0)
            .expect("adjacent pixel hit");
        let w0 = geodesy::world(Geodetic::surface(p0.0, p0.1));
        let w1 = geodesy::world(Geodetic::surface(p1.0, p1.1));
        let probed_m = (w1 - w0).length();
        let gsd_m = camera.metres_per_pixel(height_px);
        assert!(
            (probed_m - gsd_m).abs() / gsd_m < 0.01,
            "at {height_m} m: probed {probed_m} m vs GSD {gsd_m} m"
        );
    }
}

#[test]
fn non_finite_inputs_do_not_poison_camera() {
    let mut cam = Camera::at(f64::NAN, f64::INFINITY, f64::NAN);
    assert!(cam.eye_ecef().is_finite());
    assert_eq!(cam.height_m(), MIN_HEIGHT_M);
    assert!(cam.view_projection(0.0).is_finite());
    assert!(cam.view_projection(f64::NAN).is_finite());

    let before = cam;
    cam.focus(f64::NAN, 10.0, 1.0);
    cam.fly_to(45.0, f64::INFINITY, 1000.0);
    cam.orbit(f64::NAN, 0.1);
    cam.zoom_at(f64::NAN, 0.0, 0.0);
    assert_eq!(cam, before);

    // Exact pole operations stay finite and orthonormal.
    for pole in [90.0, -90.0] {
        let mut p = Camera::at(pole, 0.0, 10_000.0);
        assert!(p.eye_ecef().is_finite());
        assert_eq!(p.heading_deg(), 0.0);
        assert!(p.probe(0.5, 0.5, 1.5).is_some());
        assert!(!p.reset_north());
        p.orbit(0.1, 0.1);
        p.zoom_at(1.0, 0.1, -0.1);
        assert!(p.eye_ecef().is_finite());
        assert!(p.at.dot(p.up).abs() < 1e-9);
    }
}

#[test]
fn focusing_on_the_whole_earth_fills_the_frame() {
    let brim = 90.0 - FOV_Y_DEG * 0.5;
    let at_brim = height_for(brim);
    for asked in [brim, 80.0, 90.0, 117.0, 1000.0] {
        assert!((height_for(asked) - at_brim).abs() < 1.0);
    }
    let camera = at(0.0, 0.0, at_brim);
    let radius = 1.0 + at_brim / geodesy::SEMI_MAJOR_M;
    let limb = (1.0 / radius).asin().to_degrees();
    assert!((limb - FOV_Y_DEG * 0.5).abs() < 1e-6);
    assert!((camera.visible_deg() - brim).abs() < 1e-6);
}

#[test]
fn orbit_keeps_the_frame_orthonormal() {
    let mut camera = Camera::default();
    for step in 0..500 {
        let sign = if step % 7 < 3 { -1.0 } else { 1.0 };
        camera.orbit(0.37 * sign, 0.29);
        assert!((camera.at.dot(camera.at) - 1.0).abs() < 1e-9);
        assert!((camera.up.dot(camera.up) - 1.0).abs() < 1e-9);
        assert!(camera.at.dot(camera.up).abs() < 1e-9);
        let (side, up, forward) = camera.basis();
        for axis in [side, up, forward] {
            assert!((axis.dot(axis) - 1.0).abs() < 1e-6);
        }
    }
}

#[test]
fn a_drag_moves_the_arc_it_promises() {
    for lat in [0.0, 45.0, 80.0, 89.5] {
        let mut camera = at(lat, 10.0, 1_000_000.0);
        let span = 2.0 * camera.visible_deg();
        let before = camera.at;
        camera.orbit(0.25, 0.0);
        let moved = before.dot(camera.at).clamp(-1.0, 1.0).acos().to_degrees();
        assert!((moved - span * 0.25).abs() < 1e-6);
    }
}

#[test]
fn a_drag_carries_the_camera_over_the_pole() {
    let mut camera = at(80.0, 30.0, 1_000_000.0);
    let span = 2.0 * camera.visible_deg();
    let steps = (20.0 / span).ceil() as usize;
    for _ in 0..steps {
        camera.orbit(0.0, 1.0);
    }

    let travelled = span * steps as f64;
    let (lat, lon) = camera.center();
    assert!(travelled > 10.0);
    assert!((lat - (180.0 - (80.0 + travelled))).abs() < 1e-6);
    assert!((geodesy::unwind(210.0, lon) - 210.0).abs() < 1e-6);
    assert!(camera.up.dot(north_up(camera.at).expect("not at pole")) < 0.0);
    assert!(camera.heading_deg().abs() > 170.0);

    // Reset-North rotates heading smoothly back to 0 degrees without moving center.
    assert!(camera.reset_north());
    assert!(camera.is_flying());
    camera.advance(10.0);
    assert!(!camera.is_flying());
    assert!(!camera.reset_north());
    assert!(camera.heading_deg().abs() < 1e-6);
    let (after_lat, after_lon) = camera.center();
    assert!((after_lat - lat).abs() < 1e-9);
    assert!((geodesy::unwind(lon, after_lon) - lon).abs() < 1e-9);
}

#[test]
fn an_oblique_gesture_undoes_itself() {
    let mut camera = at(60.0, -20.0, 14_000_000.0);
    let (at_before, up_before) = (camera.at, camera.up);
    camera.orbit(0.37, 0.29);
    camera.orbit(-0.37, -0.29);
    assert!(at_before.dot(camera.at) > 1.0 - 1e-12);
    assert!(up_before.dot(camera.up) > 1.0 - 1e-12);
}

#[test]
fn a_flight_lands_exactly_where_asked() {
    let mut camera = Camera::default();
    camera.focus(-33.9, 151.2, 0.5);

    let mut ticks = 0;
    while camera.advance(1.0 / 60.0) {
        ticks += 1;
        assert!(ticks < 1000);
    }
    assert!(ticks > 1);

    let (lat, lon) = camera.center();
    assert!((lat + 33.9).abs() < 1e-9);
    assert!((lon - 151.2).abs() < 1e-9);
    assert_eq!(camera.height_m(), height_for(0.5 * FRAME_MARGIN));
    assert!(camera.at.dot(camera.up).abs() < 1e-12);
    assert!(!camera.advance(1.0 / 60.0));
}

#[test]
fn a_flight_passes_between_its_ends() {
    let mut camera = at(0.0, 0.0, 10_000_000.0);
    let target = height_for(1.0 * FRAME_MARGIN);
    camera.focus(0.0, 40.0, 1.0);
    camera.advance(FLIGHT_S * 0.5);

    let (lat, lon) = camera.center();
    assert!(lat.abs() < 1e-9);
    assert!((lon - 20.0).abs() < 1e-4);
    let middle = (10_000_000.0_f64 * target).sqrt();
    assert!((camera.height_m() / middle - 1.0).abs() < 1e-5);
}

#[test]
fn a_repeated_focus_does_not_restart_the_flight() {
    let mut camera = Camera::default();
    camera.focus(-33.9, 151.2, 0.5);
    for _ in 0..30 {
        camera.advance(1.0 / 60.0);
        camera.focus(-33.9, 151.2, 0.5);
    }
    assert!(camera.advance(0.2));
    assert!(!camera.advance(0.01));

    let (lat, lon) = camera.center();
    assert!((lat + 33.9).abs() < 1e-9 && (lon - 151.2).abs() < 1e-9);
}

#[test]
fn a_gesture_calls_off_the_flight() {
    let mut camera = Camera::default();
    camera.focus(-33.9, 151.2, 0.5);
    camera.advance(0.1);
    camera.orbit(0.1, 0.0);
    assert!(!camera.advance(1.0));

    camera.focus(-33.9, 151.2, 0.5);
    camera.zoom_at(1.0, 0.0, 0.0);
    assert!(!camera.advance(1.0));
}

#[test]
fn zoom_respects_height_range() {
    let mut camera = Camera::default();
    camera.zoom_at(1e4, 0.0, 0.0);
    assert_eq!(camera.height_m(), MIN_HEIGHT_M);
    camera.zoom_at(-1e4, 0.0, 0.0);
    assert_eq!(camera.height_m(), MAX_HEIGHT_M);
}

fn apart(from: DVec3, to: DVec3) -> f64 {
    from.cross(to).length().atan2(from.dot(to))
}

fn pixels_per_tangent(rows: f64) -> f64 {
    rows * 0.5 / (FOV_Y_DEG * 0.5).to_radians().tan()
}

fn held_at(camera: &Camera, hold_x: f64, hold_y: f64) -> (DVec3, (f64, f64)) {
    let (right, above) = frame_offset(hold_x, -hold_y);
    let point =
        geodesy::intersect_at(camera.eye_ecef(), camera.look(right, above), 0.0).expect("hit");
    (point, (right, above))
}

#[test]
fn the_hold_lets_go_at_the_limb() {
    let scale = pixels_per_tangent(1300.0);
    for (dx, dy) in [(1.0, 0.0), (0.0, 1.0), (0.6, -0.8)] {
        let swing = |reach: f64, steps: f64| {
            let before = Camera::default();
            let mut camera = before;
            let (held, (right, above)) = held_at(&camera, reach * dx, reach * dy);
            camera.zoom_at(steps, reach * dx, reach * dy);
            let (now_right, now_above) = camera.seen_at(held).expect("in front");
            (
                apart(before.at, camera.at).to_degrees(),
                (now_right - right).hypot(now_above - above) * scale,
            )
        };

        let turns: Vec<f64> = [0.1, 0.2, 0.3, 0.33, 0.345, 0.348, 0.35]
            .iter()
            .map(|r| swing(*r, 1.0).0)
            .collect();
        let peak = turns
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(at, _)| at)
            .expect("turns");
        assert!(peak > 0 && peak + 1 < turns.len());
        assert!(turns[turns.len() - 1] < turns[peak] * 0.5);

        let missed = swing(0.30, 1.0).1;
        assert!(missed < 0.02);

        let outward: Vec<f64> = [0.20, 0.25, 0.27, 0.28, 0.29, 0.295, 0.2974]
            .iter()
            .map(|r| swing(*r, -1.0).0)
            .collect();
        let (rise, edge) = (
            outward.iter().copied().fold(0.0, f64::max),
            outward[outward.len() - 2],
        );
        assert!(edge < rise * 0.25);
    }
}

#[test]
fn a_notch_back_undoes_a_notch_in() {
    for (before, hold_x, first) in [
        (at(45.0, 60.0, 1_000_000.0), 0.2, 1.0),
        (Camera::default(), 0.32, -1.0),
    ] {
        let mut camera = before;
        camera.zoom_at(first, hold_x, -0.1);
        camera.zoom_at(-first, hold_x, -0.1);

        let drift =
            apart(before.at, camera.at) * geodesy::SEMI_MAJOR_M / before.metres_per_pixel(1300.0);
        assert!(drift < 0.25, "drift {drift:.3} px");
        assert!((camera.height_m() / before.height_m() - 1.0).abs() < 1e-9);
    }
}

#[test]
fn a_zoom_holds_whatever_it_took_hold_of() {
    let scale = pixels_per_tangent(1300.0);
    let mut checked = 0;
    for height_m in [
        MIN_HEIGHT_M,
        10.0,
        200.0,
        2_000.0,
        20_000.0,
        100_000.0,
        500_000.0,
        1_500_000.0,
        3_000_000.0,
        4_500_000.0,
        6_000_000.0,
        9_000_000.0,
        14_000_000.0,
        30_000_000.0,
        80_000_000.0,
    ] {
        for lat_deg in [0.0, 45.0, 89.0] {
            for hold_x in [-0.8, -0.4, -0.1, -0.03, 0.0, 0.03, 0.1, 0.4, 0.8] {
                for hold_y in [-0.5, -0.2, -0.04, 0.0, 0.3, 0.5] {
                    for steps in [1.0, -1.0] {
                        let start = at(lat_deg, 60.0, height_m);
                        let (right, above) = frame_offset(hold_x, -hold_y);
                        let alpha_seen = right.hypot(above).atan();
                        let far = Camera {
                            height_m: (height_m * ZOOM_PER_STEP.powf(steps))
                                .clamp(HEIGHT_RANGE_M.0, HEIGHT_RANGE_M.1),
                            ..start
                        };
                        let share = start.grip(alpha_seen).min(far.grip(alpha_seen));
                        let (right, above) = (right * share, above * share);

                        let held =
                            geodesy::intersect_at(start.eye_ecef(), start.look(right, above), 0.0)
                                .expect("grip kept ray on Earth");
                        let mut camera = start;
                        camera.zoom_at(steps, hold_x, hold_y);

                        let (now_right, now_above) = camera.seen_at(held).expect("in front");
                        let slipped = (now_right - right).hypot(now_above - above) * scale;
                        checked += 1;
                        assert!(
                            slipped < 0.005,
                            "at {height_m} m lat {lat_deg} hold ({hold_x}, {hold_y}) steps {steps}: slipped {slipped:.4} px"
                        );
                    }
                }
            }
        }
    }
    assert!(checked > 1000);
}

#[test]
fn a_notch_arrives_whole_or_in_a_hundred_parts() {
    for (start, hold_x) in [(at(45.0, 60.0, 3_000_000.0), 0.2), (Camera::default(), 0.2)] {
        let (mut once, mut parts) = (start, start);
        once.zoom_at(1.0, hold_x, -0.15);
        for _ in 0..100 {
            parts.zoom_at(0.01, hold_x, -0.15);
        }

        let apart_px =
            apart(once.at, parts.at) * geodesy::SEMI_MAJOR_M / start.metres_per_pixel(1300.0);
        assert!(apart_px < 0.05);
        assert!((parts.height_m() / once.height_m() - 1.0).abs() < 1e-7);
    }

    let start = Camera::default();
    let cut = |parts: u32| {
        let mut camera = start;
        for _ in 0..parts {
            camera.zoom_at(1.0 / f64::from(parts), 0.34, -0.15);
        }
        camera
    };
    let (slow, fast) = (cut(112), cut(266));
    let apart_px = apart(slow.at, fast.at) * geodesy::SEMI_MAJOR_M / start.metres_per_pixel(1300.0);
    assert!(apart_px < 1.0);
}

#[test]
fn a_notch_in_parts_is_undone_by_a_notch_in_parts() {
    let start = Camera::default();
    let mut camera = start;
    for steps in [-0.01, 0.01] {
        for _ in 0..100 {
            camera.zoom_at(steps, 0.32, -0.1);
        }
    }

    let drift = apart(start.at, camera.at) * geodesy::SEMI_MAJOR_M / start.metres_per_pixel(1300.0);
    assert!(drift < 0.5);
    assert!((camera.height_m() / start.height_m() - 1.0).abs() < 1e-7);
}

#[test]
fn a_pointer_holds_the_place_the_probe_names() {
    let (width, height) = (2048.0, 1334.0);
    let aspect = width / height;
    for (px, py) in [(1400.0, 500.0), (600.0, 1000.0), (1024.0, 667.0)] {
        let mut camera = at(45.0, 60.0, 3_000_000.0);
        let before = camera.probe(px / width, py / height, aspect).expect("hit");
        camera.zoom_at(
            1.0,
            (px - width / 2.0) / height,
            (py - height / 2.0) / height,
        );
        let after = camera.probe(px / width, py / height, aspect).expect("hit");

        let slipped = geodesy::separation(before, after).to_radians() * geodesy::SEMI_MAJOR_M;
        assert!(slipped < 20.0, "slipped {slipped:.1} m");
    }
}

#[test]
fn zoom_at_the_edges_only_stops_moving() {
    for (height_m, steps) in [(MIN_HEIGHT_M, 3.0), (MAX_HEIGHT_M, -3.0)] {
        let before = at(45.0, 60.0, height_m);
        let mut camera = before;
        camera.zoom_at(steps, 0.3, 0.2);
        assert_eq!(camera.at, before.at);
        assert_eq!(camera.up, before.up);
        assert_eq!(camera.height_m(), height_m);
    }
}

#[test]
fn zoom_past_the_limb_holds_the_middle() {
    let scale = pixels_per_tangent(1300.0);
    let before = Camera::default();
    let mut camera = before;
    let (middle, (right, above)) = held_at(&camera, 0.0, 0.0);
    camera.zoom_at(1.0, 0.45, 0.0);

    assert!(camera.height_m() < before.height_m());
    let (now_right, now_above) = camera.seen_at(middle).expect("in front");
    let missed = (now_right - right).hypot(now_above - above) * scale;
    assert!(missed < 0.01);
}

#[test]
fn alternating_parts_bring_the_camera_back() {
    let start = at(45.0, 60.0, 1_000_000.0);
    let mut camera = start;
    for step in 0..500 {
        let sign = if step % 2 == 0 { 1.0 } else { -1.0 };
        camera.zoom_at(0.02 * sign, 0.2, -0.15);
    }

    let drift = apart(start.at, camera.at) * geodesy::SEMI_MAJOR_M / start.metres_per_pixel(1300.0);
    assert!(drift < 0.01);
    assert!((camera.height_m() / start.height_m() - 1.0).abs() < 1e-9);
}

#[test]
fn focus_reaches_the_pole() {
    let mut camera = Camera::default();
    camera.orbit(0.4, 0.4);
    camera.focus(89.7, 15.0, 0.2);
    camera.advance(10.0);
    let (lat, lon) = camera.center();
    assert!((lat - 89.7).abs() < 1e-9);
    assert!((lon - 15.0).abs() < 1e-9);
    assert!(camera.at.dot(camera.up).abs() < 1e-9);
}

#[test]
fn fly_to_current_pose_is_a_noop() {
    let mut camera = at(45.0, 60.0, 10_000.0);
    camera.fly_to(45.0, 60.0, 10_000.0);
    assert!(!camera.is_flying());
    assert!(!camera.advance(1.0 / 60.0));
}

#[test]
fn ray_and_probe_guard_invalid_aspect_ratios() {
    let camera = at(0.0, 0.0, 10_000.0);
    let expected = camera.ray(0.75, 0.25, 1.0);
    for bad_aspect in [0.0, -1.5, f64::NAN, f64::INFINITY] {
        assert_eq!(camera.ray(0.75, 0.25, bad_aspect), expected);
        assert!(camera.probe(0.75, 0.25, bad_aspect).is_some());
    }
}

#[test]
fn geodetic_nadir_aim_matches_center_with_zero_mid_latitude_offset() {
    for lat_deg in [-85.0, -45.0, 0.0, 45.0, 85.0] {
        for height_m in [MIN_HEIGHT_M, 1_000.0, 100_000.0, 14_000_000.0, MAX_HEIGHT_M] {
            let cam = at(lat_deg, 39.7015, height_m);
            let center = cam.center();
            let probed = cam.probe(0.5, 0.5, 16.0 / 9.0).expect("nadir hits WGS84");
            assert!(
                (probed.0 - center.0).abs() < 1e-11,
                "lat drift at ({lat_deg}, {height_m} m): probed {probed:?} vs center {center:?}"
            );
            assert!(
                (geodesy::unwind(center.1, probed.1) - center.1).abs() < 1e-11,
                "lon drift at ({lat_deg}, {height_m} m): probed {probed:?} vs center {center:?}"
            );
        }
    }

    // Zooming straight in from orbit to 2 m at viewport center at 45 deg lat
    // must keep center pinned with zero drift (eliminating the legacy 14.7 km drift).
    let mut cam = at(45.0, 60.0, 14_000_000.0);
    let before = cam.center();
    for _ in 0..80 {
        cam.zoom_at(1.0, 0.0, 0.0);
    }
    assert_eq!(cam.height_m(), MIN_HEIGHT_M);
    let after = cam.center();
    let drift_m = geodesy::separation(before, after).to_radians() * geodesy::SEMI_MAJOR_M;
    assert!(
        drift_m < 1e-3,
        "orbit-to-2m center zoom drifted by {drift_m} m"
    );
}
