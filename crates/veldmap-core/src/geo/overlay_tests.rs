use super::*;
use crate::geo::camera::Camera;
use crate::geo::projection::System;
use crate::tile::{cell_image_rect, level_count};

/// Measure the worst horizontal surface placement drift (in metres) across a
/// dense sample grid inside `cell_rect` between the CPU `f64` RTC warp patch
/// (including `f32` vertex quantization and triangle interpolation) and the
/// exact georeference binding.
fn max_surface_drift_m(frame: &Frame, width: u32, height: u32, cell_rect: [f64; 4]) -> f64 {
    let patch = build_warp_patch(frame, width, height, cell_rect);
    let w = f64::from(width);
    let h = f64::from(height);
    let samples = 25usize;
    let mut worst_m = 0.0_f64;

    for sy in 0..=samples {
        let ty = sy as f64 / samples as f64;
        for sx in 0..=samples {
            let tx = sx as f64 / samples as f64;
            let px = cell_rect[0] + (cell_rect[2] - cell_rect[0]) * tx;
            let py = cell_rect[1] + (cell_rect[3] - cell_rect[1]) * ty;
            let (exact_lat, exact_lon) = frame.geodetic(px / w, py / h);
            let exact_ecef = WGS84.to_ecef(Geodetic::surface(exact_lat, exact_lon));

            let drawn_ecef = evaluate_warp_patch_ecef(&patch, tx, ty);
            // Project drawn point onto WGS84 ellipsoid surface to measure
            // surface placement drift (separating radial chord sag).
            let drawn_geo = WGS84.from_ecef(drawn_ecef);
            let drawn_surface =
                WGS84.to_ecef(Geodetic::surface(drawn_geo.lat_deg, drawn_geo.lon_deg));
            let drift_m = exact_ecef.distance(drawn_surface);
            worst_m = worst_m.max(drift_m);
        }
    }
    worst_m
}

#[test]
fn a_raster_lands_where_its_binding_puts_it() {
    let utm37 = Projection::from_epsg(32637).expect("UTM 37N");
    let gk7 = Projection::from_epsg(28407).expect("Gauss-Kruger zone 7");
    let utm37s = Projection::from_epsg(32737).expect("UTM 37S");

    struct Case {
        name: &'static str,
        frame: Frame,
        width: u32,
        height: u32,
        level: u32,
        cell_x: u32,
        cell_y: u32,
    }

    // 30-degree rotated 8000x5000 orthophoto at 2 cm/px in UTM 37N.
    let angle = 30.0_f64.to_radians();
    let gsd = 0.02;
    let (sin_a, cos_a) = angle.sin_cos();
    let rotated_affine = [
        gsd * cos_a,
        gsd * sin_a,
        550_000.0,
        gsd * sin_a,
        -gsd * cos_a,
        5_230_000.0,
    ];
    let rotated_frame =
        Frame::from_pixel_affine(utm37, rotated_affine, 8_000, 5_000).expect("rotated frame");

    let cases = [
        // 1. Drone orthophoto at 2 cm/px (UTM 37N)
        Case {
            name: "utm37_drone_2cm",
            frame: Frame::utm(utm37, 550_000.0, 5_230_000.0, 550_200.0, 5_230_200.0),
            width: 10_000,
            height: 10_000,
            level: 0,
            cell_x: 3,
            cell_y: 5,
        },
        // 2. Sentinel-2 granule at 10 m/px (UTM 37N near zone edge)
        Case {
            name: "utm37_sentinel2_10m",
            frame: Frame::utm(utm37, 600_000.0, 5_200_000.0, 709_800.0, 5_309_800.0),
            width: 10_980,
            height: 10_980,
            level: 0,
            cell_x: 20,
            cell_y: 1,
        },
        // 3. Landsat scene at 30 m/px (UTM 37N)
        Case {
            name: "utm37_landsat_30m",
            frame: Frame::utm(utm37, 450_000.0, 5_100_000.0, 681_000.0, 5_331_000.0),
            width: 7_700,
            height: 7_700,
            level: 0,
            cell_x: 14,
            cell_y: 2,
        },
        // 4. Sentinel-2 granule at pyramid level 4 (coarse UTM zone edge)
        Case {
            name: "utm37_sentinel2_level4_edge",
            frame: Frame::utm(utm37, 600_000.0, 5_200_000.0, 709_800.0, 5_309_800.0),
            width: 10_980,
            height: 10_980,
            level: 4,
            cell_x: 1,
            cell_y: 0,
        },
        // 5. 8000x5000 orthophoto rotated by 30 degrees (UTM 37N)
        Case {
            name: "utm37_rotated_30deg_8000x5000",
            frame: rotated_frame,
            width: 8_000,
            height: 5_000,
            level: 0,
            cell_x: 7,
            cell_y: 4,
        },
        // 6. Second CRS fixture: Gauss-Kruger Pulkovo 1942 zone 7 (EPSG:28407)
        Case {
            name: "gk_pulkovo1942_zone7_10m",
            frame: Frame::utm(gk7, 7_450_000.0, 5_200_000.0, 7_559_800.0, 5_309_800.0),
            width: 10_980,
            height: 10_980,
            level: 4,
            cell_x: 0,
            cell_y: 0,
        },
        // 7. Southern Hemisphere UTM 37S (EPSG:32737)
        Case {
            name: "utm37s_southern_hemisphere_native",
            frame: Frame::utm(utm37s, 500_000.0, 6_000_000.0, 500_200.0, 6_000_200.0),
            width: 10_000,
            height: 10_000,
            level: 0,
            cell_x: 4,
            cell_y: 4,
        },
    ];

    for case in cases {
        let cell = cell_image_rect(
            case.cell_x,
            case.cell_y,
            case.level,
            case.width,
            case.height,
        );
        let drift_m = max_surface_drift_m(&case.frame, case.width, case.height, cell);
        let base_px_m = case
            .frame
            .ground_m_per_px(case.width)
            .expect("measurable binding");
        let level_px_m = base_px_m * f64::from(1u32 << case.level);
        let drift_px = drift_m / level_px_m;

        assert!(
            drift_px < 1e-3,
            "{}: drift {drift_px:.6e} px ({drift_m:.6} m) exceeded 1e-3 px",
            case.name
        );
        assert!(
            drift_m < 0.02,
            "{}: drift {drift_m:.6} m exceeded 2 cm pyramid tolerance",
            case.name
        );
        if case.level == 0 {
            assert!(
                drift_m < 0.002,
                "{}: native drift {drift_m:.6} m exceeded 2 mm native tolerance",
                case.name
            );
        }
    }
}

#[test]
fn negative_determinant_and_south_up_affines_wind_outward() {
    let utm37 = Projection::from_epsg(32637).expect("UTM 37N");
    let cell = [0.0, 0.0, 512.0, 512.0];

    // Standard north-up: x_per_i > 0, y_per_j < 0 -> normal CCW winding.
    let north_up = Frame::from_pixel_affine(
        utm37,
        [10.0, 0.0, 500_000.0, 0.0, -10.0, 5_250_000.0],
        512,
        512,
    )
    .expect("north-up");
    let patch_normal = build_warp_patch(&north_up, 512, 512, cell);
    assert!(!patch_normal.flip_winding);

    // South-up: x_per_i > 0, y_per_j > 0 -> flipped winding to face outward.
    let south_up = Frame::from_pixel_affine(
        utm37,
        [10.0, 0.0, 500_000.0, 0.0, 10.0, 5_240_000.0],
        512,
        512,
    )
    .expect("south-up");
    let patch_south = build_warp_patch(&south_up, 512, 512, cell);
    assert!(patch_south.flip_winding);

    // Horizontally mirrored: x_per_i < 0, y_per_j < 0 -> flipped winding.
    let mirrored = Frame::from_pixel_affine(
        utm37,
        [-10.0, 0.0, 505_120.0, 0.0, -10.0, 5_250_000.0],
        512,
        512,
    )
    .expect("mirrored");
    let patch_mirrored = build_warp_patch(&mirrored, 512, 512, cell);
    assert!(patch_mirrored.flip_winding);

    // Verify the final emitted triangle winding always has outward normals.
    for patch in [&patch_normal, &patch_south, &patch_mirrored] {
        let side = (patch.segments + 1) as usize;
        let v = |r: usize, c: usize| {
            let p = patch.rel_positions[r * side + c];
            patch.origin_ecef_m + DVec3::new(f64::from(p[0]), f64::from(p[1]), f64::from(p[2]))
        };
        let p0 = v(0, 0);
        let p1 = v(1, 0);
        let p2 = v(0, 1);
        let tri_n = if patch.flip_winding {
            (p2 - p0).cross(p1 - p0)
        } else {
            (p1 - p0).cross(p2 - p0)
        };
        assert!(
            tri_n.dot(p0) > 0.0,
            "triangle winding must face outward from WGS84 ellipsoid"
        );
    }
}

#[test]
fn horizon_plane_uniform_matches_exact_ellipsoidal_horizon() {
    for height_m in [2.0, 500.0, 2_000_000.0] {
        let cam = Camera::at(47.2, 39.7, height_m);
        let eye = cam.eye_ecef();
        let frustum = cam.frustum(16.0 / 9.0);
        let (towards, _horizon) = frustum.horizon_plane();
        let margin = frustum.horizon_eye_margin();

        // Point directly beneath camera is strictly front-facing.
        let nadir = WGS84.to_ecef(Geodetic::surface(47.2, 39.7));
        let world_nadir = (nadir - eye).as_vec3().as_dvec3();
        assert!(world_nadir.dot(towards) > margin);

        // Antipodal point is strictly behind the horizon plane.
        let antipode = WGS84.to_ecef(Geodetic::surface(-47.2, 39.7 - 180.0));
        let world_anti = (antipode - eye).as_vec3().as_dvec3();
        assert!(world_anti.dot(towards) < margin);

        // Point 5 degrees of great-circle arc past the horizon (at the equator
        // so longitude delta equals true angular separation) must be rejected.
        let cam_eq = Camera::at(0.0, 39.7, height_m);
        let eye_eq = cam_eq.eye_ecef();
        let frustum_eq = cam_eq.frustum(16.0 / 9.0);
        let (towards_eq, _) = frustum_eq.horizon_plane();
        let margin_eq = frustum_eq.horizon_eye_margin();
        let horizon_arc_deg = (geodesy::SEMI_MAJOR_M / (geodesy::SEMI_MAJOR_M + height_m))
            .acos()
            .to_degrees();
        let past_limb = WGS84.to_ecef(Geodetic::surface(
            0.0,
            39.7 + (horizon_arc_deg + 5.0).min(179.0),
        ));
        let world_past = (past_limb - eye_eq).as_vec3().as_dvec3();
        assert!(
            world_past.dot(towards_eq) < margin_eq,
            "height={height_m}: point past horizon must fail horizon plane test"
        );
    }
}

#[test]
fn lazy_cull_tree_prunes_offscreen_subtrees_and_preserves_limb_visibility() {
    let utm37 = Projection::from_epsg(32637).expect("UTM 37N");
    let frame = Frame::utm(utm37, 500_000.0, 5_200_000.0, 600_000.0, 5_300_000.0);
    let width = 16_384;
    let height = 16_384;
    let max_level = (level_count(width, height) - 1) as u8;
    let mut tree = OverlayCullTree::new(width, height, max_level);

    // Zoom into the center of the raster: only a handful of level-0 tiles are
    // visible out of 32 x 32 = 1024, so lazy descent should materialize far
    // fewer balls than eager level-0 construction.
    let (center_lat, center_lon) = frame.geodetic(0.5, 0.5);
    let cam = Camera::at(center_lat, center_lon, 5_000.0);
    let aspect = 16.0 / 9.0;
    let vp = cam.view_projection(aspect);
    let eye = cam.eye_ecef();
    let frustum = cam.frustum(aspect);

    let visible = tree.visible_cells(&frame, 0, &frustum, &vp, eye);
    assert!(!visible.is_empty());
    assert!(
        tree.cached_ball_count() < 200,
        "lazy culling should prune offscreen subtrees, materialized {} balls",
        tree.cached_ball_count()
    );

    // Every cell on the opposite side of the Earth must be rejected.
    let cam_far = Camera::at(-center_lat, center_lon - 180.0, 5_000_000.0);
    let vp_far = cam_far.view_projection(aspect);
    let eye_far = cam_far.eye_ecef();
    let frustum_far = cam_far.frustum(aspect);
    assert!(
        tree.visible_cells(&frame, 0, &frustum_far, &vp_far, eye_far)
            .is_empty()
    );
}

#[test]
fn frame_fraction_inverse_round_trips_for_projected_and_tie_grid() {
    let utm37 = Projection::Transverse(System::utm(37, true));
    let frame = Frame::utm(utm37, 500_000.0, 5_200_000.0, 510_000.0, 5_210_000.0);
    let (lat, lon) = frame.geodetic(0.37, 0.64);
    let (fx, fy) = frame.fraction_at(lat, lon).expect("inside projected frame");
    assert!((fx - 0.37).abs() < 1e-7);
    assert!((fy - 0.64).abs() < 1e-7);
    assert!(frame.fraction_at(0.0, 0.0).is_none());

    // Axial geographic TieGrid
    let ties = [
        Tie {
            fx: 0.0,
            fy: 0.0,
            lat: 48.0,
            lon: 39.0,
        },
        Tie {
            fx: 1.0,
            fy: 0.0,
            lat: 48.0,
            lon: 41.0,
        },
        Tie {
            fx: 0.0,
            fy: 1.0,
            lat: 46.0,
            lon: 39.0,
        },
        Tie {
            fx: 1.0,
            fy: 1.0,
            lat: 46.0,
            lon: 41.0,
        },
    ];
    let grid = TieGrid::new(&ties).expect("valid axial grid");
    assert!(grid.is_axial());
    let gframe = Frame::Grid(grid);
    let (glat, glon) = gframe.geodetic(0.25, 0.75);
    let (gfx, gfy) = gframe.fraction_at(glat, glon).expect("inside tie grid");
    assert!((gfx - 0.25).abs() < 1e-9);
    assert!((gfy - 0.75).abs() < 1e-9);
    assert!(gframe.fraction_at(50.0, 40.0).is_none());
}
