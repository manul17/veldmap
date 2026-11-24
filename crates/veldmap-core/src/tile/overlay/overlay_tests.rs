use glam::DVec3;

use super::*;
use crate::geo::{Camera, Frame, Geodetic, Projection, WGS84};
use crate::tile::{cell_image_rect, grid_dim, level_count};

/// Measure the worst horizontal surface placement drift (in metres) across a
/// dense sample grid inside `cell_rect` between the CPU `f64` RTC warp patch
/// (including `f32` vertex quantization and triangle interpolation) and the
/// exact georeference binding.
fn max_surface_drift_m(
    frame: &Frame,
    width: u32,
    height: u32,
    cell_rect: [f64; 4],
    segments: u32,
) -> f64 {
    let patch = build_warp_patch(frame, width, height, cell_rect, segments);
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

            let drawn_ecef = patch.evaluate_ecef(tx, ty).expect("valid patch eval");
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
        level: u8,
        cell_x: u32,
        cell_y: u32,
    }

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
    let rotated_frame = Frame::from_pixel_affine(utm37.clone(), rotated_affine, 8_000, 5_000)
        .expect("rotated frame");

    // WebMercator axial benchmark (~10 m/px around 45N).
    let (wm_x0, wm_y0) = Projection::WebMercator.from_geodetic(45.0, 39.0);
    let (wm_x1, wm_y1) = Projection::WebMercator.from_geodetic(45.25, 39.35);
    let wm_frame =
        Frame::projected_bounds(Projection::WebMercator, wm_x0, wm_y0, wm_x1, wm_y1).expect("WM");

    let cases = [
        // 1. Drone orthophoto at 2 cm/px (UTM 37N)
        Case {
            name: "utm37_drone_2cm",
            frame: Frame::projected_bounds(
                utm37.clone(),
                550_000.0,
                5_230_000.0,
                550_200.0,
                5_230_200.0,
            )
            .expect("drone"),
            width: 10_000,
            height: 10_000,
            level: 0,
            cell_x: 3,
            cell_y: 5,
        },
        // 2. Sentinel-2 granule at 10 m/px (UTM 37N near zone edge)
        Case {
            name: "utm37_sentinel2_10m",
            frame: Frame::projected_bounds(
                utm37.clone(),
                600_000.0,
                5_200_000.0,
                709_800.0,
                5_309_800.0,
            )
            .expect("s2"),
            width: 10_980,
            height: 10_980,
            level: 0,
            cell_x: 20,
            cell_y: 1,
        },
        // 3. Landsat scene at 30 m/px (UTM 37N)
        Case {
            name: "utm37_landsat_30m",
            frame: Frame::projected_bounds(
                utm37.clone(),
                450_000.0,
                5_100_000.0,
                681_000.0,
                5_331_000.0,
            )
            .expect("landsat"),
            width: 7_700,
            height: 7_700,
            level: 0,
            cell_x: 14,
            cell_y: 2,
        },
        // 4. Sentinel-2 granule at pyramid level 4 (coarse UTM zone edge)
        Case {
            name: "utm37_sentinel2_level4_edge",
            frame: Frame::projected_bounds(utm37, 600_000.0, 5_200_000.0, 709_800.0, 5_309_800.0)
                .expect("s2 l4"),
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
            frame: Frame::projected_bounds(gk7, 7_450_000.0, 5_200_000.0, 7_559_800.0, 5_309_800.0)
                .expect("gk7"),
            width: 10_980,
            height: 10_980,
            level: 4,
            cell_x: 0,
            cell_y: 0,
        },
        // 7. Southern Hemisphere UTM 37S (EPSG:32737)
        Case {
            name: "utm37s_southern_hemisphere_native",
            frame: Frame::projected_bounds(utm37s, 500_000.0, 6_000_000.0, 500_200.0, 6_000_200.0)
                .expect("utm37s"),
            width: 10_000,
            height: 10_000,
            level: 0,
            cell_x: 4,
            cell_y: 4,
        },
        // 8. Axial WebMercator separable trig warp-mesh path
        Case {
            name: "web_mercator_axial_native",
            frame: wm_frame,
            width: 4_096,
            height: 4_096,
            level: 0,
            cell_x: 3,
            cell_y: 2,
        },
    ];

    for case in cases {
        let cell = cell_image_rect(
            case.cell_x,
            case.cell_y,
            u32::from(case.level),
            case.width,
            case.height,
        );
        let segs = level_segments(&case.frame, case.width, case.height, case.level);
        let drift_m = max_surface_drift_m(&case.frame, case.width, case.height, cell, segs);
        let (base_px_m, _) = case
            .frame
            .pixel_size_m(case.width, case.height)
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
fn negative_determinant_and_south_up_affines_wind_outward_and_emit_outward_normals() {
    let utm37 = Projection::from_epsg(32637).expect("UTM 37N");
    let cell = [0.0, 0.0, 512.0, 512.0];

    let north_up = Frame::from_pixel_affine(
        utm37.clone(),
        [10.0, 0.0, 500_000.0, 0.0, -10.0, 5_250_000.0],
        512,
        512,
    )
    .expect("north-up");
    let patch_normal = build_warp_patch(&north_up, 512, 512, cell, MIN_WARP_SEGMENTS);
    assert!(!patch_normal.flip_winding);

    let south_up = Frame::from_pixel_affine(
        utm37.clone(),
        [10.0, 0.0, 500_000.0, 0.0, 10.0, 5_240_000.0],
        512,
        512,
    )
    .expect("south-up");
    let patch_south = build_warp_patch(&south_up, 512, 512, cell, MIN_WARP_SEGMENTS);
    assert!(patch_south.flip_winding);

    let mirrored = Frame::from_pixel_affine(
        utm37,
        [-10.0, 0.0, 505_120.0, 0.0, -10.0, 5_250_000.0],
        512,
        512,
    )
    .expect("mirrored");
    let patch_mirrored = build_warp_patch(&mirrored, 512, 512, cell, MIN_WARP_SEGMENTS);
    assert!(patch_mirrored.flip_winding);

    for patch in [&patch_normal, &patch_south, &patch_mirrored] {
        let side = (patch.segments + 1) as usize;
        assert_eq!(patch.rel_positions.len(), side * side);
        assert_eq!(patch.normals.len(), side * side);
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
        assert!(tri_n.dot(p0) > 0.0);
        for (&rel, &n) in patch.rel_positions.iter().zip(&patch.normals) {
            let n_vec = DVec3::new(f64::from(n[0]), f64::from(n[1]), f64::from(n[2]));
            let p_vec = patch.origin_ecef_m
                + DVec3::new(f64::from(rel[0]), f64::from(rel[1]), f64::from(rel[2]));
            assert!((n_vec.length() - 1.0).abs() < 1e-5);
            assert!(n_vec.dot(p_vec) > 0.0);
        }
    }
}

#[test]
fn lazy_cull_tree_matches_brute_force_prunes_offscreen_and_keeps_limb_cells() {
    let utm37 = Projection::from_epsg(32637).expect("UTM 37N");
    let frame = Frame::projected_bounds(utm37, 500_000.0, 5_200_000.0, 600_000.0, 5_300_000.0)
        .expect("utm37");
    let width = 16_384;
    let height = 16_384;
    let max_level = (level_count(width, height) - 1) as u8;
    let mut tree = OverlayCullTree::new(width, height, max_level);

    // 1. Center zoom: lazy pruning materializes far fewer balls than 32x32=1024
    //    and matches brute-force level-0 culling with zero false negatives.
    let (center_lat, center_lon) = frame.geodetic(0.5, 0.5);
    let cam = Camera::at(center_lat, center_lon, 5_000.0);
    let frustum = cam.frustum(16.0 / 9.0);
    let visible = tree.visible_cells(&frame, 0, &frustum);
    assert!(!visible.is_empty());
    assert!(
        tree.cached_ball_count() < 200,
        "lazy culling should prune offscreen subtrees, materialized {} balls",
        tree.cached_ball_count()
    );

    let cols = grid_dim(width);
    let rows = grid_dim(height);
    let mut brute = Vec::new();
    for y in 0..rows {
        for x in 0..cols {
            let ball = compute_cell_ball(&frame, width, height, 0, x, y);
            if frustum.admits(&ball) {
                brute.push((x, y));
            }
        }
    }
    let mut visible_sorted = visible;
    visible_sorted.sort_unstable();
    brute.sort_unstable();
    assert_eq!(visible_sorted, brute);

    // 2. Limb visibility: place camera so the raster sits right on the visible
    //    eastern limb and verify visible cells are NOT culled.
    let limb_cam = Camera::at(center_lat, center_lon - 50.0, 4_000_000.0);
    let limb_frustum = limb_cam.frustum(16.0 / 9.0);
    let limb_visible = tree.visible_cells(&frame, 0, &limb_frustum);
    assert!(
        !limb_visible.is_empty(),
        "cells on the visible limb must remain admitted"
    );

    // 3. Opposite hemisphere: every cell must be rejected.
    let far_cam = Camera::at(-center_lat, center_lon - 180.0, 5_000_000.0);
    let far_frustum = far_cam.frustum(16.0 / 9.0);
    assert!(tree.visible_cells(&frame, 0, &far_frustum).is_empty());
}

#[test]
fn uniform_level_segments_eliminates_t_junctions_and_handles_antimeridian() {
    // 700x700 regional raster (2x2 tiles with clipped right/bottom edges).
    let regional =
        Frame::quad([(10.0, 0.0), (10.0, 14.0), (-4.0, 14.0), (-4.0, 0.0)]).expect("regional");
    let segs = level_segments(&regional, 700, 700, 0);
    assert!(segs >= MIN_WARP_SEGMENTS);
    let c00 = build_warp_patch(
        &regional,
        700,
        700,
        cell_image_rect(0, 0, 0, 700, 700),
        segs,
    );
    let c10 = build_warp_patch(
        &regional,
        700,
        700,
        cell_image_rect(1, 0, 0, 700, 700),
        segs,
    );
    assert_eq!(c00.segments, c10.segments);

    // Antimeridian 2-deg cell must stay at MIN_WARP_SEGMENTS (not blow up to 64
    // from a 358-deg raw longitude wrap).
    let antimeridian = Frame::quad([(5.0, 179.0), (5.0, -179.0), (3.0, -179.0), (3.0, 179.0)])
        .expect("antimeridian");
    assert_eq!(
        level_segments(&antimeridian, 512, 512, 0),
        MIN_WARP_SEGMENTS
    );
}
