use super::*;
use crate::geo::camera::MIN_HEIGHT_M;
use crate::geo::geodesy::Geodetic;
use crate::geo::projection::WEB_MERCATOR_LIMIT_DEG;
use crate::tile::geometry::{POLAR_R_CAP, sample_tile_surface};

#[test]
fn decoupled_lod_subdivides_past_max_tex_level_with_zero_requests_above_cap() {
    let cfg = QuadtreeConfig::bundled_geodetic();
    let cam = Camera::at(47.2357, 39.7015, MIN_HEIGHT_M);
    let tiles = traverse(&cam, 1280, 800, &cfg);

    assert!(!tiles.is_empty(), "must produce visible tiles at 2 m");
    let max_geom_seen = tiles.iter().map(|t| t.id.level).max().unwrap_or(0);
    assert!(
        max_geom_seen > cfg.max_tex_level,
        "geometry must subdivide past max_tex_level={} at 2 m (got {max_geom_seen})",
        cfg.max_tex_level
    );
    assert!(
        max_geom_seen <= cfg.max_geom_level,
        "geometry must respect max_geom_level={} (got {max_geom_seen})",
        cfg.max_geom_level
    );

    let nadir_ecef = geodesy::world(Geodetic::surface(47.2357, 39.7015));
    let focal_px = cam.focal_length_px(800.0);
    let eye = cam.eye_ecef();
    let mut nadir_tile_sag = f64::INFINITY;
    for tile in &tiles {
        let max_tex = cfg.max_tex_for(tile.id.scheme);
        assert!(
            tile.tex_id.level <= max_tex,
            "texture request {:?} exceeded max_tex_level={max_tex}",
            tile.tex_id
        );
        let dist_m = (tile.ball.centre.distance(eye) - tile.ball.radius)
            .max(cam.height_m())
            .max(1.0);
        let sse_px = (tile.sag_cell_diag_m * focal_px) / dist_m;
        assert!(
            sse_px <= cfg.max_geom_sse_px + 1e-9,
            "tile {:?} exceeded SSE cap: {sse_px} px",
            tile.id
        );
        if tile.ball.centre.distance(nadir_ecef) <= tile.ball.radius {
            nadir_tile_sag = nadir_tile_sag.min(tile.sag_cell_diag_m);
        }
        assert!(
            tile.skirt_depth_m >= (4.0 * tile.sag_cell_diag_m).max(0.001) - 1e-12,
            "skirt depth too shallow: {}",
            tile.skirt_depth_m
        );
    }
    assert!(
        nadir_tile_sag < 0.001,
        "nadir tile cell diagonal sag {nadir_tile_sag} m exceeded 1 mm at 2 m altitude"
    );
}

#[test]
fn adaptive_polar_cap_covers_pole_with_sub_cm_sag_at_2m_and_exact_boundary() {
    let cfg = QuadtreeConfig::bundled_geodetic();
    for pole in [90.0, -90.0] {
        let cam = Camera::at(pole, 0.0, MIN_HEIGHT_M);
        let tiles = traverse(&cam, 1024, 768, &cfg);
        assert!(!tiles.is_empty());
        assert!(
            tiles.iter().all(
                |t| matches!(t.id.scheme, PlanetaryScheme::PolarCap { north } if north == (pole > 0.0))
            ),
            "at {pole} deg at 2 m, only matching PolarCap tiles should be visible"
        );
        for tile in &tiles {
            assert!(
                tile.sag_cell_diag_m < 0.01,
                "polar cap sag {} m exceeded 1 cm at 2 m altitude",
                tile.sag_cell_diag_m
            );
            assert!(tile.tex_id.level <= cfg.polar_max_tex_level);
        }
    }

    // Verify exact boundary alignment along |u|=1 or |v|=1 -> |lat| == WEB_MERCATOR_LIMIT_DEG.
    for north in [true, false] {
        let sign = if north { 1.0 } else { -1.0 };
        for step in -10..=10 {
            let t = f64::from(step) / 10.0;
            for (u, v) in [(1.0, t), (-1.0, t), (t, 1.0), (t, -1.0)] {
                let (lat, lon, dx, dy) = polar_cap_point(north, u, v);
                assert!(
                    (lat - sign * WEB_MERCATOR_LIMIT_DEG).abs() < 1e-11,
                    "boundary ({u},{v}) landed at lat={lat}"
                );
                assert!(((dx * dx + dy * dy) - 1.0).abs() < 1e-12);
                assert!(lon.is_finite());
            }
        }
        // Exact center maps to the exact pole.
        let (lat0, _, dx0, dy0) = polar_cap_point(north, 0.0, 0.0);
        assert!((lat0 - sign * 90.0).abs() < 1e-12);
        assert_eq!((dx0, dy0), (0.0, 0.0));
    }
}

#[test]
fn quadtree_is_2_to_1_balanced_across_visible_neighbors_and_antimeridian() {
    let cfg = QuadtreeConfig::bundled_geodetic();
    for cam in [
        Camera::at(45.0, 39.0, 500_000.0),
        Camera::at(35.0, 180.0, 2_000_000.0),
    ] {
        let tiles = traverse(&cam, 1280, 800, &cfg);
        assert!(!tiles.is_empty());
        let map: HashMap<QuadTileId, VisibleTile> = tiles.iter().map(|&t| (t.id, t)).collect();

        for &id in map.keys() {
            if id.level < 2 {
                continue;
            }
            for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                let Some(nb) = neighbor_id(id, dx, dy) else {
                    continue;
                };
                let mut anc = nb.parent().and_then(QuadTileId::parent);
                while let Some(a) = anc {
                    assert!(
                        !map.contains_key(&a),
                        "2:1 LOD violation: {:?} is adjacent to coarse leaf {:?}",
                        id,
                        a
                    );
                    anc = a.parent();
                }
            }
        }
    }
}

#[test]
fn hidpi_viewport_and_sse_config_control_subdivision_strictly() {
    // Use a high-altitude view where geometry SSE (not max_tex_level) governs refinement.
    let mut cfg = QuadtreeConfig::bundled_geodetic();
    cfg.max_tex_level = 0;
    let cam = Camera::at(35.0, 20.0, 1_500_000.0);
    let lod_1x = traverse(&cam, 600, 400, &cfg);
    let lod_2x = traverse(&cam, 2400, 1600, &cfg);
    let max_1x = lod_1x.iter().map(|t| t.id.level).max().unwrap_or(0);
    let max_2x = lod_2x.iter().map(|t| t.id.level).max().unwrap_or(0);
    assert!(
        max_2x > max_1x,
        "HiDPI 4x viewport must subdivide strictly finer than 1x: 1x={max_1x}, 4x={max_2x}"
    );

    // Relaxing max_geom_sse_px on a standard 800 px viewport must reduce subdivision.
    let mut loose = cfg;
    loose.max_geom_sse_px = 8.0;
    let tight_tiles = traverse(&cam, 1280, 800, &cfg);
    let loose_tiles = traverse(&cam, 1280, 800, &loose);
    let tight_max = tight_tiles.iter().map(|t| t.id.level).max().unwrap_or(0);
    let loose_max = loose_tiles.iter().map(|t| t.id.level).max().unwrap_or(0);
    assert!(
        loose_max < tight_max,
        "relaxing max_geom_sse_px must reduce LOD on 800 px viewport: tight={tight_max}, loose={loose_max}"
    );

    // Degenerate zero-area viewports return empty cleanly.
    assert!(traverse(&cam, 0, 800, &cfg).is_empty());
    assert!(traverse(&cam, 800, 0, &cfg).is_empty());
}

#[test]
fn web_mercator_never_falsely_culls_across_globe_and_couples_polar_seam() {
    let cfg = QuadtreeConfig::web_mercator(12);
    // Sweep across mid-latitude quadrants including (25 N, -135 W) where
    // naive z=0 subcell horizon tests falsely cull the root tile.
    for lat in [-60.0, -25.0, 0.0, 25.0, 60.0] {
        for lon in [-135.0, -45.0, 0.0, 45.0, 135.0] {
            for height_m in [MIN_HEIGHT_M, 10_000.0, 400_000.0] {
                let cam = Camera::at(lat, lon, height_m);
                let tiles = traverse(&cam, 1280, 800, &cfg);
                assert!(
                    tiles
                        .iter()
                        .any(|t| t.id.scheme == PlanetaryScheme::WebMercator),
                    "WebMercator returned 0 tiles at ({lat}, {lon}, {height_m} m)"
                );
            }
        }
    }

    // At 5 km over the 85.0511 deg polar seam with WebMercator(12), both
    // WebMercator and PolarCap must appear, PolarCap textures stay <= cap,
    // and PolarCap perimeter sag stays bounded against WebMercator skirts.
    let cam = Camera::at(WEB_MERCATOR_LIMIT_DEG, 15.0, 5_000.0);
    let tiles = traverse(&cam, 1280, 800, &cfg);
    let merc: Vec<&VisibleTile> = tiles
        .iter()
        .filter(|t| t.id.scheme == PlanetaryScheme::WebMercator)
        .collect();
    let polar: Vec<&VisibleTile> = tiles
        .iter()
        .filter(|t| t.id.scheme == PlanetaryScheme::PolarCap { north: true })
        .collect();
    assert!(!merc.is_empty() && !polar.is_empty());
    for t in &polar {
        assert!(t.tex_id.level <= cfg.polar_max_tex_level);
    }
    let max_merc_level = merc.iter().map(|t| t.id.level).max().unwrap_or(0);
    let max_polar_level = polar.iter().map(|t| t.id.level).max().unwrap_or(0);
    assert!(
        max_polar_level + 3 >= max_merc_level,
        "PolarCap perimeter lagged WebMercator at seam: merc={max_merc_level}, polar={max_polar_level}"
    );
}

#[test]
fn clamped_geodetic_polar_adjacent_skirts_cover_coarser_neighbor_sag() {
    let cfg = QuadtreeConfig::bundled_geodetic();
    let coarse = describe_tile(
        QuadTileId {
            scheme: PlanetaryScheme::Geodetic,
            level: 4,
            x: 0,
            y: 0,
        },
        &cfg,
    )
    .expect("z=4 row 0");
    let fine = describe_tile(
        QuadTileId {
            scheme: PlanetaryScheme::Geodetic,
            level: 5,
            x: 2,
            y: 0,
        },
        &cfg,
    )
    .expect("z=5 row 0");
    assert!(
        fine.skirt_depth_m >= coarse.sag_cell_diag_m,
        "clamped Geodetic z=5 skirt ({} m) shallower than adjacent z=4 sag ({} m)",
        fine.skirt_depth_m,
        coarse.sag_cell_diag_m
    );
}

#[test]
fn polar_r_cap_constant_and_bounds_guards_are_exact() {
    let expected = ((90.0 - WEB_MERCATOR_LIMIT_DEG) * 0.5).to_radians().tan();
    assert!((POLAR_R_CAP - expected).abs() < 1e-16);

    // Out-of-bounds WebMercator and PolarCap tile coordinates return None.
    assert!(
        sample_tile_surface(
            QuadTileId {
                scheme: PlanetaryScheme::WebMercator,
                level: 1,
                x: 2,
                y: 0,
            },
            0.5,
            0.5,
        )
        .is_none()
    );
    assert!(
        sample_tile_surface(
            QuadTileId {
                scheme: PlanetaryScheme::PolarCap { north: true },
                level: 0,
                x: 1,
                y: 0,
            },
            0.5,
            0.5,
        )
        .is_none()
    );
    // Huge dz in clamp_to_level must not panic.
    let deep = QuadTileId {
        scheme: PlanetaryScheme::WebMercator,
        level: 40,
        x: 0,
        y: 0,
    };
    let (clamped, _) = deep.clamp_to_level(0);
    assert_eq!(clamped.level, 0);
}
