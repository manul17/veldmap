use glam::Vec3;
use veldmap_core::geo::geodesy::Geodetic;
use veldmap_core::tile::{
    DEFAULT_GRID_CELLS, PlanetaryScheme, QuadTileId, QuadtreeConfig, describe_tile,
};

use super::*;

#[test]
fn all_planetary_and_polar_surface_triangles_have_outward_ccw_winding() {
    let cfg = QuadtreeConfig::geodetic(3);
    let shared = SharedTileIndices::build(DEFAULT_GRID_CELLS);
    let samples = [
        QuadTileId {
            scheme: PlanetaryScheme::Geodetic,
            level: 1,
            x: 2,
            y: 0,
        },
        QuadTileId {
            scheme: PlanetaryScheme::Geodetic,
            level: 3,
            x: 0,
            y: 7,
        },
        QuadTileId {
            scheme: PlanetaryScheme::WebMercator,
            level: 2,
            x: 1,
            y: 2,
        },
        QuadTileId {
            scheme: PlanetaryScheme::PolarCap { north: true },
            level: 0,
            x: 0,
            y: 0,
        },
        QuadTileId {
            scheme: PlanetaryScheme::PolarCap { north: false },
            level: 0,
            x: 0,
            y: 0,
        },
        QuadTileId {
            scheme: PlanetaryScheme::PolarCap { north: true },
            level: 2,
            x: 2,
            y: 1,
        },
    ];

    for id in samples {
        let tile = describe_tile(id, &cfg).expect("valid tile");
        let verts = build_tile_vertices(&tile, DEFAULT_GRID_CELLS);
        let vcount = verts.len() as u32;
        assert!(shared.indices.iter().all(|&i| i < vcount));

        // Every surface triangle must face outward (dot with vertex normal > 0).
        for tri in shared.indices[shared.surface.start as usize..shared.surface.end as usize]
            .chunks_exact(3)
        {
            let v0 = verts[tri[0] as usize];
            let v1 = verts[tri[1] as usize];
            let v2 = verts[tri[2] as usize];
            let p0 = Vec3::from(v0.rel_pos);
            let p1 = Vec3::from(v1.rel_pos);
            let p2 = Vec3::from(v2.rel_pos);
            let face = (p1 - p0).cross(p2 - p0);
            let normal =
                (Vec3::from(v0.normal) + Vec3::from(v1.normal) + Vec3::from(v2.normal)).normalize();
            assert!(
                face.dot(normal) > 0.0,
                "inverted surface triangle on {:?}: dot={}",
                id,
                face.dot(normal)
            );
        }

        // Every skirt triangle must face laterally outward away from the
        // tile's local RTC origin (dot with tangential edge offset > 0).
        for tri in
            shared.indices[shared.surface.end as usize..shared.full.end as usize].chunks_exact(3)
        {
            let p0 = Vec3::from(verts[tri[0] as usize].rel_pos);
            let p1 = Vec3::from(verts[tri[1] as usize].rel_pos);
            let p2 = Vec3::from(verts[tri[2] as usize].rel_pos);
            let face = (p1 - p0).cross(p2 - p0);
            let centroid = (p0 + p1 + p2) / 3.0;
            assert!(
                face.dot(centroid) > 0.0,
                "inverted skirt triangle on {:?}: dot={}",
                id,
                face.dot(centroid)
            );
        }
    }
}

#[test]
fn per_tile_rtc_preserves_sub_millimetre_precision_at_2m_zoom() {
    let cfg = QuadtreeConfig::geodetic(3);
    // z = 16 tile containing (47.2357 N, 39.7015 E).
    let level = 16u8;
    let (cols, rows) = PlanetaryScheme::Geodetic.grid_dims(level).expect("dims");
    let x = (((39.7015 + 180.0) / 360.0) * f64::from(cols)).floor() as u32;
    let y = (((90.0 - 47.2357) / 180.0) * f64::from(rows)).floor() as u32;
    let tile = describe_tile(
        QuadTileId {
            scheme: PlanetaryScheme::Geodetic,
            level,
            x,
            y,
        },
        &cfg,
    )
    .expect("z=16 tile");
    let verts = build_tile_vertices(&tile, DEFAULT_GRID_CELLS);

    // Reconstruct two adjacent surface vertices via RTC and compare against f64 ECEF.
    let v0 = verts[0];
    let v1 = verts[1];
    let step_rtc_m = f64::from((Vec3::from(v1.rel_pos) - Vec3::from(v0.rel_pos)).length());
    let (lat0, lon0, _, _, _, _) = sample_tile_surface(tile.id, 0.0, 0.0).expect("sample 0");
    let (lat1, lon1, _, _, _, _) =
        sample_tile_surface(tile.id, 1.0 / f64::from(DEFAULT_GRID_CELLS), 0.0).expect("sample 1");
    let step_exact_m = geodesy::world(Geodetic::surface(lat0, lon0))
        .distance(geodesy::world(Geodetic::surface(lat1, lon1)));
    assert!(
        (step_rtc_m - step_exact_m).abs() < 1e-4,
        "RTC vertex step drifted: rtc={step_rtc_m} m vs exact={step_exact_m} m"
    );
}

#[test]
fn skirts_drop_inward_by_at_least_four_times_cell_sag() {
    let cfg = QuadtreeConfig::geodetic(3);
    let tile = describe_tile(
        QuadTileId {
            scheme: PlanetaryScheme::Geodetic,
            level: 2,
            x: 3,
            y: 1,
        },
        &cfg,
    )
    .expect("tile");
    let verts = build_tile_vertices(&tile, DEFAULT_GRID_CELLS);
    let stride = (DEFAULT_GRID_CELLS + 1) as usize;
    let north_skirt_start = stride * stride;

    for col in 0..stride {
        let surf = verts[col];
        let skirt = verts[north_skirt_start + col];
        let delta = Vec3::from(skirt.rel_pos) - Vec3::from(surf.rel_pos);
        let inward = -delta.dot(Vec3::from(surf.normal));
        let tol = (tile.skirt_depth_m * 1e-4).max(1e-3);
        assert!(
            (f64::from(inward) - tile.skirt_depth_m).abs() < tol,
            "skirt drop {inward} m != expected {} m (tol {tol})",
            tile.skirt_depth_m
        );
        assert!(f64::from(inward) >= (4.0 * tile.sag_cell_diag_m).max(0.001) - tol);
    }
}
