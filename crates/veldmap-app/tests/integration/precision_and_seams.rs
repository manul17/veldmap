//! Bundled NASA Blue Marble landmark rendering, 2 m close-up between-vertices
//! RTC precision & coarse parent overzoom, and multi-LOD / polar seam tests.

use veldmap_app::gpu_tiles::{GpuTileCache, bundled_tile_key};
use veldmap_app::render::GlobeRenderer;
use veldmap_core::geo::camera::{Camera, MIN_HEIGHT_M};
use veldmap_core::geo::projection::WEB_MERCATOR_LIMIT_DEG;
use veldmap_core::tile::{
    DecodedTile, PixelFormat, PlanetaryScheme, QuadTileId, QuadtreeConfig, SourceId,
};
use veldmap_raster::BUNDLED_MAX_LEVEL;

use crate::common::{
    build_snapshot, is_clear_background, pixel_at, render_srgb, shared_bundled_basemap,
    shared_headless_device,
};

#[test]
fn bundled_blue_marble_renders_sahara_pacific_and_antarctica() {
    let Some((device, queue)) = shared_headless_device() else {
        return;
    };

    let basemap = shared_bundled_basemap();
    let cfg = QuadtreeConfig::geodetic(BUNDLED_MAX_LEVEL);
    let mut renderer = GlobeRenderer::new(device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut cache = GpuTileCache::new(32 * 1024 * 1024);

    for id in [
        QuadTileId {
            scheme: PlanetaryScheme::Geodetic,
            level: 0,
            x: 0,
            y: 0,
        },
        QuadTileId {
            scheme: PlanetaryScheme::Geodetic,
            level: 0,
            x: 1,
            y: 0,
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
    ] {
        let decoded = basemap.render_tile(id).expect("render bundled root");
        cache
            .upload_and_insert(device, queue, bundled_tile_key(id), &decoded)
            .expect("upload root");
    }

    let (w, h) = (65, 65);

    // 1. Sahara Desert (23 N, 13 E) -> bright tan (R > B + 35).
    let sahara_snap = build_snapshot(
        Camera::at(23.0, 13.0, 2_000_000.0),
        w,
        h,
        false,
        true,
        SourceId::BUNDLED_BASEMAP,
        10,
        &cfg,
        Some(&mut cache),
    );
    let s_mid = pixel_at(
        &render_srgb(&mut renderer, device, queue, w, h, &sahara_snap),
        w,
        w / 2,
        h / 2,
    );
    assert!(
        s_mid[0] > 140 && s_mid[0] > s_mid[2] + 35,
        "Sahara textured center did not look tan: {s_mid:?}"
    );

    // 2. Central Pacific (0 N, -150 E) -> deep ocean blue (B > R + 15, R < 50).
    let pacific_snap = build_snapshot(
        Camera::at(0.0, -150.0, 2_000_000.0),
        w,
        h,
        false,
        true,
        SourceId::BUNDLED_BASEMAP,
        11,
        &cfg,
        Some(&mut cache),
    );
    let p_mid = pixel_at(
        &render_srgb(&mut renderer, device, queue, w, h, &pacific_snap),
        w,
        w / 2,
        h / 2,
    );
    assert!(
        p_mid[2] > p_mid[0] + 15 && p_mid[0] < 50,
        "Pacific textured center did not look ocean blue: {p_mid:?}"
    );

    // 3. South Pole (-89.5 S, 0 E) -> bright Antarctica white via PolarCap texture.
    let south_snap = build_snapshot(
        Camera::at(-89.5, 0.0, 500_000.0),
        w,
        h,
        false,
        true,
        SourceId::BUNDLED_BASEMAP,
        12,
        &cfg,
        Some(&mut cache),
    );
    let sp_mid = pixel_at(
        &render_srgb(&mut renderer, device, queue, w, h, &south_snap),
        w,
        w / 2,
        h / 2,
    );
    assert!(
        sp_mid[0] > 180 && sp_mid[1] > 180 && sp_mid[2] > 180,
        "Antarctica PolarCap center did not render bright white: {sp_mid:?}"
    );
}

#[test]
fn close_up_2m_between_vertices_rtc_shift_and_parent_overzoom() {
    let Some((device, queue)) = shared_headless_device() else {
        return;
    };

    let (w, h) = (65, 65);
    let cfg = QuadtreeConfig::geodetic(BUNDLED_MAX_LEVEL);
    let mut renderer = GlobeRenderer::new(device, wgpu::TextureFormat::Rgba8UnormSrgb);

    // 1. Midpoint between-vertices sag clearance at 2 m altitude.
    let between_cam = Camera::at(45.000_73, 45.000_73, MIN_HEIGHT_M);
    let between_snap = build_snapshot(
        between_cam,
        w,
        h,
        false,
        false,
        SourceId::BUNDLED_BASEMAP,
        1,
        &cfg,
        None,
    );
    let max_level = between_snap
        .tiles
        .iter()
        .map(|t| t.visible.id.level)
        .max()
        .unwrap_or(0);
    assert!(
        max_level > cfg.max_tex_level,
        "2 m close-up must decouple z_geom ({max_level}) > max_tex_level ({})",
        cfg.max_tex_level
    );

    let between_px = render_srgb(&mut renderer, device, queue, w, h, &between_snap);
    for y in 0..h {
        for x in 0..w {
            let px = pixel_at(&between_px, w, x, y);
            assert!(
                !is_clear_background(px) && px[0] > 140 && px[1] > 130,
                "2 m between-vertices pixel ({x},{y}) clipped or leaked crack: {px:?}"
            );
        }
    }

    // 2. Sub-metre RTC + graticule phase shift at 2 m (0.25 m < 1 ULP of global f32 ECEF).
    let dlon_25cm = 0.25 / (111_320.0 * 45.0_f64.to_radians().cos());
    let s0 = build_snapshot(
        Camera::at(45.0, 45.0, MIN_HEIGHT_M),
        w,
        h,
        true,
        false,
        SourceId::BUNDLED_BASEMAP,
        2,
        &cfg,
        None,
    );
    let s1 = build_snapshot(
        Camera::at(45.0, 45.0 + dlon_25cm, MIN_HEIGHT_M),
        w,
        h,
        true,
        false,
        SourceId::BUNDLED_BASEMAP,
        3,
        &cfg,
        None,
    );
    assert_ne!(
        render_srgb(&mut renderer, device, queue, w, h, &s0),
        render_srgb(&mut renderer, device, queue, w, h, &s1),
        "0.25 m shift at 2 m altitude collapsed"
    );

    // 3. Coarse parent texture overzoom at 2 m altitude:
    //    East root (x=1, y=0) painted in distinct quadrants:
    //    NW (0..90 E, 0..90 N) = red, NE (90..180 E, 0..90 N) = green,
    //    SW (0..90 E, -90..0 S) = blue, SE (90..180 E, -90..0 S) = yellow.
    let mut quad_data = vec![0u8; 64 * 64 * 4];
    for py in 0..64u32 {
        for px in 0..64u32 {
            let color = match (px < 32, py < 32) {
                (true, true) => [220, 40, 40, 255],
                (false, true) => [40, 210, 40, 255],
                (true, false) => [40, 80, 220, 255],
                (false, false) => [220, 200, 40, 255],
            };
            let off = ((py * 64 + px) * 4) as usize;
            quad_data[off..off + 4].copy_from_slice(&color);
        }
    }
    let synthetic_east = DecodedTile::try_new(64, 64, PixelFormat::Rgba8UnormSrgb, quad_data)
        .expect("valid synthetic east tile");
    let mut overzoom_cache = GpuTileCache::new(1024 * 1024);
    let east_root = bundled_tile_key(QuadTileId {
        scheme: PlanetaryScheme::Geodetic,
        level: 0,
        x: 1,
        y: 0,
    });
    overzoom_cache
        .upload_and_insert(device, queue, east_root, &synthetic_east)
        .expect("upload synthetic root");

    // (45 N, 45 E) sits at the center of the NW quadrant (RED) of east root (x=1, y=0).
    let overzoom_snap = build_snapshot(
        Camera::at(45.0, 45.0, MIN_HEIGHT_M),
        w,
        h,
        false,
        true,
        SourceId::BUNDLED_BASEMAP,
        4,
        &cfg,
        Some(&mut overzoom_cache),
    );
    let overzoom_px = render_srgb(&mut renderer, device, queue, w, h, &overzoom_snap);
    for y in 0..h {
        for x in 0..w {
            let px = pixel_at(&overzoom_px, w, x, y);
            assert!(
                px[0] > 200 && px[1] < 60 && px[2] < 60,
                "coarse parent overzoom pixel ({x},{y}) did not sample red sub-rect: {px:?}"
            );
        }
    }

    // 4. PolarCap close-up (2 m) procedural graticule precision at (86 N, 0 E):
    //    test both 0.25 m latitude shift (colatitude delta) and 0.25 m
    //    longitude shift (cancellation-free Taylor meridian path).
    let dlat_25cm = 0.25 / 111_320.0;
    let dlon_25cm_86n = 0.25 / (111_320.0 * 86.0_f64.to_radians().cos());
    let pc_on = build_snapshot(
        Camera::at(86.0, 0.0, MIN_HEIGHT_M),
        w,
        h,
        true,
        false,
        SourceId::BUNDLED_BASEMAP,
        6,
        &cfg,
        None,
    );
    let pc_lat_shifted = build_snapshot(
        Camera::at(86.0 + dlat_25cm, 0.0, MIN_HEIGHT_M),
        w,
        h,
        true,
        false,
        SourceId::BUNDLED_BASEMAP,
        7,
        &cfg,
        None,
    );
    let pc_lon_shifted = build_snapshot(
        Camera::at(86.0, dlon_25cm_86n, MIN_HEIGHT_M),
        w,
        h,
        true,
        false,
        SourceId::BUNDLED_BASEMAP,
        8,
        &cfg,
        None,
    );
    let pc_base_px = render_srgb(&mut renderer, device, queue, w, h, &pc_on);
    assert_ne!(
        pc_base_px,
        render_srgb(&mut renderer, device, queue, w, h, &pc_lat_shifted),
        "0.25 m latitude shift on PolarCap at (86 N, 2 m) collapsed"
    );
    assert_ne!(
        pc_base_px,
        render_srgb(&mut renderer, device, queue, w, h, &pc_lon_shifted),
        "0.25 m longitude shift on PolarCap at (86 N, 2 m) collapsed"
    );
}

#[test]
fn multi_lod_polar_seam_and_web_mercator_continuity() {
    let Some((device, queue)) = shared_headless_device() else {
        return;
    };

    let cfg = QuadtreeConfig::geodetic(BUNDLED_MAX_LEVEL);
    let mut renderer = GlobeRenderer::new(device, wgpu::TextureFormat::Rgba8UnormSrgb);

    // 1. Genuine multi-LOD transition: assert `max_l > min_l` strictly and zero cracks.
    let (mw, mh) = (193, 193);
    let multi_lod_cam = Camera::at(45.0, 39.0, 3_500_000.0);
    let multi_lod_snap = build_snapshot(
        multi_lod_cam,
        mw,
        mh,
        true,
        false,
        SourceId::BUNDLED_BASEMAP,
        1,
        &cfg,
        None,
    );
    let min_l = multi_lod_snap
        .tiles
        .iter()
        .map(|t| t.visible.id.level)
        .min()
        .unwrap_or(0);
    let max_l = multi_lod_snap
        .tiles
        .iter()
        .map(|t| t.visible.id.level)
        .max()
        .unwrap_or(0);
    assert!(
        max_l > min_l,
        "expected a genuine multi-LOD transition, got min_l={min_l}, max_l={max_l}"
    );
    let multi_lod_px = render_srgb(&mut renderer, device, queue, mw, mh, &multi_lod_snap);
    for y in 0..mh {
        for x in 0..mw {
            let px = pixel_at(&multi_lod_px, mw, x, y);
            assert!(
                !is_clear_background(px),
                "multi-LOD seam leaked background at ({x},{y}): {px:?}"
            );
        }
    }

    // 2. Polar boundary seam, exact North Pole, near South Pole, and WebMercator (25 N, -135 W).
    let (w, h) = (65, 65);
    let merc_cfg = QuadtreeConfig::web_mercator(10);
    for (label, cam, active_cfg) in [
        (
            "Geodetic polar boundary seam at 50 m",
            Camera::at(WEB_MERCATOR_LIMIT_DEG, 15.0, 50.0),
            &cfg,
        ),
        (
            "WebMercator(10) polar seam at 5 km",
            Camera::at(WEB_MERCATOR_LIMIT_DEG, 15.0, 5_000.0),
            &merc_cfg,
        ),
        (
            "WebMercator(10) mid-latitude (25 N, -135 W) at 2 m",
            Camera::at(25.0, -135.0, MIN_HEIGHT_M),
            &merc_cfg,
        ),
        (
            "exact North Pole at 2 m",
            Camera::at(90.0, 0.0, MIN_HEIGHT_M),
            &cfg,
        ),
        (
            "near South Pole at 2 m",
            Camera::at(-89.995, 135.0, MIN_HEIGHT_M),
            &cfg,
        ),
    ] {
        let seam_snap = build_snapshot(
            cam,
            w,
            h,
            true,
            false,
            SourceId::BUNDLED_BASEMAP,
            5,
            active_cfg,
            None,
        );
        assert!(
            !seam_snap.tiles.is_empty(),
            "{label}: quadtree produced 0 tiles"
        );
        let seam_px = render_srgb(&mut renderer, device, queue, w, h, &seam_snap);
        for y in 0..h {
            for x in 0..w {
                let px = pixel_at(&seam_px, w, x, y);
                assert!(
                    !is_clear_background(px),
                    "{label}: background crack leaked at ({x},{y}): {px:?}"
                );
            }
        }
    }
}
