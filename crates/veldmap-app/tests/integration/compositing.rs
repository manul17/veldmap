//! Offscreen sRGB/UNORM compositing parity, 0 idle re-renders, procedural
//! graticule horizon discard & polar sharpness, and dropped-encoder recovery.

use veldmap_app::render::{GlobeRenderer, RenderParams};
use veldmap_core::geo::camera::{Camera, MIN_HEIGHT_M};
use veldmap_core::tile::{QuadtreeConfig, SourceId};
use veldmap_raster::BUNDLED_MAX_LEVEL;

use crate::common::{
    build_snapshot, is_clear_background, pixel_at, render_srgb, render_with, shared_headless_device,
};

#[test]
fn srgb_and_unorm_compositing_parity_and_zero_idle_rerender() {
    let Some((device, queue)) = shared_headless_device() else {
        return;
    };

    let (w, h) = (129, 129);
    let cfg = QuadtreeConfig::geodetic(BUNDLED_MAX_LEVEL);
    let orbital_cam = Camera::default();
    let snap = build_snapshot(
        orbital_cam,
        w,
        h,
        true,
        false,
        SourceId::BUNDLED_BASEMAP,
        1,
        &cfg,
        None,
    );
    let orbital_params = RenderParams {
        width: w,
        height: h,
        snapshot: &snap,
    };

    let mut srgb_renderer = GlobeRenderer::new(device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut unorm_renderer = GlobeRenderer::new(device, wgpu::TextureFormat::Rgba8Unorm);

    let srgb_pixels = render_with(
        &mut srgb_renderer,
        device,
        queue,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        orbital_params,
        true,
    );
    // Identical Arc<DrawSnapshot> must execute 0 idle offscreen re-renders.
    let cached_pixels = render_with(
        &mut srgb_renderer,
        device,
        queue,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        orbital_params,
        false,
    );
    assert_eq!(cached_pixels, srgb_pixels);

    let unorm_pixels = render_with(
        &mut unorm_renderer,
        device,
        queue,
        wgpu::TextureFormat::Rgba8Unorm,
        orbital_params,
        true,
    );

    assert_eq!(srgb_pixels.len(), unorm_pixels.len());
    let max_diff = srgb_pixels
        .iter()
        .zip(unorm_pixels.iter())
        .map(|(&a, &b)| a.abs_diff(b))
        .max()
        .unwrap_or(0);
    assert!(
        max_diff <= 1,
        "sRGB and UNORM composite outputs diverged by {max_diff} LSB"
    );

    // Corner at orbital zoom must be the exact sRGB-encoded deep-space background (#0B1320).
    let corner = pixel_at(&srgb_pixels, w, 2, 2);
    assert!(
        is_clear_background(corner) && corner[3] == 255,
        "unexpected orbital background corner pixel: {corner:?}"
    );

    // Center at Camera::default() (45 deg N, 60 deg E) sits on an exact 15-degree
    // procedural graticule intersection -> #A8B99B (168, 185, 155, 255).
    let center = pixel_at(&srgb_pixels, w, w / 2, h / 2);
    assert!(
        center[0].abs_diff(168) <= 3
            && center[1].abs_diff(185) <= 3
            && center[2].abs_diff(155) <= 3,
        "orbital center pixel did not match procedural graticule sRGB #A8B99B: {center:?}"
    );
}

#[test]
fn graticule_toggle_horizon_discard_and_zoomed_polar_meridians() {
    let Some((device, queue)) = shared_headless_device() else {
        return;
    };

    let (w, h) = (129, 129);
    let cfg = QuadtreeConfig::geodetic(BUNDLED_MAX_LEVEL);
    let mut renderer = GlobeRenderer::new(device, wgpu::TextureFormat::Rgba8UnormSrgb);

    let orbital_cam = Camera::default();
    let with_grid_snap = build_snapshot(
        orbital_cam,
        w,
        h,
        true,
        false,
        SourceId::BUNDLED_BASEMAP,
        1,
        &cfg,
        None,
    );
    let no_grid_snap = build_snapshot(
        orbital_cam,
        w,
        h,
        false,
        false,
        SourceId::BUNDLED_BASEMAP,
        1,
        &cfg,
        None,
    );
    let _ = render_srgb(&mut renderer, device, queue, w, h, &with_grid_snap);
    let no_grid_pixels = render_srgb(&mut renderer, device, queue, w, h, &no_grid_snap);
    let body_center = pixel_at(&no_grid_pixels, w, w / 2, h / 2);
    assert!(
        body_center[0] > 180 && body_center[1] > 170 && body_center[2] > 150,
        "center pixel without graticule did not render lit ellipsoid in sRGB: {body_center:?}"
    );

    // At (37.5 N, 52.5 E, 14,000 km), front 15-degree graticule lines lie far
    // from viewport center; central 9x9 patch must stay free of back-side bleed.
    let cam = Camera::at(37.5, 52.5, 14_000_000.0);
    let s_on = build_snapshot(
        cam,
        w,
        h,
        true,
        false,
        SourceId::BUNDLED_BASEMAP,
        2,
        &cfg,
        None,
    );
    let s_off = build_snapshot(
        cam,
        w,
        h,
        false,
        false,
        SourceId::BUNDLED_BASEMAP,
        2,
        &cfg,
        None,
    );
    let with_grid = render_srgb(&mut renderer, device, queue, w, h, &s_on);
    let without_grid = render_srgb(&mut renderer, device, queue, w, h, &s_off);
    let mid = w / 2;
    for y in (mid - 4)..=(mid + 4) {
        for x in (mid - 4)..=(mid + 4) {
            assert_eq!(
                pixel_at(&with_grid, w, x, y),
                pixel_at(&without_grid, w, x, y),
                "back-hemisphere graticule bled into front pixel ({x}, {y})"
            );
        }
    }
    assert_ne!(with_grid, without_grid);

    // At 2 m zoom near 89.0 deg N on meridian 0.00005 deg E (an exact step
    // multiple strictly inside a PolarCap tile), meridians are well separated
    // on screen (`dw_lon ~ 0.15`) and must remain visibly drawn (not suppressed
    // by a fixed-latitude polar fade).
    let polar_cam = Camera::at(89.000_025, 0.000_05, MIN_HEIGHT_M);
    let polar_on = build_snapshot(
        polar_cam,
        129,
        129,
        true,
        false,
        SourceId::BUNDLED_BASEMAP,
        3,
        &cfg,
        None,
    );
    let polar_off = build_snapshot(
        polar_cam,
        129,
        129,
        false,
        false,
        SourceId::BUNDLED_BASEMAP,
        3,
        &cfg,
        None,
    );
    let p_on = render_srgb(&mut renderer, device, queue, 129, 129, &polar_on);
    let p_off = render_srgb(&mut renderer, device, queue, 129, 129, &polar_off);
    let on_px = pixel_at(&p_on, 129, 64, 64);
    let off_px = pixel_at(&p_off, 129, 64, 64);
    assert!(
        on_px[0].abs_diff(off_px[0]) >= 30,
        "zoomed-in polar meridian at 89.0 N was too faint or faded: on={on_px:?}, off={off_px:?}"
    );
}

#[test]
fn dropped_encoder_recovery_and_zero_viewport_safety() {
    let Some((device, queue)) = shared_headless_device() else {
        return;
    };

    let cfg = QuadtreeConfig::geodetic(BUNDLED_MAX_LEVEL);
    let mut renderer = GlobeRenderer::new(device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let cam = Camera::at(37.5, 52.5, 14_000_000.0);
    let empty_snap = build_snapshot(
        cam,
        0,
        0,
        true,
        false,
        SourceId::BUNDLED_BASEMAP,
        1,
        &cfg,
        None,
    );

    let mut zero_enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    assert!(!renderer.render_if_dirty(
        device,
        queue,
        &mut zero_enc,
        RenderParams {
            width: 0,
            height: 0,
            snapshot: &empty_snap,
        },
    ));

    let snap = build_snapshot(
        cam,
        65,
        65,
        true,
        false,
        SourceId::BUNDLED_BASEMAP,
        1,
        &cfg,
        None,
    );
    let params = RenderParams {
        width: 65,
        height: 65,
        snapshot: &snap,
    };
    let mut dropped_enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    assert!(renderer.render_if_dirty(device, queue, &mut dropped_enc, params));
    drop(dropped_enc);

    let recovered = render_srgb(&mut renderer, device, queue, 65, 65, &snap);
    let center = pixel_at(&recovered, 65, 32, 32);
    assert!(
        center[0] > 180 && center[1] > 170 && center[2] > 150,
        "recovered render did not produce lit globe center: {center:?}"
    );
}
