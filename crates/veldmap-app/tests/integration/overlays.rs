//! Headless GPU smoke tests for post-body depth-less raster overlays:
//! warp-mesh draping, negative-determinant south-up outward winding, Pass 2
//! graticule-on-top ordering, sag-safe `2 m` low-altitude horizon facing,
//! limb silhouette containment, multi-layer opacity/nodata/ancestor `uv_rect`
//! compositing, `Weak<GpuTile>` bind-group sweeping, and `0` idle re-renders.

use std::sync::Arc;

use veldmap_app::gpu_tiles::GpuTileCache;
use veldmap_app::render::{
    DrawOverlayLayer, DrawOverlayPatch, DrawSnapshot, GlobeRenderer, OverlayMeshKey, RenderParams,
    SnapshotBuildParams,
};
use veldmap_core::geo::Frame;
use veldmap_core::geo::camera::Camera;
use veldmap_core::geo::projection::Projection;
use veldmap_core::tile::{
    BandKey, DecodedTile, MIN_WARP_SEGMENTS, PixelFormat, QuadtreeConfig, SourceId, TileKey,
    TileScheme, build_warp_patch, level_segments,
};

use crate::common::{
    is_clear_background, pixel_at, render_srgb, render_with, shared_headless_device,
};

fn upload_overlay_tile(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    cache: &mut GpuTileCache,
    source: SourceId,
    w: u32,
    h: u32,
    mut pixel_fn: impl FnMut(u32, u32) -> [u8; 4],
) -> (TileKey, Arc<veldmap_app::gpu_tiles::GpuTile>) {
    let key = TileKey {
        source,
        band: BandKey::Display,
        scheme: TileScheme::RasterPyramid { max_level: 0 },
        format: PixelFormat::Rgba8UnormSrgb,
        gutter: 0,
        decode_rev: 1,
        level: 0,
        x: 0,
        y: 0,
    };
    let mut data = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            data.extend_from_slice(&pixel_fn(x, y));
        }
    }
    let decoded =
        DecodedTile::try_new(w, h, PixelFormat::Rgba8UnormSrgb, data).expect("decoded tile");
    let (gpu_tile, _) = cache
        .upload_and_insert(device, queue, key, &decoded)
        .expect("uploaded overlay tile");
    (key, gpu_tile)
}

#[test]
fn overlay_warp_mesh_drapes_with_south_up_winding_graticule_on_top_and_zero_idle_rerender() {
    let Some((device, queue)) = shared_headless_device() else {
        return;
    };

    let (width, height) = (128u32, 128u32);
    let mut renderer = GlobeRenderer::new(device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut gpu_cache = GpuTileCache::new(16 * 1024 * 1024);

    let utm37 = Projection::from_epsg(32637).expect("UTM 37N");
    let north_up = Arc::new(
        Frame::from_pixel_affine(
            utm37,
            [100.0, 0.0, 500_000.0, 0.0, -100.0, 5_250_000.0],
            512,
            512,
        )
        .expect("north-up"),
    );
    let south_up = Arc::new(
        Frame::from_pixel_affine(
            utm37,
            [100.0, 0.0, 500_000.0, 0.0, 100.0, 5_198_800.0],
            512,
            512,
        )
        .expect("south-up"),
    );

    let (center_lat, center_lon) = north_up.geodetic(0.5, 0.5);
    let cam = Camera::at(center_lat, center_lon, 30_000.0);
    let config = QuadtreeConfig::default();

    for (case_idx, (frame, expected_flip)) in [(&north_up, false), (&south_up, true)]
        .into_iter()
        .enumerate()
    {
        let src = SourceId::new(900 + case_idx as u64);
        let (tile_key, gpu_tile) =
            upload_overlay_tile(device, queue, &mut gpu_cache, src, 32, 32, |_, _| {
                [255, 0, 255, 255]
            });
        let segs = level_segments(frame, 512, 512, 0);
        let warp = build_warp_patch(frame, 512, 512, [0.0, 0.0, 512.0, 512.0], segs);
        assert_eq!(warp.flip_winding, expected_flip);

        let mut snap = DrawSnapshot::build(
            SnapshotBuildParams {
                camera: cam,
                width_px: width,
                height_px: height,
                pixels_per_point: 1.0,
                show_graticule: false,
                config: &config,
            },
            None,
        );
        snap.carrier_keys.insert(tile_key);
        snap.overlays.push(DrawOverlayLayer {
            opacity: 1.0,
            frame: Arc::clone(frame),
            image_width: 512,
            image_height: 512,
            segments: segs,
            patches: vec![DrawOverlayPatch {
                mesh_key: OverlayMeshKey {
                    source: src,
                    relaid: 1,
                    level: 0,
                    x: 0,
                    y: 0,
                },
                cell_rect: [0.0, 0.0, 512.0, 512.0],
                uv_rect: [0.0, 0.0, 1.0, 1.0],
                texture: Arc::clone(&gpu_tile),
            }],
        });
        let snap = Arc::new(snap);

        let pixels = render_srgb(&mut renderer, device, queue, width, height, &snap);
        let center = pixel_at(&pixels, width, width / 2, height / 2);
        assert!(
            center[0] > 240 && center[1] < 15 && center[2] > 240,
            "case {case_idx} (flip={expected_flip}): expected magenta draped overlay at center, got {center:?}"
        );

        // Zero idle GPU re-render when the same `Arc<DrawSnapshot>` is presented again.
        let _ = render_with(
            &mut renderer,
            device,
            queue,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            RenderParams {
                width,
                height,
                snapshot: &snap,
            },
            false,
        );

        // Enable graticule in Pass 2 on top of the magenta overlay.
        let mut snap_grid = (*snap).clone();
        snap_grid.show_graticule = true;
        let snap_grid = Arc::new(snap_grid);
        let pixels_grid = render_srgb(&mut renderer, device, queue, width, height, &snap_grid);
        let changed_by_graticule = pixels
            .chunks_exact(4)
            .zip(pixels_grid.chunks_exact(4))
            .filter(|(a, b)| a[1].abs_diff(b[1]) > 15)
            .count();
        assert!(
            changed_by_graticule > 10,
            "Pass 2 graticule must draw on top of draped overlay without being buried"
        );

        drop(snap);
        drop(snap_grid);
        drop(gpu_tile);
        gpu_cache.remove_source(src);
        renderer.sweep_stale_bind_groups();
        assert_eq!(renderer.cached_tile_bind_group_count(), 0);
    }
}

#[test]
fn overlay_horizon_facing_has_zero_holes_at_2m_coarse_sag_and_zero_silhouette_leak() {
    let Some((device, queue)) = shared_headless_device() else {
        return;
    };

    let (width, height) = (128u32, 128u32);
    let mut renderer = GlobeRenderer::new(device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut gpu_cache = GpuTileCache::new(16 * 1024 * 1024);
    let config = QuadtreeConfig::default();

    // 1. Regression for P0 #2: a coarse 10 deg x 10 deg single-tile overlay
    //    has ~121 m diagonal chord sag at 0.5 deg/segment. Viewing directly
    //    above a quad interior at MIN_HEIGHT_M = 2.0 m must fill 100% of the
    //    viewport with ZERO horizon-discard holes beneath the camera.
    let coarse_frame = Arc::new(
        Frame::quad([(5.0, -5.0), (5.0, 5.0), (-5.0, 5.0), (-5.0, -5.0)]).expect("coarse quad"),
    );
    let src_coarse = SourceId::new(940);
    let (k_coarse, tex_coarse) =
        upload_overlay_tile(device, queue, &mut gpu_cache, src_coarse, 32, 32, |_, _| {
            [255, 0, 255, 255]
        });
    let segs_coarse = level_segments(&coarse_frame, 512, 512, 0);
    // Offset camera slightly to land right on a sub-quad interior midpoint
    // where radial chord sag is maximized (~121 m >> 2 m altitude).
    let half_step = 10.0 / f64::from(segs_coarse) * 0.5;
    let cam_2m = Camera::at(half_step, half_step, 2.0);
    let mut snap_2m = DrawSnapshot::build(
        SnapshotBuildParams {
            camera: cam_2m,
            width_px: width,
            height_px: height,
            pixels_per_point: 1.0,
            show_graticule: false,
            config: &config,
        },
        None,
    );
    snap_2m.carrier_keys.insert(k_coarse);
    snap_2m.overlays.push(DrawOverlayLayer {
        opacity: 1.0,
        frame: Arc::clone(&coarse_frame),
        image_width: 512,
        image_height: 512,
        segments: segs_coarse,
        patches: vec![DrawOverlayPatch {
            mesh_key: OverlayMeshKey {
                source: src_coarse,
                relaid: 1,
                level: 0,
                x: 0,
                y: 0,
            },
            cell_rect: [0.0, 0.0, 512.0, 512.0],
            uv_rect: [0.0, 0.0, 1.0, 1.0],
            texture: tex_coarse,
        }],
    });
    let pixels_2m = render_srgb(
        &mut renderer,
        device,
        queue,
        width,
        height,
        &Arc::new(snap_2m),
    );
    let non_magenta_2m = pixels_2m
        .chunks_exact(4)
        .filter(|px| !(px[0] > 235 && px[1] < 20 && px[2] > 235))
        .count();
    assert_eq!(
        non_magenta_2m, 0,
        "2 m nadir view over 121 m-sag coarse overlay must have 0 horizon-discard holes"
    );

    // 2. Orbital limb-straddling overlay: compare every pixel against a
    //    globe-only baseline render so ANY leak outside the globe silhouette
    //    (not just the 4 corners) fails immediately.
    let cam_orbit = Camera::at(0.0, 0.0, 12_000_000.0);
    let base_snap = Arc::new(DrawSnapshot::build(
        SnapshotBuildParams {
            camera: cam_orbit,
            width_px: width,
            height_px: height,
            pixels_per_point: 1.0,
            show_graticule: false,
            config: &config,
        },
        None,
    ));
    let baseline = render_srgb(&mut renderer, device, queue, width, height, &base_snap);

    let limb_frame = Arc::new(
        Frame::quad([(30.0, 20.0), (30.0, 120.0), (-30.0, 120.0), (-30.0, 20.0)])
            .expect("limb quad"),
    );
    let src_limb = SourceId::new(950);
    let (k_limb, tex_limb) =
        upload_overlay_tile(device, queue, &mut gpu_cache, src_limb, 32, 32, |_, _| {
            [255, 0, 255, 255]
        });
    let segs_limb = level_segments(&limb_frame, 512, 512, 0);
    let mut limb_snap = (*base_snap).clone();
    limb_snap.carrier_keys.insert(k_limb);
    limb_snap.overlays.push(DrawOverlayLayer {
        opacity: 1.0,
        frame: limb_frame,
        image_width: 512,
        image_height: 512,
        segments: segs_limb,
        patches: vec![DrawOverlayPatch {
            mesh_key: OverlayMeshKey {
                source: src_limb,
                relaid: 2,
                level: 0,
                x: 0,
                y: 0,
            },
            cell_rect: [0.0, 0.0, 512.0, 512.0],
            uv_rect: [0.0, 0.0, 1.0, 1.0],
            texture: tex_limb,
        }],
    });
    let limb_pixels = render_srgb(
        &mut renderer,
        device,
        queue,
        width,
        height,
        &Arc::new(limb_snap),
    );

    let mut front_magenta = 0usize;
    let mut space_leaks = 0usize;
    for (base_px, limb_px) in baseline.chunks_exact(4).zip(limb_pixels.chunks_exact(4)) {
        let is_magenta = limb_px[0] > 200 && limb_px[1] < 40 && limb_px[2] > 200;
        if is_clear_background([base_px[0], base_px[1], base_px[2], base_px[3]]) {
            if is_magenta {
                space_leaks += 1;
            }
        } else if is_magenta {
            front_magenta += 1;
        }
    }
    assert!(
        front_magenta > 40,
        "front-hemisphere portion of limb-straddling overlay must render"
    );
    assert_eq!(
        space_leaks, 0,
        "limb-straddling overlay must not leak any fragment outside the globe silhouette"
    );
}

#[test]
fn overlay_compositing_respects_painter_order_opacity_nodata_and_ancestor_uv_rect() {
    let Some((device, queue)) = shared_headless_device() else {
        return;
    };

    let (width, height) = (128u32, 128u32);
    let mut renderer = GlobeRenderer::new(device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut gpu_cache = GpuTileCache::new(16 * 1024 * 1024);

    let utm37 = Projection::from_epsg(32637).expect("UTM 37N");
    let frame = Arc::new(
        Frame::from_pixel_affine(
            utm37,
            [100.0, 0.0, 500_000.0, 0.0, -100.0, 5_250_000.0],
            512,
            512,
        )
        .expect("utm37"),
    );
    let (clat, clon) = frame.geodetic(0.5, 0.5);
    let cam = Camera::at(clat, clon, 25_000.0);
    let config = QuadtreeConfig::default();

    // Bottom layer: solid green on the right half (`x >= 16`), blue on the left half (`x < 16`).
    // Using ancestor sub-rect `uv_rect = [0.5, 0.0, 0.5, 1.0]` must sample ONLY the green right half!
    let src_bot = SourceId::new(960);
    let (k_bot, tex_bot) =
        upload_overlay_tile(device, queue, &mut gpu_cache, src_bot, 32, 32, |x, _| {
            if x < 16 {
                [0, 0, 255, 255]
            } else {
                [0, 255, 0, 255]
            }
        });
    // Top layer: left half transparent nodata (`a = 0`), right half opaque red (`255, 0, 0, 255`),
    // drawn at 50% opacity.
    let src_top = SourceId::new(961);
    let (k_top, tex_top) =
        upload_overlay_tile(device, queue, &mut gpu_cache, src_top, 32, 32, |x, _| {
            if x < 16 {
                [255, 0, 0, 0]
            } else {
                [255, 0, 0, 255]
            }
        });

    let mut snap = DrawSnapshot::build(
        SnapshotBuildParams {
            camera: cam,
            width_px: width,
            height_px: height,
            pixels_per_point: 1.0,
            show_graticule: false,
            config: &config,
        },
        None,
    );
    snap.carrier_keys.insert(k_bot);
    snap.carrier_keys.insert(k_top);
    snap.overlays.push(DrawOverlayLayer {
        opacity: 1.0,
        frame: Arc::clone(&frame),
        image_width: 512,
        image_height: 512,
        segments: MIN_WARP_SEGMENTS,
        patches: vec![DrawOverlayPatch {
            mesh_key: OverlayMeshKey {
                source: src_bot,
                relaid: 1,
                level: 0,
                x: 0,
                y: 0,
            },
            cell_rect: [0.0, 0.0, 512.0, 512.0],
            uv_rect: [0.5, 0.0, 0.5, 1.0],
            texture: tex_bot,
        }],
    });
    snap.overlays.push(DrawOverlayLayer {
        opacity: 0.5,
        frame: Arc::clone(&frame),
        image_width: 512,
        image_height: 512,
        segments: MIN_WARP_SEGMENTS,
        patches: vec![DrawOverlayPatch {
            mesh_key: OverlayMeshKey {
                source: src_top,
                relaid: 2,
                level: 0,
                x: 0,
                y: 0,
            },
            cell_rect: [0.0, 0.0, 512.0, 512.0],
            uv_rect: [0.0, 0.0, 1.0, 1.0],
            texture: tex_top,
        }],
    });

    let pixels = render_srgb(&mut renderer, device, queue, width, height, &Arc::new(snap));

    // Left of center: top layer has alpha=0 (nodata), so bottom green (from uv_rect right half) shows purely.
    let left_px = pixel_at(&pixels, width, width / 2 - 20, height / 2);
    assert!(
        left_px[0] < 20 && left_px[1] > 235 && left_px[2] < 20,
        "left nodata region must reveal pure green from bottom ancestor uv_rect, got {left_px:?}"
    );

    // Right of center: 50% linear-light blend of red over green -> sRGB (~188, ~188, 0).
    let right_px = pixel_at(&pixels, width, width / 2 + 20, height / 2);
    assert!(
        right_px[0] > 160 && right_px[1] > 160 && right_px[2] < 20,
        "right 50% opacity region must linear-blend red over green (~188, ~188, 0), got {right_px:?}"
    );
}
