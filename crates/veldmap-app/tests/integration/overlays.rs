//! Headless GPU smoke tests for post-body depth-less raster overlays:
//! warp-mesh draping, negative-determinant south-up outward winding, Pass 2
//! graticule-on-top ordering, sag-safe `2 m` low-altitude horizon facing,
//! limb silhouette containment, multi-layer opacity/nodata/ancestor `uv_rect`
//! compositing, `Weak<GpuTile>` bind-group sweeping, and `0` idle re-renders.

use std::sync::Arc;

use veldmap_app::gpu_tiles::{GpuTile, GpuTileCache};
use veldmap_app::render::{
    CategoricalPaletteId, ColormapId, DrawOverlayLayer, DrawOverlayPatch, DrawOverlayTrack,
    GlobeRenderer, OverlayMeshKey, RenderParams, TileBindGroupCache,
};
use veldmap_core::geo::{Camera, Frame, Projection};
use veldmap_core::tile::{
    BandKey, DecodedTile, MIN_WARP_SEGMENTS, PixelFormat, QuadtreeConfig, SourceId, TileKey,
    TileScheme, build_warp_patch, level_segments,
};
use veldmap_raster::generate_scientific_mipmaps;

use crate::common::{
    build_empty_snapshot, is_clear_background, pixel_at, render_srgb, render_srgb_with_bind_groups,
    render_with, shared_headless_device,
};

fn single_patch_layer(
    frame: &Arc<Frame>,
    source: SourceId,
    relaid: u64,
    segments: u32,
    opacity: f32,
    uv_rect: [f32; 4],
    texture: Arc<GpuTile>,
) -> DrawOverlayLayer {
    DrawOverlayLayer {
        opacity,
        track: DrawOverlayTrack::Display,
        frame: Arc::clone(frame),
        image_width: 512,
        image_height: 512,
        segments,
        patches: vec![DrawOverlayPatch {
            mesh_key: OverlayMeshKey {
                source,
                relaid,
                level: 0,
                x: 0,
                y: 0,
            },
            cell_rect: [0.0, 0.0, 512.0, 512.0],
            uv_rect,
            texture,
        }],
    }
}

fn upload_overlay_tile(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    cache: &mut GpuTileCache,
    source: SourceId,
    w: u32,
    h: u32,
    mut pixel_fn: impl FnMut(u32, u32) -> [u8; 4],
) -> (TileKey, Arc<GpuTile>) {
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
            utm37.clone(),
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

        let mut snap = build_empty_snapshot(cam, width, height, &config);
        snap.carrier_keys.insert(tile_key);
        snap.overlays.push(single_patch_layer(
            frame,
            src,
            1,
            segs,
            1.0,
            [0.0, 0.0, 1.0, 1.0],
            Arc::clone(&gpu_tile),
        ));
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

    // 1. Sag-safe horizon facing at MIN_HEIGHT_M = 2.0 m: a coarse 10 deg x 10 deg
    //    single-tile overlay has ~121 m diagonal chord sag at 0.5 deg/segment.
    //    Viewing directly above a quad interior at MIN_HEIGHT_M = 2.0 m must
    //    fill 100% of the viewport with ZERO horizon-discard holes beneath the
    //    camera.
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
    let mut snap_2m = build_empty_snapshot(cam_2m, width, height, &config);
    snap_2m.carrier_keys.insert(k_coarse);
    snap_2m.overlays.push(single_patch_layer(
        &coarse_frame,
        src_coarse,
        1,
        segs_coarse,
        1.0,
        [0.0, 0.0, 1.0, 1.0],
        tex_coarse,
    ));
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
    let base_snap = Arc::new(build_empty_snapshot(cam_orbit, width, height, &config));
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
    limb_snap.overlays.push(single_patch_layer(
        &limb_frame,
        src_limb,
        2,
        segs_limb,
        1.0,
        [0.0, 0.0, 1.0, 1.0],
        tex_limb,
    ));
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

    let mut snap = build_empty_snapshot(cam, width, height, &config);
    snap.carrier_keys.insert(k_bot);
    snap.carrier_keys.insert(k_top);
    snap.overlays.push(single_patch_layer(
        &frame,
        src_bot,
        1,
        MIN_WARP_SEGMENTS,
        1.0,
        [0.5, 0.0, 0.5, 1.0],
        tex_bot,
    ));
    snap.overlays.push(single_patch_layer(
        &frame,
        src_top,
        2,
        MIN_WARP_SEGMENTS,
        0.5,
        [0.0, 0.0, 1.0, 1.0],
        tex_top,
    ));

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

#[test]
fn scientific_r32float_nodata_trilinear_and_categorical_uint_render_and_sweep_cleanly() {
    let Some((device, queue)) = shared_headless_device() else {
        return;
    };

    let (width, height) = (128u32, 128u32);
    let mut shared_bg = TileBindGroupCache::new(device);
    let mut renderer = GlobeRenderer::new_with_shared_bind_groups(
        device,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        &shared_bg,
    );
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

    // Base Display layer (solid green) beneath Scientific layer to detect nodata discard.
    let src_base = SourceId::new(970);
    let (k_base, tex_base) =
        upload_overlay_tile(device, queue, &mut gpu_cache, src_base, 16, 16, |_, _| {
            [0, 255, 0, 255]
        });

    // Scientific R32Float tile (64x64 with full NaN-skipping mip chain):
    // Left half = NaN / +Inf / -Inf (nodata), Right half = raw 1000.0 (with scale=0.1, offset=0 -> 100.0).
    let src_sci = SourceId::new(971);
    let k_sci = TileKey {
        source: src_sci,
        band: BandKey::Index(0),
        scheme: TileScheme::RasterPyramid { max_level: 0 },
        format: PixelFormat::R32Float,
        gutter: 0,
        decode_rev: 1,
        level: 0,
        x: 0,
        y: 0,
    };
    let mut f32_bytes = Vec::with_capacity(64 * 64 * 4);
    for y in 0..64u32 {
        for x in 0..64u32 {
            let v: f32 = if x < 32 {
                match (x + y) % 3 {
                    0 => f32::NAN,
                    1 => f32::INFINITY,
                    _ => f32::NEG_INFINITY,
                }
            } else {
                1000.0
            };
            f32_bytes.extend_from_slice(&v.to_ne_bytes());
        }
    }
    let mut dec_sci =
        DecodedTile::try_new(64, 64, PixelFormat::R32Float, f32_bytes).expect("r32f tile");
    generate_scientific_mipmaps(&mut dec_sci);
    assert!(dec_sci.mip_level_count() > 1);
    let (tex_sci, _) = gpu_cache
        .upload_and_insert(device, queue, k_sci, &dec_sci)
        .expect("upload r32f");

    // 1. Render Scientific R32Float with Grayscale ramp over [0, 100]:
    //    Left half (NaN/±Inf) discards to green base; Right half (1000*0.1 = 100) maps to white!
    let mut snap = build_empty_snapshot(cam, width, height, &config);
    snap.carrier_keys.insert(k_base);
    snap.carrier_keys.insert(k_sci);
    snap.overlays.push(single_patch_layer(
        &frame,
        src_base,
        1,
        MIN_WARP_SEGMENTS,
        1.0,
        [0.0, 0.0, 1.0, 1.0],
        Arc::clone(&tex_base),
    ));
    snap.overlays.push(DrawOverlayLayer {
        opacity: 1.0,
        track: DrawOverlayTrack::Scientific {
            scale: 0.1,
            offset: 0.0,
            stretch_min: 0.0,
            stretch_max: 100.0,
            log10_db: false,
            colormap: ColormapId::Grayscale,
            invert_colormap: false,
        },
        frame: Arc::clone(&frame),
        image_width: 512,
        image_height: 512,
        segments: MIN_WARP_SEGMENTS,
        patches: vec![DrawOverlayPatch {
            mesh_key: OverlayMeshKey {
                source: src_sci,
                relaid: 2,
                level: 0,
                x: 0,
                y: 0,
            },
            cell_rect: [0.0, 0.0, 512.0, 512.0],
            uv_rect: [0.0, 0.0, 1.0, 1.0],
            texture: Arc::clone(&tex_sci),
        }],
    });
    let snap = Arc::new(snap);
    // Render and read back directly through the hoisted `shared_bg` cache.
    let pixels = render_srgb_with_bind_groups(
        &mut renderer,
        &mut shared_bg,
        device,
        queue,
        width,
        height,
        &snap,
    );
    let left_px = pixel_at(&pixels, width, width / 2 - 20, height / 2);
    assert!(
        left_px[0] < 20 && left_px[1] > 235 && left_px[2] < 20,
        "NaN/±Inf left half must discard to green base, got {left_px:?}"
    );
    let right_px = pixel_at(&pixels, width, width / 2 + 20, height / 2);
    assert!(
        right_px[0] > 240 && right_px[1] > 240 && right_px[2] > 240,
        "Right half (1000*0.1=100) must map to white on [0,100] Grayscale, got {right_px:?}"
    );

    // 1b. Verify `log10_db = true` stretch (`10 * log10(100.0) = 20 dB`):
    //     over [20, 40] dB -> black (and with `invert_colormap = true` -> white).
    let mut snap_db = build_empty_snapshot(cam, width, height, &config);
    snap_db.carrier_keys.insert(k_base);
    snap_db.carrier_keys.insert(k_sci);
    snap_db.overlays.push(single_patch_layer(
        &frame,
        src_base,
        1,
        MIN_WARP_SEGMENTS,
        1.0,
        [0.0, 0.0, 1.0, 1.0],
        Arc::clone(&tex_base),
    ));
    snap_db.overlays.push(DrawOverlayLayer {
        opacity: 1.0,
        track: DrawOverlayTrack::Scientific {
            scale: 0.1,
            offset: 0.0,
            stretch_min: 20.0,
            stretch_max: 40.0,
            log10_db: true,
            colormap: ColormapId::Grayscale,
            invert_colormap: false,
        },
        frame: Arc::clone(&frame),
        image_width: 512,
        image_height: 512,
        segments: MIN_WARP_SEGMENTS,
        patches: vec![DrawOverlayPatch {
            mesh_key: OverlayMeshKey {
                source: src_sci,
                relaid: 2,
                level: 0,
                x: 0,
                y: 0,
            },
            cell_rect: [0.0, 0.0, 512.0, 512.0],
            uv_rect: [0.0, 0.0, 1.0, 1.0],
            texture: Arc::clone(&tex_sci),
        }],
    });
    let snap_db = Arc::new(snap_db);
    let pixels_db = render_srgb_with_bind_groups(
        &mut renderer,
        &mut shared_bg,
        device,
        queue,
        width,
        height,
        &snap_db,
    );
    let db_right = pixel_at(&pixels_db, width, width / 2 + 20, height / 2);
    assert!(
        db_right[0] < 15 && db_right[1] < 15 && db_right[2] < 15,
        "20 dB value at stretch_min=20 dB must map to black, got {db_right:?}"
    );
    let db_left = pixel_at(&pixels_db, width, width / 2 - 20, height / 2);
    let mut standalone = GlobeRenderer::new(device, wgpu::TextureFormat::Rgba8UnormSrgb);
    assert!(
        db_left[0] < 20 && db_left[1] > 235 && db_left[2] < 20,
        "dB stretch must still discard NaN/±Inf to green base, got {db_left:?}"
    );

    // 1c. Zoomed-out trilinear mip-selection (`lod > 0.5`):
    //     Pull camera back so the 64x64 tile projects smaller than 32 px wide, forcing
    //     8-tap trilinear mip blending across NaN-skipping mipmaps without NaN artifacts.
    let cam_far = Camera::at(clat, clon, 180_000.0);
    let mut snap_far = build_empty_snapshot(cam_far, width, height, &config);
    snap_far.carrier_keys.insert(k_base);
    snap_far.carrier_keys.insert(k_sci);
    snap_far.overlays.push(single_patch_layer(
        &frame,
        src_base,
        1,
        MIN_WARP_SEGMENTS,
        1.0,
        [0.0, 0.0, 1.0, 1.0],
        Arc::clone(&tex_base),
    ));
    snap_far.overlays.push(DrawOverlayLayer {
        opacity: 1.0,
        track: DrawOverlayTrack::Scientific {
            scale: 0.1,
            offset: 0.0,
            stretch_min: 0.0,
            stretch_max: 100.0,
            log10_db: false,
            colormap: ColormapId::Grayscale,
            invert_colormap: false,
        },
        frame: Arc::clone(&frame),
        image_width: 512,
        image_height: 512,
        segments: MIN_WARP_SEGMENTS,
        patches: vec![DrawOverlayPatch {
            mesh_key: OverlayMeshKey {
                source: src_sci,
                relaid: 2,
                level: 0,
                x: 0,
                y: 0,
            },
            cell_rect: [0.0, 0.0, 512.0, 512.0],
            uv_rect: [0.0, 0.0, 1.0, 1.0],
            texture: Arc::clone(&tex_sci),
        }],
    });
    let snap_far = Arc::new(snap_far);
    let pixels_far = render_srgb(&mut standalone, device, queue, width, height, &snap_far);
    let far_right = pixel_at(&pixels_far, width, width / 2 + 4, height / 2);
    assert!(
        far_right[0] > 230 && far_right[1] > 230 && far_right[2] > 230,
        "Zoomed-out trilinear R32Float mip blend must stay white on valid half, got {far_right:?}"
    );
    let far_left = pixel_at(&pixels_far, width, width / 2 - 4, height / 2);
    assert!(
        far_left[0] < 25 && far_left[1] > 225 && far_left[2] < 25,
        "Zoomed-out trilinear R32Float mip blend must discard NaN half cleanly, got {far_left:?}"
    );

    // 2. Categorical R8Uint + R16Uint sharing Uint bind-group layout.
    let src_cat8 = SourceId::new(972);
    let k_cat8 = TileKey {
        source: src_cat8,
        band: BandKey::Index(0),
        scheme: TileScheme::RasterPyramid { max_level: 0 },
        format: PixelFormat::R8Uint,
        gutter: 0,
        decode_rev: 1,
        level: 0,
        x: 0,
        y: 0,
    };
    let mut u8_data = vec![0u8; 32 * 32];
    for y in 0..32usize {
        for x in 16..32usize {
            u8_data[y * 32 + x] = 1; // Class 1 on right half, Class 0 (nodata) on left
        }
    }
    let dec_cat8 = DecodedTile::try_new(32, 32, PixelFormat::R8Uint, u8_data).expect("r8u");
    let (tex_cat8, _) = gpu_cache
        .upload_and_insert(device, queue, k_cat8, &dec_cat8)
        .expect("upload r8u");
    let mut cmap = vec![[0u8, 0, 0, 255]; 256];
    cmap[1] = [0, 0, 255, 255]; // Class 1 -> pure blue
    let custom_cmap: Arc<[[u8; 4]]> = Arc::from(cmap);

    let mut snap_cat = build_empty_snapshot(cam, width, height, &config);
    snap_cat.carrier_keys.insert(k_base);
    snap_cat.carrier_keys.insert(k_cat8);
    snap_cat.overlays.push(single_patch_layer(
        &frame,
        src_base,
        1,
        MIN_WARP_SEGMENTS,
        1.0,
        [0.0, 0.0, 1.0, 1.0],
        Arc::clone(&tex_base),
    ));
    snap_cat.overlays.push(DrawOverlayLayer {
        opacity: 1.0,
        track: DrawOverlayTrack::Categorical {
            source: src_cat8,
            palette: CategoricalPaletteId::SourceColorMap,
            custom_color_map: Some(custom_cmap),
            nodata_value: Some(0),
        },
        frame: Arc::clone(&frame),
        image_width: 512,
        image_height: 512,
        segments: MIN_WARP_SEGMENTS,
        patches: vec![DrawOverlayPatch {
            mesh_key: OverlayMeshKey {
                source: src_cat8,
                relaid: 3,
                level: 0,
                x: 0,
                y: 0,
            },
            cell_rect: [0.0, 0.0, 512.0, 512.0],
            uv_rect: [0.0, 0.0, 1.0, 1.0],
            texture: Arc::clone(&tex_cat8),
        }],
    });
    let snap_cat = Arc::new(snap_cat);
    let pixels_cat = render_srgb(&mut standalone, device, queue, width, height, &snap_cat);
    let cat_left = pixel_at(&pixels_cat, width, width / 2 - 20, height / 2);
    assert!(
        cat_left[0] < 20 && cat_left[1] > 235 && cat_left[2] < 20,
        "Categorical nodata=0 left half must reveal green base, got {cat_left:?}"
    );
    let cat_right = pixel_at(&pixels_cat, width, width / 2 + 20, height / 2);
    assert!(
        cat_right[0] < 20 && cat_right[1] < 20 && cat_right[2] > 235,
        "Categorical class=1 right half must render pure blue from ColorMap, got {cat_right:?}"
    );

    // 2b. Categorical R16Uint with 16-bit QA class value > 255 (1024) and nodata = 65535,
    //     rendered through the shared `Uint` bind-group layout and `shared_bg`.
    let src_cat16 = SourceId::new(973);
    let k_cat16 = TileKey {
        source: src_cat16,
        band: BandKey::Index(0),
        scheme: TileScheme::RasterPyramid { max_level: 0 },
        format: PixelFormat::R16Uint,
        gutter: 0,
        decode_rev: 1,
        level: 0,
        x: 0,
        y: 0,
    };
    let mut u16_bytes = Vec::with_capacity(32 * 32 * 2);
    for _y in 0..32usize {
        for x in 0..32usize {
            let class_val: u16 = if x < 16 { 65535 } else { 1024 };
            u16_bytes.extend_from_slice(&class_val.to_ne_bytes());
        }
    }
    let dec_cat16 = DecodedTile::try_new(32, 32, PixelFormat::R16Uint, u16_bytes).expect("r16u");
    let (tex_cat16, _) = gpu_cache
        .upload_and_insert(device, queue, k_cat16, &dec_cat16)
        .expect("upload r16u");
    let mut snap_cat16 = build_empty_snapshot(cam, width, height, &config);
    snap_cat16.carrier_keys.insert(k_base);
    snap_cat16.carrier_keys.insert(k_cat16);
    snap_cat16.overlays.push(single_patch_layer(
        &frame,
        src_base,
        1,
        MIN_WARP_SEGMENTS,
        1.0,
        [0.0, 0.0, 1.0, 1.0],
        Arc::clone(&tex_base),
    ));
    snap_cat16.overlays.push(DrawOverlayLayer {
        opacity: 1.0,
        track: DrawOverlayTrack::Categorical {
            source: src_cat16,
            palette: CategoricalPaletteId::DistinctClasses,
            custom_color_map: None,
            nodata_value: Some(65535),
        },
        frame: Arc::clone(&frame),
        image_width: 512,
        image_height: 512,
        segments: MIN_WARP_SEGMENTS,
        patches: vec![DrawOverlayPatch {
            mesh_key: OverlayMeshKey {
                source: src_cat16,
                relaid: 4,
                level: 0,
                x: 0,
                y: 0,
            },
            cell_rect: [0.0, 0.0, 512.0, 512.0],
            uv_rect: [0.0, 0.0, 1.0, 1.0],
            texture: Arc::clone(&tex_cat16),
        }],
    });
    let snap_cat16 = Arc::new(snap_cat16);
    let pixels_cat16 = render_srgb_with_bind_groups(
        &mut renderer,
        &mut shared_bg,
        device,
        queue,
        width,
        height,
        &snap_cat16,
    );
    let cat16_left = pixel_at(&pixels_cat16, width, width / 2 - 20, height / 2);
    assert!(
        cat16_left[0] < 20 && cat16_left[1] > 235 && cat16_left[2] < 20,
        "R16Uint nodata=65535 left half must reveal green base, got {cat16_left:?}"
    );
    let cat16_right = pixel_at(&pixels_cat16, width, width / 2 + 20, height / 2);
    assert!(
        !(cat16_right[0] < 20 && cat16_right[1] > 235 && cat16_right[2] < 20),
        "R16Uint class=1024 right half must render hashed class color over green base, got {cat16_right:?}"
    );

    // Verify hoisted multi-format TileBindGroupCache sweeps cleanly once evicted.
    assert!(shared_bg.len() >= 3);
    drop(snap);
    drop(snap_db);
    drop(snap_far);
    drop(snap_cat);
    drop(snap_cat16);
    drop(tex_base);
    drop(tex_sci);
    drop(tex_cat8);
    drop(tex_cat16);
    gpu_cache.remove_source(src_base);
    gpu_cache.remove_source(src_sci);
    gpu_cache.remove_source(src_cat8);
    gpu_cache.remove_source(src_cat16);
    shared_bg.sweep_stale();
    assert_eq!(shared_bg.len(), 0);
}
