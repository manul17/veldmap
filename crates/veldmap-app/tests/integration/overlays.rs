//! Headless GPU smoke tests for post-body depth-less raster overlays:
//! warp-mesh draping, negative-determinant south-up outward winding, Pass 2
//! graticule-on-top ordering, sag-safe `2 m` low-altitude horizon facing,
//! limb silhouette containment, multi-layer opacity/nodata/ancestor `uv_rect`
//! compositing, `Weak<GpuTile>` bind-group sweeping, and `0` idle re-renders.

use std::sync::Arc;

use veldmap_app::gpu_tiles::{GpuTile, GpuTileCache};
use veldmap_app::render::{
    BandMathFormula, CategoricalPaletteId, ColormapId, DrawBandEpoch, DrawBandMathLayer,
    DrawDisplayEpoch, DrawOverlayContent, DrawOverlayLayer, DrawOverlayPatch, DrawOverlayTrack,
    DrawScratchBand, DrawScratchMask, DrawTemporalCrossFadeLayer, DrawTemporalCrossFadeSource,
    DrawTemporalDiffLayer, OverlayMeshKey, RenderParams, compute_band_math_scissor_rect,
};
use veldmap_core::geo::{Camera, Frame, Projection};
use veldmap_core::tile::{
    BandKey, DecodedTile, MIN_WARP_SEGMENTS, PixelFormat, QuadtreeConfig, SourceId, TileKey,
    TileScheme, build_warp_patch, level_segments,
};
use veldmap_raster::generate_tile_mipmaps;

use crate::common::{
    TestGlobe, build_empty_snapshot, is_clear_background, pixel_at, render_srgb, render_with,
    shared_headless_device,
};

fn single_patch_layer(
    frame: &Arc<Frame>,
    source: SourceId,
    relaid: u64,
    segments: u32,
    track: DrawOverlayTrack,
    texture: Arc<GpuTile>,
) -> DrawOverlayLayer {
    DrawOverlayLayer {
        layer_id: relaid,
        opacity: 1.0,
        content: DrawOverlayContent::Direct {
            track,
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
                uv_rect: [0.0, 0.0, 1.0, 1.0],
                texture,
            }],
        },
    }
}

fn utm37_test_frame() -> Arc<Frame> {
    let utm37 = Projection::from_epsg(32637).expect("UTM 37N");
    Arc::new(
        Frame::from_pixel_affine(
            utm37,
            [100.0, 0.0, 500_000.0, 0.0, -100.0, 5_250_000.0],
            512,
            512,
        )
        .expect("utm37"),
    )
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
    let mut renderer = TestGlobe::new_srgb(device);
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
        let (_, gpu_tile) =
            upload_overlay_tile(device, queue, &mut gpu_cache, src, 32, 32, |_, _| {
                [255, 0, 255, 255]
            });
        let segs = level_segments(frame, 512, 512, 0);
        let warp = build_warp_patch(frame, 512, 512, [0.0, 0.0, 512.0, 512.0], segs);
        assert_eq!(warp.flip_winding, expected_flip);

        let mut snap = build_empty_snapshot(cam, width, height, &config);
        snap.overlays.push(single_patch_layer(
            frame,
            src,
            1,
            segs,
            DrawOverlayTrack::Display,
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
                swipe: None,
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
        renderer.bind_groups.sweep_stale();
        assert_eq!(renderer.bind_groups.len(), 0);
    }
}

#[test]
fn overlay_horizon_facing_has_zero_holes_at_2m_coarse_sag_and_zero_silhouette_leak() {
    let Some((device, queue)) = shared_headless_device() else {
        return;
    };

    let (width, height) = (128u32, 128u32);
    let mut renderer = TestGlobe::new_srgb(device);
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
    let (_, tex_coarse) =
        upload_overlay_tile(device, queue, &mut gpu_cache, src_coarse, 32, 32, |_, _| {
            [255, 0, 255, 255]
        });
    let segs_coarse = level_segments(&coarse_frame, 512, 512, 0);
    // Offset camera slightly to land right on a sub-quad interior midpoint
    // where radial chord sag is maximized (~121 m >> 2 m altitude).
    let half_step = 10.0 / f64::from(segs_coarse) * 0.5;
    let cam_2m = Camera::at(half_step, half_step, 2.0);
    let mut snap_2m = build_empty_snapshot(cam_2m, width, height, &config);
    snap_2m.overlays.push(single_patch_layer(
        &coarse_frame,
        src_coarse,
        1,
        segs_coarse,
        DrawOverlayTrack::Display,
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
    let (_, tex_limb) =
        upload_overlay_tile(device, queue, &mut gpu_cache, src_limb, 32, 32, |_, _| {
            [255, 0, 255, 255]
        });
    let segs_limb = level_segments(&limb_frame, 512, 512, 0);
    let mut limb_snap = (*base_snap).clone();
    limb_snap.overlays.push(single_patch_layer(
        &limb_frame,
        src_limb,
        2,
        segs_limb,
        DrawOverlayTrack::Display,
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
    let mut renderer = TestGlobe::new_srgb(device);
    let mut gpu_cache = GpuTileCache::new(16 * 1024 * 1024);

    let frame = utm37_test_frame();
    let (clat, clon) = frame.geodetic(0.5, 0.5);
    let cam = Camera::at(clat, clon, 25_000.0);
    let config = QuadtreeConfig::default();

    // Bottom layer: solid green on the right half (`x >= 16`), blue on the left half (`x < 16`).
    // Using ancestor sub-rect `uv_rect = [0.5, 0.0, 0.5, 1.0]` must sample ONLY the green right half!
    let src_bot = SourceId::new(960);
    let (_, tex_bot) =
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
    let (_, tex_top) =
        upload_overlay_tile(device, queue, &mut gpu_cache, src_top, 32, 32, |x, _| {
            if x < 16 {
                [255, 0, 0, 0]
            } else {
                [255, 0, 0, 255]
            }
        });

    let mut snap = build_empty_snapshot(cam, width, height, &config);
    let mut bot_layer = single_patch_layer(
        &frame,
        src_bot,
        1,
        MIN_WARP_SEGMENTS,
        DrawOverlayTrack::Display,
        tex_bot,
    );
    if let DrawOverlayContent::Direct { patches, .. } = &mut bot_layer.content {
        patches[0].uv_rect = [0.5, 0.0, 0.5, 1.0];
    }
    snap.overlays.push(bot_layer);

    let mut top_layer = single_patch_layer(
        &frame,
        src_top,
        2,
        MIN_WARP_SEGMENTS,
        DrawOverlayTrack::Display,
        tex_top,
    );
    top_layer.opacity = 0.5;
    snap.overlays.push(top_layer);

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
fn scientific_r32float_nodata_linear_db_and_trilinear_render_and_sweep_cleanly() {
    let Some((device, queue)) = shared_headless_device() else {
        return;
    };

    let (width, height) = (128u32, 128u32);
    let mut globe = TestGlobe::new_srgb(device);
    let mut gpu_cache = GpuTileCache::new(16 * 1024 * 1024);

    let frame = utm37_test_frame();
    let (clat, clon) = frame.geodetic(0.5, 0.5);
    let cam = Camera::at(clat, clon, 25_000.0);
    let config = QuadtreeConfig::default();

    // Base Display layer (solid green) beneath Scientific layer to detect nodata discard.
    let src_base = SourceId::new(970);
    let (_, tex_base) =
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
    generate_tile_mipmaps(&mut dec_sci);
    assert!(dec_sci.mip_level_count() > 1);
    let (tex_sci, _) = gpu_cache
        .upload_and_insert(device, queue, k_sci, &dec_sci)
        .expect("upload r32f");

    let build_sci_snap = |camera: Camera, track: DrawOverlayTrack| {
        let mut snap = build_empty_snapshot(camera, width, height, &config);
        snap.overlays.push(single_patch_layer(
            &frame,
            src_base,
            1,
            MIN_WARP_SEGMENTS,
            DrawOverlayTrack::Display,
            Arc::clone(&tex_base),
        ));
        snap.overlays.push(single_patch_layer(
            &frame,
            src_sci,
            2,
            MIN_WARP_SEGMENTS,
            track,
            Arc::clone(&tex_sci),
        ));
        Arc::new(snap)
    };

    // 1. Render Scientific R32Float with Grayscale ramp over [0, 100]:
    //    Left half (NaN/±Inf) discards to green base; Right half (1000*0.1 = 100) maps to white!
    let linear_track = DrawOverlayTrack::Scientific {
        scale: 0.1,
        offset: 0.0,
        stretch_min: 0.0,
        stretch_max: 100.0,
        log10_db: false,
        colormap: ColormapId::Grayscale,
        invert_colormap: false,
    };
    let snap = build_sci_snap(cam, linear_track.clone());
    let pixels = render_srgb(&mut globe, device, queue, width, height, &snap);
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

    // 2. Verify `log10_db = true` stretch (`10 * log10(100.0) = 20 dB`):
    //    over [20, 40] dB -> black (and NaN/±Inf still discards to green base).
    let snap_db = build_sci_snap(
        cam,
        DrawOverlayTrack::Scientific {
            scale: 0.1,
            offset: 0.0,
            stretch_min: 20.0,
            stretch_max: 40.0,
            log10_db: true,
            colormap: ColormapId::Grayscale,
            invert_colormap: false,
        },
    );
    let pixels_db = render_srgb(&mut globe, device, queue, width, height, &snap_db);
    let db_right = pixel_at(&pixels_db, width, width / 2 + 20, height / 2);
    assert!(
        db_right[0] < 15 && db_right[1] < 15 && db_right[2] < 15,
        "20 dB value at stretch_min=20 dB must map to black, got {db_right:?}"
    );
    let db_left = pixel_at(&pixels_db, width, width / 2 - 20, height / 2);
    assert!(
        db_left[0] < 20 && db_left[1] > 235 && db_left[2] < 20,
        "dB stretch must still discard NaN/±Inf to green base, got {db_left:?}"
    );

    // 3. Zoomed-out trilinear mip-selection (`lod > 0.5`):
    //    Pull camera back so the 64x64 tile projects smaller than 32 px wide, forcing
    //    8-tap trilinear mip blending across NaN-skipping mipmaps without NaN artifacts.
    let cam_far = Camera::at(clat, clon, 180_000.0);
    let snap_far = build_sci_snap(cam_far, linear_track);
    let pixels_far = render_srgb(&mut globe, device, queue, width, height, &snap_far);
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

    assert!(globe.bind_groups.len() >= 2);
    drop(snap);
    drop(snap_db);
    drop(snap_far);
    drop(tex_base);
    drop(tex_sci);
    gpu_cache.remove_source(src_base);
    gpu_cache.remove_source(src_sci);
    globe.bind_groups.sweep_stale();
    assert_eq!(globe.bind_groups.len(), 0);
}

#[test]
fn categorical_r8uint_and_r16uint_render_and_sweep_cleanly() {
    let Some((device, queue)) = shared_headless_device() else {
        return;
    };

    let (width, height) = (128u32, 128u32);
    let mut globe = TestGlobe::new_srgb(device);
    let mut gpu_cache = GpuTileCache::new(16 * 1024 * 1024);

    let frame = utm37_test_frame();
    let (clat, clon) = frame.geodetic(0.5, 0.5);
    let cam = Camera::at(clat, clon, 25_000.0);
    let config = QuadtreeConfig::default();

    // Base Display layer (solid green) beneath Categorical layers to detect nodata discard.
    let src_base = SourceId::new(974);
    let (_, tex_base) =
        upload_overlay_tile(device, queue, &mut gpu_cache, src_base, 16, 16, |_, _| {
            [0, 255, 0, 255]
        });

    // 1. Categorical R8Uint with source ColorMap (Class 1 -> blue, Class 0 -> nodata).
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
    snap_cat.overlays.push(single_patch_layer(
        &frame,
        src_base,
        1,
        MIN_WARP_SEGMENTS,
        DrawOverlayTrack::Display,
        Arc::clone(&tex_base),
    ));
    snap_cat.overlays.push(single_patch_layer(
        &frame,
        src_cat8,
        3,
        MIN_WARP_SEGMENTS,
        DrawOverlayTrack::Categorical {
            source: src_cat8,
            palette: CategoricalPaletteId::SourceColorMap,
            custom_color_map: Some(custom_cmap),
            nodata_value: Some(0),
        },
        Arc::clone(&tex_cat8),
    ));
    let snap_cat = Arc::new(snap_cat);
    let pixels_cat = render_srgb(&mut globe, device, queue, width, height, &snap_cat);
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

    // 2. Categorical R16Uint with 16-bit QA class value > 255 (1024) and nodata = 65535,
    //    rendered through the shared `Uint` bind-group layout and `shared_bg`.
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
    snap_cat16.overlays.push(single_patch_layer(
        &frame,
        src_base,
        1,
        MIN_WARP_SEGMENTS,
        DrawOverlayTrack::Display,
        Arc::clone(&tex_base),
    ));
    snap_cat16.overlays.push(single_patch_layer(
        &frame,
        src_cat16,
        4,
        MIN_WARP_SEGMENTS,
        DrawOverlayTrack::Categorical {
            source: src_cat16,
            palette: CategoricalPaletteId::DistinctClasses,
            custom_color_map: None,
            nodata_value: Some(65535),
        },
        Arc::clone(&tex_cat16),
    ));
    let snap_cat16 = Arc::new(snap_cat16);
    let pixels_cat16 = render_srgb(&mut globe, device, queue, width, height, &snap_cat16);
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
    assert!(globe.bind_groups.len() >= 3);
    drop(snap_cat);
    drop(snap_cat16);
    drop(tex_base);
    drop(tex_cat8);
    drop(tex_cat16);
    gpu_cache.remove_source(src_base);
    gpu_cache.remove_source(src_cat8);
    gpu_cache.remove_source(src_cat16);
    globe.bind_groups.sweep_stale();
    assert_eq!(globe.bind_groups.len(), 0);
}

#[expect(
    clippy::too_many_arguments,
    reason = "Concise headless GPU test helper for uploading synthetic R32Float tiles."
)]
fn upload_r32f_tile(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    cache: &mut GpuTileCache,
    source: SourceId,
    band: BandKey,
    w: u32,
    h: u32,
    mut val_fn: impl FnMut(u32, u32) -> f32,
) -> (TileKey, Arc<GpuTile>) {
    let key = TileKey {
        source,
        band,
        scheme: TileScheme::RasterPyramid { max_level: 0 },
        format: PixelFormat::R32Float,
        gutter: 0,
        decode_rev: 1,
        level: 0,
        x: 0,
        y: 0,
    };
    let mut bytes = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            bytes.extend_from_slice(&val_fn(x, y).to_ne_bytes());
        }
    }
    let mut decoded = DecodedTile::try_new(w, h, PixelFormat::R32Float, bytes).expect("r32f");
    generate_tile_mipmaps(&mut decoded);
    let (gpu_tile, _) = cache
        .upload_and_insert(device, queue, key, &decoded)
        .expect("upload r32f");
    (key, gpu_tile)
}

fn upload_r8u_tile(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    cache: &mut GpuTileCache,
    source: SourceId,
    w: u32,
    h: u32,
    mut class_fn: impl FnMut(u32, u32) -> u8,
) -> (TileKey, Arc<GpuTile>) {
    let key = TileKey {
        source,
        band: BandKey::Index(0),
        scheme: TileScheme::RasterPyramid { max_level: 0 },
        format: PixelFormat::R8Uint,
        gutter: 0,
        decode_rev: 1,
        level: 0,
        x: 0,
        y: 0,
    };
    let mut bytes = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            bytes.push(class_fn(x, y));
        }
    }
    let decoded = DecodedTile::try_new(w, h, PixelFormat::R8Uint, bytes).expect("r8u");
    let (gpu_tile, _) = cache
        .upload_and_insert(device, queue, key, &decoded)
        .expect("upload r8u");
    (key, gpu_tile)
}

#[expect(
    clippy::too_many_arguments,
    reason = "Concise headless GPU test helper for building DrawScratchBand fixtures."
)]
fn scratch_band(
    frame: &Arc<Frame>,
    source: SourceId,
    relaid: u64,
    img_w: u32,
    img_h: u32,
    scale: f32,
    offset: f32,
    texture: Arc<GpuTile>,
) -> DrawScratchBand {
    DrawScratchBand {
        frame: Arc::clone(frame),
        image_width: img_w,
        image_height: img_h,
        segments: MIN_WARP_SEGMENTS,
        scale,
        offset,
        patches: vec![DrawOverlayPatch {
            mesh_key: OverlayMeshKey {
                source,
                relaid,
                level: 0,
                x: 0,
                y: 0,
            },
            cell_rect: [0.0, 0.0, f64::from(img_w), f64::from(img_h)],
            uv_rect: [0.0, 0.0, 1.0, 1.0],
            texture,
        }],
    }
}

#[test]
fn band_math_nan_fill_preserves_zero_reflectance_and_scl_masking() {
    let Some((device, queue)) = shared_headless_device() else {
        return;
    };

    let (width, height) = (128u32, 128u32);
    let mut globe = TestGlobe::new_srgb(device);
    let mut gpu_cache = GpuTileCache::new(16 * 1024 * 1024);

    let frame = utm37_test_frame();
    let (clat, clon) = frame.geodetic(0.5, 0.5);
    let cam = Camera::at(clat, clon, 25_000.0);
    let config = QuadtreeConfig::default();

    // Solid blue base layer beneath BandMath so discarded/masked pixels show pure blue.
    let src_base = SourceId::new(980);
    let (_, tex_base) =
        upload_overlay_tile(device, queue, &mut gpu_cache, src_base, 16, 16, |_, _| {
            [0, 0, 255, 255]
        });

    // Quadrants (32x32):
    //   Top-left     (x<16, y<16): NIR = 0.0 (valid zero!), Red = 0.5 -> NDVI = -1.0 (RdYlGn low = red)
    //   Top-right    (x>=16, y<16): NIR = 0.8, Red = 0.1 -> NDVI = +0.778 (RdYlGn high = green)
    //   Bottom-left  (x<16, y>=16): NIR = NaN, Red = 0.5 -> nodata discard (blue base shows)
    //   Bottom-right (x>=16, y>=16): NIR = 0.8, Red = 0.1, SCL = 9 (Cloud High Prob) -> masked to blue base!
    let src_nir = SourceId::new(981);
    let (_, tex_nir) = upload_r32f_tile(
        device,
        queue,
        &mut gpu_cache,
        src_nir,
        BandKey::Index(0),
        32,
        32,
        |x, y| {
            if x < 16 && y < 16 {
                0.0
            } else if x < 16 && y >= 16 {
                f32::NAN
            } else {
                0.8
            }
        },
    );
    let src_red = SourceId::new(982);
    let (_, tex_red) = upload_r32f_tile(
        device,
        queue,
        &mut gpu_cache,
        src_red,
        BandKey::Index(0),
        32,
        32,
        |x, _| if x < 16 { 0.5 } else { 0.1 },
    );
    // SCL covers only the right half (cell_rect x in [256..512]) so left half remains
    // uncovered `255` sentinel — testing that class 0 in scl_mask_bits NEVER falsely masks uncovered pixels!
    let src_scl = SourceId::new(983);
    let (_, tex_scl) = upload_r8u_tile(device, queue, &mut gpu_cache, src_scl, 16, 32, |_, y| {
        if y < 16 { 4 } else { 9 } // top-right = Vegetation (4), bottom-right = Cloud High (9)
    });

    let scl_mask_bits = (1u32 << 0) | (1u32 << 9); // mask NoData(0) and Cloud High(9)
    let make_bm = |scl_enabled: bool| DrawBandMathLayer {
        formula: BandMathFormula::NormalizedDiff,
        colormap: ColormapId::RdYlGn,
        invert_colormap: false,
        stretch_min: -1.0,
        stretch_max: 1.0,
        rgb_min: [0.0; 3],
        rgb_max: [1.0; 3],
        rgb_gamma: 1.0,
        scl_mask_enabled: scl_enabled,
        scl_mask_bits,
        bands: vec![
            scratch_band(&frame, src_nir, 1, 512, 512, 1.0, 0.0, Arc::clone(&tex_nir)),
            scratch_band(&frame, src_red, 2, 512, 512, 1.0, 0.0, Arc::clone(&tex_red)),
        ],
        scl: Some(DrawScratchMask {
            frame: Arc::clone(&frame),
            image_width: 512,
            image_height: 512,
            segments: MIN_WARP_SEGMENTS,
            nodata_value: None,
            patches: vec![DrawOverlayPatch {
                mesh_key: OverlayMeshKey {
                    source: src_scl,
                    relaid: 3,
                    level: 0,
                    x: 0,
                    y: 0,
                },
                cell_rect: [256.0, 0.0, 512.0, 512.0],
                uv_rect: [0.0, 0.0, 1.0, 1.0],
                texture: Arc::clone(&tex_scl),
            }],
        }),
    };

    let mut snap = build_empty_snapshot(cam, width, height, &config);
    snap.overlays.push(single_patch_layer(
        &frame,
        src_base,
        1,
        MIN_WARP_SEGMENTS,
        DrawOverlayTrack::Display,
        Arc::clone(&tex_base),
    ));
    snap.overlays.push(DrawOverlayLayer {
        layer_id: 9001,
        opacity: 1.0,
        content: DrawOverlayContent::BandMath(Box::new(make_bm(true))),
    });
    let pixels = render_srgb(&mut globe, device, queue, width, height, &Arc::new(snap));

    let is_blue_base = |px: [u8; 4]| px[0] < 20 && px[1] < 20 && px[2] > 235;
    let tl = pixel_at(&pixels, width, width / 2 - 20, height / 2 - 20);
    let tr = pixel_at(&pixels, width, width / 2 + 20, height / 2 - 20);
    let bl = pixel_at(&pixels, width, width / 2 - 20, height / 2 + 20);
    let br = pixel_at(&pixels, width, width / 2 + 20, height / 2 + 20);

    // Top-left: valid 0.0 NIR + uncovered SCL 255 sentinel must render RdYlGn low (reddish), NOT blue base.
    assert!(
        !is_blue_base(tl) && tl[0] > 140 && tl[2] < 80,
        "valid 0.0 reflectance & uncovered 255 SCL sentinel must render NDVI=-1.0 (red), got {tl:?}"
    );
    // Top-right: SCL class 4 (Vegetation, unmasked) + high NDVI must render RdYlGn high (greenish).
    assert!(
        !is_blue_base(tr) && tr[1] > 100 && tr[1] > tr[0],
        "unmasked SCL=4 high NDVI must render RdYlGn green, got {tr:?}"
    );
    // Bottom-left: NaN NIR must discard and reveal blue base.
    assert!(
        is_blue_base(bl),
        "NaN input band must discard to reveal blue base, got {bl:?}"
    );
    // Bottom-right: SCL class 9 (Cloud High Probability) must be masked to reveal blue base.
    assert!(
        is_blue_base(br),
        "SCL=9 cloud-masked quadrant must reveal blue base, got {br:?}"
    );

    // Toggling SCL mask OFF must immediately reveal the NDVI vegetation color in bottom-right!
    let mut snap_unmasked = build_empty_snapshot(cam, width, height, &config);
    snap_unmasked.overlays.push(single_patch_layer(
        &frame,
        src_base,
        1,
        MIN_WARP_SEGMENTS,
        DrawOverlayTrack::Display,
        Arc::clone(&tex_base),
    ));
    snap_unmasked.overlays.push(DrawOverlayLayer {
        layer_id: 9001,
        opacity: 1.0,
        content: DrawOverlayContent::BandMath(Box::new(make_bm(false))),
    });
    let pixels_unmasked = render_srgb(
        &mut globe,
        device,
        queue,
        width,
        height,
        &Arc::new(snap_unmasked),
    );
    let br_unmasked = pixel_at(&pixels_unmasked, width, width / 2 + 20, height / 2 + 20);
    assert!(
        !is_blue_base(br_unmasked) && br_unmasked[1] > 100 && br_unmasked[1] > br_unmasked[0],
        "disabling SCL mask must reveal NDVI green in bottom-right, got {br_unmasked:?}"
    );
}

#[test]
fn band_math_mrt_vs_non_mrt_parity_scissor_painter_order_and_lazy_scratch_release() {
    let Some((device, queue)) = shared_headless_device() else {
        return;
    };

    let (width, height) = (128u32, 128u32);
    let mut globe = TestGlobe::new_srgb(device);
    let mut gpu_cache = GpuTileCache::new(16 * 1024 * 1024);

    let frame_10m = utm37_test_frame();
    // Same geographic footprint represented at half image resolution (256x256, 200 m/px)
    // to exercise the non-MRT multi-GSD path (`can_use_mrt() == false`)!
    let utm37 = Projection::from_epsg(32637).expect("UTM 37N");
    let frame_20m = Arc::new(
        Frame::from_pixel_affine(
            utm37,
            [200.0, 0.0, 500_000.0, 0.0, -200.0, 5_250_000.0],
            256,
            256,
        )
        .expect("utm37 20m"),
    );

    let (clat, clon) = frame_10m.geodetic(0.5, 0.5);
    let cam = Camera::at(clat, clon, 150_000.0);
    let config = QuadtreeConfig::default();

    // Physical RGB: R = 0.8 (DN=9000, scale=0.0001, offset=-0.1), G = 0.4, B = 0.2.
    let src_r = SourceId::new(990);
    let src_g = SourceId::new(991);
    let src_b = SourceId::new(992);
    let (_, tex_r) = upload_r32f_tile(
        device,
        queue,
        &mut gpu_cache,
        src_r,
        BandKey::Index(0),
        16,
        16,
        |_, _| 9000.0,
    );
    let (_, tex_g) = upload_r32f_tile(
        device,
        queue,
        &mut gpu_cache,
        src_g,
        BandKey::Index(0),
        16,
        16,
        |_, _| 5000.0,
    );
    let (_, tex_b) = upload_r32f_tile(
        device,
        queue,
        &mut gpu_cache,
        src_b,
        BandKey::Index(0),
        16,
        16,
        |_, _| 3000.0,
    );

    // 1. MRT path (all 3 bands share 512x512 frame_10m).
    let bm_mrt = DrawBandMathLayer {
        formula: BandMathFormula::RgbComposite,
        colormap: ColormapId::Grayscale,
        invert_colormap: false,
        stretch_min: 0.0,
        stretch_max: 1.0,
        rgb_min: [0.0, 0.0, 0.0],
        rgb_max: [1.0, 1.0, 1.0],
        rgb_gamma: 1.0,
        scl_mask_enabled: false,
        scl_mask_bits: 0,
        bands: vec![
            scratch_band(
                &frame_10m,
                src_r,
                1,
                512,
                512,
                0.0001,
                -0.1,
                Arc::clone(&tex_r),
            ),
            scratch_band(
                &frame_10m,
                src_g,
                2,
                512,
                512,
                0.0001,
                -0.1,
                Arc::clone(&tex_g),
            ),
            scratch_band(
                &frame_10m,
                src_b,
                3,
                512,
                512,
                0.0001,
                -0.1,
                Arc::clone(&tex_b),
            ),
        ],
        scl: None,
    };
    assert!(
        bm_mrt.can_use_mrt(),
        "co-grid 3-band layer must take MRT path"
    );

    // Verify scissor rect is tight (strictly smaller than full viewport) on-screen,
    // None off-screen, and full viewport when straddling the horizon limb.
    let scissor_on = compute_band_math_scissor_rect(&cam, width, height, &bm_mrt)
        .expect("on-screen scissor rect");
    assert!(
        scissor_on[2] > 0 && scissor_on[3] > 0 && (scissor_on[2] < width || scissor_on[3] < height),
        "on-screen patch scissor should be tight, got {scissor_on:?}"
    );
    let cam_far_side = Camera::at(-clat, (clon + 180.0 + 180.0) % 360.0 - 180.0, 500_000.0);
    let scissor_back = compute_band_math_scissor_rect(&cam_far_side, width, height, &bm_mrt);
    assert_eq!(
        scissor_back,
        Some([0, 0, width, height]),
        "back-hemisphere / horizon fallback must return full viewport safely"
    );
    // 3 deg north is inside the ~6.8 deg horizon limb at 45 km altitude, yet well off-screen (~333 km >> 42 km FOV).
    let cam_offscreen = Camera::at(clat + 3.0, clon, 45_000.0);
    assert_eq!(
        compute_band_math_scissor_rect(&cam_offscreen, width, height, &bm_mrt),
        None,
        "completely off-screen on-hemisphere patch must return empty/None scissor"
    );

    // 2. Non-MRT path (band 2 on 256x256 frame_20m -> mixed-GSD fallback).
    let bm_non_mrt = DrawBandMathLayer {
        bands: vec![
            scratch_band(
                &frame_10m,
                src_r,
                1,
                512,
                512,
                0.0001,
                -0.1,
                Arc::clone(&tex_r),
            ),
            scratch_band(
                &frame_10m,
                src_g,
                2,
                512,
                512,
                0.0001,
                -0.1,
                Arc::clone(&tex_g),
            ),
            scratch_band(
                &frame_20m,
                src_b,
                4,
                256,
                256,
                0.0001,
                -0.1,
                Arc::clone(&tex_b),
            ),
        ],
        ..bm_mrt.clone()
    };
    assert!(
        !bm_non_mrt.can_use_mrt(),
        "mixed-GSD 3-band layer must take non-MRT per-band scratch path"
    );

    let mut snap_mrt = build_empty_snapshot(cam, width, height, &config);
    snap_mrt.overlays.push(DrawOverlayLayer {
        layer_id: 9002,
        opacity: 1.0,
        content: DrawOverlayContent::BandMath(Box::new(bm_mrt.clone())),
    });
    let pixels_mrt = render_srgb(
        &mut globe,
        device,
        queue,
        width,
        height,
        &Arc::new(snap_mrt),
    );
    // Verify lazy scratch pool allocated exact 3x R32Float (no mask) = 12 B/px.
    assert_eq!(
        globe.renderer.scratch_vram_bytes(),
        (width * height * 12) as usize,
        "lazy scratch pool must allocate exact 3x R32Float without mask"
    );

    let mut snap_non_mrt = build_empty_snapshot(cam, width, height, &config);
    snap_non_mrt.overlays.push(DrawOverlayLayer {
        layer_id: 9003,
        opacity: 1.0,
        content: DrawOverlayContent::BandMath(Box::new(bm_non_mrt)),
    });
    let pixels_non_mrt = render_srgb(
        &mut globe,
        device,
        queue,
        width,
        height,
        &Arc::new(snap_non_mrt),
    );

    // Exact pixel parity between MRT and non-MRT paths at patch center!
    let px_mrt = pixel_at(&pixels_mrt, width, width / 2, height / 2);
    let px_non_mrt = pixel_at(&pixels_non_mrt, width, width / 2, height / 2);
    assert!(
        px_mrt[0] > 195 && px_mrt[1] > 90 && px_mrt[1] < 115 && px_mrt[2] > 40 && px_mrt[2] < 65,
        "expected sRGB (~204, ~102, ~51) from physical (0.8, 0.4, 0.2), got {px_mrt:?}"
    );
    for c in 0..3 {
        assert!(
            px_mrt[c].abs_diff(px_non_mrt[c]) <= 1,
            "MRT {px_mrt:?} vs non-MRT {px_non_mrt:?} mismatch on channel {c}"
        );
    }

    // 3. Pass-2 painter's-order interleaving: Direct bottom (green) -> BandMath -> Direct top (50% blue).
    let src_top = SourceId::new(993);
    let (_, tex_top) =
        upload_overlay_tile(device, queue, &mut gpu_cache, src_top, 16, 16, |_, _| {
            [0, 0, 255, 255]
        });
    let mut top_direct = single_patch_layer(
        &frame_10m,
        src_top,
        5,
        MIN_WARP_SEGMENTS,
        DrawOverlayTrack::Display,
        tex_top,
    );
    top_direct.opacity = 0.5;
    let mut snap_interleaved = build_empty_snapshot(cam, width, height, &config);
    snap_interleaved.overlays.push(DrawOverlayLayer {
        layer_id: 9004,
        opacity: 1.0,
        content: DrawOverlayContent::BandMath(Box::new(bm_mrt)),
    });
    snap_interleaved.overlays.push(top_direct);
    let pixels_interleaved = render_srgb(
        &mut globe,
        device,
        queue,
        width,
        height,
        &Arc::new(snap_interleaved),
    );
    let px_inter = pixel_at(&pixels_interleaved, width, width / 2, height / 2);
    assert!(
        px_inter[2] > px_mrt[2] + 80 && px_inter[0] < px_mrt[0] - 20,
        "Direct layer above BandMath must composite over BandMath in Pass-2 painter order: before={px_mrt:?}, after={px_inter:?}"
    );

    // 4. Removing the BandMath layer releases all scratch VRAM immediately on the next frame!
    let snap_empty = build_empty_snapshot(cam, width, height, &config);
    let _ = render_srgb(
        &mut globe,
        device,
        queue,
        width,
        height,
        &Arc::new(snap_empty),
    );
    assert_eq!(
        globe.renderer.scratch_vram_bytes(),
        0,
        "scratch pool must release all VRAM immediately when no BandMath layer is visible"
    );

    // 5. H2 regression: distinct projections sharing identical 4 corner lat/lons
    //    (geographic quad vs Web Mercator EPSG:3857) MUST NOT batch into MRT.
    let merc_frame = Arc::new(
        Frame::from_crs_pixel_affine(
            &veldmap_core::geo::CrsSpec::from_epsg(3857).expect("epsg 3857"),
            [8_000.0, 0.0, 0.0, 0.0, -8_000.0, 8_000_000.0],
            512,
            512,
        )
        .expect("mercator frame"),
    );
    let geo_frame = Arc::new(
        Frame::quad([
            merc_frame.geodetic(0.0, 0.0),
            merc_frame.geodetic(1.0, 0.0),
            merc_frame.geodetic(0.0, 1.0),
            merc_frame.geodetic(1.0, 1.0),
        ])
        .expect("quad frame with identical 4 corners"),
    );
    let bm_diff_proj = DrawBandMathLayer {
        formula: BandMathFormula::Difference,
        colormap: ColormapId::RdBu,
        invert_colormap: false,
        stretch_min: -1.0,
        stretch_max: 1.0,
        rgb_min: [0.0; 3],
        rgb_max: [1.0; 3],
        rgb_gamma: 1.0,
        scl_mask_enabled: false,
        scl_mask_bits: 0,
        bands: vec![
            scratch_band(
                &geo_frame,
                src_r,
                10,
                512,
                512,
                1.0,
                0.0,
                Arc::clone(&tex_r),
            ),
            scratch_band(
                &merc_frame,
                src_g,
                11,
                512,
                512,
                1.0,
                0.0,
                Arc::clone(&tex_g),
            ),
        ],
        scl: None,
    };
    assert!(
        !bm_diff_proj.can_use_mrt(),
        "EPSG:4326 and EPSG:3857 frames sharing 4 corners must not batch into MRT"
    );

    // 6. H3 regression: a trailing empty BandMath layer with no allocated scratch
    //    pool MUST NOT drop the post-overlay graticule pass.
    let mut snap_grat_ref = build_empty_snapshot(cam, width, height, &config);
    snap_grat_ref.show_graticule = true;
    let pixels_grat_ref = render_srgb(
        &mut globe,
        device,
        queue,
        width,
        height,
        &Arc::new(snap_grat_ref),
    );
    let mut snap_trailing_empty = build_empty_snapshot(cam, width, height, &config);
    snap_trailing_empty.show_graticule = true;
    snap_trailing_empty.overlays.push(DrawOverlayLayer {
        layer_id: 9005,
        opacity: 1.0,
        content: DrawOverlayContent::BandMath(Box::new(DrawBandMathLayer {
            bands: Vec::new(),
            scl: None,
            ..bm_diff_proj
        })),
    });
    let pixels_trailing_empty = render_srgb(
        &mut globe,
        device,
        queue,
        width,
        height,
        &Arc::new(snap_trailing_empty),
    );
    assert_eq!(
        pixels_grat_ref, pixels_trailing_empty,
        "trailing empty BandMath layer must preserve post-overlay graticule rendering"
    );
}

#[test]
fn split_screen_swipe_clips_direct_and_band_math_without_rebuilding_snapshot_or_zero_scissors() {
    use veldmap_app::render::SwipeRenderState;

    let Some((device, queue)) = shared_headless_device() else {
        return;
    };

    let (width, height) = (161, 120);
    let config = QuadtreeConfig::geodetic(2);
    let frame = Arc::new(
        Frame::quad([(45.3, 10.0), (45.3, 10.6), (44.7, 10.6), (44.7, 10.0)]).expect("frame"),
    );
    let cam = Camera::at(45.0, 10.3, 30_000.0);
    let mut gpu_cache = GpuTileCache::new(64 * 1024 * 1024);

    // Left layer: Direct solid cyan (layer_id = 101).
    let src_direct = SourceId::new(980);
    let (_, tex_cyan) =
        upload_overlay_tile(device, queue, &mut gpu_cache, src_direct, 16, 16, |_, _| {
            [0, 230, 230, 255]
        });
    let mut direct_layer = single_patch_layer(
        &frame,
        src_direct,
        1,
        MIN_WARP_SEGMENTS,
        DrawOverlayTrack::Display,
        tex_cyan,
    );
    direct_layer.layer_id = 101;

    // Right layer: BandMath RGB solid orange (layer_id = 202).
    let src_r = SourceId::new(981);
    let src_g = SourceId::new(982);
    let src_b = SourceId::new(983);
    let (_, tex_r) = upload_r32f_tile(
        device,
        queue,
        &mut gpu_cache,
        src_r,
        BandKey::Index(0),
        16,
        16,
        |_, _| 1.0,
    );
    let (_, tex_g) = upload_r32f_tile(
        device,
        queue,
        &mut gpu_cache,
        src_g,
        BandKey::Index(0),
        16,
        16,
        |_, _| 0.5,
    );
    let (_, tex_b) = upload_r32f_tile(
        device,
        queue,
        &mut gpu_cache,
        src_b,
        BandKey::Index(0),
        16,
        16,
        |_, _| 0.0,
    );
    let bm_layer = DrawOverlayLayer {
        layer_id: 202,
        opacity: 1.0,
        content: DrawOverlayContent::BandMath(Box::new(DrawBandMathLayer {
            formula: BandMathFormula::RgbComposite,
            colormap: ColormapId::Grayscale,
            invert_colormap: false,
            stretch_min: 0.0,
            stretch_max: 1.0,
            rgb_min: [0.0; 3],
            rgb_max: [1.0; 3],
            rgb_gamma: 1.0,
            scl_mask_enabled: false,
            scl_mask_bits: 0,
            bands: vec![
                scratch_band(&frame, src_r, 2, 512, 512, 1.0, 0.0, tex_r),
                scratch_band(&frame, src_g, 3, 512, 512, 1.0, 0.0, tex_g),
                scratch_band(&frame, src_b, 4, 512, 512, 1.0, 0.0, tex_b),
            ],
            scl: None,
        })),
    };

    let mut snap = build_empty_snapshot(cam, width, height, &config);
    snap.overlays.push(bm_layer);
    snap.overlays.push(direct_layer);
    snap.show_graticule = true;
    let snap = Arc::new(snap);
    let mut globe = TestGlobe::new_srgb(device);

    // 1. Mid-split (`split_t = 0.43` on odd width `161`): verify exact physical
    //    seam pixel (`split_x - 1` is cyan Direct, `split_x` is orange BandMath).
    let mid_swipe = SwipeRenderState {
        split_t: 0.43,
        left_layer: Some(101),
        right_layer: Some(202),
    };
    let split_x = veldmap_app::render::snap_split_x(mid_swipe.split_t, width);
    let px_mid = render_with(
        &mut globe,
        device,
        queue,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        RenderParams {
            width,
            height,
            snapshot: &snap,
            swipe: Some(mid_swipe),
        },
        true,
    );
    // Probe off the exact 45.0° N graticule parallel (`y = height / 2`) for
    // pure cyan/orange seam verification, and also assert the graticule renders
    // unclipped on BOTH sides of `split_x` after the trailing clipped Direct pass.
    let probe_y = height / 2 + 12;
    let left_seam = pixel_at(&px_mid, width, split_x - 1, probe_y);
    let right_seam = pixel_at(&px_mid, width, split_x, probe_y);
    assert!(
        left_seam[0] < 20 && left_seam[1] > 210 && left_seam[2] > 210,
        "pixel split_x-1 ({}) must render cyan Direct layer, got {left_seam:?}",
        split_x - 1
    );
    assert!(
        right_seam[0] > 230 && right_seam[1] > 110 && right_seam[1] < 145 && right_seam[2] < 20,
        "pixel split_x ({split_x}) must render orange BandMath layer, got {right_seam:?}"
    );
    // The 45.0° N graticule parallel at `y = height / 2` must blend visibly
    // across BOTH half-planes (confirming full-viewport scissor restore after
    // the trailing clipped Direct layer).
    let left_grat = pixel_at(&px_mid, width, split_x - 4, height / 2);
    let right_grat = pixel_at(&px_mid, width, split_x + 4, height / 2);
    assert!(
        left_grat[0] > left_seam[0] + 40,
        "graticule must remain visible on left half-plane: seam={left_seam:?}, grat={left_grat:?}"
    );
    assert!(
        right_grat[2] > right_seam[2] + 40,
        "graticule must remain visible on right half-plane: seam={right_seam:?}, grat={right_grat:?}"
    );

    // 2. Identical snapshot + identical swipe triggers 0 idle GPU re-renders.
    let _ = render_with(
        &mut globe,
        device,
        queue,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        RenderParams {
            width,
            height,
            snapshot: &snap,
            swipe: Some(mid_swipe),
        },
        false,
    );

    // 3. Curtain dragged to `split_t = 1.0` on the SAME `Arc<DrawSnapshot>`:
    //    `LastRenderKey` must trigger a re-render, Direct covers full width, and
    //    BandMath has empty scissor intersection (skipped cleanly with zero wgpu error!).
    let px_full_left = render_with(
        &mut globe,
        device,
        queue,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        RenderParams {
            width,
            height,
            snapshot: &snap,
            swipe: Some(SwipeRenderState {
                split_t: 1.0,
                left_layer: Some(101),
                right_layer: Some(202),
            }),
        },
        true,
    );
    let right_at_1 = pixel_at(&px_full_left, width, (width * 3) / 4, probe_y);
    assert!(
        right_at_1[0] < 20 && right_at_1[1] > 210 && right_at_1[2] > 210,
        "at split_t=1.0 right side must show cyan Direct layer (BandMath skipped), got {right_at_1:?}"
    );

    // 4. Curtain dragged to `split_t = 0.0` on the SAME `Arc<DrawSnapshot>`:
    //    Direct has empty scissor (skipped cleanly), BandMath covers full width.
    let px_full_right = render_with(
        &mut globe,
        device,
        queue,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        RenderParams {
            width,
            height,
            snapshot: &snap,
            swipe: Some(SwipeRenderState {
                split_t: 0.0,
                left_layer: Some(101),
                right_layer: Some(202),
            }),
        },
        true,
    );
    let left_at_0 = pixel_at(&px_full_right, width, width / 4, probe_y);
    assert!(
        left_at_0[0] > 230 && left_at_0[1] > 110 && left_at_0[1] < 145 && left_at_0[2] < 20,
        "at split_t=0.0 left side must show orange BandMath layer (Direct skipped), got {left_at_0:?}"
    );
}

#[test]
fn temporal_scalar_diff_n1_and_n2_preserve_zero_diff_and_independent_scl_masks() {
    let Some((device, queue)) = shared_headless_device() else {
        return;
    };

    let (width, height) = (128u32, 128u32);
    let mut globe = TestGlobe::new_srgb(device);
    let mut gpu_cache = GpuTileCache::new(16 * 1024 * 1024);

    let frame_a = utm37_test_frame();
    // Non-co-registered Epoch B on a 256x256 (half-res) grid covering the same
    // geographic footprint to prove screen-space diff works across mismatched
    // carrier grids without CPU reprojection.
    let utm37 = Projection::from_epsg(32637).expect("UTM 37N");
    let frame_b = Arc::new(
        Frame::from_pixel_affine(
            utm37,
            [200.0, 0.0, 500_000.0, 0.0, -200.0, 5_250_000.0],
            256,
            256,
        )
        .expect("utm37 20m"),
    );

    let (clat, clon) = frame_a.geodetic(0.5, 0.5);
    let cam = Camera::at(clat, clon, 25_000.0);
    let config = QuadtreeConfig::default();

    // Solid green base layer beneath TemporalDiff so discarded/masked pixels
    // are unmistakably pure green (0, 255, 0), whereas valid 0.0 diff on RdBu
    // over [-1, 1] is near-white (~247, ~247, ~247).
    let src_base = SourceId::new(1100);
    let (_, tex_base) =
        upload_overlay_tile(device, queue, &mut gpu_cache, src_base, 16, 16, |_, _| {
            [0, 255, 0, 255]
        });
    let is_green_base = |px: [u8; 4]| px[0] < 25 && px[1] > 230 && px[2] < 25;

    // 1. N = 2 scalar diff (dNDVI = NDVI(B) - NDVI(A)) across 4 quadrants:
    //   TL (x<16, y<16): Epoch A NDVI = +0.5, Epoch B NDVI = +0.5 -> dNDVI = 0.0 (valid zero -> RdBu white!)
    //   TR (x>=16, y<16): Epoch A NDVI = -0.5, Epoch B NDVI = +0.5 -> dNDVI = +1.0 (RdBu high = blue)
    //   BL (x<16, y>=16): Epoch A has SCL = 9 (Cloud High), Epoch B has NO SCL mask -> A is masked -> discard to green base!
    //   BR (x>=16, y>=16): Epoch A has valid SCL = 4 (Vegetation), Epoch B has NO SCL mask, NDVI(A)=+0.5, NDVI(B)=-0.5 -> dNDVI = -1.0 (RdBu low = red), proving Epoch A's mask does NOT bleed into Epoch B!
    let (src_a_nir, src_a_red, src_a_scl) = (
        SourceId::new(1101),
        SourceId::new(1102),
        SourceId::new(1103),
    );
    let (src_b_nir, src_b_red) = (SourceId::new(1104), SourceId::new(1105));

    let (_, tex_a_nir) = upload_r32f_tile(
        device,
        queue,
        &mut gpu_cache,
        src_a_nir,
        BandKey::Index(0),
        32,
        32,
        |x, y| if x >= 16 && y < 16 { 0.25 } else { 0.75 },
    );
    let (_, tex_a_red) = upload_r32f_tile(
        device,
        queue,
        &mut gpu_cache,
        src_a_red,
        BandKey::Index(0),
        32,
        32,
        |x, y| if x >= 16 && y < 16 { 0.75 } else { 0.25 },
    );
    let (_, tex_a_scl) =
        upload_r8u_tile(device, queue, &mut gpu_cache, src_a_scl, 32, 32, |x, y| {
            if x < 16 && y >= 16 { 9 } else { 4 }
        });
    let (_, tex_b_nir) = upload_r32f_tile(
        device,
        queue,
        &mut gpu_cache,
        src_b_nir,
        BandKey::Index(0),
        32,
        32,
        |x, y| if x >= 16 && y >= 16 { 0.25 } else { 0.75 },
    );
    let (_, tex_b_red) = upload_r32f_tile(
        device,
        queue,
        &mut gpu_cache,
        src_b_red,
        BandKey::Index(0),
        32,
        32,
        |x, y| if x >= 16 && y >= 16 { 0.75 } else { 0.25 },
    );

    let diff_n2 = DrawTemporalDiffLayer {
        formula: Some(BandMathFormula::NormalizedDiff),
        colormap: ColormapId::RdBu,
        invert_colormap: false,
        stretch_min: -1.0,
        stretch_max: 1.0,
        scl_mask_enabled: true,
        scl_mask_bits: 1u32 << 9,
        epoch_a: DrawBandEpoch {
            bands: vec![
                scratch_band(
                    &frame_a,
                    src_a_nir,
                    1,
                    512,
                    512,
                    1.0,
                    0.0,
                    Arc::clone(&tex_a_nir),
                ),
                scratch_band(
                    &frame_a,
                    src_a_red,
                    2,
                    512,
                    512,
                    1.0,
                    0.0,
                    Arc::clone(&tex_a_red),
                ),
            ],
            scl: Some(DrawScratchMask {
                frame: Arc::clone(&frame_a),
                image_width: 512,
                image_height: 512,
                segments: MIN_WARP_SEGMENTS,
                nodata_value: None,
                patches: vec![DrawOverlayPatch {
                    mesh_key: OverlayMeshKey {
                        source: src_a_scl,
                        relaid: 3,
                        level: 0,
                        x: 0,
                        y: 0,
                    },
                    cell_rect: [0.0, 0.0, 512.0, 512.0],
                    uv_rect: [0.0, 0.0, 1.0, 1.0],
                    texture: Arc::clone(&tex_a_scl),
                }],
            }),
        },
        epoch_b: DrawBandEpoch {
            bands: vec![
                scratch_band(
                    &frame_b,
                    src_b_nir,
                    4,
                    256,
                    256,
                    1.0,
                    0.0,
                    Arc::clone(&tex_b_nir),
                ),
                scratch_band(
                    &frame_b,
                    src_b_red,
                    5,
                    256,
                    256,
                    1.0,
                    0.0,
                    Arc::clone(&tex_b_red),
                ),
            ],
            scl: None,
        },
    };

    let mut snap_n2 = build_empty_snapshot(cam, width, height, &config);
    snap_n2.overlays.push(single_patch_layer(
        &frame_a,
        src_base,
        1,
        MIN_WARP_SEGMENTS,
        DrawOverlayTrack::Display,
        Arc::clone(&tex_base),
    ));
    snap_n2.overlays.push(DrawOverlayLayer {
        layer_id: 9101,
        opacity: 1.0,
        content: DrawOverlayContent::TemporalDiff(Box::new(diff_n2)),
    });
    let pixels_n2 = render_srgb(&mut globe, device, queue, width, height, &Arc::new(snap_n2));

    // Scratch demand for N=2 + SCL: 3x R32Float (12 B/px) + 1x R8Uint (1 B/px) = 13 B/px.
    assert_eq!(
        globe.renderer.scratch_vram_bytes(),
        (width * height * 13) as usize,
        "N=2 scalar diff with SCL must allocate exact 3x R32Float + 1x R8Uint (13 B/px)"
    );

    let tl = pixel_at(&pixels_n2, width, width / 2 - 20, height / 2 - 20);
    let tr = pixel_at(&pixels_n2, width, width / 2 + 20, height / 2 - 20);
    let bl = pixel_at(&pixels_n2, width, width / 2 - 20, height / 2 + 20);
    let br = pixel_at(&pixels_n2, width, width / 2 + 20, height / 2 + 20);
    assert!(
        !is_green_base(tl) && tl[0] > 210 && tl[1] > 210 && tl[2] > 210,
        "TL valid 0.0 dNDVI must render RdBu neutral white (not discarded to green base), got {tl:?}"
    );
    // ColorBrewer RdBu endpoints: +1.0 = deep blue #053061 (5, 48, 97),
    // -1.0 = deep crimson #67001f (103, 0, 31), 0.0 = neutral white ~#f7f7f7.
    assert!(
        !is_green_base(tr) && tr[2] > 80 && tr[2] > tr[0] + 50,
        "TR +1.0 dNDVI must render RdBu blue (#053061), got {tr:?}"
    );
    assert!(
        is_green_base(bl),
        "BL Epoch A SCL=9 cloud mask must reduce to quiet-NaN and reveal green base, got {bl:?}"
    );
    assert!(
        !is_green_base(br) && br[0] > 80 && br[0] > br[2] + 50,
        "BR -1.0 dNDVI (unmasked Epoch A & no-SCL Epoch B) must render RdBu red (#67001f), got {br:?}"
    );

    // 2. N = 1 single-band passthrough diff (`formula: None`, e.g. SAR / DEM delta):
    //   Left half: A = 10.0, B = 10.0 -> diff = 0.0 (valid zero -> Grayscale mid-gray ~128 over [-10, 10])
    //   Right half: A = 2.0, B = 12.0 -> diff = +10.0 (Grayscale white ~255)
    let (src_s_a, src_s_b) = (SourceId::new(1106), SourceId::new(1107));
    let (_, tex_s_a) = upload_r32f_tile(
        device,
        queue,
        &mut gpu_cache,
        src_s_a,
        BandKey::Index(0),
        32,
        32,
        |x, _| if x < 16 { 10.0 } else { 2.0 },
    );
    let (_, tex_s_b) = upload_r32f_tile(
        device,
        queue,
        &mut gpu_cache,
        src_s_b,
        BandKey::Index(0),
        32,
        32,
        |x, _| if x < 16 { 10.0 } else { 12.0 },
    );
    let diff_n1 = DrawTemporalDiffLayer {
        formula: None,
        colormap: ColormapId::Grayscale,
        invert_colormap: false,
        stretch_min: -10.0,
        stretch_max: 10.0,
        scl_mask_enabled: false,
        scl_mask_bits: 0,
        epoch_a: DrawBandEpoch {
            bands: vec![scratch_band(
                &frame_a, src_s_a, 6, 512, 512, 1.0, 0.0, tex_s_a,
            )],
            scl: None,
        },
        epoch_b: DrawBandEpoch {
            bands: vec![scratch_band(
                &frame_b, src_s_b, 7, 256, 256, 1.0, 0.0, tex_s_b,
            )],
            scl: None,
        },
    };
    let mut snap_n1 = build_empty_snapshot(cam, width, height, &config);
    snap_n1.overlays.push(single_patch_layer(
        &frame_a,
        src_base,
        1,
        MIN_WARP_SEGMENTS,
        DrawOverlayTrack::Display,
        tex_base,
    ));
    snap_n1.overlays.push(DrawOverlayLayer {
        layer_id: 9102,
        opacity: 1.0,
        content: DrawOverlayContent::TemporalDiff(Box::new(diff_n1)),
    });
    let pixels_n1 = render_srgb(&mut globe, device, queue, width, height, &Arc::new(snap_n1));
    assert_eq!(
        globe.renderer.scratch_vram_bytes(),
        (width * height * 8) as usize,
        "N=1 scalar diff without SCL must allocate exact 2x R32Float (8 B/px)"
    );
    let left_n1 = pixel_at(&pixels_n1, width, width / 2 - 20, height / 2);
    let right_n1 = pixel_at(&pixels_n1, width, width / 2 + 20, height / 2);
    assert!(
        !is_green_base(left_n1)
            && left_n1[0].abs_diff(left_n1[1]) <= 2
            && left_n1[0] > 100
            && left_n1[0] < 160,
        "N=1 valid 0.0 diff must render mid-gray (~128), got {left_n1:?}"
    );
    assert!(
        right_n1[0] > 240 && right_n1[1] > 240 && right_n1[2] > 240,
        "N=1 +10.0 diff must render white, got {right_n1:?}"
    );
}

#[test]
fn temporal_rgb_cross_fade_band_math_and_direct_display_have_zero_basemap_bleed_through() {
    use veldmap_app::render::SwipeRenderState;

    let Some((device, queue)) = shared_headless_device() else {
        return;
    };

    let (width, height) = (128u32, 128u32);
    let mut globe = TestGlobe::new_srgb(device);
    let mut gpu_cache = GpuTileCache::new(16 * 1024 * 1024);

    let frame = utm37_test_frame();
    let (clat, clon) = frame.geodetic(0.5, 0.5);
    let cam = Camera::at(clat, clon, 25_000.0);
    let config = QuadtreeConfig::default();

    // Bright green base layer beneath the cross-fade layer to catch ANY
    // basemap bleed-through at t = 0.5 (if two separate 50% alpha passes were
    // drawn over green, the green channel would be >= 64!).
    let src_base = SourceId::new(1120);
    let (_, tex_base) =
        upload_overlay_tile(device, queue, &mut gpu_cache, src_base, 16, 16, |_, _| {
            [0, 255, 0, 255]
        });

    // 1. DirectDisplay cross-fade (`float_bands = 0`, `rgba_targets = 2`, 8 B/px):
    //   Epoch A: left+right opaque pure red (255, 0, 0, 255)
    //   Epoch B: left half transparent nodata (0, 0, 0, 0), right half opaque pure blue (0, 0, 255, 255)
    //   At blend_t = 0.5:
    //     - Right half (both valid): pure red<->blue blend with ZERO green bleed-through (`g < 5`)!
    //     - Left half (only A valid): graceful single-valid fade of A at 50% opacity over green base (`r > 120 && g > 120 && b < 15`)!
    let (src_da, src_db) = (SourceId::new(1121), SourceId::new(1122));
    let (_, tex_da) = upload_overlay_tile(device, queue, &mut gpu_cache, src_da, 32, 32, |_, _| {
        [255, 0, 0, 255]
    });
    let (_, tex_db) = upload_overlay_tile(device, queue, &mut gpu_cache, src_db, 32, 32, |x, _| {
        if x < 16 {
            [0, 0, 0, 0]
        } else {
            [0, 0, 255, 255]
        }
    });

    let cf_direct = DrawTemporalCrossFadeLayer {
        blend_t: 0.5,
        source: DrawTemporalCrossFadeSource::DirectDisplay {
            epoch_a: DrawDisplayEpoch {
                frame: Arc::clone(&frame),
                image_width: 512,
                image_height: 512,
                segments: MIN_WARP_SEGMENTS,
                patches: vec![DrawOverlayPatch {
                    mesh_key: OverlayMeshKey {
                        source: src_da,
                        relaid: 1,
                        level: 0,
                        x: 0,
                        y: 0,
                    },
                    cell_rect: [0.0, 0.0, 512.0, 512.0],
                    uv_rect: [0.0, 0.0, 1.0, 1.0],
                    texture: tex_da,
                }],
            },
            epoch_b: DrawDisplayEpoch {
                frame: Arc::clone(&frame),
                image_width: 512,
                image_height: 512,
                segments: MIN_WARP_SEGMENTS,
                patches: vec![DrawOverlayPatch {
                    mesh_key: OverlayMeshKey {
                        source: src_db,
                        relaid: 2,
                        level: 0,
                        x: 0,
                        y: 0,
                    },
                    cell_rect: [0.0, 0.0, 512.0, 512.0],
                    uv_rect: [0.0, 0.0, 1.0, 1.0],
                    texture: tex_db,
                }],
            },
        },
    };

    let mut snap_disp = build_empty_snapshot(cam, width, height, &config);
    snap_disp.overlays.push(single_patch_layer(
        &frame,
        src_base,
        1,
        MIN_WARP_SEGMENTS,
        DrawOverlayTrack::Display,
        Arc::clone(&tex_base),
    ));
    snap_disp.overlays.push(DrawOverlayLayer {
        layer_id: 9201,
        opacity: 1.0,
        content: DrawOverlayContent::TemporalCrossFade(Box::new(cf_direct)),
    });
    let pixels_disp = render_srgb(
        &mut globe,
        device,
        queue,
        width,
        height,
        &Arc::new(snap_disp),
    );
    assert_eq!(
        globe.renderer.scratch_vram_bytes(),
        (width * height * 8) as usize,
        "DirectDisplay cross-fade must allocate exact 2x Rgba8UnormSrgb (8 B/px) and 0 R32Float"
    );
    let disp_right = pixel_at(&pixels_disp, width, width / 2 + 20, height / 2);
    assert!(
        disp_right[0] > 160 && disp_right[1] < 5 && disp_right[2] > 160,
        "DirectDisplay cross-fade at t=0.5 must blend red+blue with ZERO green bleed-through, got {disp_right:?}"
    );
    let disp_left = pixel_at(&pixels_disp, width, width / 2 - 20, height / 2);
    assert!(
        disp_left[0] > 140 && disp_left[1] > 140 && disp_left[2] < 15,
        "DirectDisplay single-valid left half must fade Epoch A smoothly over green base, got {disp_left:?}"
    );

    // 2. BandMathRgb cross-fade (`float_bands = 3`, `needs_u8_mask = true`, `rgba_targets = 1`, 17 B/px):
    //   Epoch A: pure red (1.0, 0.0, 0.0), SCL = 4 everywhere
    //   Epoch B: pure blue (0.0, 0.0, 1.0), SCL = 9 (Cloud) on left half, 4 on right half
    //   At blend_t = 0.5:
    //     - Right half (both unmasked): red<->blue with ZERO green bleed-through (`g < 5`)!
    //     - Left half (Epoch B cloud-masked): fades Epoch A red over green base!
    let (sa_r, sa_g, sa_b, sa_scl) = (
        SourceId::new(1130),
        SourceId::new(1131),
        SourceId::new(1132),
        SourceId::new(1133),
    );
    let (sb_r, sb_g, sb_b, sb_scl) = (
        SourceId::new(1134),
        SourceId::new(1135),
        SourceId::new(1136),
        SourceId::new(1137),
    );
    let (_, tex_one) = upload_r32f_tile(
        device,
        queue,
        &mut gpu_cache,
        sa_r,
        BandKey::Index(0),
        16,
        16,
        |_, _| 1.0,
    );
    let (_, tex_zero) = upload_r32f_tile(
        device,
        queue,
        &mut gpu_cache,
        sa_g,
        BandKey::Index(0),
        16,
        16,
        |_, _| 0.0,
    );
    let (_, tex_scl_clear) =
        upload_r8u_tile(device, queue, &mut gpu_cache, sa_scl, 16, 16, |_, _| 4);
    let (_, tex_scl_half) =
        upload_r8u_tile(device, queue, &mut gpu_cache, sb_scl, 32, 32, |x, _| {
            if x < 16 { 9 } else { 4 }
        });

    let cf_bm = DrawTemporalCrossFadeLayer {
        blend_t: 0.5,
        source: DrawTemporalCrossFadeSource::BandMathRgb {
            rgb_min: [0.0; 3],
            rgb_max: [1.0; 3],
            rgb_gamma: 1.0,
            scl_mask_enabled: true,
            scl_mask_bits: 1u32 << 9,
            epoch_a: DrawBandEpoch {
                bands: vec![
                    scratch_band(&frame, sa_r, 10, 512, 512, 1.0, 0.0, Arc::clone(&tex_one)),
                    scratch_band(&frame, sa_g, 11, 512, 512, 1.0, 0.0, Arc::clone(&tex_zero)),
                    scratch_band(&frame, sa_b, 12, 512, 512, 1.0, 0.0, Arc::clone(&tex_zero)),
                ],
                scl: Some(DrawScratchMask {
                    frame: Arc::clone(&frame),
                    image_width: 512,
                    image_height: 512,
                    segments: MIN_WARP_SEGMENTS,
                    nodata_value: None,
                    patches: vec![DrawOverlayPatch {
                        mesh_key: OverlayMeshKey {
                            source: sa_scl,
                            relaid: 13,
                            level: 0,
                            x: 0,
                            y: 0,
                        },
                        cell_rect: [0.0, 0.0, 512.0, 512.0],
                        uv_rect: [0.0, 0.0, 1.0, 1.0],
                        texture: tex_scl_clear,
                    }],
                }),
            },
            epoch_b: DrawBandEpoch {
                bands: vec![
                    scratch_band(&frame, sb_r, 14, 512, 512, 1.0, 0.0, Arc::clone(&tex_zero)),
                    scratch_band(&frame, sb_g, 15, 512, 512, 1.0, 0.0, Arc::clone(&tex_zero)),
                    scratch_band(&frame, sb_b, 16, 512, 512, 1.0, 0.0, Arc::clone(&tex_one)),
                ],
                scl: Some(DrawScratchMask {
                    frame: Arc::clone(&frame),
                    image_width: 512,
                    image_height: 512,
                    segments: MIN_WARP_SEGMENTS,
                    nodata_value: None,
                    patches: vec![DrawOverlayPatch {
                        mesh_key: OverlayMeshKey {
                            source: sb_scl,
                            relaid: 17,
                            level: 0,
                            x: 0,
                            y: 0,
                        },
                        cell_rect: [0.0, 0.0, 512.0, 512.0],
                        uv_rect: [0.0, 0.0, 1.0, 1.0],
                        texture: tex_scl_half,
                    }],
                }),
            },
        },
    };
    let mut snap_bm = build_empty_snapshot(cam, width, height, &config);
    snap_bm.overlays.push(single_patch_layer(
        &frame,
        src_base,
        1,
        MIN_WARP_SEGMENTS,
        DrawOverlayTrack::Display,
        tex_base,
    ));
    snap_bm.overlays.push(DrawOverlayLayer {
        layer_id: 9202,
        opacity: 1.0,
        content: DrawOverlayContent::TemporalCrossFade(Box::new(cf_bm)),
    });
    let snap_bm = Arc::new(snap_bm);
    let pixels_bm = render_srgb(&mut globe, device, queue, width, height, &snap_bm);
    assert_eq!(
        globe.renderer.scratch_vram_bytes(),
        (width * height * 17) as usize,
        "BandMathRgb cross-fade with SCL must allocate exact 3x R32Float + 1x R8Uint + 1x Rgba8 (17 B/px)"
    );
    let bm_right = pixel_at(&pixels_bm, width, width / 2 + 20, height / 2);
    assert!(
        bm_right[0] > 160 && bm_right[1] < 5 && bm_right[2] > 160,
        "BandMathRgb cross-fade at t=0.5 must have ZERO green bleed-through, got {bm_right:?}"
    );
    let bm_left = pixel_at(&pixels_bm, width, width / 2 - 20, height / 2);
    assert!(
        bm_left[0] > 140 && bm_left[1] > 140 && bm_left[2] < 15,
        "BandMathRgb cross-fade where Epoch B is SCL-masked must fade Epoch A over green base, got {bm_left:?}"
    );

    // 3. Swipe scissor clipping on TemporalCrossFade (right half only -> left half reveals pure green base).
    let pixels_swiped = render_with(
        &mut globe,
        device,
        queue,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        RenderParams {
            width,
            height,
            snapshot: &snap_bm,
            swipe: Some(SwipeRenderState {
                split_t: 0.5,
                left_layer: None,
                right_layer: Some(9202),
            }),
        },
        true,
    );
    let swiped_left = pixel_at(&pixels_swiped, width, width / 4, height / 2);
    let swiped_right = pixel_at(&pixels_swiped, width, (width * 3) / 4, height / 2);
    assert!(
        swiped_left[0] < 20 && swiped_left[1] > 235 && swiped_left[2] < 20,
        "swipe left side must reveal pure green base, got {swiped_left:?}"
    );
    assert!(
        swiped_right[0] > 160 && swiped_right[1] < 5 && swiped_right[2] > 160,
        "swipe right side must keep zero-bleed-through BandMathRgb cross-fade, got {swiped_right:?}"
    );
}
