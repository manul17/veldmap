//! Headless GPU smoke tests for Slice 3A-1b post-body depth-less raster
//! overlays: warp-mesh draping, negative-determinant south-up outward winding,
//! Pass 2 graticule-on-top ordering, exact CPU `f64` horizon-plane limb
//! clipping, `Weak<GpuTile>` bind-group sweeping, and `0` idle GPU re-renders.

use std::sync::Arc;

use veldmap_app::gpu_tiles::GpuTileCache;
use veldmap_app::render::{
    DrawOverlayLayer, DrawOverlayPatch, DrawSnapshot, GlobeRenderer, OverlayMeshKey, RenderParams,
    SnapshotBuildParams,
};
use veldmap_core::geo::camera::Camera;
use veldmap_core::geo::overlay::{Frame, build_warp_patch};
use veldmap_core::geo::projection::Projection;
use veldmap_core::tile::{
    BandKey, DecodedTile, PixelFormat, QuadtreeConfig, SourceId, TileKey, TileScheme,
};

use crate::common::{
    is_clear_background, pixel_at, render_srgb, render_with, shared_headless_device,
};

fn upload_solid_overlay_tile(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    cache: &mut GpuTileCache,
    source: SourceId,
    rgba: [u8; 4],
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
    let (w, h) = (32u32, 32u32);
    let mut data = Vec::with_capacity((w * h * 4) as usize);
    for _ in 0..(w * h) {
        data.extend_from_slice(&rgba);
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
    let north_up = Frame::from_pixel_affine(
        utm37,
        [100.0, 0.0, 500_000.0, 0.0, -100.0, 5_250_000.0],
        512,
        512,
    )
    .expect("north-up");
    // Negative-determinant (south-up) affine covering the exact same ground box.
    let south_up = Frame::from_pixel_affine(
        utm37,
        [100.0, 0.0, 500_000.0, 0.0, 100.0, 5_198_800.0],
        512,
        512,
    )
    .expect("south-up");

    let (center_lat, center_lon) = north_up.geodetic(0.5, 0.5);
    let cam = Camera::at(center_lat, center_lon, 30_000.0);
    let config = QuadtreeConfig::default();

    for (case_idx, (frame, expected_flip)) in [(&north_up, false), (&south_up, true)]
        .into_iter()
        .enumerate()
    {
        let src = SourceId::new(900 + case_idx as u64);
        let (tile_key, gpu_tile) =
            upload_solid_overlay_tile(device, queue, &mut gpu_cache, src, [255, 0, 255, 255]);
        let warp = build_warp_patch(frame, 512, 512, [0.0, 0.0, 512.0, 512.0]);
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
            patches: vec![DrawOverlayPatch {
                mesh_key: OverlayMeshKey {
                    source: src,
                    relaid: 1,
                    level: 0,
                    x: 0,
                    y: 0,
                },
                origin_ecef_m: warp.origin_ecef_m,
                segments: warp.segments,
                flip_winding: warp.flip_winding,
                rel_positions: Arc::from(warp.rel_positions),
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

        // Enable graticule in Pass 2 on top of the magenta overlay: graticule
        // lines must remain visible (non-magenta green-tinted pixels present).
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

        // Drop snapshots and evict tile: `Weak<GpuTile>` bind-group sweeping frees it.
        drop(snap);
        drop(snap_grid);
        drop(gpu_tile);
        gpu_cache.remove_source(src);
        renderer.sweep_stale_bind_groups();
        assert_eq!(renderer.cached_tile_bind_group_count(), 0);
    }
}

#[test]
fn overlay_exact_horizon_plane_clips_coarse_limb_chords_without_back_hemisphere_leak() {
    let Some((device, queue)) = shared_headless_device() else {
        return;
    };

    let (width, height) = (160u32, 160u32);
    let mut renderer = GlobeRenderer::new(device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut gpu_cache = GpuTileCache::new(16 * 1024 * 1024);

    // Camera over (0N, 0E) at 12,000 km altitude: the full globe and eastern
    // limb (~69.7 deg from nadir) sit inside the 50-degree viewport. Place a
    // quad overlay spanning 20E..120E across the eastern limb into the back
    // hemisphere so coarse triangles straddle the horizon plane.
    let cam = Camera::at(0.0, 0.0, 12_000_000.0);
    let frame = Frame::quad([(30.0, 20.0), (30.0, 120.0), (-30.0, 120.0), (-30.0, 20.0)]);
    let src = SourceId::new(950);
    let (tile_key, gpu_tile) =
        upload_solid_overlay_tile(device, queue, &mut gpu_cache, src, [255, 0, 255, 255]);
    let warp = build_warp_patch(&frame, 512, 512, [0.0, 0.0, 512.0, 512.0]);

    let config = QuadtreeConfig::default();
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
        patches: vec![DrawOverlayPatch {
            mesh_key: OverlayMeshKey {
                source: src,
                relaid: 1,
                level: 0,
                x: 0,
                y: 0,
            },
            origin_ecef_m: warp.origin_ecef_m,
            segments: warp.segments,
            flip_winding: warp.flip_winding,
            rel_positions: Arc::from(warp.rel_positions),
            uv_rect: [0.0, 0.0, 1.0, 1.0],
            texture: gpu_tile,
        }],
    });
    let snap = Arc::new(snap);

    let pixels = render_srgb(&mut renderer, device, queue, width, height, &snap);

    // Front-hemisphere portion of the overlay must render magenta.
    let mut magenta_count = 0usize;
    for y in 0..height {
        for x in 0..width {
            let px = pixel_at(&pixels, width, x, y);
            if px[0] > 200 && px[1] < 40 && px[2] > 200 {
                magenta_count += 1;
            }
        }
    }
    assert!(
        magenta_count > 50,
        "front-hemisphere portion of limb-straddling overlay must render"
    );

    // Corner pixels outside the globe silhouette must remain pure space
    // background (zero back-hemisphere chord leak past the horizon).
    for (x, y) in [
        (0, 0),
        (width - 1, 0),
        (0, height - 1),
        (width - 1, height - 1),
    ] {
        let px = pixel_at(&pixels, width, x, y);
        assert!(
            is_clear_background(px),
            "space corner ({x}, {y}) leaked overlay fragment: {px:?}"
        );
    }

    // Also verify a patch lying entirely on the back hemisphere emits ZERO
    // magenta pixels anywhere in the viewport.
    let back_frame = Frame::quad([(30.0, 120.0), (30.0, 170.0), (-30.0, 170.0), (-30.0, 120.0)]);
    let back_warp = build_warp_patch(&back_frame, 512, 512, [0.0, 0.0, 512.0, 512.0]);
    let mut back_snap = (*snap).clone();
    back_snap.overlays[0].patches[0].mesh_key.relaid = 2;
    back_snap.overlays[0].patches[0].origin_ecef_m = back_warp.origin_ecef_m;
    back_snap.overlays[0].patches[0].segments = back_warp.segments;
    back_snap.overlays[0].patches[0].flip_winding = back_warp.flip_winding;
    back_snap.overlays[0].patches[0].rel_positions = Arc::from(back_warp.rel_positions);
    let back_snap = Arc::new(back_snap);

    let back_pixels = render_srgb(&mut renderer, device, queue, width, height, &back_snap);
    let back_magenta = back_pixels
        .chunks_exact(4)
        .filter(|px| px[0] > 200 && px[1] < 40 && px[2] > 200)
        .count();
    assert_eq!(
        back_magenta, 0,
        "back-hemisphere overlay patch must be completely clipped by exact horizon plane"
    );
}
