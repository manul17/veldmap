//! Headless offscreen `wgpu` smoke and accuracy tests verifying:
//! 1. Exact 1:1 sRGB/UNORM compositing parity via `view_formats` and `0` idle
//!    re-renders when camera/snapshot are unchanged.
//! 2. Post-overlay procedural `fwidth()` graticule toggle, back-hemisphere
//!    horizon discard, and zoomed-in polar meridian sharpness (`89.96° N`).
//! 3. Dropped-encoder recovery and `0x0` viewport safety.
//! 4. `PendingUploads` coarsest-first budgeted uploads (`>= 1` anti-starvation,
//!    duplicate-upload skipping), `GpuTileCache` VRAM byte-cap eviction +
//!    automatic `Weak<GpuTile>` bind-group sweeping, and real NASA Blue Marble
//!    bundled basemap rendering.
//! 5. Close-up (`2 m`) between-vertices accuracy: cell-midpoint sag, sub-metre
//!    per-tile RTC shift, coarse parent texture overzoom (`z_geom >> max_tex`),
//!    genuine multi-LOD seam continuity (`max_l > min_l`), `WebMercator`
//!    `(25° N, -135° W)` rendering, and adaptive Polar Stereographic caps.
//!
//! Set `VELDMAP_REQUIRE_GPU_TEST=1` (enabled on Linux Lavapipe and Windows WARP
//! CI) to fail loudly if no `Backends::PRIMARY` adapter is available.

#![expect(
    clippy::expect_used,
    reason = "INVARIANT: integration test helpers panic on headless GPU harness failure"
)]

use std::collections::HashSet;
use std::sync::{OnceLock, mpsc};
use std::time::Duration;

use veldmap_app::bundled_basemap::BUNDLED_BASEMAP_JPEG;
use veldmap_app::gpu_tiles::{
    GpuTileCache, PendingUploads, UploadPassParams, bundled_tile_key, source_tile_key,
    texture_byte_size,
};
use veldmap_app::render::{DrawSnapshot, GlobeRenderer, RenderParams};
use veldmap_core::geo::camera::{Camera, MIN_HEIGHT_M};
use veldmap_core::geo::projection::WEB_MERCATOR_LIMIT_DEG;
use veldmap_core::tile::{
    DecodedTile, PixelFormat, QuadTileId, QuadtreeConfig, SourceId, TileScheme,
};
use veldmap_raster::{BundledBasemap, generate_linear_mipmaps};

fn shared_headless_device() -> Option<&'static (wgpu::Device, wgpu::Queue)> {
    static DEVICE: OnceLock<Option<(wgpu::Device, wgpu::Queue)>> = OnceLock::new();
    DEVICE.get_or_init(request_headless_device).as_ref()
}

fn shared_bundled_basemap() -> &'static BundledBasemap {
    static BASEMAP: OnceLock<BundledBasemap> = OnceLock::new();
    BASEMAP.get_or_init(|| {
        BundledBasemap::from_encoded_bytes(BUNDLED_BASEMAP_JPEG)
            .expect("decode bundled NASA Blue Marble")
    })
}

const NATIVE_PRIMARY_BACKENDS: wgpu::Backends =
    wgpu::Backends::PRIMARY.difference(wgpu::Backends::BROWSER_WEBGPU);

fn request_headless_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
    desc.backends &= NATIVE_PRIMARY_BACKENDS;
    if desc.backends.is_empty() {
        desc.backends = NATIVE_PRIMARY_BACKENDS;
    }
    let instance = wgpu::Instance::new(desc);

    let adapter = pollster::block_on(async {
        if let Ok(adapter) = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
                ..Default::default()
            })
            .await
        {
            return Some(adapter);
        }

        instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::None,
                compatible_surface: None,
                force_fallback_adapter: true,
                ..Default::default()
            })
            .await
            .ok()
    });

    let Some(adapter) = adapter else {
        if std::env::var("VELDMAP_REQUIRE_GPU_TEST").as_deref() == Ok("1") {
            panic!("VELDMAP_REQUIRE_GPU_TEST=1 is set, but no Backends::PRIMARY adapter was found");
        }
        eprintln!("Skipping headless GPU smoke test: no Backends::PRIMARY adapter available");
        return None;
    };

    let info = adapter.get_info();
    eprintln!(
        "Running headless GPU smoke test on {} ({:?}, {:?})",
        info.name, info.backend, info.device_type
    );

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("headless smoke test device"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_defaults().using_resolution(adapter.limits()),
        memory_hints: wgpu::MemoryHints::default(),
        trace: wgpu::Trace::Off,
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
    }))
    .expect("create headless wgpu device");

    Some((device, queue))
}

fn readback_rgba8(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
) -> Vec<u8> {
    let bytes_per_pixel = 4u32;
    let unpadded_bytes_per_row = width * bytes_per_pixel;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(align) * align;

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback staging buffer"),
        size: u64::from(padded_bytes_per_row) * u64::from(height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("readback copy encoder"),
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll device");
    rx.recv()
        .expect("map_async callback")
        .expect("buffer mapping succeeded");

    let mapped = slice.get_mapped_range().expect("get mapped range");
    let mut pixels = Vec::with_capacity((width * height * bytes_per_pixel) as usize);
    for row in 0..height as usize {
        let start = row * padded_bytes_per_row as usize;
        let end = start + unpadded_bytes_per_row as usize;
        pixels.extend_from_slice(&mapped[start..end]);
    }
    drop(mapped);
    staging.unmap();
    pixels
}

fn render_with(
    renderer: &mut GlobeRenderer,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    target_format: wgpu::TextureFormat,
    params: RenderParams<'_>,
    expect_rendered: bool,
) -> Vec<u8> {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("smoke render encoder"),
    });

    let rendered = renderer.render_if_dirty(device, queue, &mut encoder, params);
    assert_eq!(rendered, expect_rendered);

    let composite_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("composite target"),
        size: wgpu::Extent3d {
            width: params.width,
            height: params.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: target_format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let composite_view = composite_tex.create_view(&wgpu::TextureViewDescriptor::default());

    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("composite pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &composite_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        renderer.composite_into(&mut pass);
    }

    queue.submit(Some(encoder.finish()));
    readback_rgba8(device, queue, &composite_tex, params.width, params.height)
}

fn render_srgb(
    renderer: &mut GlobeRenderer,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    width: u32,
    height: u32,
    snapshot: &DrawSnapshot,
) -> Vec<u8> {
    render_with(
        renderer,
        device,
        queue,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        RenderParams {
            width,
            height,
            snapshot,
        },
        true,
    )
}

fn pixel_at(pixels: &[u8], width: u32, x: u32, y: u32) -> [u8; 4] {
    let idx = ((y * width + x) * 4) as usize;
    [
        pixels[idx],
        pixels[idx + 1],
        pixels[idx + 2],
        pixels[idx + 3],
    ]
}

fn is_clear_background(px: [u8; 4]) -> bool {
    px[0].abs_diff(11) <= 2 && px[1].abs_diff(19) <= 2 && px[2].abs_diff(32) <= 2
}

#[test]
fn srgb_and_unorm_compositing_parity_and_zero_idle_rerender() {
    let Some((device, queue)) = shared_headless_device() else {
        return;
    };

    let (w, h) = (129, 129);
    let cfg = QuadtreeConfig::bundled_geodetic();
    let orbital_cam = Camera::default();
    let snap = DrawSnapshot::build(
        orbital_cam,
        w,
        h,
        1.0,
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
    // Identical snapshot must execute 0 idle offscreen re-renders.
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
    let cfg = QuadtreeConfig::bundled_geodetic();
    let mut renderer = GlobeRenderer::new(device, wgpu::TextureFormat::Rgba8UnormSrgb);

    let orbital_cam = Camera::default();
    let with_grid_snap = DrawSnapshot::build(
        orbital_cam,
        w,
        h,
        1.0,
        true,
        false,
        SourceId::BUNDLED_BASEMAP,
        1,
        &cfg,
        None,
    );
    let no_grid_snap = DrawSnapshot::build(
        orbital_cam,
        w,
        h,
        1.0,
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
    let s_on = DrawSnapshot::build(
        cam,
        w,
        h,
        1.0,
        true,
        false,
        SourceId::BUNDLED_BASEMAP,
        2,
        &cfg,
        None,
    );
    let s_off = DrawSnapshot::build(
        cam,
        w,
        h,
        1.0,
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
    let polar_on = DrawSnapshot::build(
        polar_cam,
        129,
        129,
        1.0,
        true,
        false,
        SourceId::BUNDLED_BASEMAP,
        3,
        &cfg,
        None,
    );
    let polar_off = DrawSnapshot::build(
        polar_cam,
        129,
        129,
        1.0,
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

    let cfg = QuadtreeConfig::bundled_geodetic();
    let mut renderer = GlobeRenderer::new(device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let cam = Camera::at(37.5, 52.5, 14_000_000.0);
    let empty_snap = DrawSnapshot::build(
        cam,
        0,
        0,
        1.0,
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

    let snap = DrawSnapshot::build(
        cam,
        65,
        65,
        1.0,
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

#[test]
fn pending_uploads_budget_and_end_to_end_weak_bind_group_sweeping() {
    let Some((device, queue)) = shared_headless_device() else {
        return;
    };

    let basemap = shared_bundled_basemap();
    let cfg = QuadtreeConfig::bundled_geodetic();

    let mut pending = PendingUploads::new();
    let tx = pending.sender().expect("pending sender");

    let fine_id = QuadTileId {
        scheme: TileScheme::Geodetic,
        level: 2,
        x: 4,
        y: 1,
    };
    let root_west_id = QuadTileId {
        scheme: TileScheme::Geodetic,
        level: 0,
        x: 0,
        y: 0,
    };
    let root_east_id = QuadTileId {
        scheme: TileScheme::Geodetic,
        level: 0,
        x: 1,
        y: 0,
    };
    let south_cap_id = QuadTileId {
        scheme: TileScheme::PolarCap { north: false },
        level: 0,
        x: 0,
        y: 0,
    };

    // Send finer tile BEFORE coarser roots to verify coarsest-first priority.
    for id in [fine_id, root_west_id, root_east_id, south_cap_id] {
        tx.send((
            bundled_tile_key(id),
            basemap.render_tile_sized(id, 64).expect("render tile"),
        ))
        .expect("send tile");
    }

    // Cap cache to exactly 2 tiles (32 KiB) so uploading 4 tiles triggers real LRU eviction.
    let tile_bytes = 64 * 64 * 4;
    let mut cache = GpuTileCache::new(tile_bytes * 2);
    let mut renderer = GlobeRenderer::new(device, wgpu::TextureFormat::Rgba8UnormSrgb);

    // Budget of 1 byte must STILL upload 1 coarsest tile (anti-starvation).
    let r1 = pending.drain_and_upload(
        device,
        queue,
        &mut cache,
        UploadPassParams {
            pinned_root_sources: &[SourceId::BUNDLED_BASEMAP],
            observed_source: Some(SourceId::BUNDLED_BASEMAP),
            wanted: None,
            protected: None,
            byte_budget: 1,
            time_budget: Duration::from_millis(0),
        },
    );
    assert_eq!(r1.uploaded, 1);
    assert_eq!(r1.remaining, 3);
    assert!(
        !cache.contains(&bundled_tile_key(fine_id)),
        "coarsest z=0 must upload before z=2"
    );

    // Render using the first uploaded tile so GlobeRenderer caches its BindGroup.
    {
        let snap1 = DrawSnapshot::build(
            Camera::at(0.0, -90.0, 4_000_000.0),
            65,
            65,
            1.0,
            false,
            true,
            SourceId::BUNDLED_BASEMAP,
            1,
            &cfg,
            Some(&mut cache),
        );
        let _ = render_srgb(&mut renderer, device, queue, 65, 65, &snap1);
        assert_eq!(renderer.cached_tile_bind_group_count(), 1);
    }

    // Upload the remaining 3 tiles -> evicts earlier tiles from GpuTileCache.
    let r2 = pending.drain_and_upload(
        device,
        queue,
        &mut cache,
        UploadPassParams {
            pinned_root_sources: &[SourceId::BUNDLED_BASEMAP],
            observed_source: Some(SourceId::BUNDLED_BASEMAP),
            wanted: None,
            protected: None,
            byte_budget: 1024 * 1024,
            time_budget: Duration::from_secs(1),
        },
    );
    assert_eq!(r2.uploaded, 3);
    assert_eq!(r2.remaining, 0);
    assert!(r2.evicted >= 2);
    assert!(cache.total_bytes() <= cache.max_bytes());

    // Re-sending an already-cached key must be skipped (0 duplicate GPU uploads).
    let cached_id = if cache.contains(&bundled_tile_key(fine_id)) {
        fine_id
    } else {
        south_cap_id
    };
    tx.send((
        bundled_tile_key(cached_id),
        basemap.render_tile_sized(cached_id, 64).expect("dup tile"),
    ))
    .expect("send dup");
    let r_dup = pending.drain_and_upload(
        device,
        queue,
        &mut cache,
        UploadPassParams {
            pinned_root_sources: &[SourceId::BUNDLED_BASEMAP],
            observed_source: Some(SourceId::BUNDLED_BASEMAP),
            wanted: None,
            protected: None,
            byte_budget: 1024 * 1024,
            time_budget: Duration::from_secs(1),
        },
    );
    assert_eq!(r_dup.uploaded, 0);

    // Sweeping stale Weak<GpuTile> entries drops the evicted tile's BindGroup automatically.
    renderer.sweep_stale_bind_groups();
    assert_eq!(
        renderer.cached_tile_bind_group_count(),
        0,
        "evicted tile BindGroup was not swept via Weak<GpuTile>"
    );
}

#[test]
fn bundled_blue_marble_renders_sahara_pacific_and_antarctica() {
    let Some((device, queue)) = shared_headless_device() else {
        return;
    };

    let basemap = shared_bundled_basemap();
    let cfg = QuadtreeConfig::bundled_geodetic();
    let mut renderer = GlobeRenderer::new(device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut cache = GpuTileCache::new(32 * 1024 * 1024);

    for id in [
        QuadTileId {
            scheme: TileScheme::Geodetic,
            level: 0,
            x: 0,
            y: 0,
        },
        QuadTileId {
            scheme: TileScheme::Geodetic,
            level: 0,
            x: 1,
            y: 0,
        },
        QuadTileId {
            scheme: TileScheme::PolarCap { north: true },
            level: 0,
            x: 0,
            y: 0,
        },
        QuadTileId {
            scheme: TileScheme::PolarCap { north: false },
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
    let sahara_snap = DrawSnapshot::build(
        Camera::at(23.0, 13.0, 2_000_000.0),
        w,
        h,
        1.0,
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
    let pacific_snap = DrawSnapshot::build(
        Camera::at(0.0, -150.0, 2_000_000.0),
        w,
        h,
        1.0,
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
    let south_snap = DrawSnapshot::build(
        Camera::at(-89.5, 0.0, 500_000.0),
        w,
        h,
        1.0,
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
    let cfg = QuadtreeConfig::bundled_geodetic();
    let mut renderer = GlobeRenderer::new(device, wgpu::TextureFormat::Rgba8UnormSrgb);

    // 1. Midpoint between-vertices sag clearance at 2 m altitude.
    let between_cam = Camera::at(45.000_73, 45.000_73, MIN_HEIGHT_M);
    let between_snap = DrawSnapshot::build(
        between_cam,
        w,
        h,
        1.0,
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
    let s0 = DrawSnapshot::build(
        Camera::at(45.0, 45.0, MIN_HEIGHT_M),
        w,
        h,
        1.0,
        true,
        false,
        SourceId::BUNDLED_BASEMAP,
        2,
        &cfg,
        None,
    );
    let s1 = DrawSnapshot::build(
        Camera::at(45.0, 45.0 + dlon_25cm, MIN_HEIGHT_M),
        w,
        h,
        1.0,
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
    let synthetic_east = DecodedTile::new(64, 64, PixelFormat::Rgba8UnormSrgb, quad_data);
    let mut overzoom_cache = GpuTileCache::new(1024 * 1024);
    let east_root = bundled_tile_key(QuadTileId {
        scheme: TileScheme::Geodetic,
        level: 0,
        x: 1,
        y: 0,
    });
    overzoom_cache
        .upload_and_insert(device, queue, east_root, &synthetic_east)
        .expect("upload synthetic root");

    // (45 N, 45 E) sits at the center of the NW quadrant (RED) of east root (x=1, y=0).
    let overzoom_snap = DrawSnapshot::build(
        Camera::at(45.0, 45.0, MIN_HEIGHT_M),
        w,
        h,
        1.0,
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
    let pc_on = DrawSnapshot::build(
        Camera::at(86.0, 0.0, MIN_HEIGHT_M),
        w,
        h,
        1.0,
        true,
        false,
        SourceId::BUNDLED_BASEMAP,
        6,
        &cfg,
        None,
    );
    let pc_lat_shifted = DrawSnapshot::build(
        Camera::at(86.0 + dlat_25cm, 0.0, MIN_HEIGHT_M),
        w,
        h,
        1.0,
        true,
        false,
        SourceId::BUNDLED_BASEMAP,
        7,
        &cfg,
        None,
    );
    let pc_lon_shifted = DrawSnapshot::build(
        Camera::at(86.0, dlon_25cm_86n, MIN_HEIGHT_M),
        w,
        h,
        1.0,
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

    let cfg = QuadtreeConfig::bundled_geodetic();
    let mut renderer = GlobeRenderer::new(device, wgpu::TextureFormat::Rgba8UnormSrgb);

    // 1. Genuine multi-LOD transition: assert `max_l > min_l` strictly and zero cracks.
    let (mw, mh) = (193, 193);
    let multi_lod_cam = Camera::at(45.0, 39.0, 3_500_000.0);
    let multi_lod_snap = DrawSnapshot::build(
        multi_lod_cam,
        mw,
        mh,
        1.0,
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
        let seam_snap = DrawSnapshot::build(
            cam,
            w,
            h,
            1.0,
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

#[test]
fn live_source_mipmaps_polar_fallback_and_active_frame_protection() {
    let Some((device, queue)) = shared_headless_device() else {
        return;
    };

    let live_source = SourceId::from_stable_key("eox-s2cloudless-test");
    let merc_cfg = QuadtreeConfig::web_mercator(14);
    let mut renderer = GlobeRenderer::new(device, wgpu::TextureFormat::Rgba8UnormSrgb);

    // 1. CPU linear-light mipmaps upload into GpuTileCache with exact mip byte accounting.
    let mut bundled_wm = DecodedTile::new(
        64,
        64,
        PixelFormat::Rgba8UnormSrgb,
        [40, 80, 210, 255].repeat(64 * 64),
    );
    generate_linear_mipmaps(&mut bundled_wm);
    assert_eq!(bundled_wm.mip_level_count(), 7);
    let expected_bytes = texture_byte_size(64, 64, PixelFormat::Rgba8UnormSrgb, 7);
    assert_eq!(bundled_wm.byte_size(), expected_bytes);

    let mut bundled_polar = DecodedTile::new(
        64,
        64,
        PixelFormat::Rgba8UnormSrgb,
        [235, 235, 235, 255].repeat(64 * 64),
    );
    generate_linear_mipmaps(&mut bundled_polar);

    let mut live_tile = DecodedTile::new(
        64,
        64,
        PixelFormat::Rgba8UnormSrgb,
        [40, 200, 70, 255].repeat(64 * 64),
    );
    generate_linear_mipmaps(&mut live_tile);

    // Budget holds exactly 2 mipmapped 64x64 tiles so Step 4 genuinely
    // exercises active-frame eviction protection and refusal deferral.
    let mut cache = GpuTileCache::new(expected_bytes * 2);
    let wm_bundled_root = bundled_tile_key(QuadTileId {
        scheme: TileScheme::WebMercator,
        level: 0,
        x: 0,
        y: 0,
    });
    let south_polar_root = bundled_tile_key(QuadTileId {
        scheme: TileScheme::PolarCap { north: false },
        level: 0,
        x: 0,
        y: 0,
    });
    cache
        .upload_and_insert(device, queue, wm_bundled_root, &bundled_wm)
        .expect("upload bundled wm fallback");
    cache
        .upload_and_insert(device, queue, south_polar_root, &bundled_polar)
        .expect("upload bundled polar cap");
    assert_eq!(cache.total_bytes(), expected_bytes * 2);

    // 2. While live tiles have not arrived yet, DrawSnapshot parameterized by
    //    `live_source` falls back cleanly to bundled WebMercator blue.
    let (w, h) = (65, 65);
    let fallback_snap = DrawSnapshot::build(
        Camera::at(0.0, 0.0, 2_000_000.0),
        w,
        h,
        1.0,
        false,
        true,
        live_source,
        1,
        &merc_cfg,
        Some(&mut cache),
    );
    assert!(fallback_snap.carrier_keys.contains(&wm_bundled_root));
    let fb_px = pixel_at(
        &render_srgb(&mut renderer, device, queue, w, h, &fallback_snap),
        w,
        w / 2,
        h / 2,
    );
    assert!(
        fb_px[2] > fb_px[1] + 50,
        "expected blue bundled offline fallback, got {fb_px:?}"
    );

    // 3. PolarCap under active `live_source` ALWAYS resolves to bundled polar
    //    white (never live_source). This touches `south_polar_root`, making
    //    `wm_bundled_root` the LRU tail.
    let polar_snap = DrawSnapshot::build(
        Camera::at(-89.5, 0.0, 400_000.0),
        w,
        h,
        1.0,
        false,
        true,
        live_source,
        2,
        &merc_cfg,
        Some(&mut cache),
    );
    assert!(polar_snap.carrier_keys.contains(&south_polar_root));
    let polar_px = pixel_at(
        &render_srgb(&mut renderer, device, queue, w, h, &polar_snap),
        w,
        w / 2,
        h / 2,
    );
    assert!(
        polar_px[0] > 180 && polar_px[1] > 180 && polar_px[2] > 180,
        "expected bundled polar white under live source, got {polar_px:?}"
    );

    // 4. Upload live WebMercator root into the FULL 2-tile cache while
    //    `wm_bundled_root` (currently LRU tail!) is protected by `fallback_snap`.
    //    Active-frame protection MUST skip `wm_bundled_root` and evict
    //    `south_polar_root` instead.
    let live_root = source_tile_key(
        live_source,
        QuadTileId {
            scheme: TileScheme::WebMercator,
            level: 0,
            x: 0,
            y: 0,
        },
    );
    let mut pending = PendingUploads::new();
    let tx = pending.sender().expect("sender");
    tx.send((live_root, live_tile.clone())).expect("send live");
    let pinned_live = [live_source, SourceId::BUNDLED_BASEMAP];
    let rep = pending.drain_and_upload(
        device,
        queue,
        &mut cache,
        UploadPassParams {
            pinned_root_sources: &pinned_live,
            observed_source: Some(live_source),
            wanted: Some(&HashSet::from([live_root])),
            protected: Some(&fallback_snap.carrier_keys),
            byte_budget: 1024 * 1024,
            time_budget: Duration::from_secs(1),
        },
    );
    assert_eq!(rep.uploaded, 1);
    assert_eq!(rep.evicted, 1);
    assert!(
        cache.contains(&wm_bundled_root),
        "protected LRU tail must survive"
    );
    assert!(
        !cache.contains(&south_polar_root),
        "unprotected tile must be evicted"
    );
    assert!(cache.contains(&live_root));

    // 4b. When ALL cached tiles are protected, an incoming tile is deferred
    //     (NOT silently discarded) with `remaining_ready == 0, deferred_protected == 1`,
    //     and uploads cleanly once protection lifts without re-sending.
    let all_protected = HashSet::from([wm_bundled_root, live_root]);
    tx.send((south_polar_root, bundled_polar))
        .expect("re-send polar cap");
    let deferred_rep = pending.drain_and_upload(
        device,
        queue,
        &mut cache,
        UploadPassParams {
            pinned_root_sources: &pinned_live,
            observed_source: Some(live_source),
            wanted: Some(&HashSet::from([south_polar_root])),
            protected: Some(&all_protected),
            byte_budget: 1024 * 1024,
            time_budget: Duration::from_secs(1),
        },
    );
    assert_eq!(deferred_rep.uploaded, 0);
    assert_eq!(deferred_rep.remaining_ready, 0);
    assert_eq!(deferred_rep.deferred_protected, 1);
    assert!(
        pending.contains(&south_polar_root),
        "protected-refused tile must stay deferred in queue"
    );

    let live_snap = DrawSnapshot::build(
        Camera::at(0.0, 0.0, 2_000_000.0),
        w,
        h,
        1.0,
        false,
        true,
        live_source,
        3,
        &merc_cfg,
        Some(&mut cache),
    );
    let live_px = pixel_at(
        &render_srgb(&mut renderer, device, queue, w, h, &live_snap),
        w,
        w / 2,
        h / 2,
    );
    assert!(
        live_px[1] > live_px[2] + 50,
        "expected green live tile override, got {live_px:?}"
    );

    // Now `live_snap` protects only `live_root` (unprotecting `wm_bundled_root`),
    // so the deferred `south_polar_root` uploads immediately without re-sending.
    let resumed_rep = pending.drain_and_upload(
        device,
        queue,
        &mut cache,
        UploadPassParams {
            pinned_root_sources: &pinned_live,
            observed_source: Some(live_source),
            wanted: Some(&HashSet::from([south_polar_root])),
            protected: Some(&live_snap.carrier_keys),
            byte_budget: 1024 * 1024,
            time_budget: Duration::from_secs(1),
        },
    );
    assert_eq!(resumed_rep.uploaded, 1);
    assert_eq!(resumed_rep.evicted, 1);
    assert!(cache.contains(&south_polar_root));
    assert!(cache.contains(&live_root));

    // 5. Removing `live_source` clears only live tiles and preserves bundled fallback.
    assert_eq!(cache.remove_source(live_source), 1);
    assert!(!cache.contains(&live_root));
    assert!(cache.contains(&south_polar_root));
}
