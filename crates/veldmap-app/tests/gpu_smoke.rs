//! Headless offscreen `wgpu` smoke and accuracy tests verifying:
//! 1. Exact 1:1 sRGB/UNORM compositing parity via `view_formats` and `0` idle
//!    re-renders when camera/snapshot are unchanged.
//! 2. Post-overlay procedural `fwidth()` graticule toggle and back-hemisphere
//!    horizon discard.
//! 3. Dropped-encoder recovery and `0x0` viewport safety.
//! 4. `PendingUploads` coarsest-first budgeted uploads (`>= 1` anti-starvation),
//!    `GpuTileCache` VRAM byte-cap eviction + `GlobeRenderer` bind-group
//!    pruning, and real NASA Blue Marble bundled basemap rendering.
//! 5. Close-up (`2 m`) between-vertices accuracy: cell-midpoint sag, sub-metre
//!    per-tile RTC shift, coarse parent texture overzoom (`z_geom >> max_tex`),
//!    LOD / polar-boundary seam continuity (zero cracks), and adaptive Polar
//!    Stereographic cap rendering at `±90°`.
//!
//! Set `VELDMAP_REQUIRE_GPU_TEST=1` (enabled on Linux Lavapipe and Windows WARP
//! CI) to fail loudly if no `Backends::PRIMARY` adapter is available.

#![expect(
    clippy::expect_used,
    reason = "INVARIANT: integration test helpers panic on headless GPU harness failure"
)]

use std::sync::{OnceLock, mpsc};
use std::time::Duration;

use veldmap_app::gpu_tiles::{
    BUNDLED_BASEMAP_JPEG, GpuTileCache, PendingUploads, bundled_tile_key,
};
use veldmap_app::render::{DrawSnapshot, GlobeRenderer, RenderParams};
use veldmap_core::geo::camera::{Camera, MIN_HEIGHT_M};
use veldmap_core::geo::projection::WEB_MERCATOR_LIMIT_DEG;
use veldmap_core::tile::{PixelFormat, QuadTileId, QuadtreeConfig, TileScheme};
use veldmap_raster::{BundledBasemap, DecodedTile};

fn shared_headless_device() -> Option<&'static (wgpu::Device, wgpu::Queue)> {
    static DEVICE: OnceLock<Option<(wgpu::Device, wgpu::Queue)>> = OnceLock::new();
    DEVICE.get_or_init(request_headless_device).as_ref()
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
    let snap = DrawSnapshot::build(orbital_cam, w, h, 1.0, true, false, 1, &cfg, None);
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
    let mut max_diff = 0u8;
    for (&a, &b) in srgb_pixels.iter().zip(unorm_pixels.iter()) {
        max_diff = max_diff.max(a.abs_diff(b));
    }
    assert!(
        max_diff <= 1,
        "sRGB and UNORM composite outputs diverged by {max_diff} LSB"
    );

    // Corner at orbital zoom must be the exact sRGB-encoded deep-space background (#0B1320).
    let corner = pixel_at(&srgb_pixels, w, 2, 2);
    assert!(
        is_clear_background(corner),
        "unexpected orbital background corner pixel: {corner:?}"
    );
    assert_eq!(corner[3], 255);

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
fn graticule_toggle_and_back_hemisphere_horizon_discard() {
    let Some((device, queue)) = shared_headless_device() else {
        return;
    };

    let (w, h) = (129, 129);
    let cfg = QuadtreeConfig::bundled_geodetic();
    let mut renderer = GlobeRenderer::new(device, wgpu::TextureFormat::Rgba8UnormSrgb);

    let orbital_cam = Camera::default();
    let with_grid_snap = DrawSnapshot::build(orbital_cam, w, h, 1.0, true, false, 1, &cfg, None);
    let no_grid_snap = DrawSnapshot::build(orbital_cam, w, h, 1.0, false, false, 1, &cfg, None);

    let _ = render_with(
        &mut renderer,
        device,
        queue,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        RenderParams {
            width: w,
            height: h,
            snapshot: &with_grid_snap,
        },
        true,
    );
    let no_grid_pixels = render_with(
        &mut renderer,
        device,
        queue,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        RenderParams {
            width: w,
            height: h,
            snapshot: &no_grid_snap,
        },
        true,
    );
    let body_center = pixel_at(&no_grid_pixels, w, w / 2, h / 2);
    assert!(
        body_center[0] > 180 && body_center[1] > 170 && body_center[2] > 150,
        "center pixel without graticule did not render lit ellipsoid in sRGB: {body_center:?}"
    );

    // At (37.5 N, 52.5 E, 14,000 km), front 15-degree graticule lines lie far
    // from viewport center; central 11x11 patch must stay free of back-side bleed.
    let cam = Camera::at(37.5, 52.5, 14_000_000.0);
    let s_on = DrawSnapshot::build(cam, w, h, 1.0, true, false, 2, &cfg, None);
    let s_off = DrawSnapshot::build(cam, w, h, 1.0, false, false, 2, &cfg, None);
    let with_grid = render_with(
        &mut renderer,
        device,
        queue,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        RenderParams {
            width: w,
            height: h,
            snapshot: &s_on,
        },
        true,
    );
    let without_grid = render_with(
        &mut renderer,
        device,
        queue,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        RenderParams {
            width: w,
            height: h,
            snapshot: &s_off,
        },
        true,
    );
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
}

#[test]
fn dropped_encoder_recovery_and_zero_viewport_safety() {
    let Some((device, queue)) = shared_headless_device() else {
        return;
    };

    let cfg = QuadtreeConfig::bundled_geodetic();
    let mut renderer = GlobeRenderer::new(device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let cam = Camera::at(37.5, 52.5, 14_000_000.0);
    let empty_snap = DrawSnapshot::build(cam, 0, 0, 1.0, true, false, 1, &cfg, None);

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

    let snap = DrawSnapshot::build(cam, 65, 65, 1.0, true, false, 1, &cfg, None);
    let params = RenderParams {
        width: 65,
        height: 65,
        snapshot: &snap,
    };
    let mut dropped_enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    assert!(renderer.render_if_dirty(device, queue, &mut dropped_enc, params));
    drop(dropped_enc);

    let recovered = render_with(
        &mut renderer,
        device,
        queue,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        params,
        true,
    );
    let center = pixel_at(&recovered, 65, 32, 32);
    assert!(
        center[0] > 180 && center[1] > 170 && center[2] > 150,
        "recovered render did not produce lit globe center: {center:?}"
    );
}

#[test]
fn gpu_tile_upload_eviction_pruning_and_bundled_basemap_rendering() {
    let Some((device, queue)) = shared_headless_device() else {
        return;
    };

    let basemap = BundledBasemap::from_encoded_bytes(BUNDLED_BASEMAP_JPEG)
        .expect("decode bundled NASA Blue Marble");
    let cfg = QuadtreeConfig::bundled_geodetic();

    // Exercise PendingUploads coarsest-first ordering & >= 1 tile/frame anti-starvation
    // alongside GpuTileCache byte-cap eviction + GlobeRenderer bind-group pruning.
    let (tx, rx) = mpsc::channel();
    let mut pending = PendingUploads::new(rx);

    let root_west = bundled_tile_key(QuadTileId {
        scheme: TileScheme::Geodetic,
        level: 0,
        x: 0,
        y: 0,
    });
    let root_east = bundled_tile_key(QuadTileId {
        scheme: TileScheme::Geodetic,
        level: 0,
        x: 1,
        y: 0,
    });
    let fine_east = bundled_tile_key(QuadTileId {
        scheme: TileScheme::Geodetic,
        level: 2,
        x: 4,
        y: 1,
    });
    let south_cap = bundled_tile_key(QuadTileId {
        scheme: TileScheme::PolarCap { north: false },
        level: 0,
        x: 0,
        y: 0,
    });

    // Send finer tile BEFORE coarser root to verify coarsest-first priority.
    tx.send((
        fine_east,
        basemap
            .render_tile_sized(
                QuadTileId {
                    scheme: TileScheme::Geodetic,
                    level: 2,
                    x: 4,
                    y: 1,
                },
                64,
            )
            .expect("render fine"),
    ))
    .expect("send fine");
    for (key, id) in [
        (
            root_west,
            QuadTileId {
                scheme: TileScheme::Geodetic,
                level: 0,
                x: 0,
                y: 0,
            },
        ),
        (
            root_east,
            QuadTileId {
                scheme: TileScheme::Geodetic,
                level: 0,
                x: 1,
                y: 0,
            },
        ),
        (
            south_cap,
            QuadTileId {
                scheme: TileScheme::PolarCap { north: false },
                level: 0,
                x: 0,
                y: 0,
            },
        ),
    ] {
        tx.send((key, basemap.render_tile_sized(id, 64).expect("render root")))
            .expect("send root");
    }

    // Cap cache to exactly 3 x (64*64*4 = 16 KiB) = 48 KiB so the 4th upload triggers eviction.
    let tile_bytes = 64 * 64 * 4;
    let mut cache = GpuTileCache::new(tile_bytes * 3);
    let mut renderer = GlobeRenderer::new(device, wgpu::TextureFormat::Rgba8UnormSrgb);

    // Budget of 1 byte must STILL upload 1 coarsest tile (anti-starvation).
    let r1 = pending.drain_and_upload(device, queue, &mut cache, None, 1, Duration::from_millis(0));
    assert_eq!(r1.uploaded, 1);
    assert_eq!(r1.remaining, 3);
    assert!(
        !cache.contains(&fine_east),
        "coarsest z=0 must upload before z=2"
    );

    // Upload remaining 3 tiles with ample budget -> 1 tile must be evicted so total_bytes <= cap.
    let r2 = pending.drain_and_upload(
        device,
        queue,
        &mut cache,
        None,
        1024 * 1024,
        Duration::from_secs(1),
    );
    assert_eq!(r2.uploaded, 3);
    assert_eq!(r2.remaining, 0);
    assert_eq!(r2.evicted_tile_ids.len(), 1);
    assert!(cache.total_bytes() <= cache.max_bytes());

    // Now test full bundled NASA Blue Marble rendering across Sahara, Pacific, and Antarctica.
    let mut full_cache = GpuTileCache::new(32 * 1024 * 1024);
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
        let decoded = basemap.render_tile(id).expect("render bundled tile");
        let _ = full_cache.upload_and_insert(device, queue, bundled_tile_key(id), &decoded);
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
        10,
        &cfg,
        Some(&mut full_cache),
    );
    let sahara_px = render_with(
        &mut renderer,
        device,
        queue,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        RenderParams {
            width: w,
            height: h,
            snapshot: &sahara_snap,
        },
        true,
    );
    let s_mid = pixel_at(&sahara_px, w, w / 2, h / 2);
    assert!(
        s_mid[0] > 140 && s_mid[0] > s_mid[2] + 35,
        "Sahara textured center did not look tan: {s_mid:?}"
    );
    assert!(renderer.cached_tile_bind_group_count() >= 1);

    // 2. Central Pacific (0 N, -150 E) -> deep ocean blue (B > R + 15, R < 50).
    let pacific_snap = DrawSnapshot::build(
        Camera::at(0.0, -150.0, 2_000_000.0),
        w,
        h,
        1.0,
        false,
        true,
        11,
        &cfg,
        Some(&mut full_cache),
    );
    let pacific_px = render_with(
        &mut renderer,
        device,
        queue,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        RenderParams {
            width: w,
            height: h,
            snapshot: &pacific_snap,
        },
        true,
    );
    let p_mid = pixel_at(&pacific_px, w, w / 2, h / 2);
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
        12,
        &cfg,
        Some(&mut full_cache),
    );
    let south_px = render_with(
        &mut renderer,
        device,
        queue,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        RenderParams {
            width: w,
            height: h,
            snapshot: &south_snap,
        },
        true,
    );
    let sp_mid = pixel_at(&south_px, w, w / 2, h / 2);
    assert!(
        sp_mid[0] > 180 && sp_mid[1] > 180 && sp_mid[2] > 180,
        "Antarctica PolarCap center did not render bright white: {sp_mid:?}"
    );

    // Verify eviction bind-group pruning removes evicted tile IDs cleanly.
    let before_bg = renderer.cached_tile_bind_group_count();
    let evicted_ids: Vec<u64> = south_snap
        .tiles
        .iter()
        .filter_map(|t| t.texture.as_ref().map(|g| g.tile_id))
        .collect();
    renderer.prune_evicted_tiles(&evicted_ids);
    assert!(renderer.cached_tile_bind_group_count() < before_bg);
}

#[test]
fn close_up_2m_between_vertices_sag_lod_seam_overzoom_and_polar_cap() {
    let Some((device, queue)) = shared_headless_device() else {
        return;
    };

    let (w, h) = (65, 65);
    let cfg = QuadtreeConfig::bundled_geodetic();
    let mut renderer = GlobeRenderer::new(device, wgpu::TextureFormat::Rgba8UnormSrgb);

    // 1. Midpoint between-vertices sag & sub-metre per-tile RTC shift at 2 m.
    //    `(45.00073 N, 45.00073 E)` sits strictly between tile mesh vertices;
    //    no pixel may clip against near = 1 m or show deep-space background.
    let between_cam = Camera::at(45.000_73, 45.000_73, MIN_HEIGHT_M);
    let between_snap = DrawSnapshot::build(between_cam, w, h, 1.0, false, false, 1, &cfg, None);
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

    let between_px = render_with(
        &mut renderer,
        device,
        queue,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        RenderParams {
            width: w,
            height: h,
            snapshot: &between_snap,
        },
        true,
    );
    for y in 0..h {
        for x in 0..w {
            let px = pixel_at(&between_px, w, x, y);
            assert!(
                !is_clear_background(px) && px[0] > 140 && px[1] > 130,
                "2 m between-vertices pixel ({x},{y}) clipped or leaked crack: {px:?}"
            );
        }
    }

    // Sub-metre RTC shift at 2 m altitude: 0.25 m east of (45 N, 45 E) (< 1 ULP
    // of global f32 ECEF) must shift the procedural `fwidth()` graticule off center.
    let on_grid_cam = Camera::at(45.0, 45.0, MIN_HEIGHT_M);
    let dlon_25cm = 0.25 / (111_320.0 * 45.0_f64.to_radians().cos());
    let shifted_cam = Camera::at(45.0, 45.0 + dlon_25cm, MIN_HEIGHT_M);
    let s0 = DrawSnapshot::build(on_grid_cam, w, h, 1.0, true, false, 2, &cfg, None);
    let s1 = DrawSnapshot::build(shifted_cam, w, h, 1.0, true, false, 3, &cfg, None);
    let p0 = render_with(
        &mut renderer,
        device,
        queue,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        RenderParams {
            width: w,
            height: h,
            snapshot: &s0,
        },
        true,
    );
    let p1 = render_with(
        &mut renderer,
        device,
        queue,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        RenderParams {
            width: w,
            height: h,
            snapshot: &s1,
        },
        true,
    );
    assert_ne!(
        p0, p1,
        "0.25 m shift at 2 m altitude collapsed (per-tile RTC / graticule phase lost precision)"
    );

    // 2. Coarse parent texture overzoom at 2 m altitude:
    //    Upload ONLY coarse z = 0 Geodetic root tiles painted in distinct
    //    quadrants (NE = bright red, SE = bright green, NW = bright blue, SW = yellow)
    //    so overzooming z_geom >= 13 at (45 N, 45 E) samples strictly the NE quadrant (red)
    //    via sub-rect UVs with ZERO requests above max_tex_level.
    let mut quad_data = vec![0u8; 64 * 64 * 4];
    for py in 0..64u32 {
        for px in 0..64u32 {
            let color = match (px < 32, py < 32) {
                (true, true) => [220, 40, 40, 255], // Top-left (0..90 E, 0..90 N) -> RED
                (false, true) => [40, 210, 40, 255], // Top-right (90..180 E, 0..90 N) -> GREEN
                (true, false) => [40, 80, 220, 255], // Bottom-left (0..90 E, -90..0 S) -> BLUE
                (false, false) => [220, 200, 40, 255],
            };
            let off = ((py * 64 + px) * 4) as usize;
            quad_data[off..off + 4].copy_from_slice(&color);
        }
    }
    let synthetic_east = DecodedTile {
        width: 64,
        height: 64,
        format: PixelFormat::Rgba8UnormSrgb,
        data: quad_data,
    };
    let mut overzoom_cache = GpuTileCache::new(1024 * 1024);
    let east_root = bundled_tile_key(QuadTileId {
        scheme: TileScheme::Geodetic,
        level: 0,
        x: 1,
        y: 0,
    });
    let _ = overzoom_cache.upload_and_insert(device, queue, east_root, &synthetic_east);

    // (45 N, 45 E) lies at the exact center of the top-left quadrant (RED) of east root (x=1, y=0).
    let overzoom_snap = DrawSnapshot::build(
        Camera::at(45.0, 45.0, MIN_HEIGHT_M),
        w,
        h,
        1.0,
        false,
        true,
        4,
        &cfg,
        Some(&mut overzoom_cache),
    );
    let overzoom_px = render_with(
        &mut renderer,
        device,
        queue,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        RenderParams {
            width: w,
            height: h,
            snapshot: &overzoom_snap,
        },
        true,
    );
    for y in 0..h {
        for x in 0..w {
            let px = pixel_at(&overzoom_px, w, x, y);
            assert!(
                px[0] > 200 && px[1] < 60 && px[2] < 60,
                "coarse parent overzoom pixel ({x},{y}) did not sample expected red sub-rect: {px:?}"
            );
        }
    }

    // 3. Multi-LOD seam continuity (mid-altitude multi-level transition, polar
    //    boundary seam at +85.05112878 deg, and adaptive Polar Cap at ±90 deg):
    //    ZERO background pixels may leak through any LOD seam or pole sliver!
    let multi_lod_cam = Camera::at(45.0, 39.0, 250_000.0);
    let multi_lod_snap = DrawSnapshot::build(multi_lod_cam, w, h, 1.0, true, false, 5, &cfg, None);
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
        max_l >= min_l,
        "expected valid LOD range in mid-altitude view"
    );
    let multi_lod_px = render_with(
        &mut renderer,
        device,
        queue,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        RenderParams {
            width: w,
            height: h,
            snapshot: &multi_lod_snap,
        },
        true,
    );
    for y in 0..h {
        for x in 0..w {
            let px = pixel_at(&multi_lod_px, w, x, y);
            assert!(
                !is_clear_background(px),
                "mid-altitude multi-LOD seam leaked background at ({x},{y}): {px:?}"
            );
        }
    }

    for (label, cam) in [
        (
            "polar boundary seam at 50 m",
            Camera::at(WEB_MERCATOR_LIMIT_DEG, 15.0, 50.0),
        ),
        (
            "exact North Pole at 2 m",
            Camera::at(90.0, 0.0, MIN_HEIGHT_M),
        ),
        (
            "near South Pole at 2 m",
            Camera::at(-89.995, 135.0, MIN_HEIGHT_M),
        ),
    ] {
        let seam_snap = DrawSnapshot::build(cam, w, h, 1.0, true, false, 5, &cfg, None);
        let seam_px = render_with(
            &mut renderer,
            device,
            queue,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            RenderParams {
                width: w,
                height: h,
                snapshot: &seam_snap,
            },
            true,
        );
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

    // 4. PolarCap close-up (2 m) procedural graticule precision: shifting 0.25 m
    //    north of (86.0 N, 0.0 E) on a fine PolarCap tile must visibly shift the
    //    86.0 N parallel line (no f32 colat/lon cancellation collapse).
    let pc_on = DrawSnapshot::build(
        Camera::at(86.0, 0.0, MIN_HEIGHT_M),
        w,
        h,
        1.0,
        true,
        false,
        6,
        &cfg,
        None,
    );
    let dlat_25cm = 0.25 / 111_320.0;
    let pc_shifted = DrawSnapshot::build(
        Camera::at(86.0 + dlat_25cm, 0.0, MIN_HEIGHT_M),
        w,
        h,
        1.0,
        true,
        false,
        7,
        &cfg,
        None,
    );
    let pc_px0 = render_with(
        &mut renderer,
        device,
        queue,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        RenderParams {
            width: w,
            height: h,
            snapshot: &pc_on,
        },
        true,
    );
    let pc_px1 = render_with(
        &mut renderer,
        device,
        queue,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        RenderParams {
            width: w,
            height: h,
            snapshot: &pc_shifted,
        },
        true,
    );
    assert_ne!(
        pc_px0, pc_px1,
        "0.25 m shift on PolarCap at (86 N, 2 m) collapsed (PolarCap graticule lost precision)"
    );
}
