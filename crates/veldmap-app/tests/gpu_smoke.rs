//! Headless offscreen `wgpu` smoke tests verifying:
//! 1. Reverse-Z depth rendering from orbit (14,000 km) down to 2 m altitude.
//! 2. Exact 1:1 sRGB compositing onto both `Rgba8UnormSrgb` and `Rgba8Unorm`
//!    targets via `view_formats` (zero double-gamma shift).
//! 3. Dirty-flag redraw skipping.
//!
//! Set `VELDMAP_REQUIRE_GPU_TEST=1` (enabled on Linux Lavapipe and Windows WARP
//! CI) to fail loudly if no `Backends::PRIMARY` adapter is available.

#![expect(
    clippy::expect_used,
    reason = "INVARIANT: integration test helpers panic on headless GPU harness failure"
)]

use veldmap_app::render::GlobeRenderer;
use veldmap_core::geo::camera::{Camera, MIN_HEIGHT_M};

fn request_headless_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());

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
            panic!("VELDMAP_REQUIRE_GPU_TEST=1 is set, but no wgpu adapter was found");
        }
        eprintln!("Skipping headless GPU smoke test: no wgpu adapter available");
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
    let (tx, rx) = std::sync::mpsc::channel();
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

fn render_and_composite(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    target_format: wgpu::TextureFormat,
    camera: &Camera,
    width: u32,
    height: u32,
) -> Vec<u8> {
    let mut renderer = GlobeRenderer::new(device, target_format);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("smoke render encoder"),
    });

    let rendered = renderer.render_if_dirty(
        device,
        queue,
        &mut encoder,
        width,
        height,
        1.0,
        camera,
        true,
        1,
    );
    assert!(rendered, "initial frame must render");

    // Second call with identical key must skip redraw via dirty flag.
    let rendered_again = renderer.render_if_dirty(
        device,
        queue,
        &mut encoder,
        width,
        height,
        1.0,
        camera,
        true,
        1,
    );
    assert!(!rendered_again, "unchanged frame must skip redraw");

    let composite_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("composite target"),
        size: wgpu::Extent3d {
            width,
            height,
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
    readback_rgba8(device, queue, &composite_tex, width, height)
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

#[test]
fn headless_reverse_z_and_exact_srgb_compositing() {
    let Some((device, queue)) = request_headless_device() else {
        return;
    };

    let (w, h) = (128, 128);
    let orbital_cam = Camera::default();

    // 1. Composite onto BOTH sRGB and UNORM targets and verify exact 1:1 parity.
    let srgb_pixels = render_and_composite(
        &device,
        &queue,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        &orbital_cam,
        w,
        h,
    );
    let unorm_pixels = render_and_composite(
        &device,
        &queue,
        wgpu::TextureFormat::Rgba8Unorm,
        &orbital_cam,
        w,
        h,
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

    // Corner at orbital zoom must be the exact sRGB-encoded deep-space background (#0B1320 = 11, 19, 32).
    let corner = pixel_at(&srgb_pixels, w, 2, 2);
    assert!(
        corner[0].abs_diff(11) <= 1 && corner[1].abs_diff(19) <= 1 && corner[2].abs_diff(32) <= 1,
        "unexpected orbital background corner pixel: {corner:?}"
    );
    assert_eq!(corner[3], 255);

    // Center at Camera::default() (45 deg N, 60 deg E) sits on an exact 15-degree
    // graticule intersection -> exact sRGB #A8B99B (168, 185, 155, 255).
    let center = pixel_at(&srgb_pixels, w, w / 2, h / 2);
    assert!(
        center[0].abs_diff(168) <= 2
            && center[1].abs_diff(185) <= 2
            && center[2].abs_diff(155) <= 2,
        "orbital center pixel did not match exact draped graticule sRGB #A8B99B: {center:?}"
    );

    // Slightly off the 45/60 graticule lines must hit the bright lit ellipsoid body.
    let body_px = pixel_at(&srgb_pixels, w, w / 2 + 5, h / 2 + 5);
    assert!(
        body_px[0] > 180 && body_px[1] > 170 && body_px[2] > 150,
        "orbital body pixel did not render lit ellipsoid in sRGB: {body_px:?}"
    );

    // 2. Reverse-Z close-up at 2 m altitude: surface must fill the viewport with
    // zero near-plane clipping and zero background bleed.
    let close_cam = Camera::at(47.2357, 39.7015, MIN_HEIGHT_M);
    let close_pixels = render_and_composite(
        &device,
        &queue,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        &close_cam,
        w,
        h,
    );
    for (x, y) in [(w / 2, h / 2), (8, 8), (w - 8, h - 8), (8, h - 8)] {
        let px = pixel_at(&close_pixels, w, x, y);
        assert!(
            px[0] > 150 && px[1] > 140 && px[2] > 120,
            "2 m altitude Reverse-Z pixel at ({x},{y}) clipped or missed surface: {px:?}"
        );
    }
}
