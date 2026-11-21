//! Headless offscreen `wgpu` smoke tests verifying:
//! 1. Reverse-Z rendering from orbit (14,000 km) down to 2 m altitude directly
//!    above a mesh vertex (`near = 1 m` vs surface at `2 m`).
//! 2. Exact 1:1 sRGB compositing onto both `Rgba8UnormSrgb` and `Rgba8Unorm`
//!    targets via `view_formats` (zero double-gamma shift).
//! 3. Dirty-flag redraw skipping, `show_graticule` invalidation, resize
//!    recreation, and `0x0` viewport safety.
//!
//! Set `VELDMAP_REQUIRE_GPU_TEST=1` (enabled on Linux Lavapipe and Windows WARP
//! CI) to fail loudly if no `Backends::PRIMARY` adapter is available.

#![expect(
    clippy::expect_used,
    reason = "INVARIANT: integration test helpers panic on headless GPU harness failure"
)]

use veldmap_app::render::{GlobeRenderer, RenderParams};
use veldmap_core::geo::camera::{Camera, MIN_HEIGHT_M};

fn request_headless_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
    desc.backends &= wgpu::Backends::PRIMARY;
    if desc.backends.is_empty() {
        desc.backends = wgpu::Backends::PRIMARY;
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

#[test]
fn headless_reverse_z_and_exact_srgb_compositing() {
    let Some((device, queue)) = request_headless_device() else {
        return;
    };

    // Odd viewport size places NDC (0, 0) cleanly at the center of pixel (64, 64),
    // avoiding cross-driver line tie-break ambiguity on even boundaries.
    let (w, h) = (129, 129);
    let orbital_cam = Camera::default();

    let mut srgb_renderer = GlobeRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut unorm_renderer = GlobeRenderer::new(&device, wgpu::TextureFormat::Rgba8Unorm);

    // 0x0 viewport must skip cleanly without panicking.
    let mut zero_enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    assert!(!srgb_renderer.render_if_dirty(
        &device,
        &queue,
        &mut zero_enc,
        RenderParams {
            width: 0,
            height: 0,
            camera: &orbital_cam,
            show_graticule: true,
            generation: 1,
        },
    ));

    let orbital_params = RenderParams {
        width: w,
        height: h,
        camera: &orbital_cam,
        show_graticule: true,
        generation: 1,
    };

    // 1. Composite onto BOTH sRGB and UNORM targets and verify exact 1:1 parity.
    let srgb_pixels = render_with(
        &mut srgb_renderer,
        &device,
        &queue,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        orbital_params,
        true,
    );
    // Identical key must skip offscreen redraw via dirty-key cache.
    let _ = render_with(
        &mut srgb_renderer,
        &device,
        &queue,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        orbital_params,
        false,
    );
    let unorm_pixels = render_with(
        &mut unorm_renderer,
        &device,
        &queue,
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

    // 2. Toggling show_graticule MUST re-render even with unchanged generation,
    // replacing the center graticule intersection with the lit ellipsoid body.
    let no_grid_pixels = render_with(
        &mut srgb_renderer,
        &device,
        &queue,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        RenderParams {
            show_graticule: false,
            ..orbital_params
        },
        true,
    );
    let body_center = pixel_at(&no_grid_pixels, w, w / 2, h / 2);
    assert!(
        body_center[0] > 180 && body_center[1] > 170 && body_center[2] > 150,
        "center pixel without graticule did not render lit ellipsoid in sRGB: {body_center:?}"
    );

    // 3. Resize existing renderer and verify Reverse-Z close-up at 2 m altitude
    // directly over an exact ellipsoid mesh vertex (45 N, 45 E -> 2.0 m from eye,
    // 1.0 m past the near plane) fills the viewport with zero near-plane clipping.
    let close_cam = Camera::at(45.0, 45.0, MIN_HEIGHT_M);
    let (rw, rh) = (65, 65);
    let close_pixels = render_with(
        &mut srgb_renderer,
        &device,
        &queue,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        RenderParams {
            width: rw,
            height: rh,
            camera: &close_cam,
            show_graticule: false,
            generation: 2,
        },
        true,
    );
    for (x, y) in [(rw / 2, rh / 2), (4, 4), (rw - 5, rh - 5), (4, rh - 5)] {
        let px = pixel_at(&close_pixels, rw, x, y);
        assert!(
            px[0] > 150 && px[1] > 140 && px[2] > 120,
            "2 m altitude Reverse-Z pixel at ({x},{y}) clipped or missed surface: {px:?}"
        );
    }
}
