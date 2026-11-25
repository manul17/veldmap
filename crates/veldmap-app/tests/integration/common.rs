//! Shared headless `wgpu` device, bundled NASA Blue Marble fixture, and
//! readback/snapshot helpers for `veldmap-app` GPU integration tests.

use std::sync::{Arc, OnceLock, mpsc};

use veldmap_app::basemap_policy::BasemapCarrierPolicy;
use veldmap_app::gpu_tiles::GpuTileCache;
use veldmap_app::render::{
    DrawSnapshot, GlobeRenderer, RenderParams, SnapshotBuildParams, TileBindGroupCache,
    TileCarrierPolicy, clamp_to_primary_backends,
};
use veldmap_app::{BUNDLED_BASEMAP_CONFIG, BUNDLED_BASEMAP_JPEG};
use veldmap_core::geo::Camera;
use veldmap_core::tile::{QuadtreeConfig, SourceId};
use veldmap_raster::BundledBasemap;

pub(crate) fn shared_headless_device() -> Option<&'static (wgpu::Device, wgpu::Queue)> {
    static DEVICE: OnceLock<Option<(wgpu::Device, wgpu::Queue)>> = OnceLock::new();
    DEVICE.get_or_init(request_headless_device).as_ref()
}

pub(crate) fn shared_bundled_basemap() -> &'static BundledBasemap {
    static BASEMAP: OnceLock<BundledBasemap> = OnceLock::new();
    BASEMAP.get_or_init(|| {
        BundledBasemap::from_encoded_bytes_with_config(BUNDLED_BASEMAP_JPEG, BUNDLED_BASEMAP_CONFIG)
            .expect("decode bundled NASA Blue Marble")
    })
}

fn request_headless_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
    desc.backends = clamp_to_primary_backends(desc.backends);
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

fn build_raw_snapshot(
    camera: Camera,
    width_px: u32,
    height_px: u32,
    show_graticule: bool,
    config: &QuadtreeConfig,
    carrier: Option<(&dyn TileCarrierPolicy, &mut GpuTileCache)>,
) -> (
    DrawSnapshot,
    std::collections::HashSet<veldmap_core::tile::TileKey>,
) {
    DrawSnapshot::build(
        SnapshotBuildParams {
            camera,
            width_px,
            height_px,
            pixels_per_point: 1.0,
            show_graticule,
            config,
        },
        carrier,
    )
}

pub(crate) fn build_snapshot_with_carriers(
    camera: Camera,
    width_px: u32,
    height_px: u32,
    show_graticule: bool,
    active_source: SourceId,
    config: &QuadtreeConfig,
    cache: Option<&mut GpuTileCache>,
) -> (
    Arc<DrawSnapshot>,
    std::collections::HashSet<veldmap_core::tile::TileKey>,
) {
    let policy = BasemapCarrierPolicy::new(active_source);
    let carrier = cache.map(|gpu_cache| (&policy as &dyn TileCarrierPolicy, gpu_cache));
    let (snap, carriers) =
        build_raw_snapshot(camera, width_px, height_px, show_graticule, config, carrier);
    (Arc::new(snap), carriers)
}

pub(crate) fn build_snapshot(
    camera: Camera,
    width_px: u32,
    height_px: u32,
    show_graticule: bool,
    active_source: SourceId,
    config: &QuadtreeConfig,
    cache: Option<&mut GpuTileCache>,
) -> Arc<DrawSnapshot> {
    build_snapshot_with_carriers(
        camera,
        width_px,
        height_px,
        show_graticule,
        active_source,
        config,
        cache,
    )
    .0
}

pub(crate) fn build_empty_snapshot(
    camera: Camera,
    width_px: u32,
    height_px: u32,
    config: &QuadtreeConfig,
) -> DrawSnapshot {
    build_raw_snapshot(camera, width_px, height_px, false, config, None).0
}

pub(crate) fn readback_rgba8(
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

/// Headless test harness pairing a [`GlobeRenderer`] with its hoisted
/// [`TileBindGroupCache`].
pub(crate) struct TestGlobe {
    pub(crate) renderer: GlobeRenderer,
    pub(crate) bind_groups: TileBindGroupCache,
}

impl TestGlobe {
    pub(crate) fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let bind_groups = TileBindGroupCache::new(device);
        let renderer = GlobeRenderer::new(device, target_format, &bind_groups);
        Self {
            renderer,
            bind_groups,
        }
    }

    pub(crate) fn new_srgb(device: &wgpu::Device) -> Self {
        Self::new(device, wgpu::TextureFormat::Rgba8UnormSrgb)
    }

    pub(crate) fn render_if_dirty(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        params: RenderParams<'_>,
    ) -> bool {
        self.renderer
            .render_if_dirty(device, queue, encoder, params, &mut self.bind_groups)
    }
}

pub(crate) fn render_with(
    globe: &mut TestGlobe,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    target_format: wgpu::TextureFormat,
    params: RenderParams<'_>,
    expect_rendered: bool,
) -> Vec<u8> {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("smoke render encoder"),
    });

    let rendered = globe.render_if_dirty(device, queue, &mut encoder, params);
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
        globe.renderer.composite_into(&mut pass);
    }

    queue.submit(Some(encoder.finish()));
    readback_rgba8(device, queue, &composite_tex, params.width, params.height)
}

pub(crate) fn render_srgb(
    globe: &mut TestGlobe,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    width: u32,
    height: u32,
    snapshot: &Arc<DrawSnapshot>,
) -> Vec<u8> {
    render_with(
        globe,
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

pub(crate) fn pixel_at(pixels: &[u8], width: u32, x: u32, y: u32) -> [u8; 4] {
    let idx = ((y * width + x) * 4) as usize;
    [
        pixels[idx],
        pixels[idx + 1],
        pixels[idx + 2],
        pixels[idx + 3],
    ]
}

pub(crate) fn is_clear_background(px: [u8; 4]) -> bool {
    px[0].abs_diff(11) <= 2 && px[1].abs_diff(19) <= 2 && px[2].abs_diff(32) <= 2
}
