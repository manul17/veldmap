//! Pointer-identity dirty checks (`LastRenderKey`) and encapsulated
//! per-viewport offscreen target / uniform arena lifecycle for
//! [`super::GlobeRenderer`].

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};

use super::DrawSnapshot;
use super::pipelines::GlobePipelines;
use super::uniforms::{CameraUniform, DynamicUniformArena};

/// Parameters for a single offscreen globe frame.
#[derive(Clone, Copy, Debug)]
pub struct RenderParams<'a> {
    pub width: u32,
    pub height: u32,
    pub snapshot: &'a Arc<DrawSnapshot>,
}

/// O(1) pointer-identity dirty check (`Weak<DrawSnapshot>` + target dimensions)
/// avoiding per-frame tile hashing and ABA pointer reuse without pinning old
/// snapshots or their `Arc<GpuTile>` references.
struct LastRenderKey {
    width: u32,
    height: u32,
    snapshot: Weak<DrawSnapshot>,
}

impl LastRenderKey {
    fn from_params(width: u32, height: u32, snapshot: &Arc<DrawSnapshot>) -> Self {
        Self {
            width,
            height,
            snapshot: Arc::downgrade(snapshot),
        }
    }

    fn matches(&self, width: u32, height: u32, snapshot: &Arc<DrawSnapshot>) -> bool {
        self.width == width
            && self.height == height
            && self
                .snapshot
                .upgrade()
                .is_some_and(|prev| Arc::ptr_eq(&prev, snapshot))
    }
}

pub(super) struct OffscreenTarget {
    width: u32,
    height: u32,
    pub(super) color_srgb_view: wgpu::TextureView,
    pub(super) depth_view: wgpu::TextureView,
    pub(super) composite_bind_group: wgpu::BindGroup,
}

/// Per-viewport GPU buffers and offscreen targets grouped cleanly so resizing
/// never recreates `camera_buffer` or the frame-wide `uniform_arena`.
pub(super) struct ViewportState {
    target_format: wgpu::TextureFormat,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    uniform_arena: DynamicUniformArena,
    offscreen: Option<OffscreenTarget>,
    last_render: Option<LastRenderKey>,
    /// Tracks whether the most recent offscreen pass reached `composite_into`.
    pending_present: AtomicBool,
}

impl ViewportState {
    pub(super) fn new(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        pipelines: &GlobePipelines,
    ) -> Self {
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("globe camera uniform"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("globe camera bind group"),
            layout: &pipelines.layouts.camera_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        let uniform_arena =
            DynamicUniformArena::new(device, &pipelines.layouts.dynamic_uniform_layouts);

        Self {
            target_format,
            camera_buffer,
            camera_bind_group,
            uniform_arena,
            offscreen: None,
            last_render: None,
            pending_present: AtomicBool::new(false),
        }
    }

    pub(super) fn camera_buffer(&self) -> &wgpu::Buffer {
        &self.camera_buffer
    }

    pub(super) fn camera_bind_group(&self) -> &wgpu::BindGroup {
        &self.camera_bind_group
    }

    pub(super) fn uniform_arena(&self) -> &DynamicUniformArena {
        &self.uniform_arena
    }

    pub(super) fn uniform_arena_mut(&mut self) -> &mut DynamicUniformArena {
        &mut self.uniform_arena
    }

    pub(super) fn offscreen(&self) -> Option<&OffscreenTarget> {
        self.offscreen.as_ref()
    }

    pub(super) fn is_clean_for(
        &self,
        width: u32,
        height: u32,
        snapshot: &Arc<DrawSnapshot>,
    ) -> bool {
        !self.pending_present.load(Ordering::Relaxed)
            && self
                .last_render
                .as_ref()
                .is_some_and(|last| last.matches(width, height, snapshot))
            && self
                .offscreen
                .as_ref()
                .is_some_and(|t| t.width == width && t.height == height)
    }

    pub(super) fn mark_rendered(&mut self, width: u32, height: u32, snapshot: &Arc<DrawSnapshot>) {
        self.last_render = Some(LastRenderKey::from_params(width, height, snapshot));
        self.pending_present.store(true, Ordering::Relaxed);
    }

    pub(super) fn mark_presented(&self) {
        self.pending_present.store(false, Ordering::Relaxed);
    }

    pub(super) fn ensure_offscreen(
        &mut self,
        device: &wgpu::Device,
        pipelines: &GlobePipelines,
        width: u32,
        height: u32,
    ) {
        if self
            .offscreen
            .as_ref()
            .is_some_and(|t| t.width == width && t.height == height)
        {
            return;
        }

        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let color_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("globe offscreen color (Rgba8UnormSrgb)"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[wgpu::TextureFormat::Rgba8Unorm],
        });

        let color_srgb_view = color_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("globe offscreen srgb view"),
            format: Some(wgpu::TextureFormat::Rgba8UnormSrgb),
            ..Default::default()
        });

        let target_expects_linear = self.target_format.is_srgb()
            || matches!(
                self.target_format,
                wgpu::TextureFormat::Rgba16Float | wgpu::TextureFormat::Rgba32Float
            );
        let unorm_view = (!target_expects_linear).then(|| {
            color_texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("globe composite unorm view"),
                format: Some(wgpu::TextureFormat::Rgba8Unorm),
                ..Default::default()
            })
        });
        let composite_view = unorm_view.as_ref().unwrap_or(&color_srgb_view);

        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("globe offscreen depth (Depth32Float Reverse-Z)"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TRANSIENT_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let composite_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("globe composite bind group"),
            layout: &pipelines.composite.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(composite_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&pipelines.composite.sampler),
                },
            ],
        });

        self.offscreen = Some(OffscreenTarget {
            width,
            height,
            color_srgb_view,
            depth_view,
            composite_bind_group,
        });
    }
}
