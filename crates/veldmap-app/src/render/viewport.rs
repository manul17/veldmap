//! GPU uniform layouts, pointer-identity dirty checks, and per-viewport
//! offscreen target lifecycle for [`super::GlobeRenderer`].

use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Weak};

use bytemuck::{Pod, Zeroable};

use super::DrawSnapshot;

/// `TileUniform::tile_kind` tag for `Geodetic` / `WebMercator` tiles.
pub(super) const TILE_KIND_GEO: f32 = 0.0;
/// `TileUniform::tile_kind` tag for North PolarCap tiles near the pole.
pub(super) const TILE_KIND_POLAR_NORTH: f32 = 1.0;
/// `TileUniform::tile_kind` tag for South PolarCap tiles near the pole.
pub(super) const TILE_KIND_POLAR_SOUTH: f32 = -1.0;
/// Scale multiplier applied to `TILE_KIND_POLAR_*` (`±2.0`) when a PolarCap
/// tile is strictly clear of the pole and uses cancellation-free relative
/// longitude deltas in `shaders/globe.wgsl` (`abs(tile.tile_kind) > 1.5`).
pub(super) const TILE_KIND_POLAR_CLEAR_SCALE: f32 = 2.0;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct CameraUniform {
    pub(super) view_proj: [[f32; 4]; 4],
    pub(super) grid_step_deg: f32,
    pub(super) grid_width_px: f32,
    pub(super) polar_r_cap: f32,
    pub(super) _pad0: f32,
}

const _: () = {
    assert!(std::mem::size_of::<CameraUniform>() == 80);
    assert!(std::mem::offset_of!(CameraUniform, view_proj) == 0);
    assert!(std::mem::offset_of!(CameraUniform, grid_step_deg) == 64);
    assert!(std::mem::offset_of!(CameraUniform, grid_width_px) == 68);
    assert!(std::mem::offset_of!(CameraUniform, polar_r_cap) == 72);
};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct TileUniform {
    pub(super) offset_eye: [f32; 3],
    pub(super) tile_kind: f32,
    pub(super) uv_rect: [f32; 4],
    pub(super) grid_anchor: [f32; 4],
}

const _: () = {
    assert!(std::mem::size_of::<TileUniform>() == 48);
    assert!(std::mem::offset_of!(TileUniform, offset_eye) == 0);
    assert!(std::mem::offset_of!(TileUniform, tile_kind) == 12);
    assert!(std::mem::offset_of!(TileUniform, uv_rect) == 16);
    assert!(std::mem::offset_of!(TileUniform, grid_anchor) == 32);
};

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
pub(super) struct LastRenderKey {
    width: u32,
    height: u32,
    snapshot: Weak<DrawSnapshot>,
}

impl LastRenderKey {
    pub(super) fn from_params(width: u32, height: u32, snapshot: &Arc<DrawSnapshot>) -> Self {
        Self {
            width,
            height,
            snapshot: Arc::downgrade(snapshot),
        }
    }

    pub(super) fn matches(&self, width: u32, height: u32, snapshot: &Arc<DrawSnapshot>) -> bool {
        self.width == width
            && self.height == height
            && self
                .snapshot
                .upgrade()
                .is_some_and(|prev| Arc::ptr_eq(&prev, snapshot))
    }
}

pub(super) struct OffscreenTarget {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) color_srgb_view: wgpu::TextureView,
    pub(super) depth_view: wgpu::TextureView,
    pub(super) composite_bind_group: wgpu::BindGroup,
}

/// Per-viewport GPU buffers and offscreen targets grouped cleanly so resizing
/// never recreates `camera_buffer` or `tile_uniform_buffer`.
pub(super) struct ViewportState {
    pub(super) camera_buffer: wgpu::Buffer,
    pub(super) camera_bind_group: wgpu::BindGroup,
    pub(super) tile_uniform_buffer: wgpu::Buffer,
    pub(super) tile_uniform_bind_group: wgpu::BindGroup,
    pub(super) tile_uniform_capacity: usize,
    pub(super) offscreen: Option<OffscreenTarget>,
    pub(super) last_render: Option<LastRenderKey>,
    /// Tracks whether the most recent offscreen pass reached `composite_into`.
    pub(super) pending_present: AtomicBool,
}

impl ViewportState {
    pub(super) fn new(
        device: &wgpu::Device,
        camera_layout: &wgpu::BindGroupLayout,
        tile_uniform_layout: &wgpu::BindGroupLayout,
        tile_uniform_stride: u32,
    ) -> Self {
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("globe camera uniform"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("globe camera bind group"),
            layout: camera_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        let initial_cap = 128usize;
        let (tile_uniform_buffer, tile_uniform_bind_group) = Self::alloc_tile_uniforms(
            device,
            tile_uniform_layout,
            tile_uniform_stride,
            initial_cap,
        );

        Self {
            camera_buffer,
            camera_bind_group,
            tile_uniform_buffer,
            tile_uniform_bind_group,
            tile_uniform_capacity: initial_cap,
            offscreen: None,
            last_render: None,
            pending_present: AtomicBool::new(false),
        }
    }

    fn alloc_tile_uniforms(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        stride: u32,
        capacity: usize,
    ) -> (wgpu::Buffer, wgpu::BindGroup) {
        let size = (capacity.max(1) as u64) * u64::from(stride);
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("globe per-tile dynamic uniform"),
            size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("globe per-tile dynamic bind group"),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &buffer,
                    offset: 0,
                    size: wgpu::BufferSize::new(std::mem::size_of::<TileUniform>() as u64),
                }),
            }],
        });
        (buffer, bind_group)
    }

    pub(super) fn ensure_tile_uniform_capacity(
        &mut self,
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        stride: u32,
        needed: usize,
    ) {
        if needed <= self.tile_uniform_capacity {
            return;
        }
        let next_cap = needed.next_power_of_two().max(128);
        let (buf, bg) = Self::alloc_tile_uniforms(device, layout, stride, next_cap);
        self.tile_uniform_buffer = buf;
        self.tile_uniform_bind_group = bg;
        self.tile_uniform_capacity = next_cap;
    }
}
