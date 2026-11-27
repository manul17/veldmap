//! GPU uniform buffer layouts and frame-wide dynamic uniform arena shared
//! across [`super::pipelines`], [`super::viewport`], [`super::overlay`],
//! [`super::vector`], and [`super::globe`].

use bytemuck::{Pod, Zeroable};

use crate::colormap::PALETTE_LUT_LEN;

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

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct OverlayPatchUniform {
    pub(super) offset_eye: [f32; 3],
    pub(super) opacity: f32,
    pub(super) uv_rect: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct ScientificPatchUniform {
    pub(super) offset_eye: [f32; 3],
    pub(super) opacity: f32,
    pub(super) uv_rect: [f32; 4],
    pub(super) scale: f32,
    pub(super) offset: f32,
    pub(super) stretch_min: f32,
    pub(super) stretch_inv_range: f32,
    pub(super) stretch_mode: u32,
    pub(super) invert_colormap: u32,
    pub(super) mip_count: u32,
    pub(super) _pad0: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct CategoricalPatchUniform {
    pub(super) offset_eye: [f32; 3],
    pub(super) opacity: f32,
    pub(super) uv_rect: [f32; 4],
    pub(super) has_nodata: u32,
    pub(super) nodata_value: u32,
    pub(super) _pad0: u32,
    pub(super) _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct PaletteUniform {
    pub(super) entries: [[f32; 4]; PALETTE_LUT_LEN],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct VectorUniform {
    pub(super) offset_eye: [f32; 3],
    pub(super) point_radius_m: f32,
    pub(super) color: [f32; 4],
    pub(super) viewport_inv_size: [f32; 2],
    pub(super) stroke_half_width_px: f32,
    pub(super) hatch_spacing_px: f32,
}

const _: () = {
    assert!(std::mem::size_of::<OverlayPatchUniform>() == 32);
    assert!(std::mem::offset_of!(OverlayPatchUniform, offset_eye) == 0);
    assert!(std::mem::offset_of!(OverlayPatchUniform, opacity) == 12);
    assert!(std::mem::offset_of!(OverlayPatchUniform, uv_rect) == 16);

    assert!(std::mem::size_of::<ScientificPatchUniform>() == 64);
    assert!(std::mem::offset_of!(ScientificPatchUniform, offset_eye) == 0);
    assert!(std::mem::offset_of!(ScientificPatchUniform, opacity) == 12);
    assert!(std::mem::offset_of!(ScientificPatchUniform, uv_rect) == 16);
    assert!(std::mem::offset_of!(ScientificPatchUniform, scale) == 32);
    assert!(std::mem::offset_of!(ScientificPatchUniform, offset) == 36);
    assert!(std::mem::offset_of!(ScientificPatchUniform, stretch_min) == 40);
    assert!(std::mem::offset_of!(ScientificPatchUniform, stretch_inv_range) == 44);
    assert!(std::mem::offset_of!(ScientificPatchUniform, stretch_mode) == 48);
    assert!(std::mem::offset_of!(ScientificPatchUniform, invert_colormap) == 52);
    assert!(std::mem::offset_of!(ScientificPatchUniform, mip_count) == 56);

    assert!(std::mem::size_of::<CategoricalPatchUniform>() == 48);
    assert!(std::mem::offset_of!(CategoricalPatchUniform, offset_eye) == 0);
    assert!(std::mem::offset_of!(CategoricalPatchUniform, opacity) == 12);
    assert!(std::mem::offset_of!(CategoricalPatchUniform, uv_rect) == 16);
    assert!(std::mem::offset_of!(CategoricalPatchUniform, has_nodata) == 32);
    assert!(std::mem::offset_of!(CategoricalPatchUniform, nodata_value) == 36);

    assert!(std::mem::size_of::<PaletteUniform>() == 4096);

    assert!(std::mem::size_of::<VectorUniform>() == 48);
    assert!(std::mem::offset_of!(VectorUniform, offset_eye) == 0);
    assert!(std::mem::offset_of!(VectorUniform, point_radius_m) == 12);
    assert!(std::mem::offset_of!(VectorUniform, color) == 16);
    assert!(std::mem::offset_of!(VectorUniform, viewport_inv_size) == 32);
    assert!(std::mem::offset_of!(VectorUniform, stroke_half_width_px) == 40);
    assert!(std::mem::offset_of!(VectorUniform, hatch_spacing_px) == 44);
};

/// Number of typed dynamic-uniform bind-group slots backed by
/// [`DynamicUniformArena`].
pub(super) const UNIFORM_SLOT_COUNT: usize = 5;

/// Typed index into the frame-wide [`DynamicUniformArena`] bind-group slots and
/// [`super::pipelines::SharedLayouts`] dynamic-uniform layouts.
#[repr(usize)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum UniformSlot {
    Tile = 0,
    Overlay = 1,
    Scientific = 2,
    Categorical = 3,
    Vector = 4,
}

impl UniformSlot {
    pub(super) const ALL: [Self; UNIFORM_SLOT_COUNT] = [
        Self::Tile,
        Self::Overlay,
        Self::Scientific,
        Self::Categorical,
        Self::Vector,
    ];

    pub(super) const fn index(self) -> usize {
        self as usize
    }

    const fn bind_group_label(self) -> &'static str {
        match self {
            Self::Tile => "arena tile dynamic uniform bind group",
            Self::Overlay => "arena overlay dynamic uniform bind group",
            Self::Scientific => "arena scientific overlay dynamic uniform bind group",
            Self::Categorical => "arena categorical overlay dynamic uniform bind group",
            Self::Vector => "arena vector dynamic uniform bind group",
        }
    }

    const fn binding_size(self) -> usize {
        match self {
            Self::Tile => std::mem::size_of::<TileUniform>(),
            Self::Overlay => std::mem::size_of::<OverlayPatchUniform>(),
            Self::Scientific => std::mem::size_of::<ScientificPatchUniform>(),
            Self::Categorical => std::mem::size_of::<CategoricalPatchUniform>(),
            Self::Vector => std::mem::size_of::<VectorUniform>(),
        }
    }
}

const _: () = {
    let mut i = 0;
    while i < UNIFORM_SLOT_COUNT {
        assert!(UniformSlot::ALL[i].index() == i);
        i += 1;
    }
};

struct UniformSlotEntry {
    label: &'static str,
    binding_size: usize,
    layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
}

fn dynamic_uniform_bind_group(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::BindGroupLayout,
    buffer: &wgpu::Buffer,
    binding_size: usize,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer,
                offset: 0,
                size: wgpu::BufferSize::new(binding_size as u64),
            }),
        }],
    })
}

/// Frame-wide dynamic uniform arena shared across globe body, post-body
/// display/scientific/categorical raster overlays, vector overlays, and
/// graticule passes.
pub(super) struct DynamicUniformArena {
    alignment: u32,
    staging: Vec<u8>,
    last_offset: usize,
    buffer: wgpu::Buffer,
    capacity_bytes: usize,
    slots: [UniformSlotEntry; UNIFORM_SLOT_COUNT],
}

impl DynamicUniformArena {
    pub(super) fn new(
        device: &wgpu::Device,
        layouts: &[wgpu::BindGroupLayout; UNIFORM_SLOT_COUNT],
    ) -> Self {
        let alignment = device.limits().min_uniform_buffer_offset_alignment.max(1);
        let min_slot = alignment as usize;
        let initial_bytes = (128 * min_slot).max(4096);
        let buffer = Self::create_buffer(device, initial_bytes);
        let slots = std::array::from_fn(|idx| {
            let slot = UniformSlot::ALL[idx];
            let label = slot.bind_group_label();
            let binding_size = slot.binding_size();
            let layout = &layouts[idx];
            let bind_group =
                dynamic_uniform_bind_group(device, label, layout, &buffer, binding_size);
            UniformSlotEntry {
                label,
                binding_size,
                layout: layout.clone(),
                bind_group,
            }
        });
        Self {
            alignment,
            staging: Vec::with_capacity(initial_bytes),
            last_offset: 0,
            buffer,
            capacity_bytes: initial_bytes,
            slots,
        }
    }

    fn create_buffer(device: &wgpu::Device, size_bytes: usize) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame dynamic uniform arena"),
            size: size_bytes.max(256) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    pub(super) fn reset(&mut self) {
        self.staging.clear();
        self.last_offset = 0;
    }

    /// Append `value` at the next hardware-aligned dynamic offset and return
    /// that byte offset for `RenderPass::set_bind_group`.
    pub(super) fn push<T: Pod>(&mut self, value: &T) -> u32 {
        let align = self.alignment as usize;
        let offset = self.staging.len().div_ceil(align) * align;
        if offset > self.staging.len() {
            self.staging.resize(offset, 0);
        }
        let bytes = bytemuck::bytes_of(value);
        self.staging.extend_from_slice(bytes);
        self.last_offset = offset;
        offset as u32
    }

    /// Ensure the GPU arena buffer fits all pushed uniforms plus the final
    /// entry's declared binding range, then upload the frame's staging bytes
    /// in a single write.
    pub(super) fn flush(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.staging.is_empty() {
            return;
        }
        let max_binding = self
            .slots
            .iter()
            .map(|slot| slot.binding_size)
            .max()
            .unwrap_or(0);
        let copy_len = self.staging.len().div_ceil(4) * 4;
        let required_bytes = copy_len.max(self.last_offset.saturating_add(max_binding));
        if required_bytes > self.capacity_bytes {
            let next_cap = required_bytes.next_power_of_two().max(4096);
            self.buffer = Self::create_buffer(device, next_cap);
            for slot in &mut self.slots {
                slot.bind_group = dynamic_uniform_bind_group(
                    device,
                    slot.label,
                    &slot.layout,
                    &self.buffer,
                    slot.binding_size,
                );
            }
            self.capacity_bytes = next_cap;
        }
        if copy_len > self.staging.len() {
            self.staging.resize(copy_len, 0);
        }
        queue.write_buffer(&self.buffer, 0, &self.staging[..copy_len]);
    }

    pub(super) fn bind_group(&self, slot: UniformSlot) -> &wgpu::BindGroup {
        &self.slots[slot.index()].bind_group
    }
}
