//! GPU uniform buffer layouts and frame-wide dynamic uniform arena shared
//! across [`super::pipelines`], [`super::viewport`], [`super::overlay`],
//! [`super::vector`], and [`super::globe`].

use bytemuck::{Pod, Zeroable};

use super::colormap::PALETTE_LUT_LEN;

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

    assert!(std::mem::size_of::<VectorUniform>() == 32);
    assert!(std::mem::offset_of!(VectorUniform, offset_eye) == 0);
    assert!(std::mem::offset_of!(VectorUniform, point_radius_m) == 12);
    assert!(std::mem::offset_of!(VectorUniform, color) == 16);
};

#[repr(usize)]
#[derive(Clone, Copy)]
enum UniformSlot {
    Tile = 0,
    Overlay = 1,
    ScientificOverlay = 2,
    CategoricalOverlay = 3,
    Vector = 4,
}

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
    slots: [UniformSlotEntry; 5],
}

impl DynamicUniformArena {
    pub(super) fn new(
        device: &wgpu::Device,
        tile_layout: &wgpu::BindGroupLayout,
        overlay_layout: &wgpu::BindGroupLayout,
        scientific_layout: &wgpu::BindGroupLayout,
        categorical_layout: &wgpu::BindGroupLayout,
        vector_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let alignment = device.limits().min_uniform_buffer_offset_alignment.max(1);
        let min_slot = alignment as usize;
        let initial_bytes = (128 * min_slot).max(4096);
        let buffer = Self::create_buffer(device, initial_bytes);
        let make_slot =
            |label: &'static str, binding_size: usize, layout: &wgpu::BindGroupLayout| {
                let bind_group =
                    dynamic_uniform_bind_group(device, label, layout, &buffer, binding_size);
                UniformSlotEntry {
                    label,
                    binding_size,
                    layout: layout.clone(),
                    bind_group,
                }
            };
        let slots = [
            make_slot(
                "arena tile dynamic uniform bind group",
                std::mem::size_of::<TileUniform>(),
                tile_layout,
            ),
            make_slot(
                "arena overlay dynamic uniform bind group",
                std::mem::size_of::<OverlayPatchUniform>(),
                overlay_layout,
            ),
            make_slot(
                "arena scientific overlay dynamic uniform bind group",
                std::mem::size_of::<ScientificPatchUniform>(),
                scientific_layout,
            ),
            make_slot(
                "arena categorical overlay dynamic uniform bind group",
                std::mem::size_of::<CategoricalPatchUniform>(),
                categorical_layout,
            ),
            make_slot(
                "arena vector dynamic uniform bind group",
                std::mem::size_of::<VectorUniform>(),
                vector_layout,
            ),
        ];
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

    fn bind_group(&self, slot: UniformSlot) -> &wgpu::BindGroup {
        &self.slots[slot as usize].bind_group
    }

    pub(super) fn tile_bind_group(&self) -> &wgpu::BindGroup {
        self.bind_group(UniformSlot::Tile)
    }

    pub(super) fn overlay_bind_group(&self) -> &wgpu::BindGroup {
        self.bind_group(UniformSlot::Overlay)
    }

    pub(super) fn scientific_overlay_bind_group(&self) -> &wgpu::BindGroup {
        self.bind_group(UniformSlot::ScientificOverlay)
    }

    pub(super) fn categorical_overlay_bind_group(&self) -> &wgpu::BindGroup {
        self.bind_group(UniformSlot::CategoricalOverlay)
    }

    pub(super) fn vector_bind_group(&self) -> &wgpu::BindGroup {
        self.bind_group(UniformSlot::Vector)
    }
}
