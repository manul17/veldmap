//! GPU uniform buffer layouts and frame-wide dynamic uniform arena shared
//! across [`super::pipelines`], [`super::viewport`], [`super::overlay`], and
//! [`super::globe`].

use bytemuck::{Pod, Zeroable};
use veldmap_core::tile::{PlanetaryScheme, QuadTileId};

use super::mesh::TileGridAnchor;

/// `TileUniform::tile_kind` tag for `Geodetic` / `WebMercator` tiles.
pub(super) const TILE_KIND_GEO: f32 = 0.0;
/// `TileUniform::tile_kind` tag for North PolarCap tiles near the pole.
const TILE_KIND_POLAR_NORTH: f32 = 1.0;
/// `TileUniform::tile_kind` tag for South PolarCap tiles near the pole.
const TILE_KIND_POLAR_SOUTH: f32 = -1.0;
/// Scale multiplier applied to `TILE_KIND_POLAR_*` (`±2.0`) when a PolarCap
/// tile is strictly clear of the pole and uses cancellation-free relative
/// longitude deltas in `shaders/globe.wgsl` (`abs(tile.tile_kind) > 1.5`).
const TILE_KIND_POLAR_CLEAR_SCALE: f32 = 2.0;

/// Compute per-tile `(tile_kind, grid_anchor)` uniform values from a cached
/// [`TileGridAnchor`] so `GlobeRenderer` never re-samples `sample_tile_surface`
/// on every dirty frame.
pub(super) fn tile_graticule_uniform(
    id: QuadTileId,
    cached_anchor: Option<TileGridAnchor>,
    step_deg: f64,
) -> (f32, [f32; 4]) {
    let anchor = cached_anchor.unwrap_or_else(|| TileGridAnchor::for_tile(id));
    match id.scheme {
        PlanetaryScheme::PolarCap { north } => {
            let colat_phase = (90.0 - anchor.lat0.abs()).rem_euclid(step_deg);
            let lon_phase = anchor.lon0.rem_euclid(step_deg);

            // Use cancellation-free relative longitude deltas when the tile's
            // disk center is strictly clear of the pole (`rho0 > 4 * tile_half_diag`).
            let rho0 = anchor.dx0.hypot(anchor.dy0);
            let n = f64::from(1u32 << id.level.min(30));
            let tile_half_diag = std::f64::consts::SQRT_2 / n;
            let clear_of_pole = id.level >= 2 && rho0 > 4.0 * tile_half_diag;
            let sign = if north {
                TILE_KIND_POLAR_NORTH
            } else {
                TILE_KIND_POLAR_SOUTH
            };
            let kind = if clear_of_pole {
                sign * TILE_KIND_POLAR_CLEAR_SCALE
            } else {
                sign
            };
            (
                kind,
                [
                    anchor.dx0 as f32,
                    anchor.dy0 as f32,
                    colat_phase as f32,
                    lon_phase as f32,
                ],
            )
        }
        PlanetaryScheme::Geodetic | PlanetaryScheme::WebMercator => {
            // Wrap tile anchor phase modulo `grid_step_deg` in f64 on CPU so
            // `f32` fragment arithmetic preserves sub-millimetre accuracy even
            // at `2 m` altitude.
            let lon_phase = anchor.lon0.rem_euclid(step_deg);
            let lat_phase = anchor.lat0.rem_euclid(step_deg);
            (
                TILE_KIND_GEO,
                [lon_phase as f32, lat_phase as f32, 0.0, 0.0],
            )
        }
    }
}

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

const _: () = {
    assert!(std::mem::size_of::<OverlayPatchUniform>() == 32);
    assert!(std::mem::offset_of!(OverlayPatchUniform, offset_eye) == 0);
    assert!(std::mem::offset_of!(OverlayPatchUniform, opacity) == 12);
    assert!(std::mem::offset_of!(OverlayPatchUniform, uv_rect) == 16);
};

/// Frame-wide dynamic uniform arena shared across globe body, post-body
/// overlays, and graticule passes.
///
/// Accumulates all aligned per-draw uniforms in a single CPU staging buffer and
/// flushes them with one `queue.write_buffer` call per dirty frame, preventing
/// `queue.write_buffer(..., 0, ...)` clobbering across multi-pass renders.
pub(super) struct DynamicUniformArena {
    alignment: u32,
    staging: Vec<u8>,
    last_offset: usize,
    buffer: wgpu::Buffer,
    capacity_bytes: usize,
    tile_bind_group: wgpu::BindGroup,
    overlay_bind_group: wgpu::BindGroup,
}

impl DynamicUniformArena {
    pub(super) fn new(
        device: &wgpu::Device,
        tile_layout: &wgpu::BindGroupLayout,
        overlay_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let alignment = device.limits().min_uniform_buffer_offset_alignment.max(1);
        let min_slot = alignment as usize;
        let initial_bytes = (128 * min_slot).max(4096);
        let (buffer, tile_bind_group, overlay_bind_group) =
            Self::alloc(device, tile_layout, overlay_layout, initial_bytes);
        Self {
            alignment,
            staging: Vec::with_capacity(initial_bytes),
            last_offset: 0,
            buffer,
            capacity_bytes: initial_bytes,
            tile_bind_group,
            overlay_bind_group,
        }
    }

    fn alloc(
        device: &wgpu::Device,
        tile_layout: &wgpu::BindGroupLayout,
        overlay_layout: &wgpu::BindGroupLayout,
        size_bytes: usize,
    ) -> (wgpu::Buffer, wgpu::BindGroup, wgpu::BindGroup) {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame dynamic uniform arena"),
            size: size_bytes.max(256) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let tile_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("arena tile dynamic uniform bind group"),
            layout: tile_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &buffer,
                    offset: 0,
                    size: wgpu::BufferSize::new(std::mem::size_of::<TileUniform>() as u64),
                }),
            }],
        });
        let overlay_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("arena overlay dynamic uniform bind group"),
            layout: overlay_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &buffer,
                    offset: 0,
                    size: wgpu::BufferSize::new(std::mem::size_of::<OverlayPatchUniform>() as u64),
                }),
            }],
        });
        (buffer, tile_bind_group, overlay_bind_group)
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
    pub(super) fn flush(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        tile_layout: &wgpu::BindGroupLayout,
        overlay_layout: &wgpu::BindGroupLayout,
    ) {
        if self.staging.is_empty() {
            return;
        }
        let max_binding =
            std::mem::size_of::<TileUniform>().max(std::mem::size_of::<OverlayPatchUniform>());
        let copy_len = self.staging.len().div_ceil(4) * 4;
        let required_bytes = copy_len.max(self.last_offset.saturating_add(max_binding));
        if required_bytes > self.capacity_bytes {
            let next_cap = required_bytes.next_power_of_two().max(4096);
            let (buf, tile_bg, overlay_bg) =
                Self::alloc(device, tile_layout, overlay_layout, next_cap);
            self.buffer = buf;
            self.tile_bind_group = tile_bg;
            self.overlay_bind_group = overlay_bg;
            self.capacity_bytes = next_cap;
        }
        if copy_len > self.staging.len() {
            self.staging.resize(copy_len, 0);
        }
        queue.write_buffer(&self.buffer, 0, &self.staging[..copy_len]);
    }

    pub(super) fn tile_bind_group(&self) -> &wgpu::BindGroup {
        &self.tile_bind_group
    }

    pub(super) fn overlay_bind_group(&self) -> &wgpu::BindGroup {
        &self.overlay_bind_group
    }
}
