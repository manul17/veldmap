//! `Weak<GpuTile>`-guarded GPU texture bind-group cache shared across
//! planetary basemap tiles and draped raster overlays.

use std::collections::HashMap;
use std::sync::{Arc, Weak};

use veldmap_core::tile::PixelFormat;

use crate::gpu_tiles::GpuTile;

use super::pipelines::GlobePipelines;

/// `Weak<GpuTile>`-guarded cache of `wgpu::BindGroup` entries for display
/// textures, keyed by process-unique `GpuTile::tile_id` and guarded on
/// [`PixelFormat`] so incompatible texture formats can never bind to the
/// filterable sRGB layout.
#[derive(Default)]
pub(super) struct TileBindGroupCache {
    entries: HashMap<u64, (Weak<GpuTile>, wgpu::BindGroup)>,
}

impl TileBindGroupCache {
    /// Ensure a bind group exists for `gpu_tile` when its format is
    /// [`PixelFormat::Rgba8UnormSrgb`].
    pub(super) fn ensure(
        &mut self,
        device: &wgpu::Device,
        pipelines: &GlobePipelines,
        gpu_tile: &Arc<GpuTile>,
    ) {
        if gpu_tile.format != PixelFormat::Rgba8UnormSrgb {
            return;
        }
        let reuse = self
            .entries
            .get(&gpu_tile.tile_id)
            .is_some_and(|(weak, _)| {
                weak.strong_count() > 0 && std::ptr::eq(weak.as_ptr(), Arc::as_ptr(gpu_tile))
            });
        if reuse {
            return;
        }
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tile Rgba8UnormSrgb bind group"),
            layout: &pipelines.tile_texture_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&gpu_tile.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&pipelines.tile_sampler),
                },
            ],
        });
        self.entries
            .insert(gpu_tile.tile_id, (Arc::downgrade(gpu_tile), bg));
    }

    /// Look up the cached bind group for `gpu_tile` only if it matches the
    /// live `Arc<GpuTile>` pointer and `expected_format`.
    pub(super) fn get(
        &self,
        gpu_tile: &Arc<GpuTile>,
        expected_format: PixelFormat,
    ) -> Option<&wgpu::BindGroup> {
        if gpu_tile.format != expected_format || expected_format != PixelFormat::Rgba8UnormSrgb {
            return None;
        }
        self.entries.get(&gpu_tile.tile_id).and_then(|(weak, bg)| {
            (weak.strong_count() > 0 && std::ptr::eq(weak.as_ptr(), Arc::as_ptr(gpu_tile)))
                .then_some(bg)
        })
    }

    /// Drop entries whose backing `GpuTile` has been evicted and released.
    pub(super) fn sweep_stale(&mut self) {
        self.entries.retain(|_, (weak, _)| weak.strong_count() > 0);
    }

    /// Total cached bind groups.
    pub(super) fn len(&self) -> usize {
        self.entries.len()
    }
}
