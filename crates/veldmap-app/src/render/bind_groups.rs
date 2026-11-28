//! `Weak<GpuTile>`-guarded multi-format GPU texture bind-group cache shared
//! across planetary basemap tiles, draped display/scientific/categorical
//! overlays, and future orthographic 2D views via `egui_wgpu::CallbackResources`.

use std::collections::HashMap;
use std::sync::{Arc, Weak};

use veldmap_core::tile::PixelFormat;

use crate::gpu_tiles::GpuTile;

use super::pipelines::texture_2d_layout_entry;

type CachedBindGroupEntry = (Weak<GpuTile>, PixelFormat, wgpu::BindGroup);
type CachedMrtBindGroupEntry = ([Weak<GpuTile>; 4], wgpu::BindGroup, u64);

/// Soft ceiling on combinatorial live 4-band MRT bind-group tuples pruned on
/// frame start (strictly keeping entries touched during the active frame).
const MAX_MRT_BIND_GROUPS: usize = 512;

fn entry_matches(
    entry: &CachedBindGroupEntry,
    gpu_tile: &Arc<GpuTile>,
    expected_format: PixelFormat,
) -> bool {
    let (weak, fmt, _) = entry;
    *fmt == expected_format
        && weak.strong_count() > 0
        && std::ptr::eq(weak.as_ptr(), Arc::as_ptr(gpu_tile))
}

fn mrt_entry_matches(entry: &CachedMrtBindGroupEntry, tiles: [&Arc<GpuTile>; 4]) -> bool {
    entry.0.iter().zip(tiles).all(|(weak, tile)| {
        weak.strong_count() > 0 && std::ptr::eq(weak.as_ptr(), Arc::as_ptr(tile))
    })
}

/// `Weak<GpuTile>`-guarded cache of `wgpu::BindGroup` entries across all
/// supported tile pixel formats (`Rgba8UnormSrgb`, `R32Float`, `R8Uint`,
/// `R16Uint`, plus 4-slot MRT `R32Float` bundles), keyed by process-unique
/// `GpuTile::tile_id`.
///
/// Hoisted into `egui_wgpu::CallbackResources` so `GlobeRenderer` and future
/// multi-viewport renderers (e.g. `Image2DRenderer`) share identical layouts
/// and cached texture bind groups without duplicating GPU resources.
pub struct TileBindGroupCache {
    rgba8_layout: wgpu::BindGroupLayout,
    r32f_layout: wgpu::BindGroupLayout,
    r32f_mrt_layout: wgpu::BindGroupLayout,
    uint_layout: wgpu::BindGroupLayout,
    tile_sampler: wgpu::Sampler,
    entries: HashMap<u64, CachedBindGroupEntry>,
    mrt_entries: HashMap<[u64; 4], CachedMrtBindGroupEntry>,
    mrt_epoch: u64,
}

impl TileBindGroupCache {
    /// Allocate the shared format-specific bind-group layouts (`Rgba8UnormSrgb`
    /// filterable + linear sampler, single & 4-slot MRT `R32Float` unfilterable
    /// `textureLoad`, and `Uint` `textureLoad` for `R8Uint`/`R16Uint`).
    pub fn new(device: &wgpu::Device) -> Self {
        let rgba8_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tile Rgba8UnormSrgb texture layout"),
            entries: &[
                texture_2d_layout_entry(0, wgpu::TextureSampleType::Float { filterable: true }),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let unfilterable_float = wgpu::TextureSampleType::Float { filterable: false };
        let r32f_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tile R32Float unfilterable texture layout"),
            entries: &[texture_2d_layout_entry(0, unfilterable_float)],
        });

        let r32f_mrt_entries: [wgpu::BindGroupLayoutEntry; 4] = std::array::from_fn(|binding| {
            texture_2d_layout_entry(binding as u32, unfilterable_float)
        });
        let r32f_mrt_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tile R32Float MRT 4-band texture layout"),
            entries: &r32f_mrt_entries,
        });

        let uint_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tile Uint categorical texture layout"),
            entries: &[texture_2d_layout_entry(0, wgpu::TextureSampleType::Uint)],
        });

        let tile_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("globe tile trilinear clamp sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        Self {
            rgba8_layout,
            r32f_layout,
            r32f_mrt_layout,
            uint_layout,
            tile_sampler,
            entries: HashMap::new(),
            mrt_entries: HashMap::new(),
            mrt_epoch: 0,
        }
    }

    pub(super) fn rgba8_layout(&self) -> &wgpu::BindGroupLayout {
        &self.rgba8_layout
    }

    pub(super) fn r32f_layout(&self) -> &wgpu::BindGroupLayout {
        &self.r32f_layout
    }

    pub(super) fn r32f_mrt_layout(&self) -> &wgpu::BindGroupLayout {
        &self.r32f_mrt_layout
    }

    pub(super) fn uint_layout(&self) -> &wgpu::BindGroupLayout {
        &self.uint_layout
    }

    /// Ensure a format-appropriate bind group exists for `gpu_tile`.
    pub(super) fn ensure(&mut self, device: &wgpu::Device, gpu_tile: &Arc<GpuTile>) {
        let reuse = self
            .entries
            .get(&gpu_tile.tile_id)
            .is_some_and(|entry| entry_matches(entry, gpu_tile, gpu_tile.format));
        if reuse {
            return;
        }
        let bg = match gpu_tile.format {
            PixelFormat::Rgba8UnormSrgb => device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tile Rgba8UnormSrgb bind group"),
                layout: &self.rgba8_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&gpu_tile.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.tile_sampler),
                    },
                ],
            }),
            PixelFormat::R32Float => device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tile R32Float bind group"),
                layout: &self.r32f_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&gpu_tile.view),
                }],
            }),
            PixelFormat::R8Uint | PixelFormat::R16Uint => {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("tile Uint bind group"),
                    layout: &self.uint_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&gpu_tile.view),
                    }],
                })
            }
        };
        self.entries.insert(
            gpu_tile.tile_id,
            (Arc::downgrade(gpu_tile), gpu_tile.format, bg),
        );
    }

    /// Look up the cached bind group for `gpu_tile` only if it matches the
    /// live `Arc<GpuTile>` pointer and `expected_format`.
    pub(super) fn get(
        &self,
        gpu_tile: &Arc<GpuTile>,
        expected_format: PixelFormat,
    ) -> Option<&wgpu::BindGroup> {
        if gpu_tile.format != expected_format {
            return None;
        }
        self.entries
            .get(&gpu_tile.tile_id)
            .and_then(|entry| entry_matches(entry, gpu_tile, expected_format).then_some(&entry.2))
    }

    /// Look up a cached `Uint` bind group for either `R8Uint` or `R16Uint`.
    pub(super) fn get_uint(&self, gpu_tile: &Arc<GpuTile>) -> Option<&wgpu::BindGroup> {
        if !matches!(gpu_tile.format, PixelFormat::R8Uint | PixelFormat::R16Uint) {
            return None;
        }
        self.get(gpu_tile, gpu_tile.format)
    }

    /// Advance the MRT frame epoch and prune stale unused combinatorial tuples
    /// when `mrt_entries` exceeds [`MAX_MRT_BIND_GROUPS`]. Must be called once
    /// per frame *before* `ensure_mrt` so active-frame tuples are never evicted.
    pub(super) fn begin_mrt_frame(&mut self) {
        self.mrt_epoch = self.mrt_epoch.wrapping_add(1);
        if self.mrt_entries.len() > MAX_MRT_BIND_GROUPS {
            let prev_epoch = self.mrt_epoch.wrapping_sub(1);
            self.mrt_entries.retain(|_, (weaks, _, epoch)| {
                *epoch == prev_epoch && weaks.iter().all(|w| w.strong_count() > 0)
            });
        }
    }

    /// Ensure a 4-texture `R32Float` MRT bind group exists for `tiles`.
    pub(super) fn ensure_mrt(&mut self, device: &wgpu::Device, tiles: [&Arc<GpuTile>; 4]) {
        if tiles.iter().any(|t| t.format != PixelFormat::R32Float) {
            return;
        }
        let key = tiles.map(|t| t.tile_id);
        if let Some(entry) = self.mrt_entries.get_mut(&key)
            && mrt_entry_matches(entry, tiles)
        {
            entry.2 = self.mrt_epoch;
            return;
        }
        let entries: [wgpu::BindGroupEntry<'_>; 4] =
            std::array::from_fn(|i| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: wgpu::BindingResource::TextureView(&tiles[i].view),
            });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tile R32Float MRT 4-band bind group"),
            layout: &self.r32f_mrt_layout,
            entries: &entries,
        });
        self.mrt_entries
            .insert(key, (tiles.map(Arc::downgrade), bg, self.mrt_epoch));
    }

    /// Look up the cached 4-texture MRT bind group for `tiles`.
    pub(super) fn get_mrt(&self, tiles: [&Arc<GpuTile>; 4]) -> Option<&wgpu::BindGroup> {
        let key = tiles.map(|t| t.tile_id);
        self.mrt_entries
            .get(&key)
            .and_then(|entry| mrt_entry_matches(entry, tiles).then_some(&entry.1))
    }

    /// Drop entries whose backing `GpuTile` has been evicted and released.
    pub fn sweep_stale(&mut self) {
        self.entries
            .retain(|_, (weak, _, _)| weak.strong_count() > 0);
        self.mrt_entries
            .retain(|_, (weaks, _, _)| weaks.iter().all(|w| w.strong_count() > 0));
    }

    /// Total cached bind groups across all pixel formats.
    #[cfg(any(test, feature = "test-support"))]
    pub fn len(&self) -> usize {
        self.entries.len() + self.mrt_entries.len()
    }

    /// Whether the cache currently holds zero tile bind groups.
    #[cfg(any(test, feature = "test-support"))]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty() && self.mrt_entries.is_empty()
    }
}
