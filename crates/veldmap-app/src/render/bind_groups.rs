//! `Weak<GpuTile>`-guarded multi-format GPU texture bind-group cache shared
//! across planetary basemap tiles, draped display/scientific/categorical
//! overlays, and future orthographic 2D views via `egui_wgpu::CallbackResources`.

use std::collections::HashMap;
use std::sync::{Arc, Weak};

use veldmap_core::tile::PixelFormat;

use crate::gpu_tiles::GpuTile;

type CachedBindGroupEntry = (Weak<GpuTile>, PixelFormat, wgpu::BindGroup);

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

/// `Weak<GpuTile>`-guarded cache of `wgpu::BindGroup` entries across all
/// supported tile pixel formats (`Rgba8UnormSrgb`, `R32Float`, `R8Uint`,
/// `R16Uint`), keyed by process-unique `GpuTile::tile_id`.
///
/// Hoisted into `egui_wgpu::CallbackResources` so `GlobeRenderer` and future
/// multi-viewport renderers (e.g. `Image2DRenderer`) share identical layouts
/// and cached texture bind groups without duplicating GPU resources.
pub struct TileBindGroupCache {
    rgba8_layout: wgpu::BindGroupLayout,
    r32f_layout: wgpu::BindGroupLayout,
    uint_layout: wgpu::BindGroupLayout,
    tile_sampler: wgpu::Sampler,
    entries: HashMap<u64, CachedBindGroupEntry>,
}

impl TileBindGroupCache {
    /// Allocate the shared format-specific bind-group layouts (`Rgba8UnormSrgb`
    /// filterable + linear sampler, `R32Float` unfilterable `textureLoad`, and
    /// `Uint` `textureLoad` for `R8Uint`/`R16Uint`).
    pub fn new(device: &wgpu::Device) -> Self {
        let rgba8_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tile Rgba8UnormSrgb texture layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let r32f_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tile R32Float unfilterable texture layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });

        let uint_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tile Uint categorical texture layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Uint,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
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
            uint_layout,
            tile_sampler,
            entries: HashMap::new(),
        }
    }

    pub(super) fn rgba8_layout(&self) -> &wgpu::BindGroupLayout {
        &self.rgba8_layout
    }

    pub(super) fn r32f_layout(&self) -> &wgpu::BindGroupLayout {
        &self.r32f_layout
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

    /// Drop entries whose backing `GpuTile` has been evicted and released.
    pub fn sweep_stale(&mut self) {
        self.entries
            .retain(|_, (weak, _, _)| weak.strong_count() > 0);
    }

    /// Total cached bind groups across all pixel formats.
    #[cfg(any(test, feature = "test-support"))]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache currently holds zero tile bind groups.
    #[cfg(any(test, feature = "test-support"))]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}
