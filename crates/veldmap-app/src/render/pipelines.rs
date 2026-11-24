//! GPU bind-group layouts, `PixelFormat`-partitioned texture bind-group cache,
//! samplers, and render pipelines for [`super::GlobeRenderer`] and
//! [`super::overlay::OverlayRenderer`].

use std::collections::HashMap;
use std::sync::{Arc, Weak};

use veldmap_core::tile::PixelFormat;

use crate::gpu_tiles::GpuTile;

use super::mesh::TileVertex;
use super::uniforms::{CameraUniform, OverlayPatchUniform, TileUniform};
use super::{COMPOSITE_WGSL, GLOBE_WGSL};

struct GlobePipelineSpec<'a> {
    label: &'a str,
    fragment_entry: &'a str,
    blend: wgpu::BlendState,
    depth_stencil: Option<wgpu::DepthStencilState>,
}

fn create_globe_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
    spec: GlobePipelineSpec<'_>,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(spec.label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            buffers: &[Some(TileVertex::LAYOUT)],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some(spec.fragment_entry),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                blend: Some(spec.blend),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: Some(wgpu::Face::Back),
            front_face: wgpu::FrontFace::Ccw,
            ..Default::default()
        },
        depth_stencil: spec.depth_stencil,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

/// `PixelFormat`-partitioned cache of `Weak<GpuTile>`-guarded `wgpu::BindGroup`
/// entries so filterable sRGB display tiles (`Rgba8UnormSrgb`), non-filterable
/// scientific float tiles (`R32Float`), and unsigned integer categorical tiles
/// (`R8Uint`/`R16Uint`) never share incompatible bind-group layouts.
#[derive(Default)]
pub(super) struct TileBindGroupCache {
    rgba8_srgb: HashMap<u64, (Weak<GpuTile>, wgpu::BindGroup)>,
    r32_float: HashMap<u64, (Weak<GpuTile>, wgpu::BindGroup)>,
    r8_uint: HashMap<u64, (Weak<GpuTile>, wgpu::BindGroup)>,
    r16_uint: HashMap<u64, (Weak<GpuTile>, wgpu::BindGroup)>,
}

impl TileBindGroupCache {
    fn map_for(&self, format: PixelFormat) -> &HashMap<u64, (Weak<GpuTile>, wgpu::BindGroup)> {
        match format {
            PixelFormat::Rgba8UnormSrgb => &self.rgba8_srgb,
            PixelFormat::R32Float => &self.r32_float,
            PixelFormat::R8Uint => &self.r8_uint,
            PixelFormat::R16Uint => &self.r16_uint,
        }
    }

    fn map_for_mut(
        &mut self,
        format: PixelFormat,
    ) -> &mut HashMap<u64, (Weak<GpuTile>, wgpu::BindGroup)> {
        match format {
            PixelFormat::Rgba8UnormSrgb => &mut self.rgba8_srgb,
            PixelFormat::R32Float => &mut self.r32_float,
            PixelFormat::R8Uint => &mut self.r8_uint,
            PixelFormat::R16Uint => &mut self.r16_uint,
        }
    }

    /// Ensure a bind group matching `gpu_tile.format` exists for `gpu_tile`.
    pub(super) fn ensure(
        &mut self,
        device: &wgpu::Device,
        pipelines: &GlobePipelines,
        gpu_tile: &Arc<GpuTile>,
    ) {
        let format = gpu_tile.format;
        let map = self.map_for_mut(format);
        let reuse = map.get(&gpu_tile.tile_id).is_some_and(|(weak, _)| {
            weak.strong_count() > 0 && std::ptr::eq(weak.as_ptr(), Arc::as_ptr(gpu_tile))
        });
        if reuse {
            return;
        }
        let bg = match format {
            PixelFormat::Rgba8UnormSrgb => device.create_bind_group(&wgpu::BindGroupDescriptor {
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
            }),
            PixelFormat::R32Float => device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tile R32Float bind group"),
                layout: &pipelines.float_texture_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&gpu_tile.view),
                }],
            }),
            PixelFormat::R8Uint | PixelFormat::R16Uint => {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("tile Uint bind group"),
                    layout: &pipelines.uint_texture_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&gpu_tile.view),
                    }],
                })
            }
        };
        map.insert(gpu_tile.tile_id, (Arc::downgrade(gpu_tile), bg));
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
        self.map_for(expected_format)
            .get(&gpu_tile.tile_id)
            .and_then(|(weak, bg)| {
                (weak.strong_count() > 0 && std::ptr::eq(weak.as_ptr(), Arc::as_ptr(gpu_tile)))
                    .then_some(bg)
            })
    }

    /// Drop entries whose backing `GpuTile` has been evicted and released.
    pub(super) fn sweep_stale(&mut self) {
        self.rgba8_srgb
            .retain(|_, (weak, _)| weak.strong_count() > 0);
        self.r32_float
            .retain(|_, (weak, _)| weak.strong_count() > 0);
        self.r8_uint.retain(|_, (weak, _)| weak.strong_count() > 0);
        self.r16_uint.retain(|_, (weak, _)| weak.strong_count() > 0);
    }

    /// Total cached bind groups across all `PixelFormat` partitions.
    pub(super) fn len(&self) -> usize {
        self.rgba8_srgb.len() + self.r32_float.len() + self.r8_uint.len() + self.r16_uint.len()
    }
}

pub(super) struct GlobePipelines {
    pub(super) camera_layout: wgpu::BindGroupLayout,
    pub(super) tile_uniform_layout: wgpu::BindGroupLayout,
    pub(super) overlay_uniform_layout: wgpu::BindGroupLayout,
    pub(super) tile_texture_layout: wgpu::BindGroupLayout,
    pub(super) float_texture_layout: wgpu::BindGroupLayout,
    pub(super) uint_texture_layout: wgpu::BindGroupLayout,
    pub(super) tile_sampler: wgpu::Sampler,
    pub(super) body_ellipsoid_pipeline: wgpu::RenderPipeline,
    pub(super) body_textured_pipeline: wgpu::RenderPipeline,
    pub(super) grid_pipeline: wgpu::RenderPipeline,
    pub(super) composite_layout: wgpu::BindGroupLayout,
    pub(super) composite_sampler: wgpu::Sampler,
    pub(super) composite_pipeline: wgpu::RenderPipeline,
}

struct SharedLayouts {
    camera_layout: wgpu::BindGroupLayout,
    tile_uniform_layout: wgpu::BindGroupLayout,
    overlay_uniform_layout: wgpu::BindGroupLayout,
    tile_texture_layout: wgpu::BindGroupLayout,
    float_texture_layout: wgpu::BindGroupLayout,
    uint_texture_layout: wgpu::BindGroupLayout,
}

fn create_shared_layouts(device: &wgpu::Device) -> SharedLayouts {
    let camera_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("globe camera bind group layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<CameraUniform>() as u64
                    ),
                },
                count: None,
            }],
        });

    let tile_uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("globe per-tile dynamic uniform layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<TileUniform>() as u64),
            },
            count: None,
        }],
    });

    let overlay_uniform_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("overlay patch dynamic uniform layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<OverlayPatchUniform>() as u64,
                    ),
                },
                count: None,
            }],
        });

    let tile_texture_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

    let float_texture_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("tile R32Float non-filterable texture layout"),
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

    let uint_texture_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

    SharedLayouts {
        camera_layout,
        tile_uniform_layout,
        overlay_uniform_layout,
        tile_texture_layout,
        float_texture_layout,
        uint_texture_layout,
    }
}

fn create_composite_resources(
    device: &wgpu::Device,
    target_format: wgpu::TextureFormat,
) -> (wgpu::BindGroupLayout, wgpu::Sampler, wgpu::RenderPipeline) {
    let composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("composite wgsl"),
        source: wgpu::ShaderSource::Wgsl(COMPOSITE_WGSL.into()),
    });

    let composite_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("composite bind group layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                count: None,
            },
        ],
    });

    let composite_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("composite nearest sampler"),
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        ..Default::default()
    });

    let composite_pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("composite pipeline layout"),
            bind_group_layouts: &[Some(&composite_layout)],
            immediate_size: 0,
        });

    let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("composite pipeline"),
        layout: Some(&composite_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &composite_shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &composite_shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: target_format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });

    (composite_layout, composite_sampler, composite_pipeline)
}

impl GlobePipelines {
    pub(super) fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let layouts = create_shared_layouts(device);

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

        let globe_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("globe wgsl"),
            source: wgpu::ShaderSource::Wgsl(GLOBE_WGSL.into()),
        });

        let untextured_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("globe untextured pipeline layout"),
                bind_group_layouts: &[
                    Some(&layouts.camera_layout),
                    Some(&layouts.tile_uniform_layout),
                ],
                immediate_size: 0,
            });

        let textured_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("globe textured pipeline layout"),
                bind_group_layouts: &[
                    Some(&layouts.camera_layout),
                    Some(&layouts.tile_uniform_layout),
                    Some(&layouts.tile_texture_layout),
                ],
                immediate_size: 0,
            });

        let body_depth = || {
            Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            })
        };

        let body_ellipsoid_pipeline = create_globe_pipeline(
            device,
            &globe_shader,
            &untextured_pipeline_layout,
            GlobePipelineSpec {
                label: "globe body ellipsoid pipeline (Reverse-Z)",
                fragment_entry: "fs_body_ellipsoid",
                blend: wgpu::BlendState::REPLACE,
                depth_stencil: body_depth(),
            },
        );

        let body_textured_pipeline = create_globe_pipeline(
            device,
            &globe_shader,
            &textured_pipeline_layout,
            GlobePipelineSpec {
                label: "globe body textured pipeline (Reverse-Z)",
                fragment_entry: "fs_body_textured",
                blend: wgpu::BlendState::REPLACE,
                depth_stencil: body_depth(),
            },
        );

        // Post-body graticule runs inside the depth-less (`depth_stencil_attachment: None`)
        // color pass so draped overlays and future interleaved analytical passes
        // share the exact same color-only pass contract.
        let grid_pipeline = create_globe_pipeline(
            device,
            &globe_shader,
            &untextured_pipeline_layout,
            GlobePipelineSpec {
                label: "globe post-overlay procedural graticule pipeline (depth-less)",
                fragment_entry: "fs_grid",
                blend: wgpu::BlendState::ALPHA_BLENDING,
                depth_stencil: None,
            },
        );

        let (composite_layout, composite_sampler, composite_pipeline) =
            create_composite_resources(device, target_format);

        Self {
            camera_layout: layouts.camera_layout,
            tile_uniform_layout: layouts.tile_uniform_layout,
            overlay_uniform_layout: layouts.overlay_uniform_layout,
            tile_texture_layout: layouts.tile_texture_layout,
            float_texture_layout: layouts.float_texture_layout,
            uint_texture_layout: layouts.uint_texture_layout,
            tile_sampler,
            body_ellipsoid_pipeline,
            body_textured_pipeline,
            grid_pipeline,
            composite_layout,
            composite_sampler,
            composite_pipeline,
        }
    }
}
