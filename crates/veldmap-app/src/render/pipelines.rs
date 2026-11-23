//! GPU bind-group layouts, samplers, and render pipelines for
//! [`super::GlobeRenderer`].

use super::mesh::TileVertex;
use super::viewport::{CameraUniform, TileUniform};
use super::{COMPOSITE_WGSL, GLOBE_WGSL};

struct GlobePipelineSpec<'a> {
    label: &'a str,
    fragment_entry: &'a str,
    blend: wgpu::BlendState,
    depth_write_enabled: bool,
    depth_compare: wgpu::CompareFunction,
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
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(spec.depth_write_enabled),
            depth_compare: Some(spec.depth_compare),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

pub(super) struct GlobePipelines {
    pub(super) camera_layout: wgpu::BindGroupLayout,
    pub(super) tile_uniform_layout: wgpu::BindGroupLayout,
    pub(super) tile_uniform_stride: u32,
    pub(super) tile_texture_layout: wgpu::BindGroupLayout,
    pub(super) tile_sampler: wgpu::Sampler,
    pub(super) body_ellipsoid_pipeline: wgpu::RenderPipeline,
    pub(super) body_textured_pipeline: wgpu::RenderPipeline,
    pub(super) grid_pipeline: wgpu::RenderPipeline,
    pub(super) composite_layout: wgpu::BindGroupLayout,
    pub(super) composite_sampler: wgpu::Sampler,
    pub(super) composite_pipeline: wgpu::RenderPipeline,
}

fn create_globe_layouts(
    device: &wgpu::Device,
    raw_tile_size: u32,
) -> (
    wgpu::BindGroupLayout,
    wgpu::BindGroupLayout,
    wgpu::BindGroupLayout,
) {
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
                min_binding_size: wgpu::BufferSize::new(u64::from(raw_tile_size)),
            },
            count: None,
        }],
    });

    let tile_texture_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("globe tile texture layout"),
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

    (camera_layout, tile_uniform_layout, tile_texture_layout)
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
        let align = device.limits().min_uniform_buffer_offset_alignment.max(1);
        let raw_tile_size = std::mem::size_of::<TileUniform>() as u32;
        let tile_uniform_stride = raw_tile_size.div_ceil(align) * align;

        let (camera_layout, tile_uniform_layout, tile_texture_layout) =
            create_globe_layouts(device, raw_tile_size);

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
                bind_group_layouts: &[Some(&camera_layout), Some(&tile_uniform_layout)],
                immediate_size: 0,
            });

        let textured_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("globe textured pipeline layout"),
                bind_group_layouts: &[
                    Some(&camera_layout),
                    Some(&tile_uniform_layout),
                    Some(&tile_texture_layout),
                ],
                immediate_size: 0,
            });

        let body_ellipsoid_pipeline = create_globe_pipeline(
            device,
            &globe_shader,
            &untextured_pipeline_layout,
            GlobePipelineSpec {
                label: "globe body ellipsoid pipeline (Reverse-Z)",
                fragment_entry: "fs_body_ellipsoid",
                blend: wgpu::BlendState::REPLACE,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::GreaterEqual,
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
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::GreaterEqual,
            },
        );

        let grid_pipeline = create_globe_pipeline(
            device,
            &globe_shader,
            &untextured_pipeline_layout,
            GlobePipelineSpec {
                label: "globe post-overlay procedural graticule pipeline",
                fragment_entry: "fs_grid",
                blend: wgpu::BlendState::ALPHA_BLENDING,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
            },
        );

        let (composite_layout, composite_sampler, composite_pipeline) =
            create_composite_resources(device, target_format);

        Self {
            camera_layout,
            tile_uniform_layout,
            tile_uniform_stride,
            tile_texture_layout,
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
