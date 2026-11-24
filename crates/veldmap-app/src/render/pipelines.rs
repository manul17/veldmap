//! GPU bind-group layouts, samplers, and render pipelines for
//! [`super::GlobeRenderer`] and [`super::overlay::OverlayRenderer`].

use super::mesh::TileVertex;
use super::overlay::OverlayVertex;
use super::uniforms::{CameraUniform, OverlayPatchUniform, TileUniform};
use super::vector::GpuVectorVertex;
use super::{COMPOSITE_WGSL, GLOBE_WGSL, OVERLAY_WGSL, VECTOR_WGSL};

struct GlobePipelineSpec<'a> {
    label: &'a str,
    vertex_layout: wgpu::VertexBufferLayout<'static>,
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
            buffers: &[Some(spec.vertex_layout)],
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

pub(super) struct GlobePipelines {
    pub(super) camera_layout: wgpu::BindGroupLayout,
    pub(super) tile_uniform_layout: wgpu::BindGroupLayout,
    pub(super) overlay_uniform_layout: wgpu::BindGroupLayout,
    pub(super) tile_texture_layout: wgpu::BindGroupLayout,
    pub(super) tile_sampler: wgpu::Sampler,
    pub(super) body_ellipsoid_pipeline: wgpu::RenderPipeline,
    pub(super) body_textured_pipeline: wgpu::RenderPipeline,
    pub(super) overlay_pipeline: wgpu::RenderPipeline,
    pub(super) vector_fill_pipeline: wgpu::RenderPipeline,
    pub(super) vector_line_pipeline: wgpu::RenderPipeline,
    pub(super) vector_point_pipeline: wgpu::RenderPipeline,
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
}

fn uniform_buffer_layout<T>(
    device: &wgpu::Device,
    label: &str,
    dynamic: bool,
) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: dynamic,
                min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<T>() as u64),
            },
            count: None,
        }],
    })
}

fn create_shared_layouts(device: &wgpu::Device) -> SharedLayouts {
    let camera_layout =
        uniform_buffer_layout::<CameraUniform>(device, "globe camera bind group layout", false);
    let tile_uniform_layout =
        uniform_buffer_layout::<TileUniform>(device, "globe per-tile dynamic uniform layout", true);
    let overlay_uniform_layout = uniform_buffer_layout::<OverlayPatchUniform>(
        device,
        "overlay patch dynamic uniform layout",
        true,
    );

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

    SharedLayouts {
        camera_layout,
        tile_uniform_layout,
        overlay_uniform_layout,
        tile_texture_layout,
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

        let overlay_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("overlay warp-mesh wgsl"),
            source: wgpu::ShaderSource::Wgsl(OVERLAY_WGSL.into()),
        });

        let overlay_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("overlay warp-mesh pipeline layout"),
                bind_group_layouts: &[
                    Some(&layouts.camera_layout),
                    Some(&layouts.overlay_uniform_layout),
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
                vertex_layout: TileVertex::LAYOUT,
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
                vertex_layout: TileVertex::LAYOUT,
                fragment_entry: "fs_body_textured",
                blend: wgpu::BlendState::REPLACE,
                depth_stencil: body_depth(),
            },
        );

        let overlay_pipeline = create_globe_pipeline(
            device,
            &overlay_shader,
            &overlay_pipeline_layout,
            GlobePipelineSpec {
                label: "overlay warp-mesh pipeline (depth-less)",
                vertex_layout: OverlayVertex::LAYOUT,
                fragment_entry: "fs_overlay",
                blend: wgpu::BlendState::ALPHA_BLENDING,
                depth_stencil: None,
            },
        );

        let vector_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vector draped wgsl"),
            source: wgpu::ShaderSource::Wgsl(VECTOR_WGSL.into()),
        });
        let vector_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("vector draped pipeline layout"),
                bind_group_layouts: &[
                    Some(&layouts.camera_layout),
                    Some(&layouts.overlay_uniform_layout),
                ],
                immediate_size: 0,
            });
        let vector_fill_pipeline = create_globe_pipeline(
            device,
            &vector_shader,
            &vector_pipeline_layout,
            GlobePipelineSpec {
                label: "vector polygon fill pipeline (depth-less)",
                vertex_layout: GpuVectorVertex::LAYOUT,
                fragment_entry: "fs_vector",
                blend: wgpu::BlendState::ALPHA_BLENDING,
                depth_stencil: None,
            },
        );
        let vector_line_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("vector polyline stroke pipeline (depth-less)"),
            layout: Some(&vector_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vector_shader,
                entry_point: Some("vs_main"),
                buffers: &[Some(GpuVectorVertex::LAYOUT)],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &vector_shader,
                entry_point: Some("fs_vector"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let vector_point_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("vector instanced point marker pipeline (depth-less)"),
                layout: Some(&vector_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &vector_shader,
                    entry_point: Some("vs_point"),
                    buffers: &[Some(GpuVectorVertex::INSTANCE_LAYOUT)],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &vector_shader,
                    entry_point: Some("fs_vector"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8UnormSrgb,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });

        // Post-body graticule runs inside the depth-less (`depth_stencil_attachment: None`)
        // color pass directly above draped raster overlays and vector layers.
        let grid_pipeline = create_globe_pipeline(
            device,
            &globe_shader,
            &untextured_pipeline_layout,
            GlobePipelineSpec {
                label: "globe post-overlay procedural graticule pipeline (depth-less)",
                vertex_layout: TileVertex::LAYOUT,
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
            tile_sampler,
            body_ellipsoid_pipeline,
            body_textured_pipeline,
            overlay_pipeline,
            vector_fill_pipeline,
            vector_line_pipeline,
            vector_point_pipeline,
            grid_pipeline,
            composite_layout,
            composite_sampler,
            composite_pipeline,
        }
    }
}
