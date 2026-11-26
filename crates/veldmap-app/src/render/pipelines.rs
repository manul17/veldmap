//! GPU bind-group layouts, samplers, and render pipelines for
//! [`super::GlobeRenderer`], [`super::overlay::OverlayRenderer`], and
//! [`super::vector::VectorRenderer`].

use super::bind_groups::TileBindGroupCache;
use super::mesh::TileVertex;
use super::overlay::OverlayVertex;
use super::uniforms::{
    CameraUniform, CategoricalPatchUniform, OverlayPatchUniform, PaletteUniform,
    ScientificPatchUniform, TileUniform, VectorUniform,
};
use super::vector::{GpuRibbonSegment, GpuVectorVertex};
use super::{
    COMPOSITE_WGSL, GLOBE_WGSL, OVERLAY_CATEGORICAL_WGSL, OVERLAY_SCIENTIFIC_WGSL, OVERLAY_WGSL,
    VECTOR_WGSL,
};

struct GlobePipelineSpec<'a> {
    label: &'a str,
    vertex_entry: &'a str,
    vertex_layout: wgpu::VertexBufferLayout<'static>,
    fragment_entry: &'a str,
    topology: wgpu::PrimitiveTopology,
    cull_mode: Option<wgpu::Face>,
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
            entry_point: Some(spec.vertex_entry),
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
            topology: spec.topology,
            cull_mode: spec.cull_mode,
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
    pub(super) layouts: SharedLayouts,
    pub(super) body_ellipsoid_pipeline: wgpu::RenderPipeline,
    pub(super) body_textured_pipeline: wgpu::RenderPipeline,
    pub(super) overlay_pipeline: wgpu::RenderPipeline,
    pub(super) overlay_scientific_pipeline: wgpu::RenderPipeline,
    pub(super) overlay_categorical_pipeline: wgpu::RenderPipeline,
    pub(super) vector_fill_pipeline: wgpu::RenderPipeline,
    pub(super) vector_line_pipeline: wgpu::RenderPipeline,
    pub(super) vector_point_pipeline: wgpu::RenderPipeline,
    pub(super) grid_pipeline: wgpu::RenderPipeline,
    pub(super) composite_layout: wgpu::BindGroupLayout,
    pub(super) composite_sampler: wgpu::Sampler,
    pub(super) composite_pipeline: wgpu::RenderPipeline,
}

pub(super) struct SharedLayouts {
    pub(super) camera_layout: wgpu::BindGroupLayout,
    pub(super) tile_uniform_layout: wgpu::BindGroupLayout,
    pub(super) overlay_uniform_layout: wgpu::BindGroupLayout,
    pub(super) scientific_uniform_layout: wgpu::BindGroupLayout,
    pub(super) categorical_uniform_layout: wgpu::BindGroupLayout,
    pub(super) palette_uniform_layout: wgpu::BindGroupLayout,
    pub(super) vector_uniform_layout: wgpu::BindGroupLayout,
}

fn uniform_buffer_layout<T>(
    device: &wgpu::Device,
    label: &str,
    visibility: wgpu::ShaderStages,
    dynamic: bool,
) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: dynamic,
                min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<T>() as u64),
            },
            count: None,
        }],
    })
}

fn non_filtering_texture_sampler_layout(
    device: &wgpu::Device,
    label: &str,
) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
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
    })
}

fn create_shared_layouts(device: &wgpu::Device) -> SharedLayouts {
    let vf = wgpu::ShaderStages::VERTEX_FRAGMENT;
    let camera_layout =
        uniform_buffer_layout::<CameraUniform>(device, "globe camera bind group layout", vf, false);
    let tile_uniform_layout = uniform_buffer_layout::<TileUniform>(
        device,
        "globe per-tile dynamic uniform layout",
        vf,
        true,
    );
    let overlay_uniform_layout = uniform_buffer_layout::<OverlayPatchUniform>(
        device,
        "overlay patch dynamic uniform layout",
        vf,
        true,
    );
    let scientific_uniform_layout = uniform_buffer_layout::<ScientificPatchUniform>(
        device,
        "scientific overlay patch dynamic uniform layout",
        vf,
        true,
    );
    let categorical_uniform_layout = uniform_buffer_layout::<CategoricalPatchUniform>(
        device,
        "categorical overlay patch dynamic uniform layout",
        vf,
        true,
    );
    let palette_uniform_layout = uniform_buffer_layout::<PaletteUniform>(
        device,
        "overlay palette LUT uniform layout",
        wgpu::ShaderStages::FRAGMENT,
        false,
    );
    let vector_uniform_layout =
        uniform_buffer_layout::<VectorUniform>(device, "vector dynamic uniform layout", vf, true);

    SharedLayouts {
        camera_layout,
        tile_uniform_layout,
        overlay_uniform_layout,
        scientific_uniform_layout,
        categorical_uniform_layout,
        palette_uniform_layout,
        vector_uniform_layout,
    }
}

fn create_body_and_grid_pipelines(
    device: &wgpu::Device,
    layouts: &SharedLayouts,
    rgba8_tile_layout: &wgpu::BindGroupLayout,
) -> (
    wgpu::RenderPipeline,
    wgpu::RenderPipeline,
    wgpu::RenderPipeline,
) {
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

    let textured_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("globe textured pipeline layout"),
        bind_group_layouts: &[
            Some(&layouts.camera_layout),
            Some(&layouts.tile_uniform_layout),
            Some(rgba8_tile_layout),
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
            vertex_entry: "vs_main",
            vertex_layout: TileVertex::LAYOUT,
            fragment_entry: "fs_body_ellipsoid",
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: Some(wgpu::Face::Back),
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
            vertex_entry: "vs_main",
            vertex_layout: TileVertex::LAYOUT,
            fragment_entry: "fs_body_textured",
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: Some(wgpu::Face::Back),
            blend: wgpu::BlendState::REPLACE,
            depth_stencil: body_depth(),
        },
    );

    // Post-body graticule runs inside the depth-less (`depth_stencil_attachment: None`)
    // color pass directly above draped raster overlays and vector layers.
    let grid_pipeline = create_globe_pipeline(
        device,
        &globe_shader,
        &untextured_pipeline_layout,
        GlobePipelineSpec {
            label: "globe post-overlay procedural graticule pipeline (depth-less)",
            vertex_entry: "vs_main",
            vertex_layout: TileVertex::LAYOUT,
            fragment_entry: "fs_grid",
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: Some(wgpu::Face::Back),
            blend: wgpu::BlendState::ALPHA_BLENDING,
            depth_stencil: None,
        },
    );

    (
        body_ellipsoid_pipeline,
        body_textured_pipeline,
        grid_pipeline,
    )
}

fn create_overlay_track_pipeline(
    device: &wgpu::Device,
    shader_label: &str,
    shader_source: &str,
    layout_label: &str,
    bind_group_layouts: &[Option<&wgpu::BindGroupLayout>],
    pipeline_label: &'static str,
    fragment_entry: &'static str,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(shader_label),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(layout_label),
        bind_group_layouts,
        immediate_size: 0,
    });
    create_globe_pipeline(
        device,
        &shader,
        &layout,
        GlobePipelineSpec {
            label: pipeline_label,
            vertex_entry: "vs_main",
            vertex_layout: OverlayVertex::LAYOUT,
            fragment_entry,
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: Some(wgpu::Face::Back),
            blend: wgpu::BlendState::ALPHA_BLENDING,
            depth_stencil: None,
        },
    )
}

fn create_overlay_pipelines(
    device: &wgpu::Device,
    layouts: &SharedLayouts,
    tile_bind_groups: &TileBindGroupCache,
) -> (
    wgpu::RenderPipeline,
    wgpu::RenderPipeline,
    wgpu::RenderPipeline,
) {
    let display_pipeline = create_overlay_track_pipeline(
        device,
        "overlay warp-mesh wgsl",
        OVERLAY_WGSL,
        "overlay warp-mesh pipeline layout",
        &[
            Some(&layouts.camera_layout),
            Some(&layouts.overlay_uniform_layout),
            Some(tile_bind_groups.rgba8_layout()),
        ],
        "overlay display warp-mesh pipeline (depth-less)",
        "fs_overlay",
    );
    let scientific_pipeline = create_overlay_track_pipeline(
        device,
        "overlay scientific R32Float wgsl",
        OVERLAY_SCIENTIFIC_WGSL,
        "overlay scientific pipeline layout",
        &[
            Some(&layouts.camera_layout),
            Some(&layouts.scientific_uniform_layout),
            Some(tile_bind_groups.r32f_layout()),
            Some(&layouts.palette_uniform_layout),
        ],
        "overlay scientific R32Float pipeline (depth-less)",
        "fs_scientific",
    );
    let categorical_pipeline = create_overlay_track_pipeline(
        device,
        "overlay categorical Uint wgsl",
        OVERLAY_CATEGORICAL_WGSL,
        "overlay categorical pipeline layout",
        &[
            Some(&layouts.camera_layout),
            Some(&layouts.categorical_uniform_layout),
            Some(tile_bind_groups.uint_layout()),
            Some(&layouts.palette_uniform_layout),
        ],
        "overlay categorical Uint pipeline (depth-less)",
        "fs_categorical",
    );

    (display_pipeline, scientific_pipeline, categorical_pipeline)
}

fn create_vector_pipelines(
    device: &wgpu::Device,
    layouts: &SharedLayouts,
) -> (
    wgpu::RenderPipeline,
    wgpu::RenderPipeline,
    wgpu::RenderPipeline,
) {
    let vector_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("vector draped wgsl"),
        source: wgpu::ShaderSource::Wgsl(VECTOR_WGSL.into()),
    });
    let vector_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("vector draped pipeline layout"),
        bind_group_layouts: &[
            Some(&layouts.camera_layout),
            Some(&layouts.vector_uniform_layout),
        ],
        immediate_size: 0,
    });
    let vector_fill_pipeline = create_globe_pipeline(
        device,
        &vector_shader,
        &vector_pipeline_layout,
        GlobePipelineSpec {
            label: "vector polygon fill pipeline (depth-less)",
            vertex_entry: "vs_main",
            vertex_layout: GpuVectorVertex::LAYOUT,
            fragment_entry: "fs_vector",
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: Some(wgpu::Face::Back),
            blend: wgpu::BlendState::ALPHA_BLENDING,
            depth_stencil: None,
        },
    );
    let vector_line_pipeline = create_globe_pipeline(
        device,
        &vector_shader,
        &vector_pipeline_layout,
        GlobePipelineSpec {
            label: "vector HiDPI ribbon stroke pipeline (depth-less)",
            vertex_entry: "vs_ribbon",
            vertex_layout: GpuRibbonSegment::INSTANCE_LAYOUT,
            fragment_entry: "fs_vector",
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: None,
            blend: wgpu::BlendState::ALPHA_BLENDING,
            depth_stencil: None,
        },
    );
    let vector_point_pipeline = create_globe_pipeline(
        device,
        &vector_shader,
        &vector_pipeline_layout,
        GlobePipelineSpec {
            label: "vector instanced point marker pipeline (depth-less)",
            vertex_entry: "vs_point",
            vertex_layout: GpuVectorVertex::INSTANCE_LAYOUT,
            fragment_entry: "fs_vector",
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: Some(wgpu::Face::Back),
            blend: wgpu::BlendState::ALPHA_BLENDING,
            depth_stencil: None,
        },
    );
    (
        vector_fill_pipeline,
        vector_line_pipeline,
        vector_point_pipeline,
    )
}

fn create_composite_resources(
    device: &wgpu::Device,
    target_format: wgpu::TextureFormat,
) -> (wgpu::BindGroupLayout, wgpu::Sampler, wgpu::RenderPipeline) {
    let composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("composite wgsl"),
        source: wgpu::ShaderSource::Wgsl(COMPOSITE_WGSL.into()),
    });

    let composite_layout =
        non_filtering_texture_sampler_layout(device, "composite bind group layout");

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
    pub(super) fn new(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        tile_bind_groups: &TileBindGroupCache,
    ) -> Self {
        let layouts = create_shared_layouts(device);

        let (body_ellipsoid_pipeline, body_textured_pipeline, grid_pipeline) =
            create_body_and_grid_pipelines(device, &layouts, tile_bind_groups.rgba8_layout());
        let (overlay_pipeline, overlay_scientific_pipeline, overlay_categorical_pipeline) =
            create_overlay_pipelines(device, &layouts, tile_bind_groups);
        let (vector_fill_pipeline, vector_line_pipeline, vector_point_pipeline) =
            create_vector_pipelines(device, &layouts);
        let (composite_layout, composite_sampler, composite_pipeline) =
            create_composite_resources(device, target_format);

        Self {
            layouts,
            body_ellipsoid_pipeline,
            body_textured_pipeline,
            overlay_pipeline,
            overlay_scientific_pipeline,
            overlay_categorical_pipeline,
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
