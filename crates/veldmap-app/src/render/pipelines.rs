//! GPU bind-group layouts, samplers, and render pipelines for
//! [`super::GlobeRenderer`], [`super::overlay::OverlayRenderer`], and
//! [`super::vector::VectorRenderer`].

use super::bind_groups::TileBindGroupCache;
use super::mesh::TileVertex;
use super::overlay::OverlayVertex;
use super::uniforms::{
    CameraUniform, PaletteUniform, ScratchClearUniform, UNIFORM_SLOT_COUNT, UniformSlot,
};
use super::vector_pack::{GpuRibbonSegment, GpuVectorVertex};
use super::{
    BAND_MATH_COMPOSITE_WGSL, BAND_MATH_MASK_WGSL, BAND_MATH_SCRATCH_MRT_WGSL,
    BAND_MATH_SCRATCH_WGSL, COMPOSITE_WGSL, GLOBE_WGSL, OVERLAY_CATEGORICAL_WGSL,
    OVERLAY_SCIENTIFIC_WGSL, OVERLAY_WGSL, SCRATCH_NAN_FILL_WGSL, VECTOR_WGSL,
};

fn wgsl_module(device: &wgpu::Device, label: &str, source: &str) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    })
}

fn pipeline_layout(
    device: &wgpu::Device,
    label: &str,
    bind_group_layouts: &[Option<&wgpu::BindGroupLayout>],
) -> wgpu::PipelineLayout {
    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts,
        immediate_size: 0,
    })
}

fn srgb_target(blend: wgpu::BlendState) -> Option<wgpu::ColorTargetState> {
    Some(wgpu::ColorTargetState {
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        blend: Some(blend),
        write_mask: wgpu::ColorWrites::ALL,
    })
}

struct MeshPipelineSpec<'a> {
    label: &'a str,
    vertex_entry: &'a str,
    vertex_layout: wgpu::VertexBufferLayout<'static>,
    fragment_entry: &'a str,
    targets: &'a [Option<wgpu::ColorTargetState>],
    cull_mode: Option<wgpu::Face>,
    depth_stencil: Option<wgpu::DepthStencilState>,
}

fn create_mesh_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
    spec: MeshPipelineSpec<'_>,
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
            targets: spec.targets,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
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

fn create_fullscreen_pipeline(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    fragment_entry: &str,
    targets: &[Option<wgpu::ColorTargetState>],
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some(fragment_entry),
            targets,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

/// Owned globe body (Pass 1) and procedural graticule (Pass 2) pipelines.
pub(super) struct BodyPipelines {
    pub(super) ellipsoid: wgpu::RenderPipeline,
    pub(super) textured: wgpu::RenderPipeline,
    pub(super) grid: wgpu::RenderPipeline,
}

/// Owned Pass-2 raster overlay render pipelines passed by reference into
/// [`super::overlay::OverlayRenderer::record_direct_layer`].
pub(super) struct OverlayPipelines {
    pub(super) display: wgpu::RenderPipeline,
    pub(super) scientific: wgpu::RenderPipeline,
    pub(super) categorical: wgpu::RenderPipeline,
}

/// Owned Pass-2 vector render pipelines passed by reference into
/// [`super::vector::VectorRenderer::record_pass`].
pub(super) struct VectorPipelines {
    pub(super) fill: wgpu::RenderPipeline,
    pub(super) line: wgpu::RenderPipeline,
    pub(super) point: wgpu::RenderPipeline,
}

/// Owned Band Math scratch-clear, single-band/MRT float scratch, `R8Uint` mask
/// scratch, and fullscreen composite pipelines.
pub(super) struct BandMathPipelines {
    pub(super) clear_layout: wgpu::BindGroupLayout,
    pub(super) clear: [wgpu::RenderPipeline; 4],
    pub(super) band_single: wgpu::RenderPipeline,
    pub(super) band_mrt: [wgpu::RenderPipeline; 3], // 2, 3, 4 attachments
    pub(super) mask_u8: wgpu::RenderPipeline,
    pub(super) composite_scratch_layout: wgpu::BindGroupLayout,
    pub(super) composite: wgpu::RenderPipeline,
}

impl BandMathPipelines {
    pub(super) fn clear_for_count(&self, count: usize) -> &wgpu::RenderPipeline {
        &self.clear[count.clamp(1, 4) - 1]
    }

    pub(super) fn mrt_for_count(&self, count: usize) -> &wgpu::RenderPipeline {
        &self.band_mrt[count.clamp(2, 4) - 2]
    }
}

/// Owned 1:1 offscreen-to-swapchain composite bind-group layout, sampler, and
/// render pipeline.
pub(super) struct CompositePipelines {
    pub(super) layout: wgpu::BindGroupLayout,
    pub(super) sampler: wgpu::Sampler,
    pub(super) pipeline: wgpu::RenderPipeline,
}

pub(super) struct GlobePipelines {
    pub(super) layouts: SharedLayouts,
    pub(super) body: BodyPipelines,
    pub(super) overlay: OverlayPipelines,
    pub(super) band_math: BandMathPipelines,
    pub(super) vector: VectorPipelines,
    pub(super) composite: CompositePipelines,
}

pub(super) struct SharedLayouts {
    pub(super) camera_layout: wgpu::BindGroupLayout,
    pub(super) dynamic_uniform_layouts: [wgpu::BindGroupLayout; UNIFORM_SLOT_COUNT],
    pub(super) palette_uniform_layout: wgpu::BindGroupLayout,
}

impl SharedLayouts {
    pub(super) fn dynamic_uniform_layout(&self, slot: UniformSlot) -> &wgpu::BindGroupLayout {
        &self.dynamic_uniform_layouts[slot.index()]
    }
}

fn uniform_buffer_layout(
    device: &wgpu::Device,
    label: &str,
    min_size_bytes: usize,
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
                min_binding_size: wgpu::BufferSize::new(min_size_bytes as u64),
            },
            count: None,
        }],
    })
}

fn texture_2d_layout_entry(
    binding: u32,
    sample_type: wgpu::TextureSampleType,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn non_filtering_texture_sampler_layout(
    device: &wgpu::Device,
    label: &str,
) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[
            texture_2d_layout_entry(0, wgpu::TextureSampleType::Float { filterable: false }),
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
    let camera_layout = uniform_buffer_layout(
        device,
        "globe camera bind group layout",
        std::mem::size_of::<CameraUniform>(),
        vf,
        false,
    );
    let dynamic_uniform_layouts = std::array::from_fn(|idx| {
        let slot = UniformSlot::ALL[idx];
        uniform_buffer_layout(device, slot.layout_label(), slot.binding_size(), vf, true)
    });
    let palette_uniform_layout = uniform_buffer_layout(
        device,
        "overlay palette LUT uniform layout",
        std::mem::size_of::<PaletteUniform>(),
        wgpu::ShaderStages::FRAGMENT,
        false,
    );

    SharedLayouts {
        camera_layout,
        dynamic_uniform_layouts,
        palette_uniform_layout,
    }
}

fn create_body_pipelines(
    device: &wgpu::Device,
    layouts: &SharedLayouts,
    rgba8_tile_layout: &wgpu::BindGroupLayout,
) -> BodyPipelines {
    let globe_shader = wgsl_module(device, "globe wgsl", GLOBE_WGSL);
    let tile_layout = layouts.dynamic_uniform_layout(UniformSlot::Tile);
    let untextured_pipeline_layout = pipeline_layout(
        device,
        "globe untextured pipeline layout",
        &[Some(&layouts.camera_layout), Some(tile_layout)],
    );
    let textured_pipeline_layout = pipeline_layout(
        device,
        "globe textured pipeline layout",
        &[
            Some(&layouts.camera_layout),
            Some(tile_layout),
            Some(rgba8_tile_layout),
        ],
    );

    let body_depth = || {
        Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        })
    };
    let replace = [srgb_target(wgpu::BlendState::REPLACE)];
    let alpha = [srgb_target(wgpu::BlendState::ALPHA_BLENDING)];

    let ellipsoid = create_mesh_pipeline(
        device,
        &globe_shader,
        &untextured_pipeline_layout,
        MeshPipelineSpec {
            label: "globe body ellipsoid pipeline (Reverse-Z)",
            vertex_entry: "vs_main",
            vertex_layout: TileVertex::LAYOUT,
            fragment_entry: "fs_body_ellipsoid",
            targets: &replace,
            cull_mode: Some(wgpu::Face::Back),
            depth_stencil: body_depth(),
        },
    );
    let textured = create_mesh_pipeline(
        device,
        &globe_shader,
        &textured_pipeline_layout,
        MeshPipelineSpec {
            label: "globe body textured pipeline (Reverse-Z)",
            vertex_entry: "vs_main",
            vertex_layout: TileVertex::LAYOUT,
            fragment_entry: "fs_body_textured",
            targets: &replace,
            cull_mode: Some(wgpu::Face::Back),
            depth_stencil: body_depth(),
        },
    );

    // Post-body graticule runs inside the depth-less (`depth_stencil_attachment: None`)
    // color pass directly above draped raster overlays and vector layers.
    let grid = create_mesh_pipeline(
        device,
        &globe_shader,
        &untextured_pipeline_layout,
        MeshPipelineSpec {
            label: "globe post-overlay procedural graticule pipeline (depth-less)",
            vertex_entry: "vs_main",
            vertex_layout: TileVertex::LAYOUT,
            fragment_entry: "fs_grid",
            targets: &alpha,
            cull_mode: Some(wgpu::Face::Back),
            depth_stencil: None,
        },
    );

    BodyPipelines {
        ellipsoid,
        textured,
        grid,
    }
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
    let shader = wgsl_module(device, shader_label, shader_source);
    let layout = pipeline_layout(device, layout_label, bind_group_layouts);
    let alpha = [srgb_target(wgpu::BlendState::ALPHA_BLENDING)];
    create_mesh_pipeline(
        device,
        &shader,
        &layout,
        MeshPipelineSpec {
            label: pipeline_label,
            vertex_entry: "vs_main",
            vertex_layout: OverlayVertex::LAYOUT,
            fragment_entry,
            targets: &alpha,
            cull_mode: Some(wgpu::Face::Back),
            depth_stencil: None,
        },
    )
}

fn create_overlay_pipelines(
    device: &wgpu::Device,
    layouts: &SharedLayouts,
    tile_bind_groups: &TileBindGroupCache,
) -> OverlayPipelines {
    let display = create_overlay_track_pipeline(
        device,
        "overlay warp-mesh wgsl",
        OVERLAY_WGSL,
        "overlay warp-mesh pipeline layout",
        &[
            Some(&layouts.camera_layout),
            Some(layouts.dynamic_uniform_layout(UniformSlot::Overlay)),
            Some(tile_bind_groups.rgba8_layout()),
        ],
        "overlay display warp-mesh pipeline (depth-less)",
        "fs_overlay",
    );
    let scientific = create_overlay_track_pipeline(
        device,
        "overlay scientific R32Float wgsl",
        OVERLAY_SCIENTIFIC_WGSL,
        "overlay scientific pipeline layout",
        &[
            Some(&layouts.camera_layout),
            Some(layouts.dynamic_uniform_layout(UniformSlot::Scientific)),
            Some(tile_bind_groups.r32f_layout()),
            Some(&layouts.palette_uniform_layout),
        ],
        "overlay scientific R32Float pipeline (depth-less)",
        "fs_scientific",
    );
    let categorical = create_overlay_track_pipeline(
        device,
        "overlay categorical Uint wgsl",
        OVERLAY_CATEGORICAL_WGSL,
        "overlay categorical pipeline layout",
        &[
            Some(&layouts.camera_layout),
            Some(layouts.dynamic_uniform_layout(UniformSlot::Categorical)),
            Some(tile_bind_groups.uint_layout()),
            Some(&layouts.palette_uniform_layout),
        ],
        "overlay categorical Uint pipeline (depth-less)",
        "fs_categorical",
    );

    OverlayPipelines {
        display,
        scientific,
        categorical,
    }
}

fn create_vector_pipelines(device: &wgpu::Device, layouts: &SharedLayouts) -> VectorPipelines {
    let vector_shader = wgsl_module(device, "vector draped wgsl", VECTOR_WGSL);
    let vector_pipeline_layout = pipeline_layout(
        device,
        "vector draped pipeline layout",
        &[
            Some(&layouts.camera_layout),
            Some(layouts.dynamic_uniform_layout(UniformSlot::Vector)),
        ],
    );
    let alpha = [srgb_target(wgpu::BlendState::ALPHA_BLENDING)];
    let fill = create_mesh_pipeline(
        device,
        &vector_shader,
        &vector_pipeline_layout,
        MeshPipelineSpec {
            label: "vector polygon fill pipeline (depth-less)",
            vertex_entry: "vs_main",
            vertex_layout: GpuVectorVertex::LAYOUT,
            fragment_entry: "fs_vector",
            targets: &alpha,
            cull_mode: Some(wgpu::Face::Back),
            depth_stencil: None,
        },
    );
    let line = create_mesh_pipeline(
        device,
        &vector_shader,
        &vector_pipeline_layout,
        MeshPipelineSpec {
            label: "vector HiDPI ribbon stroke pipeline (depth-less)",
            vertex_entry: "vs_ribbon",
            vertex_layout: GpuRibbonSegment::INSTANCE_LAYOUT,
            fragment_entry: "fs_vector",
            targets: &alpha,
            cull_mode: None,
            depth_stencil: None,
        },
    );
    let point = create_mesh_pipeline(
        device,
        &vector_shader,
        &vector_pipeline_layout,
        MeshPipelineSpec {
            label: "vector instanced point marker pipeline (depth-less)",
            vertex_entry: "vs_point",
            vertex_layout: GpuVectorVertex::INSTANCE_LAYOUT,
            fragment_entry: "fs_vector",
            targets: &alpha,
            cull_mode: Some(wgpu::Face::Back),
            depth_stencil: None,
        },
    );
    VectorPipelines { fill, line, point }
}

fn create_composite_pipelines(
    device: &wgpu::Device,
    target_format: wgpu::TextureFormat,
) -> CompositePipelines {
    let composite_shader = wgsl_module(device, "composite wgsl", COMPOSITE_WGSL);
    let layout = non_filtering_texture_sampler_layout(device, "composite bind group layout");
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("composite nearest sampler"),
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        ..Default::default()
    });
    let composite_pipeline_layout =
        pipeline_layout(device, "composite pipeline layout", &[Some(&layout)]);
    let pipeline = create_fullscreen_pipeline(
        device,
        "composite pipeline",
        &composite_pipeline_layout,
        &composite_shader,
        "fs_main",
        &[Some(wgpu::ColorTargetState {
            format: target_format,
            blend: Some(wgpu::BlendState::REPLACE),
            write_mask: wgpu::ColorWrites::ALL,
        })],
    );

    CompositePipelines {
        layout,
        sampler,
        pipeline,
    }
}

fn create_scratch_warp_pipeline(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::PipelineLayout,
    module: &wgpu::ShaderModule,
    fragment_entry: &str,
    targets: &[Option<wgpu::ColorTargetState>],
) -> wgpu::RenderPipeline {
    create_mesh_pipeline(
        device,
        module,
        layout,
        MeshPipelineSpec {
            label,
            vertex_entry: "vs_main",
            vertex_layout: OverlayVertex::LAYOUT,
            fragment_entry,
            targets,
            cull_mode: Some(wgpu::Face::Back),
            depth_stencil: None,
        },
    )
}

fn create_band_math_pipelines(
    device: &wgpu::Device,
    layouts: &SharedLayouts,
    tile_bind_groups: &TileBindGroupCache,
) -> BandMathPipelines {
    let clear_shader = wgsl_module(device, "scratch quiet-NaN fill wgsl", SCRATCH_NAN_FILL_WGSL);
    let clear_layout = uniform_buffer_layout(
        device,
        "scratch quiet-NaN clear uniform layout",
        std::mem::size_of::<ScratchClearUniform>(),
        wgpu::ShaderStages::FRAGMENT,
        false,
    );
    let clear_pipeline_layout = pipeline_layout(
        device,
        "scratch quiet-NaN clear pipeline layout",
        &[Some(&clear_layout)],
    );
    let r32f_target = Some(wgpu::ColorTargetState {
        format: wgpu::TextureFormat::R32Float,
        blend: None,
        write_mask: wgpu::ColorWrites::ALL,
    });
    let clear_specs = [
        ("scratch quiet-NaN clear pipeline (1x)", "fs_clear_1"),
        ("scratch quiet-NaN clear pipeline (2x)", "fs_clear_2"),
        ("scratch quiet-NaN clear pipeline (3x)", "fs_clear_3"),
        ("scratch quiet-NaN clear pipeline (4x)", "fs_clear_4"),
    ];
    let clear = std::array::from_fn(|idx| {
        let (label, entry) = clear_specs[idx];
        let targets: Vec<Option<wgpu::ColorTargetState>> =
            (0..=idx).map(|_| r32f_target.clone()).collect();
        create_fullscreen_pipeline(
            device,
            label,
            &clear_pipeline_layout,
            &clear_shader,
            entry,
            &targets,
        )
    });

    let scratch_single_shader = wgsl_module(
        device,
        "band-math single scratch wgsl",
        BAND_MATH_SCRATCH_WGSL,
    );
    let scratch_single_layout = pipeline_layout(
        device,
        "band-math single scratch pipeline layout",
        &[
            Some(&layouts.camera_layout),
            Some(layouts.dynamic_uniform_layout(UniformSlot::ScratchSingle)),
            Some(tile_bind_groups.r32f_layout()),
        ],
    );
    let band_single = create_scratch_warp_pipeline(
        device,
        "band-math single R32Float scratch pipeline",
        &scratch_single_layout,
        &scratch_single_shader,
        "fs_band_single",
        std::slice::from_ref(&r32f_target),
    );

    let scratch_mrt_shader = wgsl_module(
        device,
        "band-math MRT scratch wgsl",
        BAND_MATH_SCRATCH_MRT_WGSL,
    );
    let scratch_mrt_layout = pipeline_layout(
        device,
        "band-math MRT scratch pipeline layout",
        &[
            Some(&layouts.camera_layout),
            Some(layouts.dynamic_uniform_layout(UniformSlot::ScratchMrt)),
            Some(tile_bind_groups.r32f_mrt_layout()),
        ],
    );
    let mrt_specs = [
        (
            "band-math MRT R32Float scratch pipeline (2x)",
            "fs_band_mrt2",
        ),
        (
            "band-math MRT R32Float scratch pipeline (3x)",
            "fs_band_mrt3",
        ),
        (
            "band-math MRT R32Float scratch pipeline (4x)",
            "fs_band_mrt4",
        ),
    ];
    let band_mrt = std::array::from_fn(|idx| {
        let (label, entry) = mrt_specs[idx];
        let targets: Vec<Option<wgpu::ColorTargetState>> =
            (0..(idx + 2)).map(|_| r32f_target.clone()).collect();
        create_scratch_warp_pipeline(
            device,
            label,
            &scratch_mrt_layout,
            &scratch_mrt_shader,
            entry,
            &targets,
        )
    });

    let mask_shader = wgsl_module(
        device,
        "band-math R8Uint SCL mask scratch wgsl",
        BAND_MATH_MASK_WGSL,
    );
    let mask_layout = pipeline_layout(
        device,
        "band-math R8Uint mask pipeline layout",
        &[
            Some(&layouts.camera_layout),
            Some(layouts.dynamic_uniform_layout(UniformSlot::ScratchSingle)),
            Some(tile_bind_groups.uint_layout()),
        ],
    );
    let mask_u8 = create_scratch_warp_pipeline(
        device,
        "band-math R8Uint SCL mask scratch pipeline",
        &mask_layout,
        &mask_shader,
        "fs_mask_u8",
        &[Some(wgpu::ColorTargetState {
            format: wgpu::TextureFormat::R8Uint,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        })],
    );

    let composite_entries: [wgpu::BindGroupLayoutEntry; 5] = std::array::from_fn(|binding| {
        let sample_type = if binding < 4 {
            wgpu::TextureSampleType::Float { filterable: false }
        } else {
            wgpu::TextureSampleType::Uint
        };
        texture_2d_layout_entry(binding as u32, sample_type)
    });
    let composite_scratch_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("band-math composite scratch textures layout"),
            entries: &composite_entries,
        });

    let composite_shader = wgsl_module(
        device,
        "band-math fullscreen composite wgsl",
        BAND_MATH_COMPOSITE_WGSL,
    );
    let composite_pipeline_layout = pipeline_layout(
        device,
        "band-math fullscreen composite pipeline layout",
        &[
            Some(&composite_scratch_layout),
            Some(layouts.dynamic_uniform_layout(UniformSlot::BandMathComposite)),
            Some(&layouts.palette_uniform_layout),
        ],
    );
    let composite = create_fullscreen_pipeline(
        device,
        "band-math fullscreen composite pipeline (depth-less)",
        &composite_pipeline_layout,
        &composite_shader,
        "fs_composite",
        &[srgb_target(wgpu::BlendState::ALPHA_BLENDING)],
    );

    BandMathPipelines {
        clear_layout,
        clear,
        band_single,
        band_mrt,
        mask_u8,
        composite_scratch_layout,
        composite,
    }
}

impl GlobePipelines {
    pub(super) fn new(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        tile_bind_groups: &TileBindGroupCache,
    ) -> Self {
        let layouts = create_shared_layouts(device);
        let body = create_body_pipelines(device, &layouts, tile_bind_groups.rgba8_layout());
        let overlay = create_overlay_pipelines(device, &layouts, tile_bind_groups);
        let band_math = create_band_math_pipelines(device, &layouts, tile_bind_groups);
        let vector = create_vector_pipelines(device, &layouts);
        let composite = create_composite_pipelines(device, target_format);

        Self {
            layouts,
            body,
            overlay,
            band_math,
            vector,
            composite,
        }
    }
}
