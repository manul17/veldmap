//! Offscreen `wgpu` WGS84 globe renderer decoupled from `egui`.
//!
//! Renders the WGS84 ellipsoid and depth-less draped graticule into an
//! offscreen `Rgba8UnormSrgb` + `Depth32Float` Reverse-Z target, then
//! composites 1:1 with `FilterMode::Nearest` onto `egui_wgpu`'s target format
//! via `view_formats` (`Rgba8UnormSrgb` when the target expects linear output,
//! `Rgba8Unorm` for non-sRGB 8-bit swapchains).

use std::ops::Range;

use bytemuck::{Pod, Zeroable};
use eframe::egui;
use eframe::egui_wgpu::{CallbackResources, CallbackTrait, ScreenDescriptor};
use glam::DVec3;
use veldmap_core::geo::camera::Camera;
use veldmap_core::geo::geodesy;
use wgpu::util::DeviceExt;

use crate::mesh::{GlobeMesh, GlobeVertex};

/// Embedded WGSL source for the globe body and draped graticule.
pub const GLOBE_WGSL: &str = include_str!("shaders/globe.wgsl");
/// Embedded WGSL source for 1:1 nearest-neighbor offscreen compositing.
pub const COMPOSITE_WGSL: &str = include_str!("shaders/composite.wgsl");

/// Deep-space clear color in exact linear RGB (`#0B1320` in sRGB).
pub const CLEAR_COLOR_LINEAR: wgpu::Color = wgpu::Color {
    r: 0.003_347,
    g: 0.006_512,
    b: 0.014_444,
    a: 1.0,
};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ChunkUniform {
    origin_minus_eye_hi: [f32; 3],
    _pad0: f32,
    origin_minus_eye_lo: [f32; 3],
    _pad1: f32,
}

/// Parameters for a single offscreen globe frame.
#[derive(Clone, Copy, Debug)]
pub struct RenderParams<'a> {
    pub width: u32,
    pub height: u32,
    pub camera: &'a Camera,
    pub show_graticule: bool,
    pub generation: u64,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct RenderKey {
    width: u32,
    height: u32,
    show_graticule: bool,
    generation: u64,
}

struct OffscreenTarget {
    width: u32,
    height: u32,
    _color_texture: wgpu::Texture,
    color_srgb_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
    composite_bind_group: wgpu::BindGroup,
}

/// Standalone offscreen `wgpu` WGS84 globe renderer.
pub struct GlobeRenderer {
    target_format: wgpu::TextureFormat,
    body_indices: Range<u32>,
    graticule_indices: Range<u32>,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    chunk_buffer: wgpu::Buffer,
    chunk_bind_group: wgpu::BindGroup,
    body_pipeline: wgpu::RenderPipeline,
    grid_pipeline: wgpu::RenderPipeline,
    composite_layout: wgpu::BindGroupLayout,
    composite_sampler: wgpu::Sampler,
    composite_pipeline: wgpu::RenderPipeline,
    offscreen: Option<OffscreenTarget>,
    last_render: Option<RenderKey>,
}

impl GlobeRenderer {
    /// Create all static GPU buffers and pipelines for the globe and 1:1
    /// composite pass targeting `target_format`.
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let mesh = GlobeMesh::build();
        let body_indices = mesh.body;
        let graticule_indices = mesh.graticule;

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("globe vertex buffer"),
            contents: bytemuck::cast_slice(&mesh.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("globe index buffer"),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("globe camera uniform"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("globe camera bind group layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
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

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("globe camera bind group"),
            layout: &camera_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        let chunk_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("globe chunk origin uniform"),
            size: std::mem::size_of::<ChunkUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let chunk_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("globe chunk bind group layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<ChunkUniform>() as u64
                    ),
                },
                count: None,
            }],
        });

        let chunk_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("globe chunk bind group"),
            layout: &chunk_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: chunk_buffer.as_entire_binding(),
            }],
        });

        let globe_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("globe wgsl"),
            source: wgpu::ShaderSource::Wgsl(GLOBE_WGSL.into()),
        });

        let globe_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("globe pipeline layout"),
                bind_group_layouts: &[Some(&camera_layout), Some(&chunk_layout)],
                immediate_size: 0,
            });

        let body_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("globe body pipeline (Reverse-Z)"),
            layout: Some(&globe_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &globe_shader,
                entry_point: Some("vs_main"),
                buffers: &[Some(GlobeVertex::LAYOUT)],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &globe_shader,
                entry_point: Some("fs_body"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    blend: Some(wgpu::BlendState::REPLACE),
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
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let grid_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("globe draped graticule pipeline"),
            layout: Some(&globe_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &globe_shader,
                entry_point: Some("vs_main"),
                buffers: &[Some(GlobeVertex::LAYOUT)],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &globe_shader,
                entry_point: Some("fs_grid"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

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

        Self {
            target_format,
            body_indices,
            graticule_indices,
            vertex_buffer,
            index_buffer,
            camera_buffer,
            camera_bind_group,
            chunk_buffer,
            chunk_bind_group,
            body_pipeline,
            grid_pipeline,
            composite_layout,
            composite_sampler,
            composite_pipeline,
            offscreen: None,
            last_render: None,
        }
    }

    /// Ensure the offscreen target matches `(params.width, params.height)`
    /// clamped to device limits and re-render the globe if the target or scene
    /// parameters changed. Returns `true` if a new offscreen frame was
    /// rendered.
    pub fn render_if_dirty(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        params: RenderParams<'_>,
    ) -> bool {
        if params.width == 0 || params.height == 0 {
            return false;
        }
        let max_dim = device.limits().max_texture_dimension_2d;
        let width = params.width.clamp(1, max_dim);
        let height = params.height.clamp(1, max_dim);

        let key = RenderKey {
            width,
            height,
            show_graticule: params.show_graticule,
            generation: params.generation,
        };
        if self.last_render == Some(key)
            && self
                .offscreen
                .as_ref()
                .is_some_and(|t| t.width == width && t.height == height)
        {
            return false;
        }

        self.ensure_offscreen(device, width, height);
        self.upload_uniforms(queue, width, height, params.camera);
        self.record_offscreen_pass(encoder, params.show_graticule);
        self.last_render = Some(key);
        true
    }

    /// Record the 1:1 `FilterMode::Nearest` composite draw into an existing
    /// render pass targeting `self.target_format`.
    pub fn composite_into(&self, render_pass: &mut wgpu::RenderPass<'_>) {
        let Some(target) = &self.offscreen else {
            return;
        };
        render_pass.set_pipeline(&self.composite_pipeline);
        render_pass.set_bind_group(0, &target.composite_bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }

    fn ensure_offscreen(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if self
            .offscreen
            .as_ref()
            .is_some_and(|t| t.width == width && t.height == height)
        {
            return;
        }

        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let color_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("globe offscreen color (Rgba8UnormSrgb)"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[wgpu::TextureFormat::Rgba8Unorm],
        });

        let color_srgb_view = color_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("globe offscreen srgb view"),
            format: Some(wgpu::TextureFormat::Rgba8UnormSrgb),
            ..Default::default()
        });

        // Match swapchain color space via view_formats so compositing is exact 1:1:
        // sRGB and linear float targets expect linear shader output (sample via
        // `Rgba8UnormSrgb`); non-sRGB 8-bit targets expect raw sRGB-encoded bytes
        // (sample via `Rgba8Unorm`).
        let target_expects_linear = self.target_format.is_srgb()
            || matches!(
                self.target_format,
                wgpu::TextureFormat::Rgba16Float | wgpu::TextureFormat::Rgba32Float
            );
        let unorm_view = (!target_expects_linear).then(|| {
            color_texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("globe composite unorm view"),
                format: Some(wgpu::TextureFormat::Rgba8Unorm),
                ..Default::default()
            })
        });
        let composite_view = unorm_view.as_ref().unwrap_or(&color_srgb_view);

        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("globe offscreen depth (Depth32Float Reverse-Z)"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let composite_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("globe composite bind group"),
            layout: &self.composite_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(composite_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.composite_sampler),
                },
            ],
        });

        self.offscreen = Some(OffscreenTarget {
            width,
            height,
            _color_texture: color_texture,
            color_srgb_view,
            depth_view,
            composite_bind_group,
        });
    }

    fn upload_uniforms(&self, queue: &wgpu::Queue, width: u32, height: u32, camera: &Camera) {
        let aspect = width as f32 / height.max(1) as f32;
        let cam_uniform = CameraUniform {
            view_proj: camera.view_projection_f32(aspect).to_cols_array_2d(),
        };
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&cam_uniform));

        let (hi, lo) = geodesy::parts(DVec3::ZERO - camera.eye_ecef());
        let chunk = ChunkUniform {
            origin_minus_eye_hi: hi,
            _pad0: 0.0,
            origin_minus_eye_lo: lo,
            _pad1: 0.0,
        };
        queue.write_buffer(&self.chunk_buffer, 0, bytemuck::bytes_of(&chunk));
    }

    fn record_offscreen_pass(&self, encoder: &mut wgpu::CommandEncoder, show_graticule: bool) {
        let Some(target) = &self.offscreen else {
            return;
        };

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("globe offscreen pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target.color_srgb_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(CLEAR_COLOR_LINEAR),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &target.depth_view,
                depth_ops: Some(wgpu::Operations {
                    // Reverse-Z: infinity clears to 0.0, near plane maps to 1.0.
                    load: wgpu::LoadOp::Clear(0.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        pass.set_bind_group(0, &self.camera_bind_group, &[]);
        pass.set_bind_group(1, &self.chunk_bind_group, &[]);

        pass.set_pipeline(&self.body_pipeline);
        pass.draw_indexed(self.body_indices.clone(), 0, 0..1);

        if show_graticule {
            pass.set_pipeline(&self.grid_pipeline);
            pass.draw_indexed(self.graticule_indices.clone(), 0, 0..1);
        }
    }
}

/// `egui_wgpu` paint callback bridging the offscreen [`GlobeRenderer`] into the
/// viewport panel.
pub struct GlobePaintCallback {
    pub rect: egui::Rect,
    pub clip_rect: egui::Rect,
    pub camera: Camera,
    pub show_graticule: bool,
    pub generation: u64,
}

impl CallbackTrait for GlobePaintCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        screen_descriptor: &ScreenDescriptor,
        egui_encoder: &mut wgpu::CommandEncoder,
        callback_resources: &mut CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let info = egui::PaintCallbackInfo {
            viewport: self.rect,
            clip_rect: self.clip_rect,
            pixels_per_point: screen_descriptor.pixels_per_point,
            screen_size_px: screen_descriptor.size_in_pixels,
        };
        let vp = info.viewport_in_pixels();
        if vp.width_px <= 0 || vp.height_px <= 0 {
            return Vec::new();
        }

        if let Some(renderer) = callback_resources.get_mut::<GlobeRenderer>() {
            renderer.render_if_dirty(
                device,
                queue,
                egui_encoder,
                RenderParams {
                    width: vp.width_px as u32,
                    height: vp.height_px as u32,
                    camera: &self.camera,
                    show_graticule: self.show_graticule,
                    generation: self.generation,
                },
            );
        }
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        callback_resources: &CallbackResources,
    ) {
        if let Some(renderer) = callback_resources.get::<GlobeRenderer>() {
            renderer.composite_into(render_pass);
        }
    }
}
