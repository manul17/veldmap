//! Offscreen `wgpu` WGS84 quadtree globe renderer decoupled from `egui`.
//!
//! Renders per-tile CPU `f64` RTC quadtree meshes (textured sRGB basemap or
//! untextured WGS84 ellipsoid shading) and the post-overlay procedural
//! `fwidth()` graticule into an offscreen `Rgba8UnormSrgb` + `Depth32Float`
//! Reverse-Z target, then composites 1:1 with `FilterMode::Nearest` onto
//! `egui_wgpu`'s target format via `view_formats`.

use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};

use bytemuck::{Pod, Zeroable};
use veldmap_core::geo::camera::Camera;
use veldmap_core::tile::{
    DEFAULT_GRID_CELLS, POLAR_R_CAP, QuadTileId, QuadtreeConfig, SourceId, TileCache, TileKey,
    TileScheme, VisibleTile, adaptive_grid_step_deg, compose_uv_rect, sample_tile_surface,
    tile_grid_anchor, traverse,
};
use veldmap_raster::{BUNDLED_MAX_LEVEL, BUNDLED_POLAR_MAX_LEVEL};
use wgpu::util::DeviceExt;

use crate::gpu_tiles::{GpuTile, GpuTileCache, bundled_tile_key, source_tile_key};
use crate::mesh::{SharedTileIndices, TileVertex, build_tile_vertices};

/// Embedded WGSL source for the quadtree globe body and procedural graticule.
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

/// Default VRAM budget for cached per-tile geometry vertex buffers (`16 MiB`,
/// ~1,000 cached `16 x 16` tiles).
const DEFAULT_MESH_CACHE_BYTES: usize = 16 * 1024 * 1024;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
    grid_step_deg: f32,
    grid_width_px: f32,
    polar_r_cap: f32,
    _pad0: f32,
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
struct TileUniform {
    offset_eye: [f32; 3],
    tile_kind: f32,
    uv_rect: [f32; 4],
    grid_anchor: [f32; 4],
}

const _: () = {
    assert!(std::mem::size_of::<TileUniform>() == 48);
    assert!(std::mem::offset_of!(TileUniform, offset_eye) == 0);
    assert!(std::mem::offset_of!(TileUniform, tile_kind) == 12);
    assert!(std::mem::offset_of!(TileUniform, uv_rect) == 16);
    assert!(std::mem::offset_of!(TileUniform, grid_anchor) == 32);
};

/// Single tile draw entry inside an immutable [`DrawSnapshot`].
#[derive(Clone, Debug)]
pub struct DrawTile {
    pub visible: VisibleTile,
    /// Resolved GPU carrier texture (`None` in Ellipsoid-only mode or before
    /// any coarse ancestor tile has uploaded).
    pub texture: Option<Arc<GpuTile>>,
    /// Final UV sub-rectangle in `texture`'s coordinate space.
    pub uv_rect: [f32; 4],
}

/// Immutable per-frame draw snapshot built on `VeldmapApp` (cached by
/// `(camera_key, scene_generation)` so 120 Hz mouse hover never rebuilds tile
/// lists) and handed to `GlobePaintCallback`.
#[derive(Clone, Debug)]
pub struct DrawSnapshot {
    pub camera: Camera,
    pub tiles: Vec<DrawTile>,
    /// Exact GPU carrier tile keys referenced by `tiles`, used for active-frame
    /// `TileCache` protection so uploads never evict currently visible tiles.
    pub carrier_keys: HashSet<TileKey>,
    pub grid_cells: u32,
    pub show_graticule: bool,
    pub grid_step_deg: f64,
    pub grid_width_px: f32,
    pub generation: u64,
}

impl DrawSnapshot {
    /// Build a draw snapshot from a camera pose, viewport dimensions, active
    /// basemap [`SourceId`], and optional [`GpuTileCache`] (passing
    /// `textured = false` renders untextured WGS84 ellipsoid shading through
    /// the exact same quadtree).
    ///
    /// Enforces the explicit `PolarCap` + live-basemap contract: `PolarCap`
    /// tiles always resolve against `SourceId::BUNDLED_BASEMAP` (never a live
    /// HTTP source), and non-polar live tiles seamlessly fall back to cached
    /// `SourceId::BUNDLED_BASEMAP` tiles while streaming or offline.
    #[expect(
        clippy::too_many_arguments,
        reason = "snapshot builder groups explicit viewport, source, graticule, and cache inputs"
    )]
    pub fn build(
        camera: Camera,
        width_px: u32,
        height_px: u32,
        pixels_per_point: f32,
        show_graticule: bool,
        textured: bool,
        active_source: SourceId,
        generation: u64,
        config: &QuadtreeConfig,
        cache: Option<&mut GpuTileCache>,
    ) -> Self {
        let visible = traverse(&camera, width_px, height_px, config);
        let ppp = f64::from(pixels_per_point.max(0.5));
        let height_pt = (f64::from(height_px.max(1)) / ppp).max(1.0);
        let m_per_pt = camera.metres_per_pixel(height_pt);
        let grid_step_deg = adaptive_grid_step_deg(m_per_pt);
        let grid_width_px = (0.75 * pixels_per_point.max(1.0)).clamp(0.65, 3.0);

        let mut tiles = Vec::with_capacity(visible.len());
        let mut carrier_keys = HashSet::new();
        match (textured, cache) {
            (true, Some(gpu_cache)) => {
                for vis in visible {
                    let resolved = if matches!(vis.tex_id.scheme, TileScheme::PolarCap { .. }) {
                        let (polar_id, polar_uv) =
                            vis.tex_id.clamp_to_level(BUNDLED_POLAR_MAX_LEVEL);
                        let key = bundled_tile_key(polar_id);
                        gpu_cache.carrier(&key).map(|(ck, c_uv, tile)| {
                            let uv = compose_uv_rect(c_uv, compose_uv_rect(polar_uv, vis.uv_rect));
                            (ck, uv, tile)
                        })
                    } else {
                        let primary_key = source_tile_key(active_source, vis.tex_id);
                        if let Some((ck, c_uv, tile)) = gpu_cache.carrier(&primary_key) {
                            Some((ck, compose_uv_rect(c_uv, vis.uv_rect), tile))
                        } else if active_source != SourceId::BUNDLED_BASEMAP {
                            // Offline / Frame-1 fallback to bundled basemap in matching scheme.
                            let (fb_id, fb_uv) = vis.tex_id.clamp_to_level(BUNDLED_MAX_LEVEL);
                            let fb_key = bundled_tile_key(fb_id);
                            gpu_cache.carrier(&fb_key).map(|(ck, c_uv, tile)| {
                                let uv = compose_uv_rect(c_uv, compose_uv_rect(fb_uv, vis.uv_rect));
                                (ck, uv, tile)
                            })
                        } else {
                            None
                        }
                    };

                    if let Some((carrier_key, uv_rect, gpu_tile)) = resolved {
                        carrier_keys.insert(carrier_key);
                        tiles.push(DrawTile {
                            visible: vis,
                            texture: Some(gpu_tile),
                            uv_rect,
                        });
                    } else {
                        tiles.push(DrawTile {
                            visible: vis,
                            texture: None,
                            uv_rect: vis.uv_rect,
                        });
                    }
                }
            }
            _ => {
                for vis in visible {
                    tiles.push(DrawTile {
                        visible: vis,
                        texture: None,
                        uv_rect: vis.uv_rect,
                    });
                }
            }
        }

        Self {
            camera,
            tiles,
            carrier_keys,
            grid_cells: config.grid_cells.max(1),
            show_graticule,
            grid_step_deg,
            grid_width_px,
            generation,
        }
    }
}

/// Parameters for a single offscreen globe frame.
#[derive(Clone, Copy, Debug)]
pub struct RenderParams<'a> {
    pub width: u32,
    pub height: u32,
    pub snapshot: &'a DrawSnapshot,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct RenderKey {
    width: u32,
    height: u32,
    grid_cells: u32,
    eye_bits: [u64; 3],
    up_bits: [u64; 3],
    show_graticule: bool,
    grid_step_bits: u64,
    grid_width_bits: u32,
    generation: u64,
    tiles_hash: u64,
}

impl RenderKey {
    fn from_params(width: u32, height: u32, params: &RenderParams<'_>) -> Self {
        let snap = params.snapshot;
        let eye = snap.camera.eye_ecef();
        let (_, up, _) = snap.camera.basis();
        let mut h = 0xcbf2_9ce4_8422_2325_u64;
        let mix = |h: &mut u64, v: u64| {
            *h ^= v;
            *h = h.wrapping_mul(0x0000_0100_0000_01b3);
        };
        mix(&mut h, snap.tiles.len() as u64);
        for t in &snap.tiles {
            let scheme_tag: u64 = match t.visible.id.scheme {
                TileScheme::Geodetic => 1,
                TileScheme::WebMercator => 2,
                TileScheme::PolarCap { north: true } => 3,
                TileScheme::PolarCap { north: false } => 4,
                TileScheme::RasterPyramid { max_level } => 100 + u64::from(max_level),
            };
            mix(&mut h, scheme_tag);
            mix(&mut h, u64::from(t.visible.id.level));
            mix(&mut h, u64::from(t.visible.id.x));
            mix(&mut h, u64::from(t.visible.id.y));
            mix(&mut h, t.texture.as_ref().map_or(0, |g| g.tile_id));
            for c in t.uv_rect {
                mix(&mut h, u64::from(c.to_bits()));
            }
        }
        Self {
            width,
            height,
            grid_cells: snap.grid_cells,
            eye_bits: eye.to_array().map(f64::to_bits),
            up_bits: up.to_array().map(f64::to_bits),
            show_graticule: snap.show_graticule,
            grid_step_bits: snap.grid_step_deg.to_bits(),
            grid_width_bits: snap.grid_width_px.to_bits(),
            generation: snap.generation,
            tiles_hash: h,
        }
    }
}

struct OffscreenTarget {
    width: u32,
    height: u32,
    color_srgb_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
    composite_bind_group: wgpu::BindGroup,
}

/// Per-viewport GPU buffers and offscreen targets grouped cleanly so resizing
/// never recreates `camera_buffer` or `tile_uniform_buffer`.
struct ViewportState {
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    tile_uniform_buffer: wgpu::Buffer,
    tile_uniform_bind_group: wgpu::BindGroup,
    tile_uniform_capacity: usize,
    offscreen: Option<OffscreenTarget>,
    last_render: Option<RenderKey>,
    /// Tracks whether the most recent offscreen pass reached `composite_into`.
    pending_present: AtomicBool,
}

impl ViewportState {
    fn new(
        device: &wgpu::Device,
        camera_layout: &wgpu::BindGroupLayout,
        tile_uniform_layout: &wgpu::BindGroupLayout,
        tile_uniform_stride: u32,
    ) -> Self {
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("globe camera uniform"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("globe camera bind group"),
            layout: camera_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        let initial_cap = 128usize;
        let (tile_uniform_buffer, tile_uniform_bind_group) = Self::alloc_tile_uniforms(
            device,
            tile_uniform_layout,
            tile_uniform_stride,
            initial_cap,
        );

        Self {
            camera_buffer,
            camera_bind_group,
            tile_uniform_buffer,
            tile_uniform_bind_group,
            tile_uniform_capacity: initial_cap,
            offscreen: None,
            last_render: None,
            pending_present: AtomicBool::new(false),
        }
    }

    fn alloc_tile_uniforms(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        stride: u32,
        capacity: usize,
    ) -> (wgpu::Buffer, wgpu::BindGroup) {
        let size = (capacity.max(1) as u64) * u64::from(stride);
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("globe per-tile dynamic uniform"),
            size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("globe per-tile dynamic bind group"),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &buffer,
                    offset: 0,
                    size: wgpu::BufferSize::new(std::mem::size_of::<TileUniform>() as u64),
                }),
            }],
        });
        (buffer, bind_group)
    }

    fn ensure_tile_uniform_capacity(
        &mut self,
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        stride: u32,
        needed: usize,
    ) {
        if needed <= self.tile_uniform_capacity {
            return;
        }
        let next_cap = needed.next_power_of_two().max(128);
        let (buf, bg) = Self::alloc_tile_uniforms(device, layout, stride, next_cap);
        self.tile_uniform_buffer = buf;
        self.tile_uniform_bind_group = bg;
        self.tile_uniform_capacity = next_cap;
    }
}

/// Standalone offscreen `wgpu` WGS84 quadtree globe renderer.
pub struct GlobeRenderer {
    target_format: wgpu::TextureFormat,
    grid_cells: u32,
    surface_indices: Range<u32>,
    full_indices: Range<u32>,
    index_buffer: wgpu::Buffer,
    tile_uniform_layout: wgpu::BindGroupLayout,
    tile_uniform_stride: u32,
    tile_texture_layout: wgpu::BindGroupLayout,
    tile_sampler: wgpu::Sampler,
    body_ellipsoid_pipeline: wgpu::RenderPipeline,
    body_textured_pipeline: wgpu::RenderPipeline,
    grid_pipeline: wgpu::RenderPipeline,
    composite_layout: wgpu::BindGroupLayout,
    composite_sampler: wgpu::Sampler,
    composite_pipeline: wgpu::RenderPipeline,
    /// Bounded O(1) LRU cache of per-tile CPU `f64` RTC vertex buffers.
    tile_meshes: TileCache<QuadTileId, wgpu::Buffer>,
    /// Lightweight `BindGroup` cache keyed by `GpuTile::tile_id` and guarded
    /// by `Weak<GpuTile>`, so evicted tiles are swept automatically without
    /// losing eviction events across discarded `egui` passes.
    tile_bind_groups: HashMap<u64, (Weak<GpuTile>, wgpu::BindGroup)>,
    /// Reusable staging scratch buffer for dynamic per-tile uniforms.
    uniform_staging: Vec<u8>,
    viewport: ViewportState,
}

impl GlobeRenderer {
    /// Create all shared GPU pipelines, static quadtree index buffer, and
    /// primary `ViewportState` targeting `target_format`.
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let grid_cells = DEFAULT_GRID_CELLS;
        let shared = SharedTileIndices::build(grid_cells);
        let surface_indices = shared.surface;
        let full_indices = shared.full;

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("quadtree shared index buffer"),
            contents: bytemuck::cast_slice(&shared.indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let align = device.limits().min_uniform_buffer_offset_alignment.max(1);
        let raw_tile_size = std::mem::size_of::<TileUniform>() as u32;
        let tile_uniform_stride = raw_tile_size.div_ceil(align) * align;

        let camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

        let tile_uniform_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

        let tile_texture_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

        let body_ellipsoid_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("globe body ellipsoid pipeline (Reverse-Z)"),
                layout: Some(&untextured_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &globe_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[Some(TileVertex::LAYOUT)],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &globe_shader,
                    entry_point: Some("fs_body_ellipsoid"),
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

        let body_textured_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("globe body textured pipeline (Reverse-Z)"),
                layout: Some(&textured_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &globe_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[Some(TileVertex::LAYOUT)],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &globe_shader,
                    entry_point: Some("fs_body_textured"),
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
            label: Some("globe post-overlay procedural graticule pipeline"),
            layout: Some(&untextured_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &globe_shader,
                entry_point: Some("vs_main"),
                buffers: &[Some(TileVertex::LAYOUT)],
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
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                front_face: wgpu::FrontFace::Ccw,
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

        let viewport = ViewportState::new(
            device,
            &camera_layout,
            &tile_uniform_layout,
            tile_uniform_stride,
        );

        Self {
            target_format,
            grid_cells,
            surface_indices,
            full_indices,
            index_buffer,
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
            tile_meshes: TileCache::new(DEFAULT_MESH_CACHE_BYTES),
            tile_bind_groups: HashMap::new(),
            uniform_staging: Vec::new(),
            viewport,
        }
    }

    /// Drop cached `BindGroup` entries whose backing `GpuTile` has been
    /// evicted from `GpuTileCache` and released by all active snapshots.
    pub fn sweep_stale_bind_groups(&mut self) {
        self.tile_bind_groups
            .retain(|_, (weak, _)| weak.strong_count() > 0);
    }

    /// Number of currently cached tile texture bind groups.
    pub fn cached_tile_bind_group_count(&self) -> usize {
        self.tile_bind_groups.len()
    }

    /// Ensure the offscreen target matches `(params.width, params.height)`
    /// clamped to device limits and re-render the globe if the target or draw
    /// snapshot changed. Returns `true` if a new offscreen frame was rendered.
    pub fn render_if_dirty(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        params: RenderParams<'_>,
    ) -> bool {
        self.sweep_stale_bind_groups();
        if params.width == 0 || params.height == 0 {
            return false;
        }
        let max_dim = device.limits().max_texture_dimension_2d;
        let width = params.width.clamp(1, max_dim);
        let height = params.height.clamp(1, max_dim);

        let key = RenderKey::from_params(width, height, &params);
        if !self.viewport.pending_present.load(Ordering::Relaxed)
            && self.viewport.last_render == Some(key)
            && self
                .viewport
                .offscreen
                .as_ref()
                .is_some_and(|t| t.width == width && t.height == height)
        {
            return false;
        }

        self.ensure_offscreen(device, width, height);
        self.prepare_tile_resources(device, params.snapshot);
        self.upload_uniforms(device, queue, width, height, params.snapshot);
        self.record_offscreen_pass(encoder, params.snapshot);

        self.viewport.last_render = Some(key);
        self.viewport.pending_present.store(true, Ordering::Relaxed);
        true
    }

    /// Record the 1:1 `FilterMode::Nearest` composite draw into an existing
    /// render pass targeting `self.target_format`.
    pub fn composite_into(&self, render_pass: &mut wgpu::RenderPass<'_>) {
        let Some(target) = &self.viewport.offscreen else {
            return;
        };
        self.viewport
            .pending_present
            .store(false, Ordering::Relaxed);
        render_pass.set_pipeline(&self.composite_pipeline);
        render_pass.set_bind_group(0, &target.composite_bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }

    fn ensure_offscreen(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if self
            .viewport
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

        self.viewport.offscreen = Some(OffscreenTarget {
            width,
            height,
            color_srgb_view,
            depth_view,
            composite_bind_group,
        });
    }

    fn ensure_grid_cells(&mut self, device: &wgpu::Device, grid_cells: u32) {
        let cells = grid_cells.max(1);
        if cells == self.grid_cells {
            return;
        }
        let shared = SharedTileIndices::build(cells);
        self.grid_cells = cells;
        self.surface_indices = shared.surface;
        self.full_indices = shared.full;
        self.index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("quadtree shared index buffer"),
            contents: bytemuck::cast_slice(&shared.indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        let _ = self.tile_meshes.clear();
    }

    fn prepare_tile_resources(&mut self, device: &wgpu::Device, snapshot: &DrawSnapshot) {
        self.ensure_grid_cells(device, snapshot.grid_cells);
        let stride = (self.grid_cells + 1) as usize;
        let single_tile_bytes = (stride * stride + 4 * stride) * std::mem::size_of::<TileVertex>();
        self.tile_meshes
            .ensure_max_bytes(snapshot.tiles.len().saturating_mul(single_tile_bytes));

        for draw in &snapshot.tiles {
            let id = draw.visible.id;
            if !self.tile_meshes.touch(&id) {
                let verts = build_tile_vertices(&draw.visible, self.grid_cells);
                let byte_size = std::mem::size_of_val(verts.as_slice());
                let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("quadtree tile RTC vertex buffer"),
                    contents: bytemuck::cast_slice(&verts),
                    usage: wgpu::BufferUsages::VERTEX,
                });
                let _ = self.tile_meshes.insert(id, vbuf, byte_size);
            }

            if let Some(gpu_tile) = &draw.texture {
                self.tile_bind_groups
                    .entry(gpu_tile.tile_id)
                    .or_insert_with(|| {
                        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("globe tile texture bind group"),
                            layout: &self.tile_texture_layout,
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
                        });
                        (Arc::downgrade(gpu_tile), bg)
                    });
            }
        }
    }

    fn upload_uniforms(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        snapshot: &DrawSnapshot,
    ) {
        let aspect = f64::from(width) / f64::from(height.max(1));
        let cam_uniform = CameraUniform {
            view_proj: snapshot
                .camera
                .view_projection(aspect)
                .as_mat4()
                .to_cols_array_2d(),
            grid_step_deg: snapshot.grid_step_deg as f32,
            grid_width_px: snapshot.grid_width_px,
            polar_r_cap: POLAR_R_CAP as f32,
            _pad0: 0.0,
        };
        queue.write_buffer(
            &self.viewport.camera_buffer,
            0,
            bytemuck::bytes_of(&cam_uniform),
        );

        let tile_count = snapshot.tiles.len();
        if tile_count == 0 {
            return;
        }
        self.viewport.ensure_tile_uniform_capacity(
            device,
            &self.tile_uniform_layout,
            self.tile_uniform_stride,
            tile_count,
        );

        let stride = self.tile_uniform_stride as usize;
        let total_bytes = tile_count * stride;
        self.uniform_staging.clear();
        self.uniform_staging.resize(total_bytes, 0);

        let eye = snapshot.camera.eye_ecef();
        let step_deg = snapshot.grid_step_deg.max(1e-6);

        for (i, draw) in snapshot.tiles.iter().enumerate() {
            let offset_eye = (draw.visible.origin_ecef_m - eye).as_vec3().to_array();
            let (tile_kind, grid_anchor) = match draw.visible.id.scheme {
                TileScheme::PolarCap { north } => {
                    let (lat0, lon0, dx0, dy0) = sample_tile_surface(draw.visible.id, 0.5, 0.5)
                        .map(|(lat, lon, _, _, dx, dy)| (lat, lon, dx, dy))
                        .unwrap_or((if north { 90.0 } else { -90.0 }, 0.0, 0.0, 0.0));
                    let colat0_deg = 90.0 - lat0.abs();
                    let colat_phase = colat0_deg.rem_euclid(step_deg);
                    let lon_phase = lon0.rem_euclid(step_deg);

                    // Use cancellation-free relative longitude deltas when the
                    // tile's disk center is strictly clear of the pole (`rho0 > 4 * tile_half_diag`).
                    let rho0 = dx0.hypot(dy0);
                    let n = f64::from(1u32 << draw.visible.id.level.min(30));
                    let tile_half_diag = std::f64::consts::SQRT_2 / n;
                    let clear_of_pole = draw.visible.id.level >= 2 && rho0 > 4.0 * tile_half_diag;
                    let sign = if north { 1.0_f32 } else { -1.0_f32 };
                    let kind = if clear_of_pole { sign * 2.0 } else { sign };
                    (
                        kind,
                        [dx0 as f32, dy0 as f32, colat_phase as f32, lon_phase as f32],
                    )
                }
                _ => {
                    let (anchor_a, anchor_b) =
                        tile_grid_anchor(draw.visible.id).unwrap_or((0.0, 0.0));
                    // Wrap tile anchor phase modulo `grid_step_deg` in f64 on
                    // CPU so `f32` fragment arithmetic preserves sub-millimetre
                    // accuracy even at `2 m` altitude.
                    let lon_phase = anchor_a.rem_euclid(step_deg);
                    let lat_phase = anchor_b.rem_euclid(step_deg);
                    (0.0_f32, [lon_phase as f32, lat_phase as f32, 0.0, 0.0])
                }
            };

            let u = TileUniform {
                offset_eye,
                tile_kind,
                uv_rect: draw.uv_rect,
                grid_anchor,
            };
            let bytes = bytemuck::bytes_of(&u);
            let dst_start = i * stride;
            self.uniform_staging[dst_start..dst_start + bytes.len()].copy_from_slice(bytes);
        }

        queue.write_buffer(
            &self.viewport.tile_uniform_buffer,
            0,
            &self.uniform_staging[..total_bytes],
        );
    }

    fn record_offscreen_pass(&self, encoder: &mut wgpu::CommandEncoder, snapshot: &DrawSnapshot) {
        let Some(target) = &self.viewport.offscreen else {
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
                    store: wgpu::StoreOp::Discard,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        pass.set_bind_group(0, &self.viewport.camera_bind_group, &[]);

        // Pass 1: Opaque WGS84 quadtree surface + downward skirts (Reverse-Z).
        let mut last_textured: Option<bool> = None;
        let mut last_tex_id: Option<u64> = None;
        for (i, draw) in snapshot.tiles.iter().enumerate() {
            let Some(vbuf) = self.tile_meshes.peek(&draw.visible.id) else {
                continue;
            };
            let dyn_offset = (i as u32) * self.tile_uniform_stride;
            pass.set_vertex_buffer(0, vbuf.slice(..));
            pass.set_bind_group(1, &self.viewport.tile_uniform_bind_group, &[dyn_offset]);

            if let Some(gpu_tile) = &draw.texture
                && let Some((_, tex_bg)) = self.tile_bind_groups.get(&gpu_tile.tile_id)
            {
                if last_textured != Some(true) {
                    pass.set_pipeline(&self.body_textured_pipeline);
                    last_textured = Some(true);
                }
                if last_tex_id != Some(gpu_tile.tile_id) {
                    pass.set_bind_group(2, tex_bg, &[]);
                    last_tex_id = Some(gpu_tile.tile_id);
                }
            } else if last_textured != Some(false) {
                pass.set_pipeline(&self.body_ellipsoid_pipeline);
                last_textured = Some(false);
                last_tex_id = None;
            }
            pass.draw_indexed(self.full_indices.clone(), 0, 0..1);
        }

        // Pass 2: Post-overlay transparent procedural `fwidth()` graticule
        // reusing the exact same quadtree vertex/index buffers (`surface` only,
        // skipping skirts) so draped raster overlays never bury it.
        if snapshot.show_graticule {
            pass.set_pipeline(&self.grid_pipeline);
            for (i, draw) in snapshot.tiles.iter().enumerate() {
                let Some(vbuf) = self.tile_meshes.peek(&draw.visible.id) else {
                    continue;
                };
                let dyn_offset = (i as u32) * self.tile_uniform_stride;
                pass.set_vertex_buffer(0, vbuf.slice(..));
                pass.set_bind_group(1, &self.viewport.tile_uniform_bind_group, &[dyn_offset]);
                pass.draw_indexed(self.surface_indices.clone(), 0, 0..1);
            }
        }
    }
}
