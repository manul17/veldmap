//! Standalone offscreen `wgpu` WGS84 quadtree globe renderer (`GlobeRenderer`).
//!
//! Renders in two distinct passes:
//! 1. **Pass 1 (Opaque globe body):** `Rgba8UnormSrgb` color clear +
//!    `Depth32Float` Reverse-Z (`TRANSIENT_ATTACHMENT`, `StoreOp::Discard`).
//! 2. **Pass 2 (Post-body overlays + procedural graticule):**
//!    `depth_stencil_attachment: None`, `LoadOp::Load` over the globe body so
//!    draped raster warp meshes and future interleaved analytical passes share
//!    the exact same depth-less color pipelines without duplicating shaders.

use veldmap_core::tile::{POLAR_R_CAP, PixelFormat};

use super::CLEAR_COLOR_LINEAR;
use super::mesh::TileMeshCache;
use super::overlay::OverlayRenderer;
use super::pipelines::{GlobePipelines, TileBindGroupCache};
use super::snapshot::DrawSnapshot;
use super::uniforms::{CameraUniform, TILE_KIND_GEO, TileUniform, tile_graticule_uniform};
use super::viewport::{RenderParams, ViewportState};

/// Standalone offscreen `wgpu` WGS84 quadtree globe renderer.
pub struct GlobeRenderer {
    pipelines: GlobePipelines,
    /// Shared static quadtree index buffer + bounded LRU cache of per-tile
    /// CPU `f64` RTC vertex buffers.
    meshes: TileMeshCache,
    /// Extracted post-body warp-mesh overlay renderer.
    overlay: OverlayRenderer,
    /// `Weak<GpuTile>`-guarded texture `BindGroup` cache (`Rgba8UnormSrgb`).
    tile_bind_groups: TileBindGroupCache,
    /// Per-frame dynamic uniform byte offsets for `snapshot.tiles`.
    tile_uniform_offsets: Vec<u32>,
    /// Per-frame dynamic uniform byte offsets for `snapshot.overlays`.
    overlay_uniform_offsets: Vec<u32>,
    viewport: ViewportState,
}

impl GlobeRenderer {
    /// Create all shared GPU pipelines, static quadtree index buffer, overlay
    /// renderer, and primary `ViewportState` targeting `target_format`.
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let pipelines = GlobePipelines::new(device, target_format);
        let meshes = TileMeshCache::new(device);
        let overlay = OverlayRenderer::new(device, &pipelines);
        let viewport = ViewportState::new(
            device,
            target_format,
            &pipelines.camera_layout,
            &pipelines.tile_uniform_layout,
            &pipelines.overlay_uniform_layout,
        );

        Self {
            pipelines,
            meshes,
            overlay,
            tile_bind_groups: TileBindGroupCache::default(),
            tile_uniform_offsets: Vec::new(),
            overlay_uniform_offsets: Vec::new(),
            viewport,
        }
    }

    /// Drop cached `BindGroup` entries whose backing `GpuTile` has been
    /// evicted from `GpuTileCache` and released by all active snapshots.
    pub fn sweep_stale_bind_groups(&mut self) {
        self.tile_bind_groups.sweep_stale();
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

        if self.viewport.is_clean_for(width, height, params.snapshot) {
            return false;
        }

        self.viewport.ensure_offscreen(
            device,
            &self.pipelines.composite_layout,
            &self.pipelines.composite_sampler,
            width,
            height,
        );
        self.prepare_tile_resources(device, params.snapshot);
        self.upload_uniforms(device, queue, width, height, params.snapshot);
        self.record_offscreen_passes(encoder, params.snapshot);
        self.viewport.mark_rendered(width, height, params.snapshot);
        true
    }

    /// Record the 1:1 `FilterMode::Nearest` composite draw into an existing
    /// render pass targeting the configured swapchain format.
    pub fn composite_into(&self, render_pass: &mut wgpu::RenderPass<'_>) {
        let Some(target) = self.viewport.offscreen() else {
            return;
        };
        self.viewport.mark_presented();
        render_pass.set_pipeline(&self.pipelines.composite_pipeline);
        render_pass.set_bind_group(0, &target.composite_bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }

    fn prepare_tile_resources(&mut self, device: &wgpu::Device, snapshot: &DrawSnapshot) {
        self.meshes
            .prepare_for_tiles(device, snapshot.grid_cells, &snapshot.tiles);

        for draw in &snapshot.tiles {
            if let Some(gpu_tile) = &draw.texture {
                self.tile_bind_groups
                    .ensure(device, &self.pipelines, gpu_tile);
            }
        }

        self.overlay.prepare_resources(
            device,
            &self.pipelines,
            &mut self.tile_bind_groups,
            snapshot,
        );
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
        let vp = snapshot.camera.view_projection(aspect);
        let eye = snapshot.camera.eye_ecef();

        let cam_uniform = CameraUniform {
            view_proj: vp.as_mat4().to_cols_array_2d(),
            grid_step_deg: snapshot.grid_step_deg as f32,
            grid_width_px: snapshot.grid_width_px,
            polar_r_cap: POLAR_R_CAP as f32,
            _pad0: 0.0,
        };
        queue.write_buffer(
            self.viewport.camera_buffer(),
            0,
            bytemuck::bytes_of(&cam_uniform),
        );

        let arena = self.viewport.uniform_arena_mut();
        arena.reset();
        self.tile_uniform_offsets.clear();

        let step_deg = snapshot.grid_step_deg.max(1e-6);
        for draw in &snapshot.tiles {
            let offset_eye = (draw.visible.origin_ecef_m - eye).as_vec3().to_array();
            let (tile_kind, grid_anchor) = if snapshot.show_graticule {
                let cached_anchor = self.meshes.peek(&draw.visible.id).map(|m| m.grid_anchor);
                tile_graticule_uniform(draw.visible.id, cached_anchor, step_deg)
            } else {
                (TILE_KIND_GEO, [0.0; 4])
            };

            let u = TileUniform {
                offset_eye,
                tile_kind,
                uv_rect: draw.uv_rect,
                grid_anchor,
            };
            self.tile_uniform_offsets.push(arena.push(&u));
        }

        self.overlay
            .push_uniforms(arena, snapshot, eye, &mut self.overlay_uniform_offsets);

        arena.flush(
            device,
            queue,
            &self.pipelines.tile_uniform_layout,
            &self.pipelines.overlay_uniform_layout,
        );
    }

    fn record_body_pass<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>, snapshot: &DrawSnapshot) {
        let mut last_textured: Option<bool> = None;
        let mut last_tex_id: Option<u64> = None;
        let full_indices = self.meshes.full_indices();
        let arena = self.viewport.uniform_arena();

        for (i, draw) in snapshot.tiles.iter().enumerate() {
            let Some(mesh) = self.meshes.peek(&draw.visible.id) else {
                continue;
            };
            let dyn_offset = self.tile_uniform_offsets[i];
            pass.set_vertex_buffer(0, mesh.buffer.slice(..));
            pass.set_bind_group(1, arena.tile_bind_group(), &[dyn_offset]);

            if let Some(gpu_tile) = &draw.texture
                && let Some(tex_bg) = self
                    .tile_bind_groups
                    .get(gpu_tile, PixelFormat::Rgba8UnormSrgb)
            {
                if last_textured != Some(true) {
                    pass.set_pipeline(&self.pipelines.body_textured_pipeline);
                    last_textured = Some(true);
                }
                if last_tex_id != Some(gpu_tile.tile_id) {
                    pass.set_bind_group(2, tex_bg, &[]);
                    last_tex_id = Some(gpu_tile.tile_id);
                }
            } else if last_textured != Some(false) {
                pass.set_pipeline(&self.pipelines.body_ellipsoid_pipeline);
                last_textured = Some(false);
                last_tex_id = None;
            }
            pass.draw_indexed(full_indices.clone(), 0, 0..1);
        }
    }

    fn record_graticule_pass<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        snapshot: &DrawSnapshot,
    ) {
        pass.set_pipeline(&self.pipelines.grid_pipeline);
        pass.set_index_buffer(
            self.meshes.index_buffer().slice(..),
            wgpu::IndexFormat::Uint32,
        );
        let surface_indices = self.meshes.surface_indices();
        let arena = self.viewport.uniform_arena();

        for (i, draw) in snapshot.tiles.iter().enumerate() {
            let Some(mesh) = self.meshes.peek(&draw.visible.id) else {
                continue;
            };
            let dyn_offset = self.tile_uniform_offsets[i];
            pass.set_vertex_buffer(0, mesh.buffer.slice(..));
            pass.set_bind_group(1, arena.tile_bind_group(), &[dyn_offset]);
            pass.draw_indexed(surface_indices.clone(), 0, 0..1);
        }
    }

    fn record_offscreen_passes(&self, encoder: &mut wgpu::CommandEncoder, snapshot: &DrawSnapshot) {
        let Some(target) = self.viewport.offscreen() else {
            return;
        };

        // Pass 1: Opaque WGS84 quadtree surface + downward skirts (Reverse-Z,
        // memoryless `TRANSIENT_ATTACHMENT` + `StoreOp::Discard`).
        {
            let mut body_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("globe opaque body pass"),
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

            body_pass.set_index_buffer(
                self.meshes.index_buffer().slice(..),
                wgpu::IndexFormat::Uint32,
            );
            body_pass.set_bind_group(0, self.viewport.camera_bind_group(), &[]);
            self.record_body_pass(&mut body_pass, snapshot);
        }

        // Pass 2: Depth-less (`depth_stencil_attachment: None`, `LoadOp::Load`)
        // post-body raster overlays + procedural `fwidth()` graticule.
        if !snapshot.overlays.is_empty() || snapshot.show_graticule {
            let mut post_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("globe post-body overlay & graticule pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target.color_srgb_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            post_pass.set_bind_group(0, self.viewport.camera_bind_group(), &[]);
            self.overlay.record_pass(
                &mut post_pass,
                self.viewport.uniform_arena(),
                &self.tile_bind_groups,
                snapshot,
                &self.overlay_uniform_offsets,
            );
            if snapshot.show_graticule {
                self.record_graticule_pass(&mut post_pass, snapshot);
            }
        }
    }
}
