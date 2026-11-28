//! Standalone offscreen `wgpu` WGS84 quadtree globe renderer (`GlobeRenderer`).
//!
//! Renders in two distinct phases:
//! 1. **Pass 1 (Opaque globe body):** `Rgba8UnormSrgb` color clear (`COPY_SRC`
//!    for clean-frame PNG export readback) + `Depth32Float` Reverse-Z
//!    (`TRANSIENT_ATTACHMENT`, `StoreOp::Discard`).
//! 2. **Pass 2 (Segmented depth-less post-body overlays, Band Math scratch
//!    passes, split-screen swipe scissors, vectors & procedural graticule):**
//!    `depth_stencil_attachment: None`, `LoadOp::Load` over the globe body.
//!    Consecutive direct display / scientific (`R32Float`) / categorical
//!    (`R8Uint`/`R16Uint`) warp meshes batch together, while each multi-asset
//!    `BandMath` layer interleaves lazy shared `R32Float` + `R8Uint` `SCL`
//!    scratch passes ([`super::scratch::ScratchTargetPool`]) and a scissored
//!    fullscreen composite in exact painter's order (intersected with
//!    [`super::swipe::ResolvedSwipe`] before scratch passes) before restoring
//!    the full viewport scissor for draped vectors and the procedural graticule.

use veldmap_core::tile::{POLAR_R_CAP, PixelFormat};

use super::CLEAR_COLOR_LINEAR;
use super::bind_groups::TileBindGroupCache;
use super::graticule::{TILE_KIND_GEO, tile_graticule_uniform};
use super::mesh::TileMeshCache;
use super::overlay::{BandMathPassCtx, OverlayRenderer};
use super::pipelines::GlobePipelines;
use super::scratch::ScratchDemand;
use super::snapshot::{DrawOverlayContent, DrawSnapshot};
use super::swipe::ResolvedSwipe;
use super::uniforms::{CameraUniform, TileUniform, UniformSlot};
use super::vector::VectorRenderer;
use super::viewport::{RenderParams, ViewportState};

/// Staging buffer holding a GPU-to-CPU copy of [`GlobeRenderer`]'s offscreen
/// `Rgba8UnormSrgb` color target with `256 B`-aligned rows.
pub struct OffscreenReadback {
    buffer: wgpu::Buffer,
    width: u32,
    height: u32,
    padded_bytes_per_row: u32,
}

impl OffscreenReadback {
    /// Physical pixel width of the captured offscreen color target.
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Physical pixel height of the captured offscreen color target.
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Initiate a non-blocking `map_async(MapMode::Read)` on the staging
    /// buffer; call strictly after submitting the encoder that recorded the
    /// copy.
    pub fn map_async(
        &self,
        callback: impl FnOnce(Result<(), wgpu::BufferAsyncError>) + Send + 'static,
    ) {
        self.buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, callback);
    }

    /// Unpad the mapped staging buffer into tightly packed `width * height * 4`
    /// sRGB RGBA8 bytes and unmap the buffer.
    pub fn read_tight_rgba8(self) -> Result<Vec<u8>, String> {
        let row_bytes = (self.width as usize).saturating_mul(4);
        let stride = self.padded_bytes_per_row as usize;
        let height = self.height as usize;
        let mut out = Vec::with_capacity(row_bytes.saturating_mul(height));
        let map_res = {
            match self.buffer.slice(..).get_mapped_range() {
                Ok(mapped) => {
                    let mut truncated = None;
                    for row in 0..height {
                        let start = row.saturating_mul(stride);
                        let end = start.saturating_add(row_bytes);
                        if end > mapped.len() {
                            truncated = Some(format!(
                                "staging buffer truncated at row {row}/{height} ({end} > {})",
                                mapped.len()
                            ));
                            break;
                        }
                        out.extend_from_slice(&mapped[start..end]);
                    }
                    match truncated {
                        Some(msg) => Err(msg),
                        None => Ok(()),
                    }
                }
                Err(err) => Err(err.to_string()),
            }
        };
        self.buffer.unmap();
        map_res.map(|()| out)
    }
}

/// Standalone offscreen `wgpu` WGS84 quadtree globe renderer.
pub struct GlobeRenderer {
    pipelines: GlobePipelines,
    /// Shared static quadtree index buffer + bounded LRU cache of per-tile
    /// CPU `f64` RTC vertex buffers.
    meshes: TileMeshCache,
    /// Extracted post-body warp-mesh overlay renderer.
    overlay: OverlayRenderer,
    /// Post-body draped vector geometry renderer (GeoJSON, measurements, and
    /// STAC AOI / footprints / highlights).
    vector: VectorRenderer,
    /// Per-frame dynamic uniform byte offsets for `snapshot.tiles`.
    tile_uniform_offsets: Vec<u32>,
    viewport: ViewportState,
}

impl GlobeRenderer {
    /// Create a `GlobeRenderer` using the hoisted [`TileBindGroupCache`]
    /// (stored in `egui_wgpu::CallbackResources` in the app, or owned by a
    /// test harness in headless tests).
    pub fn new(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        shared_bind_groups: &TileBindGroupCache,
    ) -> Self {
        let pipelines = GlobePipelines::new(device, target_format, shared_bind_groups);
        let meshes = TileMeshCache::new(device);
        let overlay = OverlayRenderer::new();
        let vector = VectorRenderer::new();
        let viewport = ViewportState::new(device, target_format, &pipelines);

        Self {
            pipelines,
            meshes,
            overlay,
            vector,
            tile_uniform_offsets: Vec::new(),
            viewport,
        }
    }

    /// Ensure the offscreen target matches `(params.width, params.height)`
    /// clamped to device limits and re-render the globe if the target or draw
    /// snapshot changed, sharing the hoisted `bind_groups` cache.
    pub fn render_if_dirty(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        params: RenderParams<'_>,
        bind_groups: &mut TileBindGroupCache,
    ) -> bool {
        bind_groups.sweep_stale();
        if params.width == 0 || params.height == 0 {
            self.viewport.release_targets();
            return false;
        }
        let max_dim = device.limits().max_texture_dimension_2d;
        let width = params.width.clamp(1, max_dim);
        let height = params.height.clamp(1, max_dim);
        let resolved_swipe = params
            .swipe
            .map(|state| ResolvedSwipe::from_state(state, width));

        if self
            .viewport
            .is_clean_for(width, height, params.snapshot, resolved_swipe)
        {
            return false;
        }

        self.viewport
            .ensure_offscreen(device, &self.pipelines, width, height);
        let mut scratch_demand = ScratchDemand::NONE;
        for layer in &params.snapshot.overlays {
            scratch_demand = scratch_demand.merge(layer.scratch_requirements());
        }
        self.viewport.ensure_scratch(
            device,
            queue,
            &self.pipelines.band_math,
            width,
            height,
            scratch_demand,
        );
        self.prepare_tile_resources(device, bind_groups, params.snapshot);
        self.upload_uniforms(
            device,
            queue,
            width,
            height,
            params.snapshot,
            resolved_swipe,
        );
        self.record_offscreen_passes(encoder, bind_groups, params.snapshot, resolved_swipe);
        self.viewport
            .mark_rendered(width, height, params.snapshot, resolved_swipe);
        true
    }

    /// Encode a `256 B`-aligned GPU-to-CPU copy of the current offscreen
    /// `Rgba8UnormSrgb` color texture into a `MAP_READ` staging buffer.
    ///
    /// Works identically on both freshly rendered and clean frames
    /// (`is_clean_for == true`) whenever an offscreen target is allocated.
    pub fn encode_offscreen_readback(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Option<OffscreenReadback> {
        let target = self.viewport.offscreen()?;
        if target.width == 0 || target.height == 0 {
            return None;
        }
        let unpadded_bpr = target.width.checked_mul(4)?;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row = unpadded_bpr.div_ceil(align).checked_mul(align)?;
        let buffer_size = u64::from(padded_bytes_per_row).checked_mul(u64::from(target.height))?;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("globe offscreen PNG readback staging"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            target.color_texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(target.height),
                },
            },
            wgpu::Extent3d {
                width: target.width,
                height: target.height,
                depth_or_array_layers: 1,
            },
        );
        Some(OffscreenReadback {
            buffer,
            width: target.width,
            height: target.height,
            padded_bytes_per_row,
        })
    }

    /// Drop offscreen color/depth and Band Math scratch targets immediately
    /// when the viewport collapses/minimizes.
    pub fn release_viewport_targets(&mut self) {
        self.viewport.release_targets();
    }

    /// Currently allocated viewport scratch target VRAM in bytes (0 when no
    /// analytical Band Math layer is visible).
    #[cfg(any(test, feature = "test-support"))]
    pub fn scratch_vram_bytes(&self) -> usize {
        self.viewport.scratch_vram_bytes()
    }

    /// Record the 1:1 `FilterMode::Nearest` composite draw into an existing
    /// render pass targeting the configured swapchain format.
    pub fn composite_into(&self, render_pass: &mut wgpu::RenderPass<'_>) {
        let Some(target) = self.viewport.offscreen() else {
            return;
        };
        self.viewport.mark_presented();
        render_pass.set_pipeline(&self.pipelines.composite.pipeline);
        render_pass.set_bind_group(0, &target.composite_bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }

    fn prepare_tile_resources(
        &mut self,
        device: &wgpu::Device,
        bind_groups: &mut TileBindGroupCache,
        snapshot: &DrawSnapshot,
    ) {
        self.meshes
            .prepare_for_tiles(device, snapshot.grid_cells, &snapshot.tiles);
        bind_groups.begin_mrt_frame();

        for draw in &snapshot.tiles {
            if let Some(gpu_tile) = &draw.texture {
                bind_groups.ensure(device, gpu_tile);
            }
        }

        self.overlay.prepare_resources(
            device,
            &self.pipelines.layouts.palette_uniform_layout,
            bind_groups,
            snapshot,
        );
        self.vector.prepare_resources(device, snapshot);
    }

    fn upload_uniforms(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        snapshot: &DrawSnapshot,
        swipe: Option<ResolvedSwipe>,
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
            .push_uniforms(arena, snapshot, eye, width, height, swipe);
        self.vector
            .push_uniforms(arena, snapshot, eye, width, height);

        arena.flush(device, queue);
    }

    fn record_body_pass<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        bind_groups: &'a TileBindGroupCache,
        snapshot: &DrawSnapshot,
    ) {
        let mut last_textured: Option<bool> = None;
        let mut last_tex_id: Option<u64> = None;
        let full_indices = self.meshes.full_indices();
        let arena = self.viewport.uniform_arena();

        for (draw, &dyn_offset) in snapshot.tiles.iter().zip(self.tile_uniform_offsets.iter()) {
            let Some(mesh) = self.meshes.peek(&draw.visible.id) else {
                continue;
            };
            pass.set_vertex_buffer(0, mesh.buffer.slice(..));
            pass.set_bind_group(1, arena.bind_group(UniformSlot::Tile), &[dyn_offset]);

            if let Some(gpu_tile) = &draw.texture
                && let Some(tex_bg) = bind_groups.get(gpu_tile, PixelFormat::Rgba8UnormSrgb)
            {
                if last_textured != Some(true) {
                    pass.set_pipeline(&self.pipelines.body.textured);
                    last_textured = Some(true);
                }
                if last_tex_id != Some(gpu_tile.tile_id) {
                    pass.set_bind_group(2, tex_bg, &[]);
                    last_tex_id = Some(gpu_tile.tile_id);
                }
            } else if last_textured != Some(false) {
                pass.set_pipeline(&self.pipelines.body.ellipsoid);
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
        pass.set_pipeline(&self.pipelines.body.grid);
        pass.set_index_buffer(
            self.meshes.index_buffer().slice(..),
            wgpu::IndexFormat::Uint32,
        );
        let surface_indices = self.meshes.surface_indices();
        let arena = self.viewport.uniform_arena();

        for (draw, &dyn_offset) in snapshot.tiles.iter().zip(self.tile_uniform_offsets.iter()) {
            let Some(mesh) = self.meshes.peek(&draw.visible.id) else {
                continue;
            };
            pass.set_vertex_buffer(0, mesh.buffer.slice(..));
            pass.set_bind_group(1, arena.bind_group(UniformSlot::Tile), &[dyn_offset]);
            pass.draw_indexed(surface_indices.clone(), 0, 0..1);
        }
    }

    fn record_vector_and_graticule_tail<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        snapshot: &'a DrawSnapshot,
    ) {
        self.vector.record_pass(
            pass,
            &self.pipelines.vector,
            self.viewport.uniform_arena(),
            snapshot,
        );
        if snapshot.show_graticule {
            self.record_graticule_pass(pass, snapshot);
        }
    }

    fn record_offscreen_passes(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_groups: &TileBindGroupCache,
        snapshot: &DrawSnapshot,
        swipe: Option<ResolvedSwipe>,
    ) {
        let Some(target) = self.viewport.offscreen() else {
            return;
        };

        // Pass 1: Opaque WGS84 quadtree surface + downward skirts (Reverse-Z).
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
            self.record_body_pass(&mut body_pass, bind_groups, snapshot);
        }

        // Pass 2: Segmented depth-less (`depth_stencil_attachment: None`,
        // `LoadOp::Load`) post-body overlays, vectors, and procedural graticule.
        //
        // Consecutive direct overlays batch into a single color pass; each
        // scratch-backed `BandMath` layer interleaves `[shared <=4-target
        // scratch passes] -> [depth-less color LoadOp::Load composite pass]` in
        // exact painter's order so transient scratch VRAM stays bounded to one
        // viewport pool regardless of layer count.
        let has_post_work = !snapshot.overlays.is_empty()
            || !snapshot.vectors.is_empty()
            || snapshot.show_graticule;
        if !has_post_work {
            return;
        }

        fn begin_color_load_pass<'a>(
            encoder: &'a mut wgpu::CommandEncoder,
            color_view: &'a wgpu::TextureView,
        ) -> wgpu::RenderPass<'a> {
            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("globe post-body overlay, vector & graticule pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
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
            })
        }

        let has_tail = !snapshot.vectors.is_empty() || snapshot.show_graticule;
        let mut tail_drawn = false;
        let mut layer_cursor = 0usize;
        let overlay_len = snapshot.overlays.len();

        while layer_cursor < overlay_len {
            let bm_to_composite = if let DrawOverlayContent::BandMath(bm) =
                &snapshot.overlays[layer_cursor].content
            {
                let bm_idx = layer_cursor;
                layer_cursor += 1;
                let Some(scratch) = self.viewport.scratch() else {
                    continue;
                };
                let bm_ctx = BandMathPassCtx {
                    camera_bg: self.viewport.camera_bind_group(),
                    pipelines: &self.pipelines.band_math,
                    scratch,
                    arena: self.viewport.uniform_arena(),
                    bind_groups,
                };
                let drawn = self
                    .overlay
                    .record_band_math_scratch_passes(encoder, bm_idx, bm, &bm_ctx);
                if !drawn {
                    continue;
                }
                Some((bm_idx, bm_ctx))
            } else {
                None
            };

            // Batch an optional BandMath fullscreen composite plus any
            // immediately following direct overlays (and final vectors /
            // graticule if this reaches the tail) into one `LoadOp::Load` pass.
            let mut post_pass = begin_color_load_pass(encoder, &target.color_srgb_view);
            if let Some((bm_idx, bm_ctx)) = bm_to_composite {
                self.overlay.record_band_math_composite(
                    &mut post_pass,
                    bm_idx,
                    target.width,
                    target.height,
                    &bm_ctx,
                );
            }
            post_pass.set_bind_group(0, self.viewport.camera_bind_group(), &[]);
            let full_scissor = [0, 0, target.width, target.height];
            let mut active_scissor = full_scissor;
            while layer_cursor < overlay_len
                && matches!(
                    snapshot.overlays[layer_cursor].content,
                    DrawOverlayContent::Direct { .. }
                )
            {
                let layer = &snapshot.overlays[layer_cursor];
                let layer_scissor = match swipe {
                    Some(sw) => sw.scissor_for_layer(layer.layer_id, target.width, target.height),
                    None => Some(full_scissor),
                };
                if let Some(scissor) = layer_scissor {
                    if scissor != active_scissor {
                        post_pass.set_scissor_rect(scissor[0], scissor[1], scissor[2], scissor[3]);
                        active_scissor = scissor;
                    }
                    self.overlay.record_direct_layer(
                        &mut post_pass,
                        layer_cursor,
                        layer,
                        &self.pipelines.overlay,
                        self.viewport.uniform_arena(),
                        bind_groups,
                    );
                }
                layer_cursor += 1;
            }
            if active_scissor != full_scissor {
                post_pass.set_scissor_rect(0, 0, target.width, target.height);
            }
            if layer_cursor == overlay_len && has_tail {
                self.record_vector_and_graticule_tail(&mut post_pass, snapshot);
                tail_drawn = true;
            }
        }

        // If no overlay color pass reached the tail (zero overlays, or trailing
        // skipped/off-screen BandMath layers), record vectors & graticule once.
        if has_tail && !tail_drawn {
            let mut post_pass = begin_color_load_pass(encoder, &target.color_srgb_view);
            post_pass.set_bind_group(0, self.viewport.camera_bind_group(), &[]);
            self.record_vector_and_graticule_tail(&mut post_pass, snapshot);
        }
    }
}
