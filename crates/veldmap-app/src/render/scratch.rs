//! Lazy shared viewport-sized `R32Float` + `R8Uint` scratch target pool
//! ([`ScratchTargetPool`]) and conservative projected-AABB scissor-rect
//! calculation ([`compute_band_math_scissor_rect`]) for screen-space Band Math
//! and `SCL` masking.
//!
//! Scratch targets strictly use `RENDER_ATTACHMENT | TEXTURE_BINDING` (never
//! `TRANSIENT_ATTACHMENT`), are allocated lazily for the exact active band
//! count (`1..=4` `R32Float` + optional `R8Uint`), and are released immediately
//! when no analytical Band Math layer is visible.

use veldmap_core::geo::{Camera, WGS84};
use veldmap_core::glam::{DVec3, DVec4};
use wgpu::util::DeviceExt;

use super::pipelines::BandMathPipelines;
use super::snapshot::DrawBandMathLayer;
use super::uniforms::ScratchClearUniform;

/// Quiet-NaN bit pattern (`0x7FC0_0000u`) passed via runtime uniform to
/// `scratch_nan_fill.wgsl` (avoiding WGSL compile-time NaN const-eval).
pub(super) const QUIET_NAN_BITS: u32 = 0x7FC0_0000;

/// Uncovered sentinel written via `LoadOp::Clear(255.0)` into the `R8Uint`
/// `SCL` scratch target so pixels outside the mask footprint are never falsely
/// masked by `SCL` class `0`.
pub(super) const MASK_UNCOVERED_SENTINEL: u8 = 255;

/// Compute exact VRAM bytes occupied by viewport offscreen + scratch targets
/// (`OffscreenTarget` = `8 B/px`, each `R32Float` scratch = `4 B/px`, optional
/// `R8Uint` mask = `1 B/px`).
pub(crate) fn estimate_viewport_target_vram_bytes(
    width: u32,
    height: u32,
    float_bands: usize,
    needs_mask: bool,
) -> usize {
    if width == 0 || height == 0 {
        return 0;
    }
    let pixels = (width as usize).saturating_mul(height as usize);
    let offscreen_bpp = 8usize; // Rgba8UnormSrgb (4) + Depth32Float (4)
    let scratch_bpp = float_bands.min(4).saturating_mul(4) + usize::from(needs_mask);
    pixels.saturating_mul(offscreen_bpp.saturating_add(scratch_bpp))
}

struct AllocatedScratch {
    width: u32,
    height: u32,
    float_views: Vec<wgpu::TextureView>,
    mask_view: Option<wgpu::TextureView>,
    composite_bind_group: wgpu::BindGroup,
}

#[cfg(any(test, feature = "test-support"))]
impl AllocatedScratch {
    fn vram_bytes(&self) -> usize {
        let pixels = (self.width as usize).saturating_mul(self.height as usize);
        let bpp = self.float_views.len().saturating_mul(4) + usize::from(self.mask_view.is_some());
        pixels.saturating_mul(bpp)
    }
}

/// Lazily allocated shared viewport scratch targets (`1..=4` `R32Float` +
/// optional `R8Uint` mask) plus immutable 1×1 dummy textures and the runtime
/// quiet-NaN clear uniform buffer.
pub(super) struct ScratchTargetPool {
    _nan_clear_buffer: wgpu::Buffer,
    nan_clear_bind_group: wgpu::BindGroup,
    dummy_float_view: wgpu::TextureView,
    dummy_mask_view: wgpu::TextureView,
    allocated: Option<AllocatedScratch>,
}

impl ScratchTargetPool {
    pub(super) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pipelines: &BandMathPipelines,
    ) -> Self {
        let clear_u = ScratchClearUniform {
            nan_bits: QUIET_NAN_BITS,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let nan_clear_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("scratch quiet-NaN clear uniform"),
            contents: bytemuck::bytes_of(&clear_u),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let nan_clear_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scratch quiet-NaN clear bind group"),
            layout: &pipelines.clear_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: nan_clear_buffer.as_entire_binding(),
            }],
        });

        let dummy_float_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("scratch 1x1 dummy R32Float"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            dummy_float_tex.as_image_copy(),
            &QUIET_NAN_BITS.to_ne_bytes(),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let dummy_float_view = dummy_float_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let dummy_mask_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("scratch 1x1 dummy R8Uint"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            dummy_mask_tex.as_image_copy(),
            &[MASK_UNCOVERED_SENTINEL],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(1),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let dummy_mask_view = dummy_mask_tex.create_view(&wgpu::TextureViewDescriptor::default());

        Self {
            _nan_clear_buffer: nan_clear_buffer,
            nan_clear_bind_group,
            dummy_float_view,
            dummy_mask_view,
            allocated: None,
        }
    }

    /// Ensure scratch targets match `(width, height, float_bands, needs_mask)`,
    /// releasing all scratch VRAM immediately when `float_bands == 0`.
    pub(super) fn ensure(
        &mut self,
        device: &wgpu::Device,
        pipelines: &BandMathPipelines,
        width: u32,
        height: u32,
        float_bands: usize,
        needs_mask: bool,
    ) {
        let float_bands = float_bands.min(4);
        if float_bands == 0 || width == 0 || height == 0 {
            self.allocated = None;
            return;
        }
        if let Some(cur) = &self.allocated
            && cur.width == width
            && cur.height == height
            && cur.float_views.len() == float_bands
            && cur.mask_view.is_some() == needs_mask
        {
            return;
        }

        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let mut float_views = Vec::with_capacity(float_bands);
        for i in 0..float_bands {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(match i {
                    0 => "band-math scratch R32Float [0]",
                    1 => "band-math scratch R32Float [1]",
                    2 => "band-math scratch R32Float [2]",
                    _ => "band-math scratch R32Float [3]",
                }),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                // Never combine `TRANSIENT_ATTACHMENT` with `TEXTURE_BINDING`.
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            float_views.push(tex.create_view(&wgpu::TextureViewDescriptor::default()));
        }

        let mask_view = needs_mask.then(|| {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("band-math scratch R8Uint SCL mask"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R8Uint,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            tex.create_view(&wgpu::TextureViewDescriptor::default())
        });

        let view_for = |idx: usize| -> &wgpu::TextureView {
            float_views.get(idx).unwrap_or(&self.dummy_float_view)
        };
        let mask_binding = mask_view.as_ref().unwrap_or(&self.dummy_mask_view);
        let composite_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("band-math composite scratch textures bind group"),
            layout: &pipelines.composite_scratch_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(view_for(0)),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(view_for(1)),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(view_for(2)),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(view_for(3)),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(mask_binding),
                },
            ],
        });

        self.allocated = Some(AllocatedScratch {
            width,
            height,
            float_views,
            mask_view,
            composite_bind_group,
        });
    }

    pub(super) fn release(&mut self) {
        self.allocated = None;
    }

    #[cfg(any(test, feature = "test-support"))]
    pub(super) fn vram_bytes(&self) -> usize {
        self.allocated
            .as_ref()
            .map_or(0, AllocatedScratch::vram_bytes)
    }

    pub(super) fn nan_clear_bind_group(&self) -> &wgpu::BindGroup {
        &self.nan_clear_bind_group
    }

    pub(super) fn float_views(&self) -> &[wgpu::TextureView] {
        self.allocated
            .as_ref()
            .map_or(&[], |a| a.float_views.as_slice())
    }

    pub(super) fn mask_view(&self) -> Option<&wgpu::TextureView> {
        self.allocated.as_ref().and_then(|a| a.mask_view.as_ref())
    }

    pub(super) fn composite_bind_group(&self) -> Option<&wgpu::BindGroup> {
        self.allocated.as_ref().map(|a| &a.composite_bind_group)
    }
}

/// Compute a conservative clamped screen-space scissor rectangle `[x, y, w, h]`
/// enclosing all visible patches of `bm`, falling back to full viewport
/// `[0, 0, width, height]` whenever any projected sample has `clip.w <= near`,
/// straddles the ellipsoid horizon limb, or spans a wide angular arc (`> 4°`),
/// and returning `None` when the clamped scissor rectangle is empty.
pub fn compute_band_math_scissor_rect(
    camera: &Camera,
    width: u32,
    height: u32,
    bm: &DrawBandMathLayer,
) -> Option<[u32; 4]> {
    if width == 0 || height == 0 {
        return None;
    }
    let full = [0, 0, width, height];
    let aspect = f64::from(width) / f64::from(height);
    let vp = camera.view_projection(aspect);
    let eye = camera.eye_ecef();
    let near_m = camera.height_m() * 0.5;
    let wf = f64::from(width);
    let hf = f64::from(height);

    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    let mut any_sample = false;

    let mut visit_patch =
        |frame: &veldmap_core::geo::Frame, img_w: u32, img_h: u32, cell_rect: [f64; 4]| -> bool {
            let inv_w = 1.0 / f64::from(img_w.max(1));
            let inv_h = 1.0 / f64::from(img_h.max(1));
            let fx0 = (cell_rect[0] * inv_w).clamp(0.0, 1.0);
            let fy0 = (cell_rect[1] * inv_h).clamp(0.0, 1.0);
            let fx1 = (cell_rect[2] * inv_w).clamp(0.0, 1.0);
            let fy1 = (cell_rect[3] * inv_h).clamp(0.0, 1.0);

            let mut lat_lo = f64::INFINITY;
            let mut lat_hi = f64::NEG_INFINITY;
            let mut lon_lo = f64::INFINITY;
            let mut lon_hi = f64::NEG_INFINITY;

            // Project a 3x3 grid across the cell (4 corners + 4 edge midpoints + center)
            // to capture ellipsoid curvature and horizon-limb straddling cleanly.
            for iy in 0..=2 {
                let fy = fy0 + (fy1 - fy0) * (f64::from(iy) * 0.5);
                for ix in 0..=2 {
                    let fx = fx0 + (fx1 - fx0) * (f64::from(ix) * 0.5);
                    let (lat, lon) = frame.geodetic(fx, fy);
                    if !lat.is_finite() || !lon.is_finite() {
                        return false;
                    }
                    lat_lo = lat_lo.min(lat);
                    lat_hi = lat_hi.max(lat);
                    lon_lo = lon_lo.min(lon);
                    lon_hi = lon_hi.max(lon);

                    let (ecef, normal) = WGS84.surface_ecef_and_normal(lat, lon);
                    let rel: DVec3 = ecef - eye;
                    // Back-hemisphere / horizon limb straddle -> fall back to full target.
                    if normal.dot(-rel) <= 0.0 {
                        return false;
                    }
                    let clip = vp * DVec4::new(rel.x, rel.y, rel.z, 1.0);
                    if !clip.w.is_finite() || clip.w <= near_m {
                        return false;
                    }
                    let ndc_x = clip.x / clip.w;
                    let ndc_y = clip.y / clip.w;
                    if !ndc_x.is_finite() || !ndc_y.is_finite() {
                        return false;
                    }
                    let sx = (ndc_x * 0.5 + 0.5) * wf;
                    let sy = (0.5 - ndc_y * 0.5) * hf;
                    min_x = min_x.min(sx);
                    max_x = max_x.max(sx);
                    min_y = min_y.min(sy);
                    max_y = max_y.max(sy);
                    any_sample = true;
                }
            }
            // Wide angular cells (>4 deg) can curve outside a 3x3 sample AABB.
            if (lat_hi - lat_lo) > 4.0 || (lon_hi - lon_lo) > 4.0 {
                return false;
            }
            true
        };

    for band in &bm.bands {
        for patch in &band.patches {
            if !visit_patch(
                &band.frame,
                band.image_width,
                band.image_height,
                patch.cell_rect,
            ) {
                return Some(full);
            }
        }
    }
    if let Some(scl) = &bm.scl
        && bm.scl_mask_enabled
    {
        for patch in &scl.patches {
            if !visit_patch(
                &scl.frame,
                scl.image_width,
                scl.image_height,
                patch.cell_rect,
            ) {
                return Some(full);
            }
        }
    }

    if !any_sample {
        return None;
    }

    // Pad conservatively by 2 physical pixels to cover sub-pixel triangle edges.
    let x0 = (min_x.floor() - 2.0).clamp(0.0, wf) as u32;
    let y0 = (min_y.floor() - 2.0).clamp(0.0, hf) as u32;
    let x1 = (max_x.ceil() + 2.0).clamp(0.0, wf) as u32;
    let y1 = (max_y.ceil() + 2.0).clamp(0.0, hf) as u32;
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    Some([x0, y0, x1 - x0, y1 - y0])
}
