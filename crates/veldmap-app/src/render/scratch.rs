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

/// Viewport scratch target requirements aggregated across active analytical
/// layers (Band Math, dual-epoch scalar diff, dual-epoch RGB cross-fade, and
/// Slice 5B DEM relief scratch targets).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ScratchDemand {
    /// Maximum concurrent `R32Float` band targets required (`0..=4`).
    pub(crate) float_bands: usize,
    /// Whether an `R8Uint` `SCL` mask target is required.
    pub(crate) needs_u8_mask: bool,
    /// Maximum concurrent `Rgba8UnormSrgb` scratch targets required (`0..=2`).
    pub(crate) rgba_targets: usize,
}

impl ScratchDemand {
    /// Zero scratch targets required.
    pub(crate) const NONE: Self = Self {
        float_bands: 0,
        needs_u8_mask: false,
        rgba_targets: 0,
    };

    /// Whether zero scratch targets of any kind are requested.
    pub(crate) const fn is_empty(self) -> bool {
        self.float_bands == 0 && !self.needs_u8_mask && self.rgba_targets == 0
    }

    /// Combine two demands into the shared pool capacity covering both.
    pub(crate) const fn merge(self, other: Self) -> Self {
        Self {
            float_bands: if self.float_bands >= other.float_bands {
                self.float_bands
            } else {
                other.float_bands
            },
            needs_u8_mask: self.needs_u8_mask || other.needs_u8_mask,
            rgba_targets: if self.rgba_targets >= other.rgba_targets {
                self.rgba_targets
            } else {
                other.rgba_targets
            },
        }
    }
}

/// Compute exact VRAM bytes occupied by viewport offscreen + scratch targets
/// (`OffscreenTarget` = `8 B/px`, each `R32Float` scratch = `4 B/px`, optional
/// `R8Uint` mask = `1 B/px`, each `Rgba8UnormSrgb` scratch = `4 B/px`).
pub(crate) fn estimate_viewport_target_vram_bytes(
    width: u32,
    height: u32,
    demand: ScratchDemand,
) -> usize {
    if width == 0 || height == 0 {
        return 0;
    }
    let pixels = (width as usize).saturating_mul(height as usize);
    let offscreen_bpp = 8usize; // Rgba8UnormSrgb (4) + Depth32Float (4)
    let scratch_bpp = demand.float_bands.min(4).saturating_mul(4)
        + usize::from(demand.needs_u8_mask)
        + demand.rgba_targets.min(2).saturating_mul(4);
    pixels.saturating_mul(offscreen_bpp.saturating_add(scratch_bpp))
}

struct AllocatedScratch {
    width: u32,
    height: u32,
    float_views: Vec<wgpu::TextureView>,
    mask_view: Option<wgpu::TextureView>,
    rgba_views: Vec<wgpu::TextureView>,
    composite_bind_group: wgpu::BindGroup,
    rgba_bind_group: wgpu::BindGroup,
    /// Non-overlapping read-only reduce bind groups for `N = 1, 2, 3` input
    /// bands (`index = N - 1`): binds ONLY `float_views[0..N]` + `mask` (never
    /// `float_views[N]`, which is the render attachment).
    reduce_bind_groups: [Option<wgpu::BindGroup>; 3],
}

#[cfg(any(test, feature = "test-support"))]
impl AllocatedScratch {
    fn vram_bytes(&self) -> usize {
        let pixels = (self.width as usize).saturating_mul(self.height as usize);
        let bpp = self.float_views.len().saturating_mul(4)
            + usize::from(self.mask_view.is_some())
            + self.rgba_views.len().saturating_mul(4);
        pixels.saturating_mul(bpp)
    }
}

/// Lazily allocated shared viewport scratch targets (`0..=4` `R32Float`,
/// optional `R8Uint` mask, `0..=2` `Rgba8UnormSrgb`) plus immutable 1×1 dummy
/// textures and the runtime quiet-NaN clear uniform buffer.
pub(super) struct ScratchTargetPool {
    _nan_clear_buffer: wgpu::Buffer,
    nan_clear_bind_group: wgpu::BindGroup,
    dummy_float_view: wgpu::TextureView,
    dummy_mask_view: wgpu::TextureView,
    dummy_rgba_view: wgpu::TextureView,
    dummy_rgba_bind_group: wgpu::BindGroup,
    allocated: Option<AllocatedScratch>,
}

fn create_dummy_1x1_view(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &'static str,
    format: wgpu::TextureFormat,
    texel_bytes: &[u8],
) -> wgpu::TextureView {
    let size = wgpu::Extent3d {
        width: 1,
        height: 1,
        depth_or_array_layers: 1,
    };
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        tex.as_image_copy(),
        texel_bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(texel_bytes.len() as u32),
            rows_per_image: Some(1),
        },
        size,
    );
    tex.create_view(&wgpu::TextureViewDescriptor::default())
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

        let dummy_float_view = create_dummy_1x1_view(
            device,
            queue,
            "scratch 1x1 dummy R32Float",
            wgpu::TextureFormat::R32Float,
            &QUIET_NAN_BITS.to_ne_bytes(),
        );
        let dummy_mask_view = create_dummy_1x1_view(
            device,
            queue,
            "scratch 1x1 dummy R8Uint",
            wgpu::TextureFormat::R8Uint,
            &[MASK_UNCOVERED_SENTINEL],
        );
        let dummy_rgba_view = create_dummy_1x1_view(
            device,
            queue,
            "scratch 1x1 dummy Rgba8UnormSrgb",
            wgpu::TextureFormat::Rgba8UnormSrgb,
            &[0, 0, 0, 0],
        );
        let dummy_rgba_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scratch dummy rgba bind group (@group(3))"),
            layout: &pipelines.rgba_scratch_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&dummy_rgba_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&dummy_rgba_view),
                },
            ],
        });

        Self {
            _nan_clear_buffer: nan_clear_buffer,
            nan_clear_bind_group,
            dummy_float_view,
            dummy_mask_view,
            dummy_rgba_view,
            dummy_rgba_bind_group,
            allocated: None,
        }
    }

    /// Ensure scratch targets match `(width, height, demand)`, releasing all
    /// scratch VRAM immediately when `demand.is_empty()`.
    pub(super) fn ensure(
        &mut self,
        device: &wgpu::Device,
        pipelines: &BandMathPipelines,
        width: u32,
        height: u32,
        demand: ScratchDemand,
    ) {
        let float_bands = demand.float_bands.min(4);
        let needs_mask = demand.needs_u8_mask;
        let rgba_targets = demand.rgba_targets.min(2);
        if demand.is_empty() || width == 0 || height == 0 {
            self.allocated = None;
            return;
        }
        if let Some(cur) = &self.allocated
            && cur.width == width
            && cur.height == height
            && cur.float_views.len() == float_bands
            && cur.mask_view.is_some() == needs_mask
            && cur.rgba_views.len() == rgba_targets
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

        let mut rgba_views = Vec::with_capacity(rgba_targets);
        for i in 0..rgba_targets {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(if i == 0 {
                    "scratch Rgba8UnormSrgb [0]"
                } else {
                    "scratch Rgba8UnormSrgb [1]"
                }),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            rgba_views.push(tex.create_view(&wgpu::TextureViewDescriptor::default()));
        }

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

        let rgba_for = |idx: usize| -> &wgpu::TextureView {
            rgba_views.get(idx).unwrap_or(&self.dummy_rgba_view)
        };
        let rgba_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scratch rgba bind group (@group(3))"),
            layout: &pipelines.rgba_scratch_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(rgba_for(0)),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(rgba_for(1)),
                },
            ],
        });

        // Build non-overlapping reduce bind groups for `N = 1, 2, 3` whenever
        // `float_bands >= N + 1`: bind ONLY `float_views[0..N]` + `mask`, NEVER
        // `float_views[N]` (which serves as the reduce render target).
        let reduce_bind_groups: [Option<wgpu::BindGroup>; 3] = std::array::from_fn(|idx| {
            let n = idx + 1;
            if float_bands <= n {
                return None;
            }
            let r_view = |slot: usize| -> &wgpu::TextureView {
                if slot < n {
                    &float_views[slot]
                } else {
                    &self.dummy_float_view
                }
            };
            Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(match n {
                    1 => "band-math reduce non-overlapping bind group (N=1)",
                    2 => "band-math reduce non-overlapping bind group (N=2)",
                    _ => "band-math reduce non-overlapping bind group (N=3)",
                }),
                layout: &pipelines.reduce_scratch_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(r_view(0)),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(r_view(1)),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(r_view(2)),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(mask_binding),
                    },
                ],
            }))
        });

        self.allocated = Some(AllocatedScratch {
            width,
            height,
            float_views,
            mask_view,
            rgba_views,
            composite_bind_group,
            rgba_bind_group,
            reduce_bind_groups,
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

    pub(super) fn rgba_views(&self) -> &[wgpu::TextureView] {
        self.allocated
            .as_ref()
            .map_or(&[], |a| a.rgba_views.as_slice())
    }

    pub(super) fn composite_bind_group(&self) -> Option<&wgpu::BindGroup> {
        self.allocated.as_ref().map(|a| &a.composite_bind_group)
    }

    pub(super) fn rgba_bind_group(&self) -> Option<&wgpu::BindGroup> {
        self.allocated.as_ref().map(|a| &a.rgba_bind_group)
    }

    pub(super) fn dummy_rgba_bind_group(&self) -> &wgpu::BindGroup {
        &self.dummy_rgba_bind_group
    }

    /// Return the non-overlapping read-only reduce bind group for `input_bands`
    /// (`1..=3`), guaranteeing `float_views[input_bands]` is never bound while
    /// being written.
    pub(super) fn reduce_bind_group(&self, input_bands: usize) -> Option<&wgpu::BindGroup> {
        let idx = input_bands.checked_sub(1)?;
        self.allocated
            .as_ref()
            .and_then(|a| a.reduce_bind_groups.get(idx))
            .and_then(Option::as_ref)
    }
}

/// Compute a conservative clamped screen-space scissor rectangle `[x, y, w, h]`
/// enclosing all visible patches across `patch_sets`, falling back to full
/// viewport `[0, 0, width, height]` whenever any projected sample has
/// `clip.w <= near`, straddles the ellipsoid horizon limb, or spans a wide
/// angular arc (`> 4°`), and returning `None` when the clamped scissor
/// rectangle is empty.
pub(crate) fn compute_patch_sets_scissor_rect<'a>(
    camera: &Camera,
    width: u32,
    height: u32,
    patch_sets: impl IntoIterator<Item = super::snapshot::WarpPatchSetRef<'a>>,
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

    for set in patch_sets {
        for patch in set.patches {
            if !visit_patch(
                set.frame,
                set.image_width,
                set.image_height,
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

/// Compute a conservative clamped screen-space scissor rectangle `[x, y, w, h]`
/// enclosing all visible patches of `bm`.
pub fn compute_band_math_scissor_rect(
    camera: &Camera,
    width: u32,
    height: u32,
    bm: &DrawBandMathLayer,
) -> Option<[u32; 4]> {
    compute_patch_sets_scissor_rect(camera, width, height, bm.active_warp_patch_sets(false))
}
