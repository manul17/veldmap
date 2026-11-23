//! `egui_wgpu` viewport bridge (`SnapshotKey`, `GlobePaintCallback`, and
//! logical-to-physical viewport pixel mapping) extracted from `app.rs`.

use std::sync::Arc;

use eframe::egui;
use eframe::egui_wgpu::{CallbackResources, CallbackTrait, ScreenDescriptor};
use veldmap_core::geo::camera::Camera;
use veldmap_core::tile::SourceId;

use crate::render::{DrawSnapshot, GlobeRenderer, RenderParams};
use crate::settings::BasemapSelection;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SnapshotKey {
    eye_bits: [u64; 3],
    up_bits: [u64; 3],
    width_px: u32,
    height_px: u32,
    ppp_bits: u32,
    basemap: BasemapSelection,
    active_source: SourceId,
    show_graticule: bool,
    scene_generation: u64,
}

impl SnapshotKey {
    pub(crate) fn new(
        camera: &Camera,
        viewport_px: [u32; 2],
        pixels_per_point: f32,
        basemap: BasemapSelection,
        active_source: SourceId,
        show_graticule: bool,
        scene_generation: u64,
    ) -> Self {
        let eye = camera.eye_ecef();
        let (_, up, _) = camera.basis();
        Self {
            eye_bits: eye.to_array().map(f64::to_bits),
            up_bits: up.to_array().map(f64::to_bits),
            width_px: viewport_px[0],
            height_px: viewport_px[1],
            ppp_bits: pixels_per_point.to_bits(),
            basemap,
            active_source,
            show_graticule,
            scene_generation,
        }
    }
}

/// Estimate physical viewport dimensions from `egui::Context` during `App::ui`
/// (where `egui_wgpu::ScreenDescriptor::size_in_pixels` is not exposed yet;
/// `GlobePaintCallback::prepare` uses the exact framebuffer size during
/// rendering).
pub(crate) fn viewport_pixels(ctx: &egui::Context, rect: egui::Rect) -> (u32, u32, f64) {
    let ppp = ctx.pixels_per_point();
    let screen_rect = ctx.content_rect();
    let screen_px = [
        (screen_rect.width() * ppp).round().max(1.0) as u32,
        (screen_rect.height() * ppp).round().max(1.0) as u32,
    ];
    let vp = egui::epaint::ViewportInPixels::from_points(&rect, ppp, screen_px);
    let width_px = vp.width_px.max(1) as u32;
    let height_px = vp.height_px.max(1) as u32;
    (
        width_px,
        height_px,
        f64::from(width_px) / f64::from(height_px),
    )
}

/// `egui_wgpu` paint callback bridging an immutable `Arc<DrawSnapshot>` into
/// the offscreen [`GlobeRenderer`].
pub(crate) struct GlobePaintCallback {
    pub(crate) rect: egui::Rect,
    pub(crate) snapshot: Arc<DrawSnapshot>,
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
        let Some(renderer) = callback_resources.get_mut::<GlobeRenderer>() else {
            return Vec::new();
        };

        let vp = egui::epaint::ViewportInPixels::from_points(
            &self.rect,
            screen_descriptor.pixels_per_point,
            screen_descriptor.size_in_pixels,
        );
        if vp.width_px <= 0 || vp.height_px <= 0 {
            renderer.sweep_stale_bind_groups();
            return Vec::new();
        }

        renderer.render_if_dirty(
            device,
            queue,
            egui_encoder,
            RenderParams {
                width: vp.width_px as u32,
                height: vp.height_px as u32,
                snapshot: &self.snapshot,
            },
        );
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
