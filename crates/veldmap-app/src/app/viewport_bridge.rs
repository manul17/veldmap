//! `egui_wgpu` viewport bridge ([`GlobePaintCallback`] and logical-to-physical
//! viewport pixel mapping) using the hoisted [`TileBindGroupCache`] inside
//! `egui_wgpu::CallbackResources`.

use std::sync::{Arc, Mutex};

use eframe::egui;
use eframe::egui_wgpu::{CallbackResources, CallbackTrait, RenderState, ScreenDescriptor};
use veldmap_core::sync::{Wakeup, lock_unpoisoned};

use crate::render::{DrawSnapshot, GlobeRenderer, RenderParams, TileBindGroupCache};

use super::GpuFaultState;

/// Install `wgpu` uncaptured-error / device-lost hooks and register the shared
/// [`TileBindGroupCache`] and [`GlobeRenderer`] inside
/// `egui_wgpu::CallbackResources`.
pub(super) fn install_wgpu_resources(
    render_state: &RenderState,
    gpu_faults: &Arc<GpuFaultState>,
    wakeup: &Wakeup,
) {
    {
        let faults = Arc::clone(gpu_faults);
        let wake = Arc::clone(wakeup);
        render_state
            .device
            .on_uncaptured_error(Arc::new(move |err| {
                let msg = err.to_string();
                tracing::error!(error = %msg, "Uncaptured wgpu device error");
                if faults.record_uncaptured_error(msg) {
                    (wake)();
                }
            }));
    }
    {
        let faults = Arc::clone(gpu_faults);
        let wake = Arc::clone(wakeup);
        render_state
            .device
            .set_device_lost_callback(move |reason, message| {
                // Normal shutdown destroys the wgpu device; ignore expected teardown.
                if reason == wgpu::DeviceLostReason::Destroyed {
                    tracing::debug!(?reason, %message, "wgpu device destroyed during teardown");
                    return;
                }
                tracing::error!(?reason, %message, "wgpu device lost");
                if faults.record_device_lost(message) {
                    (wake)();
                }
            });
    }

    let bind_groups = TileBindGroupCache::new(&render_state.device);
    let renderer = GlobeRenderer::new(
        &render_state.device,
        render_state.target_format,
        &bind_groups,
    );
    let mut wgpu_renderer = render_state.renderer.write();
    wgpu_renderer
        .callback_resources
        .insert(Arc::new(Mutex::new(bind_groups)));
    wgpu_renderer.callback_resources.insert(renderer);
}

/// Estimate physical viewport dimensions from `egui::Context` during `App::ui`
/// (where `egui_wgpu::ScreenDescriptor::size_in_pixels` is not exposed yet;
/// `GlobePaintCallback::prepare` uses the exact framebuffer size during
/// rendering).
pub(super) fn viewport_pixels(ctx: &egui::Context, rect: egui::Rect) -> (u32, u32, f64) {
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
/// the offscreen [`GlobeRenderer`] and hoisted [`TileBindGroupCache`].
pub(super) struct GlobePaintCallback {
    pub(super) rect: egui::Rect,
    pub(super) snapshot: Arc<DrawSnapshot>,
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
        let vp = egui::epaint::ViewportInPixels::from_points(
            &self.rect,
            screen_descriptor.pixels_per_point,
            screen_descriptor.size_in_pixels,
        );
        let Some(bind_groups_arc) = callback_resources
            .get::<Arc<Mutex<TileBindGroupCache>>>()
            .cloned()
        else {
            debug_assert!(
                false,
                "hoisted Arc<Mutex<TileBindGroupCache>> missing from CallbackResources"
            );
            return Vec::new();
        };

        if vp.width_px <= 0 || vp.height_px <= 0 {
            lock_unpoisoned(&bind_groups_arc).sweep_stale();
            return Vec::new();
        }

        if let Some(renderer) = callback_resources.get_mut::<GlobeRenderer>() {
            let mut bind_groups = lock_unpoisoned(&bind_groups_arc);
            renderer.render_if_dirty(
                device,
                queue,
                egui_encoder,
                RenderParams {
                    width: vp.width_px as u32,
                    height: vp.height_px as u32,
                    snapshot: &self.snapshot,
                },
                &mut bind_groups,
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
