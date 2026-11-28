//! `egui_wgpu` viewport bridge ([`GlobePaintCallback`], [`GpuFaultState`], and
//! logical-to-physical viewport pixel mapping) using the hoisted
//! [`TileBindGroupCache`] inside `egui_wgpu::CallbackResources`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use eframe::egui;
use eframe::egui_wgpu::{CallbackResources, CallbackTrait, RenderState, ScreenDescriptor};
use veldmap_core::sync::{Wakeup, lock_unpoisoned};

use crate::render::{
    DrawSnapshot, GlobeRenderer, OffscreenReadback, RenderParams, SwipeRenderState,
    TileBindGroupCache,
};

/// Shared synchronization slot bridging a single-frame offscreen PNG readback
/// from `GlobePaintCallback::prepare` back to `VeldmapApp` without ever
/// blocking the UI thread.
pub(super) struct PngReadbackBridge {
    /// Set by `GlobePaintCallback::prepare` once `copy_texture_to_buffer` is
    /// submitted and `map_async` has been initiated.
    pub(super) readback: Option<OffscreenReadback>,
    /// Set asynchronously by the `wgpu::BufferSlice::map_async` callback.
    pub(super) mapped: Option<Result<(), String>>,
    /// Set if the offscreen target was unavailable during `prepare`.
    pub(super) capture_error: Option<String>,
    pub(super) wakeup: Wakeup,
}

impl PngReadbackBridge {
    pub(super) fn new(wakeup: Wakeup) -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Self {
            readback: None,
            mapped: None,
            capture_error: None,
            wakeup,
        }))
    }
}

#[derive(Default)]
struct GpuFaultInner {
    device_lost: Option<String>,
    uncaptured_error: Option<String>,
}

/// Lock-free fast-path holder for asynchronous `wgpu` device faults surfaced in
/// the UI banner without per-frame mutex locks or error-frame repaint storms.
#[derive(Default)]
pub(super) struct GpuFaultState {
    has_fault: AtomicBool,
    inner: Mutex<GpuFaultInner>,
}

impl GpuFaultState {
    fn record_device_lost(&self, msg: String) -> bool {
        let mut guard = lock_unpoisoned(&self.inner);
        let changed = guard.device_lost.as_deref() != Some(msg.as_str());
        guard.device_lost = Some(msg);
        self.has_fault.store(true, Ordering::Release);
        changed
    }

    fn record_uncaptured_error(&self, msg: String) -> bool {
        let mut guard = lock_unpoisoned(&self.inner);
        if guard.device_lost.is_some() {
            return false;
        }
        if guard.uncaptured_error.as_deref() == Some(msg.as_str()) {
            return false;
        }
        guard.uncaptured_error = Some(msg);
        self.has_fault.store(true, Ordering::Release);
        true
    }

    pub(super) fn summary(&self) -> Option<(String, bool)> {
        if !self.has_fault.load(Ordering::Acquire) {
            return None;
        }
        let guard = lock_unpoisoned(&self.inner);
        if let Some(lost) = &guard.device_lost {
            return Some((
                format!("GPU device lost (restart recommended): {lost}"),
                true,
            ));
        }
        guard
            .uncaptured_error
            .as_ref()
            .map(|err| (format!("GPU error: {err}"), false))
    }

    pub(super) fn dismiss(&self) {
        if !self.has_fault.load(Ordering::Acquire) {
            return;
        }
        let mut guard = lock_unpoisoned(&self.inner);
        guard.device_lost = None;
        guard.uncaptured_error = None;
        self.has_fault.store(false, Ordering::Release);
    }
}

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

/// Release offscreen/scratch viewport GPU targets and sweep stale bind groups
/// when the globe viewport collapses or minimizes.
pub(super) fn release_viewport_resources(frame: &eframe::Frame) {
    let Some(render_state) = frame.wgpu_render_state() else {
        return;
    };
    let bind_groups = {
        let mut res = render_state.renderer.write();
        if let Some(renderer) = res.callback_resources.get_mut::<GlobeRenderer>() {
            renderer.release_viewport_targets();
        }
        res.callback_resources
            .get::<Arc<Mutex<TileBindGroupCache>>>()
            .cloned()
    };
    if let Some(bg) = bind_groups {
        lock_unpoisoned(&bg).sweep_stale();
    }
}

/// Physical viewport origin and dimensions computed via
/// `ViewportInPixels::from_points` so `egui::Painter` overlays (such as the
/// split-screen swipe divider) align to the exact physical scissor seam without
/// fractional-HiDPI 1-px drift.
#[derive(Clone, Copy, Debug)]
pub(super) struct ViewportPixels {
    pub(super) left_px: i32,
    pub(super) width_px: u32,
    pub(super) height_px: u32,
    pub(super) aspect: f64,
}

impl ViewportPixels {
    /// Compute the exact HiDPI-aligned logical X coordinate and physical pixel
    /// offset `(div_x_pt, split_x_px)` of the split-screen swipe curtain.
    pub(super) fn swipe_divider_x(self, split_t: f32, ppp: f32) -> (f32, u32) {
        let split_x = crate::render::snap_split_x(split_t, self.width_px);
        let div_x = (self.left_px as f32 + split_x as f32) / ppp.max(0.5);
        (div_x, split_x)
    }
}

/// Estimate physical viewport origin and dimensions from `egui::Context` during
/// `App::ui` (where `egui_wgpu::ScreenDescriptor::size_in_pixels` is not
/// exposed yet; `GlobePaintCallback::prepare` uses the exact framebuffer size
/// during rendering).
pub(super) fn viewport_pixels(ctx: &egui::Context, rect: egui::Rect) -> ViewportPixels {
    let ppp = ctx.pixels_per_point();
    let screen_rect = ctx.content_rect();
    let screen_px = [
        (screen_rect.width() * ppp).round().max(1.0) as u32,
        (screen_rect.height() * ppp).round().max(1.0) as u32,
    ];
    let vp = egui::epaint::ViewportInPixels::from_points(&rect, ppp, screen_px);
    let width_px = vp.width_px.max(1) as u32;
    let height_px = vp.height_px.max(1) as u32;
    ViewportPixels {
        left_px: vp.left_px,
        width_px,
        height_px,
        aspect: f64::from(width_px) / f64::from(height_px),
    }
}

/// `egui_wgpu` paint callback bridging an immutable `Arc<DrawSnapshot>`,
/// optional split-screen [`SwipeRenderState`], and optional clean-frame PNG
/// readback request into the offscreen [`GlobeRenderer`] and hoisted
/// [`TileBindGroupCache`].
pub(super) struct GlobePaintCallback {
    pub(super) rect: egui::Rect,
    pub(super) snapshot: Arc<DrawSnapshot>,
    pub(super) swipe: Option<SwipeRenderState>,
    pub(super) png_readback: Option<Arc<Mutex<PngReadbackBridge>>>,
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
        let fail_readback = |bridge: &Arc<Mutex<PngReadbackBridge>>, msg: &str| {
            let mut guard = lock_unpoisoned(bridge);
            guard.capture_error = Some(msg.to_owned());
            let wake = Arc::clone(&guard.wakeup);
            drop(guard);
            (wake)();
        };

        let Some(bind_groups_arc) = callback_resources
            .get::<Arc<Mutex<TileBindGroupCache>>>()
            .cloned()
        else {
            debug_assert!(
                false,
                "hoisted Arc<Mutex<TileBindGroupCache>> missing from CallbackResources"
            );
            if let Some(bridge) = &self.png_readback {
                fail_readback(bridge, "GPU tile bind-group cache is unavailable.");
            }
            return Vec::new();
        };

        if vp.width_px <= 0 || vp.height_px <= 0 {
            if let Some(renderer) = callback_resources.get_mut::<GlobeRenderer>() {
                renderer.release_viewport_targets();
            }
            lock_unpoisoned(&bind_groups_arc).sweep_stale();
            if let Some(bridge) = &self.png_readback {
                fail_readback(bridge, "Viewport is collapsed; cannot capture screenshot.");
            }
            return Vec::new();
        }

        if let Some(renderer) = callback_resources.get_mut::<GlobeRenderer>() {
            let mut bind_groups = lock_unpoisoned(&bind_groups_arc);
            let params = RenderParams {
                width: vp.width_px as u32,
                height: vp.height_px as u32,
                snapshot: &self.snapshot,
                swipe: self.swipe,
            };

            if let Some(bridge) = &self.png_readback {
                // Use a dedicated command encoder submitted immediately so
                // `map_async` can begin right away on BOTH dirty and clean
                // frames (`is_clean_for == true`) without blocking the UI thread.
                let mut capture_encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("globe offscreen PNG capture encoder"),
                    });
                renderer.render_if_dirty(
                    device,
                    queue,
                    &mut capture_encoder,
                    params,
                    &mut bind_groups,
                );
                let readback = renderer.encode_offscreen_readback(device, &mut capture_encoder);
                queue.submit(Some(capture_encoder.finish()));

                if let Some(rb) = readback {
                    let cb_bridge = Arc::clone(bridge);
                    rb.map_async(move |res| {
                        let wake = {
                            let mut guard = lock_unpoisoned(&cb_bridge);
                            guard.mapped = Some(res.map_err(|e| e.to_string()));
                            Arc::clone(&guard.wakeup)
                        };
                        (wake)();
                    });
                    let mut guard = lock_unpoisoned(bridge);
                    guard.readback = Some(rb);
                    let wake = Arc::clone(&guard.wakeup);
                    drop(guard);
                    (wake)();
                } else {
                    fail_readback(
                        bridge,
                        "Offscreen color target is unavailable for PNG readback.",
                    );
                }
            } else {
                renderer.render_if_dirty(device, queue, egui_encoder, params, &mut bind_groups);
            }
        } else if let Some(bridge) = &self.png_readback {
            fail_readback(bridge, "Globe renderer is unavailable for PNG readback.");
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_fault_state_deduplicates_errors_preserves_device_lost_and_dismisses() {
        let faults = GpuFaultState::default();
        assert!(faults.summary().is_none());

        // Identical repeated uncaptured errors return false (no repaint storm).
        assert!(faults.record_uncaptured_error("validation fail".to_owned()));
        assert!(!faults.record_uncaptured_error("validation fail".to_owned()));
        let (msg, is_lost) = faults.summary().expect("uncaptured summary");
        assert!(!is_lost);
        assert!(msg.contains("validation fail"));

        // Device lost overrides uncaptured errors and cannot be overwritten by later errors.
        assert!(faults.record_device_lost("driver reset".to_owned()));
        assert!(!faults.record_uncaptured_error("later error".to_owned()));
        let (lost_msg, is_lost) = faults.summary().expect("device lost summary");
        assert!(is_lost);
        assert!(lost_msg.contains("driver reset"));

        faults.dismiss();
        assert!(faults.summary().is_none());
    }
}
