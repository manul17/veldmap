//! Main `eframe::App` state, globe input handling, and GIS HUD layout.

use std::sync::Arc;

use eframe::egui;
use eframe::egui_wgpu::{self, CallbackResources, CallbackTrait, ScreenDescriptor};
use thiserror::Error;
use veldmap_core::geo::camera::{Camera, ZOOM_PER_CLICK};
use veldmap_core::geo::coords::parse_location;

use crate::hud;
use crate::render::{GlobeRenderer, RenderParams};

/// Logical scroll points mapped to one discrete globe zoom notch.
const SCROLL_PT_PER_STEP: f64 = 48.0;

/// Compute discrete zoom notches from `egui` scroll and pinch/`Ctrl+scroll`
/// deltas. When an active pinch/zoom gesture is present, it takes precedence
/// over decaying `smooth_scroll_delta` momentum so the two do not double-count.
fn zoom_steps_from_input(scroll_y: f32, zoom_delta: f32) -> f64 {
    if zoom_delta.is_finite() && zoom_delta > 0.0 && (zoom_delta - 1.0).abs() > 1e-6 {
        f64::from(zoom_delta).ln() / ZOOM_PER_CLICK.ln()
    } else if scroll_y.is_finite() {
        f64::from(scroll_y) / SCROLL_PT_PER_STEP
    } else {
        0.0
    }
}

/// `egui_wgpu` paint callback bridging the offscreen [`GlobeRenderer`] into the
/// viewport panel.
struct GlobePaintCallback {
    rect: egui::Rect,
    camera: Camera,
    show_graticule: bool,
    generation: u64,
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

#[derive(Debug, Error)]
pub enum AppError {
    #[error("eframe did not provide a wgpu render state")]
    MissingWgpu,
}

pub struct VeldmapApp {
    adapter_label: String,
    camera: Camera,
    show_graticule: bool,
    /// Reserved for non-camera scene mutations (e.g. tile uploads/evictions in
    /// Slice 2A); camera pose and `show_graticule` are tracked directly in
    /// `GlobeRenderer`'s `RenderKey`.
    scene_generation: u64,
    cursor_geo: Option<(f64, f64)>,
    last_viewport_rect: Option<egui::Rect>,
    last_compass_rect: Option<egui::Rect>,
    last_viewport_height_px: f64,
    goto_input: String,
    goto_error: Option<String>,
}

impl VeldmapApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Result<Self, AppError> {
        let render_state = cc.wgpu_render_state.as_ref().ok_or(AppError::MissingWgpu)?;
        let info = render_state.adapter.get_info();
        let adapter_label = format!("{} ({:?})", info.name, info.backend);
        tracing::info!(
            adapter = %info.name,
            backend = ?info.backend,
            device_type = ?info.device_type,
            target_format = ?render_state.target_format,
            "Initialized wgpu render state"
        );

        render_state.device.on_uncaptured_error(Arc::new(|err| {
            tracing::error!(error = %err, "Uncaptured wgpu device error");
        }));
        render_state
            .device
            .set_device_lost_callback(|reason, message| {
                tracing::error!(?reason, %message, "wgpu device lost");
            });

        let renderer = GlobeRenderer::new(&render_state.device, render_state.target_format);
        render_state
            .renderer
            .write()
            .callback_resources
            .insert(renderer);

        Ok(Self {
            adapter_label,
            camera: Camera::default(),
            show_graticule: true,
            scene_generation: 1,
            cursor_geo: None,
            last_viewport_rect: None,
            last_compass_rect: None,
            last_viewport_height_px: 800.0,
            goto_input: String::new(),
            goto_error: None,
        })
    }

    fn trigger_goto(&mut self) {
        match parse_location(&self.goto_input) {
            Ok(loc) => {
                tracing::info!(
                    lat = loc.lat_deg,
                    lon = loc.lon_deg,
                    radius_deg = loc.radius_deg,
                    kind = ?loc.kind,
                    "Flying camera to parsed location"
                );
                self.camera.focus(loc.lat_deg, loc.lon_deg, loc.radius_deg);
                self.goto_error = None;
            }
            Err(err) => {
                self.goto_error = Some(err.to_string());
            }
        }
    }

    fn viewport_pixels(ctx: &egui::Context, rect: egui::Rect) -> (f64, f64, f64) {
        let ppp = ctx.pixels_per_point();
        let screen_rect = ctx.content_rect();
        let screen_px = [
            (screen_rect.width() * ppp).round().max(1.0) as u32,
            (screen_rect.height() * ppp).round().max(1.0) as u32,
        ];
        let vp = egui::epaint::ViewportInPixels::from_points(&rect, ppp, screen_px);
        let width_px = f64::from(vp.width_px.max(1));
        let height_px = f64::from(vp.height_px.max(1));
        (width_px, height_px, width_px / height_px)
    }

    fn probe_pointer(
        &self,
        ctx: &egui::Context,
        rect: egui::Rect,
        pos: egui::Pos2,
    ) -> Option<(f64, f64)> {
        let (_, _, aspect) = Self::viewport_pixels(ctx, rect);
        let width_pt = f64::from(rect.width()).max(1.0);
        let height_pt = f64::from(rect.height()).max(1.0);
        let nx = f64::from(pos.x - rect.left()) / width_pt;
        let ny = f64::from(pos.y - rect.top()) / height_pt;
        self.camera.probe(nx, ny, aspect)
    }
}

impl eframe::App for VeldmapApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();

        // Advance active quaternion camera flights (cap dt to 50 ms so a
        // frame hitch never jumps abruptly through a 0.6 s flight).
        let dt_s = f64::from(ctx.input(|i| i.stable_dt)).clamp(0.0, 0.05);
        self.camera.advance(dt_s);

        // Global shortcut `N` (without any modifiers) resets North when no
        // text field owns keyboard focus.
        let reset_north_key = !ctx.egui_wants_keyboard_input()
            && ctx.input(|i| i.key_pressed(egui::Key::N) && i.modifiers.is_none());
        if reset_north_key {
            self.camera.reset_north();
        }

        // Top toolbar: title, Go-To DD/DMS/MGRS input, graticule toggle, GPU badge.
        egui::Panel::top("top_toolbar").show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.strong(veldmap_core::APP_NAME);
                ui.separator();

                ui.label("Go to:");
                let text_edit = egui::TextEdit::singleline(&mut self.goto_input)
                    .desired_width(180.0)
                    .hint_text("DD, DMS, or MGRS (e.g. 47.23, 39.70 / 37UDQ)");
                let edit_resp = ui.add(text_edit);
                if edit_resp.changed() {
                    self.goto_error = None;
                }
                let enter_pressed =
                    edit_resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                if ui.button("Fly").clicked() || enter_pressed {
                    self.trigger_goto();
                }

                ui.separator();
                ui.checkbox(&mut self.show_graticule, "Graticule");

                if let Some(err) = &self.goto_error {
                    ui.separator();
                    ui.add(
                        egui::Label::new(
                            egui::RichText::new(err)
                                .color(egui::Color32::from_rgb(235, 110, 95))
                                .small(),
                        )
                        .truncate(),
                    )
                    .on_hover_text(err);
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.add(
                        egui::Label::new(egui::RichText::new(&self.adapter_label).weak())
                            .truncate(),
                    )
                    .on_hover_text(&self.adapter_label);
                });
            });
        });

        // Probe the current pointer against the previous frame's viewport rect
        // before laying out the bottom status bar so hover readouts stay in
        // sync without scheduling a duplicate repaint on every mouse move.
        if let Some(rect) = self.last_viewport_rect.filter(|r| r.is_positive()) {
            let hover_pos = ctx.input(|i| i.pointer.hover_pos());
            self.cursor_geo = hover_pos.and_then(|pos| {
                let over_compass = self.last_compass_rect.is_some_and(|c| c.contains(pos));
                (rect.contains(pos) && !over_compass)
                    .then(|| self.probe_pointer(&ctx, rect, pos))
                    .flatten()
            });
        }

        let camera_before = self.camera;
        let rect_before = self.last_viewport_rect;

        // Bottom GIS status bar: wraps cleanly down to the 640 px minimum width.
        egui::Panel::bottom("gis_status_bar").show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                let cursor_text = match self.cursor_geo {
                    Some((lat, lon)) => format!("Cursor: {}", hud::format_lat_lon(lat, lon)),
                    None => "Cursor: Off globe".to_owned(),
                };
                ui.monospace(cursor_text);
                ui.separator();

                let (clat, clon) = self.camera.center();
                ui.monospace(format!("Center: {}", hud::format_lat_lon(clat, clon)));
                ui.separator();

                ui.monospace(format!(
                    "Alt: {}",
                    hud::format_distance_m(self.camera.height_m())
                ));
                ui.separator();

                let mpp = self.camera.metres_per_pixel(self.last_viewport_height_px);
                ui.monospace(format!("GSD: {}", hud::format_resolution(mpp)));
            });
        });

        // Central 3D WGS84 globe viewport.
        let mut viewport_rect = None;
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ui, |ui| {
                let available = ui.available_size();
                let (response, painter) =
                    ui.allocate_painter(available, egui::Sense::click_and_drag());
                let rect = response.rect;

                if rect.is_positive() {
                    viewport_rect = Some(rect);
                    self.last_viewport_rect = Some(rect);
                    let (_, height_px, _) = Self::viewport_pixels(&ctx, rect);
                    let height_pt = f64::from(rect.height()).max(1.0);
                    self.last_viewport_height_px = height_px;

                    if response.dragged_by(egui::PointerButton::Primary) {
                        ctx.set_cursor_icon(egui::CursorIcon::Grabbing);
                        let delta = response.drag_delta();
                        if delta != egui::Vec2::ZERO {
                            self.camera.orbit(
                                f64::from(delta.x) / height_pt,
                                f64::from(delta.y) / height_pt,
                            );
                        }
                    } else if response.hovered() {
                        ctx.set_cursor_icon(egui::CursorIcon::Grab);
                    }

                    // Scroll-wheel and trackpad pinch / Ctrl+scroll zoom holding ground under cursor.
                    if response.hovered() {
                        let (scroll_y, zoom_delta) =
                            ctx.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta()));
                        let steps = zoom_steps_from_input(scroll_y, zoom_delta);
                        if steps.abs() > 0.0 {
                            let (hold_x, hold_y) = match response.hover_pos() {
                                Some(pos) => (
                                    f64::from(pos.x - rect.center().x) / height_pt,
                                    f64::from(pos.y - rect.center().y) / height_pt,
                                ),
                                None => (0.0, 0.0),
                            };
                            self.camera.zoom_at(steps, hold_x, hold_y);
                        }
                    }

                    // Register offscreen globe + 1:1 nearest composite paint callback.
                    painter.add(egui_wgpu::Callback::new_paint_callback(
                        rect,
                        GlobePaintCallback {
                            rect,
                            camera: self.camera,
                            show_graticule: self.show_graticule,
                            generation: self.scene_generation,
                        },
                    ));

                    // Floating dynamic metric scale bar.
                    let m_per_pt = self.camera.metres_per_pixel(height_pt);
                    hud::paint_scale_bar(&painter, rect, m_per_pt);
                }
            });

        // Floating North compass in its own foreground Area (no pointer bleed).
        if let Some(rect) = viewport_rect {
            let (clicked, compass_rect) =
                hud::show_north_compass(&ctx, rect, self.camera.heading_deg());
            self.last_compass_rect = Some(compass_rect);
            if clicked {
                self.camera.reset_north();
            }
        }

        // Request a follow-up repaint only for active flights, camera moves, or
        // viewport resizes (never on plain mouse hover).
        if self.camera.is_flying()
            || self.camera != camera_before
            || self.last_viewport_rect != rect_before
        {
            ctx.request_repaint();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zoom_steps_prefers_active_pinch_over_residual_scroll() {
        assert!((zoom_steps_from_input(48.0, 1.0) - 1.0).abs() < 1e-12);
        assert!((zoom_steps_from_input(-24.0, 1.0) + 0.5).abs() < 1e-12);
        // Active pinch (zoom_delta = ZOOM_PER_CLICK) yields 1 step without adding scroll_y.
        let pinch = zoom_steps_from_input(96.0, ZOOM_PER_CLICK as f32);
        assert!((pinch - 1.0).abs() < 1e-6, "pinch={pinch}");
        // Non-finite inputs never produce NaN steps.
        assert_eq!(zoom_steps_from_input(f32::NAN, 1.0), 0.0);
        assert_eq!(zoom_steps_from_input(0.0, f32::NAN), 0.0);
    }
}
