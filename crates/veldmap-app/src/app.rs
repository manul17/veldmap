//! Main `eframe::App` state, globe input handling, and GIS HUD layout.

use std::sync::Arc;

use eframe::{egui, egui_wgpu};
use thiserror::Error;
use veldmap_core::geo::camera::Camera;
use veldmap_core::geo::coords::parse_location;

use crate::hud;
use crate::render::{GlobePaintCallback, GlobeRenderer};

#[derive(Debug, Error)]
pub enum AppError {
    #[error("eframe did not provide a wgpu render state")]
    MissingWgpu,
}

pub struct VeldmapApp {
    adapter_label: String,
    camera: Camera,
    show_graticule: bool,
    scene_generation: u64,
    cursor_geo: Option<(f64, f64)>,
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
            last_viewport_height_px: 800.0,
            goto_input: String::new(),
            goto_error: None,
        })
    }

    fn mark_dirty(&mut self) {
        self.scene_generation = self.scene_generation.wrapping_add(1);
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
                self.mark_dirty();
            }
            Err(err) => {
                self.goto_error = Some(err.to_string());
            }
        }
    }
}

impl eframe::App for VeldmapApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();

        // Advance active quaternion camera flights (cap dt to 50 ms so a
        // frame hitch never jumps abruptly through a 0.6 s flight).
        let dt_s = f64::from(ctx.input(|i| i.unstable_dt)).clamp(0.0, 0.05);
        if self.camera.advance(dt_s) {
            self.mark_dirty();
        }

        // Global shortcut `N` (without modifiers) resets North when no text
        // field owns keyboard focus.
        let reset_north_key = !ctx.egui_wants_keyboard_input()
            && ctx.input(|i| {
                i.key_pressed(egui::Key::N)
                    && !i.modifiers.command
                    && !i.modifiers.ctrl
                    && !i.modifiers.alt
            });
        if reset_north_key && self.camera.reset_north() {
            self.mark_dirty();
        }

        // Top toolbar: title, Go-To DD/DMS/MGRS input, graticule toggle, GPU badge.
        egui::Panel::top("top_toolbar").show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.strong(veldmap_core::APP_NAME);
                ui.separator();

                ui.label("Go to:");
                let text_edit = egui::TextEdit::singleline(&mut self.goto_input)
                    .desired_width(220.0)
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
                if let Some(err) = &self.goto_error {
                    ui.colored_label(egui::Color32::from_rgb(235, 110, 95), "Invalid")
                        .on_hover_text(err);
                }

                ui.separator();
                if ui.checkbox(&mut self.show_graticule, "Graticule").changed() {
                    self.mark_dirty();
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.weak(&self.adapter_label);
                });
            });
        });

        // Snapshot status-bar inputs before CentralPanel processes this frame's
        // pointer/scroll/resize updates so we can request one cheap UI-only
        // follow-up repaint whenever a displayed status value changes.
        let status_before = (
            self.cursor_geo,
            self.camera.center(),
            self.camera.height_m().to_bits(),
            self.last_viewport_height_px.to_bits(),
        );

        // Bottom GIS status bar: Cursor Lat/Lon, Center Lat/Lon, Altitude, m/px.
        egui::Panel::bottom("gis_status_bar").show(ui, |ui| {
            ui.horizontal(|ui| {
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
                    let ppp = f64::from(ctx.pixels_per_point());
                    let width_pt = f64::from(rect.width()).max(1.0);
                    let height_pt = f64::from(rect.height()).max(1.0);
                    let aspect = width_pt / height_pt;
                    self.last_viewport_height_px = (height_pt * ppp).max(1.0);

                    // Left-drag orbit.
                    if response.dragged_by(egui::PointerButton::Primary) {
                        let delta = response.drag_delta();
                        if delta != egui::Vec2::ZERO {
                            self.camera.orbit(
                                f64::from(delta.x) / height_pt,
                                f64::from(delta.y) / height_pt,
                            );
                            self.mark_dirty();
                        }
                    }

                    // Scroll-wheel zoom holding ground under cursor.
                    if response.hovered() {
                        let (scroll_y, hover_pos) =
                            ctx.input(|i| (i.smooth_scroll_delta.y, i.pointer.hover_pos()));
                        if scroll_y.abs() > 0.0 {
                            let steps = f64::from(scroll_y) / 48.0;
                            let (hold_x, hold_y) = match hover_pos {
                                Some(pos) => (
                                    f64::from(pos.x - rect.center().x) / height_pt,
                                    f64::from(pos.y - rect.center().y) / height_pt,
                                ),
                                None => (0.0, 0.0),
                            };
                            self.camera.zoom_at(steps, hold_x, hold_y);
                            self.mark_dirty();
                        }
                    }

                    // Cursor geographic probe.
                    self.cursor_geo = response.hover_pos().and_then(|pos| {
                        let nx = f64::from(pos.x - rect.left()) / width_pt;
                        let ny = f64::from(pos.y - rect.top()) / height_pt;
                        self.camera.probe(nx, ny, aspect)
                    });

                    // Register offscreen globe + 1:1 nearest composite paint callback.
                    painter.add(egui_wgpu::Callback::new_paint_callback(
                        rect,
                        GlobePaintCallback {
                            rect,
                            clip_rect: ui.clip_rect(),
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
        if let Some(rect) = viewport_rect
            && hud::show_north_compass(&ctx, rect, self.camera.heading_deg())
            && self.camera.reset_north()
        {
            self.mark_dirty();
        }

        let status_after = (
            self.cursor_geo,
            self.camera.center(),
            self.camera.height_m().to_bits(),
            self.last_viewport_height_px.to_bits(),
        );
        if self.camera.is_flying() || status_after != status_before {
            ctx.request_repaint();
        }
    }
}
