//! Viewport pointer probing, camera drag/heading-rotation/zoom/keyboard input
//! handling, measurement vertex clicks, and viewport/compass geometry tracking
//! (`ViewportInteraction`).

use eframe::egui;
use veldmap_core::geo::{Camera, GeoPoint, ZOOM_PER_CLICK};

/// Logical scroll points mapped to one discrete globe zoom notch.
const SCROLL_PT_PER_STEP: f64 = 48.0;

/// Radians of clockwise heading rotation per logical drag point.
const ROTATE_RAD_PER_PT: f64 = 0.006;

/// Keyboard pan speed in viewport-height fractions per second.
const KEY_PAN_FRAC_PER_S: f64 = 0.65;

/// Keyboard zoom speed in scroll notches per second.
const KEY_ZOOM_STEPS_PER_S: f64 = 4.5;

/// Keyboard heading rotation speed in radians per second (`Q` / `E`).
const KEY_ROTATE_RAD_PER_S: f64 = 1.4;

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

/// Active primary-pointer interaction mode on the 3D globe viewport.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum ViewportToolMode {
    #[default]
    Navigate,
    Measure,
    DrawAoiBox,
}

/// Interactive geodetic event captured by [`ViewportInteraction::handle_viewport_input`].
#[derive(Clone, Copy, Debug)]
pub(super) enum ViewportEvent {
    MeasurePoint(GeoPoint),
    AoiBox { bbox: [f64; 4], finished: bool },
}

/// Tracks globe viewport geometry, HUD overlay bounds, hovered geodetic
/// cursor coordinates, and pointer/keyboard navigation gestures.
pub(super) struct ViewportInteraction {
    cursor_geo: Option<(f64, f64)>,
    aoi_drag_start: Option<(f64, f64)>,
    last_viewport_rect: Option<egui::Rect>,
    last_compass_rect: Option<egui::Rect>,
    last_scale_bar_rect: Option<egui::Rect>,
    last_attribution_rect: Option<egui::Rect>,
    last_colorbar_rect: Option<egui::Rect>,
    last_viewport_height_px: f64,
    last_viewport_aspect: f64,
}

impl Default for ViewportInteraction {
    fn default() -> Self {
        Self {
            cursor_geo: None,
            aoi_drag_start: None,
            last_viewport_rect: None,
            last_compass_rect: None,
            last_scale_bar_rect: None,
            last_attribution_rect: None,
            last_colorbar_rect: None,
            last_viewport_height_px: 800.0,
            last_viewport_aspect: 1.6,
        }
    }
}

impl ViewportInteraction {
    pub(super) fn cursor_geo(&self) -> Option<(f64, f64)> {
        self.cursor_geo
    }

    pub(super) fn viewport_rect(&self) -> Option<egui::Rect> {
        self.last_viewport_rect
    }

    pub(super) fn viewport_height_px(&self) -> f64 {
        self.last_viewport_height_px
    }

    pub(super) fn record_viewport(&mut self, rect: egui::Rect, height_px: u32, aspect: f64) {
        self.last_viewport_rect = Some(rect);
        self.last_viewport_height_px = f64::from(height_px);
        self.last_viewport_aspect = aspect;
    }

    pub(super) fn clear_viewport(&mut self) {
        self.last_viewport_rect = None;
        self.last_compass_rect = None;
        self.last_scale_bar_rect = None;
        self.last_attribution_rect = None;
        self.last_colorbar_rect = None;
    }

    pub(super) fn set_compass_rect(&mut self, rect: egui::Rect) {
        self.last_compass_rect = Some(rect);
    }

    pub(super) fn set_hud_rects(
        &mut self,
        scale_bar: Option<egui::Rect>,
        attribution: Option<egui::Rect>,
        colorbar: Option<egui::Rect>,
    ) {
        self.last_scale_bar_rect = scale_bar;
        self.last_attribution_rect = attribution;
        self.last_colorbar_rect = colorbar;
    }

    fn is_over_hud(&self, pos: egui::Pos2) -> bool {
        self.last_compass_rect.is_some_and(|r| r.contains(pos))
            || self.last_scale_bar_rect.is_some_and(|r| r.contains(pos))
            || self.last_attribution_rect.is_some_and(|r| r.contains(pos))
            || self.last_colorbar_rect.is_some_and(|r| r.contains(pos))
    }

    fn probe_pointer(
        &self,
        camera: &Camera,
        rect: egui::Rect,
        pos: egui::Pos2,
    ) -> Option<(f64, f64)> {
        let width_pt = f64::from(rect.width()).max(1.0);
        let height_pt = f64::from(rect.height()).max(1.0);
        let nx = f64::from(pos.x - rect.left()) / width_pt;
        let ny = f64::from(pos.y - rect.top()) / height_pt;
        camera.probe(nx, ny, self.last_viewport_aspect)
    }

    /// Probe the current pointer against the previous frame's viewport rect
    /// before laying out the bottom status bar so hover readouts stay in sync
    /// without scheduling a duplicate repaint on every mouse move.
    pub(super) fn update_cursor_probe(&mut self, ctx: &egui::Context, camera: &Camera) {
        if let Some(rect) = self.last_viewport_rect.filter(|r| r.is_positive()) {
            let hover_pos = ctx.input(|i| i.pointer.hover_pos());
            self.cursor_geo = hover_pos.and_then(|pos| {
                // Skip probing whenever a floating window, menu, or compass
                // `Area` sits above the globe `CentralPanel` (`LayerId::background`).
                let covered_by_ui_layer = ctx
                    .layer_id_at(pos)
                    .is_some_and(|layer| layer != egui::LayerId::background());
                (rect.contains(pos) && !covered_by_ui_layer && !self.is_over_hud(pos))
                    .then(|| self.probe_pointer(camera, rect, pos))
                    .flatten()
            });
        } else {
            self.cursor_geo = None;
        }
    }

    /// Apply keyboard navigation (`WASD`/Arrow pan, `+`/`=`/`-` zoom, `Q`/`E`
    /// heading rotation, `N` reset North) when no text field holds focus.
    /// Returns `true` if any continuous navigation key is held so the app can
    /// schedule another frame.
    pub(super) fn handle_keyboard_navigation(
        &self,
        ctx: &egui::Context,
        camera: &mut Camera,
        dt_s: f64,
    ) -> bool {
        if ctx.egui_wants_keyboard_input() {
            return false;
        }
        let mut active = false;
        ctx.input(|i| {
            if i.modifiers.ctrl || i.modifiers.command || i.modifiers.alt {
                return;
            }
            if i.key_pressed(egui::Key::N) && i.modifiers.is_none() {
                camera.reset_north();
            }

            let mut pan_x = 0.0_f64;
            let mut pan_y = 0.0_f64;
            if i.key_down(egui::Key::ArrowLeft) || i.key_down(egui::Key::A) {
                pan_x += 1.0;
            }
            if i.key_down(egui::Key::ArrowRight) || i.key_down(egui::Key::D) {
                pan_x -= 1.0;
            }
            if i.key_down(egui::Key::ArrowUp) || i.key_down(egui::Key::W) {
                pan_y += 1.0;
            }
            if i.key_down(egui::Key::ArrowDown) || i.key_down(egui::Key::S) {
                pan_y -= 1.0;
            }
            if pan_x != 0.0 || pan_y != 0.0 {
                let norm = pan_x.hypot(pan_y);
                let step = KEY_PAN_FRAC_PER_S * dt_s;
                camera.orbit((pan_x / norm) * step, (pan_y / norm) * step);
                active = true;
            }

            let mut zoom_dir = 0.0_f64;
            if i.key_down(egui::Key::Plus) || i.key_down(egui::Key::Equals) {
                zoom_dir += 1.0;
            }
            if i.key_down(egui::Key::Minus) {
                zoom_dir -= 1.0;
            }
            if zoom_dir != 0.0 {
                camera.zoom_at(zoom_dir * KEY_ZOOM_STEPS_PER_S * dt_s, 0.0, 0.0);
                active = true;
            }

            let mut rot_dir = 0.0_f64;
            if i.key_down(egui::Key::Q) {
                rot_dir -= 1.0;
            }
            if i.key_down(egui::Key::E) {
                rot_dir += 1.0;
            }
            if rot_dir != 0.0 {
                camera.rotate_heading(rot_dir * KEY_ROTATE_RAD_PER_S * dt_s);
                active = true;
            }
        });
        active
    }

    /// Approximate `[min_lon, min_lat, max_lon, max_lat]` covering the current
    /// visible globe viewport by probing a 5×5 screen grid, or `None` if fewer
    /// than two rays intersect the WGS84 ellipsoid.
    pub(super) fn viewport_bbox(&self, camera: &Camera) -> Option<[f64; 4]> {
        let mut min_lon = f64::INFINITY;
        let mut max_lon = f64::NEG_INFINITY;
        let mut min_lat = f64::INFINITY;
        let mut max_lat = f64::NEG_INFINITY;
        let mut hits = 0usize;
        for gy in 0..=4 {
            let ny = 0.08 + 0.84 * (f64::from(gy) / 4.0);
            for gx in 0..=4 {
                let nx = 0.08 + 0.84 * (f64::from(gx) / 4.0);
                if let Some((lat, lon)) = camera.probe(nx, ny, self.last_viewport_aspect) {
                    min_lon = min_lon.min(lon);
                    max_lon = max_lon.max(lon);
                    min_lat = min_lat.min(lat);
                    max_lat = max_lat.max(lat);
                    hits += 1;
                }
            }
        }
        if hits < 2 || (max_lon - min_lon) < 1e-6 || (max_lat - min_lat) < 1e-6 {
            return None;
        }
        // Guard against antimeridian wrap-around producing a bogus >180 deg span.
        if (max_lon - min_lon) > 180.0 {
            let (lat, lon) = camera.probe(0.5, 0.5, self.last_viewport_aspect)?;
            let half = 5.0;
            return Some([
                (lon - half).clamp(-180.0, 180.0),
                (lat - half).clamp(-89.5, 89.5),
                (lon + half).clamp(-180.0, 180.0),
                (lat + half).clamp(-89.5, 89.5),
            ]);
        }
        Some([min_lon, min_lat, max_lon, max_lat])
    }

    /// Apply primary-button orbit dragging, secondary / `Shift+Primary` manual
    /// heading rotation, scroll/pinch cursor-anchored zoom, and optional
    /// measurement click or AOI box drag capture to `camera`.
    pub(super) fn handle_viewport_input(
        &mut self,
        ctx: &egui::Context,
        camera: &mut Camera,
        response: &egui::Response,
        rect: egui::Rect,
        height_pt: f64,
        tool_mode: ViewportToolMode,
    ) -> Option<ViewportEvent> {
        let shift_held = ctx.input(|i| i.modifiers.shift);
        let rotating = response.dragged_by(egui::PointerButton::Secondary)
            || (shift_held && response.dragged_by(egui::PointerButton::Primary));

        let mut captured_event = None;
        if tool_mode == ViewportToolMode::DrawAoiBox && !shift_held {
            if response.hovered() || self.aoi_drag_start.is_some() {
                ctx.set_cursor_icon(egui::CursorIcon::Crosshair);
            }
            if response.drag_started_by(egui::PointerButton::Primary)
                && let Some(pos) = response.interact_pointer_pos()
                && !self.is_over_hud(pos)
            {
                self.aoi_drag_start = self.probe_pointer(camera, rect, pos);
            }
            if let Some((start_lat, start_lon)) = self.aoi_drag_start {
                if response.dragged_by(egui::PointerButton::Primary)
                    && let Some(pos) = response.interact_pointer_pos()
                    && let Some((cur_lat, cur_lon)) = self.probe_pointer(camera, rect, pos)
                {
                    let bbox = [
                        start_lon.min(cur_lon),
                        start_lat.min(cur_lat),
                        start_lon.max(cur_lon),
                        start_lat.max(cur_lat),
                    ];
                    if (bbox[2] - bbox[0]) > 1e-4 && (bbox[3] - bbox[1]) > 1e-4 {
                        captured_event = Some(ViewportEvent::AoiBox {
                            bbox,
                            finished: false,
                        });
                    }
                } else if response.drag_stopped_by(egui::PointerButton::Primary)
                    || !ctx.input(|i| i.pointer.primary_down())
                {
                    self.aoi_drag_start = None;
                    if let Some(pos) = response.interact_pointer_pos()
                        && let Some((end_lat, end_lon)) = self.probe_pointer(camera, rect, pos)
                    {
                        let bbox = [
                            start_lon.min(end_lon),
                            start_lat.min(end_lat),
                            start_lon.max(end_lon),
                            start_lat.max(end_lat),
                        ];
                        if (bbox[2] - bbox[0]) > 1e-4 && (bbox[3] - bbox[1]) > 1e-4 {
                            captured_event = Some(ViewportEvent::AoiBox {
                                bbox,
                                finished: true,
                            });
                        }
                    }
                }
            }
            if rotating {
                ctx.set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
                let delta = response.drag_delta();
                if delta.x != 0.0 {
                    camera.rotate_heading(f64::from(delta.x) * ROTATE_RAD_PER_PT);
                }
            }
        } else {
            self.aoi_drag_start = None;
            if rotating {
                ctx.set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
                let delta = response.drag_delta();
                if delta.x != 0.0 {
                    camera.rotate_heading(f64::from(delta.x) * ROTATE_RAD_PER_PT);
                }
            } else if response.dragged_by(egui::PointerButton::Primary) {
                ctx.set_cursor_icon(egui::CursorIcon::Grabbing);
                let delta = response.drag_delta();
                if delta != egui::Vec2::ZERO {
                    camera.orbit(
                        f64::from(delta.x) / height_pt,
                        f64::from(delta.y) / height_pt,
                    );
                }
            } else if response.hovered() {
                ctx.set_cursor_icon(if tool_mode == ViewportToolMode::Measure {
                    egui::CursorIcon::Crosshair
                } else {
                    egui::CursorIcon::Grab
                });
            }
        }

        // Scroll-wheel and trackpad pinch / Ctrl+scroll zoom holding ground under cursor.
        if response.hovered() {
            let (scroll_y, zoom_delta) = ctx.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta()));
            let steps = zoom_steps_from_input(scroll_y, zoom_delta);
            if steps.abs() > 0.0 {
                let (hold_x, hold_y) = match response.hover_pos() {
                    Some(pos) => (
                        f64::from(pos.x - rect.center().x) / height_pt,
                        f64::from(pos.y - rect.center().y) / height_pt,
                    ),
                    None => (0.0, 0.0),
                };
                camera.zoom_at(steps, hold_x, hold_y);
            }
        }

        if captured_event.is_some() {
            return captured_event;
        }

        if tool_mode == ViewportToolMode::Measure
            && response.clicked_by(egui::PointerButton::Primary)
            && !shift_held
            && let Some(pos) = response.interact_pointer_pos()
            && !self.is_over_hud(pos)
            && let Some((lat, lon)) = self.probe_pointer(camera, rect, pos)
        {
            return Some(ViewportEvent::MeasurePoint(GeoPoint::new(lat, lon)));
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zoom_steps_prefers_active_pinch_over_residual_scroll() {
        assert!((zoom_steps_from_input(48.0, 1.0) - 1.0).abs() < 1e-12);
        assert!((zoom_steps_from_input(-24.0, 1.0) + 0.5).abs() < 1e-12);
        let pinch = zoom_steps_from_input(96.0, ZOOM_PER_CLICK as f32);
        assert!((pinch - 1.0).abs() < 1e-6, "pinch={pinch}");
        assert_eq!(zoom_steps_from_input(f32::NAN, 1.0), 0.0);
        assert_eq!(zoom_steps_from_input(0.0, f32::NAN), 0.0);
    }
}
