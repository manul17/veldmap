//! Viewport pointer probing, camera drag/zoom input handling, and
//! viewport/compass layout geometry tracking (`ViewportInteraction`).

use eframe::egui;
use veldmap_core::geo::Camera;
use veldmap_core::geo::camera::ZOOM_PER_CLICK;

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

/// Tracks globe viewport geometry, foreground compass bounds, hovered geodetic
/// cursor coordinates, and pointer orbit/zoom gestures.
pub(crate) struct ViewportInteraction {
    cursor_geo: Option<(f64, f64)>,
    last_viewport_rect: Option<egui::Rect>,
    last_compass_rect: Option<egui::Rect>,
    last_viewport_height_px: f64,
    last_viewport_aspect: f64,
}

impl Default for ViewportInteraction {
    fn default() -> Self {
        Self {
            cursor_geo: None,
            last_viewport_rect: None,
            last_compass_rect: None,
            last_viewport_height_px: 800.0,
            last_viewport_aspect: 1.6,
        }
    }
}

impl ViewportInteraction {
    pub(crate) fn cursor_geo(&self) -> Option<(f64, f64)> {
        self.cursor_geo
    }

    pub(crate) fn viewport_rect(&self) -> Option<egui::Rect> {
        self.last_viewport_rect
    }

    pub(crate) fn viewport_height_px(&self) -> f64 {
        self.last_viewport_height_px
    }

    pub(crate) fn record_viewport(&mut self, rect: egui::Rect, height_px: u32, aspect: f64) {
        self.last_viewport_rect = Some(rect);
        self.last_viewport_height_px = f64::from(height_px);
        self.last_viewport_aspect = aspect;
    }

    pub(crate) fn clear_viewport(&mut self) {
        self.last_viewport_rect = None;
        self.last_compass_rect = None;
    }

    pub(crate) fn set_compass_rect(&mut self, rect: egui::Rect) {
        self.last_compass_rect = Some(rect);
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
    pub(crate) fn update_cursor_probe(&mut self, ctx: &egui::Context, camera: &Camera) {
        if let Some(rect) = self.last_viewport_rect.filter(|r| r.is_positive()) {
            let hover_pos = ctx.input(|i| i.pointer.hover_pos());
            self.cursor_geo = hover_pos.and_then(|pos| {
                let over_compass = self.last_compass_rect.is_some_and(|c| c.contains(pos));
                (rect.contains(pos) && !over_compass)
                    .then(|| self.probe_pointer(camera, rect, pos))
                    .flatten()
            });
        } else {
            self.cursor_geo = None;
        }
    }

    /// Apply primary-button orbit dragging and scroll/pinch cursor-anchored
    /// zoom gestures to `camera`.
    pub(crate) fn handle_viewport_input(
        &self,
        ctx: &egui::Context,
        camera: &mut Camera,
        response: &egui::Response,
        rect: egui::Rect,
        height_pt: f64,
    ) {
        if response.dragged_by(egui::PointerButton::Primary) {
            ctx.set_cursor_icon(egui::CursorIcon::Grabbing);
            let delta = response.drag_delta();
            if delta != egui::Vec2::ZERO {
                camera.orbit(
                    f64::from(delta.x) / height_pt,
                    f64::from(delta.y) / height_pt,
                );
            }
        } else if response.hovered() {
            ctx.set_cursor_icon(egui::CursorIcon::Grab);
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
