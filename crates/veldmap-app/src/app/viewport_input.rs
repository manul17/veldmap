//! Viewport pointer probing, camera drag/heading-rotation/zoom/keyboard input
//! handling, measurement vertex clicks, STAC AOI box drag capture / viewport
//! bounding-box extraction, and viewport/compass geometry tracking
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

/// Compute the minimal RFC 7946 `[west, south, east, north]` bounding box
/// covering `points` (supporting both wide hemispheric extents and antimeridian
/// crossings where `west > east` via the largest circular longitude gap).
fn minimal_circular_bbox(points: &[(f64, f64)]) -> Option<[f64; 4]> {
    if points.len() < 2 {
        return None;
    }
    let mut min_lat = f64::INFINITY;
    let mut max_lat = f64::NEG_INFINITY;
    let mut lons = Vec::with_capacity(points.len());
    for &(lat, lon) in points {
        if !lat.is_finite() || !lon.is_finite() {
            continue;
        }
        min_lat = min_lat.min(lat);
        max_lat = max_lat.max(lat);
        lons.push(lon);
    }
    if lons.len() < 2 || (max_lat - min_lat) < 1e-6 {
        return None;
    }
    lons.sort_by(f64::total_cmp);

    let mut max_gap = (lons[0] + 360.0) - lons[lons.len() - 1];
    let mut gap_after_idx: Option<usize> = None;
    for i in 0..lons.len() - 1 {
        let gap = lons[i + 1] - lons[i];
        if gap > max_gap {
            max_gap = gap;
            gap_after_idx = Some(i);
        }
    }
    let lon_span = 360.0 - max_gap;
    if lon_span < 1e-6 {
        return None;
    }
    let (west, east) = match gap_after_idx {
        None => (lons[0], lons[lons.len() - 1]),
        Some(k) => (lons[k + 1], lons[k]),
    };
    Some([west, min_lat, east, max_lat])
}

/// Compute an RFC 7946 `[west, south, east, north]` drag rectangle between two
/// globe points using the shortest longitude arc (handling ±180° crossings).
fn aoi_drag_bbox(start: (f64, f64), end: (f64, f64)) -> Option<[f64; 4]> {
    let (start_lat, start_lon) = start;
    let (end_lat, end_lon) = end;
    let dlon = (end_lon - start_lon + 540.0).rem_euclid(360.0) - 180.0;
    let south = start_lat.min(end_lat);
    let north = start_lat.max(end_lat);
    if dlon.abs() <= 1e-4 || (north - south) <= 1e-4 {
        return None;
    }
    let (west, east) = if dlon >= 0.0 {
        (start_lon, end_lon)
    } else {
        (end_lon, start_lon)
    };
    Some([west, south, east, north])
}

/// Active primary-pointer interaction mode on the 3D globe viewport.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum ViewportToolMode {
    #[default]
    Navigate,
    Measure,
    DrawAoiBox,
    Swipe,
}

/// Interactive geodetic or swipe event captured by
/// [`ViewportInteraction::handle_viewport_input`].
#[derive(Clone, Copy, Debug)]
pub(super) enum ViewportEvent {
    MeasurePoint(GeoPoint),
    AoiBox { bbox: [f64; 4], finished: bool },
    AoiBoxCancelled,
    SwipeSplitT { split_t: f32, finished: bool },
}

/// Per-frame viewport geometry and interaction context passed to
/// [`ViewportInteraction::handle_viewport_input`].
#[derive(Clone, Copy)]
pub(super) struct ViewportInputCtx<'a> {
    pub(super) ctx: &'a egui::Context,
    pub(super) response: &'a egui::Response,
    pub(super) rect: egui::Rect,
    pub(super) tool_mode: ViewportToolMode,
    pub(super) swipe_divider_x: Option<f32>,
}

/// Half-width hit radius (logical points) for split-screen swipe divider
/// hover highlighting, resize cursor, and drag capture.
pub(super) const SWIPE_HIT_HALF_PT: f32 = 14.0;

/// Tracks globe viewport geometry, HUD overlay bounds, hovered geodetic
/// cursor coordinates, and pointer/keyboard navigation gestures.
pub(super) struct ViewportInteraction {
    cursor_geo: Option<(f64, f64)>,
    cursor_screen_pos: Option<egui::Pos2>,
    aoi_drag_start: Option<(f64, f64)>,
    swipe_dragging: bool,
    last_viewport_rect: Option<egui::Rect>,
    last_compass_rect: Option<egui::Rect>,
    last_scale_bar_rect: Option<egui::Rect>,
    last_attribution_rect: Option<egui::Rect>,
    last_colorbar_rect: Option<egui::Rect>,
    last_swipe_bar_rect: Option<egui::Rect>,
    last_viewport_height_px: f64,
    last_viewport_aspect: f64,
}

impl Default for ViewportInteraction {
    fn default() -> Self {
        Self {
            cursor_geo: None,
            cursor_screen_pos: None,
            aoi_drag_start: None,
            swipe_dragging: false,
            last_viewport_rect: None,
            last_compass_rect: None,
            last_scale_bar_rect: None,
            last_attribution_rect: None,
            last_colorbar_rect: None,
            last_swipe_bar_rect: None,
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
        self.last_swipe_bar_rect = None;
        self.swipe_dragging = false;
    }

    pub(super) fn set_compass_rect(&mut self, rect: egui::Rect) {
        self.last_compass_rect = Some(rect);
    }

    pub(super) fn set_swipe_bar_rect(&mut self, rect: Option<egui::Rect>) {
        self.last_swipe_bar_rect = rect;
    }

    pub(super) fn is_swipe_dragging(&self) -> bool {
        self.swipe_dragging
    }

    pub(super) fn cancel_swipe_drag(&mut self) {
        self.swipe_dragging = false;
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

    pub(super) fn is_over_hud(&self, pos: egui::Pos2) -> bool {
        self.last_compass_rect.is_some_and(|r| r.contains(pos))
            || self.last_scale_bar_rect.is_some_and(|r| r.contains(pos))
            || self.last_attribution_rect.is_some_and(|r| r.contains(pos))
            || self.last_colorbar_rect.is_some_and(|r| r.contains(pos))
            || self.last_swipe_bar_rect.is_some_and(|r| r.contains(pos))
    }

    pub(super) fn cursor_screen_pos(&self) -> Option<egui::Pos2> {
        self.cursor_screen_pos
    }

    pub(super) fn is_hovering_swipe_divider(&self, response: &egui::Response, div_x: f32) -> bool {
        response
            .hover_pos()
            .is_some_and(|p| !self.is_over_hud(p) && (p.x - div_x).abs() <= SWIPE_HIT_HALF_PT)
    }

    pub(super) fn probe_screen_pos(
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
            let mut valid_pos = None;
            self.cursor_geo = hover_pos.and_then(|pos| {
                // Skip probing whenever a floating window, menu, or compass
                // `Area` sits above the globe `CentralPanel` (`LayerId::background`).
                let covered_by_ui_layer = ctx
                    .layer_id_at(pos)
                    .is_some_and(|layer| layer != egui::LayerId::background());
                let over_globe =
                    rect.contains(pos) && !covered_by_ui_layer && !self.is_over_hud(pos);
                if over_globe {
                    valid_pos = Some(pos);
                }
                over_globe
                    .then(|| self.probe_screen_pos(camera, rect, pos))
                    .flatten()
            });
            self.cursor_screen_pos = valid_pos;
        } else {
            self.cursor_geo = None;
            self.cursor_screen_pos = None;
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

    /// Cancel any in-progress AOI box drag anchor.
    pub(super) fn cancel_aoi_drag(&mut self) {
        self.aoi_drag_start = None;
    }

    /// Approximate `[west, south, east, north]` covering the current visible
    /// globe viewport by probing a 5×5 screen grid, or `None` if fewer than
    /// two rays intersect the WGS84 ellipsoid.
    pub(super) fn viewport_bbox(&self, camera: &Camera) -> Option<[f64; 4]> {
        let mut hits = Vec::with_capacity(25);
        for gy in 0..=4 {
            let ny = 0.08 + 0.84 * (f64::from(gy) / 4.0);
            for gx in 0..=4 {
                let nx = 0.08 + 0.84 * (f64::from(gx) / 4.0);
                if let Some(pt) = camera.probe(nx, ny, self.last_viewport_aspect) {
                    hits.push(pt);
                }
            }
        }
        minimal_circular_bbox(&hits)
    }

    fn handle_swipe_drag(
        &mut self,
        input: ViewportInputCtx<'_>,
        shift_held: bool,
    ) -> Option<ViewportEvent> {
        let Some(div_x) = input
            .swipe_divider_x
            .filter(|_| input.tool_mode == ViewportToolMode::Swipe)
        else {
            self.swipe_dragging = false;
            return None;
        };

        let width_pt = input.rect.width().max(1.0);
        if !self.swipe_dragging
            && !shift_held
            && input.response.drag_started_by(egui::PointerButton::Primary)
            && let Some(pos) = input
                .ctx
                .input(|i| i.pointer.press_origin())
                .or_else(|| input.response.interact_pointer_pos())
            && !self.is_over_hud(pos)
            && (pos.x - div_x).abs() <= SWIPE_HIT_HALF_PT
        {
            self.swipe_dragging = true;
        }
        if !self.swipe_dragging {
            return None;
        }

        input
            .ctx
            .set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
        if input.response.dragged_by(egui::PointerButton::Primary)
            && let Some(pos) = input.response.interact_pointer_pos()
        {
            let split_t = ((pos.x - input.rect.left()) / width_pt).clamp(0.0, 1.0);
            Some(ViewportEvent::SwipeSplitT {
                split_t,
                finished: false,
            })
        } else if input.response.drag_stopped_by(egui::PointerButton::Primary)
            || !input.ctx.input(|i| i.pointer.primary_down())
        {
            self.swipe_dragging = false;
            let split_t = input.response.interact_pointer_pos().map_or(
                ((div_x - input.rect.left()) / width_pt).clamp(0.0, 1.0),
                |pos| ((pos.x - input.rect.left()) / width_pt).clamp(0.0, 1.0),
            );
            Some(ViewportEvent::SwipeSplitT {
                split_t,
                finished: true,
            })
        } else {
            None
        }
    }

    fn handle_aoi_drag(
        &mut self,
        camera: &Camera,
        input: ViewportInputCtx<'_>,
        shift_held: bool,
    ) -> Option<ViewportEvent> {
        if input.tool_mode != ViewportToolMode::DrawAoiBox || shift_held {
            return self
                .aoi_drag_start
                .take()
                .map(|_| ViewportEvent::AoiBoxCancelled);
        }

        if input.response.hovered() || self.aoi_drag_start.is_some() {
            input.ctx.set_cursor_icon(egui::CursorIcon::Crosshair);
        }
        if input.response.drag_started_by(egui::PointerButton::Primary)
            && let Some(pos) = input.response.interact_pointer_pos()
            && !self.is_over_hud(pos)
        {
            self.aoi_drag_start = self.probe_screen_pos(camera, input.rect, pos);
        }
        let start = self.aoi_drag_start?;
        if input.response.dragged_by(egui::PointerButton::Primary)
            && let Some(pos) = input.response.interact_pointer_pos()
            && let Some(cur) = self.probe_screen_pos(camera, input.rect, pos)
        {
            aoi_drag_bbox(start, cur).map(|bbox| ViewportEvent::AoiBox {
                bbox,
                finished: false,
            })
        } else if input.response.drag_stopped_by(egui::PointerButton::Primary)
            || !input.ctx.input(|i| i.pointer.primary_down())
        {
            self.aoi_drag_start = None;
            let final_bbox = input
                .response
                .interact_pointer_pos()
                .and_then(|pos| self.probe_screen_pos(camera, input.rect, pos))
                .and_then(|end| aoi_drag_bbox(start, end));
            Some(match final_bbox {
                Some(bbox) => ViewportEvent::AoiBox {
                    bbox,
                    finished: true,
                },
                None => ViewportEvent::AoiBoxCancelled,
            })
        } else {
            None
        }
    }

    fn handle_camera_gestures(
        &self,
        camera: &mut Camera,
        input: ViewportInputCtx<'_>,
        height_pt: f64,
        rotating: bool,
    ) {
        if rotating {
            input
                .ctx
                .set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
            let delta = input.response.drag_delta();
            if delta.x != 0.0 {
                camera.rotate_heading(f64::from(delta.x) * ROTATE_RAD_PER_PT);
            }
        } else if input.tool_mode != ViewportToolMode::DrawAoiBox && !self.swipe_dragging {
            if input.response.dragged_by(egui::PointerButton::Primary) {
                input.ctx.set_cursor_icon(egui::CursorIcon::Grabbing);
                let delta = input.response.drag_delta();
                if delta != egui::Vec2::ZERO {
                    camera.orbit(
                        f64::from(delta.x) / height_pt,
                        f64::from(delta.y) / height_pt,
                    );
                }
            } else if input.response.hovered() {
                let over_swipe_handle = input.tool_mode == ViewportToolMode::Swipe
                    && input
                        .swipe_divider_x
                        .is_some_and(|div_x| self.is_hovering_swipe_divider(input.response, div_x));
                input.ctx.set_cursor_icon(if over_swipe_handle {
                    egui::CursorIcon::ResizeHorizontal
                } else if input.tool_mode == ViewportToolMode::Measure {
                    egui::CursorIcon::Crosshair
                } else {
                    egui::CursorIcon::Grab
                });
            }
        }
    }

    fn handle_scroll_zoom(camera: &mut Camera, input: ViewportInputCtx<'_>, height_pt: f64) {
        if !input.response.hovered() {
            return;
        }
        let (scroll_y, zoom_delta) = input
            .ctx
            .input(|i| (i.smooth_scroll_delta.y, i.zoom_delta()));
        let steps = zoom_steps_from_input(scroll_y, zoom_delta);
        if steps.abs() > 0.0 {
            let (hold_x, hold_y) = match input.response.hover_pos() {
                Some(pos) => (
                    f64::from(pos.x - input.rect.center().x) / height_pt,
                    f64::from(pos.y - input.rect.center().y) / height_pt,
                ),
                None => (0.0, 0.0),
            };
            camera.zoom_at(steps, hold_x, hold_y);
        }
    }

    /// Apply primary-button orbit dragging, split-screen swipe curtain dragging,
    /// secondary / `Shift+Primary` manual heading rotation, scroll/pinch
    /// cursor-anchored zoom, and optional measurement click or AOI box drag
    /// capture to `camera`.
    pub(super) fn handle_viewport_input(
        &mut self,
        camera: &mut Camera,
        input: ViewportInputCtx<'_>,
    ) -> Option<ViewportEvent> {
        let shift_held = input.ctx.input(|i| i.modifiers.shift);
        let height_pt = f64::from(input.rect.height()).max(1.0);

        // Split-screen swipe curtain drag capture (within ±SWIPE_HIT_HALF_PT of
        // divider). An already-started drag continues until release even if
        // `Shift` is pressed mid-drag so session persistence is never skipped.
        let swipe_ev = self.handle_swipe_drag(input, shift_held);

        let rotating = input.response.dragged_by(egui::PointerButton::Secondary)
            || (!self.swipe_dragging
                && shift_held
                && input.response.dragged_by(egui::PointerButton::Primary));

        let aoi_ev = self.handle_aoi_drag(camera, input, shift_held);
        let captured_event = aoi_ev.or(swipe_ev);

        self.handle_camera_gestures(camera, input, height_pt, rotating);
        Self::handle_scroll_zoom(camera, input, height_pt);

        if captured_event.is_some() {
            return captured_event;
        }

        if input.tool_mode == ViewportToolMode::Measure
            && input.response.clicked_by(egui::PointerButton::Primary)
            && !shift_held
            && let Some(pos) = input.response.interact_pointer_pos()
            && !self.is_over_hud(pos)
            && let Some((lat, lon)) = self.probe_screen_pos(camera, input.rect, pos)
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

    #[test]
    fn circular_bbox_and_aoi_drag_handle_wide_views_and_antimeridian() {
        // Wide non-antimeridian view (-95..+95 = 190 deg) must stay [-95, +95],
        // never collapsing to a 10 deg patch.
        let wide = minimal_circular_bbox(&[
            (-20.0, -95.0),
            (-20.0, -30.0),
            (0.0, 0.0),
            (20.0, 30.0),
            (20.0, 95.0),
        ])
        .expect("wide");
        assert_eq!(wide, [-95.0, -20.0, 95.0, 20.0]);

        // True antimeridian crossing (+170..-170 = 20 deg span across 180).
        let pac =
            minimal_circular_bbox(&[(-10.0, 170.0), (0.0, 178.0), (10.0, -172.0), (10.0, -170.0)])
                .expect("antimeridian");
        assert_eq!(pac, [170.0, -10.0, -170.0, 10.0]);

        // Antimeridian drag eastbound and westbound.
        assert_eq!(
            aoi_drag_bbox((-5.0, 175.0), (5.0, -175.0)),
            Some([175.0, -5.0, -175.0, 5.0])
        );
        assert_eq!(
            aoi_drag_bbox((5.0, -175.0), (-5.0, 175.0)),
            Some([175.0, -5.0, -175.0, 5.0])
        );
    }
}
