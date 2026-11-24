//! Thin `eframe::App` shell coordinating globe input, GIS HUD layout,
//! storage persistence, and [`SceneState`] frame updates.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use eframe::egui;
use eframe::egui_wgpu;
use thiserror::Error;
use veldmap_core::geo::camera::{Camera, ZOOM_PER_CLICK};
use veldmap_core::geo::footprint::FootprintBounds;
use veldmap_core::sync::Wakeup;
use veldmap_storage::detect_memory_budget;

use crate::gpu_tiles::UploadChannelClosed;
use crate::layers::OverlayIoEvent;
use crate::render::GlobeRenderer;
use crate::scene::SceneState;
use crate::storage::AppStorage;
use crate::ui::basemap::{BasemapUiAction, BasemapUiState};
use crate::ui::hud;
use crate::ui::layers::{LayerUiAction, LayersPanelState, show_layers_panel};
use crate::ui::show_settings_window;
use crate::ui::toolbar::{self, GoToBarState, ToolbarAction, ToolbarInputs};
use crate::viewport_bridge::{GlobePaintCallback, viewport_pixels};

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

#[derive(Debug, Error)]
pub enum AppError {
    #[error("eframe did not provide a wgpu render state")]
    MissingWgpu,
    #[error("tile upload completion channel was closed during initialization")]
    UploadChannelClosed,
}

impl From<UploadChannelClosed> for AppError {
    fn from(_: UploadChannelClosed) -> Self {
        Self::UploadChannelClosed
    }
}

pub struct VeldmapApp {
    adapter_label: String,
    device: wgpu::Device,
    queue: wgpu::Queue,
    camera: Camera,
    scene: SceneState,
    storage: AppStorage,
    basemap_ui: BasemapUiState,
    layers_ui: LayersPanelState,
    cursor_geo: Option<(f64, f64)>,
    last_viewport_rect: Option<egui::Rect>,
    last_compass_rect: Option<egui::Rect>,
    last_viewport_height_px: f64,
    last_viewport_aspect: f64,
    goto_bar: GoToBarState,
}

impl Drop for VeldmapApp {
    fn drop(&mut self) {
        // Deterministic shutdown order:
        // 1. Cancel tokens / signal background workers
        // 2. Close bounded upload queue (`PendingUploads::close`)
        // 3. Join shared CPU decode pool -> shut down `tokio::Runtime` -> join disk-cache worker
        self.scene.begin_shutdown();
        self.storage.begin_shutdown();
        self.scene.close_uploads();
        self.scene.finish_shutdown();
        self.storage.finish_shutdown();
    }
}

impl VeldmapApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Result<Self, AppError> {
        Self::new_with_startup_files(cc, Vec::new())
    }

    pub fn new_with_startup_files(
        cc: &eframe::CreationContext<'_>,
        startup_files: Vec<PathBuf>,
    ) -> Result<Self, AppError> {
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

        let device = render_state.device.clone();
        let queue = render_state.queue.clone();

        let egui_ctx = cc.egui_ctx.clone();
        let wakeup: Wakeup = Arc::new(move || {
            egui_ctx.request_repaint();
        });

        let budget = detect_memory_budget();
        let storage = AppStorage::open(Arc::clone(&wakeup), None);
        let scene = SceneState::new(storage.settings(), storage.disk_cache(), wakeup, &budget)?;
        let basemap_ui = BasemapUiState::new(storage.settings());

        let mut app = Self {
            adapter_label,
            device,
            queue,
            camera: Camera::default(),
            scene,
            storage,
            basemap_ui,
            layers_ui: LayersPanelState::default(),
            cursor_geo: None,
            last_viewport_rect: None,
            last_compass_rect: None,
            last_viewport_height_px: 800.0,
            last_viewport_aspect: 1.6,
            goto_bar: GoToBarState::default(),
        };

        for path in startup_files {
            app.queue_open_overlay_path(&path, true);
        }

        Ok(app)
    }

    fn focus_footprint(&mut self, bounds: FootprintBounds) {
        self.camera
            .focus(bounds.lat, bounds.lon, bounds.radius_deg.max(0.000_2));
    }

    fn queue_open_overlay_path(&mut self, path: &Path, fly_to: bool) -> bool {
        if let Err(err) = self.scene.begin_open_overlay_file(path, fly_to) {
            tracing::warn!(path = %path.display(), error = %err, "Failed to queue overlay open");
            self.layers_ui.push_error(err);
        }
        true
    }

    fn drain_overlay_io_events(&mut self) -> bool {
        let events = self.scene.poll_overlay_io_events();
        if events.is_empty() {
            return false;
        }
        let mut had_error = false;
        let mut had_open = false;
        for event in events {
            match event {
                OverlayIoEvent::Opened { id, bounds, fly_to } => {
                    tracing::info!(?id, "Opened local overlay layer");
                    self.layers_ui.open = true;
                    self.layers_ui.path_input.clear();
                    self.layers_ui.open_prompt_visible = false;
                    had_open = true;
                    if fly_to && let Some(b) = bounds {
                        self.focus_footprint(b);
                    }
                }
                OverlayIoEvent::Reloaded { id } => {
                    tracing::info!(?id, "Reloaded local overlay layer");
                    self.layers_ui.clear_error();
                }
                OverlayIoEvent::Failed { path, error } => {
                    tracing::warn!(path = %path.display(), error = %error, "Overlay I/O failed");
                    self.layers_ui.push_error(error);
                    had_error = true;
                }
            }
        }
        if had_open && !had_error {
            self.layers_ui.clear_error();
        }
        true
    }

    fn apply_layer_actions(&mut self, actions: Vec<LayerUiAction>) -> bool {
        let mut needs_repaint = false;
        for action in actions {
            match action {
                LayerUiAction::OpenPath(path) => {
                    self.layers_ui.clear_error();
                    needs_repaint |= self.queue_open_overlay_path(&path, true);
                }
                LayerUiAction::ToggleVisibility(id, visible) => {
                    needs_repaint |= self.scene.set_overlay_visible(id, visible);
                }
                LayerUiAction::SetOpacity(id, opacity) => {
                    needs_repaint |= self.scene.set_overlay_opacity(id, opacity);
                }
                LayerUiAction::ZoomToExtent(bounds) => {
                    self.focus_footprint(bounds);
                    needs_repaint = true;
                }
                LayerUiAction::ReloadBypassCache(id) => {
                    if let Err(err) = self.scene.begin_reload_overlay_bypass_cache(id) {
                        tracing::warn!(?id, error = %err, "Failed to queue overlay reload");
                        self.layers_ui.push_error(err);
                    }
                    needs_repaint = true;
                }
                LayerUiAction::Remove(id) => {
                    needs_repaint |= self.scene.remove_overlay(id);
                }
            }
        }
        needs_repaint
    }

    fn apply_basemap_action(&mut self, action: BasemapUiAction) -> bool {
        let mut needs_repaint = false;
        match action {
            BasemapUiAction::SelectBasemap(selection) => {
                needs_repaint |= self
                    .storage
                    .mutate_settings(|s| s.active_basemap = selection);
                needs_repaint |= self.scene.sync_basemap_settings(self.storage.settings());
                // Preserve unsaved edits in an open Settings window; only
                // synchronize the active selection field.
                self.basemap_ui
                    .set_draft_active_basemap(self.storage.settings().active_basemap);
            }
            BasemapUiAction::ApplyDraft(draft) => {
                needs_repaint |= self.storage.mutate_settings(|s| draft.apply_into(s));
                needs_repaint |= self.scene.sync_basemap_settings(self.storage.settings());
                self.basemap_ui.sync_from_settings(self.storage.settings());
            }
            BasemapUiAction::SettingsOpened => {
                self.storage.maybe_refresh_disk_stats(true);
            }
            BasemapUiAction::ClearDiskCache => {
                self.storage.request_clear_disk_cache();
                needs_repaint = true;
            }
            BasemapUiAction::ClearDiskCacheNotice => {
                self.storage.clear_disk_cache_notice();
            }
            BasemapUiAction::RetryLive => {
                self.scene.retry_live_and_reconcile();
                needs_repaint = true;
            }
        }
        needs_repaint
    }

    fn apply_basemap_actions(&mut self, actions: Vec<BasemapUiAction>) -> bool {
        let mut needs_repaint = false;
        for action in actions {
            needs_repaint |= self.apply_basemap_action(action);
        }
        needs_repaint
    }

    fn probe_pointer(&self, rect: egui::Rect, pos: egui::Pos2) -> Option<(f64, f64)> {
        let width_pt = f64::from(rect.width()).max(1.0);
        let height_pt = f64::from(rect.height()).max(1.0);
        let nx = f64::from(pos.x - rect.left()) / width_pt;
        let ny = f64::from(pos.y - rect.top()) / height_pt;
        self.camera.probe(nx, ny, self.last_viewport_aspect)
    }

    fn update_cursor_probe(&mut self, ctx: &egui::Context) {
        // Probe the current pointer against the previous frame's viewport rect
        // before laying out the bottom status bar so hover readouts stay in
        // sync without scheduling a duplicate repaint on every mouse move.
        if let Some(rect) = self.last_viewport_rect.filter(|r| r.is_positive()) {
            let hover_pos = ctx.input(|i| i.pointer.hover_pos());
            self.cursor_geo = hover_pos.and_then(|pos| {
                let over_compass = self.last_compass_rect.is_some_and(|c| c.contains(pos));
                (rect.contains(pos) && !over_compass)
                    .then(|| self.probe_pointer(rect, pos))
                    .flatten()
            });
        } else {
            self.cursor_geo = None;
        }
    }

    fn handle_viewport_input(
        &mut self,
        ctx: &egui::Context,
        response: &egui::Response,
        rect: egui::Rect,
        height_pt: f64,
    ) {
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
                self.camera.zoom_at(steps, hold_x, hold_y);
            }
        }
    }

    fn show_globe_viewport(
        &mut self,
        ctx: &egui::Context,
        ui: &mut egui::Ui,
    ) -> (Option<egui::Rect>, bool) {
        let mut viewport_rect = None;
        let mut reconciled = false;
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
                    let (width_px, height_px, aspect) = viewport_pixels(ctx, rect);
                    let height_pt = f64::from(rect.height()).max(1.0);
                    self.last_viewport_height_px = f64::from(height_px);
                    self.last_viewport_aspect = aspect;

                    self.handle_viewport_input(ctx, &response, rect, height_pt);

                    let show_graticule = self.storage.settings().show_graticule;
                    let (snapshot, did_reconcile) = self.scene.snapshot_for_frame(
                        self.camera,
                        width_px,
                        height_px,
                        ctx.pixels_per_point(),
                        show_graticule,
                    );
                    reconciled = did_reconcile;

                    // Register offscreen globe + 1:1 nearest composite paint callback.
                    painter.add(egui_wgpu::Callback::new_paint_callback(
                        rect,
                        GlobePaintCallback { rect, snapshot },
                    ));

                    // Floating dynamic metric scale bar.
                    let m_per_pt = self.camera.metres_per_pixel(height_pt);
                    hud::paint_scale_bar(&painter, rect, m_per_pt);

                    // Visible basemap attribution pill (NASA GIBS / EOX CC BY 4.0 / Custom).
                    if let Some(attr) = self.scene.attribution() {
                        hud::paint_attribution_badge(&painter, rect, attr);
                    }
                } else {
                    self.scene.clear_cached_snapshot();
                    self.last_viewport_rect = None;
                    self.last_compass_rect = None;
                }
            });
        (viewport_rect, reconciled)
    }
}

impl eframe::App for VeldmapApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        let mut needs_repaint = self.drain_overlay_io_events();
        let upload_report = self.scene.drain_tile_uploads(&self.device, &self.queue);

        let mut reconciled_this_frame = false;
        if self.storage.poll_clear_completed() {
            self.scene.retry_live_and_reconcile();
            reconciled_this_frame = true;
            needs_repaint = true;
        }

        // Advance active quaternion camera flights (cap dt to 50 ms so a
        // frame hitch never jumps abruptly through a 0.6 s flight).
        let dt_s = f64::from(ctx.input(|i| i.stable_dt)).clamp(0.0, 0.05);
        self.camera.advance(dt_s);

        // Global shortcut `N` (without any modifiers) resets North when no
        // text field owns keyboard focus.
        if !ctx.egui_wants_keyboard_input()
            && ctx.input(|i| i.key_pressed(egui::Key::N) && i.modifiers.is_none())
        {
            self.camera.reset_north();
        }

        // Handle OS drag-and-drop of local GeoTIFF/BigTIFF/COG files.
        let dropped_paths: Vec<PathBuf> = ctx.input(|i| {
            i.raw
                .dropped_files
                .iter()
                .map(|f| f.path().to_path_buf())
                .filter(|p| !p.as_os_str().is_empty())
                .collect()
        });
        if !dropped_paths.is_empty() {
            self.layers_ui.clear_error();
            for path in dropped_paths {
                needs_repaint |= self.queue_open_overlay_path(&path, true);
            }
        }

        // Top toolbar and Settings window.
        let basemap_status = self.scene.basemap_status();
        let layer_summaries = self.scene.layer_summaries();
        let toolbar_actions = toolbar::show_top_toolbar(
            ui,
            ToolbarInputs {
                settings: self.storage.settings(),
                basemap_status: &basemap_status,
                adapter_label: &self.adapter_label,
                layer_count: layer_summaries.len(),
                layers_open: self.layers_ui.open,
            },
            &mut self.goto_bar,
            &mut self.basemap_ui,
        );
        for action in toolbar_actions {
            match action {
                ToolbarAction::OpenFileTrigger => {
                    self.layers_ui.show_open_prompt();
                    needs_repaint = true;
                }
                ToolbarAction::ToggleLayersPanel => {
                    self.layers_ui.open = !self.layers_ui.open;
                    needs_repaint = true;
                }
                ToolbarAction::FocusLocation(loc) => {
                    tracing::info!(
                        lat = loc.lat_deg,
                        lon = loc.lon_deg,
                        radius_deg = loc.radius_deg,
                        kind = ?loc.kind,
                        "Flying camera to parsed location"
                    );
                    self.camera.focus(loc.lat_deg, loc.lon_deg, loc.radius_deg);
                    needs_repaint = true;
                }
                ToolbarAction::SetGraticule(show_graticule) => {
                    needs_repaint |= self
                        .storage
                        .mutate_settings(|s| s.show_graticule = show_graticule);
                }
                ToolbarAction::Basemap(basemap_action) => {
                    needs_repaint |= self.apply_basemap_action(basemap_action);
                }
            }
        }

        if self.basemap_ui.is_settings_open() {
            self.storage.maybe_refresh_disk_stats(false);
            let disk_status = self.storage.disk_cache_status();
            let settings_actions = show_settings_window(
                &ctx,
                self.storage.settings(),
                &disk_status,
                &mut self.basemap_ui,
            );
            needs_repaint |= self.apply_basemap_actions(settings_actions);
        }

        self.update_cursor_probe(&ctx);
        let cursor_readout = match self.cursor_geo {
            Some((lat, lon)) => self.scene.cursor_overlay_readout(lat, lon),
            None => {
                self.scene.cancel_cursor_readout();
                None
            }
        };
        let topmost_source_px = self.scene.topmost_visible_source_pixel_size_m();
        let camera_before = self.camera;
        let rect_before = self.last_viewport_rect;

        toolbar::show_status_bar(
            ui,
            self.cursor_geo,
            cursor_readout.as_ref(),
            topmost_source_px,
            &self.camera,
            self.last_viewport_height_px,
        );

        let layer_actions = show_layers_panel(ui, &layer_summaries, &mut self.layers_ui);
        needs_repaint |= self.apply_layer_actions(layer_actions);

        let (viewport_rect, viewport_reconciled) = self.show_globe_viewport(&ctx, ui);
        reconciled_this_frame |= viewport_reconciled;

        // If uploads progressed or queued tiles remain deferred and the
        // viewport did not already rebuild/reconcile providers this frame
        // (e.g. collapsed window or zero-upload deferred batch), reconcile now.
        if !reconciled_this_frame
            && (upload_report.uploaded > 0
                || upload_report.remaining_ready > 0
                || upload_report.deferred_protected > 0)
        {
            self.scene.reconcile_providers();
        }

        // Floating North compass in its own foreground Area (no pointer bleed).
        if let Some(rect) = viewport_rect {
            let (clicked, compass_rect) =
                hud::show_north_compass(&ctx, rect, self.camera.heading_deg());
            self.last_compass_rect = Some(compass_rect);
            if clicked {
                self.camera.reset_north();
            }
        }

        // Request a follow-up repaint only for active flights, camera changes,
        // viewport resizes, scene/action changes, budget-deferred uploads, or
        // partial upload progress that may have unprotected a deferred tile.
        if self.camera.is_flying()
            || self.camera != camera_before
            || self.last_viewport_rect != rect_before
            || needs_repaint
            || upload_report.remaining_ready > 0
            || (upload_report.uploaded > 0 && upload_report.deferred_protected > 0)
        {
            ctx.request_repaint();
        } else if let Some(delay) = self.scene.overlay_retry_delay() {
            ctx.request_repaint_after(delay);
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
