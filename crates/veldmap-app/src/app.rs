//! Thin `eframe::App` shell coordinating globe input, GIS HUD layout,
//! storage persistence, and [`SceneState`] frame updates.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use eframe::egui;
use eframe::egui_wgpu;
use thiserror::Error;
use veldmap_core::geo::{Camera, FootprintBounds};
use veldmap_core::sync::Wakeup;
use veldmap_storage::detect_memory_budget;

use crate::gpu_tiles::UploadChannelClosed;
use crate::layers::OverlayIoEvent;
use crate::render::GlobeRenderer;
use crate::scene::{SceneRepaintHint, SceneState};
use crate::storage::AppStorage;
use crate::ui::basemap::{BasemapUiAction, BasemapUiState};
use crate::ui::hud;
use crate::ui::layers::{LayerUiAction, LayersPanelState, show_layers_panel};
use crate::ui::show_settings_window;
use crate::ui::toolbar::{self, GoToBarState, ToolbarAction, ToolbarInputs};
use crate::viewport_bridge::{GlobePaintCallback, viewport_pixels};
use crate::viewport_input::ViewportInteraction;

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
    viewport: ViewportInteraction,
    goto_bar: GoToBarState,
}

impl Drop for VeldmapApp {
    fn drop(&mut self) {
        // Deterministic shutdown order:
        // 1. Cancel tokens / signal background workers
        // 2. Close bounded upload queue (`PendingUploads::close`)
        // 3. Join shared CPU decode pool -> join overlay I/O worker -> shut down `tokio::Runtime` -> join disk-cache worker
        self.scene.begin_shutdown();
        self.storage.begin_shutdown();
        self.scene.close_uploads();
        self.scene.finish_shutdown();
        self.storage.finish_shutdown();
    }
}

impl VeldmapApp {
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
            viewport: ViewportInteraction::default(),
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

    fn queue_open_overlay_path(&mut self, path: &Path, fly_to: bool) {
        if let Err(err) = self.scene.begin_open_overlay_file(path, fly_to) {
            tracing::warn!(path = %path.display(), error = %err, "Failed to queue overlay open");
            self.layers_ui.push_error(err);
        }
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
                    self.layers_ui.on_layer_opened();
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
                    self.queue_open_overlay_path(&path, true);
                    needs_repaint = true;
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

    fn show_globe_viewport(
        &mut self,
        ctx: &egui::Context,
        ui: &mut egui::Ui,
    ) -> Option<egui::Rect> {
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
                    let (width_px, height_px, aspect) = viewport_pixels(ctx, rect);
                    let height_pt = f64::from(rect.height()).max(1.0);
                    self.viewport.record_viewport(rect, height_px, aspect);

                    self.viewport.handle_viewport_input(
                        ctx,
                        &mut self.camera,
                        &response,
                        rect,
                        height_pt,
                    );

                    let show_graticule = self.storage.settings().show_graticule;
                    let snapshot = self.scene.snapshot_for_frame(
                        self.camera,
                        width_px,
                        height_px,
                        ctx.pixels_per_point(),
                        show_graticule,
                    );

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
                    self.viewport.clear_viewport();
                }
            });
        viewport_rect
    }
    fn handle_global_input_and_drops(&mut self, ctx: &egui::Context) -> bool {
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
        if dropped_paths.is_empty() {
            return false;
        }
        self.layers_ui.clear_error();
        for path in dropped_paths {
            self.queue_open_overlay_path(&path, true);
        }
        true
    }

    fn apply_toolbar_actions(&mut self, actions: Vec<ToolbarAction>) -> bool {
        let mut needs_repaint = false;
        for action in actions {
            match action {
                ToolbarAction::OpenFileTrigger => {
                    self.layers_ui.show_open_prompt();
                    needs_repaint = true;
                }
                ToolbarAction::ToggleLayersPanel => {
                    self.layers_ui.toggle_open();
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
        needs_repaint
    }

    fn show_settings_if_open(&mut self, ctx: &egui::Context) -> bool {
        if !self.basemap_ui.is_settings_open() {
            return false;
        }
        self.storage.maybe_refresh_disk_stats(false);
        let disk_status = self.storage.disk_cache_status();
        let settings_actions = show_settings_window(
            ctx,
            self.storage.settings(),
            &disk_status,
            &mut self.basemap_ui,
        );
        self.apply_basemap_actions(settings_actions)
    }

    fn schedule_next_repaint(
        &self,
        ctx: &egui::Context,
        camera_before: Camera,
        rect_before: Option<egui::Rect>,
        needs_repaint: bool,
        scene_hint: SceneRepaintHint,
    ) {
        // Request a follow-up repaint only for active flights, camera changes,
        // viewport resizes, scene/action changes, or scene-driven upload/retry hints.
        if self.camera.is_flying()
            || self.camera != camera_before
            || self.viewport.viewport_rect() != rect_before
            || needs_repaint
            || matches!(scene_hint, SceneRepaintHint::Immediate)
        {
            ctx.request_repaint();
        } else if let SceneRepaintHint::After(delay) = scene_hint {
            ctx.request_repaint_after(delay);
        }
    }
}

impl eframe::App for VeldmapApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        // Drain overlay I/O events BEFORE `begin_frame` so reloaded/removed
        // sources purge stale tiles before upload budget is spent on them.
        let mut needs_repaint = self.drain_overlay_io_events();
        self.scene.begin_frame(&self.device, &self.queue);

        if self.storage.poll_clear_completed() {
            self.scene.retry_live_and_reconcile();
            needs_repaint = true;
        }

        needs_repaint |= self.handle_global_input_and_drops(&ctx);

        let basemap_status = self.scene.basemap_status();
        let layer_summaries = self.scene.layer_summaries();
        let toolbar_actions = toolbar::show_top_toolbar(
            ui,
            ToolbarInputs {
                settings: self.storage.settings(),
                basemap_status: &basemap_status,
                adapter_label: &self.adapter_label,
                layer_count: layer_summaries.len(),
                layers_open: self.layers_ui.is_open(),
            },
            &mut self.goto_bar,
            &mut self.basemap_ui,
        );
        needs_repaint |= self.apply_toolbar_actions(toolbar_actions);
        needs_repaint |= self.show_settings_if_open(&ctx);

        self.viewport.update_cursor_probe(&ctx, &self.camera);
        let cursor_geo = self.viewport.cursor_geo();
        let cursor_readout = match cursor_geo {
            Some((lat, lon)) => self.scene.cursor_overlay_readout(lat, lon),
            None => {
                self.scene.cancel_cursor_readout();
                None
            }
        };
        let topmost_source_px = self.scene.topmost_visible_source_pixel_size_m();
        let camera_before = self.camera;
        let rect_before = self.viewport.viewport_rect();

        toolbar::show_status_bar(
            ui,
            cursor_geo,
            cursor_readout.as_ref(),
            topmost_source_px,
            &self.camera,
            self.viewport.viewport_height_px(),
        );

        let layer_actions = show_layers_panel(ui, &layer_summaries, &mut self.layers_ui);
        needs_repaint |= self.apply_layer_actions(layer_actions);

        let viewport_rect = self.show_globe_viewport(&ctx, ui);
        let scene_hint = self.scene.end_frame();

        // Floating North compass in its own foreground Area (no pointer bleed).
        if let Some(rect) = viewport_rect {
            let (clicked, compass_rect) =
                hud::show_north_compass(&ctx, rect, self.camera.heading_deg());
            self.viewport.set_compass_rect(compass_rect);
            if clicked {
                self.camera.reset_north();
            }
        }

        self.schedule_next_repaint(&ctx, camera_before, rect_before, needs_repaint, scene_hint);
    }
}
