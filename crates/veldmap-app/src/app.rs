//! Thin `eframe::App` shell coordinating globe input, GIS HUD layout,
//! storage/session persistence, native file dialogs, geodesic measurement,
//! CRS overrides, and [`SceneState`] frame updates.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use eframe::egui;
use eframe::egui_wgpu;
use thiserror::Error;
use veldmap_core::geo::{Camera, FootprintBounds};
use veldmap_core::sync::Wakeup;
use veldmap_storage::detect_memory_budget;

use crate::dialogs::{DialogHost, FileDialogEvent};
use crate::fonts::build_font_definitions;
use crate::gpu_tiles::UploadChannelClosed;
use crate::layers::{OverlayIoEvent, is_geojson_path};
use crate::render::GlobeRenderer;
use crate::scene::{SceneRepaintHint, SceneState};
use crate::session::{CameraBookmark, SavedCameraPose, SavedCrsOverride, SavedLayerEntry};
use crate::storage::AppStorage;
use crate::ui::basemap::{BasemapUiAction, BasemapUiState};
use crate::ui::crs_picker::{CrsPickerAction, CrsPickerState, show_crs_picker_window};
use crate::ui::hud;
use crate::ui::layers::{LayerUiAction, LayersPanelState, show_layers_panel};
use crate::ui::measure::{MeasureAction, MeasureState, show_measure_window};
use crate::ui::toolbar::{self, GoToBarState, ToolbarAction, ToolbarInputs};
use crate::ui::{show_settings_window, status_bar};
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
    dialogs: DialogHost,
    basemap_ui: BasemapUiState,
    layers_ui: LayersPanelState,
    crs_picker_ui: CrsPickerState,
    measure_ui: MeasureState,
    viewport: ViewportInteraction,
    goto_bar: GoToBarState,
    /// Layers queued for background restoration from `session.json` at startup
    /// whose `OverlayIoEvent` has not yet drained; preserved on early exit so
    /// quitting before background open finishes never wipes persisted layers.
    pending_restored_layers: Vec<SavedLayerEntry>,
}

impl Drop for VeldmapApp {
    fn drop(&mut self) {
        // Persist final camera pose & open layer/vector list before closing the
        // serialized disk worker so `session.json` drains durably before exit.
        self.persist_session_snapshot(true);

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
        // Install permissively licensed DejaVu Sans / Mono non-Latin fallback
        // fonts for Cyrillic, international file paths, and GIS symbols.
        cc.egui_ctx.set_fonts(build_font_definitions());

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
        let dialogs = DialogHost::new(Arc::clone(&wakeup));
        let scene = SceneState::new(storage.settings(), storage.disk_cache(), wakeup, &budget)?;
        let basemap_ui = BasemapUiState::new(storage.settings());

        let restored_session = storage.session().clone();
        let mut camera = Camera::default();
        if let Some(pose) = restored_session.last_camera {
            camera = Camera::at(pose.lat_deg, pose.lon_deg, pose.height_m);
            if pose.heading_deg.abs() > 1e-6 {
                camera.rotate_heading(pose.heading_deg.to_radians());
            }
        }

        let mut app = Self {
            adapter_label,
            device,
            queue,
            camera,
            scene,
            storage,
            dialogs,
            basemap_ui,
            layers_ui: LayersPanelState::default(),
            crs_picker_ui: CrsPickerState::default(),
            measure_ui: MeasureState::default(),
            viewport: ViewportInteraction::default(),
            goto_bar: GoToBarState::default(),
            pending_restored_layers: Vec::new(),
        };

        // Restore persisted open layers/vectors from `session.json` silently if
        // they still exist on disk.
        for saved in restored_session.open_layers {
            if saved.path.is_file()
                && app
                    .scene
                    .begin_open_overlay_file_with_options(
                        &saved.path,
                        false,
                        saved.visible,
                        saved.opacity,
                        saved.crs_override.clone(),
                    )
                    .is_ok()
            {
                app.pending_restored_layers.push(saved);
            }
        }
        for saved_vec in restored_session.open_vectors {
            if saved_vec.path.is_file()
                && app
                    .scene
                    .open_vector_file(&saved_vec.path, saved_vec.visible, saved_vec.opacity)
                    .is_ok()
            {
                app.layers_ui.on_layer_opened();
            }
        }

        for path in startup_files {
            app.open_local_path(&path, true, None);
        }

        Ok(app)
    }

    fn persist_session_snapshot(&mut self, include_camera: bool) {
        let mut open_layers = self.scene.saved_layer_entries();
        for pending in &self.pending_restored_layers {
            if !open_layers.iter().any(|l| l.path == pending.path) {
                open_layers.push(pending.clone());
            }
        }
        let open_vectors = self.scene.saved_vector_entries();
        let cam_pose = if include_camera {
            let (lat_deg, lon_deg) = self.camera.center();
            Some(SavedCameraPose {
                lat_deg,
                lon_deg,
                height_m: self.camera.height_m(),
                heading_deg: self.camera.heading_deg(),
            })
        } else {
            self.storage.session().last_camera
        };
        let _ = self.storage.mutate_session(|s| {
            s.open_layers = open_layers;
            s.open_vectors = open_vectors;
            s.last_camera = cam_pose;
        });
    }

    fn focus_footprint(&mut self, bounds: FootprintBounds) {
        self.camera
            .focus(bounds.lat, bounds.lon, bounds.radius_deg.max(0.000_2));
    }

    fn open_local_path(
        &mut self,
        path: &Path,
        fly_to: bool,
        crs_override: Option<SavedCrsOverride>,
    ) {
        if is_geojson_path(path) {
            match self.scene.open_vector_file(path, true, 1.0) {
                Ok((id, bounds)) => {
                    tracing::info!(?id, path = %path.display(), "Opened GeoJSON vector layer");
                    self.layers_ui.on_layer_opened();
                    self.layers_ui.clear_error();
                    if fly_to && let Some(b) = bounds {
                        self.focus_footprint(b);
                    }
                    let owned = path.to_path_buf();
                    let _ = self
                        .storage
                        .mutate_session(|s| s.record_recent_file(&owned));
                    self.persist_session_snapshot(false);
                }
                Err(err) => {
                    tracing::warn!(path = %path.display(), error = %err, "Failed to open GeoJSON");
                    self.layers_ui.push_error(err);
                }
            }
            return;
        }

        if let Err(err) =
            self.scene
                .begin_open_overlay_file_with_options(path, fly_to, true, 1.0, crs_override)
        {
            tracing::warn!(path = %path.display(), error = %err, "Failed to queue overlay open");
            self.layers_ui.push_error(err);
        } else {
            let owned = path.to_path_buf();
            let _ = self
                .storage
                .mutate_session(|s| s.record_recent_file(&owned));
        }
    }

    fn trigger_native_open_dialog(&mut self) {
        if !self.dialogs.spawn_open_files() && !self.dialogs.is_busy() {
            // Fallback to inline path prompt if spawning the dialog thread failed.
            self.layers_ui.show_open_prompt();
        }
    }

    fn drain_file_dialog_events(&mut self) -> bool {
        let events = self.dialogs.poll();
        if events.is_empty() {
            return false;
        }
        for ev in events {
            match ev {
                FileDialogEvent::OpenFiles(paths) => {
                    self.layers_ui.clear_error();
                    for path in paths {
                        self.open_local_path(&path, true, None);
                    }
                }
                FileDialogEvent::SaveGeoJson(path) => {
                    if let Some(json) = self.measure_ui.export_geojson_string() {
                        match std::fs::write(&path, json.as_bytes()) {
                            Ok(()) => {
                                self.measure_ui
                                    .set_notice(format!("Exported to {}", path.display()));
                            }
                            Err(err) => {
                                self.measure_ui.set_notice(format!("Export failed: {err}"));
                            }
                        }
                    }
                }
            }
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
                    self.pending_restored_layers.retain(|p| p.path != path);
                    self.layers_ui.note_failed_path(path, error);
                    had_error = true;
                }
            }
        }
        if had_open {
            if !self.pending_restored_layers.is_empty() {
                let opened_paths: Vec<PathBuf> = self
                    .scene
                    .saved_layer_entries()
                    .into_iter()
                    .map(|l| l.path)
                    .collect();
                self.pending_restored_layers
                    .retain(|p| !opened_paths.iter().any(|open| open == &p.path));
            }
            self.persist_session_snapshot(false);
            if !had_error {
                self.layers_ui.clear_error();
            }
        }
        true
    }

    fn apply_layer_actions(&mut self, actions: Vec<LayerUiAction>) -> bool {
        let mut needs_repaint = false;
        let mut session_dirty = false;
        for action in actions {
            match action {
                LayerUiAction::OpenFileDialog => {
                    self.trigger_native_open_dialog();
                    needs_repaint = true;
                }
                LayerUiAction::OpenPath(path) => {
                    self.layers_ui.clear_error();
                    self.open_local_path(&path, true, None);
                    needs_repaint = true;
                }
                LayerUiAction::ToggleVisibility(id, visible) => {
                    let changed = self.scene.set_overlay_visible(id, visible);
                    needs_repaint |= changed;
                    session_dirty |= changed;
                }
                LayerUiAction::SetOpacity(id, opacity) => {
                    let changed = self.scene.set_overlay_opacity(id, opacity);
                    needs_repaint |= changed;
                    session_dirty |= changed;
                }
                LayerUiAction::MoveUp(id) => {
                    let changed = self.scene.move_overlay_up(id);
                    needs_repaint |= changed;
                    session_dirty |= changed;
                }
                LayerUiAction::MoveDown(id) => {
                    let changed = self.scene.move_overlay_down(id);
                    needs_repaint |= changed;
                    session_dirty |= changed;
                }
                LayerUiAction::OpenCrsPicker(id, name) => {
                    self.crs_picker_ui.open_for_layer(id, name);
                    needs_repaint = true;
                }
                LayerUiAction::OpenCrsPickerForPendingPath(path) => {
                    self.crs_picker_ui.open_for_pending_path(path);
                    needs_repaint = true;
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
                    let changed = self.scene.remove_overlay(id);
                    needs_repaint |= changed;
                    session_dirty |= changed;
                }
                LayerUiAction::ToggleVectorVisibility(id, visible) => {
                    let changed = self.scene.set_vector_visible(id, visible);
                    needs_repaint |= changed;
                    session_dirty |= changed;
                }
                LayerUiAction::SetVectorOpacity(id, opacity) => {
                    let changed = self.scene.set_vector_opacity(id, opacity);
                    needs_repaint |= changed;
                    session_dirty |= changed;
                }
                LayerUiAction::RemoveVector(id) => {
                    let changed = self.scene.remove_vector(id);
                    needs_repaint |= changed;
                    session_dirty |= changed;
                }
            }
        }
        if session_dirty {
            self.persist_session_snapshot(false);
        }
        needs_repaint
    }

    fn apply_crs_picker_action(&mut self, action: CrsPickerAction) -> bool {
        match action {
            CrsPickerAction::ApplyToLayer(id, crs_override) => {
                match self.scene.override_overlay_crs(id, crs_override) {
                    Ok(bounds) => {
                        self.layers_ui.clear_error();
                        if let Some(b) = bounds {
                            self.focus_footprint(b);
                        }
                        self.persist_session_snapshot(false);
                    }
                    Err(err) => {
                        self.layers_ui.push_error(err);
                    }
                }
                true
            }
            CrsPickerAction::OpenPathWithCrs(path, crs_override) => {
                self.layers_ui.clear_error();
                self.open_local_path(&path, true, Some(crs_override));
                true
            }
        }
    }

    fn apply_basemap_action(&mut self, action: BasemapUiAction) -> bool {
        let mut needs_repaint = false;
        match action {
            BasemapUiAction::SelectBasemap(selection) => {
                needs_repaint |= self
                    .storage
                    .mutate_settings(|s| s.active_basemap = selection);
                needs_repaint |= self.scene.sync_basemap_settings(self.storage.settings());
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

                    if let Some(clicked_pt) = self.viewport.handle_viewport_input(
                        ctx,
                        &mut self.camera,
                        &response,
                        rect,
                        height_pt,
                        self.measure_ui.is_active(),
                    ) {
                        self.measure_ui.add_point(clicked_pt);
                    }

                    self.scene
                        .set_transient_vector(self.measure_ui.draw_vector_layer());

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

        // Keyboard pan (`WASD`/Arrows), zoom (`+`/`-`), heading (`Q`/`E`), and
        // North reset (`N`) when no text field holds focus.
        let mut active = self
            .viewport
            .handle_keyboard_navigation(ctx, &mut self.camera, dt_s);

        // Handle OS drag-and-drop of local GeoTIFF/PNG/JPEG/GeoJSON files.
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
                self.open_local_path(&path, true, None);
            }
            active = true;
        }
        active
    }

    fn apply_toolbar_actions(&mut self, actions: Vec<ToolbarAction>) -> bool {
        let mut needs_repaint = false;
        for action in actions {
            match action {
                ToolbarAction::OpenFileDialog => {
                    self.trigger_native_open_dialog();
                    needs_repaint = true;
                }
                ToolbarAction::OpenPathPrompt => {
                    self.layers_ui.show_open_prompt();
                    needs_repaint = true;
                }
                ToolbarAction::OpenRecentFile(path) => {
                    self.layers_ui.clear_error();
                    self.open_local_path(&path, true, None);
                    needs_repaint = true;
                }
                ToolbarAction::ToggleLayersPanel => {
                    self.layers_ui.toggle_open();
                    needs_repaint = true;
                }
                ToolbarAction::ToggleMeasureTool => {
                    self.measure_ui.toggle_open();
                    needs_repaint = true;
                }
                ToolbarAction::SaveBookmark(raw_name) => {
                    let (lat_deg, lon_deg) = self.camera.center();
                    let name = if raw_name.is_empty() {
                        hud::format_lat_lon(lat_deg, lon_deg)
                    } else {
                        raw_name
                    };
                    let bm = CameraBookmark {
                        name,
                        lat_deg,
                        lon_deg,
                        height_m: self.camera.height_m(),
                        heading_deg: self.camera.heading_deg(),
                    };
                    let _ = self.storage.mutate_session(|s| s.bookmarks.push(bm));
                    needs_repaint = true;
                }
                ToolbarAction::GoToBookmark(bm) => {
                    self.camera
                        .fly_to_pose(bm.lat_deg, bm.lon_deg, bm.height_m, bm.heading_deg);
                    needs_repaint = true;
                }
                ToolbarAction::DeleteBookmark(idx) => {
                    let _ = self.storage.mutate_session(|s| {
                        if idx < s.bookmarks.len() {
                            s.bookmarks.remove(idx);
                        }
                    });
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
        // Drain native file dialogs & overlay I/O events BEFORE `begin_frame`.
        let mut needs_repaint = self.drain_file_dialog_events();
        needs_repaint |= self.drain_overlay_io_events();
        self.scene.begin_frame(&self.device, &self.queue);

        if self.storage.poll_clear_completed() {
            self.scene.retry_live_and_reconcile();
            needs_repaint = true;
        }

        needs_repaint |= self.handle_global_input_and_drops(&ctx);

        let basemap_status = self.scene.basemap_status();
        let layer_summaries = self.scene.layer_summaries();
        let vector_summaries = self.scene.vector_summaries();
        let toolbar_actions = toolbar::show_top_toolbar(
            ui,
            ToolbarInputs {
                settings: self.storage.settings(),
                basemap_status: &basemap_status,
                adapter_label: &self.adapter_label,
                layer_count: self.scene.total_layer_count(),
                layers_open: self.layers_ui.is_open(),
                measure_active: self.measure_ui.is_active(),
                recent_files: &self.storage.session().recent_files,
                bookmarks: &self.storage.session().bookmarks,
            },
            &mut self.goto_bar,
            &mut self.basemap_ui,
        );
        needs_repaint |= self.apply_toolbar_actions(toolbar_actions);
        needs_repaint |= self.show_settings_if_open(&ctx);

        if let Some(crs_action) = show_crs_picker_window(&ctx, &mut self.crs_picker_ui) {
            needs_repaint |= self.apply_crs_picker_action(crs_action);
        }

        if let Some(MeasureAction::ExportGeoJson) = show_measure_window(&ctx, &mut self.measure_ui)
        {
            let _ = self.dialogs.spawn_save_geojson("measurement.geojson");
            needs_repaint = true;
        }

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

        status_bar::show_status_bar(
            ui,
            cursor_geo,
            cursor_readout.as_ref(),
            topmost_source_px,
            &self.camera,
            self.viewport.viewport_height_px(),
        );

        let layer_actions =
            show_layers_panel(ui, &layer_summaries, &vector_summaries, &mut self.layers_ui);
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
