//! Thin `eframe::App` shell coordinating globe input, GIS HUD layout,
//! storage/session persistence, native file dialogs, geodesic measurement,
//! CRS overrides, and [`SceneState`] frame updates.

mod actions;
mod dialogs;
mod fonts;
mod session_sync;
mod viewport_bridge;
mod viewport_input;

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use eframe::egui;
use eframe::egui_wgpu;
use thiserror::Error;
use veldmap_core::geo::{Camera, FootprintBounds};
use veldmap_core::sync::{Wakeup, lock_unpoisoned};
use veldmap_storage::detect_memory_budget;

use crate::gpu_tiles::UploadChannelClosed;
use crate::scene::{SceneRepaintHint, SceneState};
use crate::storage::AppStorage;
use crate::storage::session::SavedCameraPose;
use crate::storage::settings::AppSettings;
use crate::ui::basemap::BasemapUiState;
use crate::ui::crs_picker::{CrsPickerState, show_crs_picker_window};
use crate::ui::hud;
use crate::ui::layers::{LayersPanelState, show_layers_panel};
use crate::ui::measure::{MeasureUiState, show_measure_window};
use crate::ui::toolbar::{self, ToolbarInputs, ToolbarState};
use crate::ui::{self, status_bar};

use dialogs::DialogHost;
use fonts::build_font_definitions;
use session_sync::SessionRestoreTracker;
use viewport_bridge::{GlobePaintCallback, install_wgpu_resources, viewport_pixels};
use viewport_input::ViewportInteraction;

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

/// Transient UI state for all floating windows, panels, and toolbar inputs.
struct UiPanels {
    basemap: BasemapUiState,
    layers: LayersPanelState,
    crs_picker: CrsPickerState,
    measure: MeasureUiState,
    toolbar: ToolbarState,
}

impl UiPanels {
    fn new(settings: &AppSettings) -> Self {
        Self {
            basemap: BasemapUiState::new(settings),
            layers: LayersPanelState::default(),
            crs_picker: CrsPickerState::default(),
            measure: MeasureUiState::default(),
            toolbar: ToolbarState::default(),
        }
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
struct GpuFaultState {
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

    fn summary(&self) -> Option<(String, bool)> {
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

    fn dismiss(&self) {
        if !self.has_fault.load(Ordering::Acquire) {
            return;
        }
        let mut guard = lock_unpoisoned(&self.inner);
        guard.device_lost = None;
        guard.uncaptured_error = None;
        self.has_fault.store(false, Ordering::Release);
    }
}

pub struct VeldmapApp {
    adapter_label: String,
    device: wgpu::Device,
    queue: wgpu::Queue,
    gpu_faults: Arc<GpuFaultState>,
    camera: Camera,
    scene: SceneState,
    storage: AppStorage,
    dialogs: DialogHost,
    panels: UiPanels,
    viewport: ViewportInteraction,
    restore_tracker: SessionRestoreTracker,
}

impl Drop for VeldmapApp {
    fn drop(&mut self) {
        // Persist final camera pose & open layer/vector list before closing the
        // serialized disk worker so `session.json` drains durably before exit.
        self.persist_session_snapshot(true, &[]);

        // Deterministic shutdown order:
        // 1. Cancel tokens / signal background workers
        // 2. Close bounded upload queue (`PendingUploads::close`)
        // 3. Join shared CPU decode pool -> join overlay & vector I/O workers -> shut down `tokio::Runtime` -> join disk-cache worker
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

        let egui_ctx = cc.egui_ctx.clone();
        let wakeup: Wakeup = Arc::new(move || {
            egui_ctx.request_repaint();
        });

        let gpu_faults = Arc::new(GpuFaultState::default());
        install_wgpu_resources(render_state, &gpu_faults, &wakeup);

        let device = render_state.device.clone();
        let queue = render_state.queue.clone();

        let budget = detect_memory_budget();
        let storage = AppStorage::open(Arc::clone(&wakeup), None);
        let dialogs = DialogHost::new(Arc::clone(&wakeup));
        let scene = SceneState::new(storage.settings(), storage.disk_cache(), wakeup, &budget)?;
        let panels = UiPanels::new(storage.settings());

        let restored_session = storage.session().clone();
        let camera = restored_session
            .last_camera
            .map_or_else(Camera::default, SavedCameraPose::to_camera);

        let mut app = Self {
            adapter_label,
            device,
            queue,
            gpu_faults,
            camera,
            scene,
            storage,
            dialogs,
            panels,
            viewport: ViewportInteraction::default(),
            restore_tracker: SessionRestoreTracker::default(),
        };

        app.queue_restored_session(restored_session.open_layers, restored_session.open_vectors);
        app.open_user_paths(startup_files, None);

        Ok(app)
    }

    fn focus_footprint(&mut self, bounds: FootprintBounds) {
        self.camera
            .focus(bounds.lat, bounds.lon, bounds.radius_deg.max(0.000_2));
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
                        self.scene.measure().is_active(),
                    ) {
                        self.scene.mutate_measure(|m| m.add_point(clicked_pt));
                        self.panels.measure.clear_notice();
                    }

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

                    // Floating dynamic metric scale bar, scientific colorbar, and basemap attribution pill.
                    let m_per_pt = self.camera.metres_per_pixel(height_pt);
                    let scale_rect = hud::paint_scale_bar(&painter, rect, m_per_pt);
                    let attr_rect = self
                        .scene
                        .basemap()
                        .attribution()
                        .and_then(|attr| hud::paint_attribution_badge(&painter, rect, attr));
                    let colorbar_rect = self
                        .scene
                        .overlays()
                        .topmost_colorbar_info()
                        .and_then(|cb| hud::paint_colorbar(&painter, rect, scale_rect, &cb));
                    self.viewport
                        .set_hud_rects(scale_rect, attr_rect, colorbar_rect);
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
            self.open_user_paths(dropped_paths, None);
            active = true;
        }
        active
    }

    fn show_settings_if_open(&mut self, ctx: &egui::Context) -> bool {
        if !self.panels.basemap.is_settings_open() {
            return false;
        }
        self.storage.maybe_refresh_disk_stats(false);
        let disk_status = self.storage.disk_cache_status();
        let settings_actions = ui::basemap::show_settings_window(
            ctx,
            self.storage.settings(),
            &disk_status,
            &mut self.panels.basemap,
        );
        let mut needs_repaint = false;
        for action in settings_actions {
            needs_repaint |= self.apply_basemap_action(action);
        }
        needs_repaint
    }

    fn show_system_fault_banner(&self, ui: &mut egui::Ui) -> bool {
        let gpu_fault = self.gpu_faults.summary();
        let storage_fault = self.storage.persistence_fault_summary();
        let action = ui::show_system_fault_banner(
            ui,
            gpu_fault
                .as_ref()
                .map(|(msg, is_lost)| (msg.as_str(), *is_lost)),
            storage_fault.as_deref(),
        );
        if action.dismiss_gpu {
            self.gpu_faults.dismiss();
        }
        if action.dismiss_storage {
            self.storage.dismiss_persistence_fault();
        }
        action.dismiss_gpu || action.dismiss_storage
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
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        // Drain native file dialogs & scene I/O events BEFORE `begin_frame`.
        let mut needs_repaint = self.drain_file_dialog_events();
        needs_repaint |= self.drain_scene_io_events();
        self.scene.begin_frame(&self.device, &self.queue);

        if self.storage.poll_clear_completed() {
            self.scene.retry_live_and_reconcile();
            needs_repaint = true;
        }

        needs_repaint |= self.handle_global_input_and_drops(&ctx);

        let basemap_status = self.scene.basemap().status_snapshot();
        let toolbar_actions = toolbar::show_top_toolbar(
            ui,
            ToolbarInputs {
                settings: self.storage.settings(),
                basemap_status: &basemap_status,
                adapter_label: &self.adapter_label,
                layer_count: self.scene.total_layer_count(),
                unavailable_layer_count: self.restore_tracker.unavailable_count(),
                layers_open: self.panels.layers.is_open(),
                measure_active: self.scene.measure().is_active(),
                recent_files: &self.storage.session().recent_files,
                bookmarks: &self.storage.session().bookmarks,
            },
            &mut self.panels.toolbar,
            &mut self.panels.basemap,
        );
        needs_repaint |= self.apply_toolbar_actions(toolbar_actions, frame);
        needs_repaint |= self.show_system_fault_banner(ui);
        needs_repaint |= self.show_settings_if_open(&ctx);

        if let Some(crs_action) = show_crs_picker_window(&ctx, &mut self.panels.crs_picker) {
            needs_repaint |= self.apply_crs_picker_action(crs_action);
        }

        if let Some(measure_action) =
            show_measure_window(&ctx, self.scene.measure(), &self.panels.measure)
        {
            needs_repaint |= self.handle_measure_action(measure_action, frame);
        }

        self.viewport.update_cursor_probe(&ctx, &self.camera);
        let cursor_geo = self.viewport.cursor_geo();
        let cursor_readout = self.scene.update_cursor_readout(cursor_geo);
        let topmost_source_px = self.scene.overlays().topmost_visible_source_pixel_size_m();
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

        if self.panels.layers.is_open() {
            let layer_summaries = self.scene.overlays().layer_summaries();
            let vector_summaries = self.scene.vectors().summaries();
            let visible_notices = self.restore_tracker.visible_notices();
            let layer_actions = show_layers_panel(
                ui,
                &layer_summaries,
                &vector_summaries,
                &visible_notices,
                &mut self.panels.layers,
            );
            needs_repaint |= self.apply_layer_actions(layer_actions, frame);
        }

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
