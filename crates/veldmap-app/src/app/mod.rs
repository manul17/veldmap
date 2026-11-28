//! Thin `eframe::App` shell coordinating globe input, GIS HUD layout,
//! collapsible `Layers` / `Cloud / STAC` workbench, split-screen swipe,
//! cartographic PNG export, `session.json` & `.veldmap.json` project
//! persistence, native file dialogs, geodesic measurement, AOI drawing, CRS
//! overrides, and [`SceneState`] frame updates.

mod actions;
mod dialogs;
mod fonts;
mod png_export;
mod restore_tracker;
mod session_sync;
mod swipe;
mod viewport_bridge;
mod viewport_input;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use eframe::egui;
use eframe::egui_wgpu;
use thiserror::Error;
use veldmap_core::geo::{Camera, FootprintBounds};
use veldmap_core::sync::Wakeup;
use veldmap_storage::detect_memory_budget;

use crate::format;
use crate::gpu_tiles::UploadChannelClosed;
use crate::layers::LayerSummary;
use crate::measure::{MeasureMode, MeasureSession};
use crate::scene::{SceneRepaintHint, SceneState};
use crate::stac::{AuthTrust, StacSession};
use crate::storage::AppStorage;
use crate::storage::project::is_project_path;
use crate::storage::session::{
    SavedCameraPose, SavedLayerEntry, SavedSwipeState, SavedVectorEntry,
};
use crate::storage::settings::AppSettings;
use crate::ui;
use crate::ui::crs_picker::{CrsPickerState, show_crs_picker_window};
use crate::ui::export_png::{ExportPngUiAction, ExportPngWindowInputs, show_export_png_window};
use crate::ui::hud;
use crate::ui::layers::LayersPanelState;
use crate::ui::measure::{MeasureUiState, show_measure_window};
use crate::ui::settings::{SettingsTab, SettingsWindowState, show_settings_window};
use crate::ui::stac::StacUiState;
use crate::ui::status_bar::{self, StatusBarInputs};
use crate::ui::toolbar::{self, ToolbarInputs, ToolbarState};
use crate::ui::workbench::{WorkbenchInputs, WorkbenchTab, show_workbench_panel};

use dialogs::DialogHost;
use fonts::build_font_definitions;
use png_export::PngExportController;
use restore_tracker::SessionRestoreTracker;
use swipe::SwipeState;
use viewport_bridge::{GlobePaintCallback, GpuFaultState, install_wgpu_resources, viewport_pixels};
use viewport_input::{ViewportInputCtx, ViewportInteraction, ViewportToolMode};

const STATUS_NOTICE_DURATION: Duration = Duration::from_secs(4);
const PNG_GPU_POLL_INTERVAL: Duration = Duration::from_millis(8);

type DeferredStartupSession = (
    Vec<SavedLayerEntry>,
    Vec<SavedVectorEntry>,
    Option<SavedSwipeState>,
);

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
    settings: SettingsWindowState,
    layers: LayersPanelState,
    stac: StacUiState,
    workbench_open: bool,
    workbench_tab: WorkbenchTab,
    draw_aoi_active: bool,
    crs_picker: CrsPickerState,
    measure: MeasureUiState,
    toolbar: ToolbarState,
}

impl UiPanels {
    fn new(settings: &AppSettings) -> Self {
        Self {
            settings: SettingsWindowState::new(settings),
            layers: LayersPanelState::default(),
            stac: StacUiState::default(),
            workbench_open: false,
            workbench_tab: WorkbenchTab::Layers,
            draw_aoi_active: false,
            crs_picker: CrsPickerState::default(),
            measure: MeasureUiState::default(),
            toolbar: ToolbarState::default(),
        }
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
    deferred_startup_session: Option<DeferredStartupSession>,
    swipe: SwipeState,
    png_export: PngExportController,
    context_menu_geo: Option<(f64, f64)>,
    status_notice: Option<(String, Instant)>,
}

impl Drop for VeldmapApp {
    fn drop(&mut self) {
        // Persist final camera pose & open layer/vector list before closing the
        // serialized disk worker so `session.json` drains durably before exit.
        self.persist_session_snapshot(true, &[]);

        // Deterministic shutdown order:
        // 1. Cancel tokens / signal background workers (`overlays`, `vectors`, `basemap`, `stac`, `decode_shutdown`, `NetContext`, `storage`)
        // 2. Close bounded upload queue (`PendingUploads::close`)
        // 3. Join background PNG export, `TwoLaneDecodePool`, `veldmap-remote-io`, overlay/tile-write & vector I/O workers -> shut down `LiveTilePipeline` -> shut down `NetContext` (`tokio::Runtime`) -> join `DiskCacheWorker`
        self.scene.begin_shutdown();
        self.storage.begin_shutdown();
        self.scene.close_uploads();
        self.png_export.join_background_encode();
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
        let png_export = PngExportController::new(Arc::clone(&wakeup));
        let settings = storage.settings();
        let scene = SceneState::new(
            settings.basemap_config(),
            settings.gpu_tile_cache_override_bytes(),
            storage.disk_cache(),
            wakeup,
            &budget,
        )?;
        let panels = UiPanels::new(settings);

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
            deferred_startup_session: None,
            swipe: SwipeState::default(),
            png_export,
            context_menu_geo: None,
            status_notice: None,
        };

        if startup_files.iter().any(|p| is_project_path(p)) {
            app.deferred_startup_session = Some((
                restored_session.open_layers,
                restored_session.open_vectors,
                restored_session.swipe,
            ));
        } else {
            app.queue_restored_session(
                restored_session.open_layers,
                restored_session.open_vectors,
                restored_session.swipe,
                AuthTrust::LocalSession,
            );
        }
        app.open_user_paths(startup_files, None);

        Ok(app)
    }

    pub(super) fn set_status_notice(&mut self, msg: String) {
        self.status_notice = Some((msg, Instant::now()));
    }

    fn active_status_notice(&mut self) -> Option<&str> {
        if let Some((_, ts)) = &self.status_notice
            && ts.elapsed() >= STATUS_NOTICE_DURATION
        {
            self.status_notice = None;
        }
        self.status_notice.as_ref().map(|(s, _)| s.as_str())
    }

    pub(super) fn default_export_caption(&self) -> String {
        let summaries = self.scene.overlays().layer_summaries();
        if let Some(top) = summaries.iter().rev().find(|l| l.visible) {
            return top.name.clone();
        }
        let settings = self.storage.settings();
        if !settings.gibs_date.trim().is_empty() {
            return format!("VeldMap · {}", settings.gibs_date.trim());
        }
        "VeldMap".to_owned()
    }

    fn focus_footprint(&mut self, bounds: FootprintBounds) {
        self.camera
            .focus(bounds.lat, bounds.lon, bounds.radius_deg.max(0.000_2));
    }

    /// Open the unified Settings modal on `tab`, synchronizing the draft when
    /// newly opened, refreshing disk cache statistics, and clearing any stale
    /// disk-cache notice.
    fn open_settings_tab(&mut self, tab: SettingsTab) {
        self.panels.settings.open_tab(self.storage.settings(), tab);
        self.storage.maybe_refresh_disk_stats(true);
        self.storage.clear_disk_cache_notice();
    }

    /// Toggle the unified Settings modal open or closed from the toolbar,
    /// preserving the last selected tab when reopening.
    fn toggle_settings(&mut self) {
        if self.panels.settings.is_open() {
            self.panels.settings.close();
        } else {
            let tab = self.panels.settings.active_tab();
            self.open_settings_tab(tab);
        }
    }

    /// Currently active exclusive viewport interaction tool.
    fn active_viewport_tool(&self) -> ViewportToolMode {
        if self.panels.draw_aoi_active {
            ViewportToolMode::DrawAoiBox
        } else if self.scene.measure().is_active() {
            ViewportToolMode::Measure
        } else if self.swipe.is_active() {
            ViewportToolMode::Swipe
        } else {
            ViewportToolMode::Navigate
        }
    }

    /// Clear AOI box drawing state, active drag, and live preview without
    /// disturbing an unrelated active measurement session.
    fn deactivate_draw_aoi(&mut self) {
        self.panels.draw_aoi_active = false;
        self.viewport.cancel_aoi_drag();
        self.scene.mutate_stac(StacSession::clear_aoi_preview);
    }

    /// Single transition point enforcing mutual exclusion between `Navigate`,
    /// `DrawAoiBox`, `Measure`, and `Swipe` (preserving remembered
    /// `last_active_mode` when `explicit_measure_mode` is `None`).
    pub(in crate::app) fn set_viewport_tool(
        &mut self,
        tool: ViewportToolMode,
        explicit_measure_mode: Option<MeasureMode>,
    ) {
        if tool != ViewportToolMode::DrawAoiBox {
            self.deactivate_draw_aoi();
        }
        if tool != ViewportToolMode::Measure {
            self.scene.mutate_measure(|m| m.set_mode(MeasureMode::Off));
        }
        if tool != ViewportToolMode::Swipe && self.swipe.is_active() {
            self.swipe.deactivate();
            self.restore_tracker.clear_pending_swipe();
            self.viewport.cancel_swipe_drag();
            self.viewport.set_swipe_bar_rect(None);
            self.persist_session_snapshot(false, &[]);
        }
        match tool {
            ViewportToolMode::Navigate => {}
            ViewportToolMode::DrawAoiBox => {
                self.panels.draw_aoi_active = true;
            }
            ViewportToolMode::Measure => match explicit_measure_mode {
                Some(mode) => {
                    self.scene.mutate_measure(|m| m.set_mode(mode));
                }
                None => {
                    if !self.scene.measure().is_active() {
                        self.scene.mutate_measure(MeasureSession::toggle);
                    }
                }
            },
            ViewportToolMode::Swipe => {
                let summaries = self.scene.overlays().layer_summaries();
                if !summaries.is_empty() {
                    self.restore_tracker.clear_pending_swipe();
                    self.swipe.activate(&summaries);
                    self.persist_session_snapshot(false, &[]);
                }
            }
        }
    }

    /// Toggle split-screen swipe mode cleanly via unified tool transitions.
    pub(super) fn toggle_swipe_tool(&mut self) {
        if self.swipe.is_active() {
            self.set_viewport_tool(ViewportToolMode::Navigate, None);
        } else {
            self.set_viewport_tool(ViewportToolMode::Swipe, None);
        }
    }

    /// Enable or disable interactive AOI box drawing, enforcing mutual
    /// exclusion with geodesic measurement and swipe.
    fn set_draw_aoi_active(&mut self, active: bool) {
        if active {
            self.set_viewport_tool(ViewportToolMode::DrawAoiBox, None);
        } else {
            self.deactivate_draw_aoi();
        }
    }

    /// Toggle geodesic measurement mode while preserving
    /// [`MeasureSession`]'s remembered `last_active_mode`.
    fn toggle_measure_tool(&mut self) {
        if self.scene.measure().is_active() {
            self.set_viewport_tool(ViewportToolMode::Navigate, None);
        } else {
            self.set_viewport_tool(ViewportToolMode::Measure, None);
        }
    }

    /// Set geodesic measurement mode explicitly (`Distance`, `Area`, or `Off`),
    /// enforcing mutual exclusion with AOI box drawing and swipe.
    fn set_measure_mode(&mut self, mode: MeasureMode) {
        if mode == MeasureMode::Off {
            self.set_viewport_tool(ViewportToolMode::Navigate, None);
        } else {
            self.set_viewport_tool(ViewportToolMode::Measure, Some(mode));
        }
    }

    /// Clear transient Cloud/STAC interaction modes (AOI drag, live drag
    /// preview, hovered footprint highlight) whenever leaving the STAC tab or
    /// closing the workbench so orbit navigation and highlights never stick.
    fn cleanup_stac_transient_modes(&mut self) {
        self.deactivate_draw_aoi();
        self.scene.mutate_stac(|s| s.set_hovered(None));
    }

    fn open_workbench_tab(&mut self, tab: WorkbenchTab) {
        let leaving_stac = self.panels.workbench_open
            && self.panels.workbench_tab == WorkbenchTab::CloudStac
            && tab != WorkbenchTab::CloudStac;
        self.panels.workbench_open = true;
        self.panels.workbench_tab = tab;
        if leaving_stac {
            self.cleanup_stac_transient_modes();
        }
    }

    fn close_workbench(&mut self) {
        if self.panels.workbench_open {
            self.panels.workbench_open = false;
            self.cleanup_stac_transient_modes();
        }
    }

    fn toggle_workbench_tab(&mut self, tab: WorkbenchTab) {
        if self.panels.workbench_open && self.panels.workbench_tab == tab {
            self.close_workbench();
        } else {
            self.open_workbench_tab(tab);
        }
    }

    fn show_toolbar(&mut self, ui: &mut egui::Ui, frame: &eframe::Frame) -> bool {
        let settings = self.storage.settings();
        let basemap_status = self.scene.basemap().status_snapshot();
        let toolbar_actions = toolbar::show_top_toolbar(
            ui,
            ToolbarInputs {
                active_basemap: settings.active_basemap,
                custom_xyz_configured: !settings.custom_xyz.url_template.trim().is_empty(),
                show_graticule: settings.show_graticule,
                basemap_status: &basemap_status,
                adapter_label: &self.adapter_label,
                layer_count: self.scene.total_layer_count(),
                raster_layer_count: self.scene.overlays().layer_count(),
                unavailable_layer_count: self.restore_tracker.unavailable_count(),
                layers_open: self.panels.workbench_open
                    && self.panels.workbench_tab == WorkbenchTab::Layers,
                cloud_open: self.panels.workbench_open
                    && self.panels.workbench_tab == WorkbenchTab::CloudStac,
                measure_active: self.scene.measure().is_active(),
                swipe_active: self.swipe.is_active(),
                recent_files: &self.storage.session().recent_files,
                bookmarks: &self.storage.session().bookmarks,
            },
            &mut self.panels.toolbar,
        );
        self.apply_toolbar_actions(toolbar_actions, frame)
    }

    fn show_settings_if_open(&mut self, ctx: &egui::Context) -> bool {
        if !self.panels.settings.is_open() {
            return false;
        }
        self.storage.maybe_refresh_disk_stats(false);
        let disk_status = self.storage.disk_cache_status();
        let settings_actions = show_settings_window(
            ctx,
            self.storage.settings(),
            &disk_status,
            &mut self.panels.settings,
        );
        let mut needs_repaint = false;
        for action in settings_actions {
            needs_repaint |= self.apply_settings_action(action);
        }
        needs_repaint
    }

    fn show_floating_windows(&mut self, ctx: &egui::Context, frame: &eframe::Frame) -> bool {
        let mut needs_repaint = self.show_settings_if_open(ctx);
        if let Some(crs_action) = show_crs_picker_window(ctx, &mut self.panels.crs_picker) {
            needs_repaint |= self.apply_crs_picker_action(crs_action);
        }
        let measure_summary = self.scene.measure().summary();
        if let Some(measure_action) =
            show_measure_window(ctx, measure_summary, &self.panels.measure)
        {
            needs_repaint |= self.apply_measure_action(measure_action, frame);
        }
        let has_colorbar = self.scene.overlays().topmost_colorbar_info().is_some();
        let export_busy = self.png_export.is_busy();
        let export_notice = self.png_export.last_notice().map(str::to_owned);
        let (export_open, export_opts) = self.png_export.modal_state_mut();
        if let Some(ExportPngUiAction::SaveAs) = show_export_png_window(
            ctx,
            export_open,
            export_opts,
            ExportPngWindowInputs {
                has_colorbar,
                busy: export_busy,
                last_notice: export_notice.as_deref(),
            },
        ) {
            self.trigger_export_png_save_dialog(frame);
            needs_repaint = true;
        }
        needs_repaint
    }

    fn show_workbench_if_open(
        &mut self,
        ctx: &egui::Context,
        ui: &mut egui::Ui,
        frame: &eframe::Frame,
    ) -> (bool, Option<Vec<LayerSummary>>) {
        if !self.panels.workbench_open {
            return (false, None);
        }
        let total_layers = self.scene.total_layer_count();
        let fetched_layer_summaries = self.panels.workbench_tab == WorkbenchTab::Layers;
        let layer_summaries = if fetched_layer_summaries {
            self.scene.overlays().layer_summaries()
        } else {
            Vec::new()
        };
        let vector_summaries = self.scene.vectors().summaries();
        let visible_notices = if fetched_layer_summaries {
            self.restore_tracker.visible_notices()
        } else {
            Vec::new()
        };
        let stac_view = self
            .scene
            .stac()
            .view_snapshot(&self.storage.settings().stac);
        let wb_out = show_workbench_panel(
            ui,
            WorkbenchInputs {
                active_tab: self.panels.workbench_tab,
                total_layers,
                layer_summaries: &layer_summaries,
                vector_summaries: &vector_summaries,
                unavailable_notices: &visible_notices,
                stac: stac_view,
                draw_aoi_active: self.panels.draw_aoi_active,
            },
            &mut self.panels.layers,
            &mut self.panels.stac,
        );
        let reusable_summaries =
            (fetched_layer_summaries && wb_out.layer_actions.is_empty()).then_some(layer_summaries);
        let mut needs_repaint = self.apply_layer_actions(wb_out.layer_actions, frame);
        needs_repaint |= self.apply_stac_actions(wb_out.stac_actions, ctx);
        if wb_out.close_requested {
            self.close_workbench();
            needs_repaint = true;
        } else if let Some(tab) = wb_out.selected_tab {
            self.open_workbench_tab(tab);
            needs_repaint = true;
        }
        (needs_repaint, reusable_summaries)
    }

    fn copy_coord_to_clipboard(&mut self, ctx: &egui::Context, label: &str, text: String) {
        ctx.copy_text(text.clone());
        self.set_status_notice(format!("Copied {label}: {text}"));
    }

    fn handle_viewport_collapsed(&mut self, frame: &eframe::Frame) -> bool {
        let was_positive = self.viewport.viewport_rect().is_some();
        self.scene.on_viewport_collapsed();
        self.viewport.clear_viewport();
        let mut needs_repaint = false;
        if let Some(err) = self
            .png_export
            .abort_unsubmitted_capture("Viewport is collapsed; cannot capture screenshot.")
        {
            self.set_status_notice(err);
            needs_repaint = true;
        }
        if was_positive {
            viewport_bridge::release_viewport_resources(frame);
        }
        needs_repaint
    }

    fn show_globe_viewport(
        &mut self,
        ctx: &egui::Context,
        ui: &mut egui::Ui,
        frame: &eframe::Frame,
        cached_layer_summaries: Option<Vec<LayerSummary>>,
    ) -> (Option<egui::Rect>, bool) {
        let mut viewport_rect = None;
        let mut viewport_needs_repaint = false;
        let swipe_reconcile_summaries = if self.swipe.is_active() {
            let s =
                cached_layer_summaries.unwrap_or_else(|| self.scene.overlays().layer_summaries());
            if self.swipe.reconcile_layers(&s) {
                self.persist_session_snapshot(false, &[]);
            }
            s
        } else {
            Vec::new()
        };

        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ui, |ui| {
                let available = ui.available_size();
                let (response, painter) =
                    ui.allocate_painter(available, egui::Sense::click_and_drag());
                let rect = response.rect;

                if rect.is_positive() {
                    viewport_rect = Some(rect);
                    let vp = viewport_pixels(ctx, rect);
                    let ppp = ctx.pixels_per_point().max(0.5);
                    let height_pt = f64::from(rect.height()).max(1.0);
                    self.viewport.record_viewport(rect, vp.height_px, vp.aspect);

                    // Open the Copy Coordinates popup strictly on clean
                    // secondary clicks over the globe (never over painted HUD
                    // cards or conflicting with secondary-drag heading rotation).
                    let open_coord_menu = if response.secondary_clicked()
                        && let Some(pos) = response.interact_pointer_pos()
                        && !self.viewport.is_over_hud(pos)
                    {
                        self.context_menu_geo =
                            self.viewport.probe_screen_pos(&self.camera, rect, pos);
                        Some(egui::SetOpenCommand::Bool(true))
                    } else {
                        None
                    };

                    if let Some((label, text)) = hud::show_coord_context_menu(
                        &response,
                        open_coord_menu,
                        self.context_menu_geo,
                        self.camera.center(),
                    ) {
                        self.copy_coord_to_clipboard(ctx, label, text);
                        viewport_needs_repaint = true;
                    }

                    let swipe_divider_x = self
                        .swipe
                        .is_active()
                        .then(|| vp.swipe_divider_x(self.swipe.split_t(), ppp).0);

                    let tool_mode = self.active_viewport_tool();
                    if let Some(ev) = self.viewport.handle_viewport_input(
                        &mut self.camera,
                        ViewportInputCtx {
                            ctx,
                            response: &response,
                            rect,
                            tool_mode,
                            swipe_divider_x,
                        },
                    ) {
                        self.apply_viewport_event(ev);
                    }

                    let m_per_pt = self.camera.metres_per_pixel(height_pt);
                    let show_graticule = self.storage.settings().show_graticule;
                    let snapshot = self.scene.snapshot_for_frame(
                        self.camera,
                        vp.width_px,
                        vp.height_px,
                        ppp,
                        show_graticule,
                    );
                    let swipe_render = self.swipe.to_render_state();
                    self.png_export.refresh_capture_metrics(
                        ppp,
                        m_per_pt,
                        self.camera.heading_deg(),
                    );
                    let png_readback = self.png_export.take_bridge_for_callback();

                    // Register offscreen globe + 1:1 nearest composite paint callback.
                    painter.add(egui_wgpu::Callback::new_paint_callback(
                        rect,
                        GlobePaintCallback {
                            rect,
                            snapshot,
                            swipe: swipe_render,
                            png_readback,
                        },
                    ));

                    // Paint HiDPI-snapped split-screen swipe curtain divider & handle.
                    if self.swipe.is_active() {
                        let (div_x, _) = vp.swipe_divider_x(self.swipe.split_t(), ppp);
                        let (left_lbl, right_lbl) =
                            self.swipe.side_labels(&swipe_reconcile_summaries);
                        let hovered_or_drag = self.viewport.is_swipe_dragging()
                            || self.viewport.is_hovering_swipe_divider(&response, div_x);
                        hud::paint_swipe_curtain(
                            &painter,
                            rect,
                            div_x,
                            &left_lbl,
                            &right_lbl,
                            hovered_or_drag,
                        );
                    }

                    // Floating dynamic metric scale bar, scientific colorbar, and basemap attribution pill.
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
                    viewport_needs_repaint |= self.handle_viewport_collapsed(frame);
                }
            });

        // Floating Swipe controls bar in its own Background-ordered Area.
        if let Some(rect) = viewport_rect
            && self.swipe.is_active()
        {
            let (action, bar_rect) = hud::show_swipe_controls(
                ctx,
                rect,
                &swipe_reconcile_summaries,
                self.swipe.left_layer(),
                self.swipe.right_layer(),
            );
            self.viewport.set_swipe_bar_rect(Some(bar_rect));
            if let Some(act) = action {
                viewport_needs_repaint |= self.apply_swipe_action(act);
            }
        } else {
            self.viewport.set_swipe_bar_rect(None);
        }

        (viewport_rect, viewport_needs_repaint)
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

        if !ctx.egui_wants_keyboard_input() {
            if (self.panels.draw_aoi_active || self.swipe.is_active())
                && ctx.input(|i| i.key_pressed(egui::Key::Escape))
            {
                self.set_viewport_tool(ViewportToolMode::Navigate, None);
                active = true;
            }
            // `Ctrl+Shift+C` / `Cmd+Shift+C`: quick copy cursor (or center) DD coordinates.
            if ctx.input(|i| {
                (i.modifiers.ctrl || i.modifiers.command)
                    && i.modifiers.shift
                    && i.key_pressed(egui::Key::C)
            }) {
                let (label, (lat, lon)) = match self.viewport.cursor_geo() {
                    Some(pt) => ("Cursor DD", pt),
                    None => ("Center DD", self.camera.center()),
                };
                self.copy_coord_to_clipboard(ctx, label, format::format_signed_dd(lat, lon));
                active = true;
            }
        }

        // Handle OS drag-and-drop of local GeoTIFF/PNG/JPEG/GeoJSON/.veldmap.json files.
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
            return;
        }
        let mut min_delay: Option<Duration> = match scene_hint {
            SceneRepaintHint::After(delay) => Some(delay),
            SceneRepaintHint::None | SceneRepaintHint::Immediate => None,
        };
        if self.png_export.needs_gpu_poll() {
            min_delay =
                Some(min_delay.map_or(PNG_GPU_POLL_INTERVAL, |d| d.min(PNG_GPU_POLL_INTERVAL)));
        }
        if let Some((_, ts)) = &self.status_notice {
            let remaining = STATUS_NOTICE_DURATION.saturating_sub(ts.elapsed());
            min_delay = Some(min_delay.map_or(remaining, |d| d.min(remaining)));
        }
        if let Some(delay) = min_delay {
            ctx.request_repaint_after(delay);
        }
    }
}

impl eframe::App for VeldmapApp {
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        // Drain native file dialogs, PNG readback/encode completions & scene I/O events BEFORE `begin_frame`.
        let mut needs_repaint = self.drain_file_dialog_events(&ctx);
        if let Some(done) = self.png_export.poll(&self.device) {
            match done.result {
                Ok(()) => self.set_status_notice(format!(
                    "Saved PNG {}",
                    crate::display_name_from_path(&done.path)
                )),
                Err(err) => self.set_status_notice(format!("PNG export failed: {err}")),
            }
            needs_repaint = true;
        }
        needs_repaint |= self.drain_scene_io_events();
        self.scene.begin_frame(&self.device, &self.queue);

        if self.storage.poll_clear_completed() {
            self.scene.retry_live_and_reconcile();
            needs_repaint = true;
        }

        needs_repaint |= self.handle_global_input_and_drops(&ctx);
        needs_repaint |= self.show_toolbar(ui, frame);
        needs_repaint |= self.show_system_fault_banner(ui);
        needs_repaint |= self.show_floating_windows(&ctx, frame);

        self.viewport.update_cursor_probe(&ctx, &self.camera);
        let cursor_geo = self.viewport.cursor_geo();
        let swipe_excluded_layer = if self.swipe.is_active()
            && let Some(cursor_pos) = self.viewport.cursor_screen_pos()
            && let Some(rect) = self.viewport.viewport_rect()
        {
            let vp = viewport_pixels(&ctx, rect);
            let (div_x, split_x) = vp.swipe_divider_x(self.swipe.split_t(), ctx.pixels_per_point());
            self.swipe
                .excluded_layer_for_cursor(cursor_pos.x, div_x, split_x, vp.width_px)
        } else {
            None
        };
        let cursor_readout = self
            .scene
            .update_cursor_readout(cursor_geo, swipe_excluded_layer);
        let topmost_source_px = self.scene.overlays().topmost_visible_source_pixel_size_m();
        let camera_before = self.camera;
        let rect_before = self.viewport.viewport_rect();
        let status_notice = self.active_status_notice().map(str::to_owned);

        status_bar::show_status_bar(
            ui,
            StatusBarInputs {
                cursor_geo,
                cursor_readout: cursor_readout.as_ref(),
                topmost_source_px,
                camera: &self.camera,
                viewport_height_px: self.viewport.viewport_height_px(),
                status_notice: status_notice.as_deref(),
            },
        );

        let (wb_repaint, cached_summaries) = self.show_workbench_if_open(&ctx, ui, frame);
        needs_repaint |= wb_repaint;

        let (viewport_rect, viewport_repaint) =
            self.show_globe_viewport(&ctx, ui, frame, cached_summaries);
        needs_repaint |= viewport_repaint;
        let scene_hint = self.scene.end_frame();

        // Floating North compass in its own Background-ordered Area (above the
        // globe, beneath floating windows; no pointer bleed).
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
