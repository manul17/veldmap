//! Main `eframe::App` state, GPU tile cache / upload scheduling, cached
//! quadtree draw snapshots, globe input handling, and GIS HUD layout.

use std::sync::Arc;

use eframe::egui;
use eframe::egui_wgpu::{self, CallbackResources, CallbackTrait, ScreenDescriptor};
use thiserror::Error;
use veldmap_core::geo::camera::{Camera, ZOOM_PER_CLICK};
use veldmap_core::geo::coords::parse_location;
use veldmap_core::io::Wakeup;
use veldmap_core::tile::SourceId;

use crate::basemap::BasemapManager;
use crate::basemap_ui::{self, BasemapUiAction, BasemapUiState};
use crate::gpu_tiles::{
    DEFAULT_GPU_TILE_CACHE_BYTES, DEFAULT_UPLOAD_BYTE_BUDGET, DEFAULT_UPLOAD_TIME_BUDGET,
    GpuTileCache, PendingUploads, UploadPassParams,
};
use crate::hud;
use crate::render::{DrawSnapshot, GlobeRenderer, RenderParams};
use crate::settings::BasemapSelection;
use crate::storage::AppStorage;

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SnapshotKey {
    eye_bits: [u64; 3],
    up_bits: [u64; 3],
    width_px: u32,
    height_px: u32,
    ppp_bits: u32,
    basemap: BasemapSelection,
    active_source: SourceId,
    show_graticule: bool,
    scene_generation: u64,
}

impl SnapshotKey {
    #[expect(
        clippy::too_many_arguments,
        reason = "snapshot cache key hashes explicit camera, viewport, basemap, and generation fields"
    )]
    fn new(
        camera: &Camera,
        width_px: u32,
        height_px: u32,
        pixels_per_point: f32,
        basemap: BasemapSelection,
        active_source: SourceId,
        show_graticule: bool,
        scene_generation: u64,
    ) -> Self {
        let eye = camera.eye_ecef();
        let (_, up, _) = camera.basis();
        Self {
            eye_bits: eye.to_array().map(f64::to_bits),
            up_bits: up.to_array().map(f64::to_bits),
            width_px,
            height_px,
            ppp_bits: pixels_per_point.to_bits(),
            basemap,
            active_source,
            show_graticule,
            scene_generation,
        }
    }
}

/// `egui_wgpu` paint callback bridging an immutable `Arc<DrawSnapshot>` into
/// the offscreen [`GlobeRenderer`].
struct GlobePaintCallback {
    rect: egui::Rect,
    snapshot: Arc<DrawSnapshot>,
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
        let Some(renderer) = callback_resources.get_mut::<GlobeRenderer>() else {
            return Vec::new();
        };

        let vp = egui::epaint::ViewportInPixels::from_points(
            &self.rect,
            screen_descriptor.pixels_per_point,
            screen_descriptor.size_in_pixels,
        );
        if vp.width_px <= 0 || vp.height_px <= 0 {
            renderer.sweep_stale_bind_groups();
            return Vec::new();
        }

        renderer.render_if_dirty(
            device,
            queue,
            egui_encoder,
            RenderParams {
                width: vp.width_px as u32,
                height: vp.height_px as u32,
                snapshot: &self.snapshot,
            },
        );
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
    #[error("tile upload completion channel was closed during initialization")]
    UploadChannelClosed,
}

pub struct VeldmapApp {
    adapter_label: String,
    device: wgpu::Device,
    queue: wgpu::Queue,
    camera: Camera,
    gpu_tile_cache: GpuTileCache,
    pending_uploads: PendingUploads,
    storage: AppStorage,
    basemap: BasemapManager,
    basemap_ui: BasemapUiState,
    cached_snapshot: Option<(SnapshotKey, Arc<DrawSnapshot>)>,
    scene_generation: u64,
    cursor_geo: Option<(f64, f64)>,
    last_viewport_rect: Option<egui::Rect>,
    last_compass_rect: Option<egui::Rect>,
    last_viewport_height_px: f64,
    goto_input: String,
    goto_error: Option<String>,
}

impl Drop for VeldmapApp {
    fn drop(&mut self) {
        // Deterministic shutdown order:
        // 1. Cancel tokens / signal background workers
        // 2. Drop completion channel receiver (`PendingUploads::close`)
        // 3. Join CPU decode, bundled & disk-cache threads -> shut down `tokio::Runtime`
        self.basemap.begin_shutdown();
        self.storage.begin_shutdown();
        self.pending_uploads.close();
        self.basemap.finish_shutdown();
        self.storage.finish_shutdown();
    }
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

        let device = render_state.device.clone();
        let queue = render_state.queue.clone();

        let egui_ctx = cc.egui_ctx.clone();
        let wakeup: Wakeup = Arc::new(move || {
            egui_ctx.request_repaint();
        });

        let pending_uploads = PendingUploads::new();
        let done_tx = pending_uploads
            .sender()
            .ok_or(AppError::UploadChannelClosed)?;
        let storage = AppStorage::open(Arc::clone(&wakeup), None);
        let basemap =
            BasemapManager::new(storage.settings(), storage.disk_cache(), done_tx, wakeup);
        let basemap_ui = BasemapUiState::new(storage.settings());

        Ok(Self {
            adapter_label,
            device,
            queue,
            camera: Camera::default(),
            gpu_tile_cache: GpuTileCache::new(DEFAULT_GPU_TILE_CACHE_BYTES),
            pending_uploads,
            storage,
            basemap,
            basemap_ui,
            cached_snapshot: None,
            scene_generation: 1,
            cursor_geo: None,
            last_viewport_rect: None,
            last_compass_rect: None,
            last_viewport_height_px: 800.0,
            goto_input: String::new(),
            goto_error: None,
        })
    }

    fn apply_basemap_actions(&mut self, actions: Vec<BasemapUiAction>) -> (bool, bool) {
        let mut scene_changed = false;
        let mut needs_repaint = false;
        for action in actions {
            match action {
                BasemapUiAction::SelectBasemap(selection) => {
                    scene_changed |= self
                        .storage
                        .mutate_settings(|s| s.active_basemap = selection);
                    scene_changed |= self.basemap.sync_settings(
                        self.storage.settings(),
                        &mut self.gpu_tile_cache,
                        &mut self.pending_uploads,
                    );
                    // Preserve unsaved edits in an open Settings window; only
                    // synchronize the active selection field.
                    self.basemap_ui
                        .set_draft_active_basemap(self.storage.settings().active_basemap);
                }
                BasemapUiAction::ApplyDraft(draft) => {
                    scene_changed |= self.storage.mutate_settings(|s| draft.apply_into(s));
                    scene_changed |= self.basemap.sync_settings(
                        self.storage.settings(),
                        &mut self.gpu_tile_cache,
                        &mut self.pending_uploads,
                    );
                    self.basemap_ui.sync_from_settings(self.storage.settings());
                }
                BasemapUiAction::ClearDiskCache => {
                    self.storage.request_clear_disk_cache();
                    needs_repaint = true;
                }
                BasemapUiAction::ClearDiskCacheNotice => {
                    self.storage.clear_disk_cache_notice();
                }
                BasemapUiAction::RetryLive => {
                    self.basemap.retry_live();
                    self.basemap
                        .reconcile_providers(&self.gpu_tile_cache, &self.pending_uploads);
                    needs_repaint = true;
                }
            }
        }
        (scene_changed, needs_repaint)
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

    fn viewport_pixels(ctx: &egui::Context, rect: egui::Rect) -> (u32, u32, f64) {
        let ppp = ctx.pixels_per_point();
        let screen_rect = ctx.content_rect();
        let screen_px = [
            (screen_rect.width() * ppp).round().max(1.0) as u32,
            (screen_rect.height() * ppp).round().max(1.0) as u32,
        ];
        let vp = egui::epaint::ViewportInPixels::from_points(&rect, ppp, screen_px);
        let width_px = vp.width_px.max(1) as u32;
        let height_px = vp.height_px.max(1) as u32;
        (
            width_px,
            height_px,
            f64::from(width_px) / f64::from(height_px),
        )
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

    /// Return the cached immutable `Arc<DrawSnapshot>` when `(camera, viewport,
    /// basemap, graticule, scene_generation)` is unchanged (avoiding quadtree
    /// traversal on 120 Hz mouse hover), or rebuild and schedule missing tiles.
    fn snapshot_for_frame(
        &mut self,
        width_px: u32,
        height_px: u32,
        pixels_per_point: f32,
    ) -> Arc<DrawSnapshot> {
        let active_source = self.basemap.active_source();
        let selection = self.storage.settings().active_basemap;
        let show_graticule = self.storage.settings().show_graticule;
        let key = SnapshotKey::new(
            &self.camera,
            width_px,
            height_px,
            pixels_per_point,
            selection,
            active_source,
            show_graticule,
            self.scene_generation,
        );
        if let Some((cached_key, snapshot)) = &self.cached_snapshot
            && *cached_key == key
        {
            return Arc::clone(snapshot);
        }

        let textured = self.basemap.is_textured();
        let quadtree_config = self.basemap.quadtree_config();
        let snapshot = Arc::new(DrawSnapshot::build(
            self.camera,
            width_px,
            height_px,
            pixels_per_point,
            show_graticule,
            textured,
            active_source,
            self.scene_generation,
            &quadtree_config,
            textured.then_some(&mut self.gpu_tile_cache),
        ));

        self.basemap.schedule_for_tiles(
            snapshot.visible_tex_ids(),
            &self.gpu_tile_cache,
            &self.pending_uploads,
            1,
        );

        self.cached_snapshot = Some((key, Arc::clone(&snapshot)));
        snapshot
    }
}

impl eframe::App for VeldmapApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();

        // Drain background tile completions and upload coarsest-first within
        // the per-frame VRAM/time budget while protecting active-frame
        // carriers from eviction.
        let active_source = self.basemap.active_source();
        let pinned_sources = [active_source, SourceId::BUNDLED_BASEMAP];
        let protected = self
            .cached_snapshot
            .as_ref()
            .map(|(_, snap)| &snap.carrier_keys);
        let upload_report = self.pending_uploads.drain_and_upload(
            &self.device,
            &self.queue,
            &mut self.gpu_tile_cache,
            UploadPassParams {
                pinned_root_sources: &pinned_sources,
                observed_source: Some(active_source),
                wanted: Some(self.basemap.wanted_tiles()),
                protected,
                byte_budget: DEFAULT_UPLOAD_BYTE_BUDGET,
                time_budget: DEFAULT_UPLOAD_TIME_BUDGET,
            },
        );
        self.basemap.record_upload_report(&upload_report);
        if upload_report.uploaded > 0 {
            self.scene_generation = self.scene_generation.wrapping_add(1);
        }
        // When `uploaded > 0` on a visible viewport, `snapshot_for_frame` will
        // rebuild and call `schedule_for_tiles` -> `reconcile_providers`, so
        // only reconcile here when no snapshot rebuild will occur.
        let viewport_hidden = self.last_viewport_rect.is_none_or(|r| !r.is_positive());
        if (upload_report.uploaded == 0
            && (upload_report.remaining_ready > 0 || upload_report.deferred_protected > 0))
            || (upload_report.uploaded > 0 && viewport_hidden)
        {
            self.basemap
                .reconcile_providers(&self.gpu_tile_cache, &self.pending_uploads);
        }

        let mut needs_repaint = false;
        if self.storage.poll_clear_completed() {
            self.basemap.retry_live();
            self.basemap
                .reconcile_providers(&self.gpu_tile_cache, &self.pending_uploads);
            needs_repaint = true;
        }

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

        // Top toolbar: wraps cleanly at the 640 px minimum window width.
        let mut basemap_changed = false;
        let mut toolbar_actions = Vec::new();
        let mut graticule_toggled = None;
        egui::Panel::top("top_toolbar").show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.strong(veldmap_core::APP_NAME);
                ui.separator();

                ui.label("Go to:");
                let text_edit = egui::TextEdit::singleline(&mut self.goto_input)
                    .desired_width(170.0)
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
                let basemap_status = self.basemap.status_snapshot();
                toolbar_actions = basemap_ui::show_toolbar_controls(
                    ui,
                    self.storage.settings(),
                    &basemap_status,
                    &mut self.basemap_ui,
                );

                ui.separator();
                let mut show_graticule = self.storage.settings().show_graticule;
                if ui.checkbox(&mut show_graticule, "Graticule").changed() {
                    graticule_toggled = Some(show_graticule);
                }

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

                ui.separator();
                ui.add(
                    egui::Label::new(egui::RichText::new(&self.adapter_label).weak()).truncate(),
                )
                .on_hover_text(&self.adapter_label);
            });
        });

        // Apply toolbar actions BEFORE showing Settings so Settings sees fresh
        // active basemap / config_error on the same frame.
        let (tb_changed, tb_repaint) = self.apply_basemap_actions(toolbar_actions);
        basemap_changed |= tb_changed;
        needs_repaint |= tb_repaint;

        if let Some(show_graticule) = graticule_toggled {
            basemap_changed |= self
                .storage
                .mutate_settings(|s| s.show_graticule = show_graticule);
        }

        // Event-driven off-UI-thread SQLite stats refresh (zero idle queries).
        self.storage
            .maybe_refresh_disk_stats(self.basemap_ui.settings_open());

        let disk_ui = self.storage.disk_cache_ui_state();
        let settings_actions = basemap_ui::show_settings_window(
            &ctx,
            self.storage.settings(),
            &disk_ui,
            &mut self.basemap_ui,
        );
        let (st_changed, st_repaint) = self.apply_basemap_actions(settings_actions);
        basemap_changed |= st_changed;
        needs_repaint |= st_repaint;

        if basemap_changed {
            self.scene_generation = self.scene_generation.wrapping_add(1);
        }

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
        } else {
            self.cursor_geo = None;
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
                    let (width_px, height_px, _) = Self::viewport_pixels(&ctx, rect);
                    let height_pt = f64::from(rect.height()).max(1.0);
                    self.last_viewport_height_px = f64::from(height_px);

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

                    let snapshot =
                        self.snapshot_for_frame(width_px, height_px, ctx.pixels_per_point());

                    // Register offscreen globe + 1:1 nearest composite paint callback.
                    painter.add(egui_wgpu::Callback::new_paint_callback(
                        rect,
                        GlobePaintCallback { rect, snapshot },
                    ));

                    // Floating dynamic metric scale bar.
                    let m_per_pt = self.camera.metres_per_pixel(height_pt);
                    hud::paint_scale_bar(&painter, rect, m_per_pt);

                    // Visible basemap attribution pill (NASA GIBS / EOX CC BY 4.0 / Custom).
                    if let Some(attr) = self.basemap.attribution() {
                        basemap_ui::paint_attribution_badge(&painter, rect, attr);
                    }
                } else {
                    // Release cached snapshot references and stale viewport/compass
                    // rects when collapsed/minimized so evicted `Arc<GpuTile>`s are
                    // not pinned and `viewport_hidden` reflects the collapsed state.
                    self.cached_snapshot = None;
                    self.last_viewport_rect = None;
                    self.last_compass_rect = None;
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

        // Request a follow-up repaint only for active flights, camera changes,
        // viewport resizes, basemap/action changes, budget-deferred uploads, or
        // partial upload progress that may have unprotected a deferred tile.
        if self.camera.is_flying()
            || self.camera != camera_before
            || self.last_viewport_rect != rect_before
            || basemap_changed
            || needs_repaint
            || upload_report.remaining_ready > 0
            || (upload_report.uploaded > 0 && upload_report.deferred_protected > 0)
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
        let pinch = zoom_steps_from_input(96.0, ZOOM_PER_CLICK as f32);
        assert!((pinch - 1.0).abs() < 1e-6, "pinch={pinch}");
        assert_eq!(zoom_steps_from_input(f32::NAN, 1.0), 0.0);
        assert_eq!(zoom_steps_from_input(0.0, f32::NAN), 0.0);
    }
}
