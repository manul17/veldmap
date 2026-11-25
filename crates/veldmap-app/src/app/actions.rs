//! UI action dispatch (layers, CRS picker, basemap settings, toolbar,
//! geodesic measurement, and native file dialog completion) for [`VeldmapApp`].

use std::path::{Path, PathBuf};

use crate::layers::CrsOverride;
use crate::measure::{MeasureMode, MeasureSession};
use crate::storage::session::{CameraBookmark, SavedCameraPose};
use crate::ui::basemap::BasemapUiAction;
use crate::ui::crs_picker::CrsPickerAction;
use crate::ui::format;
use crate::ui::layers::LayerUiAction;
use crate::ui::measure::MeasureAction;
use crate::ui::toolbar::ToolbarAction;
use crate::vectors::is_geojson_path;

use super::VeldmapApp;
use super::dialogs::{DialogSpawnError, FileDialogEvent};

impl VeldmapApp {
    /// Open a user-initiated batch of local paths (drag-and-drop, native file
    /// dialog, CLI args, or path prompt), clearing previous layer errors once
    /// per batch so multi-file errors are preserved together.
    pub(super) fn open_user_paths(
        &mut self,
        paths: Vec<PathBuf>,
        crs_override: Option<CrsOverride>,
    ) {
        if paths.is_empty() {
            return;
        }
        self.panels.layers.clear_error();
        for path in paths {
            self.open_local_path(&path, true, crs_override.clone());
        }
    }

    fn open_local_path(&mut self, path: &Path, fly_to: bool, crs_override: Option<CrsOverride>) {
        if is_geojson_path(path) {
            if let Err(err) = self.scene.vectors().begin_open(path, fly_to, true, 1.0) {
                tracing::warn!(path = %path.display(), error = %err, "Failed to queue GeoJSON open");
                self.panels.layers.push_error(err);
            }
            return;
        }

        if let Err(err) = self.scene.overlays().begin_open_file_with_options(
            path,
            fly_to,
            true,
            1.0,
            crs_override,
        ) {
            tracing::warn!(path = %path.display(), error = %err, "Failed to queue overlay open");
            self.panels.layers.push_error(err);
        }
    }

    fn trigger_native_open_dialog(&mut self, frame: &eframe::Frame) {
        if let Err(DialogSpawnError::SpawnFailed) = self.dialogs.spawn_open_files(frame) {
            // Fallback to inline path prompt if spawning the dialog thread failed.
            self.panels.layers.show_open_prompt();
        }
    }

    pub(super) fn drain_file_dialog_events(&mut self) -> bool {
        let events = self.dialogs.poll();
        if events.is_empty() {
            return false;
        }
        for ev in events {
            match ev {
                FileDialogEvent::OpenFiles(paths) => {
                    self.open_user_paths(paths, None);
                }
                FileDialogEvent::GeoJsonSaved { path, result } => match result {
                    Ok(()) => {
                        self.panels
                            .measure
                            .set_notice(format!("Exported to {}", path.display()));
                    }
                    Err(err) => {
                        self.panels
                            .measure
                            .set_notice(format!("Export failed: {err}"));
                    }
                },
            }
        }
        true
    }

    pub(super) fn apply_layer_actions(
        &mut self,
        actions: Vec<LayerUiAction>,
        frame: &eframe::Frame,
    ) -> bool {
        let mut needs_repaint = false;
        let mut session_dirty = false;
        for action in actions {
            match action {
                LayerUiAction::OpenFileDialog => {
                    self.trigger_native_open_dialog(frame);
                    needs_repaint = true;
                }
                LayerUiAction::OpenPath(path) => {
                    self.open_user_paths(vec![path], None);
                    needs_repaint = true;
                }
                LayerUiAction::ToggleVisibility(id, visible) => {
                    let changed = self.scene.set_overlay_visible(id, visible);
                    needs_repaint |= changed;
                    session_dirty |= changed;
                }
                LayerUiAction::SetOpacity(id, opacity, commit) => {
                    let changed = self.scene.set_overlay_opacity(id, opacity);
                    needs_repaint |= changed;
                    session_dirty |= commit;
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
                    let existing = self.scene.overlays().layer_crs_override(id);
                    self.panels.crs_picker.open_for_layer(id, name, existing);
                    needs_repaint = true;
                }
                LayerUiAction::OpenCrsPickerForPendingPath(path) => {
                    self.panels.crs_picker.open_for_pending_path(path);
                    needs_repaint = true;
                }
                LayerUiAction::ZoomToExtent(bounds) => {
                    self.focus_footprint(bounds);
                    needs_repaint = true;
                }
                LayerUiAction::ReloadBypassCache(id) => {
                    if let Err(err) = self.scene.begin_reload_overlay_bypass_cache(id) {
                        tracing::warn!(?id, error = %err, "Failed to queue overlay reload");
                        self.panels.layers.push_error(err);
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
                LayerUiAction::SetVectorOpacity(id, opacity, commit) => {
                    let changed = self.scene.set_vector_opacity(id, opacity);
                    needs_repaint |= changed;
                    session_dirty |= commit;
                }
                LayerUiAction::RemoveVector(id) => {
                    let changed = self.scene.remove_vector(id);
                    needs_repaint |= changed;
                    session_dirty |= changed;
                }
            }
        }
        if session_dirty {
            self.persist_session_snapshot(false, &[]);
        }
        needs_repaint
    }

    pub(super) fn apply_crs_picker_action(&mut self, action: CrsPickerAction) -> bool {
        match action {
            CrsPickerAction::ApplyToLayer(id, crs_override) => {
                match self.scene.override_overlay_crs(id, crs_override) {
                    Ok(bounds) => {
                        self.panels.layers.clear_error();
                        if let Some(b) = bounds {
                            self.focus_footprint(b);
                        }
                        self.persist_session_snapshot(false, &[]);
                    }
                    Err(err) => {
                        self.panels.layers.push_error(err);
                    }
                }
                true
            }
            CrsPickerAction::OpenPathWithCrs(path, crs_override) => {
                self.open_user_paths(vec![path], Some(crs_override));
                true
            }
        }
    }

    pub(super) fn apply_basemap_action(&mut self, action: BasemapUiAction) -> bool {
        let mut needs_repaint = false;
        match action {
            BasemapUiAction::SelectBasemap(selection) => {
                needs_repaint |= self
                    .storage
                    .mutate_settings(|s| s.active_basemap = selection);
                needs_repaint |= self.scene.sync_basemap_settings(self.storage.settings());
                self.panels
                    .basemap
                    .set_draft_active_basemap(self.storage.settings().active_basemap);
            }
            BasemapUiAction::ApplyDraft(draft) => {
                needs_repaint |= self.storage.mutate_settings(|s| draft.apply_into(s));
                needs_repaint |= self.scene.sync_basemap_settings(self.storage.settings());
                self.panels
                    .basemap
                    .sync_from_settings(self.storage.settings());
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

    pub(super) fn apply_toolbar_actions(
        &mut self,
        actions: Vec<ToolbarAction>,
        frame: &eframe::Frame,
    ) -> bool {
        let mut needs_repaint = false;
        for action in actions {
            match action {
                ToolbarAction::OpenFileDialog => {
                    self.trigger_native_open_dialog(frame);
                    needs_repaint = true;
                }
                ToolbarAction::OpenPathPrompt => {
                    self.panels.layers.show_open_prompt();
                    needs_repaint = true;
                }
                ToolbarAction::OpenRecentFile(path) => {
                    self.open_user_paths(vec![path], None);
                    needs_repaint = true;
                }
                ToolbarAction::ClearRecentFiles => {
                    let _ = self.storage.mutate_session(|s| s.recent_files.clear());
                    needs_repaint = true;
                }
                ToolbarAction::ToggleLayersPanel => {
                    self.panels.layers.toggle_open();
                    needs_repaint = true;
                }
                ToolbarAction::ToggleMeasureTool => {
                    self.scene.mutate_measure(MeasureSession::toggle);
                    needs_repaint = true;
                }
                ToolbarAction::SaveBookmark(raw_name) => {
                    let pose = SavedCameraPose::from_camera(&self.camera);
                    let name = if raw_name.is_empty() {
                        format::format_lat_lon(pose.lat_deg, pose.lon_deg)
                    } else {
                        raw_name
                    };
                    let bm = CameraBookmark { name, pose };
                    let _ = self.storage.mutate_session(|s| s.bookmarks.push(bm));
                    needs_repaint = true;
                }
                ToolbarAction::GoToBookmark(bm) => {
                    self.camera.fly_to_pose(
                        bm.pose.lat_deg,
                        bm.pose.lon_deg,
                        bm.pose.height_m,
                        bm.pose.heading_deg,
                    );
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

    pub(super) fn handle_measure_action(
        &mut self,
        action: MeasureAction,
        frame: &eframe::Frame,
    ) -> bool {
        match action {
            MeasureAction::SetMode(mode) => {
                self.scene.mutate_measure(|m| m.set_mode(mode));
                true
            }
            MeasureAction::Undo => {
                self.scene.mutate_measure(MeasureSession::undo);
                self.panels.measure.clear_notice();
                true
            }
            MeasureAction::Clear => {
                self.scene.mutate_measure(MeasureSession::clear);
                self.panels.measure.clear_notice();
                true
            }
            MeasureAction::Close => {
                self.scene.mutate_measure(|m| m.set_mode(MeasureMode::Off));
                true
            }
            MeasureAction::ExportGeoJson => {
                if let Some(payload) = self.scene.measure().export_geojson_string()
                    && let Err(err) =
                        self.dialogs
                            .spawn_save_geojson("measurement.geojson", payload, frame)
                {
                    let msg = match err {
                        DialogSpawnError::Busy => "Another file dialog is already open.",
                        DialogSpawnError::SpawnFailed => {
                            "Unable to open native save dialog on this system."
                        }
                    };
                    self.panels.measure.set_notice(msg.to_owned());
                }
                true
            }
        }
    }
}
