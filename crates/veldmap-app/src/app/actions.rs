//! UI action dispatch (layers, Cloud/STAC discovery & AOI, CRS picker,
//! unified settings, toolbar, geodesic measurement, and native file dialog
//! completion) for [`VeldmapApp`].

use std::path::{Path, PathBuf};

use eframe::egui;

use crate::layers::{CrsOverride, RemoteBandAssetSpec, RemoteBandMathRequest, RemoteCogRequest};
use crate::measure::{MeasureMode, MeasureSession};
use crate::stac::StacSession;
use crate::storage::session::SavedCameraPose;
use crate::storage::settings::BasemapSelection;
use crate::ui::basemap::BasemapUiAction;
use crate::ui::crs_picker::CrsPickerAction;
use crate::ui::format;
use crate::ui::layers::{EditPersist, LayerUiAction};
use crate::ui::measure::MeasureAction;
use crate::ui::settings::{SettingsAction, SettingsTab};
use crate::ui::stac::StacUiAction;
use crate::ui::toolbar::ToolbarAction;
use crate::ui::workbench::WorkbenchTab;
use crate::vectors::VectorLayerId;

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
        if let Err(err) = self.scene.begin_open_local_path(path, fly_to, crs_override) {
            tracing::warn!(path = %path.display(), error = %err, "Failed to queue local layer open");
            self.panels.layers.push_error(err);
            self.open_workbench_tab(WorkbenchTab::Layers);
        }
    }

    fn trigger_native_open_dialog(&mut self, frame: &eframe::Frame) {
        if let Err(DialogSpawnError::SpawnFailed) = self.dialogs.spawn_open_files(frame) {
            // Fallback to inline path prompt if spawning the dialog thread failed.
            self.panels.layers.show_open_prompt();
            self.open_workbench_tab(WorkbenchTab::Layers);
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
                LayerUiAction::RetryUnavailableSessionLayers => {
                    self.retry_unavailable_session_layers();
                    needs_repaint = true;
                }
                LayerUiAction::HideUnavailableSessionNotice => {
                    self.restore_tracker.hide_notices();
                    needs_repaint = true;
                }
                LayerUiAction::ForgetUnavailableSessionLayers => {
                    if self.restore_tracker.forget_unavailable() {
                        session_dirty = true;
                    }
                    needs_repaint = true;
                }
                LayerUiAction::EditRaster { id, edit, persist } => {
                    let changed = self.scene.edit_overlay(id, edit);
                    needs_repaint |= changed;
                    session_dirty |= match persist {
                        EditPersist::OnChange => changed,
                        EditPersist::OnCommit(commit) => commit,
                    };
                }
                LayerUiAction::SetStream(id, spec) => {
                    match self.scene.set_overlay_stream(id, spec) {
                        Ok(changed) => {
                            needs_repaint |= changed;
                            session_dirty |= changed;
                        }
                        Err(err) => {
                            self.panels.layers.push_error(err);
                            needs_repaint = true;
                        }
                    }
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
                LayerUiAction::EditVector { id, edit, persist } => {
                    let changed = self.scene.edit_vector(id, edit);
                    needs_repaint |= changed;
                    session_dirty |= match persist {
                        EditPersist::OnChange => changed,
                        EditPersist::OnCommit(commit) => commit,
                    };
                }
                LayerUiAction::UseVectorAsStacAoi(id) => {
                    self.use_vector_as_stac_aoi(id);
                    self.open_workbench_tab(WorkbenchTab::CloudStac);
                    needs_repaint = true;
                }
                LayerUiAction::CreateBandMath(spec) => {
                    match self.scene.create_band_math_from_loaded(spec) {
                        Ok((_id, bounds)) => {
                            self.panels.layers.clear_error();
                            if let Some(b) = bounds {
                                self.focus_footprint(b);
                            }
                        }
                        Err(err) => {
                            self.panels.layers.push_error(err);
                        }
                    }
                    needs_repaint = true;
                }
            }
        }
        if session_dirty {
            self.persist_session_snapshot(false, &[]);
        }
        needs_repaint
    }

    fn use_vector_as_stac_aoi(&mut self, id: VectorLayerId) {
        if let Some(geom) = self.scene.vectors().first_polygon_geometry(id) {
            let label = self
                .scene
                .vectors()
                .layer_name(id)
                .map_or_else(|| "GeoJSON AOI".to_owned(), str::to_owned);
            self.deactivate_draw_aoi();
            self.panels.stac.clear_copy_notice();
            self.scene.mutate_stac(|s| s.set_aoi_geometry(geom, label));
        }
    }

    pub(super) fn apply_stac_actions(
        &mut self,
        actions: Vec<StacUiAction>,
        ctx: &egui::Context,
    ) -> bool {
        let mut needs_repaint = false;
        for action in actions {
            match action {
                StacUiAction::Search(request) => {
                    let stac_settings = &self.storage.settings().stac;
                    let res = self
                        .scene
                        .mutate_stac(|s| s.begin_search(&request, stac_settings));
                    if let Err(err) = res {
                        self.scene.mutate_stac(|s| s.set_error(err));
                    }
                    needs_repaint = true;
                }
                StacUiAction::CancelSearch => {
                    self.scene.mutate_stac(StacSession::cancel);
                    needs_repaint = true;
                }
                StacUiAction::ClearResults => {
                    self.scene.mutate_stac(StacSession::clear_results);
                    needs_repaint = true;
                }
                StacUiAction::DismissError => {
                    self.scene.mutate_stac(StacSession::clear_error);
                    needs_repaint = true;
                }
                StacUiAction::CaptureViewportAoi => {
                    if let Some(bbox) = self.viewport.viewport_bbox(&self.camera) {
                        self.deactivate_draw_aoi();
                        self.scene
                            .mutate_stac(|s| s.set_aoi_bbox(bbox, "Viewport".to_owned()));
                    } else {
                        self.scene.mutate_stac(|s| {
                            s.set_error(
                                "Zoom onto the globe before capturing Viewport AOI.".to_owned(),
                            );
                        });
                    }
                    needs_repaint = true;
                }
                StacUiAction::ToggleDrawAoiBox => {
                    self.set_draw_aoi_active(!self.panels.draw_aoi_active);
                    needs_repaint = true;
                }
                StacUiAction::UseVectorAoi(id) => {
                    self.use_vector_as_stac_aoi(id);
                    needs_repaint = true;
                }
                StacUiAction::ClearAoi => {
                    self.deactivate_draw_aoi();
                    self.scene.mutate_stac(StacSession::clear_aoi);
                    needs_repaint = true;
                }
                StacUiAction::CopyAoiGeoJson => {
                    if let Some(json) = self.scene.stac().export_aoi_geojson() {
                        ctx.copy_text(json);
                        self.panels
                            .stac
                            .set_copy_notice("Copied AOI GeoJSON.".to_owned());
                        needs_repaint = true;
                    }
                }
                StacUiAction::CopyAoiWkt => {
                    if let Some(wkt) = self.scene.stac().export_aoi_wkt() {
                        ctx.copy_text(wkt);
                        self.panels
                            .stac
                            .set_copy_notice("Copied AOI WKT.".to_owned());
                        needs_repaint = true;
                    }
                }
                StacUiAction::SelectItem(idx) => {
                    needs_repaint |= self.scene.mutate_stac(|s| s.set_selected(Some(idx)));
                }
                StacUiAction::HoverItem(idx) => {
                    needs_repaint |= self.scene.mutate_stac(|s| s.set_hovered(idx));
                }
                StacUiAction::ZoomToExtent(bounds) => {
                    self.focus_footprint(bounds);
                    needs_repaint = true;
                }
                StacUiAction::StreamCog {
                    url,
                    title,
                    auth_mode,
                    external_bands,
                    is_scl,
                } => {
                    let raw_url = url.into_inner();
                    let (final_url, headers) = self.scene.stac().resolve_cog_auth(
                        &raw_url,
                        auth_mode,
                        &self.storage.settings().stac,
                    );
                    if let Err(err) = self.scene.begin_open_remote_cog(RemoteCogRequest {
                        url: final_url,
                        title,
                        extra_headers: headers,
                        external_bands,
                        is_scl,
                        fly_to: true,
                    }) {
                        self.scene.mutate_stac(|s| s.set_error(err));
                    } else {
                        self.scene.mutate_stac(StacSession::clear_error);
                    }
                    needs_repaint = true;
                }
                StacUiAction::OpenRecipe {
                    title,
                    preset,
                    auth_mode,
                    bands,
                    scl,
                } => {
                    let stac_settings = &self.storage.settings().stac;
                    let resolved_bands: Vec<RemoteBandAssetSpec> = bands
                        .into_iter()
                        .map(|b| {
                            let raw_url = b.href.into_inner();
                            let (final_url, extra_headers) = self.scene.stac().resolve_cog_auth(
                                &raw_url,
                                auth_mode,
                                stac_settings,
                            );
                            RemoteBandAssetSpec {
                                label: b.label,
                                url: final_url,
                                extra_headers,
                                external_bands: b.external_bands,
                                band_index: 0,
                            }
                        })
                        .collect();
                    let resolved_scl = scl.map(|s| {
                        let raw_url = s.href.into_inner();
                        let (final_url, extra_headers) =
                            self.scene
                                .stac()
                                .resolve_cog_auth(&raw_url, auth_mode, stac_settings);
                        RemoteBandAssetSpec {
                            label: s.label,
                            url: final_url,
                            extra_headers,
                            external_bands: s.external_bands,
                            band_index: 0,
                        }
                    });
                    if let Err(err) =
                        self.scene
                            .begin_open_remote_band_math(RemoteBandMathRequest {
                                title,
                                preset,
                                bands: resolved_bands,
                                scl: resolved_scl,
                                fly_to: true,
                            })
                    {
                        self.scene.mutate_stac(|s| s.set_error(err));
                    } else {
                        self.scene.mutate_stac(StacSession::clear_error);
                    }
                    needs_repaint = true;
                }
                StacUiAction::OpenStacSettings => {
                    self.open_settings_tab(SettingsTab::StacCloud);
                    needs_repaint = true;
                }
            }
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

    pub(super) fn apply_settings_action(&mut self, action: SettingsAction) -> bool {
        let mut needs_repaint = false;
        match action {
            SettingsAction::ApplyDraft(draft) => {
                needs_repaint |= self.storage.mutate_settings(|s| draft.apply_into(s));
                let settings = self.storage.settings();
                needs_repaint |= self.scene.sync_settings(
                    settings.basemap_config(),
                    settings.gpu_tile_cache_override_bytes(),
                );
                self.panels.settings.sync_from_settings(settings);
            }
            SettingsAction::ClearDiskCache => {
                self.storage.request_clear_disk_cache();
                needs_repaint = true;
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
                let settings = self.storage.settings();
                needs_repaint |= self.scene.sync_settings(
                    settings.basemap_config(),
                    settings.gpu_tile_cache_override_bytes(),
                );
                self.panels
                    .settings
                    .set_draft_active_basemap(settings.active_basemap);
            }
            BasemapUiAction::ConfigureCustomXyz => {
                self.open_settings_tab(SettingsTab::Basemap);
                self.panels
                    .settings
                    .set_draft_active_basemap(BasemapSelection::CustomXyz);
                needs_repaint = true;
            }
            BasemapUiAction::ToggleSettings => {
                self.toggle_settings();
                needs_repaint = true;
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
                    self.open_workbench_tab(WorkbenchTab::Layers);
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
                    self.toggle_workbench_tab(WorkbenchTab::Layers);
                    needs_repaint = true;
                }
                ToolbarAction::ToggleCloudPanel => {
                    self.toggle_workbench_tab(WorkbenchTab::CloudStac);
                    needs_repaint = true;
                }
                ToolbarAction::ShowUnavailableSessionLayers => {
                    self.restore_tracker.unhide_notices();
                    self.open_workbench_tab(WorkbenchTab::Layers);
                    needs_repaint = true;
                }
                ToolbarAction::ToggleMeasureTool => {
                    self.toggle_measure_tool();
                    needs_repaint = true;
                }
                ToolbarAction::SaveBookmark(raw_name) => {
                    let pose = SavedCameraPose::from_camera(&self.camera);
                    let name = if raw_name.is_empty() {
                        format::format_lat_lon(pose.lat_deg, pose.lon_deg)
                    } else {
                        raw_name
                    };
                    let _ = self.storage.mutate_session(|s| s.add_bookmark(name, pose));
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
                    let _ = self.storage.mutate_session(|s| s.remove_bookmark(idx));
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

    pub(super) fn apply_measure_action(
        &mut self,
        action: MeasureAction,
        frame: &eframe::Frame,
    ) -> bool {
        match action {
            MeasureAction::SetMode(mode) => {
                self.set_measure_mode(mode);
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
                self.set_measure_mode(MeasureMode::Off);
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
