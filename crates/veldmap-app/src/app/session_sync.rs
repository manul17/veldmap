//! Unified `session.json` / `.veldmap.json` snapshot persistence, restored
//! layer queuing, and asynchronous scene-I/O event draining for [`VeldmapApp`].

use std::path::{Path, PathBuf};

use veldmap_core::geo::FootprintBounds;
use veldmap_raster::ExternalBandMeta;

use crate::display_name_from_path;
use crate::layers::{
    BandMathAssetOpenSpec, LayerId, OpenOverlayRequest, OverlayIoEvent, RemoteBandMathRequest,
    RemoteCogRequest, RemoteLayerKind, TimelineCompareMode, TimelineEpochSourceSpec,
    TimelineEpochSpec, TimelineOpenRequest,
};
use crate::measure::MeasureMode;
use crate::scene::SceneIoEvent;
use crate::stac::{
    AuthTrust, build_stac_item_refresh_policy, resolve_persisted_cog_auth,
    sanitize_imported_auth_mode,
};
use crate::storage::project::ProjectFile;
use crate::storage::session::{
    SavedBandAssetSource, SavedCameraPose, SavedLayerEntry, SavedSwipeState, SavedTimelineEntry,
    SavedTimelineEpochSource, SavedVectorEntry,
};
use crate::storage::settings::StacSettings;
use crate::ui::workbench::WorkbenchTab;
use crate::vectors::VectorIoEvent;

use super::VeldmapApp;
use super::restore_tracker::RestoredEntryKind;

fn build_band_math_asset_spec(
    src: &SavedBandAssetSource,
    trust: AuthTrust,
    stac_settings: &StacSettings,
) -> BandMathAssetOpenSpec {
    match src {
        SavedBandAssetSource::Local {
            label,
            path,
            band_index,
            ..
        } => BandMathAssetOpenSpec::Local {
            label: label.clone(),
            path: path.clone(),
            crs_override: src.local_crs_override(),
            band_index: *band_index,
        },
        SavedBandAssetSource::Remote {
            label,
            url,
            external_bands,
            stac_item_href,
            asset_key,
            band_index,
            ..
        } => {
            let auth_mode = src.remote_auth_mode();
            let (final_url, extra_headers) =
                resolve_persisted_cog_auth(url, auth_mode, trust, stac_settings);
            let refresh_policy = build_stac_item_refresh_policy(
                stac_item_href.as_deref(),
                asset_key.as_deref(),
                trust,
                stac_settings,
            );
            BandMathAssetOpenSpec::Remote {
                label: label.clone(),
                url: final_url,
                auth_mode,
                extra_headers,
                external_bands: external_bands.iter().map(ExternalBandMeta::from).collect(),
                stac_item_href: stac_item_href.clone(),
                asset_key: asset_key.clone(),
                refresh_policy,
                band_index: *band_index,
            }
        }
    }
}

fn build_timeline_open_request(
    entry: &SavedTimelineEntry,
    trust: AuthTrust,
    stac_settings: &StacSettings,
) -> TimelineOpenRequest {
    let epochs = entry
        .epochs
        .iter()
        .map(|e| {
            let source = match &e.source {
                SavedTimelineEpochSource::LocalSingle { path, .. } => {
                    TimelineEpochSourceSpec::LocalSingle {
                        path: path.clone(),
                        crs_override: e.source.local_crs_override(),
                    }
                }
                SavedTimelineEpochSource::RemoteSingle {
                    url,
                    external_bands,
                    is_scl,
                    stac_item_href,
                    asset_key,
                    ..
                } => {
                    let auth_mode = e.source.remote_auth_mode();
                    let (final_url, extra_headers) =
                        resolve_persisted_cog_auth(url, auth_mode, trust, stac_settings);
                    let refresh_policy = build_stac_item_refresh_policy(
                        stac_item_href.as_deref(),
                        asset_key.as_deref(),
                        trust,
                        stac_settings,
                    );
                    TimelineEpochSourceSpec::RemoteSingle {
                        url: final_url,
                        auth_mode,
                        extra_headers,
                        external_bands: external_bands.iter().map(ExternalBandMeta::from).collect(),
                        is_scl: *is_scl,
                        stac_item_href: stac_item_href.clone(),
                        asset_key: asset_key.clone(),
                        refresh_policy,
                    }
                }
                SavedTimelineEpochSource::BandMath { bands, scl } => {
                    TimelineEpochSourceSpec::BandMath {
                        bands: bands
                            .iter()
                            .map(|b| build_band_math_asset_spec(b, trust, stac_settings))
                            .collect(),
                        scl: scl
                            .as_ref()
                            .map(|s| build_band_math_asset_spec(s, trust, stac_settings)),
                    }
                }
            };
            TimelineEpochSpec {
                label: e.label.clone(),
                date_label: e.date_label.clone(),
                sort_key: e.sort_key.clone(),
                cloud_cover: e.cloud_cover,
                source,
            }
        })
        .collect();
    TimelineOpenRequest {
        title: entry.title.clone(),
        epochs,
        recipe_preset: entry.recipe_preset(),
        band_math_style: entry.band_math_style.as_ref().map(|s| s.to_spec()),
        single_style: entry.single_style(),
        visible: entry.visible,
        opacity: entry.opacity,
        active_idx: Some(entry.active_idx),
        compare: Some(TimelineCompareMode::from(entry.compare)),
        max_cloud_cover: entry.max_cloud_cover,
        diff_colormap: Some(entry.diff_colormap()),
        diff_invert_colormap: entry.diff_invert_colormap,
        diff_stretch_min: Some(entry.diff_stretch_min),
        diff_stretch_max: Some(entry.diff_stretch_max),
        fly_to: false,
    }
}

struct OpenedLayerParams {
    kind: RestoredEntryKind,
    opened_layer_id: Option<LayerId>,
    local_path: Option<PathBuf>,
    bounds: Option<FootprintBounds>,
    fly_to: bool,
    restore_id: Option<u64>,
}

impl VeldmapApp {
    /// Queue persisted open layers (local rasters, sanitized remote COGs, and
    /// heterogeneous `BandMath` recipes) and vectors strictly off the UI thread.
    pub(super) fn queue_restored_session(
        &mut self,
        open_layers: Vec<SavedLayerEntry>,
        open_vectors: Vec<SavedVectorEntry>,
        swipe: Option<SavedSwipeState>,
        trust: AuthTrust,
    ) {
        self.restore_tracker.set_pending_swipe(swipe, trust);
        let indexed_layers: Vec<(Option<usize>, SavedLayerEntry)> = open_layers
            .into_iter()
            .enumerate()
            .map(|(i, e)| (Some(i), e))
            .collect();
        self.queue_restored_entries(indexed_layers, open_vectors, trust);
    }

    fn queue_restored_entries(
        &mut self,
        open_layers: Vec<(Option<usize>, SavedLayerEntry)>,
        open_vectors: Vec<SavedVectorEntry>,
        trust: AuthTrust,
    ) {
        let stac_settings = self.storage.settings().stac.clone();

        for (layer_idx, saved) in open_layers {
            let restore_id = self.restore_tracker.alloc_restore_id();
            let res = match &saved {
                SavedLayerEntry::LocalRaster(entry) => {
                    self.scene.begin_open_overlay(OpenOverlayRequest {
                        path: &entry.path,
                        fly_to: false,
                        visible: entry.visible,
                        opacity: entry.opacity,
                        crs_override: entry.crs_override(),
                        style: entry.style(),
                        restore_id: Some(restore_id),
                    })
                }
                SavedLayerEntry::RemoteCog(entry) => {
                    let auth_mode = entry.auth_mode();
                    let (final_url, extra_headers) =
                        resolve_persisted_cog_auth(&entry.url, auth_mode, trust, &stac_settings);
                    let refresh_policy = build_stac_item_refresh_policy(
                        entry.stac_item_href.as_deref(),
                        entry.asset_key.as_deref(),
                        trust,
                        &stac_settings,
                    );
                    self.scene.begin_open_remote_cog(RemoteCogRequest {
                        url: final_url,
                        title: entry.title.clone(),
                        auth_mode,
                        extra_headers,
                        external_bands: entry.external_band_metas(),
                        is_scl: entry.is_scl,
                        stac_item_href: entry.stac_item_href.clone(),
                        asset_key: entry.asset_key.clone(),
                        refresh_policy,
                        visible: entry.visible,
                        opacity: entry.opacity,
                        style: entry.style(),
                        fly_to: false,
                        restore_id: Some(restore_id),
                    })
                }
                SavedLayerEntry::BandMath(entry) => {
                    let style_spec = entry.style.to_spec();
                    let bands = entry
                        .bands
                        .iter()
                        .map(|b| build_band_math_asset_spec(b, trust, &stac_settings))
                        .collect();
                    let scl = entry
                        .scl
                        .as_ref()
                        .map(|s| build_band_math_asset_spec(s, trust, &stac_settings));
                    self.scene
                        .begin_open_remote_band_math(RemoteBandMathRequest {
                            title: entry.title.clone(),
                            preset: style_spec.preset,
                            bands,
                            scl,
                            visible: entry.visible,
                            opacity: entry.opacity,
                            style: Some(style_spec),
                            fly_to: false,
                            restore_id: Some(restore_id),
                        })
                }
                SavedLayerEntry::Timeline(entry) => {
                    let req = build_timeline_open_request(entry, trust, &stac_settings);
                    match self.scene.open_timeline(req) {
                        Ok(id) => {
                            self.restore_tracker
                                .push_layer(restore_id, layer_idx, saved);
                            let saved_slot = self.restore_tracker.on_opened(
                                Some(restore_id),
                                Some(id),
                                RestoredEntryKind::Timeline,
                                None,
                            );
                            self.place_restored_layer_in_order(id, saved_slot);
                        }
                        Err(err) => {
                            self.restore_tracker
                                .record_unavailable_layer(layer_idx, saved, err);
                        }
                    }
                    continue;
                }
            };
            match res {
                Ok(()) => self
                    .restore_tracker
                    .push_layer(restore_id, layer_idx, saved),
                Err(err) => self
                    .restore_tracker
                    .record_unavailable_layer(layer_idx, saved, err),
            }
        }

        for saved_vec in open_vectors {
            let restore_id = self.restore_tracker.alloc_restore_id();
            match self.scene.begin_open_vector(
                &saved_vec.path,
                false,
                saved_vec.visible,
                saved_vec.opacity,
                saved_vec.outline_only,
                Some(restore_id),
            ) {
                Ok(()) => self.restore_tracker.push_vector(restore_id, saved_vec),
                Err(err) => self
                    .restore_tracker
                    .record_unavailable_vector(saved_vec, err),
            }
        }
        // Synchronously restored `Timeline` layers may immediately satisfy a
        // restored split-screen swipe layout without emitting a deferred event.
        self.resolve_pending_swipe_if_ready();
    }

    fn resolve_pending_swipe_if_ready(&mut self) {
        if let Some((split_t, left, right)) = self.restore_tracker.try_resolve_pending_swipe()
            && matches!(
                self.active_viewport_tool(),
                super::viewport_input::ViewportToolMode::Navigate
                    | super::viewport_input::ViewportToolMode::Swipe
            )
        {
            self.swipe.restore(split_t, left, right);
        }
    }

    pub(super) fn retry_unavailable_session_layers(&mut self) {
        let trust = self.restore_tracker.auth_trust();
        let (layers, vectors) = self.restore_tracker.take_unavailable_for_retry();
        self.queue_restored_entries(layers, vectors, trust);
    }

    /// Snapshot live + preserved layers, vectors, bookmarks, camera pose, and
    /// swipe layout into a portable [`ProjectFile`] (ready for `.veldmap.json`
    /// export).
    pub(super) fn build_project_file(&self) -> ProjectFile {
        let (open_layers, open_vectors, swipe) = self.collect_workspace_state();
        ProjectFile {
            open_layers,
            open_vectors,
            bookmarks: self.storage.session().bookmarks.clone(),
            last_camera: Some(SavedCameraPose::from_camera(&self.camera)),
            swipe,
            ..ProjectFile::default()
        }
    }

    /// Replace the active workspace with a loaded `.veldmap.json` project file,
    /// enforcing imported-project credential trust boundaries and emitting a
    /// single clean `session.json` snapshot.
    pub(super) fn apply_loaded_project(&mut self, mut project: ProjectFile, source_path: &Path) {
        // Downgrade untrusted cross-origin auth modes before queuing or
        // persisting into `session.json`.
        let stac_settings = self.storage.settings().stac.clone();
        for entry in &mut project.open_layers {
            entry.map_remote_auth_modes(|url, mode| {
                sanitize_imported_auth_mode(url, mode, &stac_settings)
            });
        }

        // Reset transient tools, UI error/picker state, and in-flight opens
        // cleanly without writing intermediate partial session snapshots.
        self.deactivate_draw_aoi();
        self.scene.mutate_measure(|m| {
            m.clear();
            m.set_mode(MeasureMode::Off);
        });
        self.panels.measure.clear_notice();
        self.panels.layers.clear_error();
        self.panels.crs_picker.close();
        self.swipe.reset();
        self.viewport.cancel_swipe_drag();
        self.viewport.set_swipe_bar_rect(None);

        let _ = self.scene.clear_workspace_layers();
        self.restore_tracker
            .reset_for_project(project.swipe, AuthTrust::ImportedProject);
        if let Some(pose) = project.last_camera {
            self.camera = pose.to_camera();
        }
        let indexed_layers: Vec<(Option<usize>, SavedLayerEntry)> = project
            .open_layers
            .into_iter()
            .enumerate()
            .map(|(i, e)| (Some(i), e))
            .collect();
        self.queue_restored_entries(
            indexed_layers,
            project.open_vectors,
            AuthTrust::ImportedProject,
        );

        let (open_layers, open_vectors, swipe) = self.collect_workspace_state();
        let cam_pose = Some(SavedCameraPose::from_camera(&self.camera));
        let bookmarks = project.bookmarks;
        let _ = self.storage.mutate_session(|s| {
            s.open_layers = open_layers;
            s.open_vectors = open_vectors;
            s.bookmarks = bookmarks;
            s.last_camera = cam_pose;
            s.swipe = swipe;
            s.record_recent_file(source_path);
        });
        self.set_status_notice(format!(
            "Loaded project {}",
            display_name_from_path(source_path)
        ));
    }

    fn collect_workspace_state(
        &self,
    ) -> (
        Vec<SavedLayerEntry>,
        Vec<SavedVectorEntry>,
        Option<SavedSwipeState>,
    ) {
        let live_entries: Vec<(LayerId, SavedLayerEntry)> = self
            .scene
            .overlays()
            .saved_entries_iter()
            .map(|(id, entry)| (id, SavedLayerEntry::from_persist(entry)))
            .collect();
        let live_swipe = self.swipe.is_active().then(|| {
            (
                self.swipe.split_t(),
                self.swipe.left_layer(),
                self.swipe.right_layer(),
            )
        });
        let (open_layers, swipe) = self
            .restore_tracker
            .assemble_layers_and_swipe(live_entries, live_swipe);

        let mut open_vectors: Vec<SavedVectorEntry> = self
            .scene
            .vectors()
            .saved_entries_iter()
            .map(|entry| SavedVectorEntry {
                path: entry.path.to_path_buf(),
                visible: entry.visible,
                opacity: entry.opacity,
                outline_only: entry.outline_only,
            })
            .collect();
        open_vectors.extend(self.restore_tracker.preserved_vectors().cloned());

        (open_layers, open_vectors, swipe)
    }

    pub(super) fn persist_session_snapshot(
        &mut self,
        include_camera: bool,
        newly_opened: &[PathBuf],
    ) {
        // While a startup CLI `.veldmap.json` load is still in flight, do not
        // overwrite `session.json`'s saved layers/vectors/swipe with an empty
        // pre-load workspace.
        if self.deferred_startup_session.is_some() {
            if !newly_opened.is_empty() {
                let _ = self.storage.mutate_session(|s| {
                    for path in newly_opened {
                        s.record_recent_file(path);
                    }
                });
            }
            return;
        }
        let (open_layers, open_vectors, swipe) = self.collect_workspace_state();
        let cam_pose = if include_camera {
            Some(SavedCameraPose::from_camera(&self.camera))
        } else {
            self.storage.session().last_camera
        };
        let _ = self.storage.mutate_session(|s| {
            s.open_layers = open_layers;
            s.open_vectors = open_vectors;
            s.last_camera = cam_pose;
            s.swipe = swipe;
            for path in newly_opened {
                s.record_recent_file(path);
            }
        });
    }

    fn place_restored_layer_in_order(&mut self, id: LayerId, saved_slot: Option<usize>) {
        if let Some(slot) = saved_slot {
            let live_ids = self.scene.overlays().layer_ids();
            let target_idx = self
                .restore_tracker
                .insertion_index_for_slot(&live_ids, id, slot);
            let _ = self.scene.move_overlay_to_index(id, target_idx);
        }
    }

    /// Shared post-open bookkeeping for local rasters, local vectors, and
    /// remote COGs / BandMath layers: consumes restore tracking, places the
    /// layer in original painter's order, and optionally focuses its footprint.
    fn finish_opened_layer(
        &mut self,
        params: OpenedLayerParams,
        newly_opened_paths: &mut Vec<PathBuf>,
        session_changed: &mut bool,
    ) {
        let is_local = params.local_path.is_some();
        if is_local {
            self.panels.layers.on_layer_opened();
            if params.restore_id.is_none() {
                self.open_workbench_tab(WorkbenchTab::Layers);
            }
        } else if params.restore_id.is_none() && !self.panels.workbench_open {
            self.open_workbench_tab(WorkbenchTab::Layers);
        }

        let saved_slot = self.restore_tracker.on_opened(
            params.restore_id,
            params.opened_layer_id,
            params.kind,
            params.local_path.as_deref(),
        );
        if let Some(lid) = params.opened_layer_id {
            self.place_restored_layer_in_order(lid, saved_slot);
        }
        if params.restore_id.is_none()
            && let Some(path) = params.local_path
        {
            newly_opened_paths.push(path);
        }
        *session_changed = true;
        if params.fly_to
            && let Some(b) = params.bounds
        {
            self.focus_footprint(b);
        }
    }

    fn handle_layer_failed(
        &mut self,
        kind: RestoredEntryKind,
        path: PathBuf,
        error: String,
        can_assign_crs: bool,
        restore_id: Option<u64>,
    ) {
        if self.restore_tracker.on_failed(restore_id, &error) {
            tracing::warn!(
                kind = kind.label(),
                path = %path.display(),
                error = %error,
                "Saved workspace layer unavailable; preserving entry in session.json"
            );
        } else {
            tracing::warn!(
                kind = kind.label(),
                path = %path.display(),
                error = %error,
                "Failed to open layer"
            );
            self.panels
                .layers
                .note_failed_path(path, error, can_assign_crs);
            self.open_workbench_tab(WorkbenchTab::Layers);
        }
    }

    pub(super) fn drain_scene_io_events(&mut self) -> bool {
        let (events, stac_changed) = self.scene.poll_io_events();
        if events.is_empty() {
            return stac_changed;
        }
        let mut session_changed = false;
        let mut newly_opened_paths: Vec<PathBuf> = Vec::new();
        for event in events {
            match event {
                SceneIoEvent::Raster(OverlayIoEvent::Opened {
                    id,
                    path,
                    bounds,
                    fly_to,
                    restore_id,
                }) => {
                    tracing::info!(?id, path = %path.display(), "Opened local overlay layer");
                    self.finish_opened_layer(
                        OpenedLayerParams {
                            kind: RestoredEntryKind::LocalRaster,
                            opened_layer_id: Some(id),
                            local_path: Some(path),
                            bounds,
                            fly_to,
                            restore_id,
                        },
                        &mut newly_opened_paths,
                        &mut session_changed,
                    );
                }
                SceneIoEvent::Raster(OverlayIoEvent::Reloaded { id }) => {
                    tracing::info!(?id, "Reloaded local overlay layer");
                    self.panels.layers.clear_error();
                }
                SceneIoEvent::Raster(OverlayIoEvent::ReloadFailed {
                    path,
                    error,
                    can_assign_crs,
                }) => {
                    tracing::warn!(
                        path = %path.display(),
                        error = %error,
                        "Failed to reload local overlay layer"
                    );
                    self.panels
                        .layers
                        .note_failed_path(path, error, can_assign_crs);
                    self.open_workbench_tab(WorkbenchTab::Layers);
                }
                SceneIoEvent::Raster(OverlayIoEvent::Failed {
                    path,
                    error,
                    can_assign_crs,
                    restore_id,
                }) => {
                    self.handle_layer_failed(
                        RestoredEntryKind::LocalRaster,
                        path,
                        error,
                        can_assign_crs,
                        restore_id,
                    );
                }
                SceneIoEvent::Raster(OverlayIoEvent::RemoteOpened {
                    id,
                    title,
                    display_url,
                    kind,
                    bounds,
                    fly_to,
                    restore_id,
                }) => {
                    tracing::info!(?id, ?kind, title = %title, url = %display_url, "Opened remote overlay layer");
                    let restored_kind = match kind {
                        RemoteLayerKind::SingleCog => RestoredEntryKind::RemoteCog,
                        RemoteLayerKind::BandMath => RestoredEntryKind::BandMath,
                    };
                    self.finish_opened_layer(
                        OpenedLayerParams {
                            kind: restored_kind,
                            opened_layer_id: Some(id),
                            local_path: None,
                            bounds,
                            fly_to,
                            restore_id,
                        },
                        &mut newly_opened_paths,
                        &mut session_changed,
                    );
                }
                SceneIoEvent::Raster(OverlayIoEvent::RemoteFailed {
                    title,
                    display_url,
                    kind,
                    error,
                    restore_id,
                }) => {
                    if self.restore_tracker.on_failed(restore_id, &error) {
                        tracing::warn!(
                            ?kind,
                            title = %title,
                            url = %display_url,
                            error = %error,
                            "Saved remote overlay layer unavailable; preserving entry in session.json"
                        );
                    } else {
                        tracing::warn!(
                            ?kind,
                            title = %title,
                            url = %display_url,
                            error = %error,
                            "Failed to open remote overlay layer"
                        );
                        let msg = format!("{} '{title}': {error}", kind.label());
                        if self.panels.workbench_open
                            && self.panels.workbench_tab == WorkbenchTab::CloudStac
                        {
                            self.scene.mutate_stac(|s| s.set_error(msg));
                        } else {
                            self.panels.layers.push_error(msg);
                            self.open_workbench_tab(WorkbenchTab::Layers);
                        }
                    }
                }
                SceneIoEvent::Raster(OverlayIoEvent::TimelineOpened { id, bounds, fly_to }) => {
                    tracing::info!(?id, "Timeline initial epoch ready");
                    if fly_to && let Some(b) = bounds {
                        self.focus_footprint(b);
                    }
                    session_changed = true;
                }
                SceneIoEvent::Vector(VectorIoEvent::Opened {
                    id,
                    path,
                    bounds,
                    fly_to,
                    restore_id,
                }) => {
                    tracing::info!(?id, path = %path.display(), "Opened GeoJSON vector layer");
                    self.finish_opened_layer(
                        OpenedLayerParams {
                            kind: RestoredEntryKind::Vector,
                            opened_layer_id: None,
                            local_path: Some(path),
                            bounds,
                            fly_to,
                            restore_id,
                        },
                        &mut newly_opened_paths,
                        &mut session_changed,
                    );
                }
                SceneIoEvent::Vector(VectorIoEvent::Failed {
                    path,
                    error,
                    restore_id,
                }) => {
                    self.handle_layer_failed(
                        RestoredEntryKind::Vector,
                        path,
                        error,
                        false,
                        restore_id,
                    );
                }
            }
        }
        self.resolve_pending_swipe_if_ready();
        if session_changed || !newly_opened_paths.is_empty() {
            self.persist_session_snapshot(false, &newly_opened_paths);
        }
        true
    }
}
