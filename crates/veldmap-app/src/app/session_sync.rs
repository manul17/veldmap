//! Session restoration tracking, unified `session.json` snapshot persistence,
//! and asynchronous file-dialog / overlay-I/O event draining for [`VeldmapApp`].

use std::path::{Path, PathBuf};

use crate::dialogs::FileDialogEvent;
use crate::layers::OverlayIoEvent;
use crate::session::{SavedCameraPose, SavedLayerEntry, SavedVectorEntry};

use super::VeldmapApp;

/// Tracks layers and vectors queued from `session.json` at startup whose
/// background open events have not yet drained, so quitting early never wipes
/// persisted session entries.
#[derive(Default)]
pub(super) struct SessionRestoreTracker {
    layers: Vec<SavedLayerEntry>,
    vectors: Vec<SavedVectorEntry>,
}

impl SessionRestoreTracker {
    pub(super) fn push_layer(&mut self, entry: SavedLayerEntry) {
        self.layers.push(entry);
    }

    pub(super) fn push_vector(&mut self, entry: SavedVectorEntry) {
        self.vectors.push(entry);
    }

    pub(super) fn take_layer(&mut self, path: &Path) -> bool {
        self.layers
            .iter()
            .position(|entry| entry.path == path)
            .map(|pos| self.layers.remove(pos))
            .is_some()
    }

    pub(super) fn take_vector(&mut self, path: &Path) -> bool {
        self.vectors
            .iter()
            .position(|entry| entry.path == path)
            .map(|pos| self.vectors.remove(pos))
            .is_some()
    }

    pub(super) fn pending_layers(&self) -> &[SavedLayerEntry] {
        &self.layers
    }

    pub(super) fn pending_vectors(&self) -> &[SavedVectorEntry] {
        &self.vectors
    }
}

impl VeldmapApp {
    /// Queue persisted open layers/vectors from `session.json` strictly off
    /// the UI thread (avoiding startup `is_file()` stat stalls on network
    /// mounts and synchronous 64 MiB GeoJSON tessellation).
    pub(super) fn queue_restored_session(
        &mut self,
        open_layers: Vec<SavedLayerEntry>,
        open_vectors: Vec<SavedVectorEntry>,
    ) {
        for saved in open_layers {
            if self
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
                self.restore_tracker.push_layer(saved);
            }
        }
        for saved_vec in open_vectors {
            if self
                .scene
                .begin_open_vector_file(
                    &saved_vec.path,
                    false,
                    saved_vec.visible,
                    saved_vec.opacity,
                )
                .is_ok()
            {
                self.restore_tracker.push_vector(saved_vec);
            }
        }
    }

    pub(super) fn persist_session_snapshot(
        &mut self,
        include_camera: bool,
        newly_opened: &[PathBuf],
    ) {
        let mut open_layers = self.scene.overlays().saved_layer_entries();
        open_layers.extend(self.restore_tracker.pending_layers().iter().cloned());
        let mut open_vectors = self.scene.vectors().saved_entries();
        open_vectors.extend(self.restore_tracker.pending_vectors().iter().cloned());
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
            for path in newly_opened {
                s.record_recent_file(path);
            }
        });
    }

    pub(super) fn drain_file_dialog_events(&mut self) -> bool {
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
                FileDialogEvent::GeoJsonSaved { path, result } => match result {
                    Ok(()) => {
                        self.measure_ui
                            .set_notice(format!("Exported to {}", path.display()));
                    }
                    Err(err) => {
                        self.measure_ui.set_notice(format!("Export failed: {err}"));
                    }
                },
            }
        }
        true
    }

    pub(super) fn drain_overlay_io_events(&mut self) -> bool {
        let events = self.scene.poll_overlay_io_events();
        if events.is_empty() {
            return false;
        }
        let mut session_changed = false;
        let mut newly_opened_paths: Vec<PathBuf> = Vec::new();
        for event in events {
            match event {
                OverlayIoEvent::Opened {
                    id,
                    path,
                    bounds,
                    fly_to,
                } => {
                    tracing::info!(?id, path = %path.display(), "Opened local overlay layer");
                    self.layers_ui.on_layer_opened();
                    let _ = self.restore_tracker.take_layer(&path);
                    newly_opened_paths.push(path);
                    session_changed = true;
                    if fly_to && let Some(b) = bounds {
                        self.focus_footprint(b);
                    }
                }
                OverlayIoEvent::Reloaded { id } => {
                    tracing::info!(?id, "Reloaded local overlay layer");
                    self.layers_ui.clear_error();
                }
                OverlayIoEvent::Failed {
                    path,
                    error,
                    can_assign_crs,
                } => {
                    if self.restore_tracker.take_layer(&path) {
                        tracing::info!(
                            path = %path.display(),
                            error = %error,
                            "Skipped unavailable layer from previous session"
                        );
                        session_changed = true;
                    } else {
                        tracing::warn!(path = %path.display(), error = %error, "Overlay I/O failed");
                        self.layers_ui.note_failed_path(path, error, can_assign_crs);
                    }
                }
                OverlayIoEvent::VectorOpened {
                    id,
                    path,
                    bounds,
                    fly_to,
                } => {
                    tracing::info!(?id, path = %path.display(), "Opened GeoJSON vector layer");
                    self.layers_ui.on_layer_opened();
                    let _ = self.restore_tracker.take_vector(&path);
                    newly_opened_paths.push(path);
                    session_changed = true;
                    if fly_to && let Some(b) = bounds {
                        self.focus_footprint(b);
                    }
                }
                OverlayIoEvent::VectorFailed { path, error } => {
                    if self.restore_tracker.take_vector(&path) {
                        tracing::info!(
                            path = %path.display(),
                            error = %error,
                            "Skipped unavailable GeoJSON layer from previous session"
                        );
                        session_changed = true;
                    } else {
                        tracing::warn!(path = %path.display(), error = %error, "Failed to open GeoJSON");
                        self.layers_ui.push_error(error);
                    }
                }
            }
        }
        if session_changed || !newly_opened_paths.is_empty() {
            self.persist_session_snapshot(false, &newly_opened_paths);
        }
        true
    }
}
