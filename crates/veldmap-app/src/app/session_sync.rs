//! Session restoration tracking, unified `session.json` snapshot persistence,
//! and asynchronous scene-I/O event draining for [`VeldmapApp`].

use std::path::{Path, PathBuf};

use veldmap_core::geo::FootprintBounds;

use crate::display_name_from_path;
use crate::layers::OverlayIoEvent;
use crate::scene::SceneIoEvent;
use crate::storage::session::{SavedCameraPose, SavedLayerEntry, SavedVectorEntry};
use crate::ui::layers::UnavailableLayerNotice;
use crate::vectors::VectorIoEvent;

use super::VeldmapApp;

#[derive(Clone, Copy)]
enum LayerKind {
    Raster,
    Vector,
}

impl LayerKind {
    fn label(self) -> &'static str {
        match self {
            Self::Raster => "local overlay",
            Self::Vector => "GeoJSON vector",
        }
    }

    fn is_vector(self) -> bool {
        matches!(self, Self::Vector)
    }
}

fn take_entry_by_path<T>(
    entries: &mut Vec<T>,
    path: &Path,
    entry_path: impl Fn(&T) -> &Path,
) -> Option<T> {
    entries
        .iter()
        .position(|entry| entry_path(entry) == path)
        .map(|pos| entries.remove(pos))
}

/// Tracks layers and vectors queued from `session.json` at startup whose
/// background open events have not yet drained, plus any restored entries that
/// failed to open (e.g. unmounted drives/NAS), so quitting or saving never
/// silently wipes persisted CRS overrides or stretch styles.
#[derive(Default)]
pub(super) struct SessionRestoreTracker {
    pending_layers: Vec<SavedLayerEntry>,
    pending_vectors: Vec<SavedVectorEntry>,
    unavailable_layers: Vec<SavedLayerEntry>,
    unavailable_vectors: Vec<SavedVectorEntry>,
    unavailable_notices: Vec<UnavailableLayerNotice>,
    notices_hidden: bool,
}

impl SessionRestoreTracker {
    fn push_layer(&mut self, entry: SavedLayerEntry) {
        self.pending_layers.push(entry);
    }

    fn push_vector(&mut self, entry: SavedVectorEntry) {
        self.pending_vectors.push(entry);
    }

    fn record_unavailable_layer(&mut self, entry: SavedLayerEntry, error: String) {
        self.unavailable_notices.push(UnavailableLayerNotice {
            name: display_name_from_path(&entry.path),
            path: entry.path.clone(),
            error,
            is_vector: false,
        });
        self.unavailable_layers.push(entry);
        self.notices_hidden = false;
    }

    fn record_unavailable_vector(&mut self, entry: SavedVectorEntry, error: String) {
        self.unavailable_notices.push(UnavailableLayerNotice {
            name: display_name_from_path(&entry.path),
            path: entry.path.clone(),
            error,
            is_vector: true,
        });
        self.unavailable_vectors.push(entry);
        self.notices_hidden = false;
    }

    fn remove_unavailable_notice(&mut self, kind: LayerKind, path: &Path) {
        let is_vector = kind.is_vector();
        if let Some(pos) = self
            .unavailable_notices
            .iter()
            .position(|n| n.is_vector == is_vector && n.path == path)
        {
            self.unavailable_notices.remove(pos);
        }
    }

    /// Consume a matching entry when a layer opens: prefer `pending_*` first so
    /// duplicate paths in `session.json` consume FIFO; only when no pending
    /// restore matches does a later manual open clear a preserved `unavailable_*`
    /// entry so `session.json` never duplicates it.
    fn on_opened(&mut self, kind: LayerKind, path: &Path) {
        let consumed_pending = match kind {
            LayerKind::Raster => {
                take_entry_by_path(&mut self.pending_layers, path, |e| &e.path).is_some()
            }
            LayerKind::Vector => {
                take_entry_by_path(&mut self.pending_vectors, path, |e| &e.path).is_some()
            }
        };
        if !consumed_pending {
            let removed_unavailable = match kind {
                LayerKind::Raster => {
                    take_entry_by_path(&mut self.unavailable_layers, path, |e| &e.path).is_some()
                }
                LayerKind::Vector => {
                    take_entry_by_path(&mut self.unavailable_vectors, path, |e| &e.path).is_some()
                }
            };
            if removed_unavailable {
                self.remove_unavailable_notice(kind, path);
            }
        }
    }

    /// If `path` belonged to a pending startup restore, move its full saved
    /// entry into `unavailable_*` (preserving CRS override and style) and
    /// return `true`. Returns `false` for non-restore open failures.
    fn on_failed(&mut self, kind: LayerKind, path: &Path, error: &str) -> bool {
        match kind {
            LayerKind::Raster => {
                let Some(entry) = take_entry_by_path(&mut self.pending_layers, path, |e| &e.path)
                else {
                    return false;
                };
                self.record_unavailable_layer(entry, error.to_owned());
                true
            }
            LayerKind::Vector => {
                let Some(entry) = take_entry_by_path(&mut self.pending_vectors, path, |e| &e.path)
                else {
                    return false;
                };
                self.record_unavailable_vector(entry, error.to_owned());
                true
            }
        }
    }

    fn take_unavailable_for_retry(&mut self) -> (Vec<SavedLayerEntry>, Vec<SavedVectorEntry>) {
        self.unavailable_notices.clear();
        self.notices_hidden = false;
        (
            std::mem::take(&mut self.unavailable_layers),
            std::mem::take(&mut self.unavailable_vectors),
        )
    }

    pub(super) fn hide_notices(&mut self) {
        self.notices_hidden = true;
    }

    pub(super) fn unhide_notices(&mut self) {
        self.notices_hidden = false;
    }

    pub(super) fn forget_unavailable(&mut self) -> bool {
        let had_any = !self.unavailable_layers.is_empty() || !self.unavailable_vectors.is_empty();
        self.unavailable_layers.clear();
        self.unavailable_vectors.clear();
        self.unavailable_notices.clear();
        self.notices_hidden = false;
        had_any
    }

    pub(super) fn unavailable_count(&self) -> usize {
        self.unavailable_notices.len()
    }

    pub(super) fn visible_notices(&self) -> &[UnavailableLayerNotice] {
        if self.notices_hidden {
            &[]
        } else {
            &self.unavailable_notices
        }
    }

    fn preserved_layers(&self) -> impl Iterator<Item = &SavedLayerEntry> {
        self.pending_layers
            .iter()
            .chain(self.unavailable_layers.iter())
    }

    fn preserved_vectors(&self) -> impl Iterator<Item = &SavedVectorEntry> {
        self.pending_vectors
            .iter()
            .chain(self.unavailable_vectors.iter())
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
            match self.scene.overlays().begin_open_file_with_options(
                &saved.path,
                false,
                saved.visible,
                saved.opacity,
                saved.crs_override(),
                saved.style.clone(),
            ) {
                Ok(()) => self.restore_tracker.push_layer(saved),
                Err(err) => self.restore_tracker.record_unavailable_layer(saved, err),
            }
        }
        for saved_vec in open_vectors {
            match self.scene.vectors().begin_open(
                &saved_vec.path,
                false,
                saved_vec.visible,
                saved_vec.opacity,
            ) {
                Ok(()) => self.restore_tracker.push_vector(saved_vec),
                Err(err) => self
                    .restore_tracker
                    .record_unavailable_vector(saved_vec, err),
            }
        }
    }

    pub(super) fn retry_unavailable_session_layers(&mut self) {
        let (layers, vectors) = self.restore_tracker.take_unavailable_for_retry();
        self.queue_restored_session(layers, vectors);
    }

    pub(super) fn persist_session_snapshot(
        &mut self,
        include_camera: bool,
        newly_opened: &[PathBuf],
    ) {
        let mut open_layers: Vec<SavedLayerEntry> = self
            .scene
            .overlays()
            .saved_entries_iter()
            .map(|entry| {
                SavedLayerEntry::with_style(
                    entry.path.to_path_buf(),
                    entry.visible,
                    entry.opacity,
                    entry.crs_override,
                    entry.style,
                )
            })
            .collect();
        open_layers.extend(self.restore_tracker.preserved_layers().cloned());

        let mut open_vectors: Vec<SavedVectorEntry> = self
            .scene
            .vectors()
            .saved_entries_iter()
            .map(|entry| SavedVectorEntry {
                path: entry.path.to_path_buf(),
                visible: entry.visible,
                opacity: entry.opacity,
            })
            .collect();
        open_vectors.extend(self.restore_tracker.preserved_vectors().cloned());

        let cam_pose = if include_camera {
            Some(SavedCameraPose::from_camera(&self.camera))
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

    fn handle_layer_opened(
        &mut self,
        kind: LayerKind,
        path: PathBuf,
        bounds: Option<FootprintBounds>,
        fly_to: bool,
        newly_opened_paths: &mut Vec<PathBuf>,
        session_changed: &mut bool,
    ) {
        self.panels.layers.on_layer_opened();
        self.restore_tracker.on_opened(kind, &path);
        newly_opened_paths.push(path);
        *session_changed = true;
        if fly_to && let Some(b) = bounds {
            self.focus_footprint(b);
        }
    }

    fn handle_layer_failed(
        &mut self,
        kind: LayerKind,
        path: PathBuf,
        error: String,
        can_assign_crs: bool,
    ) {
        if self.restore_tracker.on_failed(kind, &path, &error) {
            // Preserve the entry in `unavailable_*` rather than rewriting
            // `session.json`, so temporarily unmounted drives never wipe custom
            // CRS overrides or stretch configurations.
            tracing::warn!(
                kind = kind.label(),
                path = %path.display(),
                error = %error,
                "Saved session layer unavailable; preserving entry in session.json"
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
        }
    }

    pub(super) fn drain_scene_io_events(&mut self) -> bool {
        let events = self.scene.poll_io_events();
        if events.is_empty() {
            return false;
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
                }) => {
                    tracing::info!(?id, path = %path.display(), "Opened local overlay layer");
                    self.handle_layer_opened(
                        LayerKind::Raster,
                        path,
                        bounds,
                        fly_to,
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
                }
                SceneIoEvent::Raster(OverlayIoEvent::Failed {
                    path,
                    error,
                    can_assign_crs,
                }) => {
                    self.handle_layer_failed(LayerKind::Raster, path, error, can_assign_crs);
                }
                SceneIoEvent::Vector(VectorIoEvent::Opened {
                    id,
                    path,
                    bounds,
                    fly_to,
                }) => {
                    tracing::info!(?id, path = %path.display(), "Opened GeoJSON vector layer");
                    self.handle_layer_opened(
                        LayerKind::Vector,
                        path,
                        bounds,
                        fly_to,
                        &mut newly_opened_paths,
                        &mut session_changed,
                    );
                }
                SceneIoEvent::Vector(VectorIoEvent::Failed { path, error }) => {
                    self.handle_layer_failed(LayerKind::Vector, path, error, false);
                }
            }
        }
        if session_changed || !newly_opened_paths.is_empty() {
            self.persist_session_snapshot(false, &newly_opened_paths);
        }
        true
    }
}

#[cfg(test)]
#[path = "session_sync_tests.rs"]
mod tests;
