//! Session restoration tracking, unified `session.json` snapshot persistence,
//! and asynchronous scene-I/O event draining for [`VeldmapApp`].

use std::path::{Path, PathBuf};

use veldmap_core::geo::FootprintBounds;

use crate::layers::OverlayIoEvent;
use crate::scene::SceneIoEvent;
use crate::storage::session::{SavedCameraPose, SavedLayerEntry, SavedVectorEntry};
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
}

fn take_by_path<T>(entries: &mut Vec<T>, path: &Path, entry_path: impl Fn(&T) -> &Path) -> bool {
    entries
        .iter()
        .position(|entry| entry_path(entry) == path)
        .map(|pos| entries.remove(pos))
        .is_some()
}

/// Tracks layers and vectors queued from `session.json` at startup whose
/// background open events have not yet drained, so quitting early never wipes
/// persisted session entries.
#[derive(Default)]
pub(super) struct SessionRestoreTracker {
    layers: Vec<SavedLayerEntry>,
    vectors: Vec<SavedVectorEntry>,
}

impl SessionRestoreTracker {
    fn push_layer(&mut self, entry: SavedLayerEntry) {
        self.layers.push(entry);
    }

    fn push_vector(&mut self, entry: SavedVectorEntry) {
        self.vectors.push(entry);
    }

    fn take_layer(&mut self, path: &Path) -> bool {
        take_by_path(&mut self.layers, path, |entry| &entry.path)
    }

    fn take_vector(&mut self, path: &Path) -> bool {
        take_by_path(&mut self.vectors, path, |entry| &entry.path)
    }

    fn take(&mut self, kind: LayerKind, path: &Path) -> bool {
        match kind {
            LayerKind::Raster => self.take_layer(path),
            LayerKind::Vector => self.take_vector(path),
        }
    }

    fn pending_layers(&self) -> &[SavedLayerEntry] {
        &self.layers
    }

    fn pending_vectors(&self) -> &[SavedVectorEntry] {
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
                .overlays()
                .begin_open_file_with_options(
                    &saved.path,
                    false,
                    saved.visible,
                    saved.opacity,
                    saved.crs_override(),
                    saved.style.clone(),
                )
                .is_ok()
            {
                self.restore_tracker.push_layer(saved);
            }
        }
        for saved_vec in open_vectors {
            if self
                .scene
                .vectors()
                .begin_open(&saved_vec.path, false, saved_vec.visible, saved_vec.opacity)
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
                    Some(entry.style),
                )
            })
            .collect();
        open_layers.extend(self.restore_tracker.pending_layers().iter().cloned());

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
        open_vectors.extend(self.restore_tracker.pending_vectors().iter().cloned());

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
        let _ = self.restore_tracker.take(kind, &path);
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
        session_changed: &mut bool,
    ) {
        if self.restore_tracker.take(kind, &path) {
            tracing::info!(
                kind = kind.label(),
                path = %path.display(),
                error = %error,
                "Skipped unavailable layer from previous session"
            );
            *session_changed = true;
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
                SceneIoEvent::Raster(OverlayIoEvent::Failed {
                    path,
                    error,
                    can_assign_crs,
                }) => {
                    self.handle_layer_failed(
                        LayerKind::Raster,
                        path,
                        error,
                        can_assign_crs,
                        &mut session_changed,
                    );
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
                    self.handle_layer_failed(
                        LayerKind::Vector,
                        path,
                        error,
                        false,
                        &mut session_changed,
                    );
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
mod tests {
    use super::*;

    #[test]
    fn session_restore_tracker_tracks_and_consumes_layers_and_vectors_independently() {
        let mut tracker = SessionRestoreTracker::default();
        let raster_a = PathBuf::from("/tmp/a.tif");
        let raster_b = PathBuf::from("/tmp/b.tif");
        let vector_a = PathBuf::from("/tmp/a.geojson");

        tracker.push_layer(SavedLayerEntry::new(raster_a.clone(), true, 1.0, None));
        tracker.push_layer(SavedLayerEntry::new(raster_a.clone(), false, 0.5, None));
        tracker.push_layer(SavedLayerEntry::new(raster_b.clone(), true, 0.8, None));
        tracker.push_vector(SavedVectorEntry {
            path: vector_a.clone(),
            visible: true,
            opacity: 0.7,
        });

        assert_eq!(tracker.pending_layers().len(), 3);
        assert_eq!(tracker.pending_vectors().len(), 1);

        // Taking a non-existent path returns false and leaves queues untouched.
        assert!(!tracker.take_layer(Path::new("/tmp/missing.tif")));
        assert!(!tracker.take_vector(&raster_a));

        // Duplicate paths are consumed FIFO one at a time.
        assert!(tracker.take_layer(&raster_a));
        assert_eq!(tracker.pending_layers().len(), 2);
        assert!(!tracker.pending_layers()[0].visible);
        assert!(tracker.take_layer(&raster_a));
        assert!(!tracker.take_layer(&raster_a));
        assert_eq!(tracker.pending_layers().len(), 1);
        assert_eq!(tracker.pending_layers()[0].path, raster_b);

        // Vector entries are tracked independently of raster entries.
        assert!(tracker.take_vector(&vector_a));
        assert!(!tracker.take_vector(&vector_a));
        assert!(tracker.pending_vectors().is_empty());
        assert_eq!(tracker.pending_layers().len(), 1);
    }
}
