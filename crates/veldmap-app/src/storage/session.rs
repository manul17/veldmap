//! Durable workstation session, recent-files, camera bookmarks, swipe layout,
//! and sanitized local/cloud/`BandMath` workspace persistence
//! (`config_dir/session.json` via `write_json_durable_private()`).

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use veldmap_core::geo::Camera;
use veldmap_storage::{
    StorageError, StoragePaths, read_json_or_default_with_backup, write_json_durable_private,
};

use crate::clamp_opacity;

#[cfg(test)]
pub(crate) use super::layers::SavedLocalRasterEntry;
pub(crate) use super::layers::{
    SavedBandAssetSource, SavedLayerEntry, SavedTimelineEntry, SavedTimelineEpochSource,
};
use super::layers::{default_one_f32, default_true};

/// Current on-disk schema version for `session.json` (tagged `open_layers`,
/// sanitized remote COGs & `BandMath`, vector `outline_only`, and `swipe`).
pub(super) const SESSION_SCHEMA_VERSION: u32 = 2;

/// Maximum number of recent files kept in `session.json`.
const MAX_RECENT_FILES: usize = 16;

/// Maximum number of camera bookmarks kept in `session.json`.
pub(crate) const MAX_BOOKMARKS: usize = 64;

fn default_half_f32() -> f32 {
    0.5
}

/// Saved read-only GeoJSON vector layer entry.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct SavedVectorEntry {
    pub(crate) path: PathBuf,
    #[serde(default = "default_true")]
    pub(crate) visible: bool,
    #[serde(default = "default_one_f32")]
    pub(crate) opacity: f32,
    #[serde(default)]
    pub(crate) outline_only: bool,
}

impl Default for SavedVectorEntry {
    fn default() -> Self {
        Self {
            path: PathBuf::new(),
            visible: true,
            opacity: 1.0,
            outline_only: false,
        }
    }
}

/// Persisted split-screen swipe curtain configuration referencing raster layers
/// by index inside `open_layers`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct SavedSwipeState {
    pub(crate) active: bool,
    #[serde(default = "default_half_f32")]
    pub(crate) split_t: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) left_layer_index: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) right_layer_index: Option<usize>,
}

impl Default for SavedSwipeState {
    fn default() -> Self {
        Self {
            active: false,
            split_t: 0.5,
            left_layer_index: None,
            right_layer_index: None,
        }
    }
}

impl SavedSwipeState {
    pub(super) fn normalize(&mut self, layer_count: usize) -> bool {
        if !self.split_t.is_finite() {
            self.split_t = 0.5;
        } else {
            self.split_t = self.split_t.clamp(0.0, 1.0);
        }
        if self.left_layer_index.is_some_and(|i| i >= layer_count) {
            self.left_layer_index = None;
        }
        if self.right_layer_index.is_some_and(|i| i >= layer_count) {
            self.right_layer_index = None;
        }
        self.active && layer_count > 0
    }
}

/// Persisted camera pose (shared by `last_camera` and [`CameraBookmark`]).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct SavedCameraPose {
    pub(crate) lat_deg: f64,
    pub(crate) lon_deg: f64,
    pub(crate) height_m: f64,
    #[serde(default)]
    pub(crate) heading_deg: f64,
}

impl SavedCameraPose {
    pub(crate) fn from_camera(camera: &Camera) -> Self {
        let (lat_deg, lon_deg) = camera.center();
        Self {
            lat_deg,
            lon_deg,
            height_m: camera.height_m(),
            heading_deg: camera.heading_deg(),
        }
    }

    pub(crate) fn to_camera(self) -> Camera {
        let mut camera = Camera::at(self.lat_deg, self.lon_deg, self.height_m);
        if self.heading_deg.abs() > 1e-6 {
            camera.rotate_heading(self.heading_deg.to_radians());
        }
        camera
    }

    pub(super) fn is_valid(&self) -> bool {
        self.lat_deg.is_finite()
            && (-90.0..=90.0).contains(&self.lat_deg)
            && self.lon_deg.is_finite()
            && (-360.0..=360.0).contains(&self.lon_deg)
            && self.height_m.is_finite()
            && self.height_m > 0.0
            && self.heading_deg.is_finite()
    }
}

/// Named camera bookmark.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct CameraBookmark {
    pub(crate) name: String,
    #[serde(flatten)]
    pub(crate) pose: SavedCameraPose,
}

/// Durable workstation session persisted in `session.json`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct AppSession {
    pub(crate) version: u32,
    pub(crate) open_layers: Vec<SavedLayerEntry>,
    pub(crate) open_vectors: Vec<SavedVectorEntry>,
    pub(crate) recent_files: Vec<PathBuf>,
    pub(crate) bookmarks: Vec<CameraBookmark>,
    pub(crate) last_camera: Option<SavedCameraPose>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) swipe: Option<SavedSwipeState>,
}

impl Default for AppSession {
    fn default() -> Self {
        Self {
            version: SESSION_SCHEMA_VERSION,
            open_layers: Vec::new(),
            open_vectors: Vec::new(),
            recent_files: Vec::new(),
            bookmarks: Vec::new(),
            last_camera: None,
            swipe: None,
        }
    }
}

/// Mutable borrow bundle of workspace fields shared between [`AppSession`] and
/// portable [`super::project::ProjectFile`] normalization.
pub(super) struct WorkspaceFieldsMut<'a> {
    pub(super) version: &'a mut u32,
    pub(super) open_layers: &'a mut Vec<SavedLayerEntry>,
    pub(super) open_vectors: &'a mut Vec<SavedVectorEntry>,
    pub(super) bookmarks: &'a mut Vec<CameraBookmark>,
    pub(super) last_camera: &'a mut Option<SavedCameraPose>,
    pub(super) swipe: &'a mut Option<SavedSwipeState>,
}

/// Shared workspace normalization applied identically to both [`AppSession`]
/// and portable [`super::project::ProjectFile`] instances.
pub(super) fn normalize_workspace_fields(fields: WorkspaceFieldsMut<'_>) {
    *fields.version = SESSION_SCHEMA_VERSION;
    fields.open_layers.retain_mut(SavedLayerEntry::sanitize);
    fields
        .open_vectors
        .retain(|v| !v.path.as_os_str().is_empty());
    for vec_layer in fields.open_vectors.iter_mut() {
        vec_layer.opacity = clamp_opacity(vec_layer.opacity);
    }
    fields
        .bookmarks
        .retain(|b| !b.name.trim().is_empty() && b.pose.is_valid());
    fields.bookmarks.truncate(MAX_BOOKMARKS);
    if let Some(cam) = *fields.last_camera
        && !cam.is_valid()
    {
        *fields.last_camera = None;
    }
    let layer_count = fields.open_layers.len();
    if let Some(mut sw) = *fields.swipe {
        if sw.normalize(layer_count) {
            *fields.swipe = Some(sw);
        } else {
            *fields.swipe = None;
        }
    }
}

impl AppSession {
    /// Normalize bounds, sanitize all remote URLs, deduplicate recent files,
    /// and drop invalid entries.
    pub(super) fn normalize(&mut self) {
        normalize_workspace_fields(WorkspaceFieldsMut {
            version: &mut self.version,
            open_layers: &mut self.open_layers,
            open_vectors: &mut self.open_vectors,
            bookmarks: &mut self.bookmarks,
            last_camera: &mut self.last_camera,
            swipe: &mut self.swipe,
        });
        let mut deduped = Vec::with_capacity(self.recent_files.len().min(MAX_RECENT_FILES));
        for p in self.recent_files.drain(..) {
            if !p.as_os_str().is_empty() && !deduped.contains(&p) {
                deduped.push(p);
                if deduped.len() >= MAX_RECENT_FILES {
                    break;
                }
            }
        }
        self.recent_files = deduped;
    }

    /// Push `path` to the front of `recent_files` (deduplicated, capped).
    pub(crate) fn record_recent_file(&mut self, path: &Path) {
        if path.as_os_str().is_empty() {
            return;
        }
        self.recent_files.retain(|p| p != path);
        self.recent_files.insert(0, path.to_path_buf());
        self.recent_files.truncate(MAX_RECENT_FILES);
    }

    /// Append a camera bookmark (bounded to `MAX_BOOKMARKS` during normalize).
    pub(crate) fn add_bookmark(&mut self, name: String, pose: SavedCameraPose) {
        self.bookmarks.push(CameraBookmark { name, pose });
    }

    /// Remove the bookmark at `index`, if in bounds.
    pub(crate) fn remove_bookmark(&mut self, index: usize) {
        if index < self.bookmarks.len() {
            self.bookmarks.remove(index);
        }
    }

    /// Load `session.json` from `paths` (backing up corrupt files to `.corrupt`).
    pub(super) fn load_or_default(paths: &StoragePaths) -> Self {
        let mut session: Self = read_json_or_default_with_backup(&paths.session_path());
        session.normalize();
        session
    }

    /// Persist a normalized copy to `session.json` via durable private atomic write.
    pub(super) fn save(&self, paths: &StoragePaths) -> Result<(), StorageError> {
        let mut copy = self.clone();
        copy.normalize();
        write_json_durable_private(&paths.session_path(), &copy)
    }
}

#[cfg(test)]
#[path = "session_tests.rs"]
mod tests;
