//! Explicit user-saved `.veldmap.json` project files with sanitized remote
//! COG / `BandMath` URLs (zero credentials/presigned query tokens) and
//! same-root relative local path serialization with absolute fallback.

use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};
use veldmap_storage::{StorageError, read_json_limited, write_json_durable};

use super::session::{
    CameraBookmark, SESSION_SCHEMA_VERSION, SavedBandAssetSource, SavedCameraPose, SavedLayerEntry,
    SavedSwipeState, SavedVectorEntry, normalize_workspace_fields,
};

/// File suffix identifying portable VeldMap project files.
pub(crate) const PROJECT_FILE_SUFFIX: &str = ".veldmap.json";

/// Upper size bound for loading a user-selected `.veldmap.json` project file (16 MiB).
const MAX_PROJECT_FILE_BYTES: u64 = 16 * 1024 * 1024;

/// Return whether `path` is a `.veldmap.json` project file (checked before
/// `is_geojson_path`, which matches generic `.json`).
pub(crate) fn is_project_path(path: &Path) -> bool {
    path.file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|name| name.to_ascii_lowercase().ends_with(PROJECT_FILE_SUFFIX))
}

/// Normalize a path lexically (collapsing `.` and resolving non-leading `..`)
/// without touching the filesystem so path relativization works deterministically
/// even for offline/unmounted paths.
fn lexical_clean(path: &Path) -> PathBuf {
    let mut out = Vec::new();
    for comp in path.components() {
        match comp {
            Component::CurDir => {}
            Component::ParentDir => match out.last() {
                Some(Component::Normal(_)) => {
                    out.pop();
                }
                Some(Component::RootDir | Component::Prefix(_)) => {}
                _ => out.push(comp),
            },
            other => out.push(other),
        }
    }
    let mut result = PathBuf::new();
    for comp in out {
        result.push(comp.as_os_str());
    }
    result
}

fn components_equal(a: Component<'_>, b: Component<'_>) -> bool {
    #[cfg(windows)]
    {
        use std::path::Prefix;
        match (a, b) {
            (Component::Prefix(pa), Component::Prefix(pb)) => match (pa.kind(), pb.kind()) {
                (
                    Prefix::Disk(da) | Prefix::VerbatimDisk(da),
                    Prefix::Disk(db) | Prefix::VerbatimDisk(db),
                ) => da.eq_ignore_ascii_case(&db),
                (
                    Prefix::UNC(sa, sha) | Prefix::VerbatimUNC(sa, sha),
                    Prefix::UNC(sb, shb) | Prefix::VerbatimUNC(sb, shb),
                ) => {
                    sa.to_string_lossy().to_lowercase() == sb.to_string_lossy().to_lowercase()
                        && sha.to_string_lossy().to_lowercase()
                            == shb.to_string_lossy().to_lowercase()
                }
                _ => pa
                    .as_os_str()
                    .to_string_lossy()
                    .eq_ignore_ascii_case(&pb.as_os_str().to_string_lossy()),
            },
            (Component::Normal(na), Component::Normal(nb)) => {
                na.to_string_lossy().to_lowercase() == nb.to_string_lossy().to_lowercase()
            }
            _ => a == b,
        }
    }
    #[cfg(not(windows))]
    {
        a == b
    }
}

/// Express `target` relative to `base_dir` when both are absolute paths on the
/// exact same root/drive/UNC prefix, falling back to the cleaned absolute
/// `target` across different drives or when `base_dir` is relative.
pub(crate) fn make_path_relative_if_same_root(base_dir: &Path, target: &Path) -> PathBuf {
    if !base_dir.is_absolute() || !target.is_absolute() {
        return target.to_path_buf();
    }
    let base_clean = lexical_clean(base_dir);
    let target_clean = lexical_clean(target);
    let base_comps: Vec<Component<'_>> = base_clean.components().collect();
    let target_comps: Vec<Component<'_>> = target_clean.components().collect();

    // Verify identical root/prefix (e.g. same Windows drive letter or Unix `/`).
    let mut prefix_len = 0usize;
    while prefix_len < base_comps.len()
        && prefix_len < target_comps.len()
        && matches!(
            base_comps[prefix_len],
            Component::Prefix(_) | Component::RootDir
        )
    {
        if !components_equal(base_comps[prefix_len], target_comps[prefix_len]) {
            return target_clean;
        }
        prefix_len += 1;
    }
    if prefix_len == 0
        || target_comps
            .get(prefix_len)
            .is_some_and(|c| matches!(c, Component::Prefix(_) | Component::RootDir))
    {
        return target_clean;
    }

    let mut common = prefix_len;
    while common < base_comps.len()
        && common < target_comps.len()
        && components_equal(base_comps[common], target_comps[common])
    {
        common += 1;
    }

    let mut rel = PathBuf::new();
    for _ in common..base_comps.len() {
        rel.push("..");
    }
    for comp in &target_comps[common..] {
        rel.push(comp.as_os_str());
    }
    if rel.as_os_str().is_empty() {
        PathBuf::from(".")
    } else {
        rel
    }
}

/// Resolve a `stored` path from a `.veldmap.json` file against `base_dir`
/// (returning absolute paths cleaned as-is, and joining relative paths onto
/// `base_dir`).
pub(crate) fn resolve_project_path(base_dir: &Path, stored: &Path) -> PathBuf {
    if stored.as_os_str().is_empty() || stored.is_absolute() {
        return stored.to_path_buf();
    }
    lexical_clean(&base_dir.join(stored))
}

fn map_layer_paths(layers: &mut [SavedLayerEntry], mut f: impl FnMut(&Path) -> PathBuf) {
    for layer in layers {
        match layer {
            SavedLayerEntry::LocalRaster(e) => {
                e.path = f(&e.path);
            }
            SavedLayerEntry::RemoteCog(_) => {}
            SavedLayerEntry::BandMath(e) => {
                for band in &mut e.bands {
                    if let SavedBandAssetSource::Local { path, .. } = band {
                        *path = f(path);
                    }
                }
                if let Some(SavedBandAssetSource::Local { path, .. }) = &mut e.scl {
                    *path = f(path);
                }
            }
        }
    }
}

/// Portable `.veldmap.json` workspace project file (sanitized URLs, relative
/// local paths on the same drive/root, zero credentials or recent-file history).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct ProjectFile {
    pub(crate) version: u32,
    pub(crate) open_layers: Vec<SavedLayerEntry>,
    pub(crate) open_vectors: Vec<SavedVectorEntry>,
    pub(crate) bookmarks: Vec<CameraBookmark>,
    pub(crate) last_camera: Option<SavedCameraPose>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) swipe: Option<SavedSwipeState>,
}

impl Default for ProjectFile {
    fn default() -> Self {
        Self {
            version: SESSION_SCHEMA_VERSION,
            open_layers: Vec::new(),
            open_vectors: Vec::new(),
            bookmarks: Vec::new(),
            last_camera: None,
            swipe: None,
        }
    }
}

impl ProjectFile {
    /// Normalize bounds, sanitize all remote URLs, and drop invalid entries.
    pub(crate) fn normalize(&mut self) {
        normalize_workspace_fields(
            &mut self.version,
            &mut self.open_layers,
            &mut self.open_vectors,
            &mut self.bookmarks,
            &mut self.last_camera,
            &mut self.swipe,
        );
    }

    /// Save a normalized, sanitized copy to `project_path`, relativizing local
    /// raster and vector paths against `project_path.parent()` (anchoring
    /// relative inputs against the current working directory first) when on the
    /// same root/drive.
    pub(crate) fn save_to(&self, project_path: &Path) -> Result<(), StorageError> {
        let mut copy = self.clone();
        copy.normalize();
        let cwd = std::env::current_dir().ok();
        let anchor = |p: &Path| -> PathBuf {
            if p.is_absolute() {
                p.to_path_buf()
            } else if let Some(ref dir) = cwd {
                dir.join(p)
            } else {
                p.to_path_buf()
            }
        };
        let anchored = anchor(project_path);
        if let Some(base_dir) = anchored.parent().filter(|p| p.is_absolute()) {
            map_layer_paths(&mut copy.open_layers, |p| {
                let abs = anchor(p);
                make_path_relative_if_same_root(base_dir, &abs)
            });
            for v in &mut copy.open_vectors {
                let abs = anchor(&v.path);
                v.path = make_path_relative_if_same_root(base_dir, &abs);
            }
        }
        write_json_durable(project_path, &copy)
    }

    /// Load and normalize a `.veldmap.json` project file, resolving relative
    /// local paths against `project_path.parent()` (anchored to the current
    /// working directory when `project_path` itself is relative).
    pub(crate) fn load_from(project_path: &Path) -> Result<Self, StorageError> {
        let mut project: Self = read_json_limited(project_path, MAX_PROJECT_FILE_BYTES)?;
        let anchored = if project_path.is_absolute() {
            project_path.to_path_buf()
        } else if let Ok(cwd) = std::env::current_dir() {
            cwd.join(project_path)
        } else {
            project_path.to_path_buf()
        };
        if let Some(base_dir) = anchored.parent().filter(|p| !p.as_os_str().is_empty()) {
            map_layer_paths(&mut project.open_layers, |p| {
                resolve_project_path(base_dir, p)
            });
            for v in &mut project.open_vectors {
                v.path = resolve_project_path(base_dir, &v.path);
            }
        }
        project.normalize();
        Ok(project)
    }
}

#[cfg(test)]
#[path = "project_tests.rs"]
mod tests;
