//! Injectable application storage paths (`config_dir`, `cache_dir`).

use std::path::{Path, PathBuf};

use directories::ProjectDirs;

use crate::StorageError;

/// Append `suffix` directly onto `path`'s OS string (preserving existing
/// extensions, e.g. `settings.json` -> `settings.json.corrupt` or
/// `tiles.sqlite3` -> `tiles.sqlite3-wal`).
pub(crate) fn append_path_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut os = path.as_os_str().to_os_string();
    os.push(suffix);
    PathBuf::from(os)
}

/// Concrete filesystem roots for VeldMap configuration and ephemeral caches.
/// Always injected rather than queried via globals so tests and portable
/// installations remain hermetic.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StoragePaths {
    pub config_dir: PathBuf,
    pub cache_dir: PathBuf,
}

impl StoragePaths {
    /// Resolve standard OS-native paths (`XDG` on Linux, `AppData` on Windows,
    /// `Library/Application Support` + `Library/Caches` on macOS).
    pub fn system_default() -> Result<Self, StorageError> {
        let dirs = ProjectDirs::from("io", "veldmap", veldmap_core::APP_NAME)
            .ok_or(StorageError::NoPlatformDirs)?;
        Ok(Self {
            config_dir: dirs.config_dir().to_path_buf(),
            cache_dir: dirs.cache_dir().to_path_buf(),
        })
    }

    /// Construct isolated paths rooted under `root` (for tests or portable runs).
    pub fn with_root(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        Self {
            config_dir: root.join("config"),
            cache_dir: root.join("cache"),
        }
    }

    /// Path to durable user settings JSON (`config_dir/settings.json`).
    pub fn settings_path(&self) -> PathBuf {
        self.config_dir.join("settings.json")
    }

    /// Path to durable local workstation session JSON (`config_dir/session.json`).
    pub fn session_path(&self) -> PathBuf {
        self.config_dir.join("session.json")
    }

    /// Path to the single-file SQLite WAL tile cache (`cache_dir/tiles.sqlite3`).
    pub fn tile_cache_db_path(&self) -> PathBuf {
        self.cache_dir.join("tiles.sqlite3")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn with_root_derives_consistent_subpaths() {
        let paths = StoragePaths::with_root("/tmp/veldmap-test-root");
        assert_eq!(
            paths.settings_path(),
            PathBuf::from("/tmp/veldmap-test-root/config/settings.json")
        );
        assert_eq!(
            paths.tile_cache_db_path(),
            PathBuf::from("/tmp/veldmap-test-root/cache/tiles.sqlite3")
        );
    }
}
