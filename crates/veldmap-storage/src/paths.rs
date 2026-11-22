//! Injectable application storage paths (`config_dir`, `cache_dir`, `data_dir`).

use std::path::{Path, PathBuf};

use directories::ProjectDirs;

use crate::StorageError;

/// Concrete filesystem roots for VeldMap configuration, ephemeral caches, and
/// persistent user data. Always injected rather than queried via globals so
/// tests and portable installations remain hermetic.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StoragePaths {
    pub config_dir: PathBuf,
    pub cache_dir: PathBuf,
    pub data_dir: PathBuf,
}

impl StoragePaths {
    /// Resolve standard OS-native paths (`XDG` on Linux, `AppData` on Windows,
    /// `Library/Application Support` + `Library/Caches` on macOS).
    pub fn system_default() -> Result<Self, StorageError> {
        let dirs =
            ProjectDirs::from("io", "veldmap", "VeldMap").ok_or(StorageError::NoPlatformDirs)?;
        Ok(Self {
            config_dir: dirs.config_dir().to_path_buf(),
            cache_dir: dirs.cache_dir().to_path_buf(),
            data_dir: dirs.data_dir().to_path_buf(),
        })
    }

    /// Construct isolated paths rooted under `root` (for tests or portable runs).
    pub fn with_root(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        Self {
            config_dir: root.join("config"),
            cache_dir: root.join("cache"),
            data_dir: root.join("data"),
        }
    }

    /// Path to durable user settings JSON (`config_dir/settings.json`).
    pub fn settings_path(&self) -> PathBuf {
        self.config_dir.join("settings.json")
    }

    /// Path to the single-file SQLite WAL tile cache (`cache_dir/tiles.sqlite3`).
    pub fn tile_cache_db_path(&self) -> PathBuf {
        self.cache_dir.join("tiles.sqlite3")
    }

    /// Ensure `config_dir`, `cache_dir`, and `data_dir` exist on disk.
    pub fn ensure_dirs(&self) -> Result<(), StorageError> {
        for dir in [&self.config_dir, &self.cache_dir, &self.data_dir] {
            std::fs::create_dir_all(dir)?;
        }
        Ok(())
    }

    /// Return the configured cache directory.
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
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
