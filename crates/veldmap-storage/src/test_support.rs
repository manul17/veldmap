//! RAII temporary directory helper shared by unit and integration tests.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Temporary directory automatically removed when dropped.
pub struct TestTempDir(PathBuf);

impl TestTempDir {
    /// Create a fresh process- and timestamp-unique temporary directory.
    pub fn new(label: &str) -> Self {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "veldmap-test-{label}-{}-{nanos}-{seq}",
            std::process::id()
        ));
        if let Err(err) = std::fs::create_dir_all(&dir) {
            panic!("create temp test dir {}: {err}", dir.display());
        }
        Self(dir)
    }

    /// Path to the created temporary directory.
    pub fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TestTempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
