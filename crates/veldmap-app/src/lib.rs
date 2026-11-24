//! Library surface of `veldmap-app` exposing [`VeldmapApp`] and [`AppError`]
//! for the binary entry point plus the headless-testable rendering boundary
//! (`render`, `gpu_tiles`, `basemap_policy`, [`BUNDLED_BASEMAP_CONFIG`], and
//! [`BUNDLED_BASEMAP_JPEG`]) for GPU smoke tests and `naga` shader validation.

mod app;
pub(crate) mod basemap;
pub mod gpu_tiles;
pub mod render;
mod scene;
pub(crate) mod settings;
pub(crate) mod storage;
mod ui;
mod viewport_bridge;

pub use app::{AppError, VeldmapApp};
pub use basemap::bundled::{BUNDLED_BASEMAP_CONFIG, BUNDLED_BASEMAP_JPEG, BUNDLED_MAX_LEVEL};
pub use basemap::policy as basemap_policy;

#[cfg(test)]
pub(crate) mod test_support {
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    pub(crate) struct TempTestRoot(PathBuf);

    impl TempTestRoot {
        pub(crate) fn new(label: &str) -> Self {
            static SEQ: AtomicU64 = AtomicU64::new(0);
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos());
            let seq = SEQ.fetch_add(1, Ordering::Relaxed);
            let dir = std::env::temp_dir().join(format!(
                "veldmap-app-test-{label}-{}-{nanos}-{seq}",
                std::process::id()
            ));
            std::fs::create_dir_all(&dir).expect("create temp test dir");
            Self(dir)
        }

        pub(crate) fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempTestRoot {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}
