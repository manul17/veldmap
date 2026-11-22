//! Application-wide storage ownership (`AppStorage`): platform storage paths,
//! persisted [`AppSettings`], shared SQLite WAL [`TileDiskCache`], and the
//! serialized off-UI-thread `DiskCacheWorker`.
//!
//! Stays completely decoupled from `veldmap-net` (using `veldmap_core::io::Wakeup`
//! and `TileDiskCache::write_generation()`) so local/offline storage never
//! depends on HTTP streaming.

use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use veldmap_core::io::{Wakeup, lock_unpoisoned};
use veldmap_storage::{DiskCacheStats, StoragePaths, TileDiskCache};

use crate::settings::AppSettings;

/// Minimum interval between event-driven disk-cache stats refreshes while the
/// Settings window is open and new tiles are being written.
const DISK_STATS_REFRESH_INTERVAL: Duration = Duration::from_secs(1);

/// UI-readable snapshot of asynchronous disk cache stats and maintenance state.
#[derive(Clone, Debug, Default)]
pub struct DiskCacheUiState {
    pub available: bool,
    pub stats: Option<DiskCacheStats>,
    pub busy: bool,
    pub notice: Option<String>,
    pub clear_gen: u64,
}

enum DiskCacheCmd {
    RefreshStats,
    SetMaxBytes(u64),
    Clear,
}

/// Serialized off-UI-thread worker handling SQLite stats, cap changes, and
/// full clears in FIFO order (never blocking the egui frame loop and joined
/// deterministically on shutdown).
struct DiskCacheWorker {
    tx: Option<Sender<DiskCacheCmd>>,
    state: Arc<Mutex<DiskCacheUiState>>,
    handle: Option<JoinHandle<()>>,
}

impl DiskCacheWorker {
    fn spawn(cache: Arc<TileDiskCache>, wakeup: Wakeup) -> Option<Self> {
        let (tx, rx) = mpsc::channel::<DiskCacheCmd>();
        let state = Arc::new(Mutex::new(DiskCacheUiState {
            available: true,
            stats: None,
            busy: false,
            notice: None,
            clear_gen: 0,
        }));
        let worker_state = Arc::clone(&state);
        let handle = match thread::Builder::new()
            .name("veldmap-disk-cache-ui".to_owned())
            .spawn(move || {
                while let Ok(cmd) = rx.recv() {
                    match cmd {
                        DiskCacheCmd::RefreshStats => {
                            if let Ok(stats) = cache.stats() {
                                lock_unpoisoned(&worker_state).stats = Some(stats);
                                (wakeup)();
                            }
                        }
                        DiskCacheCmd::SetMaxBytes(max_bytes) => {
                            if let Err(err) = cache.set_max_bytes(max_bytes) {
                                tracing::warn!(
                                    error = %err,
                                    "Failed to update SQLite disk cache budget"
                                );
                            }
                            if let Ok(stats) = cache.stats() {
                                lock_unpoisoned(&worker_state).stats = Some(stats);
                            }
                            (wakeup)();
                        }
                        DiskCacheCmd::Clear => {
                            let res = cache.clear();
                            let stats = cache.stats().ok();
                            {
                                let mut st = lock_unpoisoned(&worker_state);
                                st.busy = false;
                                if let Some(s) = stats {
                                    st.stats = Some(s);
                                }
                                match res {
                                    Ok(()) => {
                                        st.notice = Some("Disk cache cleared.".to_owned());
                                        st.clear_gen = st.clear_gen.wrapping_add(1);
                                    }
                                    Err(err) => {
                                        st.notice = Some(format!("Clear failed: {err}"));
                                    }
                                }
                            }
                            (wakeup)();
                        }
                    }
                }
            }) {
            Ok(h) => h,
            Err(err) => {
                tracing::warn!(error = %err, "Failed to spawn disk cache maintenance worker");
                return None;
            }
        };

        Some(Self {
            tx: Some(tx),
            state,
            handle: Some(handle),
        })
    }

    fn snapshot(&self) -> DiskCacheUiState {
        lock_unpoisoned(&self.state).clone()
    }

    fn clear_notice(&self) {
        lock_unpoisoned(&self.state).notice = None;
    }

    fn request_refresh(&self) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(DiskCacheCmd::RefreshStats);
        }
    }

    fn request_set_max_bytes(&self, max_bytes: u64) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(DiskCacheCmd::SetMaxBytes(max_bytes));
        }
    }

    fn request_clear(&self) {
        let Some(tx) = &self.tx else {
            let mut st = lock_unpoisoned(&self.state);
            st.busy = false;
            st.notice = Some("Disk cache worker unavailable.".to_owned());
            return;
        };
        {
            let mut st = lock_unpoisoned(&self.state);
            if st.busy {
                return;
            }
            st.busy = true;
            st.notice = None;
        }
        if tx.send(DiskCacheCmd::Clear).is_err() {
            let mut st = lock_unpoisoned(&self.state);
            st.busy = false;
            st.notice = Some("Clear failed: maintenance worker disconnected.".to_owned());
        }
    }

    fn close_channel(&mut self) {
        self.tx = None;
    }

    fn join(&mut self) {
        self.tx = None;
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for DiskCacheWorker {
    fn drop(&mut self) {
        self.join();
    }
}

/// Top-level application owner for platform paths, persisted [`AppSettings`],
/// shared SQLite WAL [`TileDiskCache`], and asynchronous cache maintenance.
pub struct AppStorage {
    paths: Option<StoragePaths>,
    settings: AppSettings,
    disk_cache: Option<Arc<TileDiskCache>>,
    disk_worker: Option<DiskCacheWorker>,
    last_settings_open: bool,
    last_stats_write_gen: u64,
    last_stats_refresh: Option<Instant>,
    last_handled_clear_gen: u64,
}

impl AppStorage {
    /// Load persisted user settings and open the shared SQLite WAL tile disk
    /// cache under `paths` (or platform defaults when `None`).
    pub fn open(wakeup: Wakeup, paths: Option<StoragePaths>) -> Self {
        let resolved_paths = paths.or_else(|| match StoragePaths::system_default() {
            Ok(p) => Some(p),
            Err(err) => {
                tracing::warn!(
                    error = %err,
                    "Platform storage directories unavailable; running without disk persistence"
                );
                None
            }
        });

        let settings = resolved_paths
            .as_ref()
            .map(AppSettings::load_or_default)
            .unwrap_or_default();

        let disk_cache = resolved_paths.as_ref().and_then(|p| {
            match TileDiskCache::open(p.tile_cache_db_path(), settings.disk_cache_max_bytes()) {
                Ok(cache) => Some(Arc::new(cache)),
                Err(err) => {
                    tracing::warn!(
                        error = %err,
                        "Failed to open SQLite tile disk cache; continuing without disk cache"
                    );
                    None
                }
            }
        });

        let disk_worker = disk_cache
            .as_ref()
            .and_then(|cache| DiskCacheWorker::spawn(Arc::clone(cache), wakeup));

        Self {
            paths: resolved_paths,
            settings,
            disk_cache,
            disk_worker,
            last_settings_open: false,
            last_stats_write_gen: 0,
            last_stats_refresh: None,
            last_handled_clear_gen: 0,
        }
    }

    pub fn settings(&self) -> &AppSettings {
        &self.settings
    }

    pub fn disk_cache(&self) -> Option<Arc<TileDiskCache>> {
        self.disk_cache.as_ref().map(Arc::clone)
    }

    /// Update and durably persist [`AppSettings`] when changed, forwarding any
    /// disk-cache cap change to the off-UI-thread maintenance worker. Returns
    /// `true` if settings changed.
    pub fn update_settings(&mut self, mut next: AppSettings) -> bool {
        next.normalize();
        let prev_cache_bytes = self.settings.disk_cache_max_bytes();
        let changed = next != self.settings;
        if !changed {
            return false;
        }
        self.settings = next;
        self.save_settings();

        let next_cache_bytes = self.settings.disk_cache_max_bytes();
        if next_cache_bytes != prev_cache_bytes
            && let Some(worker) = &self.disk_worker
        {
            worker.request_set_max_bytes(next_cache_bytes);
        }
        true
    }

    /// Mutate `AppSettings` in place, normalize, and persist durably if changed.
    pub fn mutate_settings(&mut self, edit: impl FnOnce(&mut AppSettings)) -> bool {
        let mut next = self.settings.clone();
        edit(&mut next);
        self.update_settings(next)
    }

    /// Persist the current [`AppSettings`] durably via
    /// `write_json_durable_private()`.
    pub fn save_settings(&self) {
        if let Some(paths) = &self.paths
            && let Err(err) = self.settings.save(paths)
        {
            tracing::warn!(error = %err, "Failed to persist user settings");
        }
    }

    /// Cached off-UI-thread disk cache status & stats snapshot (zero SQLite
    /// blocking on the UI thread).
    pub fn disk_cache_ui_state(&self) -> DiskCacheUiState {
        self.disk_worker
            .as_ref()
            .map(DiskCacheWorker::snapshot)
            .unwrap_or_default()
    }

    /// Clear transient disk cache UI notice (e.g. when reopening Settings).
    pub fn clear_disk_cache_notice(&self) {
        if let Some(worker) = &self.disk_worker {
            worker.clear_notice();
        }
    }

    /// Enqueue an asynchronous SQLite disk cache clear on the maintenance
    /// worker thread (never stalls the UI thread).
    pub fn request_clear_disk_cache(&self) {
        if let Some(worker) = &self.disk_worker {
            worker.request_clear();
        }
    }

    /// Event-driven disk-cache stats refresh: triggers an async stats query
    /// only when Settings opens or when `TileDiskCache::write_generation()`
    /// advances and at least `DISK_STATS_REFRESH_INTERVAL` has elapsed (zero
    /// idle queries/repaints).
    pub fn maybe_refresh_disk_stats(&mut self, settings_open: bool) {
        let just_opened = settings_open && !self.last_settings_open;
        self.last_settings_open = settings_open;
        if !settings_open {
            return;
        }
        let (Some(worker), Some(cache)) = (&self.disk_worker, &self.disk_cache) else {
            return;
        };
        let write_gen = cache.write_generation();
        let now = Instant::now();
        let interval_elapsed = self
            .last_stats_refresh
            .is_none_or(|t| now.duration_since(t) >= DISK_STATS_REFRESH_INTERVAL);
        if just_opened || (write_gen != self.last_stats_write_gen && interval_elapsed) {
            self.last_stats_write_gen = write_gen;
            self.last_stats_refresh = Some(now);
            worker.request_refresh();
        }
    }

    /// Poll whether an asynchronous disk cache clear has completed since the
    /// last check.
    pub fn poll_clear_completed(&mut self) -> bool {
        let Some(worker) = &self.disk_worker else {
            return false;
        };
        let clear_gen = worker.snapshot().clear_gen;
        if clear_gen != self.last_handled_clear_gen {
            self.last_handled_clear_gen = clear_gen;
            return true;
        }
        false
    }

    /// Phase 1 of deterministic shutdown: close maintenance command channel
    /// before joining threads.
    pub fn begin_shutdown(&mut self) {
        if let Some(worker) = &mut self.disk_worker {
            worker.close_channel();
        }
    }

    /// Phase 2 of deterministic shutdown: join the maintenance worker thread
    /// cleanly so the final `Arc<TileDiskCache>` flushes touches on drop.
    pub fn finish_shutdown(&mut self) {
        if let Some(worker) = &mut self.disk_worker {
            worker.join();
        }
    }
}

#[cfg(test)]
pub(crate) mod tests_support {
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

#[cfg(test)]
#[path = "storage_tests.rs"]
mod tests;
