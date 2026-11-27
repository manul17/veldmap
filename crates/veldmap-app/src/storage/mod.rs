//! Application-wide storage ownership (`AppStorage`): platform storage paths,
//! persisted [`AppSettings`] and [`AppSession`], shared SQLite WAL
//! [`TileDiskCache`], and the serialized off-UI-thread
//! [`worker::DiskCacheWorker`] (handling SQLite maintenance and durable
//! `settings.json` / `session.json` saves off the UI thread).
//!
//! `AppStorage` and [`worker::DiskCacheWorker`] stay decoupled from
//! `veldmap-net` and `LiveTilePipeline` (communicating through
//! `veldmap_core::sync::Wakeup` and `TileDiskCache::write_generation()`) so
//! local/offline storage never depends on HTTP streaming, while
//! [`settings`] reuses `veldmap-net` config and validation helpers.

pub(crate) mod session;
pub(crate) mod settings;
mod worker;

use std::sync::Arc;
use std::time::{Duration, Instant};

use veldmap_core::sync::Wakeup;
use veldmap_storage::{StorageError, StoragePaths, TileDiskCache};

use self::session::AppSession;
use self::settings::AppSettings;
pub(crate) use self::worker::{DiskCacheNotice, DiskCacheStatus};
use self::worker::{DiskCacheWorker, PersistenceFaults};

/// Minimum interval between event-driven disk-cache stats refreshes while the
/// Settings window is open and new tiles are being written.
const DISK_STATS_REFRESH_INTERVAL: Duration = Duration::from_secs(1);

/// Top-level application owner for platform paths, persisted [`AppSettings`]
/// and [`AppSession`], shared SQLite WAL [`TileDiskCache`], and asynchronous
/// cache/settings/session I/O.
pub(crate) struct AppStorage {
    paths: Option<StoragePaths>,
    settings: AppSettings,
    session: AppSession,
    disk_cache: Option<Arc<TileDiskCache>>,
    disk_worker: Option<DiskCacheWorker>,
    persistence_faults: Arc<PersistenceFaults>,
    last_stats_write_gen: u64,
    last_stats_refresh: Option<Instant>,
    last_handled_clear_gen: u64,
}

impl AppStorage {
    /// Load persisted user settings and session state and open the shared
    /// SQLite WAL tile disk cache under `paths` (or platform defaults when
    /// `None`).
    pub(crate) fn open(wakeup: Wakeup, paths: Option<StoragePaths>) -> Self {
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

        let session = resolved_paths
            .as_ref()
            .map(AppSession::load_or_default)
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

        let persistence_faults = Arc::new(PersistenceFaults::default());
        let disk_worker = DiskCacheWorker::spawn(
            disk_cache.as_ref().map(Arc::clone),
            resolved_paths.clone(),
            wakeup,
            Arc::clone(&persistence_faults),
        );

        Self {
            paths: resolved_paths,
            settings,
            session,
            disk_cache,
            disk_worker,
            persistence_faults,
            last_stats_write_gen: 0,
            last_stats_refresh: None,
            last_handled_clear_gen: 0,
        }
    }

    pub(crate) fn settings(&self) -> &AppSettings {
        &self.settings
    }

    pub(crate) fn session(&self) -> &AppSession {
        &self.session
    }

    /// Update and durably persist [`AppSession`] asynchronously when changed.
    fn update_session(&mut self, mut next: AppSession) -> bool {
        next.normalize();
        if next == self.session {
            return false;
        }
        self.session = next;
        self.save_session();
        true
    }

    /// Mutate `AppSession` in place, normalize, and enqueue durable write if changed.
    pub(crate) fn mutate_session(&mut self, edit: impl FnOnce(&mut AppSession)) -> bool {
        let mut next = self.session.clone();
        edit(&mut next);
        self.update_session(next)
    }

    fn persist_with_fallback(
        &self,
        label: &str,
        try_enqueue: impl FnOnce(&DiskCacheWorker) -> bool,
        save_direct: impl FnOnce(&StoragePaths) -> Result<(), StorageError>,
        record_direct_result: impl FnOnce(&PersistenceFaults, Option<String>) -> bool,
    ) {
        if let Some(worker) = &self.disk_worker
            && try_enqueue(worker)
        {
            return;
        }
        if let Some(paths) = &self.paths {
            match save_direct(paths) {
                Ok(()) => {
                    let _ = record_direct_result(&self.persistence_faults, None);
                }
                Err(err) => {
                    tracing::warn!(error = %err, "Failed to persist {label}");
                    let _ = record_direct_result(&self.persistence_faults, Some(err.to_string()));
                }
            }
        }
    }

    fn save_session(&self) {
        self.persist_with_fallback(
            "session",
            |worker| worker.request_save_session(self.session.clone()),
            |paths| self.session.save(paths),
            PersistenceFaults::set_session_result,
        );
    }

    /// Lock-free fast-path summary of any active `settings.json` / `session.json`
    /// save failure (`None` with zero mutex acquisitions on healthy frames).
    pub(crate) fn persistence_fault_summary(&self) -> Option<String> {
        self.persistence_faults.summary()
    }

    /// Dismiss active persistence save failure notices from the UI banner.
    pub(crate) fn dismiss_persistence_fault(&self) {
        self.persistence_faults.dismiss();
    }

    pub(crate) fn disk_cache(&self) -> Option<Arc<TileDiskCache>> {
        self.disk_cache.as_ref().map(Arc::clone)
    }

    /// Update and durably persist [`AppSettings`] asynchronously when changed,
    /// forwarding any disk-cache cap change to the off-UI-thread maintenance
    /// worker. Returns `true` if settings changed.
    fn update_settings(&mut self, mut next: AppSettings) -> bool {
        next.normalize();
        let prev_cache_bytes = self.settings.disk_cache_max_bytes();
        let changed = next != self.settings;
        if !changed {
            return false;
        }
        self.settings = next;
        self.save_settings();

        let next_cache_bytes = self.settings.disk_cache_max_bytes();
        if next_cache_bytes != prev_cache_bytes {
            if let Some(worker) = &self.disk_worker {
                worker.request_set_max_bytes(next_cache_bytes);
            } else if let Some(cache) = &self.disk_cache
                && let Err(err) = cache.set_max_bytes(next_cache_bytes)
            {
                tracing::warn!(
                    error = %err,
                    "Failed to update SQLite disk cache budget directly"
                );
            }
        }
        true
    }

    /// Mutate `AppSettings` in place, normalize, and enqueue durable
    /// persistence if changed.
    pub(crate) fn mutate_settings(&mut self, edit: impl FnOnce(&mut AppSettings)) -> bool {
        let mut next = self.settings.clone();
        edit(&mut next);
        self.update_settings(next)
    }

    /// Enqueue durable persistence of [`AppSettings`] onto
    /// [`worker::DiskCacheWorker`], falling back to direct write only if the
    /// worker is unavailable.
    fn save_settings(&self) {
        self.persist_with_fallback(
            "user settings",
            |worker| worker.request_save_settings(self.settings.clone()),
            |paths| self.settings.save(paths),
            PersistenceFaults::set_settings_result,
        );
    }

    /// Cached off-UI-thread disk cache status & stats snapshot (zero SQLite
    /// blocking on the UI thread). Call only when Settings is open.
    pub(crate) fn disk_cache_status(&self) -> DiskCacheStatus {
        self.disk_worker
            .as_ref()
            .map(DiskCacheWorker::snapshot)
            .unwrap_or_default()
    }

    /// Clear transient disk cache notice (e.g. when reopening Settings).
    pub(crate) fn clear_disk_cache_notice(&self) {
        if let Some(worker) = &self.disk_worker {
            worker.clear_notice();
        }
    }

    /// Enqueue an asynchronous SQLite disk cache clear on the maintenance
    /// worker thread (never stalls the UI thread).
    pub(crate) fn request_clear_disk_cache(&self) {
        if let Some(worker) = &self.disk_worker {
            worker.request_clear();
        }
    }

    /// Event-driven disk-cache stats refresh: triggers an async stats query
    /// only when `force_refresh` is `true` (e.g. Settings just opened) or when
    /// `TileDiskCache::write_generation()` advances and at least
    /// `DISK_STATS_REFRESH_INTERVAL` has elapsed (zero idle queries/repaints).
    pub(crate) fn maybe_refresh_disk_stats(&mut self, force_refresh: bool) {
        let (Some(worker), Some(cache)) = (&self.disk_worker, &self.disk_cache) else {
            return;
        };
        let write_gen = cache.write_generation();
        let now = Instant::now();
        let interval_elapsed = self
            .last_stats_refresh
            .is_none_or(|t| now.duration_since(t) >= DISK_STATS_REFRESH_INTERVAL);
        if force_refresh || (write_gen != self.last_stats_write_gen && interval_elapsed) {
            self.last_stats_write_gen = write_gen;
            self.last_stats_refresh = Some(now);
            worker.request_refresh();
        }
    }

    /// Lock-free atomic check whether an asynchronous disk cache clear has
    /// completed since the last check.
    pub(crate) fn poll_clear_completed(&mut self) -> bool {
        let Some(worker) = &self.disk_worker else {
            return false;
        };
        let clear_gen = worker.clear_gen();
        if clear_gen != self.last_handled_clear_gen {
            self.last_handled_clear_gen = clear_gen;
            return true;
        }
        false
    }

    /// Phase 1 of deterministic shutdown: close maintenance command channel
    /// before joining threads (all already-queued saves drain in FIFO order).
    pub(crate) fn begin_shutdown(&mut self) {
        if let Some(worker) = &mut self.disk_worker {
            worker.close_channel();
        }
    }

    /// Phase 3 of deterministic shutdown: join the maintenance worker thread
    /// cleanly so pending `SaveSettings` and `SaveSession` writes flush and
    /// the final `Arc<TileDiskCache>` flushes touches on drop.
    pub(crate) fn finish_shutdown(&mut self) {
        if let Some(worker) = &mut self.disk_worker {
            worker.join();
        }
    }
}

#[cfg(test)]
#[path = "storage_tests.rs"]
mod tests;
