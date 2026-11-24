//! Serialized off-UI-thread maintenance worker (`DiskCacheWorker`) handling
//! SQLite disk-cache stats, cap updates, full clears, and coalesced durable
//! `settings.json` / `session.json` saves in FIFO order.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use veldmap_core::sync::{Wakeup, lock_unpoisoned};
use veldmap_storage::{DiskCacheStats, StoragePaths, TileDiskCache};

use crate::session::AppSession;
use crate::settings::AppSettings;

/// Typed asynchronous maintenance outcome reported by [`DiskCacheWorker`]
/// (formatted into user-facing copy by the UI layer).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum DiskCacheNotice {
    Cleared,
    ClearFailed(String),
    WorkerUnavailable,
    WorkerDisconnected,
}

/// Snapshot of asynchronous disk cache stats and maintenance state.
#[derive(Clone, Debug, Default)]
pub(crate) struct DiskCacheStatus {
    pub(crate) available: bool,
    pub(crate) stats: Option<DiskCacheStats>,
    pub(crate) busy: bool,
    pub(crate) notice: Option<DiskCacheNotice>,
}

enum CoalesceStep<T> {
    Update(T),
    Defer(DiskCacheCmd),
}

fn drain_coalesced<T>(
    rx: &mpsc::Receiver<DiskCacheCmd>,
    pending_cmd: &mut Option<DiskCacheCmd>,
    mut latest: T,
    mut classify: impl FnMut(DiskCacheCmd) -> CoalesceStep<T>,
) -> T {
    while let Ok(cmd) = rx.try_recv() {
        match classify(cmd) {
            CoalesceStep::Update(newer) => latest = newer,
            CoalesceStep::Defer(other) => {
                *pending_cmd = Some(other);
                break;
            }
        }
    }
    latest
}

enum DiskCacheCmd {
    RefreshStats,
    SetMaxBytes(u64),
    Clear,
    SaveSettings(AppSettings),
    SaveSession(AppSession),
}

/// Serialized off-UI-thread worker handling SQLite stats, cap changes, full
/// clears, and coalesced durable `settings.json` / `session.json` writes in
/// FIFO order (never blocking the egui frame loop and draining cleanly before
/// joining on shutdown).
pub(super) struct DiskCacheWorker {
    tx: Option<Sender<DiskCacheCmd>>,
    state: Arc<Mutex<DiskCacheStatus>>,
    clear_gen: Arc<AtomicU64>,
    handle: Option<JoinHandle<()>>,
}

impl DiskCacheWorker {
    pub(super) fn spawn(
        cache: Option<Arc<TileDiskCache>>,
        paths: Option<StoragePaths>,
        wakeup: Wakeup,
    ) -> Option<Self> {
        if cache.is_none() && paths.is_none() {
            return None;
        }
        let (tx, rx) = mpsc::channel::<DiskCacheCmd>();
        let state = Arc::new(Mutex::new(DiskCacheStatus {
            available: cache.is_some(),
            stats: None,
            busy: false,
            notice: None,
        }));
        let clear_gen = Arc::new(AtomicU64::new(0));
        let worker_state = Arc::clone(&state);
        let worker_clear_gen = Arc::clone(&clear_gen);
        let handle = match thread::Builder::new()
            .name("veldmap-disk-cache-ui".to_owned())
            .spawn(move || {
                let mut pending_cmd: Option<DiskCacheCmd> = None;
                while let Some(cmd) = pending_cmd.take().or_else(|| rx.recv().ok()) {
                    match cmd {
                        DiskCacheCmd::RefreshStats => {
                            if let Some(cache) = &cache
                                && let Ok(stats) = cache.stats()
                            {
                                lock_unpoisoned(&worker_state).stats = Some(stats);
                                (wakeup)();
                            }
                        }
                        DiskCacheCmd::SetMaxBytes(max_bytes) => {
                            if let Some(cache) = &cache {
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
                        }
                        DiskCacheCmd::Clear => {
                            let Some(cache) = &cache else {
                                lock_unpoisoned(&worker_state).busy = false;
                                continue;
                            };
                            let res = cache.clear();
                            let stats = cache.stats().ok();
                            let succeeded = res.is_ok();
                            {
                                let mut st = lock_unpoisoned(&worker_state);
                                st.busy = false;
                                if let Some(s) = stats {
                                    st.stats = Some(s);
                                }
                                match res {
                                    Ok(()) => {
                                        st.notice = Some(DiskCacheNotice::Cleared);
                                    }
                                    Err(err) => {
                                        st.notice =
                                            Some(DiskCacheNotice::ClearFailed(err.to_string()));
                                    }
                                }
                            }
                            if succeeded {
                                worker_clear_gen.fetch_add(1, Ordering::Release);
                            }
                            (wakeup)();
                        }
                        DiskCacheCmd::SaveSettings(initial) => {
                            let settings =
                                drain_coalesced(&rx, &mut pending_cmd, initial, |cmd| match cmd {
                                    DiskCacheCmd::SaveSettings(newer) => {
                                        CoalesceStep::Update(newer)
                                    }
                                    other => CoalesceStep::Defer(other),
                                });
                            if let Some(paths) = &paths
                                && let Err(err) = settings.save(paths)
                            {
                                tracing::warn!(error = %err, "Failed to persist user settings");
                            }
                        }
                        DiskCacheCmd::SaveSession(initial) => {
                            let session =
                                drain_coalesced(&rx, &mut pending_cmd, initial, |cmd| match cmd {
                                    DiskCacheCmd::SaveSession(newer) => CoalesceStep::Update(newer),
                                    other => CoalesceStep::Defer(other),
                                });
                            if let Some(paths) = &paths
                                && let Err(err) = session.save(paths)
                            {
                                tracing::warn!(error = %err, "Failed to persist session");
                            }
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
            clear_gen,
            handle: Some(handle),
        })
    }

    pub(super) fn snapshot(&self) -> DiskCacheStatus {
        lock_unpoisoned(&self.state).clone()
    }

    pub(super) fn clear_gen(&self) -> u64 {
        self.clear_gen.load(Ordering::Acquire)
    }

    pub(super) fn clear_notice(&self) {
        lock_unpoisoned(&self.state).notice = None;
    }

    pub(super) fn request_refresh(&self) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(DiskCacheCmd::RefreshStats);
        }
    }

    pub(super) fn request_set_max_bytes(&self, max_bytes: u64) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(DiskCacheCmd::SetMaxBytes(max_bytes));
        }
    }

    pub(super) fn request_save_settings(&self, settings: AppSettings) -> bool {
        self.tx
            .as_ref()
            .is_some_and(|tx| tx.send(DiskCacheCmd::SaveSettings(settings)).is_ok())
    }

    pub(super) fn request_save_session(&self, session: AppSession) -> bool {
        self.tx
            .as_ref()
            .is_some_and(|tx| tx.send(DiskCacheCmd::SaveSession(session)).is_ok())
    }

    pub(super) fn request_clear(&self) {
        let Some(tx) = &self.tx else {
            let mut st = lock_unpoisoned(&self.state);
            st.busy = false;
            st.notice = Some(DiskCacheNotice::WorkerUnavailable);
            return;
        };
        {
            let mut st = lock_unpoisoned(&self.state);
            if !st.available || st.busy {
                return;
            }
            st.busy = true;
            st.notice = None;
        }
        if tx.send(DiskCacheCmd::Clear).is_err() {
            let mut st = lock_unpoisoned(&self.state);
            st.busy = false;
            st.notice = Some(DiskCacheNotice::WorkerDisconnected);
        }
    }

    pub(super) fn close_channel(&mut self) {
        self.tx = None;
    }

    pub(super) fn join(&mut self) {
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
