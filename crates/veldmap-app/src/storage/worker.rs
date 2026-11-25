//! Serialized off-UI-thread maintenance worker (`DiskCacheWorker`) handling
//! SQLite disk-cache stats, cap updates, full clears, and coalesced durable
//! `settings.json` / `session.json` saves in FIFO order.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

use veldmap_core::sync::{Wakeup, lock_unpoisoned};
use veldmap_storage::{DiskCacheStats, StoragePaths, TileDiskCache};

use crate::worker::{CommandSendError, CommandWorker};

use super::session::AppSession;
use super::settings::AppSettings;

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

struct DiskWorkerCtx<'a> {
    cache: Option<&'a TileDiskCache>,
    paths: Option<&'a StoragePaths>,
    wakeup: &'a Wakeup,
    state: &'a Mutex<DiskCacheStatus>,
    clear_gen: &'a AtomicU64,
}

fn handle_disk_cmd(
    cmd: DiskCacheCmd,
    rx: &mpsc::Receiver<DiskCacheCmd>,
    pending_cmd: &mut Option<DiskCacheCmd>,
    ctx: &DiskWorkerCtx<'_>,
) {
    match cmd {
        DiskCacheCmd::RefreshStats => {
            if let Some(cache) = ctx.cache
                && let Ok(stats) = cache.stats()
            {
                lock_unpoisoned(ctx.state).stats = Some(stats);
                (ctx.wakeup)();
            }
        }
        DiskCacheCmd::SetMaxBytes(max_bytes) => {
            if let Some(cache) = ctx.cache {
                if let Err(err) = cache.set_max_bytes(max_bytes) {
                    tracing::warn!(
                        error = %err,
                        "Failed to update SQLite disk cache budget"
                    );
                }
                if let Ok(stats) = cache.stats() {
                    lock_unpoisoned(ctx.state).stats = Some(stats);
                }
                (ctx.wakeup)();
            }
        }
        DiskCacheCmd::Clear => {
            let Some(cache) = ctx.cache else {
                lock_unpoisoned(ctx.state).busy = false;
                return;
            };
            let res = cache.clear();
            let stats = cache.stats().ok();
            let succeeded = res.is_ok();
            {
                let mut st = lock_unpoisoned(ctx.state);
                st.busy = false;
                if let Some(s) = stats {
                    st.stats = Some(s);
                }
                match res {
                    Ok(()) => {
                        st.notice = Some(DiskCacheNotice::Cleared);
                    }
                    Err(err) => {
                        st.notice = Some(DiskCacheNotice::ClearFailed(err.to_string()));
                    }
                }
            }
            if succeeded {
                ctx.clear_gen.fetch_add(1, Ordering::Release);
            }
            (ctx.wakeup)();
        }
        DiskCacheCmd::SaveSettings(initial) => {
            let settings = drain_coalesced(rx, pending_cmd, initial, |cmd| match cmd {
                DiskCacheCmd::SaveSettings(newer) => CoalesceStep::Update(newer),
                other => CoalesceStep::Defer(other),
            });
            if let Some(paths) = ctx.paths
                && let Err(err) = settings.save(paths)
            {
                tracing::warn!(error = %err, "Failed to persist user settings");
            }
        }
        DiskCacheCmd::SaveSession(initial) => {
            let session = drain_coalesced(rx, pending_cmd, initial, |cmd| match cmd {
                DiskCacheCmd::SaveSession(newer) => CoalesceStep::Update(newer),
                other => CoalesceStep::Defer(other),
            });
            if let Some(paths) = ctx.paths
                && let Err(err) = session.save(paths)
            {
                tracing::warn!(error = %err, "Failed to persist session");
            }
        }
    }
}

/// Serialized off-UI-thread worker handling SQLite stats, cap changes, full
/// clears, and coalesced durable `settings.json` / `session.json` writes in
/// FIFO order (never blocking the egui frame loop and draining cleanly before
/// joining on shutdown).
pub(super) struct DiskCacheWorker {
    worker: CommandWorker<DiskCacheCmd>,
    state: Arc<Mutex<DiskCacheStatus>>,
    clear_gen: Arc<AtomicU64>,
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
        let state = Arc::new(Mutex::new(DiskCacheStatus {
            available: cache.is_some(),
            stats: None,
            busy: false,
            notice: None,
        }));
        let clear_gen = Arc::new(AtomicU64::new(0));
        let worker_state = Arc::clone(&state);
        let worker_clear_gen = Arc::clone(&clear_gen);

        let worker = CommandWorker::try_spawn("veldmap-disk-cache-ui", move |rx| {
            let ctx = DiskWorkerCtx {
                cache: cache.as_deref(),
                paths: paths.as_ref(),
                wakeup: &wakeup,
                state: &worker_state,
                clear_gen: &worker_clear_gen,
            };
            let mut pending_cmd: Option<DiskCacheCmd> = None;
            while let Some(cmd) = pending_cmd.take().or_else(|| rx.recv().ok()) {
                handle_disk_cmd(cmd, &rx, &mut pending_cmd, &ctx);
            }
        })?;

        Some(Self {
            worker,
            state,
            clear_gen,
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
        let _ = self.worker.send(DiskCacheCmd::RefreshStats);
    }

    pub(super) fn request_set_max_bytes(&self, max_bytes: u64) {
        let _ = self.worker.send(DiskCacheCmd::SetMaxBytes(max_bytes));
    }

    pub(super) fn request_save_settings(&self, settings: AppSettings) -> bool {
        self.worker
            .send(DiskCacheCmd::SaveSettings(settings))
            .is_ok()
    }

    pub(super) fn request_save_session(&self, session: AppSession) -> bool {
        self.worker.send(DiskCacheCmd::SaveSession(session)).is_ok()
    }

    pub(super) fn request_clear(&self) {
        {
            let mut st = lock_unpoisoned(&self.state);
            if !st.available || st.busy {
                return;
            }
            st.busy = true;
            st.notice = None;
        }
        if let Err(err) = self.worker.send(DiskCacheCmd::Clear) {
            let mut st = lock_unpoisoned(&self.state);
            st.busy = false;
            st.notice = Some(match err {
                CommandSendError::Closed => DiskCacheNotice::WorkerUnavailable,
                CommandSendError::Disconnected => DiskCacheNotice::WorkerDisconnected,
            });
        }
    }

    pub(super) fn close_channel(&mut self) {
        self.worker.close_channel();
    }

    pub(super) fn join(&mut self) {
        self.worker.join();
    }
}
