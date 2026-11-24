//! Dedicated off-UI-thread [`OverlayIoWorker`] for non-blocking local file
//! fingerprinting, GeoTIFF/COG opening, and reload disk-cache invalidation.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::{self, JoinHandle};

use veldmap_core::geo::Frame;
use veldmap_core::sync::CancelToken;
use veldmap_raster::RasterSource;

use super::LayerId;
use super::opener::open_local_raster;
use super::runtime::OverlayRuntime;

/// Commands processed off the UI thread by [`OverlayIoWorker`].
pub(super) enum OverlayIoCmd {
    Open {
        path: PathBuf,
        fly_to: bool,
        cancel: CancelToken,
    },
    Reload {
        id: LayerId,
        reload_seq: u64,
        path: PathBuf,
        old_fingerprint_hex: String,
        cancel: CancelToken,
    },
}

/// Results posted back from [`OverlayIoWorker`] to `OverlayManager`.
pub(super) enum OverlayIoResult {
    Opened {
        path: PathBuf,
        fly_to: bool,
        outcome: Result<(Arc<dyn RasterSource>, Frame), String>,
    },
    Reloaded {
        id: LayerId,
        reload_seq: u64,
        path: PathBuf,
        outcome: Result<(Arc<dyn RasterSource>, Frame), String>,
    },
}

fn handle_io_cmd(cmd: OverlayIoCmd, runtime: &OverlayRuntime, res_tx: &Sender<OverlayIoResult>) {
    match cmd {
        OverlayIoCmd::Open {
            path,
            fly_to,
            cancel,
        } => {
            if cancel.is_cancelled() {
                return;
            }
            let outcome = open_local_raster(
                &path,
                &runtime.fingerprint_cache,
                &runtime.budget,
                false,
                &cancel,
            );
            if cancel.is_cancelled() {
                return;
            }
            let _ = res_tx.send(OverlayIoResult::Opened {
                path,
                fly_to,
                outcome,
            });
            (runtime.wakeup)();
        }
        OverlayIoCmd::Reload {
            id,
            reload_seq,
            path,
            old_fingerprint_hex,
            cancel,
        } => {
            if cancel.is_cancelled() {
                return;
            }
            runtime.fingerprint_cache.invalidate(&path);
            let outcome = open_local_raster(
                &path,
                &runtime.fingerprint_cache,
                &runtime.budget,
                true,
                &cancel,
            );
            if cancel.is_cancelled() {
                return;
            }
            // Only clear disk cache entries AFTER reopen succeeds so a failed
            // reload preserves existing state, and run `clear_source` strictly
            // off the UI thread.
            if let Ok((ref source, _)) = outcome
                && let Some(ref disk) = runtime.disk_cache
            {
                let _ = disk.clear_source(&old_fingerprint_hex);
                let new_fp = source.fingerprint().to_hex();
                if new_fp != old_fingerprint_hex {
                    let _ = disk.clear_source(&new_fp);
                }
            }
            let _ = res_tx.send(OverlayIoResult::Reloaded {
                id,
                reload_seq,
                path,
                outcome,
            });
            (runtime.wakeup)();
        }
    }
}

/// Dedicated single-thread background worker handling local GeoTIFF/COG
/// fingerprinting, IFD opening, and reload SQLite cache clearing strictly off
/// the UI thread (joined deterministically on shutdown).
pub(super) struct OverlayIoWorker {
    tx: Option<Sender<OverlayIoCmd>>,
    rx: Receiver<OverlayIoResult>,
    handle: Option<JoinHandle<()>>,
}

impl OverlayIoWorker {
    pub(super) fn spawn(runtime: Arc<OverlayRuntime>) -> Self {
        let (cmd_tx, cmd_rx) = mpsc::channel::<OverlayIoCmd>();
        let (res_tx, res_rx) = mpsc::channel::<OverlayIoResult>();
        let handle = match thread::Builder::new()
            .name("veldmap-overlay-io".to_owned())
            .spawn(move || {
                while let Ok(cmd) = cmd_rx.recv() {
                    if runtime.shutdown.is_cancelled() {
                        break;
                    }
                    handle_io_cmd(cmd, &runtime, &res_tx);
                }
            }) {
            Ok(h) => Some(h),
            Err(err) => {
                tracing::error!(error = %err, "Failed to spawn overlay I/O worker");
                None
            }
        };

        Self {
            tx: handle.is_some().then_some(cmd_tx),
            rx: res_rx,
            handle,
        }
    }

    pub(super) fn send(&self, cmd: OverlayIoCmd) -> bool {
        self.tx.as_ref().is_some_and(|tx| tx.send(cmd).is_ok())
    }

    pub(super) fn try_recv(&self) -> Option<OverlayIoResult> {
        self.rx.try_recv().ok()
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

impl Drop for OverlayIoWorker {
    fn drop(&mut self) {
        self.join();
    }
}
