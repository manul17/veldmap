//! Dedicated off-UI-thread [`OverlayIoWorker`] for non-blocking local raster
//! fingerprinting, GeoTIFF/PNG/JPEG opening, and reload disk-cache invalidation.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc::Sender;

use veldmap_core::geo::Frame;
use veldmap_core::sync::CancelToken;
use veldmap_raster::RasterSource;

use crate::worker::SingleThreadWorker;

use super::opener::{OpenRasterError, open_local_raster};
use super::runtime::OverlayRuntime;
use super::stack::LayerInitOptions;
use super::{CrsOverride, LayerId};

/// Parameters describing a non-blocking reload of an existing overlay layer.
pub(super) struct ReloadRequest {
    pub(super) id: LayerId,
    pub(super) reload_seq: u64,
    pub(super) crs_override_epoch: u64,
    pub(super) path: PathBuf,
    pub(super) old_fingerprint_hex: String,
    pub(super) crs_override: Option<CrsOverride>,
    pub(super) cancel: CancelToken,
}

/// Completed outcome of a background overlay reload.
pub(super) struct ReloadedPayload {
    pub(super) id: LayerId,
    pub(super) reload_seq: u64,
    pub(super) crs_override_epoch: u64,
    pub(super) path: PathBuf,
    pub(super) outcome: Result<(Arc<dyn RasterSource>, Frame), OpenRasterError>,
}

/// Commands processed off the UI thread by [`OverlayIoWorker`].
pub(super) enum OverlayIoCmd {
    Open {
        path: PathBuf,
        fly_to: bool,
        options: LayerInitOptions,
    },
    Reload(ReloadRequest),
}

/// Results posted back from [`OverlayIoWorker`] to `OverlayManager`.
pub(super) enum OverlayIoResult {
    Opened {
        path: PathBuf,
        fly_to: bool,
        options: LayerInitOptions,
        outcome: Result<(Arc<dyn RasterSource>, Frame), OpenRasterError>,
    },
    Reloaded(ReloadedPayload),
}

fn handle_io_cmd(cmd: OverlayIoCmd, runtime: &OverlayRuntime, res_tx: &Sender<OverlayIoResult>) {
    match cmd {
        OverlayIoCmd::Open {
            path,
            fly_to,
            options,
        } => {
            if runtime.shutdown.is_cancelled() {
                return;
            }
            let outcome = open_local_raster(
                &path,
                &runtime.fingerprint_cache,
                &runtime.budget,
                false,
                options.crs_override.as_ref(),
                &runtime.shutdown,
            );
            if runtime.shutdown.is_cancelled() {
                return;
            }
            let _ = res_tx.send(OverlayIoResult::Opened {
                path,
                fly_to,
                options,
                outcome,
            });
            (runtime.wakeup)();
        }
        OverlayIoCmd::Reload(ReloadRequest {
            id,
            reload_seq,
            crs_override_epoch,
            path,
            old_fingerprint_hex,
            crs_override,
            cancel,
        }) => {
            if cancel.is_cancelled() {
                return;
            }
            runtime.fingerprint_cache.invalidate(&path);
            let outcome = open_local_raster(
                &path,
                &runtime.fingerprint_cache,
                &runtime.budget,
                true,
                crs_override.as_ref(),
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
            let _ = res_tx.send(OverlayIoResult::Reloaded(ReloadedPayload {
                id,
                reload_seq,
                crs_override_epoch,
                path,
                outcome,
            }));
            (runtime.wakeup)();
        }
    }
}

/// Dedicated single-thread background worker handling local GeoTIFF/COG and
/// PNG/JPEG opening plus reload SQLite cache clearing strictly off the UI
/// thread (joined deterministically on shutdown).
pub(super) struct OverlayIoWorker {
    worker: SingleThreadWorker<OverlayIoCmd, OverlayIoResult>,
}

impl OverlayIoWorker {
    pub(super) fn spawn(runtime: Arc<OverlayRuntime>) -> Self {
        let worker = SingleThreadWorker::spawn("veldmap-overlay-io", move |cmd_rx, res_tx| {
            while let Ok(cmd) = cmd_rx.recv() {
                if runtime.shutdown.is_cancelled() {
                    break;
                }
                handle_io_cmd(cmd, &runtime, &res_tx);
            }
        });
        Self { worker }
    }

    pub(super) fn send(&self, cmd: OverlayIoCmd) -> bool {
        self.worker.send(cmd)
    }

    pub(super) fn try_recv(&self) -> Option<OverlayIoResult> {
        self.worker.try_recv()
    }

    pub(super) fn close_channel(&mut self) {
        self.worker.close_channel();
    }

    pub(super) fn join(&mut self) {
        self.worker.join();
    }
}
