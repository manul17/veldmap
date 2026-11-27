//! Non-blocking remote Cloud-Optimized GeoTIFF opening ([`RemoteOpenTracker`])
//! isolated onto `NetContext` + `veldmap-remote-io` (`remote_pool`).

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};

use veldmap_core::geo::Frame;
use veldmap_net::{HttpRangeConfig, HttpRangeSource, sanitize_url_for_display};
use veldmap_raster::{DecodeJob, DecodeJobKey, RasterSource};

use super::RemoteCogRequest;
use super::opener::open_remote_cog_source;
use super::runtime::OverlayRuntime;

/// Completed outcome of a background remote Cloud-Optimized GeoTIFF open.
pub(super) struct RemoteOpenResult {
    pub(super) title: String,
    pub(super) display_url: String,
    pub(super) fly_to: bool,
    pub(super) outcome: Result<(HttpRangeSource, Arc<dyn RasterSource>, Frame), String>,
}

/// Manages non-blocking remote COG open requests over `NetContext` +
/// `remote_pool`, isolating remote HTTP Range probing and IFD parsing from
/// local disk I/O and `fast_pool`.
pub(super) struct RemoteOpenTracker {
    next_open_id: AtomicU64,
    tx: Sender<RemoteOpenResult>,
    rx: Receiver<RemoteOpenResult>,
}

impl RemoteOpenTracker {
    pub(super) fn new() -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            next_open_id: AtomicU64::new(1),
            tx,
            rx,
        }
    }

    /// Queue a non-blocking remote Cloud-Optimized GeoTIFF open onto
    /// `NetContext` + `remote_pool`, completely isolated from local file
    /// open/reload I/O and `fast_pool`.
    pub(super) fn begin_open(
        &self,
        runtime: &Arc<OverlayRuntime>,
        request: RemoteCogRequest,
    ) -> Result<(), String> {
        let RemoteCogRequest {
            url,
            title,
            extra_headers,
            scale_offset,
            fly_to,
        } = request;
        let Some(net) = runtime.net.clone() else {
            return Err("network subsystem is unavailable".to_owned());
        };
        let trimmed = url.trim().to_owned();
        if trimmed.is_empty() {
            return Err("remote COG URL cannot be empty".to_owned());
        }
        let display_url = sanitize_url_for_display(&trimmed);
        let title = if title.trim().is_empty() {
            display_url.clone()
        } else {
            title.trim().to_owned()
        };
        let open_id = self.next_open_id.fetch_add(1, Ordering::Relaxed);
        let rt = Arc::clone(runtime);
        let tx = self.tx.clone();
        let cancel = runtime.shutdown.child_token();

        let handle = net.handle().clone();
        handle.spawn(async move {
            let cfg = HttpRangeConfig {
                extra_headers,
                ..HttpRangeConfig::default()
            };
            let range_source =
                match HttpRangeSource::open_async(Arc::clone(&net), &trimmed, cfg, &cancel).await {
                    Ok(src) => src,
                    Err(err) if err.is_cancelled() => return,
                    Err(err) => {
                        let _ = tx.send(RemoteOpenResult {
                            title,
                            display_url,
                            fly_to,
                            outcome: Err(err.to_string()),
                        });
                        (rt.wakeup)();
                        return;
                    }
                };

            if cancel.is_cancelled() {
                return;
            }

            // Parse TIFF IFD & bind geodetic Frame on `remote_pool` (a pure
            // non-Tokio thread pool so IFDs exceeding the 256 KiB head probe
            // can perform blocking range reads without deadlocking Tokio).
            let job_rt = Arc::clone(&rt);
            let job_cancel = cancel.clone();
            let tx_fail = tx.clone();
            let title_fail = title.clone();
            let url_fail = display_url.clone();
            let wakeup_fail = Arc::clone(&rt.wakeup);
            let job = DecodeJob {
                key: DecodeJobKey::Open(open_id),
                queued_bytes: 64 * 1024,
                cancel,
                work: Box::new(move || {
                    let res = open_remote_cog_source(
                        range_source.clone(),
                        scale_offset,
                        &job_rt.budget,
                        &job_cancel,
                    )
                    .map(|(source, frame)| (range_source, source, frame))
                    .map_err(|e| e.message);
                    if !job_cancel.is_cancelled() {
                        let _ = tx.send(RemoteOpenResult {
                            title,
                            display_url,
                            fly_to,
                            outcome: res,
                        });
                        (job_rt.wakeup)();
                    }
                }),
            };
            let _ = rt.try_submit_guarded(&rt.remote_pool, job, false, move || {
                let _ = tx_fail.send(RemoteOpenResult {
                    title: title_fail,
                    display_url: url_fail,
                    fly_to,
                    outcome: Err("remote I/O pool is busy or shutting down".to_owned()),
                });
                (wakeup_fail)();
            });
        });
        Ok(())
    }

    /// Drain all completed remote open results without blocking.
    pub(super) fn drain(&self) -> Vec<RemoteOpenResult> {
        let mut results = Vec::new();
        while let Ok(res) = self.rx.try_recv() {
            results.push(res);
        }
        results
    }
}
