//! Non-blocking remote Cloud-Optimized GeoTIFF opening ([`RemoteOpenTracker`])
//! for both single-asset COGs and concurrent multi-asset STAC Band Math recipes,
//! isolated onto `NetContext` + `veldmap-remote-io` (`remote_pool`).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};

use veldmap_core::geo::Frame;
use veldmap_core::sync::{CancelToken, Wakeup, lock_unpoisoned};
use veldmap_net::{HttpRangeConfig, HttpRangeSource, NetContext, sanitize_url_for_display};
use veldmap_raster::{DecodeJob, DecodeJobKey, ExternalBandMeta, RasterSource};

use super::opener::open_remote_cog_source;
use super::runtime::OverlayRuntime;
use super::{BandMathPreset, RemoteBandAssetSpec, RemoteBandMathRequest, RemoteCogRequest};

/// One opened constituent remote COG asset for a `BandMath` recipe.
pub(super) struct OpenedRemoteBandAsset {
    pub(super) label: String,
    pub(super) display_url: String,
    pub(super) range_source: HttpRangeSource,
    pub(super) source: Arc<dyn RasterSource>,
    pub(super) frame: Frame,
    pub(super) band_index: u16,
}

/// Completed outcome of a background remote Cloud-Optimized GeoTIFF open
/// (either a single COG or a multi-asset Band Math recipe).
pub(super) enum RemoteOpenResult {
    Single {
        title: String,
        display_url: String,
        is_scl: bool,
        fly_to: bool,
        outcome: Result<(HttpRangeSource, Arc<dyn RasterSource>, Frame), String>,
    },
    BandMath {
        title: String,
        display_url: String,
        preset: BandMathPreset,
        fly_to: bool,
        outcome: Result<(Vec<OpenedRemoteBandAsset>, Option<OpenedRemoteBandAsset>), String>,
    },
}

type RemoteOpenCallback =
    Box<dyn FnOnce(Result<(HttpRangeSource, Arc<dyn RasterSource>, Frame), String>) + Send>;

/// RAII drop guard guaranteeing every spawned remote COG open settles its
/// callback even if the async task panics or the `remote_pool` queue evicts
/// the `DecodeJob`, preventing multi-asset `BandMathOpenCollector` hangs.
struct RemoteOpenGuard {
    done: Arc<Mutex<Option<RemoteOpenCallback>>>,
    cancel: CancelToken,
}

impl RemoteOpenGuard {
    fn resolve(self, res: Result<(HttpRangeSource, Arc<dyn RasterSource>, Frame), String>) {
        let cb = lock_unpoisoned(&self.done).take();
        if !self.cancel.is_cancelled()
            && let Some(cb) = cb
        {
            cb(res);
        }
    }

    fn resolve_err(self, msg: String) {
        let cb = lock_unpoisoned(&self.done).take();
        if let Some(cb) = cb {
            cb(Err(msg));
        }
    }
}

impl Drop for RemoteOpenGuard {
    fn drop(&mut self) {
        let cb = lock_unpoisoned(&self.done).take();
        if !self.cancel.is_cancelled()
            && let Some(cb) = cb
        {
            cb(Err("remote COG open interrupted or evicted".to_owned()));
        }
    }
}

/// Spawn one remote COG HTTP Range probe + non-Tokio `remote_pool` IFD parse,
/// invoking `on_done` exactly once (unless `cancel` is already triggered).
#[expect(
    clippy::too_many_arguments,
    reason = "Internal spawn helper keeps single-COG and multi-asset recipe open paths unified."
)]
fn spawn_remote_asset_open<F>(
    net: Arc<NetContext>,
    rt: Arc<OverlayRuntime>,
    open_id: u64,
    url: String,
    extra_headers: Vec<(String, String)>,
    external_bands: Vec<ExternalBandMeta>,
    cancel: CancelToken,
    on_done: F,
) where
    F: FnOnce(Result<(HttpRangeSource, Arc<dyn RasterSource>, Frame), String>) + Send + 'static,
{
    let done = Arc::new(Mutex::new(Some(Box::new(on_done) as RemoteOpenCallback)));
    let guard = RemoteOpenGuard {
        done: Arc::clone(&done),
        cancel: cancel.clone(),
    };
    let handle = net.handle().clone();
    handle.spawn(async move {
        let cfg = HttpRangeConfig {
            extra_headers,
            ..HttpRangeConfig::default()
        };
        let range_source =
            match HttpRangeSource::open_async(Arc::clone(&net), &url, cfg, &cancel).await {
                Ok(src) => src,
                Err(err) if err.is_cancelled() => return,
                Err(err) => {
                    guard.resolve(Err(err.to_string()));
                    return;
                }
            };
        if cancel.is_cancelled() {
            return;
        }

        let job_rt = Arc::clone(&rt);
        let job_cancel = cancel.clone();
        let reject_done = Arc::clone(&done);
        let reject_cancel = cancel.clone();
        let job = DecodeJob {
            key: DecodeJobKey::Open(open_id),
            queued_bytes: 64 * 1024,
            cancel,
            work: Box::new(move || {
                let res = open_remote_cog_source(
                    range_source.clone(),
                    &external_bands,
                    &job_rt.budget,
                    &job_cancel,
                )
                .map(|(source, frame)| (range_source, source, frame))
                .map_err(|e| e.message);
                guard.resolve(res);
            }),
        };
        let _ = rt.try_submit_guarded(&rt.remote_pool, job, false, move || {
            let reject_guard = RemoteOpenGuard {
                done: reject_done,
                cancel: reject_cancel,
            };
            reject_guard.resolve_err("remote I/O pool is busy or shutting down".to_owned());
        });
    });
}

/// Shared synchronization barrier joining N concurrent required band opens +
/// optional `SCL` open into one `RemoteOpenResult::BandMath` emission.
struct BandMathOpenCollector {
    title: String,
    display_url: String,
    preset: BandMathPreset,
    fly_to: bool,
    remaining_required: usize,
    waiting_scl: bool,
    bands: Vec<Option<OpenedRemoteBandAsset>>,
    scl: Option<OpenedRemoteBandAsset>,
    settled: bool,
    cancel: CancelToken,
    tx: Sender<RemoteOpenResult>,
    wakeup: Wakeup,
}

impl BandMathOpenCollector {
    fn on_band_done(&mut self, slot: usize, res: Result<OpenedRemoteBandAsset, String>) {
        if self.settled {
            return;
        }
        match res {
            Ok(asset) => {
                if let Some(entry) = self.bands.get_mut(slot) {
                    *entry = Some(asset);
                }
                self.remaining_required = self.remaining_required.saturating_sub(1);
                self.maybe_finish();
            }
            Err(err) => {
                self.settled = true;
                self.cancel.cancel();
                let _ = self.tx.send(RemoteOpenResult::BandMath {
                    title: self.title.clone(),
                    display_url: self.display_url.clone(),
                    preset: self.preset,
                    fly_to: self.fly_to,
                    outcome: Err(err),
                });
                (self.wakeup)();
            }
        }
    }

    fn on_scl_done(&mut self, res: Result<OpenedRemoteBandAsset, String>) {
        if self.settled {
            return;
        }
        self.waiting_scl = false;
        match res {
            Ok(asset) => self.scl = Some(asset),
            Err(err) => {
                tracing::warn!(
                    title = %self.title,
                    error = %err,
                    "Optional SCL mask asset failed to open; continuing without cloud mask"
                );
            }
        }
        self.maybe_finish();
    }

    fn maybe_finish(&mut self) {
        if self.settled || self.remaining_required > 0 || self.waiting_scl {
            return;
        }
        self.settled = true;
        let bands: Vec<OpenedRemoteBandAsset> =
            self.bands.iter_mut().filter_map(Option::take).collect();
        let scl = self.scl.take();
        let _ = self.tx.send(RemoteOpenResult::BandMath {
            title: self.title.clone(),
            display_url: self.display_url.clone(),
            preset: self.preset,
            fly_to: self.fly_to,
            outcome: Ok((bands, scl)),
        });
        (self.wakeup)();
    }
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

    fn alloc_open_id(&self) -> u64 {
        self.next_open_id.fetch_add(1, Ordering::Relaxed)
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
            external_bands,
            is_scl,
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
        let open_id = self.alloc_open_id();
        let tx = self.tx.clone();
        let wakeup = Arc::clone(&runtime.wakeup);
        let cancel = runtime.shutdown.child_token();

        spawn_remote_asset_open(
            net,
            Arc::clone(runtime),
            open_id,
            trimmed,
            extra_headers,
            external_bands,
            cancel,
            move |outcome| {
                let _ = tx.send(RemoteOpenResult::Single {
                    title,
                    display_url,
                    is_scl,
                    fly_to,
                    outcome,
                });
                (wakeup)();
            },
        );
        Ok(())
    }

    /// Queue a non-blocking concurrent multi-asset STAC Band Math open onto
    /// `NetContext` + `remote_pool`, producing one `LayerKind::BandMath` layer
    /// and a single camera flight on completion.
    pub(super) fn begin_open_band_math(
        &self,
        runtime: &Arc<OverlayRuntime>,
        request: RemoteBandMathRequest,
    ) -> Result<(), String> {
        let RemoteBandMathRequest {
            title,
            preset,
            bands,
            scl,
            fly_to,
        } = request;
        let Some(net) = runtime.net.clone() else {
            return Err("network subsystem is unavailable".to_owned());
        };
        let required = preset.formula().required_bands();
        if bands.len() != required {
            return Err(format!(
                "recipe '{}' requires {required} bands, got {}",
                preset.short_label(),
                bands.len()
            ));
        }
        if let Some(empty_band) = bands.iter().find(|b| b.url.trim().is_empty()) {
            return Err(format!(
                "band '{}' remote COG URL cannot be empty",
                empty_band.label
            ));
        }
        if let Some(scl_spec) = &scl
            && scl_spec.url.trim().is_empty()
        {
            return Err("SCL remote COG URL cannot be empty".to_owned());
        }
        let display_url = bands
            .first()
            .map(|b| sanitize_url_for_display(b.url.trim()))
            .unwrap_or_default();
        let title = if title.trim().is_empty() {
            preset.short_label().to_owned()
        } else {
            title.trim().to_owned()
        };
        let cancel = runtime.shutdown.child_token();
        let collector = Arc::new(Mutex::new(BandMathOpenCollector {
            title,
            display_url,
            preset,
            fly_to,
            remaining_required: bands.len(),
            waiting_scl: scl.is_some(),
            bands: (0..bands.len()).map(|_| None).collect(),
            scl: None,
            settled: false,
            cancel: cancel.clone(),
            tx: self.tx.clone(),
            wakeup: Arc::clone(&runtime.wakeup),
        }));

        let spawn_band = |spec: RemoteBandAssetSpec,
                          slot: Option<usize>,
                          net: Arc<NetContext>,
                          cancel: CancelToken| {
            let RemoteBandAssetSpec {
                label,
                url,
                extra_headers,
                external_bands,
                band_index,
            } = spec;
            let trimmed = url.trim().to_owned();
            let disp = sanitize_url_for_display(&trimmed);
            let open_id = self.alloc_open_id();
            let col = Arc::clone(&collector);
            spawn_remote_asset_open(
                net,
                Arc::clone(runtime),
                open_id,
                trimmed,
                extra_headers,
                external_bands,
                cancel,
                move |res| {
                    let mapped = res
                        .map(|(range_source, source, frame)| OpenedRemoteBandAsset {
                            label: label.clone(),
                            display_url: disp,
                            range_source,
                            source,
                            frame,
                            band_index,
                        })
                        .map_err(|err| format!("{label}: {err}"));
                    let mut guard = lock_unpoisoned(&col);
                    match slot {
                        Some(idx) => guard.on_band_done(idx, mapped),
                        None => guard.on_scl_done(mapped),
                    }
                },
            );
        };

        for (slot, spec) in bands.into_iter().enumerate() {
            spawn_band(spec, Some(slot), Arc::clone(&net), cancel.clone());
        }
        if let Some(spec) = scl {
            spawn_band(spec, None, net, cancel);
        }

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
