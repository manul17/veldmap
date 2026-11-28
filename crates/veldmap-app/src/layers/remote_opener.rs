//! Non-blocking remote Cloud-Optimized GeoTIFF opening ([`RemoteOpenTracker`])
//! for both single-asset COGs and concurrent multi-asset STAC or heterogeneous
//! local/remote Band Math recipes, isolated onto `NetContext` +
//! `veldmap-remote-io` (`remote_pool`).

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};

use veldmap_core::geo::Frame;
use veldmap_core::sync::{CancelToken, Wakeup, lock_unpoisoned};
use veldmap_net::{
    HttpRangeConfig, HttpRangeSource, NetContext, sanitize_url_for_display,
    sanitize_url_for_persistence,
};
use veldmap_raster::{DecodeJob, DecodeJobKey, ExternalBandMeta, RasterSource};

use crate::clamp_opacity;

use super::asset::{RemoteAssetMeta, StacCogAuthMode};
use super::opener::{open_local_raster, open_remote_cog_source};
use super::runtime::OverlayRuntime;
use super::{
    BandMathAssetOpenSpec, BandMathPreset, BandMathStyleSpec, CrsOverride, LayerInitOptions,
    RemoteBandMathRequest, RemoteCogRequest,
};

/// Origin payload for one opened constituent Band Math asset (local file or
/// remote COG).
pub(super) enum OpenedBandAssetOrigin {
    Local {
        path: PathBuf,
        crs_override: Option<CrsOverride>,
    },
    Remote {
        meta: RemoteAssetMeta,
        range_source: HttpRangeSource,
    },
}

/// One opened constituent local or remote asset for a `BandMath` recipe.
pub(super) struct OpenedBandAsset {
    pub(super) label: String,
    pub(super) origin: OpenedBandAssetOrigin,
    pub(super) source: Arc<dyn RasterSource>,
    pub(super) frame: Frame,
    pub(super) band_index: u16,
}

/// Completed payload of a single remote COG open.
pub(super) struct RemoteSingleOpenPayload {
    pub(super) epoch: u64,
    pub(super) title: String,
    pub(super) meta: RemoteAssetMeta,
    pub(super) options: LayerInitOptions,
    pub(super) fly_to: bool,
    pub(super) restore_id: Option<u64>,
    pub(super) outcome: Result<(HttpRangeSource, Arc<dyn RasterSource>, Frame), String>,
}

/// Completed payload of a multi-asset Band Math open.
pub(super) struct RemoteBandMathOpenPayload {
    pub(super) epoch: u64,
    pub(super) title: String,
    pub(super) display_url: String,
    pub(super) preset: BandMathPreset,
    pub(super) visible: bool,
    pub(super) opacity: f32,
    pub(super) style: Option<BandMathStyleSpec>,
    pub(super) fly_to: bool,
    pub(super) restore_id: Option<u64>,
    pub(super) outcome: Result<(Vec<OpenedBandAsset>, Option<OpenedBandAsset>), String>,
}

/// Completed outcome of a background remote COG or multi-asset Band Math open.
pub(super) enum RemoteOpenResult {
    Single(RemoteSingleOpenPayload),
    BandMath(RemoteBandMathOpenPayload),
}

type OpenCallback<T> = Box<dyn FnOnce(Result<T, String>) + Send>;

/// RAII drop guard guaranteeing every spawned remote COG or local band-math
/// asset open settles its callback even if the async task panics or the
/// `remote_pool` queue evicts the `DecodeJob`, preventing multi-asset
/// `BandMathOpenCollector` hangs.
struct OpenCallbackGuard<T> {
    done: Arc<Mutex<Option<OpenCallback<T>>>>,
    cancel: CancelToken,
    interrupted_msg: &'static str,
}

impl<T: 'static> OpenCallbackGuard<T> {
    fn new_pair<F>(
        on_done: F,
        cancel: &CancelToken,
        interrupted_msg: &'static str,
    ) -> (Self, Arc<Mutex<Option<OpenCallback<T>>>>)
    where
        F: FnOnce(Result<T, String>) + Send + 'static,
    {
        let done: Arc<Mutex<Option<OpenCallback<T>>>> =
            Arc::new(Mutex::new(Some(Box::new(on_done))));
        let guard = Self {
            done: Arc::clone(&done),
            cancel: cancel.clone(),
            interrupted_msg,
        };
        (guard, done)
    }

    fn resolve(self, res: Result<T, String>) {
        let cb = lock_unpoisoned(&self.done).take();
        if !self.cancel.is_cancelled()
            && let Some(cb) = cb
        {
            cb(res);
        }
    }
}

impl<T> Drop for OpenCallbackGuard<T> {
    fn drop(&mut self) {
        let cb = lock_unpoisoned(&self.done).take();
        if !self.cancel.is_cancelled()
            && let Some(cb) = cb
        {
            cb(Err(self.interrupted_msg.to_owned()));
        }
    }
}

fn submit_guarded_open_job<T, W>(
    rt: &OverlayRuntime,
    open_id: u64,
    cancel: CancelToken,
    guard: OpenCallbackGuard<T>,
    done: Arc<Mutex<Option<OpenCallback<T>>>>,
    busy_msg: &'static str,
    work: W,
) where
    T: Send + 'static,
    W: FnOnce() -> Result<T, String> + Send + 'static,
{
    let reject_cancel = cancel.clone();
    let job = DecodeJob {
        key: DecodeJobKey::Open(open_id),
        queued_bytes: 64 * 1024,
        cancel,
        work: Box::new(move || {
            guard.resolve(work());
        }),
    };
    let _ = rt.try_submit_guarded(&rt.remote_pool, job, false, move || {
        let cb = lock_unpoisoned(&done).take();
        if !reject_cancel.is_cancelled()
            && let Some(cb) = cb
        {
            cb(Err(busy_msg.to_owned()));
        }
    });
}

struct RemoteAssetMetaInput {
    url: String,
    auth_mode: StacCogAuthMode,
    external_bands: Vec<ExternalBandMeta>,
    is_scl: bool,
    stac_item_href: Option<String>,
    asset_key: Option<String>,
}

/// Trim, validate, and sanitize a remote COG URL once, returning
/// `(trimmed_url, RemoteAssetMeta)`.
fn build_remote_asset_meta(
    input: RemoteAssetMetaInput,
    error_prefix: Option<&str>,
) -> Result<(String, RemoteAssetMeta), String> {
    let trimmed = input.url.trim().to_owned();
    if trimmed.is_empty() {
        return Err(match error_prefix {
            Some(prefix) => format!("{prefix} remote COG URL cannot be empty"),
            None => "remote COG URL cannot be empty".to_owned(),
        });
    }
    let display_url = sanitize_url_for_display(&trimmed);
    let canonical_url =
        sanitize_url_for_persistence(&trimmed).ok_or_else(|| match error_prefix {
            Some(prefix) => format!("{prefix} has an invalid remote COG URL '{display_url}'"),
            None => format!("invalid remote COG URL '{display_url}'"),
        })?;
    let meta = RemoteAssetMeta {
        canonical_url,
        display_url,
        auth_mode: input.auth_mode,
        external_bands: input.external_bands,
        is_scl: input.is_scl,
        stac_item_href: input
            .stac_item_href
            .as_deref()
            .and_then(sanitize_url_for_persistence),
        asset_key: input.asset_key,
    };
    Ok((trimmed, meta))
}

struct RemoteAssetOpenSpec {
    net: Arc<NetContext>,
    rt: Arc<OverlayRuntime>,
    open_id: u64,
    url: String,
    extra_headers: Vec<(String, String)>,
    external_bands: Vec<ExternalBandMeta>,
    cancel: CancelToken,
}

/// Spawn one remote COG HTTP Range probe + non-Tokio `remote_pool` IFD parse,
/// invoking `on_done` exactly once (unless `cancel` is already triggered).
fn spawn_remote_asset_open<F>(spec: RemoteAssetOpenSpec, on_done: F)
where
    F: FnOnce(Result<(HttpRangeSource, Arc<dyn RasterSource>, Frame), String>) + Send + 'static,
{
    let RemoteAssetOpenSpec {
        net,
        rt,
        open_id,
        url,
        extra_headers,
        external_bands,
        cancel,
    } = spec;
    let (guard, done) =
        OpenCallbackGuard::new_pair(on_done, &cancel, "remote COG open interrupted or evicted");
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
        submit_guarded_open_job(
            &rt,
            open_id,
            cancel,
            guard,
            done,
            "remote I/O pool is busy or shutting down",
            move || {
                open_remote_cog_source(
                    range_source.clone(),
                    &external_bands,
                    &job_rt.budget,
                    &job_cancel,
                )
                .map(|(source, frame)| (range_source, source, frame))
                .map_err(|e| e.message)
            },
        );
    });
}

fn spawn_local_band_open<F>(
    rt: Arc<OverlayRuntime>,
    open_id: u64,
    path: PathBuf,
    crs_override: Option<CrsOverride>,
    cancel: CancelToken,
    on_done: F,
) where
    F: FnOnce(Result<(Arc<dyn RasterSource>, Frame), String>) + Send + 'static,
{
    let (guard, done) = OpenCallbackGuard::new_pair(
        on_done,
        &cancel,
        "local band asset open interrupted or evicted",
    );
    let job_rt = Arc::clone(&rt);
    let job_cancel = cancel.clone();
    submit_guarded_open_job(
        &rt,
        open_id,
        cancel,
        guard,
        done,
        "remote/band I/O pool is busy or shutting down",
        move || {
            open_local_raster(
                &path,
                &job_rt.fingerprint_cache,
                &job_rt.budget,
                false,
                crs_override.as_ref(),
                &job_cancel,
            )
            .map_err(|e| e.message)
        },
    );
}

/// Shared synchronization barrier joining N concurrent required band opens +
/// optional `SCL` open into one `RemoteOpenResult::BandMath` emission.
struct BandMathOpenCollector {
    epoch: u64,
    title: String,
    display_url: String,
    preset: BandMathPreset,
    visible: bool,
    opacity: f32,
    style: Option<BandMathStyleSpec>,
    fly_to: bool,
    restore_id: Option<u64>,
    remaining_required: usize,
    waiting_scl: bool,
    bands: Vec<Option<OpenedBandAsset>>,
    scl: Option<OpenedBandAsset>,
    settled: bool,
    cancel: CancelToken,
    tx: Sender<RemoteOpenResult>,
    wakeup: Wakeup,
}

impl BandMathOpenCollector {
    fn emit(&mut self, outcome: Result<(Vec<OpenedBandAsset>, Option<OpenedBandAsset>), String>) {
        self.settled = true;
        let _ = self
            .tx
            .send(RemoteOpenResult::BandMath(RemoteBandMathOpenPayload {
                epoch: self.epoch,
                title: self.title.clone(),
                display_url: self.display_url.clone(),
                preset: self.preset,
                visible: self.visible,
                opacity: self.opacity,
                style: self.style.take(),
                fly_to: self.fly_to,
                restore_id: self.restore_id,
                outcome,
            }));
        (self.wakeup)();
    }

    fn on_band_done(&mut self, slot: usize, res: Result<OpenedBandAsset, String>) {
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
                self.cancel.cancel();
                self.emit(Err(err));
            }
        }
    }

    fn on_scl_done(&mut self, res: Result<OpenedBandAsset, String>) {
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
        let bands: Vec<OpenedBandAsset> = self.bands.iter_mut().filter_map(Option::take).collect();
        let scl = self.scl.take();
        self.emit(Ok((bands, scl)));
    }
}

fn finish_collector_slot(
    collector: &Mutex<BandMathOpenCollector>,
    slot: Option<usize>,
    result: Result<OpenedBandAsset, String>,
) {
    let mut guard = lock_unpoisoned(collector);
    match slot {
        Some(idx) => guard.on_band_done(idx, result),
        None => guard.on_scl_done(result),
    }
}

enum ValidatedBandAsset {
    Local {
        label: String,
        path: PathBuf,
        crs_override: Option<CrsOverride>,
        band_index: u16,
    },
    Remote {
        label: String,
        url: String,
        meta: RemoteAssetMeta,
        extra_headers: Vec<(String, String)>,
        external_bands: Vec<ExternalBandMeta>,
        band_index: u16,
    },
}

impl ValidatedBandAsset {
    fn display_label(&self) -> String {
        match self {
            Self::Local { path, .. } => path.display().to_string(),
            Self::Remote { meta, .. } => meta.display_url.clone(),
        }
    }
}

fn validate_band_math_asset(
    spec: BandMathAssetOpenSpec,
    is_scl: bool,
) -> Result<ValidatedBandAsset, String> {
    match spec {
        BandMathAssetOpenSpec::Local {
            label,
            path,
            crs_override,
            band_index,
        } => {
            if path.as_os_str().is_empty() {
                return Err(format!("band '{label}' local path cannot be empty"));
            }
            Ok(ValidatedBandAsset::Local {
                label,
                path,
                crs_override,
                band_index,
            })
        }
        BandMathAssetOpenSpec::Remote {
            label,
            url,
            auth_mode,
            extra_headers,
            external_bands,
            stac_item_href,
            asset_key,
            band_index,
        } => {
            let prefix = format!("band '{label}'");
            let (trimmed, meta) = build_remote_asset_meta(
                RemoteAssetMetaInput {
                    url,
                    auth_mode,
                    external_bands: external_bands.clone(),
                    is_scl,
                    stac_item_href,
                    asset_key,
                },
                Some(&prefix),
            )?;
            Ok(ValidatedBandAsset::Remote {
                label,
                url: trimmed,
                meta,
                extra_headers,
                external_bands,
                band_index,
            })
        }
    }
}

/// Manages non-blocking remote COG and multi-asset Band Math open requests over
/// `NetContext` + `remote_pool`, isolating remote HTTP Range probing and IFD
/// parsing from local disk I/O and `fast_pool`.
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
        epoch: u64,
        workspace_cancel: &CancelToken,
        request: RemoteCogRequest,
    ) -> Result<(), String> {
        let RemoteCogRequest {
            url,
            title,
            auth_mode,
            extra_headers,
            external_bands,
            is_scl,
            stac_item_href,
            asset_key,
            visible,
            opacity,
            style,
            fly_to,
            restore_id,
        } = request;
        let Some(net) = runtime.net.clone() else {
            return Err("network subsystem is unavailable".to_owned());
        };
        let (trimmed, meta) = build_remote_asset_meta(
            RemoteAssetMetaInput {
                url,
                auth_mode,
                external_bands: external_bands.clone(),
                is_scl,
                stac_item_href,
                asset_key,
            },
            None,
        )?;
        let title = if title.trim().is_empty() {
            meta.canonical_url.clone()
        } else {
            title.trim().to_owned()
        };
        let options = LayerInitOptions {
            visible,
            opacity: clamp_opacity(opacity),
            crs_override: None,
            style,
        };
        let open_id = self.alloc_open_id();
        let tx = self.tx.clone();
        let wakeup = Arc::clone(&runtime.wakeup);
        let cancel = workspace_cancel.child_token();

        spawn_remote_asset_open(
            RemoteAssetOpenSpec {
                net,
                rt: Arc::clone(runtime),
                open_id,
                url: trimmed,
                extra_headers,
                external_bands,
                cancel,
            },
            move |outcome| {
                let _ = tx.send(RemoteOpenResult::Single(RemoteSingleOpenPayload {
                    epoch,
                    title,
                    meta,
                    options,
                    fly_to,
                    restore_id,
                    outcome,
                }));
                (wakeup)();
            },
        );
        Ok(())
    }

    fn spawn_band_math_asset(
        &self,
        runtime: &Arc<OverlayRuntime>,
        collector: &Arc<Mutex<BandMathOpenCollector>>,
        asset: ValidatedBandAsset,
        slot: Option<usize>,
        net: Option<Arc<NetContext>>,
        cancel: CancelToken,
    ) {
        let open_id = self.alloc_open_id();
        let col = Arc::clone(collector);
        match asset {
            ValidatedBandAsset::Local {
                label,
                path,
                crs_override,
                band_index,
            } => {
                let path_for_origin = path.clone();
                let crs_for_origin = crs_override.clone();
                spawn_local_band_open(
                    Arc::clone(runtime),
                    open_id,
                    path,
                    crs_override,
                    cancel,
                    move |res| {
                        let mapped = res
                            .map(|(source, frame)| OpenedBandAsset {
                                label: label.clone(),
                                origin: OpenedBandAssetOrigin::Local {
                                    path: path_for_origin,
                                    crs_override: crs_for_origin,
                                },
                                source,
                                frame,
                                band_index,
                            })
                            .map_err(|err| format!("{label}: {err}"));
                        finish_collector_slot(&col, slot, mapped);
                    },
                );
            }
            ValidatedBandAsset::Remote {
                label,
                url,
                meta,
                extra_headers,
                external_bands,
                band_index,
            } => {
                let Some(net) = net else {
                    return;
                };
                spawn_remote_asset_open(
                    RemoteAssetOpenSpec {
                        net,
                        rt: Arc::clone(runtime),
                        open_id,
                        url,
                        extra_headers,
                        external_bands,
                        cancel,
                    },
                    move |res| {
                        let mapped = res
                            .map(|(range_source, source, frame)| OpenedBandAsset {
                                label: label.clone(),
                                origin: OpenedBandAssetOrigin::Remote { meta, range_source },
                                source,
                                frame,
                                band_index,
                            })
                            .map_err(|err| format!("{label}: {err}"));
                        finish_collector_slot(&col, slot, mapped);
                    },
                );
            }
        }
    }

    /// Queue a non-blocking concurrent multi-asset STAC or heterogeneous
    /// local/remote Band Math open, producing one `LayerKind::BandMath` layer
    /// and a single camera flight on completion.
    pub(super) fn begin_open_band_math(
        &self,
        runtime: &Arc<OverlayRuntime>,
        epoch: u64,
        workspace_cancel: &CancelToken,
        request: RemoteBandMathRequest,
    ) -> Result<(), String> {
        let RemoteBandMathRequest {
            title,
            preset,
            bands,
            scl,
            visible,
            opacity,
            style,
            fly_to,
            restore_id,
        } = request;
        let needs_net = bands.iter().any(BandMathAssetOpenSpec::is_remote)
            || scl.as_ref().is_some_and(BandMathAssetOpenSpec::is_remote);
        let net = if needs_net {
            Some(
                runtime
                    .net
                    .clone()
                    .ok_or_else(|| "network subsystem is unavailable".to_owned())?,
            )
        } else {
            None
        };
        let formula = style
            .as_ref()
            .map_or_else(|| preset.formula(), |s| s.formula);
        let required = formula.required_bands();
        if bands.len() != required {
            return Err(format!(
                "recipe '{}' requires {required} bands, got {}",
                preset.short_label(),
                bands.len()
            ));
        }

        let validated_bands: Vec<ValidatedBandAsset> = bands
            .into_iter()
            .map(|spec| validate_band_math_asset(spec, false))
            .collect::<Result<_, _>>()?;
        let validated_scl = scl
            .map(|spec| validate_band_math_asset(spec, true))
            .transpose()?;

        let display_url = validated_bands
            .first()
            .map(ValidatedBandAsset::display_label)
            .unwrap_or_default();
        let title = if title.trim().is_empty() {
            preset.short_label().to_owned()
        } else {
            title.trim().to_owned()
        };
        let cancel = workspace_cancel.child_token();
        let collector = Arc::new(Mutex::new(BandMathOpenCollector {
            epoch,
            title,
            display_url,
            preset,
            visible,
            opacity: clamp_opacity(opacity),
            style,
            fly_to,
            restore_id,
            remaining_required: validated_bands.len(),
            waiting_scl: validated_scl.is_some(),
            bands: (0..validated_bands.len()).map(|_| None).collect(),
            scl: None,
            settled: false,
            cancel: cancel.clone(),
            tx: self.tx.clone(),
            wakeup: Arc::clone(&runtime.wakeup),
        }));

        for (slot, asset) in validated_bands.into_iter().enumerate() {
            self.spawn_band_math_asset(
                runtime,
                &collector,
                asset,
                Some(slot),
                net.clone(),
                cancel.clone(),
            );
        }
        if let Some(asset) = validated_scl {
            self.spawn_band_math_asset(runtime, &collector, asset, None, net, cancel);
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
