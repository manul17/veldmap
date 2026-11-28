//! Background band-statistics sampling (`BandStatsTracker`) for scientific
//! (`R32Float`) raster layers with queue-rejection-safe RAII completion guards.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};

use veldmap_core::sync::{CancelToken, Wakeup};
use veldmap_core::tile::{BandKey, SourceId};
use veldmap_net::{HttpRangeSource, NetContext};
use veldmap_raster::{BandStats, DecodeJob, DecodeJobKey, RasterError, RasterSource};

use super::super::LayerId;
use super::super::runtime::OverlayRuntime;
use super::super::stack::{LayerStack, OverlayLayer};
use super::prefetch_coalesced_ranges;

enum BandStatsOutcome {
    Ready(Arc<BandStats>),
    Empty,
    Cancelled,
}

struct BandStatsMsg {
    layer_id: LayerId,
    job_id: u64,
    source_id: SourceId,
    band: BandKey,
    outcome: BandStatsOutcome,
}

fn compute_band_stats_outcome(
    rt: &OverlayRuntime,
    source: &dyn RasterSource,
    band: BandKey,
    cancel: &CancelToken,
) -> BandStatsOutcome {
    if cancel.is_cancelled() {
        return BandStatsOutcome::Cancelled;
    }
    let ctx = rt.decode_context(cancel);
    match source.band_stats(band, &ctx) {
        Ok(Some(stats)) => BandStatsOutcome::Ready(Arc::new(stats)),
        Ok(None) => BandStatsOutcome::Empty,
        Err(RasterError::Cancelled) => BandStatsOutcome::Cancelled,
        Err(_) if cancel.is_cancelled() => BandStatsOutcome::Cancelled,
        Err(_) => BandStatsOutcome::Empty,
    }
}

fn build_stats_decode_job(
    source: Arc<dyn RasterSource>,
    rt: Arc<OverlayRuntime>,
    cancel: CancelToken,
    pin: Option<veldmap_net::PrefetchPin>,
    guard: StatsCompletionGuard,
) -> DecodeJob {
    let source_id = guard.source_id;
    let band = guard.band;
    let layer_tag = guard.layer_id.0;
    let job_cancel = cancel.clone();
    DecodeJob {
        key: DecodeJobKey::BandStats {
            source: source_id,
            band,
            layer_tag,
        },
        queued_bytes: 16 * 1024,
        cancel,
        work: Box::new(move || {
            let _pin = pin;
            guard.finish(compute_band_stats_outcome(
                &rt,
                source.as_ref(),
                band,
                &job_cancel,
            ));
        }),
    }
}

struct StatsCompletionGuard {
    tx: Sender<BandStatsMsg>,
    wakeup: Wakeup,
    disarmed: Arc<AtomicBool>,
    layer_id: LayerId,
    job_id: u64,
    source_id: SourceId,
    band: BandKey,
    outcome: Option<BandStatsOutcome>,
}

impl StatsCompletionGuard {
    fn finish(mut self, outcome: BandStatsOutcome) {
        self.outcome = Some(outcome);
    }
}

impl Drop for StatsCompletionGuard {
    fn drop(&mut self) {
        if self.disarmed.load(Ordering::Acquire) {
            return;
        }
        let outcome = self.outcome.take().unwrap_or(BandStatsOutcome::Empty);
        let _ = self.tx.send(BandStatsMsg {
            layer_id: self.layer_id,
            job_id: self.job_id,
            source_id: self.source_id,
            band: self.band,
            outcome,
        });
        (self.wakeup)();
    }
}

struct StatsJobParams {
    id: LayerId,
    job_id: u64,
    source_id: SourceId,
    band: BandKey,
    source: Arc<dyn RasterSource>,
    cancel: CancelToken,
    rt: Arc<OverlayRuntime>,
    disarmed: Arc<AtomicBool>,
    guard: StatsCompletionGuard,
}

fn spawn_remote_band_stats(
    params: StatsJobParams,
    range_src: HttpRangeSource,
    net: Arc<NetContext>,
) {
    let StatsJobParams {
        id,
        job_id,
        source_id,
        band,
        source,
        cancel,
        rt,
        disarmed,
        guard,
    } = params;
    let handle = net.handle().clone();
    handle.spawn(async move {
        if cancel.is_cancelled() {
            guard.finish(BandStatsOutcome::Cancelled);
            return;
        }
        let raw_ranges = match source.plan_band_stats_byte_ranges(band) {
            Ok(raw) => raw,
            Err(err) => {
                tracing::warn!(
                    source = ?source_id,
                    ?band,
                    error = %err,
                    "Remote COG band-stats range planning failed"
                );
                guard.finish(BandStatsOutcome::Empty);
                return;
            }
        };
        let pin = match prefetch_coalesced_ranges(&range_src, raw_ranges, &cancel).await {
            Ok(pin) => pin,
            Err(err) if err.is_cancelled() => {
                guard.finish(BandStatsOutcome::Cancelled);
                return;
            }
            Err(err) => {
                tracing::warn!(
                    source = ?source_id,
                    ?band,
                    error = %err,
                    "Remote COG band-stats range prefetch failed"
                );
                guard.finish(BandStatsOutcome::Empty);
                return;
            }
        };
        if cancel.is_cancelled() {
            guard.finish(BandStatsOutcome::Cancelled);
            return;
        }

        // Route remote band_stats strictly to `remote_pool` (never `fast_pool`).
        // On rejection, send `Cancelled` rather than silently disarming so
        // `BandStatsState` returns to `NotRequested` and retries cleanly.
        let reject_tx = guard.tx.clone();
        let reject_wakeup = Arc::clone(&guard.wakeup);
        let job = build_stats_decode_job(source, Arc::clone(&rt), cancel, pin, guard);
        let disarm_on_reject = Arc::clone(&disarmed);
        let _ = rt.try_submit_guarded(&rt.remote_pool, job, true, move || {
            disarm_on_reject.store(true, Ordering::Release);
            let _ = reject_tx.send(BandStatsMsg {
                layer_id: id,
                job_id,
                source_id,
                band,
                outcome: BandStatsOutcome::Cancelled,
            });
            (reject_wakeup)();
        });
    });
}

fn submit_direct_band_stats(params: StatsJobParams) -> bool {
    let StatsJobParams {
        source,
        cancel,
        rt,
        disarmed,
        guard,
        ..
    } = params;
    let is_remote = source.is_remote();
    let job = build_stats_decode_job(source, Arc::clone(&rt), cancel, None, guard);
    // Preserve pool isolation (`docs/decisions.md` §11): a remote source
    // without an async prefetch handle still routes to `remote_pool`.
    let pool = rt.pool_for_source(is_remote);
    let disarm_on_reject = Arc::clone(&disarmed);
    rt.try_submit_guarded(pool, job, true, move || {
        disarm_on_reject.store(true, Ordering::Release);
    })
}

/// Manages background sampled physical [`BandStats`] jobs submitted on
/// `fast_pool` (local rasters) or `NetContext` prefetch + `remote_pool`
/// (remote COGs) and applies completions back onto matching layers.
pub(in crate::layers) struct BandStatsTracker {
    next_job_id: u64,
    tx: Sender<BandStatsMsg>,
    rx: Receiver<BandStatsMsg>,
}

impl BandStatsTracker {
    pub(in crate::layers) fn new() -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            next_job_id: 1,
            tx,
            rx,
        }
    }

    /// Submit a background band-statistics scan for `layer` if its active
    /// scientific stream still needs statistics.
    pub(in crate::layers) fn request_if_needed(
        &mut self,
        layer: &mut OverlayLayer,
        runtime: &Arc<OverlayRuntime>,
    ) {
        let Some(band) = layer.needs_band_stats() else {
            return;
        };
        let Some(asset) = layer.stats_target_asset() else {
            return;
        };
        let id = layer.id();
        let source = Arc::clone(asset.source());
        let source_id = asset.source_id();
        let remote_range = asset.remote_range().cloned();

        let job_id = self.next_job_id;
        self.next_job_id = self.next_job_id.saturating_add(1);
        let cancel = runtime.shutdown.child_token();
        layer.begin_stats_job(job_id, cancel.clone());

        // Load-bearing against `try_submit_guarded` queue-full rejection: if
        // submission fails, `disarmed` is flipped BEFORE `job` drops so
        // `StatsCompletionGuard::drop` does not send a spurious `Empty`
        // completion that would poison `BandStatsState::Empty`.
        let disarmed = Arc::new(AtomicBool::new(false));
        let guard = StatsCompletionGuard {
            tx: self.tx.clone(),
            wakeup: Arc::clone(&runtime.wakeup),
            disarmed: Arc::clone(&disarmed),
            layer_id: id,
            job_id,
            source_id,
            band,
            outcome: None,
        };
        let params = StatsJobParams {
            id,
            job_id,
            source_id,
            band,
            source: Arc::clone(&source),
            cancel,
            rt: Arc::clone(runtime),
            disarmed,
            guard,
        };

        if source.is_remote()
            && let (Some(range_src), Some(net)) = (remote_range, runtime.net.clone())
        {
            spawn_remote_band_stats(params, range_src, net);
        } else if !submit_direct_band_stats(params) {
            layer.cancel_stats_job();
        }
    }

    /// Drain all completed background band-statistics messages into `stack`,
    /// returning `true` if any layer's visible styling changed.
    pub(in crate::layers) fn drain_into(&self, stack: &mut LayerStack) -> bool {
        let mut changed = false;
        while let Ok(msg) = self.rx.try_recv() {
            match msg.outcome {
                BandStatsOutcome::Ready(stats) => {
                    for layer in stack.layers_mut() {
                        if layer.has_single_source(msg.source_id)
                            && layer.apply_band_stats_ready(
                                msg.source_id,
                                msg.band,
                                Arc::clone(&stats),
                            )
                        {
                            changed = true;
                        }
                    }
                }
                BandStatsOutcome::Empty => {
                    if let Some(layer) = stack.get_mut(msg.layer_id)
                        && layer.has_single_source(msg.source_id)
                        && layer.apply_band_stats_empty(msg.source_id, msg.band, msg.job_id)
                    {
                        changed = true;
                    }
                }
                BandStatsOutcome::Cancelled => {
                    if let Some(layer) = stack.get_mut(msg.layer_id)
                        && layer.has_single_source(msg.source_id)
                    {
                        let _ =
                            layer.apply_band_stats_cancelled(msg.source_id, msg.band, msg.job_id);
                    }
                }
            }
        }
        changed
    }
}
