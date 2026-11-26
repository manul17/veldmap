//! Background band-statistics sampling (`BandStatsTracker`) for scientific
//! (`R32Float`) raster layers with queue-rejection-safe RAII completion guards.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};

use veldmap_core::tile::{BandKey, SourceId};
use veldmap_net::PrefetchPin;
use veldmap_raster::{BandStats, DecodeJob, DecodeJobKey, RasterError, coalesce_byte_ranges};

use super::super::LayerId;
use super::super::runtime::OverlayRuntime;
use super::super::stack::{LayerStack, OverlayLayer};

enum BandStatsOutcome {
    Ready(Box<BandStats>),
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

struct StatsCompletionGuard {
    tx: Sender<BandStatsMsg>,
    wakeup: Arc<dyn Fn() + Send + Sync>,
    disarmed: Arc<AtomicBool>,
    layer_id: LayerId,
    job_id: u64,
    source_id: SourceId,
    band: BandKey,
    outcome: Option<BandStatsOutcome>,
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

/// Manages background sampled physical [`BandStats`] jobs submitted on the
/// fast decode pool and applies completions back onto matching layers.
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
        let id = layer.id();
        let source = Arc::clone(layer.source());
        let source_id = layer.source_id();

        let job_id = self.next_job_id;
        self.next_job_id = self.next_job_id.saturating_add(1);
        let cancel = runtime.shutdown.child_token();
        layer.begin_stats_job(job_id, cancel.clone());

        let rt = Arc::clone(runtime);
        let tx = self.tx.clone();

        // Load-bearing against `try_submit_guarded` queue-full rejection: if
        // submission fails, `disarmed` is flipped BEFORE `job` drops so
        // `StatsCompletionGuard::drop` does not send a spurious `Empty`
        // completion that would poison `BandStatsState::Empty`.
        let disarmed = Arc::new(AtomicBool::new(false));
        let guard = StatsCompletionGuard {
            tx,
            wakeup: Arc::clone(&rt.wakeup),
            disarmed: Arc::clone(&disarmed),
            layer_id: id,
            job_id,
            source_id,
            band,
            outcome: None,
        };

        if source.is_remote()
            && let (Some(range_src), Some(net)) =
                (layer.remote_range().cloned(), runtime.net.clone())
        {
            let handle = net.handle().clone();
            handle.spawn(async move {
                let mut guard = guard;
                if cancel.is_cancelled() {
                    guard.outcome = Some(BandStatsOutcome::Cancelled);
                    return;
                }
                let ranges: Vec<(u64, u64)> = {
                    let mut raw = source.plan_band_stats_byte_ranges(band).unwrap_or_default();
                    coalesce_byte_ranges(&mut raw, 64 * 1024, 4 * 1024 * 1024);
                    raw.into_iter().map(|r| (r.offset, r.len)).collect()
                };
                let pin: Option<PrefetchPin> = if ranges.is_empty() {
                    None
                } else {
                    match range_src.prefetch_ranges_async(&ranges, &cancel).await {
                        Ok(pin) => Some(pin),
                        Err(err) if err.is_cancelled() => {
                            guard.outcome = Some(BandStatsOutcome::Cancelled);
                            return;
                        }
                        Err(err) => {
                            tracing::warn!(
                                source = ?source_id,
                                ?band,
                                error = %err,
                                "Remote COG band-stats range prefetch failed"
                            );
                            guard.outcome = Some(BandStatsOutcome::Empty);
                            return;
                        }
                    }
                };
                if cancel.is_cancelled() {
                    guard.outcome = Some(BandStatsOutcome::Cancelled);
                    return;
                }

                // Route remote band_stats strictly to `remote_pool` (never `fast_pool`).
                // On rejection, send `Cancelled` rather than disarming so
                // `BandStatsState` returns to `NotLoaded` and retries cleanly.
                let job_rt = Arc::clone(&rt);
                let job_cancel = cancel.clone();
                let reject_tx = guard.tx.clone();
                let reject_wakeup = Arc::clone(&guard.wakeup);
                let job = DecodeJob {
                    key: DecodeJobKey::BandStats {
                        source: source_id,
                        band,
                        layer_tag: id.0,
                    },
                    queued_bytes: 16 * 1024,
                    cancel,
                    work: Box::new(move || {
                        let _pin = pin;
                        let mut guard = guard;
                        if job_cancel.is_cancelled() {
                            guard.outcome = Some(BandStatsOutcome::Cancelled);
                            return;
                        }
                        let ctx = job_rt.decode_context(&job_cancel);
                        guard.outcome = Some(match source.band_stats(band, &ctx) {
                            Ok(Some(stats)) => BandStatsOutcome::Ready(Box::new(stats)),
                            Ok(None) => BandStatsOutcome::Empty,
                            Err(RasterError::Cancelled) => BandStatsOutcome::Cancelled,
                            Err(_) if job_cancel.is_cancelled() => BandStatsOutcome::Cancelled,
                            Err(_) => BandStatsOutcome::Empty,
                        });
                    }),
                };
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
            return;
        }

        let is_remote = source.is_remote();
        let job = DecodeJob {
            key: DecodeJobKey::BandStats {
                source: source_id,
                band,
                layer_tag: id.0,
            },
            queued_bytes: 16 * 1024,
            cancel: cancel.clone(),
            work: Box::new(move || {
                let mut guard = guard;
                if cancel.is_cancelled() {
                    guard.outcome = Some(BandStatsOutcome::Cancelled);
                    return;
                }
                let ctx = rt.decode_context(&cancel);
                guard.outcome = Some(match source.band_stats(band, &ctx) {
                    Ok(Some(stats)) => BandStatsOutcome::Ready(Box::new(stats)),
                    Ok(None) => BandStatsOutcome::Empty,
                    Err(RasterError::Cancelled) => BandStatsOutcome::Cancelled,
                    Err(_) if cancel.is_cancelled() => BandStatsOutcome::Cancelled,
                    Err(_) => BandStatsOutcome::Empty,
                });
            }),
        };
        let pool = if is_remote {
            &runtime.remote_pool
        } else {
            &runtime.fast_pool
        };
        let disarm_on_reject = Arc::clone(&disarmed);
        let submitted = runtime.try_submit_guarded(pool, job, true, move || {
            disarm_on_reject.store(true, Ordering::Release);
        });
        if !submitted {
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
                        if layer.source_id() == msg.source_id
                            && layer.apply_band_stats_ready(msg.band, (*stats).clone())
                        {
                            changed = true;
                        }
                    }
                }
                BandStatsOutcome::Empty => {
                    if let Some(layer) = stack.get_mut(msg.layer_id)
                        && layer.source_id() == msg.source_id
                        && layer.apply_band_stats_empty(msg.band, msg.job_id)
                    {
                        changed = true;
                    }
                }
                BandStatsOutcome::Cancelled => {
                    if let Some(layer) = stack.get_mut(msg.layer_id)
                        && layer.source_id() == msg.source_id
                    {
                        let _ = layer.apply_band_stats_cancelled(msg.band, msg.job_id);
                    }
                }
            }
        }
        changed
    }
}
