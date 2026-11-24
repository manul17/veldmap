//! Lazy singleflight percentile-stretch computation for >8-bit GeoTIFF display
//! mapping, distinguishing leader cancellation (retryable by uncancelled
//! followers) from terminal sampling errors (propagated to current-round
//! followers without serial retry storms).

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Mutex};

use veldmap_core::budget::DecodeBudgetTracker;
use veldmap_core::sync::{CancelToken, lock_unpoisoned};

use crate::RasterError;
use crate::pipeline::grid;
use crate::pipeline::radiometry::{self, DisplayMapping, PixelLayout, percentile_stretch};
use crate::pipeline::singleflight::SingleflightSlot;

use super::chunk_reader::DecoderPool;
use super::tags::{map_tiff_error, typed_samples};

#[derive(Clone)]
enum RoundOutcome {
    Ready(DisplayMapping),
    LeaderCancelled,
    Failed(RasterError),
}

type StretchSlot = SingleflightSlot<RoundOutcome>;

enum CacheState {
    Uncomputed,
    InFlight(Arc<StretchSlot>),
    Ready(DisplayMapping),
}

struct RoundGuard<'a> {
    cache: &'a Mutex<CacheState>,
    slot: Arc<StretchSlot>,
    completed: bool,
}

impl Drop for RoundGuard<'_> {
    fn drop(&mut self) {
        if self.completed {
            return;
        }
        {
            let mut guard = lock_unpoisoned(self.cache);
            if let CacheState::InFlight(cur) = &*guard
                && Arc::ptr_eq(cur, &self.slot)
            {
                *guard = CacheState::Uncomputed;
            }
        }
        self.slot
            .resolve(RoundOutcome::Failed(RasterError::DecoderPanicked(
                "display-mapping sampling leader panicked".to_string(),
            )));
    }
}

pub(super) struct StretchCache {
    state: Mutex<CacheState>,
}

impl StretchCache {
    pub(super) fn new(initial: Option<DisplayMapping>) -> Self {
        let state = match initial {
            Some(mapping) => CacheState::Ready(mapping),
            None => CacheState::Uncomputed,
        };
        Self {
            state: Mutex::new(state),
        }
    }

    pub(super) fn get_or_compute(
        &self,
        req: &StretchRequest<'_>,
    ) -> Result<DisplayMapping, RasterError> {
        loop {
            enum Role {
                Ready(DisplayMapping),
                Leader(Arc<StretchSlot>),
                Follower(Arc<StretchSlot>),
            }

            let role = {
                let mut guard = lock_unpoisoned(&self.state);
                match &*guard {
                    CacheState::Ready(ready) => Role::Ready(*ready),
                    CacheState::InFlight(slot) => Role::Follower(Arc::clone(slot)),
                    CacheState::Uncomputed => {
                        if req.cancel.is_cancelled() {
                            return Err(RasterError::Cancelled);
                        }
                        let slot = Arc::new(SingleflightSlot::new());
                        *guard = CacheState::InFlight(Arc::clone(&slot));
                        Role::Leader(slot)
                    }
                }
            };

            match role {
                Role::Ready(ready) => return Ok(ready),
                Role::Leader(slot) => {
                    let mut guard = RoundGuard {
                        cache: &self.state,
                        slot: Arc::clone(&slot),
                        completed: false,
                    };
                    match catch_unwind(AssertUnwindSafe(|| compute_mapping(req))) {
                        Ok(Ok(computed)) => {
                            {
                                let mut cache = lock_unpoisoned(&self.state);
                                *cache = CacheState::Ready(computed);
                            }
                            slot.resolve(RoundOutcome::Ready(computed));
                            guard.completed = true;
                            return Ok(computed);
                        }
                        Ok(Err(err)) => {
                            {
                                let mut cache = lock_unpoisoned(&self.state);
                                if let CacheState::InFlight(cur) = &*cache
                                    && Arc::ptr_eq(cur, &slot)
                                {
                                    *cache = CacheState::Uncomputed;
                                }
                            }
                            if err.is_cancelled() {
                                slot.resolve(RoundOutcome::LeaderCancelled);
                            } else {
                                // Current-round followers receive this error
                                // directly instead of triggering serial retries.
                                slot.resolve(RoundOutcome::Failed(err.clone()));
                            }
                            guard.completed = true;
                            return Err(err);
                        }
                        Err(_) => {
                            drop(guard);
                            return Err(RasterError::DecoderPanicked(
                                "display-mapping sampling leader panicked".to_string(),
                            ));
                        }
                    }
                }
                Role::Follower(slot) => match slot.wait(req.cancel)? {
                    RoundOutcome::Ready(ready) => return Ok(ready),
                    RoundOutcome::Failed(err) => {
                        if req.cancel.is_cancelled() {
                            return Err(RasterError::Cancelled);
                        }
                        return Err(err);
                    }
                    RoundOutcome::LeaderCancelled => {
                        // Leader was cancelled; loop so an uncancelled
                        // follower can elect itself as the new leader.
                        if req.cancel.is_cancelled() {
                            return Err(RasterError::Cancelled);
                        }
                    }
                },
            }
        }
    }
}

pub(super) struct StretchRequest<'a> {
    pub(super) decoder_pool: &'a DecoderPool,
    pub(super) pixel_layout: PixelLayout,
    pub(super) stats_ifd: usize,
    pub(super) nodata: Option<f32>,
    pub(super) sample_bits: u8,
    pub(super) depth: u32,
    pub(super) budget_tracker: &'a DecodeBudgetTracker,
    pub(super) cancel: &'a CancelToken,
}

fn compute_mapping(req: &StretchRequest<'_>) -> Result<DisplayMapping, RasterError> {
    if req.sample_bits <= 8 {
        return Ok(DisplayMapping::identity(req.nodata));
    }
    let mut pooled = req.decoder_pool.checkout(req.cancel)?;
    pooled.seek_to_image(req.stats_ifd, req.cancel)?;
    let (w, h) = pooled
        .decoder_mut()
        .dimensions()
        .map_err(|e| map_tiff_error(e, req.cancel))?;
    let (cw, ch) = pooled.decoder_mut().chunk_dimensions();
    if w == 0 || h == 0 || cw == 0 || ch == 0 {
        return Ok(DisplayMapping::identity(req.nodata));
    }
    let pixel = req.pixel_layout;

    let total = u64::from(w.div_ceil(cw)) * u64::from(h.div_ceil(ch));
    let mut picks = Vec::new();
    if total > 0 {
        let count = total.min(9);
        if count == 1 {
            picks.push(0);
        } else {
            for i in 0..count {
                let raw_idx = (i * (total - 1)) / (count - 1);
                if let Ok(idx) = u32::try_from(raw_idx) {
                    picks.push(idx);
                }
            }
        }
        picks.dedup();
    }

    let target_per_chunk = (radiometry::STRETCH_SAMPLES / picks.len().max(1)).max(1);
    let mut values = Vec::new();
    for index in picks {
        if req.cancel.is_cancelled() {
            return Err(RasterError::Cancelled);
        }
        let (dw, dh) = pooled.decoder_mut().chunk_data_dimensions(index);
        let est_bytes = grid::chunk_working_bytes(dw, dh, req.depth, false);
        let _lease = req.budget_tracker.reserve_blocking(est_bytes, req.cancel)?;
        let raw = pooled
            .decoder_mut()
            .read_chunk(index)
            .map_err(|e| map_tiff_error(e, req.cancel))?;
        let samples = typed_samples(&raw)?;
        let pixels = samples.len() / pixel.channels.max(1);
        let step = (pixels * pixel.colors().max(1))
            .div_ceil(target_per_chunk)
            .max(1);
        for at in (0..pixels).step_by(step) {
            for channel in 0..pixel.colors() {
                let v = samples.get(at * pixel.channels + channel);
                if radiometry::is_data(v, req.nodata) {
                    values.push(v);
                }
            }
        }
    }
    values.truncate(radiometry::STRETCH_SAMPLES);
    pooled.mark_clean();

    Ok(match percentile_stretch(&mut values) {
        Some((lo, hi)) => DisplayMapping::stretched(lo, hi, req.nodata),
        None => DisplayMapping::identity(req.nodata),
    })
}
