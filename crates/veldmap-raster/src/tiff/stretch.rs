//! Lazy singleflight percentile-stretch computation for >8-bit GeoTIFF display
//! mapping, distinguishing leader cancellation (retryable by uncancelled
//! followers) from terminal sampling errors (propagated to current-round
//! followers without serial retry storms).

use std::sync::{Arc, Condvar, Mutex};

use veldmap_core::budget::DecodeBudgetTracker;
use veldmap_core::sync::{CancelToken, lock_unpoisoned};

use crate::RasterError;
use crate::pipeline::grid;
use crate::pipeline::radiometry::{self, DisplayMapping, PixelLayout, percentile_stretch};

use super::chunk_reader::DecoderPool;
use super::tags::{map_tiff_error, typed_samples};

enum RoundState {
    InFlight,
    Ready(DisplayMapping),
    Cancelled,
    Failed(RasterError),
}

struct StretchRound {
    state: Mutex<RoundState>,
    cvar: Condvar,
}

impl StretchRound {
    fn new() -> Self {
        Self {
            state: Mutex::new(RoundState::InFlight),
            cvar: Condvar::new(),
        }
    }

    fn resolve(&self, next: RoundState) {
        let mut state = lock_unpoisoned(&self.state);
        if matches!(*state, RoundState::InFlight) {
            *state = next;
            self.cvar.notify_all();
        }
    }
}

enum CacheState {
    Uncomputed,
    InFlight(Arc<StretchRound>),
    Ready(DisplayMapping),
}

struct RoundGuard<'a> {
    cache: &'a Mutex<CacheState>,
    round: Arc<StretchRound>,
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
                && Arc::ptr_eq(cur, &self.round)
            {
                *guard = CacheState::Uncomputed;
            }
        }
        self.round
            .resolve(RoundState::Failed(RasterError::InvalidFormat(
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
                Leader(Arc<StretchRound>),
                Follower(Arc<StretchRound>),
            }

            let role = {
                let mut guard = lock_unpoisoned(&self.state);
                match &*guard {
                    CacheState::Ready(ready) => Role::Ready(*ready),
                    CacheState::InFlight(round) => Role::Follower(Arc::clone(round)),
                    CacheState::Uncomputed => {
                        if req.cancel.is_cancelled() {
                            return Err(RasterError::Cancelled);
                        }
                        let round = Arc::new(StretchRound::new());
                        *guard = CacheState::InFlight(Arc::clone(&round));
                        Role::Leader(round)
                    }
                }
            };

            match role {
                Role::Ready(ready) => return Ok(ready),
                Role::Leader(round) => {
                    let mut guard = RoundGuard {
                        cache: &self.state,
                        round: Arc::clone(&round),
                        completed: false,
                    };
                    match compute_mapping(req) {
                        Ok(computed) => {
                            {
                                let mut cache = lock_unpoisoned(&self.state);
                                *cache = CacheState::Ready(computed);
                            }
                            round.resolve(RoundState::Ready(computed));
                            guard.completed = true;
                            return Ok(computed);
                        }
                        Err(err) => {
                            {
                                let mut cache = lock_unpoisoned(&self.state);
                                if let CacheState::InFlight(cur) = &*cache
                                    && Arc::ptr_eq(cur, &round)
                                {
                                    *cache = CacheState::Uncomputed;
                                }
                            }
                            if err.is_cancelled() {
                                round.resolve(RoundState::Cancelled);
                            } else {
                                // Current-round followers receive this error
                                // directly instead of triggering serial retries.
                                round.resolve(RoundState::Failed(err.clone()));
                            }
                            guard.completed = true;
                            return Err(err);
                        }
                    }
                }
                Role::Follower(round) => {
                    if req.cancel.is_cancelled() {
                        return Err(RasterError::Cancelled);
                    }
                    let wake_round = Arc::clone(&round);
                    let _cancel_guard = req.cancel.on_cancel_guarded(move || {
                        let _lock = lock_unpoisoned(&wake_round.state);
                        wake_round.cvar.notify_all();
                    });
                    let mut state = lock_unpoisoned(&round.state);
                    loop {
                        match &*state {
                            RoundState::Ready(ready) => return Ok(*ready),
                            RoundState::Failed(err) => {
                                if req.cancel.is_cancelled() {
                                    return Err(RasterError::Cancelled);
                                }
                                return Err(err.clone());
                            }
                            RoundState::Cancelled => {
                                // Leader was cancelled; loop so an uncancelled
                                // follower can elect itself as the new leader.
                                if req.cancel.is_cancelled() {
                                    return Err(RasterError::Cancelled);
                                }
                                break;
                            }
                            RoundState::InFlight => {
                                if req.cancel.is_cancelled() {
                                    return Err(RasterError::Cancelled);
                                }
                                state = match round.cvar.wait(state) {
                                    Ok(g) => g,
                                    Err(poisoned) => poisoned.into_inner(),
                                };
                            }
                        }
                    }
                }
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
    if cw == 0 || ch == 0 {
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
