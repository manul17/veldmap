//! Lazy singleflight percentile-stretch computation for >8-bit GeoTIFF display
//! mapping, distinguishing leader cancellation (retryable by uncancelled
//! followers) from terminal sampling errors (propagated to current-round
//! followers without serial retry storms).

use std::sync::{Arc, Mutex};

use veldmap_core::budget::DecodeBudgetTracker;
use veldmap_core::sync::{CancelToken, lock_unpoisoned};

use crate::RasterError;
use crate::pipeline::grid;
use crate::pipeline::radiometry::{self, DisplayMapping, percentile_stretch};
use crate::pipeline::singleflight::{SingleflightRole, SingleflightSlot, run_singleflight};

use super::TiffFormat;
use super::chunk_reader::DecoderPool;
use super::tags::{map_tiff_error, typed_samples};

type StretchSlot = SingleflightSlot<DisplayMapping>;

enum CacheState {
    Uncomputed,
    InFlight(Arc<StretchSlot>),
    Ready(DisplayMapping),
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
        self.get_or_compute_with(req.cancel, || compute_mapping(req))
    }

    fn get_or_compute_with(
        &self,
        cancel: &CancelToken,
        compute: impl FnOnce() -> Result<DisplayMapping, RasterError>,
    ) -> Result<DisplayMapping, RasterError> {
        run_singleflight(
            cancel,
            "display-mapping sampling leader panicked",
            || {
                let mut guard = lock_unpoisoned(&self.state);
                match &*guard {
                    CacheState::Ready(ready) => SingleflightRole::Ready(*ready),
                    CacheState::InFlight(slot) => SingleflightRole::Follower(Arc::clone(slot)),
                    CacheState::Uncomputed => {
                        let slot = Arc::new(SingleflightSlot::new());
                        *guard = CacheState::InFlight(Arc::clone(&slot));
                        SingleflightRole::Leader(slot)
                    }
                }
            },
            compute,
            |_slot, &computed| {
                let mut cache = lock_unpoisoned(&self.state);
                *cache = CacheState::Ready(computed);
            },
            |slot| {
                let mut cache = lock_unpoisoned(&self.state);
                if let CacheState::InFlight(cur) = &*cache
                    && Arc::ptr_eq(cur, slot)
                {
                    *cache = CacheState::Uncomputed;
                }
            },
        )
    }
}

pub(super) struct StretchRequest<'a> {
    pub(super) decoder_pool: &'a DecoderPool,
    pub(super) format: TiffFormat,
    pub(super) stats_ifd: usize,
    pub(super) budget_tracker: &'a DecodeBudgetTracker,
    pub(super) cancel: &'a CancelToken,
}

fn compute_mapping(req: &StretchRequest<'_>) -> Result<DisplayMapping, RasterError> {
    if req.format.sample_bits <= 8 {
        return Ok(DisplayMapping::identity(req.format.nodata));
    }
    let mut pooled = req.decoder_pool.checkout(req.cancel)?;
    pooled.seek_to_image(req.stats_ifd, req.cancel)?;
    let total = match pooled.ifd_chunk_layout(req.cancel) {
        Ok(total) => total,
        Err(RasterError::InvalidDimensions { .. }) => {
            return Ok(DisplayMapping::identity(req.format.nodata));
        }
        Err(err) => return Err(err),
    };
    let pixel = req.format.pixel_layout;

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
        if dw == 0 || dh == 0 {
            continue;
        }
        let est_bytes = grid::chunk_working_bytes(dw, dh, req.format.depth, false);
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
                if radiometry::is_data(v, req.format.nodata) {
                    values.push(v);
                }
            }
        }
    }
    values.truncate(radiometry::STRETCH_SAMPLES);
    pooled.mark_clean();

    Ok(match percentile_stretch(&mut values) {
        Some((lo, hi)) => DisplayMapping::stretched(lo, hi, req.format.nodata),
        None => DisplayMapping::identity(req.format.nodata),
    })
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::thread;
    use std::time::Duration;

    use super::*;

    #[test]
    fn uncancelled_leader_returning_cancelled_propagates_once_without_follower_livelock() {
        let cache = Arc::new(StretchCache::new(None));
        let started = Arc::new(AtomicBool::new(false));
        let release = CancelToken::new();

        let leader = {
            let cache = Arc::clone(&cache);
            let started = Arc::clone(&started);
            let release = release.clone();
            thread::spawn(move || {
                cache.get_or_compute_with(&CancelToken::new(), || {
                    started.store(true, Ordering::Release);
                    while !release.is_cancelled() {
                        thread::sleep(Duration::from_millis(2));
                    }
                    Err(RasterError::Cancelled)
                })
            })
        };

        while !started.load(Ordering::Acquire) {
            thread::sleep(Duration::from_millis(2));
        }

        let follower = {
            let cache = Arc::clone(&cache);
            thread::spawn(move || {
                cache.get_or_compute_with(&CancelToken::new(), || {
                    unreachable!("follower must not retry uncancelled leader error")
                })
            })
        };

        thread::sleep(Duration::from_millis(10));
        release.cancel();

        assert!(matches!(
            leader.join().unwrap(),
            Err(RasterError::Cancelled)
        ));
        assert!(matches!(
            follower.join().unwrap(),
            Err(RasterError::Cancelled)
        ));

        // Pre-cancelled callers fail immediately at loop entry even when Ready.
        let ready_cache = StretchCache::new(Some(DisplayMapping::identity(None)));
        let cancelled = CancelToken::new();
        cancelled.cancel();
        assert!(matches!(
            ready_cache.get_or_compute_with(&cancelled, || {
                unreachable!("pre-cancelled caller must not run compute")
            }),
            Err(RasterError::Cancelled)
        ));
    }
}
