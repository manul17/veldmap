//! Lazy singleflight percentile-stretch computation (for >8-bit display
//! mapping) and per-band physical statistics + histograms (`BandStatsCache`),
//! built over the stateless direct raw-chunk decoder.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use veldmap_core::budget::DecodeBudgetTracker;
use veldmap_core::io::ByteSource;
use veldmap_core::sync::{CancelToken, lock_unpoisoned};

use crate::RasterError;
use crate::pipeline::grid;
use crate::pipeline::metadata::{BandStats, SampleType};
use crate::pipeline::radiometry::{
    self, DisplayMapping, compute_band_stats, is_data_f64, percentile_stretch,
    read_native_sample_f64,
};
use crate::pipeline::singleflight::{SingleflightRole, SingleflightSlot, run_singleflight};

use super::ifd::{CachedIfd, Photometric, PlanarConfig, TiffByteOrder};
use super::raw_decode::{RawChunkPayload, decode_raw_chunk};

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
    pub(super) byte_source: &'a Arc<dyn ByteSource>,
    pub(super) byte_order: TiffByteOrder,
    pub(super) stats_ifd: &'a CachedIfd,
    pub(super) nodata_f64: Option<f64>,
    pub(super) max_chunk_bytes: usize,
    pub(super) budget_tracker: &'a DecodeBudgetTracker,
    pub(super) cancel: &'a CancelToken,
}

fn pick_sample_chunks(total: u32) -> Vec<u32> {
    let mut picks = Vec::new();
    if total == 0 {
        return picks;
    }
    let count = total.min(9);
    if count == 1 {
        picks.push(0);
    } else {
        for i in 0..count {
            let raw_idx = (u64::from(i) * u64::from(total - 1)) / u64::from(count - 1);
            if let Ok(idx) = u32::try_from(raw_idx) {
                picks.push(idx);
            }
        }
    }
    picks.dedup();
    picks
}

fn compute_mapping(req: &StretchRequest<'_>) -> Result<DisplayMapping, RasterError> {
    let ifd = req.stats_ifd;
    let invert = ifd.photometric == Photometric::WhiteIsZero;
    if ifd.sample_type == SampleType::U8 {
        return Ok(DisplayMapping::identity(req.nodata_f64).with_invert(invert));
    }
    let picks = pick_sample_chunks(ifd.chunks_per_plane);
    let colors = ifd.pixel_layout.colors().max(1);
    let target_per_chunk = (radiometry::STRETCH_SAMPLES / picks.len().max(1)).max(1);
    let mut values = Vec::new();

    for index in picks {
        if req.cancel.is_cancelled() {
            return Err(RasterError::Cancelled);
        }
        let Some((dw, dh)) = ifd.chunk_data_dims(index) else {
            continue;
        };
        let (stored_w, stored_h) = ifd.chunk_stored_dims(index).unwrap_or((dw, dh));
        let est_bytes = grid::chunk_working_bytes(stored_w, stored_h, ifd.raw_pixel_bytes(), 0);
        let _lease = req.budget_tracker.reserve_blocking(est_bytes, req.cancel)?;

        match ifd.planar {
            PlanarConfig::Chunky => {
                let (_, _, payload) = decode_raw_chunk(
                    req.byte_source.as_ref(),
                    ifd,
                    req.byte_order,
                    0,
                    index,
                    req.max_chunk_bytes,
                    req.cancel,
                )?;
                let RawChunkPayload::Bytes(data) = payload else {
                    continue;
                };
                let spp = usize::from(ifd.samples_per_pixel);
                let pixels = (dw as usize) * (dh as usize);
                let step = (pixels * colors).div_ceil(target_per_chunk).max(1);
                for at in (0..pixels).step_by(step) {
                    for ch in 0..colors {
                        if let Some(v) =
                            read_native_sample_f64(&data, ifd.sample_type, at * spp + ch)
                            && is_data_f64(v, req.nodata_f64)
                        {
                            let f = v as f32;
                            if f.is_finite() {
                                values.push(f);
                            }
                        }
                    }
                }
            }
            PlanarConfig::Planar => {
                let pixels = (dw as usize) * (dh as usize);
                let step = (pixels * colors).div_ceil(target_per_chunk).max(1);
                for ch in 0..colors {
                    let (_, _, payload) = decode_raw_chunk(
                        req.byte_source.as_ref(),
                        ifd,
                        req.byte_order,
                        ch,
                        index,
                        req.max_chunk_bytes,
                        req.cancel,
                    )?;
                    let RawChunkPayload::Bytes(data) = payload else {
                        continue;
                    };
                    for at in (0..pixels).step_by(step) {
                        if let Some(v) = read_native_sample_f64(&data, ifd.sample_type, at)
                            && is_data_f64(v, req.nodata_f64)
                        {
                            let f = v as f32;
                            if f.is_finite() {
                                values.push(f);
                            }
                        }
                    }
                }
            }
        }
    }
    values.truncate(radiometry::STRETCH_SAMPLES);

    Ok(match percentile_stretch(&mut values) {
        Some((lo, hi)) => DisplayMapping::stretched(lo, hi, req.nodata_f64).with_invert(invert),
        None => DisplayMapping::identity(req.nodata_f64).with_invert(invert),
    })
}

type BandStatsSlot = SingleflightSlot<Option<BandStats>>;

enum BandStatsEntry {
    InFlight(Arc<BandStatsSlot>),
    Ready(Option<Box<BandStats>>),
}

/// Per-band physical statistics & histogram cache with singleflight dedup.
pub(super) struct BandStatsCache {
    entries: Mutex<HashMap<u16, BandStatsEntry>>,
}

impl BandStatsCache {
    pub(super) fn new() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
        }
    }

    pub(super) fn get_or_compute(
        &self,
        band: u16,
        scale: f64,
        offset: f64,
        req: &StretchRequest<'_>,
    ) -> Result<Option<BandStats>, RasterError> {
        run_singleflight(
            req.cancel,
            "band stats leader panicked",
            || {
                let mut guard = lock_unpoisoned(&self.entries);
                match guard.get(&band) {
                    Some(BandStatsEntry::Ready(stats)) => {
                        SingleflightRole::Ready(stats.as_deref().cloned())
                    }
                    Some(BandStatsEntry::InFlight(slot)) => {
                        SingleflightRole::Follower(Arc::clone(slot))
                    }
                    None => {
                        let slot = Arc::new(SingleflightSlot::new());
                        guard.insert(band, BandStatsEntry::InFlight(Arc::clone(&slot)));
                        SingleflightRole::Leader(slot)
                    }
                }
            },
            || compute_single_band_stats(band, scale, offset, req),
            |_slot, computed| {
                let mut guard = lock_unpoisoned(&self.entries);
                guard.insert(
                    band,
                    BandStatsEntry::Ready(computed.as_ref().cloned().map(Box::new)),
                );
            },
            |slot| {
                let mut guard = lock_unpoisoned(&self.entries);
                if let Some(BandStatsEntry::InFlight(cur)) = guard.get(&band)
                    && Arc::ptr_eq(cur, slot)
                {
                    guard.remove(&band);
                }
            },
        )
    }
}

fn compute_single_band_stats(
    band: u16,
    scale: f64,
    offset: f64,
    req: &StretchRequest<'_>,
) -> Result<Option<BandStats>, RasterError> {
    let ifd = req.stats_ifd;
    let band_idx = usize::from(band);
    if band_idx >= usize::from(ifd.samples_per_pixel) {
        return Ok(None);
    }
    let picks = pick_sample_chunks(ifd.chunks_per_plane);
    let target_per_chunk = (radiometry::STRETCH_SAMPLES / picks.len().max(1)).max(1);
    let mut values = Vec::new();
    let plane = match ifd.planar {
        PlanarConfig::Planar => band_idx,
        PlanarConfig::Chunky => 0,
    };
    let (stride, ch_off) = match ifd.planar {
        PlanarConfig::Planar => (1usize, 0usize),
        PlanarConfig::Chunky => (usize::from(ifd.samples_per_pixel), band_idx),
    };

    for index in picks {
        if req.cancel.is_cancelled() {
            return Err(RasterError::Cancelled);
        }
        let Some((dw, dh)) = ifd.chunk_data_dims(index) else {
            continue;
        };
        let (stored_w, stored_h) = ifd.chunk_stored_dims(index).unwrap_or((dw, dh));
        let raw_depth = match ifd.planar {
            PlanarConfig::Planar => ifd.sample_type.byte_size() as u32,
            PlanarConfig::Chunky => ifd.raw_pixel_bytes(),
        };
        let est_bytes = grid::chunk_working_bytes(stored_w, stored_h, raw_depth, 0);
        let _lease = req.budget_tracker.reserve_blocking(est_bytes, req.cancel)?;
        let (_, _, payload) = decode_raw_chunk(
            req.byte_source.as_ref(),
            ifd,
            req.byte_order,
            plane,
            index,
            req.max_chunk_bytes,
            req.cancel,
        )?;
        let RawChunkPayload::Bytes(data) = payload else {
            continue;
        };
        let pixels = (dw as usize) * (dh as usize);
        let step = pixels.div_ceil(target_per_chunk).max(1);
        if let Some(params) = ifd.ycbcr
            && ifd.sample_type == SampleType::U8
            && band_idx < 3
        {
            for at in (0..pixels).step_by(step) {
                let off = at * 3;
                if off + 3 <= data.len() {
                    let rgb =
                        params.convert_u8_pixel_to_rgb(data[off], data[off + 1], data[off + 2]);
                    let raw = f64::from(rgb[band_idx]);
                    if is_data_f64(raw, req.nodata_f64) {
                        let phys = raw.mul_add(scale, offset);
                        if phys.is_finite() {
                            values.push(phys);
                        }
                    }
                }
            }
            continue;
        }
        for at in (0..pixels).step_by(step) {
            if let Some(raw) = read_native_sample_f64(&data, ifd.sample_type, at * stride + ch_off)
                && is_data_f64(raw, req.nodata_f64)
            {
                let phys = raw.mul_add(scale, offset);
                if phys.is_finite() {
                    values.push(phys);
                }
            }
        }
    }
    values.truncate(radiometry::STRETCH_SAMPLES);
    Ok(compute_band_stats(&mut values))
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
