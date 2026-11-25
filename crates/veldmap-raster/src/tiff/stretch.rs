//! Lazy singleflight percentile-stretch computation (for >8-bit display
//! mapping) and per-band physical statistics + histograms (`BandStatsCache`),
//! built over the stateless direct raw-chunk decoder.

use veldmap_core::budget::DecodeBudgetTracker;
use veldmap_core::sync::CancelToken;

use crate::RasterError;
use crate::pipeline::grid;
use crate::pipeline::metadata::{BandStats, SampleType};
use crate::pipeline::radiometry::{
    self, DisplayMapping, compute_band_stats, is_data_f64, percentile_stretch,
    read_native_sample_f64,
};
use crate::pipeline::singleflight::SingleflightMap;

use super::chunk_convert::for_each_band_sample_f64;
use super::ifd::{CachedIfd, Photometric, PlanarConfig, RawTiffAccess};
use super::raw_decode::{RawChunkPayload, decode_raw_chunk};

pub(super) struct StretchCache {
    map: SingleflightMap<(), DisplayMapping>,
}

impl StretchCache {
    pub(super) fn new(initial: Option<DisplayMapping>) -> Self {
        let map = match initial {
            Some(mapping) => SingleflightMap::with_ready((), mapping),
            None => SingleflightMap::new(),
        };
        Self { map }
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
        self.map.get_or_compute(
            (),
            cancel,
            "display-mapping sampling leader panicked",
            compute,
        )
    }
}

/// Per-band physical statistics & histogram cache with singleflight dedup.
pub(super) struct BandStatsCache {
    map: SingleflightMap<u16, Option<BandStats>>,
}

impl BandStatsCache {
    pub(super) fn new() -> Self {
        Self {
            map: SingleflightMap::new(),
        }
    }

    pub(super) fn get_or_compute(
        &self,
        band: u16,
        scale: f64,
        offset: f64,
        req: &StretchRequest<'_>,
    ) -> Result<Option<BandStats>, RasterError> {
        self.map
            .get_or_compute(band, req.cancel, "band stats leader panicked", || {
                compute_single_band_stats(band, scale, offset, req)
            })
    }
}

pub(super) struct StretchRequest<'a> {
    pub(super) access: &'a RawTiffAccess,
    pub(super) stats_ifd: &'a CachedIfd,
    pub(super) nodata_f64: Option<f64>,
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

/// Iterate up to 9 evenly spaced chunks in `req.stats_ifd` under cancellation
/// checks and per-chunk working-memory budget leases.
fn for_each_sampled_chunk(
    req: &StretchRequest<'_>,
    raw_depth: u32,
    mut visit: impl FnMut(u32, usize, usize) -> Result<(), RasterError>,
) -> Result<(), RasterError> {
    let ifd = req.stats_ifd;
    let picks = pick_sample_chunks(ifd.chunks_per_plane);
    let target_per_chunk = (radiometry::STRETCH_SAMPLES / picks.len().max(1)).max(1);
    for index in picks {
        if req.cancel.is_cancelled() {
            return Err(RasterError::Cancelled);
        }
        let Some(dims) = ifd.chunk_dims(index) else {
            continue;
        };
        let est_bytes = grid::chunk_working_bytes(dims.stored_w, dims.stored_h, raw_depth, 0);
        let _lease = req.budget_tracker.reserve_blocking(est_bytes, req.cancel)?;
        let pixels = (dims.data_w as usize) * (dims.data_h as usize);
        visit(index, pixels, target_per_chunk)?;
    }
    Ok(())
}

fn compute_mapping(req: &StretchRequest<'_>) -> Result<DisplayMapping, RasterError> {
    let ifd = req.stats_ifd;
    let invert = ifd.photometric == Photometric::WhiteIsZero;
    if ifd.sample_type == SampleType::U8 {
        return Ok(DisplayMapping::identity(req.nodata_f64).with_invert(invert));
    }
    let colors = ifd.pixel_layout.colors().max(1);
    let mut values = Vec::new();

    for_each_sampled_chunk(
        req,
        ifd.raw_pixel_bytes(),
        |index, pixels, target_per_chunk| {
            let step = (pixels * colors).div_ceil(target_per_chunk).max(1);
            match ifd.planar {
                PlanarConfig::Chunky => {
                    let (_, _, payload) = decode_raw_chunk(req.access, ifd, 0, index, req.cancel)?;
                    let RawChunkPayload::Bytes(data) = payload else {
                        return Ok(());
                    };
                    let spp = usize::from(ifd.samples_per_pixel);
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
                    for ch in 0..colors {
                        let (_, _, payload) =
                            decode_raw_chunk(req.access, ifd, ch, index, req.cancel)?;
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
            Ok(())
        },
    )?;
    values.truncate(radiometry::STRETCH_SAMPLES);

    Ok(match percentile_stretch(&mut values) {
        Some((lo, hi)) => DisplayMapping::stretched(lo, hi, req.nodata_f64).with_invert(invert),
        None => DisplayMapping::identity(req.nodata_f64).with_invert(invert),
    })
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
    let plane = ifd.band_plane(band_idx);
    let mut values = Vec::new();

    for_each_sampled_chunk(
        req,
        ifd.band_raw_pixel_bytes(),
        |index, pixels, target_per_chunk| {
            let (_, _, payload) = decode_raw_chunk(req.access, ifd, plane, index, req.cancel)?;
            let RawChunkPayload::Bytes(data) = payload else {
                return Ok(());
            };
            let step = pixels.div_ceil(target_per_chunk).max(1);
            for_each_band_sample_f64(
                &data,
                ifd,
                band_idx,
                pixels,
                step,
                req.nodata_f64,
                |sample| {
                    if let Some(raw) = sample {
                        let phys = raw.mul_add(scale, offset);
                        if phys.is_finite() {
                            values.push(phys);
                        }
                    }
                },
            );
            Ok(())
        },
    )?;
    values.truncate(radiometry::STRETCH_SAMPLES);
    Ok(compute_band_stats(&mut values))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
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
