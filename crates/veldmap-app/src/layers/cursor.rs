//! Non-blocking zoom-invariant CPU cursor pixel readout (`CursorProber`)
//! supporting Display RGBA8, Scientific raw+physical `f64` values with units,
//! and Categorical discrete class IDs.

use std::sync::{Arc, Mutex};

use veldmap_core::sync::{CancelToken, Wakeup, lock_unpoisoned};
use veldmap_core::tile::{BandKey, PixelFormat, SourceId};
use veldmap_raster::{BandSample, DecodeJob, DecodeJobKey, RasterSource};

use super::runtime::OverlayRuntime;
use super::stack::{LayerStack, OverlayLayer};
use super::{CursorReadout, CursorReadoutValue};

type ProbeKey = (SourceId, BandKey, PixelFormat, u32, u32);

/// RAII guard ensuring an evicted, cancelled, or panicked cursor readout job
/// never leaves `CursorProbeCache` permanently stuck in `pending`.
struct CursorProbeGuard {
    cache: Arc<Mutex<CursorProbeCache>>,
    probe_key: ProbeKey,
    wakeup: Wakeup,
    settled: bool,
}

impl CursorProbeGuard {
    fn new(cache: Arc<Mutex<CursorProbeCache>>, probe_key: ProbeKey, wakeup: Wakeup) -> Self {
        Self {
            cache,
            probe_key,
            wakeup,
            settled: false,
        }
    }

    fn resolve(mut self, value: CursorReadoutValue) {
        self.settled = true;
        let should_wake = {
            let mut guard = lock_unpoisoned(&self.cache);
            if guard.active_key == Some(self.probe_key) {
                guard.resolved_key = Some(self.probe_key);
                guard.resolved_value = value;
                guard.active_key = None;
                guard.active_cancel = None;
                true
            } else {
                false
            }
        };
        if should_wake {
            (self.wakeup)();
        }
    }

    fn clear_transient(mut self) {
        self.settled = true;
        self.cleanup_if_active();
    }

    fn cleanup_if_active(&self) {
        let should_wake = {
            let mut guard = lock_unpoisoned(&self.cache);
            if guard.active_key == Some(self.probe_key) {
                guard.active_key = None;
                guard.active_cancel = None;
                true
            } else {
                false
            }
        };
        if should_wake {
            (self.wakeup)();
        }
    }
}

impl Drop for CursorProbeGuard {
    fn drop(&mut self) {
        if !self.settled {
            self.cleanup_if_active();
        }
    }
}

struct CursorProbeCache {
    active_key: Option<ProbeKey>,
    active_cancel: Option<CancelToken>,
    resolved_key: Option<ProbeKey>,
    resolved_value: CursorReadoutValue,
}

impl Default for CursorProbeCache {
    fn default() -> Self {
        Self {
            active_key: None,
            active_cancel: None,
            resolved_key: None,
            resolved_value: CursorReadoutValue::Unavailable,
        }
    }
}

fn find_topmost_pixel(
    stack: &LayerStack,
    lat_deg: f64,
    lon_deg: f64,
) -> Option<(&OverlayLayer, (u32, u32))> {
    stack.layers().iter().rev().find_map(|layer| {
        if !layer.is_active() {
            return None;
        }
        let (fx, fy) = layer.frame().fraction_at(lat_deg, lon_deg)?;
        let desc = layer.source().description();
        let px = ((fx * f64::from(desc.width)).floor() as i64)
            .clamp(0, i64::from(desc.width.saturating_sub(1))) as u32;
        let py = ((fy * f64::from(desc.height)).floor() as i64)
            .clamp(0, i64::from(desc.height.saturating_sub(1))) as u32;
        Some((layer, (px, py)))
    })
}

fn build_readout(
    layer: &OverlayLayer,
    pixel: (u32, u32),
    value: CursorReadoutValue,
) -> CursorReadout {
    let spec = layer.meta().spec;
    let band_info = layer.source().description().band_info(spec.band);
    CursorReadout {
        layer_name: layer.name().to_owned(),
        pixel,
        source_pixel_size_m: layer.source_pixel_size_m(),
        units: band_info.and_then(|b| b.units.clone()),
        has_affine_scaling: band_info
            .is_some_and(|b| (b.scale - 1.0).abs() > 1e-12 || b.offset.abs() > 1e-12),
        log10_db: spec.format == PixelFormat::R32Float
            && layer.scientific_style().preset.is_log10_db(),
        value,
    }
}

fn sample_cursor_pixel(
    rt: &OverlayRuntime,
    source: &dyn RasterSource,
    band: BandKey,
    format: PixelFormat,
    pixel: (u32, u32),
    cancel: &CancelToken,
    probe_guard: CursorProbeGuard,
) {
    let ctx = rt.decode_context(cancel);
    match source.sample_band(band, format, pixel.0, pixel.1, &ctx) {
        Ok(BandSample::Rgba(rgba)) => probe_guard.resolve(CursorReadoutValue::Rgba(rgba)),
        Ok(BandSample::Scalar { raw, physical }) => {
            probe_guard.resolve(CursorReadoutValue::Scalar { raw, physical })
        }
        Ok(BandSample::Categorical(class_id)) => {
            probe_guard.resolve(CursorReadoutValue::Categorical(class_id))
        }
        Ok(BandSample::Nodata) => probe_guard.resolve(CursorReadoutValue::Nodata),
        Ok(BandSample::Unavailable) => probe_guard.resolve(CursorReadoutValue::Unavailable),
        Err(err) if err.is_cancelled() => probe_guard.clear_transient(),
        Err(_) => probe_guard.resolve(CursorReadoutValue::Unavailable),
    }
}

/// Manages single-flight cancelled-on-move base-resolution CPU pixel probes on
/// `fast_pool` without ever blocking the UI thread or spinning on queue saturation.
pub(super) struct CursorProber {
    cache: Arc<Mutex<CursorProbeCache>>,
}

impl CursorProber {
    pub(super) fn new() -> Self {
        Self {
            cache: Arc::new(Mutex::new(CursorProbeCache::default())),
        }
    }

    pub(super) fn cancel(&self) {
        let prev_cancel = {
            let mut state = lock_unpoisoned(&self.cache);
            state.active_key = None;
            state.active_cancel.take()
        };
        if let Some(cancel) = prev_cancel {
            cancel.cancel();
        }
    }

    pub(super) fn invalidate_source(&self, source_id: SourceId) {
        let prev_cancel = {
            let mut state = lock_unpoisoned(&self.cache);
            let matches = state
                .active_key
                .is_some_and(|(src, _, _, _, _)| src == source_id)
                || state
                    .resolved_key
                    .is_some_and(|(src, _, _, _, _)| src == source_id);
            if matches {
                let tok = state.active_cancel.take();
                *state = CursorProbeCache::default();
                tok
            } else {
                None
            }
        };
        if let Some(tok) = prev_cancel {
            tok.cancel();
        }
    }

    fn submit_probe(
        &self,
        layer: &OverlayLayer,
        pixel: (u32, u32),
        cancel: CancelToken,
        runtime: &Arc<OverlayRuntime>,
    ) -> bool {
        let spec = layer.meta().spec;
        let probe_key: ProbeKey = (layer.source_id(), spec.band, spec.format, pixel.0, pixel.1);
        let source = Arc::clone(layer.source());
        let rt = Arc::clone(runtime);
        let probe_guard = CursorProbeGuard::new(
            Arc::clone(&self.cache),
            probe_key,
            Arc::clone(&runtime.wakeup),
        );
        let job = DecodeJob {
            key: DecodeJobKey::Probe {
                source: layer.source_id(),
                band: spec.band,
                format: spec.format,
                x: pixel.0,
                y: pixel.1,
            },
            queued_bytes: 4096,
            cancel: cancel.clone(),
            work: Box::new(move || {
                sample_cursor_pixel(
                    &rt,
                    source.as_ref(),
                    spec.band,
                    spec.format,
                    pixel,
                    &cancel,
                    probe_guard,
                );
            }),
        };

        let pool = if layer.source().is_remote() {
            &runtime.remote_pool
        } else {
            &runtime.fast_pool
        };
        runtime.try_submit_guarded(pool, job, false, || {
            let mut state = lock_unpoisoned(&self.cache);
            if state.active_key == Some(probe_key) {
                state.active_key = None;
                state.active_cancel = None;
            }
        })
    }

    pub(super) fn query(
        &self,
        stack: &LayerStack,
        runtime: &Arc<OverlayRuntime>,
        lat_deg: f64,
        lon_deg: f64,
    ) -> Option<CursorReadout> {
        let Some((layer, pixel)) = find_topmost_pixel(stack, lat_deg, lon_deg) else {
            self.cancel();
            return None;
        };

        let spec = layer.meta().spec;
        let probe_key: ProbeKey = (layer.source_id(), spec.band, spec.format, pixel.0, pixel.1);
        let (submit_cancel, prev_cancel) = {
            let mut state = lock_unpoisoned(&self.cache);
            if state.resolved_key == Some(probe_key) {
                return Some(build_readout(layer, pixel, state.resolved_value));
            }
            if state.active_key != Some(probe_key) {
                let prev = state.active_cancel.take();
                let cancel = runtime.shutdown.child_token();
                state.active_key = Some(probe_key);
                state.active_cancel = Some(cancel.clone());
                (Some(cancel), prev)
            } else {
                (None, None)
            }
        };
        if let Some(prev) = prev_cancel {
            prev.cancel();
        }

        if let Some(cancel) = submit_cancel
            && !self.submit_probe(layer, pixel, cancel, runtime)
        {
            return Some(build_readout(layer, pixel, CursorReadoutValue::Unavailable));
        }

        Some(build_readout(layer, pixel, CursorReadoutValue::Pending))
    }
}
