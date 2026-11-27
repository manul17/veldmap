//! Non-blocking zoom-invariant CPU cursor pixel readout (`CursorProber`)
//! supporting Display RGBA8, Scientific raw+physical `f64` values with units,
//! Categorical discrete class IDs (with ESA Sentinel-2 `SCL` class names), and
//! multi-asset `BandMath` readouts (computed index + constituent physical band
//! values + optional `SCL` class & mask status).

use std::sync::{Arc, Mutex};

use veldmap_core::sync::{CancelToken, Wakeup, lock_unpoisoned};
use veldmap_core::tile::{BandKey, PixelFormat, SourceId};
use veldmap_raster::{BandSample, DecodeJob, DecodeJobKey, RasterSource};

use crate::colormap::{CategoricalPaletteId, sentinel2_scl_class_name};

use super::asset::{RasterAsset, has_affine_scaling};
use super::band_math::{
    BandChannelSample, BandMathFormula, BandMathReadout, MAX_SCL_MASK_CLASSES, is_scl_class_masked,
};
use super::layer::{LayerKind, OverlayLayer};
use super::runtime::OverlayRuntime;
use super::stack::LayerStack;
use super::style::LayerStyle;
use super::{CursorReadout, CursorReadoutValue, LayerId};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProbeKey {
    Single {
        source: SourceId,
        band: BandKey,
        format: PixelFormat,
        is_scl_palette: bool,
        x: u32,
        y: u32,
    },
    BandMath {
        layer_id: LayerId,
        style_rev: u64,
        sources: [Option<SourceId>; 5],
        pixels: [Option<(u32, u32)>; 5],
    },
}

impl ProbeKey {
    fn involves_source(self, source_id: SourceId) -> bool {
        match self {
            Self::Single { source, .. } => source == source_id,
            Self::BandMath { sources, .. } => sources.contains(&Some(source_id)),
        }
    }
}

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

fn asset_pixel_at(asset: &RasterAsset, lat_deg: f64, lon_deg: f64) -> Option<(u32, u32)> {
    let (fx, fy) = asset.geo().frame().fraction_at(lat_deg, lon_deg)?;
    let desc = asset.source().description();
    let px = ((fx * f64::from(desc.width)).floor() as i64)
        .clamp(0, i64::from(desc.width.saturating_sub(1))) as u32;
    let py = ((fy * f64::from(desc.height)).floor() as i64)
        .clamp(0, i64::from(desc.height.saturating_sub(1))) as u32;
    Some((px, py))
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
        let pixel = asset_pixel_at(layer.primary_asset(), lat_deg, lon_deg)?;
        Some((layer, pixel))
    })
}

fn build_readout(
    layer: &OverlayLayer,
    pixel: (u32, u32),
    value: CursorReadoutValue,
) -> CursorReadout {
    match layer.kind() {
        LayerKind::SingleRaster { asset, style, .. } => {
            let spec = asset.meta().spec;
            let band_info = asset.source().description().band_info(spec.band);
            CursorReadout {
                layer_name: layer.name().to_owned(),
                pixel,
                source_pixel_size_m: asset.geo().source_pixel_size_m(),
                units: band_info.and_then(|b| b.units.clone()),
                has_affine_scaling: band_info.is_some_and(has_affine_scaling),
                log10_db: spec.format == PixelFormat::R32Float
                    && style.scientific().preset.is_log10_db(),
                value,
            }
        }
        LayerKind::BandMath(bm) => CursorReadout {
            layer_name: layer.name().to_owned(),
            pixel,
            source_pixel_size_m: bm.primary_asset().geo().source_pixel_size_m(),
            units: bm.output_units(),
            has_affine_scaling: false,
            log10_db: false,
            value,
        },
    }
}

struct SingleProbeJobSpec {
    source: Arc<dyn RasterSource>,
    band: BandKey,
    format: PixelFormat,
    is_scl_palette: bool,
    pixel: (u32, u32),
}

fn sample_single_pixel(
    rt: &OverlayRuntime,
    spec: SingleProbeJobSpec,
    cancel: &CancelToken,
    probe_guard: CursorProbeGuard,
) {
    let ctx = rt.decode_context(cancel);
    match spec
        .source
        .sample_band(spec.band, spec.format, spec.pixel.0, spec.pixel.1, &ctx)
    {
        Ok(BandSample::Rgba(rgba)) => probe_guard.resolve(CursorReadoutValue::Rgba(rgba)),
        Ok(BandSample::Scalar { raw, physical }) => {
            probe_guard.resolve(CursorReadoutValue::Scalar { raw, physical })
        }
        Ok(BandSample::Categorical(class_id)) => {
            probe_guard.resolve(CursorReadoutValue::Categorical {
                class_id,
                class_name: spec
                    .is_scl_palette
                    .then(|| sentinel2_scl_class_name(class_id))
                    .flatten(),
            })
        }
        Ok(BandSample::Nodata) => probe_guard.resolve(CursorReadoutValue::Nodata),
        Ok(BandSample::Unavailable) => probe_guard.resolve(CursorReadoutValue::Unavailable),
        Err(err) if err.is_cancelled() => probe_guard.clear_transient(),
        Err(_) => probe_guard.resolve(CursorReadoutValue::Unavailable),
    }
}

struct BandMathBandProbeSpec {
    label: String,
    source: Arc<dyn RasterSource>,
    band: BandKey,
    format: PixelFormat,
    pixel: Option<(u32, u32)>,
    /// For `SCL` masks whose TIFF/STAC metadata declares a maskable class ID
    /// (`< 32`, e.g. `nodata = 0` = ESA "No Data"), preserve that class on
    /// `BandSample::Nodata` so CPU cursor readouts match GPU `SCL` bitmasking.
    nodata_class_lt_32: Option<u16>,
}

fn make_band_probe_spec(
    asset: &RasterAsset,
    lat_deg: f64,
    lon_deg: f64,
    is_scl: bool,
) -> BandMathBandProbeSpec {
    let spec = asset.meta().spec;
    let nodata_class_lt_32 = is_scl
        .then(|| {
            LayerStyle::categorical_nodata_value(asset.source().description(), spec.band)
                .filter(|&nd| nd < MAX_SCL_MASK_CLASSES)
                .map(|nd| nd as u16)
        })
        .flatten();
    BandMathBandProbeSpec {
        label: asset.label().to_owned(),
        source: Arc::clone(asset.source()),
        band: spec.band,
        format: spec.format,
        pixel: asset_pixel_at(asset, lat_deg, lon_deg),
        nodata_class_lt_32,
    }
}

struct BandMathProbeJobSpec {
    formula: BandMathFormula,
    formula_label: String,
    scl_mask_enabled: bool,
    scl_mask_bits: u32,
    bands: Vec<BandMathBandProbeSpec>,
    scl: Option<BandMathBandProbeSpec>,
}

fn sample_band_math_pixel(
    rt: &OverlayRuntime,
    spec: BandMathProbeJobSpec,
    cancel: &CancelToken,
    probe_guard: CursorProbeGuard,
) {
    let ctx = rt.decode_context(cancel);
    let mut channels: Vec<(String, BandChannelSample)> = Vec::with_capacity(spec.bands.len());
    let mut has_unavailable = false;

    for b in &spec.bands {
        let Some((px, py)) = b.pixel else {
            channels.push((b.label.clone(), BandChannelSample::Nodata));
            continue;
        };
        match b.source.sample_band(b.band, b.format, px, py, &ctx) {
            Ok(BandSample::Scalar { physical, .. }) => {
                channels.push((b.label.clone(), BandChannelSample::Value(physical)));
            }
            Ok(BandSample::Nodata) => {
                channels.push((b.label.clone(), BandChannelSample::Nodata));
            }
            Ok(BandSample::Unavailable) | Ok(_) => {
                has_unavailable = true;
                channels.push((b.label.clone(), BandChannelSample::Unavailable));
            }
            Err(err) if err.is_cancelled() => {
                probe_guard.clear_transient();
                return;
            }
            Err(_) => {
                has_unavailable = true;
                channels.push((b.label.clone(), BandChannelSample::Unavailable));
            }
        }
    }

    let mut scl_class: Option<u16> = None;
    if let Some(scl) = &spec.scl
        && let Some((px, py)) = scl.pixel
    {
        match scl.source.sample_band(scl.band, scl.format, px, py, &ctx) {
            Ok(BandSample::Categorical(c)) => scl_class = Some(c),
            Ok(BandSample::Nodata) => scl_class = scl.nodata_class_lt_32,
            Err(err) if err.is_cancelled() => {
                probe_guard.clear_transient();
                return;
            }
            Ok(BandSample::Unavailable) | Err(_) if spec.scl_mask_enabled => {
                has_unavailable = true;
            }
            _ => {}
        }
    }

    if channels
        .iter()
        .all(|(_, v)| matches!(v, BandChannelSample::Unavailable))
        && scl_class.is_none()
    {
        probe_guard.resolve(CursorReadoutValue::Unavailable);
        return;
    }

    let scl_masked = scl_class
        .is_some_and(|c| is_scl_class_masked(spec.scl_mask_enabled, spec.scl_mask_bits, c));

    let computed = if spec.formula.is_scalar() && channels.len() >= 2 {
        match (channels[0].1.value(), channels[1].1.value()) {
            (Some(a), Some(b)) => spec.formula.evaluate_scalar(a, b),
            _ => None,
        }
    } else {
        None
    };

    probe_guard.resolve(CursorReadoutValue::BandMath(BandMathReadout {
        formula_label: spec.formula_label,
        is_scalar: spec.formula.is_scalar(),
        computed,
        channels,
        scl_class,
        scl_class_name: scl_class.and_then(sentinel2_scl_class_name),
        scl_masked,
        has_unavailable,
    }))
}

#[derive(Clone, Copy)]
struct ProbeTarget {
    pixel: (u32, u32),
    lat_deg: f64,
    lon_deg: f64,
}

fn probe_key_for_layer(layer: &OverlayLayer, target: ProbeTarget) -> ProbeKey {
    match layer.kind() {
        LayerKind::SingleRaster { asset, style, .. } => {
            let spec = asset.meta().spec;
            let is_scl_palette = style.categorical_palette() == CategoricalPaletteId::Sentinel2Scl;
            ProbeKey::Single {
                source: asset.source_id(),
                band: spec.band,
                format: spec.format,
                is_scl_palette,
                x: target.pixel.0,
                y: target.pixel.1,
            }
        }
        LayerKind::BandMath(bm) => {
            let mut sources = [None; 5];
            let mut pixels = [None; 5];
            for (idx, asset) in bm.assets().take(5).enumerate() {
                sources[idx] = Some(asset.source_id());
                pixels[idx] = if idx == 0 {
                    Some(target.pixel)
                } else {
                    asset_pixel_at(asset, target.lat_deg, target.lon_deg)
                };
            }
            ProbeKey::BandMath {
                layer_id: layer.id(),
                style_rev: bm.style.style_rev,
                sources,
                pixels,
            }
        }
    }
}

/// Manages single-flight cancelled-on-move base-resolution CPU pixel probes on
/// `fast_pool` (local rasters) or `remote_pool` (cache-only remote COGs) without
/// ever blocking the UI thread or spinning on queue saturation.
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
                .is_some_and(|k| k.involves_source(source_id))
                || state
                    .resolved_key
                    .is_some_and(|k| k.involves_source(source_id));
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

    /// Clear a cached `Unavailable` or partial `BandMath` probe for `source_id`
    /// so newly uploaded remote COG chunks are picked up on the next hover
    /// frame without discarding complete resolved readouts or cancelling
    /// in-flight probes.
    pub(super) fn invalidate_unavailable_for_source(&self, source_id: SourceId) {
        let mut state = lock_unpoisoned(&self.cache);
        let should_clear = state
            .resolved_key
            .is_some_and(|k| k.involves_source(source_id))
            && match &state.resolved_value {
                CursorReadoutValue::Unavailable => true,
                CursorReadoutValue::BandMath(bm) => bm.has_unavailable,
                _ => false,
            };
        if should_clear {
            state.resolved_key = None;
        }
    }

    fn submit_probe(
        &self,
        layer: &OverlayLayer,
        target: ProbeTarget,
        probe_key: ProbeKey,
        cancel: CancelToken,
        runtime: &Arc<OverlayRuntime>,
    ) -> bool {
        let rt = Arc::clone(runtime);
        let probe_guard = CursorProbeGuard::new(
            Arc::clone(&self.cache),
            probe_key,
            Arc::clone(&runtime.wakeup),
        );
        let (job, is_remote) = match layer.kind() {
            LayerKind::SingleRaster { asset, style, .. } => {
                let spec = asset.meta().spec;
                let source = Arc::clone(asset.source());
                let is_remote = source.is_remote();
                let is_scl_palette =
                    style.categorical_palette() == CategoricalPaletteId::Sentinel2Scl;
                let job_spec = SingleProbeJobSpec {
                    source,
                    band: spec.band,
                    format: spec.format,
                    is_scl_palette,
                    pixel: target.pixel,
                };
                let job = DecodeJob {
                    key: DecodeJobKey::Probe {
                        source: asset.source_id(),
                        band: spec.band,
                        format: spec.format,
                        x: target.pixel.0,
                        y: target.pixel.1,
                    },
                    queued_bytes: 4096,
                    cancel: cancel.clone(),
                    work: Box::new(move || {
                        sample_single_pixel(&rt, job_spec, &cancel, probe_guard);
                    }),
                };
                (job, is_remote)
            }
            LayerKind::BandMath(bm) => {
                let any_remote = bm.assets().any(|a| a.source().is_remote());
                let bands = bm
                    .bands
                    .iter()
                    .map(|a| make_band_probe_spec(a, target.lat_deg, target.lon_deg, false))
                    .collect();
                let scl = bm
                    .scl
                    .as_ref()
                    .map(|a| make_band_probe_spec(a, target.lat_deg, target.lon_deg, true));
                let job_spec = BandMathProbeJobSpec {
                    formula: bm.style.formula,
                    formula_label: bm.style.recipe_label.clone(),
                    scl_mask_enabled: bm.style.scl_mask_enabled,
                    scl_mask_bits: bm.style.scl_mask_bits,
                    bands,
                    scl,
                };
                let primary_spec = bm.primary_asset().meta().spec;
                let job = DecodeJob {
                    key: DecodeJobKey::Probe {
                        source: bm.primary_asset().source_id(),
                        band: primary_spec.band,
                        format: primary_spec.format,
                        x: target.pixel.0,
                        y: target.pixel.1,
                    },
                    queued_bytes: 8192,
                    cancel: cancel.clone(),
                    work: Box::new(move || {
                        sample_band_math_pixel(&rt, job_spec, &cancel, probe_guard);
                    }),
                };
                (job, any_remote)
            }
        };

        let pool = if is_remote {
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

        let target = ProbeTarget {
            pixel,
            lat_deg,
            lon_deg,
        };
        let probe_key = probe_key_for_layer(layer, target);
        let (submit_cancel, prev_cancel) = {
            let mut state = lock_unpoisoned(&self.cache);
            if state.resolved_key == Some(probe_key) {
                return Some(build_readout(layer, pixel, state.resolved_value.clone()));
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
            && !self.submit_probe(layer, target, probe_key, cancel, runtime)
        {
            return Some(build_readout(layer, pixel, CursorReadoutValue::Unavailable));
        }

        Some(build_readout(layer, pixel, CursorReadoutValue::Pending))
    }
}
