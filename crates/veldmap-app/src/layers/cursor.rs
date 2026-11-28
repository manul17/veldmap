//! Non-blocking zoom-invariant CPU cursor pixel readout (`CursorProber`)
//! supporting Display RGBA8, Scientific raw+physical `f64` values with units,
//! Categorical discrete class IDs (with ESA Sentinel-2 `SCL` class names),
//! multi-asset `BandMath` readouts, and dual-epoch `Timeline` scalar diff
//! (`Δ = B - A`) readouts.

use std::sync::{Arc, Mutex};

use veldmap_core::sync::{CancelToken, Wakeup, lock_unpoisoned};
use veldmap_core::tile::{BandKey, PixelFormat, SourceId};
use veldmap_raster::{BandSample, DecodeJob, DecodeJobKey, RasterSource};

use crate::band_math::BandMathFormula;
use crate::colormap::{CategoricalPaletteId, sentinel2_scl_class_name};

use super::LayerId;
use super::asset::{RasterAsset, has_affine_scaling};
use super::band_math::{
    BandChannelSample, BandMathLayerData, BandMathReadout, is_scl_class_masked,
    scl_cpu_maskable_nodata_class,
};
use super::layer::{LayerKind, OverlayLayer};
use super::runtime::OverlayRuntime;
use super::stack::LayerStack;
use super::style::LayerStyle;
use super::timeline::{
    TimelineCompareMode, TimelineEpoch, TimelineEpochContent, TimelineLayerData,
};

/// Per-epoch scalar sample state inside [`TemporalDiffReadout`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum DiffEpochSample {
    Value(f64),
    Masked,
    Nodata,
    Unavailable,
}

/// Zoom-invariant dual-epoch scalar difference readout (`Δ = B - A`) for a
/// `Timeline` layer in `ScalarDiff` comparison mode.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TemporalDiffReadout {
    pub(crate) formula_label: String,
    pub(crate) label_a: String,
    pub(crate) label_b: String,
    pub(crate) sample_a: DiffEpochSample,
    pub(crate) sample_b: DiffEpochSample,
    pub(crate) delta: Option<f64>,
    pub(crate) has_unavailable: bool,
}

/// Classified value returned by a zoom-invariant CPU cursor probe.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum CursorReadoutValue {
    Rgba([u8; 4]),
    Scalar {
        raw: f64,
        physical: f64,
    },
    Categorical {
        class_id: u16,
        class_name: Option<&'static str>,
    },
    BandMath(BandMathReadout),
    TemporalDiff(TemporalDiffReadout),
    Nodata,
    Pending,
    Unavailable,
}

/// Zoom-invariant CPU cursor readout for the topmost visible overlay covering
/// the cursor position.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CursorReadout {
    pub(crate) layer_name: String,
    pub(crate) is_remote: bool,
    pub(crate) pixel: (u32, u32),
    pub(crate) source_pixel_size_m: Option<(f64, f64)>,
    pub(crate) units: Option<String>,
    pub(crate) has_affine_scaling: bool,
    pub(crate) log10_db: bool,
    pub(crate) value: CursorReadoutValue,
}

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
    TemporalDiff {
        layer_id: LayerId,
        style_rev: u64,
        sources_a: [Option<SourceId>; 4],
        sources_b: [Option<SourceId>; 4],
        pixels_a: [Option<(u32, u32)>; 4],
        pixels_b: [Option<(u32, u32)>; 4],
    },
}

impl ProbeKey {
    fn involves_source(self, source_id: SourceId) -> bool {
        match self {
            Self::Single { source, .. } => source == source_id,
            Self::BandMath { sources, .. } => sources.contains(&Some(source_id)),
            Self::TemporalDiff {
                sources_a,
                sources_b,
                ..
            } => sources_a.contains(&Some(source_id)) || sources_b.contains(&Some(source_id)),
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

fn content_is_remote(content: &TimelineEpochContent) -> bool {
    let mut remote = false;
    content.for_each_asset(|a| remote |= a.is_remote());
    remote
}

fn timeline_probe_epoch(tl: &TimelineLayerData) -> Option<(&TimelineEpoch, &TimelineEpochContent)> {
    let pick = |idx: usize| {
        tl.epochs
            .get(idx)
            .and_then(|e| e.ready_content().map(|c| (e, c)))
    };
    match tl.compare {
        TimelineCompareMode::CrossFade {
            ref_idx, blend_t, ..
        } if blend_t < 0.5 && ref_idx != tl.active_idx => {
            pick(ref_idx).or_else(|| pick(tl.active_idx))
        }
        _ => pick(tl.active_idx),
    }
}

fn timeline_probe_content(tl: &TimelineLayerData) -> Option<&TimelineEpochContent> {
    timeline_probe_epoch(tl).map(|(_, c)| c)
}

fn layer_pixel_at(layer: &OverlayLayer, lat_deg: f64, lon_deg: f64) -> Option<(u32, u32)> {
    match layer.kind() {
        LayerKind::SingleRaster { asset, .. } => asset_pixel_at(asset, lat_deg, lon_deg),
        LayerKind::BandMath(bm) => asset_pixel_at(bm.reference_asset(), lat_deg, lon_deg),
        LayerKind::Timeline(tl) => {
            let content = timeline_probe_content(tl)?;
            asset_pixel_at(content.reference_asset(), lat_deg, lon_deg)
        }
    }
}

fn find_topmost_pixel(
    stack: &LayerStack,
    lat_deg: f64,
    lon_deg: f64,
    excluded_layer: Option<LayerId>,
) -> Option<(&OverlayLayer, (u32, u32))> {
    stack.layers().iter().rev().find_map(|layer| {
        if !layer.is_active() || excluded_layer == Some(layer.id()) {
            return None;
        }
        let pixel = layer_pixel_at(layer, lat_deg, lon_deg)?;
        Some((layer, pixel))
    })
}

fn build_single_readout(
    layer_name: String,
    is_remote: bool,
    asset: &RasterAsset,
    style: &LayerStyle,
    pixel: (u32, u32),
    value: CursorReadoutValue,
) -> CursorReadout {
    let spec = asset.meta().spec;
    let band_info = asset.source().description().band_info(spec.band);
    CursorReadout {
        layer_name,
        is_remote,
        pixel,
        source_pixel_size_m: asset.geo().source_pixel_size_m(),
        units: band_info.and_then(|b| b.units.clone()),
        has_affine_scaling: band_info.is_some_and(has_affine_scaling),
        log10_db: spec.format == PixelFormat::R32Float && style.scientific().preset.is_log10_db(),
        value,
    }
}

fn build_readout(
    layer: &OverlayLayer,
    pixel: (u32, u32),
    value: CursorReadoutValue,
) -> CursorReadout {
    let is_remote = layer.is_remote();
    match layer.kind() {
        LayerKind::SingleRaster { asset, style, .. } => build_single_readout(
            layer.name().to_owned(),
            is_remote,
            asset,
            style,
            pixel,
            value,
        ),
        LayerKind::BandMath(bm) => CursorReadout {
            layer_name: layer.name().to_owned(),
            is_remote,
            pixel,
            source_pixel_size_m: bm.source_pixel_size_m(),
            units: bm.output_units(),
            has_affine_scaling: false,
            log10_db: false,
            value,
        },
        LayerKind::Timeline(tl) => {
            if tl.is_scalar_diff_ready() {
                let date_tag = tl
                    .active_epoch()
                    .map(|e| format!(" [{}]", e.date_label))
                    .unwrap_or_default();
                let display_name = format!("{}{date_tag}", layer.name());
                let diff_is_remote = tl.ref_ready_content().is_some_and(content_is_remote)
                    || tl.active_ready_content().is_some_and(content_is_remote);
                let units = tl.active_ready_content().and_then(|c| match c {
                    TimelineEpochContent::BandMath(bm) => bm.output_units(),
                    TimelineEpochContent::Single { asset, .. } => asset
                        .source()
                        .description()
                        .band_info(asset.meta().spec.band)
                        .and_then(|b| b.units.clone()),
                });
                return CursorReadout {
                    layer_name: display_name,
                    is_remote: diff_is_remote,
                    pixel,
                    source_pixel_size_m: tl.source_pixel_size_m(),
                    units,
                    has_affine_scaling: false,
                    log10_db: false,
                    value,
                };
            }
            match timeline_probe_epoch(tl) {
                Some((epoch, content)) => {
                    let display_name = format!("{} [{}]", layer.name(), epoch.date_label);
                    let epoch_is_remote = content_is_remote(content);
                    match content {
                        TimelineEpochContent::Single { asset, style } => build_single_readout(
                            display_name,
                            epoch_is_remote,
                            asset,
                            style,
                            pixel,
                            value,
                        ),
                        TimelineEpochContent::BandMath(bm) => CursorReadout {
                            layer_name: display_name,
                            is_remote: epoch_is_remote,
                            pixel,
                            source_pixel_size_m: bm.source_pixel_size_m(),
                            units: bm.output_units(),
                            has_affine_scaling: false,
                            log10_db: false,
                            value,
                        },
                    }
                }
                None => CursorReadout {
                    layer_name: layer.name().to_owned(),
                    is_remote,
                    pixel,
                    source_pixel_size_m: tl.source_pixel_size_m(),
                    units: None,
                    has_affine_scaling: false,
                    log10_db: false,
                    value,
                },
            }
        }
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
        .then(|| scl_cpu_maskable_nodata_class(asset))
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

struct DiffEpochProbeSpec {
    formula: Option<BandMathFormula>,
    scl_mask_enabled: bool,
    scl_mask_bits: u32,
    bands: Vec<BandMathBandProbeSpec>,
    scl: Option<BandMathBandProbeSpec>,
}

struct TemporalDiffProbeJobSpec {
    formula_label: String,
    label_a: String,
    label_b: String,
    epoch_a: DiffEpochProbeSpec,
    epoch_b: DiffEpochProbeSpec,
}

enum EpochScalarEval {
    Sample(DiffEpochSample),
    Cancelled,
}

fn eval_diff_epoch_scalar(
    rt: &OverlayRuntime,
    spec: &DiffEpochProbeSpec,
    cancel: &CancelToken,
) -> EpochScalarEval {
    let ctx = rt.decode_context(cancel);
    let mut vals: Vec<Option<f64>> = Vec::with_capacity(spec.bands.len());
    let mut has_unavailable = false;
    for b in &spec.bands {
        let Some((px, py)) = b.pixel else {
            vals.push(None);
            continue;
        };
        match b.source.sample_band(b.band, b.format, px, py, &ctx) {
            Ok(BandSample::Scalar { physical, .. }) => vals.push(Some(physical)),
            Ok(BandSample::Nodata) => vals.push(None),
            Err(err) if err.is_cancelled() => return EpochScalarEval::Cancelled,
            _ => {
                has_unavailable = true;
                vals.push(None);
            }
        }
    }
    let mut masked = false;
    let mut mask_unavailable = false;
    if spec.scl_mask_enabled
        && let Some(scl) = &spec.scl
        && let Some((px, py)) = scl.pixel
    {
        match scl.source.sample_band(scl.band, scl.format, px, py, &ctx) {
            Ok(BandSample::Categorical(c)) => {
                masked = is_scl_class_masked(true, spec.scl_mask_bits, c);
            }
            Ok(BandSample::Nodata) => {
                masked = scl
                    .nodata_class_lt_32
                    .is_some_and(|c| is_scl_class_masked(true, spec.scl_mask_bits, c));
            }
            Err(err) if err.is_cancelled() => return EpochScalarEval::Cancelled,
            Ok(BandSample::Unavailable) | Err(_) => mask_unavailable = true,
            _ => {}
        }
    }
    if masked {
        return EpochScalarEval::Sample(DiffEpochSample::Masked);
    }
    if has_unavailable || mask_unavailable {
        return EpochScalarEval::Sample(DiffEpochSample::Unavailable);
    }
    let value = match spec.formula {
        None => vals.first().copied().flatten(),
        Some(formula) if vals.len() >= 2 => match (vals[0], vals[1]) {
            (Some(v0), Some(v1)) => formula.evaluate_scalar(v0, v1),
            _ => None,
        },
        Some(_) => None,
    };
    EpochScalarEval::Sample(match value {
        Some(v) => DiffEpochSample::Value(v),
        None => DiffEpochSample::Nodata,
    })
}

fn sample_temporal_diff_pixel(
    rt: &OverlayRuntime,
    spec: TemporalDiffProbeJobSpec,
    cancel: &CancelToken,
    probe_guard: CursorProbeGuard,
) {
    let EpochScalarEval::Sample(sample_a) = eval_diff_epoch_scalar(rt, &spec.epoch_a, cancel)
    else {
        probe_guard.clear_transient();
        return;
    };
    let EpochScalarEval::Sample(sample_b) = eval_diff_epoch_scalar(rt, &spec.epoch_b, cancel)
    else {
        probe_guard.clear_transient();
        return;
    };
    let has_unavailable = matches!(sample_a, DiffEpochSample::Unavailable)
        || matches!(sample_b, DiffEpochSample::Unavailable);
    if matches!(sample_a, DiffEpochSample::Unavailable)
        && matches!(sample_b, DiffEpochSample::Unavailable)
    {
        probe_guard.resolve(CursorReadoutValue::Unavailable);
        return;
    }
    let delta = match (sample_a, sample_b) {
        (DiffEpochSample::Value(a), DiffEpochSample::Value(b)) => {
            let d = b - a;
            d.is_finite().then_some(d)
        }
        _ => None,
    };
    probe_guard.resolve(CursorReadoutValue::TemporalDiff(TemporalDiffReadout {
        formula_label: spec.formula_label,
        label_a: spec.label_a,
        label_b: spec.label_b,
        sample_a,
        sample_b,
        delta,
        has_unavailable,
    }));
}

fn probe_key_for_single(asset: &RasterAsset, style: &LayerStyle, pixel: (u32, u32)) -> ProbeKey {
    let spec = asset.meta().spec;
    let is_scl_palette = style.categorical_palette() == CategoricalPaletteId::Sentinel2Scl;
    ProbeKey::Single {
        source: asset.source_id(),
        band: spec.band,
        format: spec.format,
        is_scl_palette,
        x: pixel.0,
        y: pixel.1,
    }
}

fn probe_key_for_band_math(
    layer_id: LayerId,
    bm: &BandMathLayerData,
    target: ProbeTarget,
) -> ProbeKey {
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
        layer_id,
        style_rev: bm.style.style_rev,
        sources,
        pixels,
    }
}

type EpochKeySlots = ([Option<SourceId>; 4], [Option<(u32, u32)>; 4]);

fn fill_epoch_key_slots(content: &TimelineEpochContent, target: ProbeTarget) -> EpochKeySlots {
    let mut sources = [None; 4];
    let mut pixels = [None; 4];
    let mut idx = 0usize;
    content.for_each_asset(|asset| {
        if idx < 4 {
            sources[idx] = Some(asset.source_id());
            pixels[idx] = asset_pixel_at(asset, target.lat_deg, target.lon_deg);
            idx += 1;
        }
    });
    (sources, pixels)
}

fn probe_key_for_layer(layer: &OverlayLayer, target: ProbeTarget) -> ProbeKey {
    match layer.kind() {
        LayerKind::SingleRaster { asset, style, .. } => {
            probe_key_for_single(asset, style, target.pixel)
        }
        LayerKind::BandMath(bm) => probe_key_for_band_math(layer.id(), bm, target),
        LayerKind::Timeline(tl) => {
            if tl.is_scalar_diff_ready()
                && let (Some(a), Some(b)) = (tl.ref_ready_content(), tl.active_ready_content())
            {
                let (sources_a, pixels_a) = fill_epoch_key_slots(a, target);
                let (sources_b, pixels_b) = fill_epoch_key_slots(b, target);
                let style_rev = tl.band_math_style.as_ref().map_or(0, |s| s.style_rev);
                return ProbeKey::TemporalDiff {
                    layer_id: layer.id(),
                    style_rev,
                    sources_a,
                    sources_b,
                    pixels_a,
                    pixels_b,
                };
            }
            match timeline_probe_content(tl) {
                Some(TimelineEpochContent::Single { asset, style }) => {
                    probe_key_for_single(asset, style, target.pixel)
                }
                Some(TimelineEpochContent::BandMath(bm)) => {
                    probe_key_for_band_math(layer.id(), bm, target)
                }
                None => ProbeKey::BandMath {
                    layer_id: layer.id(),
                    style_rev: 0,
                    sources: [None; 5],
                    pixels: [None; 5],
                },
            }
        }
    }
}

fn build_single_probe_job(
    asset: &RasterAsset,
    style: &LayerStyle,
    pixel: (u32, u32),
    cancel: CancelToken,
    rt: Arc<OverlayRuntime>,
    probe_guard: CursorProbeGuard,
) -> (DecodeJob, bool) {
    let spec = asset.meta().spec;
    let source = Arc::clone(asset.source());
    let is_remote = source.is_remote();
    let is_scl_palette = style.categorical_palette() == CategoricalPaletteId::Sentinel2Scl;
    let job_spec = SingleProbeJobSpec {
        source,
        band: spec.band,
        format: spec.format,
        is_scl_palette,
        pixel,
    };
    let job = DecodeJob {
        key: DecodeJobKey::Probe {
            source: asset.source_id(),
            band: spec.band,
            format: spec.format,
            x: pixel.0,
            y: pixel.1,
        },
        queued_bytes: 4096,
        cancel: cancel.clone(),
        work: Box::new(move || {
            sample_single_pixel(&rt, job_spec, &cancel, probe_guard);
        }),
    };
    (job, is_remote)
}

fn build_band_math_probe_job(
    bm: &BandMathLayerData,
    target: ProbeTarget,
    cancel: CancelToken,
    rt: Arc<OverlayRuntime>,
    probe_guard: CursorProbeGuard,
) -> (DecodeJob, bool) {
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
    let ref_asset = bm.reference_asset();
    let ref_spec = ref_asset.meta().spec;
    let job = DecodeJob {
        key: DecodeJobKey::Probe {
            source: ref_asset.source_id(),
            band: ref_spec.band,
            format: ref_spec.format,
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
                CursorReadoutValue::TemporalDiff(td) => td.has_unavailable,
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
                build_single_probe_job(asset, style, target.pixel, cancel, rt, probe_guard)
            }
            LayerKind::BandMath(bm) => {
                build_band_math_probe_job(bm, target, cancel, rt, probe_guard)
            }
            LayerKind::Timeline(tl) => {
                if tl.is_scalar_diff_ready()
                    && let TimelineCompareMode::ScalarDiff { ref_idx } = tl.compare
                    && let (Some(content_a), Some(content_b)) =
                        (tl.ref_ready_content(), tl.active_ready_content())
                {
                    let label_a = tl
                        .epochs
                        .get(ref_idx)
                        .map_or_else(|| "A".to_owned(), |e| e.date_label.clone());
                    let label_b = tl
                        .active_epoch()
                        .map_or_else(|| "B".to_owned(), |e| e.date_label.clone());
                    let make_epoch_spec = |content: &TimelineEpochContent| match content {
                        TimelineEpochContent::Single { asset, .. } => DiffEpochProbeSpec {
                            formula: None,
                            scl_mask_enabled: false,
                            scl_mask_bits: 0,
                            bands: vec![make_band_probe_spec(
                                asset,
                                target.lat_deg,
                                target.lon_deg,
                                false,
                            )],
                            scl: None,
                        },
                        TimelineEpochContent::BandMath(bm) => DiffEpochProbeSpec {
                            formula: Some(bm.style.formula),
                            scl_mask_enabled: bm.style.scl_mask_enabled,
                            scl_mask_bits: bm.style.scl_mask_bits,
                            bands: bm
                                .bands
                                .iter()
                                .map(|a| {
                                    make_band_probe_spec(a, target.lat_deg, target.lon_deg, false)
                                })
                                .collect(),
                            scl: bm.scl.as_ref().map(|a| {
                                make_band_probe_spec(a, target.lat_deg, target.lon_deg, true)
                            }),
                        },
                    };
                    let formula_label = match content_b {
                        TimelineEpochContent::BandMath(bm) => format!("Δ{}", bm.style.recipe_label),
                        TimelineEpochContent::Single { .. } => "Δ".to_owned(),
                    };
                    let ref_asset = content_b.reference_asset();
                    let ref_spec = ref_asset.meta().spec;
                    let is_remote = content_is_remote(content_a) || content_is_remote(content_b);
                    let job_spec = TemporalDiffProbeJobSpec {
                        formula_label,
                        label_a,
                        label_b,
                        epoch_a: make_epoch_spec(content_a),
                        epoch_b: make_epoch_spec(content_b),
                    };
                    let job = DecodeJob {
                        key: DecodeJobKey::Probe {
                            source: ref_asset.source_id(),
                            band: ref_spec.band,
                            format: ref_spec.format,
                            x: target.pixel.0,
                            y: target.pixel.1,
                        },
                        queued_bytes: 12_288,
                        cancel: cancel.clone(),
                        work: Box::new(move || {
                            sample_temporal_diff_pixel(&rt, job_spec, &cancel, probe_guard);
                        }),
                    };
                    (job, is_remote)
                } else {
                    match timeline_probe_content(tl) {
                        Some(TimelineEpochContent::Single { asset, style }) => {
                            build_single_probe_job(
                                asset,
                                style,
                                target.pixel,
                                cancel,
                                rt,
                                probe_guard,
                            )
                        }
                        Some(TimelineEpochContent::BandMath(bm)) => {
                            build_band_math_probe_job(bm, target, cancel, rt, probe_guard)
                        }
                        None => return false,
                    }
                }
            }
        };

        let pool = runtime.pool_for_source(is_remote);
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
        excluded_layer: Option<LayerId>,
    ) -> Option<CursorReadout> {
        let Some((layer, pixel)) = find_topmost_pixel(stack, lat_deg, lon_deg, excluded_layer)
        else {
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
