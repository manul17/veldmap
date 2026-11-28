//! Read-only UI and HUD projections (`LayerSummary`, `ScientificSummary`,
//! `CategoricalSummary`, `ColorbarInfo`) derived from
//! [`super::layer::OverlayLayer`].

use std::path::{Path, PathBuf};
use std::sync::Arc;

use veldmap_core::geo::FootprintBounds;
use veldmap_core::tile::{PixelFormat, TileStreamSpec};
use veldmap_raster::{BandStats, RasterDescription};

use crate::band_math::BandMathFormula;
use crate::colormap::{CategoricalPaletteId, ColormapId};

use super::LayerId;
use super::asset::{RasterAsset, is_non_identity_affine};
use super::band_math::{BandMathInputSummary, BandMathLayerData, BandMathSummary};
use super::layer::{LayerKind, OverlayLayer};
use super::style::{ColorbarInfo, LayerBandOption, LayerStyle, StretchPreset, format_band_label};
use super::timeline::{
    TimelineCompareMode, TimelineEpochContent, TimelineLayerData, TimelineSummary,
};

/// Read-only scientific track summary projected into [`LayerSummary`].
#[derive(Clone, Debug)]
pub(crate) struct ScientificSummary {
    pub(crate) preset: StretchPreset,
    pub(crate) min: f64,
    pub(crate) max: f64,
    pub(crate) colormap: ColormapId,
    pub(crate) invert_colormap: bool,
    pub(crate) units: Option<String>,
    pub(crate) scale: f64,
    pub(crate) offset: f64,
    pub(crate) stats: Option<Arc<BandStats>>,
    pub(crate) stats_loading: bool,
}

impl ScientificSummary {
    pub(crate) fn has_affine_scaling(&self) -> bool {
        is_non_identity_affine(self.scale, self.offset)
    }
}

/// Read-only categorical track summary projected into [`LayerSummary`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct CategoricalSummary {
    pub(crate) palette: CategoricalPaletteId,
    pub(crate) has_source_color_map: bool,
    pub(crate) nodata_value: Option<u32>,
}

/// Read-only `SingleRaster`-specific summary fields inside [`LayerSummaryKind`].
#[derive(Clone, Debug)]
pub(crate) struct SingleRasterSummary {
    pub(crate) path: PathBuf,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) reloading: bool,
    pub(crate) can_override_crs: bool,
    pub(crate) has_crs_override: bool,
    pub(crate) active_spec: TileStreamSpec,
    pub(crate) band_options: Vec<LayerBandOption>,
    pub(crate) scientific: Option<ScientificSummary>,
    pub(crate) categorical: Option<CategoricalSummary>,
}

/// Tagged kind-specific UI projection embedded in [`LayerSummary`].
#[derive(Clone, Debug)]
pub(crate) enum LayerSummaryKind {
    SingleRaster(SingleRasterSummary),
    BandMath(BandMathSummary),
    Timeline(Box<TimelineSummary>),
}

/// Read-only summary of an overlay layer rendered in the Layer List UI.
#[derive(Clone, Debug)]
pub(crate) struct LayerSummary {
    pub(crate) id: LayerId,
    pub(crate) name: String,
    pub(crate) is_remote: bool,
    pub(crate) remote_url: Option<String>,
    pub(crate) visible: bool,
    pub(crate) opacity: f32,
    pub(crate) source_pixel_size_m: Option<(f64, f64)>,
    pub(crate) footprint: Option<FootprintBounds>,
    pub(crate) in_flight: usize,
    pub(crate) pass_active: bool,
    pub(crate) crs_label: String,
    pub(crate) datum_warning: Option<String>,
    pub(crate) sample_info: String,
    pub(crate) detail_limit_level: u8,
    pub(crate) last_error: Option<String>,
    pub(crate) can_move_up: bool,
    pub(crate) can_move_down: bool,
    pub(crate) kind: LayerSummaryKind,
}

impl LayerSummary {
    pub(crate) fn is_reloading(&self) -> bool {
        match &self.kind {
            LayerSummaryKind::SingleRaster(s) => s.reloading,
            LayerSummaryKind::BandMath(_) | LayerSummaryKind::Timeline(_) => false,
        }
    }

    pub(crate) fn dimensions(&self) -> Option<(u32, u32)> {
        match &self.kind {
            LayerSummaryKind::SingleRaster(s) => Some((s.width, s.height)),
            LayerSummaryKind::BandMath(bm) => Some((bm.width, bm.height)),
            LayerSummaryKind::Timeline(tl) => tl.width.zip(tl.height),
        }
    }
}

/// Shared + kind-specific fields assembled cleanly by
/// [`single_raster_summary_fields`] and [`band_math_summary_fields`].
struct LayerKindSummaryFields {
    is_remote: bool,
    remote_url: Option<String>,
    source_pixel_size_m: Option<(f64, f64)>,
    footprint: Option<FootprintBounds>,
    crs_label: String,
    datum_warning: Option<String>,
    sample_info: String,
    detail_limit_level: u8,
    kind: LayerSummaryKind,
}

fn band_slot_name(formula: BandMathFormula, idx: usize) -> &'static str {
    formula.slot_names().get(idx).copied().unwrap_or(match idx {
        0 => "Band A",
        1 => "Band B",
        2 => "Band C",
        _ => "Band D",
    })
}

fn single_raster_summary_fields(
    layer: &OverlayLayer,
    asset: &RasterAsset,
    style: &LayerStyle,
    desc: &RasterDescription,
) -> LayerKindSummaryFields {
    let geo = asset.geo();
    let active_spec = asset.meta().spec;
    let sample_info = desc.bands.first().map_or_else(
        || "Display RGBA".to_owned(),
        |b| {
            let nodata_str = b
                .nodata
                .map(|v| format!(", nodata={v}"))
                .unwrap_or_default();
            format!("{} ch, {:?}{nodata_str}", b.channels, b.sample_type)
        },
    );
    let active_band_info = desc.band_info(active_spec.band);
    let sci = style.scientific();
    let scientific = (active_spec.format == PixelFormat::R32Float).then(|| {
        let (scale, offset) = active_band_info.map_or((1.0, 0.0), |b| (b.scale, b.offset));
        ScientificSummary {
            preset: sci.preset,
            min: sci.min,
            max: sci.max,
            colormap: sci.colormap,
            invert_colormap: sci.invert_colormap,
            units: active_band_info.and_then(|b| b.units.clone()),
            scale,
            offset,
            stats: sci.stats().map(Arc::clone),
            stats_loading: sci.is_loading(),
        }
    });
    let categorical = matches!(
        active_spec.format,
        PixelFormat::R8Uint | PixelFormat::R16Uint
    )
    .then(|| CategoricalSummary {
        palette: style.categorical_palette(),
        has_source_color_map: style.has_source_color_map(),
        nodata_value: asset.categorical_nodata_value(),
    });
    LayerKindSummaryFields {
        is_remote: asset.is_remote(),
        remote_url: asset.remote_url().map(str::to_owned),
        source_pixel_size_m: geo.source_pixel_size_m(),
        footprint: geo.footprint(),
        crs_label: geo.frame().crs_label(),
        datum_warning: geo.frame().datum_warning().map(|w| w.message()),
        sample_info,
        detail_limit_level: asset
            .meta()
            .finest_fitting_level()
            .unwrap_or(asset.meta().max_level()),
        kind: LayerSummaryKind::SingleRaster(SingleRasterSummary {
            path: asset
                .local_path()
                .map(Path::to_path_buf)
                .unwrap_or_default(),
            width: desc.width,
            height: desc.height,
            reloading: layer.is_reloading(),
            can_override_crs: desc.can_override_crs(),
            has_crs_override: geo.crs_override().is_some(),
            active_spec,
            band_options: style.band_options().to_vec(),
            scientific,
            categorical,
        }),
    }
}

fn band_math_summary_fields(
    layer: &OverlayLayer,
    bm: &BandMathLayerData,
) -> LayerKindSummaryFields {
    let ref_asset = bm.reference_asset();
    let ref_desc = ref_asset.source().description();
    let ref_geo = ref_asset.geo();
    let detail_limit_level = bm
        .assets()
        .map(|a| {
            a.meta()
                .finest_fitting_level()
                .unwrap_or(a.meta().max_level())
        })
        .max()
        .unwrap_or(0);
    let inputs = bm
        .bands
        .iter()
        .enumerate()
        .map(|(idx, a)| BandMathInputSummary {
            slot_name: band_slot_name(bm.style.formula, idx).to_owned(),
            label: a.label().to_owned(),
            source_pixel_size_m: a.geo().source_pixel_size_m(),
        })
        .collect();
    let scl_label = bm.scl.as_ref().map(|a| a.label().to_owned());
    let sample_info = format!(
        "{} ({} bands{})",
        bm.style.recipe_label,
        bm.bands.len(),
        if bm.scl.is_some() { " + SCL" } else { "" }
    );
    LayerKindSummaryFields {
        is_remote: layer.is_remote(),
        remote_url: layer.remote_url().map(str::to_owned),
        source_pixel_size_m: ref_geo.source_pixel_size_m(),
        footprint: ref_geo.footprint(),
        crs_label: ref_geo.frame().crs_label(),
        datum_warning: ref_geo.frame().datum_warning().map(|w| w.message()),
        sample_info,
        detail_limit_level,
        kind: LayerSummaryKind::BandMath(BandMathSummary {
            width: ref_desc.width,
            height: ref_desc.height,
            formula: bm.style.formula,
            recipe_label: bm.style.recipe_label.clone(),
            colormap: bm.style.colormap,
            invert_colormap: bm.style.invert_colormap,
            stretch_min: bm.style.stretch_min,
            stretch_max: bm.style.stretch_max,
            rgb_min: bm.style.rgb_min,
            rgb_max: bm.style.rgb_max,
            rgb_gamma: bm.style.rgb_gamma,
            has_scl: bm.scl.is_some(),
            scl_mask_enabled: bm.style.scl_mask_enabled,
            scl_mask_bits: bm.style.scl_mask_bits,
            inputs,
            scl_label,
        }),
    }
}

fn timeline_summary_fields(layer: &OverlayLayer, tl: &TimelineLayerData) -> LayerKindSummaryFields {
    let ref_asset = tl
        .any_ready_content()
        .map(TimelineEpochContent::reference_asset);
    let crs_label = ref_asset
        .map(|a| a.geo().frame().crs_label())
        .unwrap_or_else(|| "Loading…".to_owned());
    let datum_warning =
        ref_asset.and_then(|a| a.geo().frame().datum_warning().map(|w| w.message()));
    let detail_limit_level = tl
        .active_ready_content()
        .map(|c| {
            let mut max_lvl = 0u8;
            c.for_each_asset(|a| {
                max_lvl = max_lvl.max(
                    a.meta()
                        .finest_fitting_level()
                        .unwrap_or(a.meta().max_level()),
                );
            });
            max_lvl
        })
        .unwrap_or(0);
    let tl_summary = tl.to_summary();
    let sample_info = format!(
        "Timeline · {} epochs{}",
        tl_summary.epochs.len(),
        tl_summary
            .band_math
            .as_ref()
            .map(|bm| format!(" ({})", bm.recipe_label))
            .unwrap_or_default()
    );
    LayerKindSummaryFields {
        is_remote: layer.is_remote(),
        remote_url: layer.remote_url().map(str::to_owned),
        source_pixel_size_m: tl.source_pixel_size_m(),
        footprint: tl.footprint(),
        crs_label,
        datum_warning,
        sample_info,
        detail_limit_level,
        kind: LayerSummaryKind::Timeline(Box::new(tl_summary)),
    }
}

impl OverlayLayer {
    pub(super) fn colorbar_info(&self) -> Option<ColorbarInfo> {
        if !self.is_active() {
            return None;
        }
        match self.kind() {
            LayerKind::SingleRaster { asset, style, .. } => {
                let spec = asset.meta().spec;
                if spec.format != PixelFormat::R32Float {
                    return None;
                }
                let info = asset.source().description().band_info(spec.band);
                let band_title = info.map(format_band_label);
                let title = match band_title {
                    Some(bt) => format!("{} · {bt}", self.name()),
                    None => self.name().to_owned(),
                };
                let sci = style.scientific();
                Some(ColorbarInfo {
                    title,
                    units: info.and_then(|b| b.units.clone()),
                    min: sci.min,
                    max: sci.max,
                    log10_db: sci.preset.is_log10_db(),
                    colormap: sci.colormap,
                    invert_colormap: sci.invert_colormap,
                })
            }
            LayerKind::BandMath(bm) => {
                if !bm.style.formula.is_scalar() {
                    return None;
                }
                Some(ColorbarInfo {
                    title: format!("{} · {}", self.name(), bm.style.recipe_label),
                    units: bm.output_units(),
                    min: bm.style.stretch_min,
                    max: bm.style.stretch_max,
                    log10_db: false,
                    colormap: bm.style.colormap,
                    invert_colormap: bm.style.invert_colormap,
                })
            }
            LayerKind::Timeline(tl) => match tl.compare {
                TimelineCompareMode::ScalarDiff { ref_idx } if tl.is_scalar_diff_ready() => {
                    let a_date = tl
                        .epochs
                        .get(ref_idx)
                        .map_or("A", |e| e.date_label.as_str());
                    let b_date = tl.active_epoch().map_or("B", |e| e.date_label.as_str());
                    let units = tl.active_ready_content().and_then(|c| match c {
                        TimelineEpochContent::BandMath(bm) => bm.output_units(),
                        TimelineEpochContent::Single { asset, .. } => asset
                            .source()
                            .description()
                            .band_info(asset.meta().spec.band)
                            .and_then(|b| b.units.clone()),
                    });
                    Some(ColorbarInfo {
                        title: format!("Δ {} ({b_date} − {a_date})", self.name()),
                        units,
                        min: tl.diff_stretch_min,
                        max: tl.diff_stretch_max,
                        log10_db: false,
                        colormap: tl.diff_colormap,
                        invert_colormap: tl.diff_invert_colormap,
                    })
                }
                TimelineCompareMode::Single | TimelineCompareMode::ScalarDiff { .. } => {
                    let active = tl.active_ready_content()?;
                    let date = tl.active_epoch().map_or("", |e| e.date_label.as_str());
                    match active {
                        TimelineEpochContent::BandMath(bm) if bm.style.formula.is_scalar() => {
                            Some(ColorbarInfo {
                                title: format!("{} [{date}]", self.name()),
                                units: bm.output_units(),
                                min: bm.style.stretch_min,
                                max: bm.style.stretch_max,
                                log10_db: false,
                                colormap: bm.style.colormap,
                                invert_colormap: bm.style.invert_colormap,
                            })
                        }
                        TimelineEpochContent::Single { asset, style }
                            if asset.meta().spec.format == PixelFormat::R32Float =>
                        {
                            let sci = style.scientific();
                            let info = asset
                                .source()
                                .description()
                                .band_info(asset.meta().spec.band);
                            Some(ColorbarInfo {
                                title: format!("{} [{date}]", self.name()),
                                units: info.and_then(|b| b.units.clone()),
                                min: sci.min,
                                max: sci.max,
                                log10_db: sci.preset.is_log10_db(),
                                colormap: sci.colormap,
                                invert_colormap: sci.invert_colormap,
                            })
                        }
                        _ => None,
                    }
                }
                TimelineCompareMode::CrossFade { .. } => None,
            },
        }
    }

    pub(super) fn to_summary(
        &self,
        in_flight: usize,
        pass_active: bool,
        can_move_up: bool,
        can_move_down: bool,
    ) -> LayerSummary {
        let kind_fields = match self.kind() {
            LayerKind::SingleRaster { asset, style } => {
                let desc = asset.source().description();
                single_raster_summary_fields(self, asset, style, desc)
            }
            LayerKind::BandMath(bm) => band_math_summary_fields(self, bm),
            LayerKind::Timeline(tl) => timeline_summary_fields(self, tl),
        };

        LayerSummary {
            id: self.id(),
            name: self.name().to_owned(),
            is_remote: kind_fields.is_remote,
            remote_url: kind_fields.remote_url,
            visible: self.is_visible(),
            opacity: self.opacity(),
            source_pixel_size_m: kind_fields.source_pixel_size_m,
            footprint: kind_fields.footprint,
            in_flight,
            pass_active,
            crs_label: kind_fields.crs_label,
            datum_warning: kind_fields.datum_warning,
            sample_info: kind_fields.sample_info,
            detail_limit_level: kind_fields.detail_limit_level,
            last_error: self.last_error().map(str::to_owned),
            can_move_up,
            can_move_down,
            kind: kind_fields.kind,
        }
    }
}
