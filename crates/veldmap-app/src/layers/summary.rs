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

/// Read-only summary of an overlay layer rendered in the Layer List UI.
#[derive(Clone, Debug)]
pub(crate) struct LayerSummary {
    pub(crate) id: LayerId,
    pub(crate) name: String,
    pub(crate) path: PathBuf,
    pub(crate) is_remote: bool,
    pub(crate) remote_url: Option<String>,
    pub(crate) visible: bool,
    pub(crate) opacity: f32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) source_pixel_size_m: Option<(f64, f64)>,
    pub(crate) footprint: Option<FootprintBounds>,
    pub(crate) in_flight: usize,
    pub(crate) pass_active: bool,
    pub(crate) reloading: bool,
    pub(crate) crs_label: String,
    pub(crate) datum_warning: Option<String>,
    pub(crate) sample_info: String,
    pub(crate) detail_limit_level: u8,
    pub(crate) can_override_crs: bool,
    pub(crate) has_crs_override: bool,
    pub(crate) last_error: Option<String>,
    pub(crate) can_move_up: bool,
    pub(crate) can_move_down: bool,
    pub(crate) active_spec: TileStreamSpec,
    pub(crate) band_options: Vec<LayerBandOption>,
    pub(crate) scientific: Option<ScientificSummary>,
    pub(crate) categorical: Option<CategoricalSummary>,
    pub(crate) band_math: Option<BandMathSummary>,
}

/// Kind-specific fields assembled cleanly by [`single_raster_summary_fields`]
/// and [`band_math_summary_fields`].
struct LayerKindSummaryFields {
    path: PathBuf,
    is_remote: bool,
    remote_url: Option<String>,
    reloading: bool,
    sample_info: String,
    detail_limit_level: u8,
    can_override_crs: bool,
    has_crs_override: bool,
    band_options: Vec<LayerBandOption>,
    scientific: Option<ScientificSummary>,
    categorical: Option<CategoricalSummary>,
    band_math: Option<BandMathSummary>,
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
        path: asset
            .local_path()
            .map(Path::to_path_buf)
            .unwrap_or_default(),
        is_remote: asset.is_remote(),
        remote_url: asset.remote_url().map(str::to_owned),
        reloading: layer.is_reloading(),
        sample_info,
        detail_limit_level: asset
            .meta()
            .finest_fitting_level()
            .unwrap_or(asset.meta().max_level()),
        can_override_crs: desc.can_override_crs(),
        has_crs_override: asset.geo().crs_override().is_some(),
        band_options: style.band_options().to_vec(),
        scientific,
        categorical,
        band_math: None,
    }
}

fn band_math_summary_fields(
    layer: &OverlayLayer,
    bm: &BandMathLayerData,
) -> LayerKindSummaryFields {
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
        path: PathBuf::new(),
        is_remote: layer.is_remote(),
        remote_url: layer.remote_url().map(str::to_owned),
        reloading: false,
        sample_info,
        detail_limit_level,
        can_override_crs: false,
        has_crs_override: false,
        band_options: Vec::new(),
        scientific: None,
        categorical: None,
        band_math: Some(BandMathSummary {
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
        }
    }

    pub(super) fn to_summary(
        &self,
        in_flight: usize,
        pass_active: bool,
        can_move_up: bool,
        can_move_down: bool,
    ) -> LayerSummary {
        let primary = self.primary_asset();
        let desc = primary.source().description();
        let geo = primary.geo();

        let kind_fields = match self.kind() {
            LayerKind::SingleRaster { asset, style } => {
                single_raster_summary_fields(self, asset, style, desc)
            }
            LayerKind::BandMath(bm) => band_math_summary_fields(self, bm),
        };

        LayerSummary {
            id: self.id(),
            name: self.name().to_owned(),
            path: kind_fields.path,
            is_remote: kind_fields.is_remote,
            remote_url: kind_fields.remote_url,
            visible: self.is_visible(),
            opacity: self.opacity(),
            width: desc.width,
            height: desc.height,
            source_pixel_size_m: geo.source_pixel_size_m(),
            footprint: geo.footprint(),
            in_flight,
            pass_active,
            reloading: kind_fields.reloading,
            crs_label: geo.frame().crs_label(),
            datum_warning: geo.frame().datum_warning().map(|w| w.message()),
            sample_info: kind_fields.sample_info,
            detail_limit_level: kind_fields.detail_limit_level,
            can_override_crs: kind_fields.can_override_crs,
            has_crs_override: kind_fields.has_crs_override,
            last_error: self.last_error().map(str::to_owned),
            can_move_up,
            can_move_down,
            active_spec: primary.meta().spec,
            band_options: kind_fields.band_options,
            scientific: kind_fields.scientific,
            categorical: kind_fields.categorical,
            band_math: kind_fields.band_math,
        }
    }
}
