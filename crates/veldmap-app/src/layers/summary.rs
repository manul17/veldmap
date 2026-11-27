//! Read-only UI and HUD projections (`LayerSummary`, `ColorbarInfo`) derived
//! from [`super::layer::OverlayLayer`].

use std::path::Path;

use veldmap_core::tile::PixelFormat;

use super::band_math::{BandMathFormula, BandMathInputSummary, BandMathSummary};
use super::layer::{LayerKind, OverlayLayer};
use super::style::{ColorbarInfo, LayerStyle, format_band_label};
use super::{CategoricalSummary, LayerSummary, ScientificSummary};

fn band_slot_name(formula: BandMathFormula, idx: usize) -> &'static str {
    match (formula, idx) {
        (BandMathFormula::RgbComposite, 0) => "Red",
        (BandMathFormula::RgbComposite, 1) => "Green",
        (BandMathFormula::RgbComposite, 2) => "Blue",
        (_, 0) => "Band A",
        (_, 1) => "Band B",
        (_, 2) => "Band C",
        _ => "Band D",
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
                let units = (bm.style.formula == BandMathFormula::Difference)
                    .then(|| bm.bands[0].band_units())
                    .flatten();
                Some(ColorbarInfo {
                    title: format!("{} · {}", self.name(), bm.style.recipe_label),
                    units,
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

        let (
            path,
            is_remote,
            remote_url,
            reloading,
            sample_info,
            detail_limit_level,
            can_override_crs,
            has_crs_override,
            band_options,
            scientific,
            categorical,
            band_math,
        ) = match self.kind() {
            LayerKind::SingleRaster { asset, style } => {
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
                    let (scale, offset) =
                        active_band_info.map_or((1.0, 0.0), |b| (b.scale, b.offset));
                    ScientificSummary {
                        preset: sci.preset,
                        min: sci.min,
                        max: sci.max,
                        colormap: sci.colormap,
                        invert_colormap: sci.invert_colormap,
                        units: active_band_info.and_then(|b| b.units.clone()),
                        scale,
                        offset,
                        stats: sci.stats().cloned(),
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
                    nodata_value: LayerStyle::categorical_nodata_value(desc, active_spec.band),
                });
                (
                    asset
                        .local_path()
                        .map(Path::to_path_buf)
                        .unwrap_or_default(),
                    asset.is_remote(),
                    asset.remote_url().map(str::to_owned),
                    self.is_reloading(),
                    sample_info,
                    asset
                        .meta()
                        .finest_fitting_level()
                        .unwrap_or(asset.meta().max_level()),
                    desc.can_override_crs(),
                    geo.crs_override().is_some(),
                    style.band_options().to_vec(),
                    scientific,
                    categorical,
                    None,
                )
            }
            LayerKind::BandMath(bm) => {
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
                (
                    Default::default(),
                    self.is_remote(),
                    self.remote_url().map(str::to_owned),
                    false,
                    sample_info,
                    detail_limit_level,
                    false,
                    false,
                    Vec::new(),
                    None,
                    None,
                    Some(BandMathSummary {
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
                )
            }
        };

        LayerSummary {
            id: self.id(),
            name: self.name().to_owned(),
            path,
            is_remote,
            remote_url,
            visible: self.is_visible(),
            opacity: self.opacity(),
            width: desc.width,
            height: desc.height,
            source_pixel_size_m: geo.source_pixel_size_m(),
            footprint: geo.footprint(),
            in_flight,
            pass_active,
            reloading,
            crs_label: geo.frame().crs_label(),
            datum_warning: geo.frame().datum_warning().map(|w| w.message()),
            sample_info,
            detail_limit_level,
            can_override_crs,
            has_crs_override,
            last_error: self.last_error().map(str::to_owned),
            can_move_up,
            can_move_down,
            active_spec: primary.meta().spec,
            band_options,
            scientific,
            categorical,
            band_math,
        }
    }
}
