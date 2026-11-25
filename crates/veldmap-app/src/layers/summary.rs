//! Read-only UI and HUD projections (`LayerSummary`, `ColorbarInfo`) derived
//! from [`super::layer::OverlayLayer`].

use veldmap_core::tile::PixelFormat;

use super::layer::OverlayLayer;
use super::style::{ColorbarInfo, format_band_label};
use super::{CategoricalSummary, LayerSummary, ScientificSummary};

impl OverlayLayer {
    pub(super) fn colorbar_info(&self) -> Option<ColorbarInfo> {
        let spec = self.meta().spec;
        if !self.is_active() || spec.format != PixelFormat::R32Float {
            return None;
        }
        let info = self.source().description().band_info(spec.band);
        let band_title = info.map(format_band_label);
        let title = match band_title {
            Some(bt) => format!("{} · {bt}", self.name()),
            None => self.name().to_owned(),
        };
        let sci = self.scientific_style();
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

    pub(super) fn to_summary(
        &self,
        in_flight: usize,
        pass_active: bool,
        can_move_up: bool,
        can_move_down: bool,
    ) -> LayerSummary {
        let desc = self.source().description();
        let active_spec = self.meta().spec;
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
        let sci = self.scientific_style();
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
                stats: sci.stats().cloned(),
                stats_loading: sci.is_loading(),
            }
        });
        let categorical = matches!(
            active_spec.format,
            PixelFormat::R8Uint | PixelFormat::R16Uint
        )
        .then(|| CategoricalSummary {
            palette: self.categorical_palette(),
            has_source_color_map: self.has_source_color_map(),
            nodata_value: self.categorical_nodata_value(active_spec.band),
        });

        LayerSummary {
            id: self.id(),
            name: self.name().to_owned(),
            path: self.path().to_path_buf(),
            visible: self.is_visible(),
            opacity: self.opacity(),
            width: desc.width,
            height: desc.height,
            source_pixel_size_m: self.source_pixel_size_m(),
            footprint: self.footprint(),
            in_flight,
            pass_active,
            reloading: self.is_reloading(),
            crs_label: self.frame().crs_label(),
            datum_warning: self.frame().datum_warning().map(|w| w.message()),
            sample_info,
            detail_limit_level: self
                .meta()
                .finest_fitting_level()
                .unwrap_or(self.meta().max_level()),
            can_override_crs: desc.can_override_crs(),
            has_crs_override: self.crs_override().is_some(),
            last_error: self.last_error().map(str::to_owned),
            can_move_up,
            can_move_down,
            active_spec,
            band_options: self.band_options().to_vec(),
            scientific,
            categorical,
        }
    }
}
