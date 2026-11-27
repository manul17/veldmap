//! STAC catalog search controller (`StacSession`), canonical AOI geometry
//! ownership, origin-aware COG auth resolution, cached UI view DTOs (including
//! 1-click STAC Scene Recipes), and redacted URL formatting.
//!
//! Kept outside `ui/` so UI panels stay strictly presentation-only and free of
//! direct `veldmap-net` dependencies.

mod aoi;
mod auth;
mod session;
mod views;

use std::fmt;

use veldmap_core::geo::FootprintBounds;
use veldmap_net::sanitize_url_for_display;
use veldmap_raster::ExternalBandMeta;

use crate::layers::BandMathPreset;

pub(crate) use auth::StacCogAuthMode;
pub(crate) use session::{StacCatalogPreset, StacSearchRequest, StacSession};

/// URL string wrapper whose [`fmt::Debug`] implementation delegates to
/// [`veldmap_net::sanitize_url_for_display`] so userinfo and query tokens
/// (such as SAS signatures) can never leak through derived `Debug` output.
#[derive(Clone, Default, PartialEq, Eq)]
pub(crate) struct RedactedUrl(String);

impl RedactedUrl {
    pub(crate) fn new(url: impl Into<String>) -> Self {
        Self(url.into())
    }

    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }

    pub(crate) fn as_mut_string(&mut self) -> &mut String {
        &mut self.0
    }

    pub(crate) fn into_inner(self) -> String {
        self.0
    }
}

impl fmt::Debug for RedactedUrl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&sanitize_url_for_display(&self.0), f)
    }
}

/// Opaque single-COG streaming request payload handed from `ui::stac` back to
/// `VeldmapApp`, keeping `veldmap_raster::ExternalBandMeta` out of `ui/`.
#[derive(Clone, Debug)]
pub(crate) struct StacCogStreamSpec {
    pub(crate) url: RedactedUrl,
    pub(crate) title: String,
    pub(crate) auth_mode: StacCogAuthMode,
    pub(crate) external_bands: Vec<ExternalBandMeta>,
    pub(crate) is_scl: bool,
}

impl StacCogStreamSpec {
    pub(crate) fn direct(url: RedactedUrl, title: String, auth_mode: StacCogAuthMode) -> Self {
        Self {
            url,
            title,
            auth_mode,
            external_bands: Vec::new(),
            is_scl: false,
        }
    }
}

/// UI-facing view of a built-in zero-auth COG sample scene.
#[derive(Clone, Debug)]
pub(crate) struct SampleSceneView {
    pub(crate) title: &'static str,
    pub(crate) description: &'static str,
    pub(crate) cog_url: &'static str,
    pub(crate) external_bands: Vec<ExternalBandMeta>,
}

impl SampleSceneView {
    pub(crate) fn to_stream_spec(&self) -> StacCogStreamSpec {
        StacCogStreamSpec {
            url: RedactedUrl::new(self.cog_url),
            title: self.title.to_owned(),
            auth_mode: StacCogAuthMode::None,
            external_bands: self.external_bands.clone(),
            is_scl: false,
        }
    }
}

/// UI-facing view of one constituent COG asset in a 1-click STAC Band Math
/// scene recipe.
#[derive(Clone, Debug)]
pub(crate) struct StacRecipeAssetView {
    pub(crate) label: String,
    pub(crate) href: RedactedUrl,
    pub(crate) external_bands: Vec<ExternalBandMeta>,
}

/// UI-facing view of one 1-click STAC Scene Recipe (`True Color`, `NDVI + SCL`,
/// `NDWI + SCL`, `False Color NIR`, `SWIR`).
#[derive(Clone, Debug)]
pub(crate) struct StacRecipeView {
    pub(crate) preset: BandMathPreset,
    pub(crate) button_label: String,
    pub(crate) tooltip: String,
    pub(crate) bands: Vec<StacRecipeAssetView>,
    pub(crate) scl: Option<StacRecipeAssetView>,
}

/// UI-facing view of a built-in zero-auth multi-band sample scene recipe.
#[derive(Clone, Debug)]
pub(crate) struct SampleRecipeView {
    pub(crate) title: &'static str,
    pub(crate) description: &'static str,
    pub(crate) preset: BandMathPreset,
    pub(crate) bands: Vec<StacRecipeAssetView>,
    pub(crate) scl: Option<StacRecipeAssetView>,
}

/// UI-facing view of a single HTTP(S) COG asset on a STAC search result item.
#[derive(Clone, Debug)]
pub(crate) struct StacCogAssetView {
    pub(crate) key: String,
    pub(crate) title: Option<String>,
    pub(crate) href: RedactedUrl,
    pub(crate) gsd: Option<f64>,
    pub(crate) external_bands: Vec<ExternalBandMeta>,
    pub(crate) is_scl: bool,
}

impl StacCogAssetView {
    pub(crate) fn to_stream_spec(
        &self,
        title: String,
        auth_mode: StacCogAuthMode,
    ) -> StacCogStreamSpec {
        StacCogStreamSpec {
            url: self.href.clone(),
            title,
            auth_mode,
            external_bands: self.external_bands.clone(),
            is_scl: self.is_scl,
        }
    }
}

/// Cached UI-facing view of a STAC search result item, rebuilt only when search
/// results change so 60–120 Hz frames borrow `&[StacItemView]` with zero
/// per-frame allocations or geometry scans.
#[derive(Clone, Debug)]
pub(crate) struct StacItemView {
    pub(crate) id: String,
    pub(crate) meta_line: Option<String>,
    pub(crate) platform: Option<String>,
    pub(crate) processing_level: Option<String>,
    pub(crate) is_surface_reflectance: bool,
    pub(crate) preferred_cog: Option<StacCogAssetView>,
    pub(crate) recipes: Vec<StacRecipeView>,
    pub(crate) cog_assets: Vec<StacCogAssetView>,
    pub(crate) zoom_bounds: Option<FootprintBounds>,
}

/// Borrowed read-only snapshot of STAC session and catalog configuration state
/// passed into `ui::stac` so UI panels never depend on [`StacSession`],
/// `AppSettings`, or `veldmap-net`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct StacViewSnapshot<'a> {
    pub(crate) is_searching: bool,
    pub(crate) has_aoi: bool,
    pub(crate) aoi_bbox: Option<[f64; 4]>,
    pub(crate) aoi_label: Option<&'a str>,
    pub(crate) last_error: Option<&'a str>,
    pub(crate) matched: Option<u64>,
    pub(crate) selected_idx: Option<usize>,
    pub(crate) hovered_idx: Option<usize>,
    pub(crate) items: &'a [StacItemView],
    pub(crate) sample_scenes: &'static [SampleSceneView],
    pub(crate) sample_recipes: &'static [SampleRecipeView],
    pub(crate) custom_endpoint_configured: bool,
}

#[cfg(test)]
#[path = "stac_tests.rs"]
mod tests;
