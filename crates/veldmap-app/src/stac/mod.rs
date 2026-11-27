//! STAC catalog search controller (`StacSession`), canonical AOI geometry
//! ownership, origin-aware COG auth resolution, cached UI view DTOs, and
//! redacted URL formatting.
//!
//! Kept outside `ui/` so UI panels stay strictly presentation-only and free of
//! direct `veldmap-net` dependencies.

mod aoi;
mod auth;
mod session;

use std::fmt;

use veldmap_core::geo::FootprintBounds;
use veldmap_net::sanitize_url_for_display;

pub(crate) use auth::{StacCogAuthMode, resolve_cog_auth};
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

/// UI-facing view of a built-in zero-auth COG sample scene.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SampleSceneView {
    pub(crate) title: &'static str,
    pub(crate) description: &'static str,
    pub(crate) cog_url: &'static str,
    pub(crate) scale_offset: Option<(f64, f64)>,
}

/// UI-facing view of a single HTTP(S) COG asset on a STAC search result item.
#[derive(Clone, Debug)]
pub(crate) struct StacCogAssetView {
    pub(crate) key: String,
    pub(crate) title: Option<String>,
    pub(crate) href: RedactedUrl,
    pub(crate) gsd: Option<f64>,
    pub(crate) scale_offset: Option<(f64, f64)>,
}

/// Cached UI-facing view of a STAC search result item, rebuilt only when search
/// results change so 60–120 Hz frames borrow `&[StacItemView]` with zero
/// per-frame allocations or geometry scans.
#[derive(Clone, Debug)]
pub(crate) struct StacItemView {
    pub(crate) id: String,
    pub(crate) short_date: Option<String>,
    pub(crate) cloud_cover: Option<f32>,
    pub(crate) mgrs_tile: Option<String>,
    pub(crate) preferred_cog: Option<StacCogAssetView>,
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
    pub(crate) custom_endpoint_configured: bool,
}

#[cfg(test)]
#[path = "stac_tests.rs"]
mod tests;
