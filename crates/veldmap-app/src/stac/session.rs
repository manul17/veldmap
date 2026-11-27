//! STAC catalog presets, async search orchestration (`StacSession`), cached
//! UI-facing view projections, and Pass-2 footprint/highlight overlay caches.

use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, OnceLock};

use veldmap_core::geo::GeoJsonGeometry;
use veldmap_core::sync::{CancelToken, Wakeup};
use veldmap_net::{
    NetContext, StacAsset, StacCatalogConfig, StacClient, StacItem, StacSearchQuery,
    StacSearchResult, builtin_sample_scenes,
};

use crate::render::{DrawVectorId, DrawVectorLayer};
use crate::storage::settings::StacSettings;

use super::aoi::{
    AoiState, build_vector_layer, footprint_bounds_from_bbox, polygon_geometry_from_bbox,
};
use super::auth::{StacCogAuthMode, resolve_cog_auth};
use super::{RedactedUrl, SampleSceneView, StacCogAssetView, StacItemView, StacViewSnapshot};

/// Default visual colors (sRGB RGB + linear alpha) for STAC overlays on globe.
const STAC_FOOTPRINT_FILL: [f32; 4] = [0.22, 0.62, 0.96, 0.08];
const STAC_FOOTPRINT_LINE: [f32; 4] = [0.35, 0.78, 1.0, 0.85];
const STAC_HIGHLIGHT_FILL: [f32; 4] = [1.0, 0.72, 0.14, 0.24];
const STAC_HIGHLIGHT_LINE: [f32; 4] = [1.0, 0.86, 0.24, 0.98];

/// Selectable STAC catalog preset inside the Cloud / STAC panel.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum StacCatalogPreset {
    #[default]
    AwsSentinel2L2a,
    AwsCopDemGlo30,
    Custom,
}

impl StacCatalogPreset {
    pub(crate) const ALL: [Self; 3] = [Self::AwsSentinel2L2a, Self::AwsCopDemGlo30, Self::Custom];

    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::AwsSentinel2L2a => "AWS Sentinel-2 L2A (Zero-Auth)",
            Self::AwsCopDemGlo30 => "AWS COP-DEM GLO-30 (Zero-Auth)",
            Self::Custom => "Custom STAC Endpoint",
        }
    }

    /// Return whether date-range and cloud-cover filters apply to this preset
    /// (false for static global elevation mosaics such as COP-DEM GLO-30).
    pub(crate) const fn supports_temporal_and_cloud_filters(self) -> bool {
        !matches!(self, Self::AwsCopDemGlo30)
    }

    fn to_catalog_config(self, stac_settings: &StacSettings) -> StacCatalogConfig {
        match self {
            Self::AwsSentinel2L2a => StacCatalogConfig::aws_sentinel2_l2a(),
            Self::AwsCopDemGlo30 => StacCatalogConfig::aws_cop_dem_glo30(),
            Self::Custom => stac_settings.to_custom_catalog(),
        }
    }

    /// Build a [`StacSearchQuery`] honoring preset-specific filter rules.
    fn build_search_query(
        self,
        aoi_geometry: Option<GeoJsonGeometry>,
        datetime_input: &str,
        filter_clouds: bool,
        max_cloud_cover: f32,
        limit: usize,
    ) -> StacSearchQuery {
        let dt = datetime_input.trim();
        let temporal = self.supports_temporal_and_cloud_filters();
        StacSearchQuery {
            bbox: None,
            intersects: aoi_geometry,
            datetime: (temporal && !dt.is_empty()).then(|| dt.to_owned()),
            max_cloud_cover: (temporal && filter_clouds).then_some(max_cloud_cover),
            limit,
        }
    }
}

/// Search parameters emitted by the Cloud / STAC UI form so [`StacSession`]
/// constructs [`StacSearchQuery`] from its own canonical AOI inside
/// `crate::stac`.
#[derive(Clone, Debug)]
pub(crate) struct StacSearchRequest {
    pub(crate) preset: StacCatalogPreset,
    pub(crate) datetime_input: String,
    pub(crate) filter_clouds: bool,
    pub(crate) max_cloud_cover: f32,
    pub(crate) limit: usize,
}

/// Cached UI-facing projection of built-in zero-auth COG sample scenes.
fn builtin_sample_scene_views() -> &'static [SampleSceneView] {
    static VIEWS: OnceLock<Vec<SampleSceneView>> = OnceLock::new();
    VIEWS.get_or_init(|| {
        builtin_sample_scenes()
            .iter()
            .map(|sample| SampleSceneView {
                title: sample.title,
                description: sample.description,
                cog_url: sample.cog_url,
                scale_offset: sample.scale_offset(),
            })
            .collect()
    })
}

fn build_asset_view(asset: &StacAsset) -> StacCogAssetView {
    StacCogAssetView {
        key: asset.key.clone(),
        title: asset.title.clone(),
        href: RedactedUrl::new(asset.href.clone()),
        gsd: asset.gsd,
        scale_offset: asset.primary_scale_offset(),
    }
}

fn build_item_view(item: &StacItem) -> StacItemView {
    let short_date = item
        .datetime
        .as_deref()
        .map(|dt| dt.split('T').next().unwrap_or(dt).to_owned());
    let preferred_cog = item.preferred_cog_asset().map(build_asset_view);
    let cog_assets = item
        .cog_assets()
        .into_iter()
        .map(build_asset_view)
        .collect();
    let zoom_bounds = item
        .bbox
        .or_else(|| item.geometry.as_ref().and_then(GeoJsonGeometry::bbox))
        .and_then(footprint_bounds_from_bbox);
    StacItemView {
        id: item.id.clone(),
        short_date,
        cloud_cover: item.cloud_cover,
        mgrs_tile: item.mgrs_tile.clone(),
        preferred_cog,
        cog_assets,
        zoom_bounds,
    }
}

fn item_geometry(item: &StacItem) -> Option<GeoJsonGeometry> {
    if let Some(geom) = &item.geometry {
        Some(geom.clone())
    } else {
        item.bbox.and_then(polygon_geometry_from_bbox)
    }
}

struct SearchDoneMsg {
    seq: u64,
    uses_custom_auth: bool,
    catalog_endpoint: String,
    outcome: Result<StacSearchResult, String>,
}

/// Domain controller for STAC catalog search, AOI management, and globe scene
/// footprint overlays.
pub(crate) struct StacSession {
    net: Option<Arc<NetContext>>,
    wakeup: Wakeup,
    shutdown: CancelToken,
    tx: Sender<SearchDoneMsg>,
    rx: Receiver<SearchDoneMsg>,
    active_seq: u64,
    active_cancel: Option<CancelToken>,
    searching: bool,
    aoi: AoiState,
    last_result: Option<StacSearchResult>,
    item_views: Vec<StacItemView>,
    result_uses_custom_auth: bool,
    result_catalog_endpoint: Option<String>,
    last_error: Option<String>,
    selected_idx: Option<usize>,
    hovered_idx: Option<usize>,
    rev: u64,
    footprints_draw: Option<DrawVectorLayer>,
    highlight_draw: Option<DrawVectorLayer>,
}

impl StacSession {
    pub(crate) fn new(net: Option<Arc<NetContext>>, wakeup: Wakeup, shutdown: CancelToken) -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            net,
            wakeup,
            shutdown,
            tx,
            rx,
            active_seq: 0,
            active_cancel: None,
            searching: false,
            aoi: AoiState::default(),
            last_result: None,
            item_views: Vec::new(),
            result_uses_custom_auth: false,
            result_catalog_endpoint: None,
            last_error: None,
            selected_idx: None,
            hovered_idx: None,
            rev: 1,
            footprints_draw: None,
            highlight_draw: None,
        }
    }

    fn bump_rev(&mut self) -> u64 {
        self.rev = self.rev.wrapping_add(1);
        self.rev
    }

    /// Monotonic revision incremented only when visual Pass-2 overlays change
    /// (AOI, preview, footprints, or highlight), never on UI-only error text.
    pub(crate) const fn rev(&self) -> u64 {
        self.rev
    }

    /// Borrow a zero-allocation read-only snapshot for the Cloud / STAC UI tab.
    pub(crate) fn view_snapshot(&self, stac_settings: &StacSettings) -> StacViewSnapshot<'_> {
        StacViewSnapshot {
            is_searching: self.searching,
            has_aoi: self.aoi.has_geometry(),
            aoi_bbox: self.aoi.bbox(),
            aoi_label: self.aoi.label(),
            last_error: self.last_error.as_deref(),
            matched: self.matched(),
            selected_idx: self.selected_idx,
            hovered_idx: self.hovered_idx,
            items: &self.item_views,
            sample_scenes: builtin_sample_scene_views(),
            custom_endpoint_configured: !stac_settings.custom_endpoint.trim().is_empty(),
        }
    }

    /// Resolve origin-aware URL & custom HTTP headers for opening a remote COG
    /// using this session's last search-result auth context and `stac_settings`.
    pub(crate) fn resolve_cog_auth(
        &self,
        url: &str,
        mode: StacCogAuthMode,
        stac_settings: &StacSettings,
    ) -> (String, Vec<(String, String)>) {
        resolve_cog_auth(
            url,
            mode,
            self.result_uses_custom_auth,
            self.result_catalog_endpoint.as_deref(),
            stac_settings,
        )
    }

    #[cfg(test)]
    pub(crate) fn last_error(&self) -> Option<&str> {
        self.last_error.as_deref()
    }

    /// Record a STAC/Cloud error as the single source of truth without bumping
    /// [`Self::rev`] so UI-only errors never rebuild `DrawSnapshot`.
    pub(crate) fn set_error(&mut self, msg: String) {
        self.last_error = Some(msg);
    }

    /// Clear the active STAC/Cloud error without bumping [`Self::rev`].
    pub(crate) fn clear_error(&mut self) {
        self.last_error = None;
    }

    #[cfg(test)]
    pub(super) fn seed_queued_search_for_test(
        &mut self,
        seq: u64,
        catalog_endpoint: &str,
        result: StacSearchResult,
    ) {
        self.active_seq = seq;
        self.searching = true;
        let _ = self.tx.send(SearchDoneMsg {
            seq,
            uses_custom_auth: false,
            catalog_endpoint: catalog_endpoint.to_owned(),
            outcome: Ok(result),
        });
    }

    #[cfg(test)]
    pub(super) fn set_result_auth_context_for_test(
        &mut self,
        uses_custom_auth: bool,
        catalog_endpoint: Option<&str>,
    ) {
        self.result_uses_custom_auth = uses_custom_auth;
        self.result_catalog_endpoint = catalog_endpoint.map(str::to_owned);
    }

    #[cfg(test)]
    pub(crate) const fn aoi_bbox(&self) -> Option<[f64; 4]> {
        self.aoi.bbox()
    }

    pub(crate) fn export_aoi_geojson(&self) -> Option<String> {
        self.aoi.export_geojson()
    }

    pub(crate) fn export_aoi_wkt(&self) -> Option<String> {
        self.aoi.export_wkt()
    }

    pub(crate) fn set_aoi_bbox(&mut self, bbox: [f64; 4], label: String) -> bool {
        let rev = self.rev.wrapping_add(1);
        if !self.aoi.set_bbox(bbox, label, rev) {
            return false;
        }
        self.rev = rev;
        true
    }

    pub(crate) fn set_aoi_geometry(&mut self, geometry: GeoJsonGeometry, label: String) {
        let rev = self.bump_rev();
        self.aoi.set_geometry(geometry, label, rev);
    }

    /// Update the live drag preview rectangle without overwriting any existing
    /// committed AOI until the drag finishes.
    pub(crate) fn set_aoi_preview_bbox(&mut self, bbox: [f64; 4]) -> bool {
        let rev = self.rev.wrapping_add(1);
        if !self.aoi.set_preview_bbox(bbox, rev) {
            return false;
        }
        self.rev = rev;
        true
    }

    /// Clear any active AOI drag preview, restoring the committed AOI overlay.
    pub(crate) fn clear_aoi_preview(&mut self) -> bool {
        if !self.aoi.clear_preview() {
            return false;
        }
        self.bump_rev();
        true
    }

    pub(crate) fn clear_aoi(&mut self) -> bool {
        if !self.aoi.clear() {
            return false;
        }
        self.bump_rev();
        true
    }

    fn items(&self) -> &[StacItem] {
        self.last_result
            .as_ref()
            .map_or(&[], |res| res.items.as_slice())
    }

    pub(super) fn matched(&self) -> Option<u64> {
        self.last_result.as_ref().and_then(|res| res.matched)
    }

    pub(crate) fn set_selected(&mut self, idx: Option<usize>) -> bool {
        let clamped = idx.filter(|&i| i < self.items().len());
        if self.selected_idx == clamped {
            return false;
        }
        self.selected_idx = clamped;
        self.rebuild_highlight();
        true
    }

    pub(crate) fn set_hovered(&mut self, idx: Option<usize>) -> bool {
        let clamped = idx.filter(|&i| i < self.items().len());
        if self.hovered_idx == clamped {
            return false;
        }
        self.hovered_idx = clamped;
        self.rebuild_highlight();
        true
    }

    pub(crate) fn clear_results(&mut self) -> bool {
        let had_overlays = self.last_result.is_some()
            || self.footprints_draw.is_some()
            || self.highlight_draw.is_some();
        let had_error = self.last_error.is_some();
        if !had_overlays && !had_error {
            return false;
        }
        self.last_result = None;
        self.item_views.clear();
        self.result_uses_custom_auth = false;
        self.result_catalog_endpoint = None;
        self.last_error = None;
        self.selected_idx = None;
        self.hovered_idx = None;
        self.footprints_draw = None;
        self.highlight_draw = None;
        if had_overlays {
            self.bump_rev();
        }
        true
    }

    /// Launch an asynchronous STAC `/search` query from `request` and the
    /// session's current AOI, cancelling any prior in-flight search immediately.
    pub(crate) fn begin_search(
        &mut self,
        request: &StacSearchRequest,
        stac_settings: &StacSettings,
    ) -> Result<(), String> {
        let Some(net) = self.net.clone() else {
            return Err("network subsystem is unavailable".to_owned());
        };
        let catalog = request.preset.to_catalog_config(stac_settings);
        let uses_custom_auth = request.preset == StacCatalogPreset::Custom;
        let query = request.preset.build_search_query(
            self.aoi.geometry().cloned(),
            &request.datetime_input,
            request.filter_clouds,
            request.max_cloud_cover,
            request.limit,
        );

        if let Some(prev) = self.active_cancel.take() {
            prev.cancel();
        }
        self.active_seq = self.active_seq.saturating_add(1);
        let seq = self.active_seq;
        let cancel = self.shutdown.child_token();
        self.active_cancel = Some(cancel.clone());
        self.searching = true;
        self.last_error = None;

        let catalog_endpoint = catalog.endpoint.clone();
        let tx = self.tx.clone();
        let wakeup = Arc::clone(&self.wakeup);
        let client = StacClient::new(Arc::clone(&net));
        let handle = net.handle().clone();
        handle.spawn(async move {
            let outcome = match client.search_async(&catalog, &query, &cancel).await {
                Ok(res) => Ok(res),
                Err(err) if err.is_cancelled() => return,
                Err(err) => Err(err.to_string()),
            };
            if !cancel.is_cancelled() {
                let _ = tx.send(SearchDoneMsg {
                    seq,
                    uses_custom_auth,
                    catalog_endpoint,
                    outcome,
                });
                (wakeup)();
            }
        });
        Ok(())
    }

    /// Cancel any in-flight STAC search and invalidate its sequence number so
    /// an already-queued response can never overwrite session state.
    pub(crate) fn cancel(&mut self) {
        if let Some(tok) = self.active_cancel.take() {
            tok.cancel();
        }
        self.active_seq = self.active_seq.saturating_add(1);
        self.searching = false;
        self.last_error = None;
    }

    /// Drain completed STAC search responses, rebuilding cached UI views and
    /// globe footprint meshes when results arrive. Returns `true` if scene
    /// overlays or UI state changed.
    pub(crate) fn poll_results(&mut self) -> bool {
        let mut changed = false;
        while let Ok(msg) = self.rx.try_recv() {
            if msg.seq != self.active_seq {
                continue;
            }
            self.searching = false;
            self.active_cancel = None;
            changed = true;
            match msg.outcome {
                Ok(result) => {
                    self.last_error = None;
                    self.selected_idx = (!result.items.is_empty()).then_some(0);
                    self.hovered_idx = None;
                    self.result_uses_custom_auth = msg.uses_custom_auth;
                    self.result_catalog_endpoint = Some(msg.catalog_endpoint);
                    self.item_views = result.items.iter().map(build_item_view).collect();
                    self.last_result = Some(result);
                    self.rebuild_footprints();
                    self.rebuild_highlight();
                }
                Err(err) => {
                    self.last_error = Some(err);
                }
            }
        }
        changed
    }

    fn rebuild_footprints(&mut self) {
        let geoms: Vec<GeoJsonGeometry> = self.items().iter().filter_map(item_geometry).collect();
        let rev = self.bump_rev();
        self.footprints_draw = build_vector_layer(
            DrawVectorId::StacFootprints,
            rev,
            geoms,
            STAC_FOOTPRINT_FILL,
            STAC_FOOTPRINT_LINE,
            1.8,
            0.0,
        );
    }

    fn rebuild_highlight(&mut self) {
        let active_idx = self.hovered_idx.or(self.selected_idx);
        let geom = active_idx
            .and_then(|idx| self.items().get(idx))
            .and_then(item_geometry);
        let rev = self.bump_rev();
        self.highlight_draw = geom.and_then(|g| {
            build_vector_layer(
                DrawVectorId::StacHighlight,
                rev,
                vec![g],
                STAC_HIGHLIGHT_FILL,
                STAC_HIGHLIGHT_LINE,
                3.0,
                10.0,
            )
        });
    }

    /// Append cached STAC scene footprint, highlight, and AOI (or live drag
    /// preview) overlays into `out` in painter's order.
    pub(crate) fn push_draw_layers(&self, out: &mut Vec<DrawVectorLayer>) {
        if let Some(fp) = &self.footprints_draw {
            out.push(fp.clone());
        }
        if let Some(hi) = &self.highlight_draw {
            out.push(hi.clone());
        }
        if let Some(aoi) = self.aoi.draw_layer() {
            out.push(aoi.clone());
        }
    }
}
