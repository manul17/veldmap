//! STAC catalog presets, async search & `links[rel="next"]` pagination
//! orchestration (`StacSession`), and Pass-2 footprint/highlight overlay caches.

use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, Sender};

use veldmap_core::geo::GeoJsonGeometry;
use veldmap_core::sync::{CancelToken, Wakeup};
use veldmap_net::{
    MAX_STAC_PAGINATED_ITEMS, NetContext, StacAuth, StacCatalogConfig, StacClient, StacItem,
    StacNextLink, StacSearchQuery, StacSearchResult, StacSortBy,
};

use crate::layers::{StacCogAuthMode, TimelineOpenRequest};
use crate::render::{DrawVectorId, DrawVectorLayer};
use crate::storage::settings::StacSettings;

use super::aoi::{AoiState, build_vector_layer};
use super::auth::{effective_persisted_auth_mode, resolve_cog_auth};
use super::timeline::{
    StacTimelineGroupView, StacTimelineMode, build_stac_timeline_request, build_timeline_groups,
};
use super::views::{
    build_item_view, builtin_sample_recipe_views, builtin_sample_scene_views, item_geometry,
};
use super::{StacItemView, StacViewSnapshot};

/// Default visual colors (sRGB RGB + linear alpha) for STAC overlays on globe.
const STAC_FOOTPRINT_FILL: [f32; 4] = [0.22, 0.62, 0.96, 0.08];
const STAC_FOOTPRINT_LINE: [f32; 4] = [0.35, 0.78, 1.0, 0.85];
const STAC_HIGHLIGHT_FILL: [f32; 4] = [1.0, 0.72, 0.14, 0.24];
const STAC_HIGHLIGHT_LINE: [f32; 4] = [1.0, 0.86, 0.24, 0.98];

/// Read-only UI summary of one configured Custom STAC profile (never exposes
/// credentials).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct StacProfileView {
    pub(crate) index: usize,
    pub(crate) name: String,
    pub(crate) default_collection: String,
    pub(crate) configured: bool,
}

/// UI-safe server-side sort selector for STAC `/search` `sortby` (decoupling
/// `ui::stac` from `veldmap-net`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum StacSortOption {
    #[default]
    Default,
    Newest,
    LeastCloud,
}

impl StacSortOption {
    pub(crate) const ALL: [Self; 3] = [Self::Default, Self::Newest, Self::LeastCloud];

    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Default => "Catalog Default",
            Self::Newest => "Newest First",
            Self::LeastCloud => "Lowest Cloud %",
        }
    }

    const fn to_net_sort(self) -> Option<StacSortBy> {
        match self {
            Self::Default => None,
            Self::Newest => Some(StacSortBy::Newest),
            Self::LeastCloud => Some(StacSortBy::LeastCloud),
        }
    }
}

/// Selectable STAC catalog preset inside the Cloud / STAC panel.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum StacCatalogPreset {
    #[default]
    AwsSentinel2L2a,
    AwsCopDemGlo30,
    CustomProfile(usize),
}

impl StacCatalogPreset {
    pub(crate) const BUILTINS: [Self; 2] = [Self::AwsSentinel2L2a, Self::AwsCopDemGlo30];

    pub(crate) const fn builtin_label(self) -> &'static str {
        match self {
            Self::AwsSentinel2L2a => "AWS Sentinel-2 L2A (Zero-Auth)",
            Self::AwsCopDemGlo30 => "AWS COP-DEM GLO-30 (Zero-Auth)",
            Self::CustomProfile(_) => "Custom STAC Profile",
        }
    }

    pub(crate) const fn default_collection(self) -> &'static str {
        match self {
            Self::AwsSentinel2L2a => "sentinel-2-l2a",
            Self::AwsCopDemGlo30 => "cop-dem-glo-30",
            Self::CustomProfile(_) => "",
        }
    }

    pub(crate) const fn is_custom(self) -> bool {
        matches!(self, Self::CustomProfile(_))
    }

    /// Return whether date-range and cloud-cover filters apply to this preset
    /// (false for static global elevation mosaics such as COP-DEM GLO-30).
    pub(crate) const fn supports_temporal_and_cloud_filters(self) -> bool {
        !matches!(self, Self::AwsCopDemGlo30)
    }

    fn to_catalog_config(
        self,
        collection_override: &str,
        stac_settings: &StacSettings,
    ) -> Result<StacCatalogConfig, String> {
        let mut cfg = match self {
            Self::AwsSentinel2L2a => StacCatalogConfig::aws_sentinel2_l2a(),
            Self::AwsCopDemGlo30 => StacCatalogConfig::aws_cop_dem_glo30(),
            Self::CustomProfile(idx) => {
                let Some(profile) = stac_settings
                    .profile(idx)
                    .or_else(|| stac_settings.active_profile())
                else {
                    return Err(
                        "No Custom STAC profile configured in Settings → STAC / Cloud.".to_owned(),
                    );
                };
                profile.to_catalog_config()
            }
        };
        let ov = collection_override.trim();
        if !ov.is_empty() {
            cfg.collection = ov.to_owned();
        }
        Ok(cfg)
    }

    /// Build a [`StacSearchQuery`] honoring preset-specific filter rules.
    ///
    /// Uses native STAC `"bbox"` when the AOI originated from a bounding box
    /// (avoiding OGC touching-polygon issues and spherical geodesic bow on wide
    /// boxes), and `"intersects"` for drawn/imported arbitrary geometries.
    fn build_search_query(self, aoi: &AoiState, request: &StacSearchRequest) -> StacSearchQuery {
        let dt = request.datetime_input.trim();
        let temporal = self.supports_temporal_and_cloud_filters();
        let (bbox, intersects) = match aoi.query_bbox() {
            Some(b) => (Some(b), None),
            None => (None, aoi.geometry().cloned()),
        };
        StacSearchQuery {
            bbox,
            intersects,
            datetime: (temporal && !dt.is_empty()).then(|| dt.to_owned()),
            max_cloud_cover: (temporal && request.filter_clouds).then_some(request.max_cloud_cover),
            sort_by: temporal.then(|| request.sort_by.to_net_sort()).flatten(),
            limit: request.limit,
        }
    }
}

/// Search parameters emitted by the Cloud / STAC UI form so [`StacSession`]
/// constructs [`StacSearchQuery`] from its own canonical AOI inside
/// `crate::stac`.
#[derive(Clone, Debug)]
pub(crate) struct StacSearchRequest {
    pub(crate) preset: StacCatalogPreset,
    pub(crate) collection_override: String,
    pub(crate) datetime_input: String,
    pub(crate) filter_clouds: bool,
    pub(crate) max_cloud_cover: f32,
    pub(crate) sort_by: StacSortOption,
    pub(crate) limit: usize,
}

struct SearchDoneMsg {
    seq: u64,
    append: bool,
    uses_custom_auth: bool,
    catalog: Option<StacCatalogConfig>,
    query: Option<StacSearchQuery>,
    outcome: Result<StacSearchResult, String>,
}

/// Hard bound on paginated STAC page fetches per search session so a server
/// emitting infinite distinct empty-page cursors cannot loop unboundedly.
const MAX_STAC_PAGES: usize = 50;

/// Domain controller for STAC catalog search, bounded pagination, AOI
/// management, and globe scene footprint overlays.
pub(crate) struct StacSession {
    net: Option<Arc<NetContext>>,
    wakeup: Wakeup,
    shutdown: CancelToken,
    tx: Sender<SearchDoneMsg>,
    rx: Receiver<SearchDoneMsg>,
    active_seq: u64,
    active_cancel: Option<CancelToken>,
    searching: bool,
    loading_more: bool,
    aoi: AoiState,
    last_result: Option<StacSearchResult>,
    last_catalog: Option<StacCatalogConfig>,
    last_query: Option<StacSearchQuery>,
    next_link: Option<StacNextLink>,
    visited_next_cursors: std::collections::HashSet<(String, u8, Option<String>, bool)>,
    pages_fetched: usize,
    truncated_at_cap: bool,
    item_views: Vec<StacItemView>,
    timeline_groups: Vec<StacTimelineGroupView>,
    result_uses_custom_auth: bool,
    result_catalog_endpoint: Option<String>,
    result_auth: Option<StacAuth>,
    last_error: Option<String>,
    selected_idx: Option<usize>,
    hovered_idx: Option<usize>,
    rev: u64,
    footprints_draw: Option<DrawVectorLayer>,
    highlight_draw: Option<DrawVectorLayer>,
}

fn next_link_cursor_key(link: &StacNextLink) -> (String, u8, Option<String>, bool) {
    let method_tag = match link.method {
        veldmap_net::StacNextMethod::Get => 0,
        veldmap_net::StacNextMethod::Post => 1,
    };
    (link.href.clone(), method_tag, link.body.clone(), link.merge)
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
            loading_more: false,
            aoi: AoiState::default(),
            last_result: None,
            last_catalog: None,
            last_query: None,
            next_link: None,
            visited_next_cursors: std::collections::HashSet::new(),
            pages_fetched: 0,
            truncated_at_cap: false,
            item_views: Vec::new(),
            timeline_groups: Vec::new(),
            result_uses_custom_auth: false,
            result_catalog_endpoint: None,
            result_auth: None,
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

    /// Borrow a read-only snapshot for the Cloud / STAC UI tab.
    pub(crate) fn view_snapshot(&self, stac_settings: &StacSettings) -> StacViewSnapshot<'_> {
        let count = self.items().len();
        let reached_page_cap = count >= MAX_STAC_PAGINATED_ITEMS
            && (self.next_link.is_some() || self.truncated_at_cap);
        let can_load_more =
            self.next_link.is_some() && !self.searching && count < MAX_STAC_PAGINATED_ITEMS;
        let custom_profiles = stac_settings
            .profiles
            .iter()
            .enumerate()
            .map(|(index, p)| StacProfileView {
                index,
                name: p.name.clone(),
                default_collection: p.collection.clone(),
                configured: p.is_configured(),
            })
            .collect();
        StacViewSnapshot {
            is_searching: self.searching,
            is_loading_more: self.loading_more,
            can_load_more,
            reached_page_cap,
            has_aoi: self.aoi.has_geometry(),
            aoi_bbox: self.aoi.bbox(),
            aoi_label: self.aoi.label(),
            last_error: self.last_error.as_deref(),
            matched: self.matched(),
            selected_idx: self.selected_idx,
            hovered_idx: self.hovered_idx,
            items: &self.item_views,
            timeline_groups: &self.timeline_groups,
            sample_scenes: builtin_sample_scene_views(),
            sample_recipes: builtin_sample_recipe_views(),
            custom_profiles,
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
            self.result_auth.as_ref(),
            stac_settings,
        )
    }

    /// Normalize `mode` against this session's active search-result auth flag
    /// for safe persistence in `session.json` / `.veldmap.json`.
    pub(crate) fn effective_persisted_auth_mode(&self, mode: StacCogAuthMode) -> StacCogAuthMode {
        effective_persisted_auth_mode(mode, self.result_uses_custom_auth)
    }

    /// Build a chronologically ordered [`TimelineOpenRequest`] from currently
    /// loaded STAC search results matching `group_key` (`mgrs_tile` or `"AOI"`).
    pub(crate) fn build_timeline_request(
        &self,
        group_key: &str,
        mode: StacTimelineMode,
        stac_settings: &StacSettings,
    ) -> Result<TimelineOpenRequest, String> {
        let auth_mode = self.effective_persisted_auth_mode(StacCogAuthMode::FromSearchResult);
        build_stac_timeline_request(
            self.items(),
            &self.item_views,
            group_key,
            mode,
            auth_mode,
            stac_settings,
            |url, m| self.resolve_cog_auth(url, m, stac_settings),
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
        self.loading_more = false;
        let _ = self.tx.send(SearchDoneMsg {
            seq,
            append: false,
            uses_custom_auth: false,
            catalog: Some(StacCatalogConfig {
                endpoint: catalog_endpoint.to_owned(),
                collection: String::new(),
                auth: StacAuth::None,
            }),
            query: Some(StacSearchQuery::default()),
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
        self.result_auth = None;
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

    pub(crate) fn items(&self) -> &[StacItem] {
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
        let prev_active = self.hovered_idx.or(self.selected_idx);
        self.selected_idx = clamped;
        if self.hovered_idx.or(self.selected_idx) != prev_active {
            self.rebuild_highlight();
        }
        true
    }

    pub(crate) fn set_hovered(&mut self, idx: Option<usize>) -> bool {
        let clamped = idx.filter(|&i| i < self.items().len());
        if self.hovered_idx == clamped {
            return false;
        }
        let prev_active = self.hovered_idx.or(self.selected_idx);
        self.hovered_idx = clamped;
        if self.hovered_idx.or(self.selected_idx) != prev_active {
            self.rebuild_highlight();
        }
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
        self.last_catalog = None;
        self.last_query = None;
        self.next_link = None;
        self.visited_next_cursors.clear();
        self.pages_fetched = 0;
        self.truncated_at_cap = false;
        self.item_views.clear();
        self.timeline_groups.clear();
        self.result_uses_custom_auth = false;
        self.result_catalog_endpoint = None;
        self.result_auth = None;
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
        let catalog = request
            .preset
            .to_catalog_config(&request.collection_override, stac_settings)?;
        let uses_custom_auth = request.preset.is_custom();
        let query = request.preset.build_search_query(&self.aoi, request);

        if let Some(prev) = self.active_cancel.take() {
            prev.cancel();
        }
        self.active_seq = self.active_seq.saturating_add(1);
        let seq = self.active_seq;
        let cancel = self.shutdown.child_token();
        self.active_cancel = Some(cancel.clone());
        self.searching = true;
        self.loading_more = false;
        self.last_error = None;

        let catalog_for_msg = catalog.clone();
        let query_for_msg = query.clone();
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
                    append: false,
                    uses_custom_auth,
                    catalog: Some(catalog_for_msg),
                    query: Some(query_for_msg),
                    outcome,
                });
                (wakeup)();
            }
        });
        Ok(())
    }

    /// Follow the current result's `links[rel="next"]` pagination link,
    /// appending up to [`MAX_STAC_PAGINATED_ITEMS`] cumulative items.
    pub(crate) fn begin_load_more(&mut self) -> Result<(), String> {
        if self.searching {
            return Ok(());
        }
        if self.items().len() >= MAX_STAC_PAGINATED_ITEMS {
            return Ok(());
        }
        let Some(net) = self.net.clone() else {
            return Err("network subsystem is unavailable".to_owned());
        };
        let Some(next_link) = self.next_link.clone() else {
            return Ok(());
        };
        if self.pages_fetched >= MAX_STAC_PAGES {
            self.next_link = None;
            return Ok(());
        }
        let cursor_key = next_link_cursor_key(&next_link);
        if !self.visited_next_cursors.insert(cursor_key) {
            self.next_link = None;
            return Ok(());
        }
        let (Some(catalog), Some(prev_query)) =
            (self.last_catalog.clone(), self.last_query.clone())
        else {
            return Err("no active STAC search context to paginate".to_owned());
        };

        if let Some(prev) = self.active_cancel.take() {
            prev.cancel();
        }
        self.active_seq = self.active_seq.saturating_add(1);
        let seq = self.active_seq;
        let cancel = self.shutdown.child_token();
        self.active_cancel = Some(cancel.clone());
        self.searching = true;
        self.loading_more = true;
        self.last_error = None;

        let uses_custom_auth = self.result_uses_custom_auth;
        let tx = self.tx.clone();
        let wakeup = Arc::clone(&self.wakeup);
        let client = StacClient::new(Arc::clone(&net));
        let handle = net.handle().clone();
        handle.spawn(async move {
            let outcome = match client
                .search_next_async(&catalog, &prev_query, &next_link, &cancel)
                .await
            {
                Ok(res) => Ok(res),
                Err(err) if err.is_cancelled() => return,
                Err(err) => Err(err.to_string()),
            };
            if !cancel.is_cancelled() {
                let _ = tx.send(SearchDoneMsg {
                    seq,
                    append: true,
                    uses_custom_auth,
                    catalog: None,
                    query: None,
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
        self.loading_more = false;
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
            self.loading_more = false;
            self.active_cancel = None;
            changed = true;
            match msg.outcome {
                Ok(mut result) => {
                    self.last_error = None;
                    let next_link = result.next_link.take();
                    if msg.append
                        && let Some(existing) = &mut self.last_result
                    {
                        self.pages_fetched = self.pages_fetched.saturating_add(1);
                        if let Some(m) = result.matched {
                            existing.matched = Some(m);
                        }
                        for item in result.items {
                            if !existing.items.iter().any(|e| e.id == item.id) {
                                if existing.items.len() >= MAX_STAC_PAGINATED_ITEMS {
                                    self.truncated_at_cap = true;
                                    break;
                                }
                                self.item_views.push(build_item_view(&item));
                                existing.items.push(item);
                            }
                        }
                        self.next_link = next_link.filter(|link| {
                            self.pages_fetched < MAX_STAC_PAGES
                                && !self
                                    .visited_next_cursors
                                    .contains(&next_link_cursor_key(link))
                        });
                        if self.selected_idx.is_none() && !existing.items.is_empty() {
                            self.selected_idx = Some(0);
                        }
                        self.timeline_groups =
                            build_timeline_groups(&existing.items, &self.item_views);
                        self.rebuild_footprints();
                        self.rebuild_highlight();
                    } else {
                        self.visited_next_cursors.clear();
                        self.pages_fetched = 1;
                        self.truncated_at_cap = result.items.len() > MAX_STAC_PAGINATED_ITEMS;
                        result.items.truncate(MAX_STAC_PAGINATED_ITEMS);
                        self.selected_idx = (!result.items.is_empty()).then_some(0);
                        self.hovered_idx = None;
                        self.result_uses_custom_auth = msg.uses_custom_auth;
                        if let Some(cat) = msg.catalog {
                            self.result_catalog_endpoint = Some(cat.endpoint.clone());
                            self.result_auth = msg.uses_custom_auth.then(|| cat.auth.clone());
                            self.last_catalog = Some(cat);
                        }
                        if let Some(q) = msg.query {
                            self.last_query = Some(q);
                        }
                        self.next_link = next_link;
                        self.item_views = result.items.iter().map(build_item_view).collect();
                        self.timeline_groups =
                            build_timeline_groups(&result.items, &self.item_views);
                        self.last_result = Some(result);
                        self.rebuild_footprints();
                        self.rebuild_highlight();
                    }
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
