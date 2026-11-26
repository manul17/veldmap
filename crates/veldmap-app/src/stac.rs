//! STAC catalog search controller (`StacSession`): catalog presets, query
//! construction, asynchronous catalog querying via [`veldmap_net::StacClient`],
//! canonical AOI geometry ownership (drawn box, viewport, or imported GeoJSON
//! polygon), GeoJSON/WKT export, origin-aware COG auth resolution, and cached
//! Pass-2 globe scene outline [`DrawVectorLayer`] generation.
//!
//! Kept outside `ui/` so UI panels stay strictly presentation-only.

use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, Sender};

use veldmap_core::geo::{
    DEFAULT_VECTOR_MAX_CHORD_DEG, FootprintBounds, GeoJsonFeature, GeoJsonFeatureCollection,
    GeoJsonGeometry, GeoPoint, Polygon, Ring, VectorMesh, export_geojson_geometry,
    export_wkt_geometry,
};
use veldmap_core::sync::{CancelToken, Wakeup};
use veldmap_net::{
    NetContext, StacAuth, StacCatalogConfig, StacClient, StacItem, StacSearchQuery,
    StacSearchResult, same_http_origin,
};

use crate::render::{DrawVectorId, DrawVectorLayer};
use crate::storage::settings::AppSettings;

/// Default visual colors (sRGB RGB + linear alpha) for STAC overlays on globe.
const STAC_FOOTPRINT_FILL: [f32; 4] = [0.22, 0.62, 0.96, 0.08];
const STAC_FOOTPRINT_LINE: [f32; 4] = [0.35, 0.78, 1.0, 0.85];
const STAC_HIGHLIGHT_FILL: [f32; 4] = [1.0, 0.72, 0.14, 0.24];
const STAC_HIGHLIGHT_LINE: [f32; 4] = [1.0, 0.86, 0.24, 0.98];
const STAC_AOI_FILL: [f32; 4] = [0.16, 0.88, 0.56, 0.14];
const STAC_AOI_LINE: [f32; 4] = [0.22, 0.96, 0.66, 0.96];

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

    pub(crate) fn to_catalog_config(self, settings: &AppSettings) -> StacCatalogConfig {
        match self {
            Self::AwsSentinel2L2a => StacCatalogConfig::aws_sentinel2_l2a(),
            Self::AwsCopDemGlo30 => StacCatalogConfig::aws_cop_dem_glo30(),
            Self::Custom => settings.stac.to_custom_catalog(),
        }
    }

    /// Build a [`StacSearchQuery`] honoring preset-specific filter rules.
    pub(crate) fn build_search_query(
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

/// Authentication policy for opening a remote COG from the Cloud / STAC panel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StacCogAuthMode {
    /// Zero-auth public scene (built-in sample scenes).
    None,
    /// Inherit auth mode & catalog origin from the STAC search that produced
    /// the displayed result list (decoupled from the live preset dropdown).
    FromSearchResult,
    /// Explicit user opt-in on a manually entered Direct COG URL.
    DirectCustom,
}

/// Return `true` if `url` already carries a cloud presigned-URL signature query
/// parameter (AWS S3, Azure SAS, GCS) so we never double-sign or attach
/// conflicting auth headers.
pub(crate) fn url_has_presigned_query(url: &str) -> bool {
    let before_frag = url.split('#').next().unwrap_or(url);
    let Some((_, query)) = before_frag.split_once('?') else {
        return false;
    };
    query.split('&').any(|pair| {
        let key = pair.split('=').next().unwrap_or("").trim();
        key.eq_ignore_ascii_case("sig")
            || key.eq_ignore_ascii_case("x-amz-signature")
            || key.eq_ignore_ascii_case("signature")
            || key.eq_ignore_ascii_case("x-goog-signature")
    })
}

/// Resolve the final URL and custom HTTP headers for opening a remote COG,
/// preventing accidental credential leaks onto public or cross-origin hosts and
/// avoiding dual-auth errors on already-presigned URLs.
pub(crate) fn resolve_cog_auth(
    url: &str,
    mode: StacCogAuthMode,
    session: &StacSession,
    settings: &AppSettings,
) -> (String, Vec<(String, String)>) {
    if mode == StacCogAuthMode::None || url_has_presigned_query(url) {
        return (url.to_owned(), Vec::new());
    }
    if mode == StacCogAuthMode::FromSearchResult && !session.result_uses_custom_auth() {
        return (url.to_owned(), Vec::new());
    }

    let auth = settings.stac.to_stac_auth();
    match &auth {
        StacAuth::None => (url.to_owned(), Vec::new()),
        // SAS query tokens are intentionally used across storage domains (e.g.
        // STAC catalog on one host, COGs on `*.blob.core.windows.net`).
        StacAuth::SasQuery(_) => (auth.apply_to_url(url), Vec::new()),
        StacAuth::Bearer(_) | StacAuth::ApiKey { .. } => {
            if mode == StacCogAuthMode::FromSearchResult
                && let Some(origin_ep) = session.result_catalog_endpoint()
                && !same_http_origin(origin_ep, url)
            {
                tracing::debug!("Skipping custom STAC auth headers for cross-origin asset URL");
                return (url.to_owned(), Vec::new());
            }
            (url.to_owned(), auth.extra_headers())
        }
    }
}

fn rect_polygon(west: f64, south: f64, east: f64, north: f64) -> Polygon {
    Polygon {
        exterior: Ring {
            points: vec![
                GeoPoint::new(south, west),
                GeoPoint::new(south, east),
                GeoPoint::new(north, east),
                GeoPoint::new(north, west),
                GeoPoint::new(south, west),
            ],
        },
        holes: Vec::new(),
    }
}

/// Construct a closed CCW WGS84 polygon (or antimeridian-split `MultiPolygon`
/// when RFC 7946 `west > east`) from `[west, south, east, north]`.
pub(crate) fn polygon_geometry_from_bbox(bbox: [f64; 4]) -> Option<GeoJsonGeometry> {
    if !bbox.iter().all(|v| v.is_finite()) {
        return None;
    }
    let west = bbox[0].clamp(-180.0, 180.0);
    let east = bbox[2].clamp(-180.0, 180.0);
    let south = bbox[1].min(bbox[3]).clamp(-89.9, 89.9);
    let north = bbox[1].max(bbox[3]).clamp(-89.9, 89.9);
    if (north - south) <= 1e-9 {
        return None;
    }

    if west <= east {
        if (east - west) <= 1e-9 {
            return None;
        }
        Some(GeoJsonGeometry::Polygon(rect_polygon(
            west, south, east, north,
        )))
    } else {
        // RFC 7946 antimeridian-crossing bbox: split into [west..=180] and [-180..=east],
        // skipping degenerate zero-width halves.
        let mut polys = Vec::with_capacity(2);
        if (180.0 - west) > 1e-9 {
            polys.push(rect_polygon(west, south, 180.0, north));
        }
        if (east - -180.0) > 1e-9 {
            polys.push(rect_polygon(-180.0, south, east, north));
        }
        match polys.len() {
            0 => None,
            1 => polys.into_iter().next().map(GeoJsonGeometry::Polygon),
            _ => Some(GeoJsonGeometry::MultiPolygon(polys)),
        }
    }
}

/// Compute `[min_lon, min_lat, max_lon, max_lat]` covering `geometry`.
pub(crate) fn geometry_bbox(geometry: &GeoJsonGeometry) -> Option<[f64; 4]> {
    let fc = GeoJsonFeatureCollection {
        features: vec![GeoJsonFeature::new(None, geometry.clone())],
    };
    fc.bbox()
}

/// Compute a camera [`FootprintBounds`] covering a `[west, south, east, north]`
/// RFC 7946 bbox (including `west > east` antimeridian crossings).
pub(crate) fn footprint_bounds_from_bbox(bbox: [f64; 4]) -> Option<FootprintBounds> {
    if !bbox.iter().all(|v| v.is_finite()) {
        return None;
    }
    let west = bbox[0].clamp(-180.0, 180.0);
    let east = bbox[2].clamp(-180.0, 180.0);
    let min_lat = bbox[1].min(bbox[3]).clamp(-90.0, 90.0);
    let max_lat = bbox[1].max(bbox[3]).clamp(-90.0, 90.0);
    let lat = 0.5 * (min_lat + max_lat);
    let (lon, lon_span) = if west <= east {
        (0.5 * (west + east), east - west)
    } else {
        let span = (east - west) + 360.0;
        let raw_mid = west + 0.5 * span;
        let wrapped = if raw_mid > 180.0 {
            raw_mid - 360.0
        } else {
            raw_mid
        };
        (wrapped, span)
    };
    let half_lat = 0.5 * (max_lat - min_lat).abs();
    let half_lon = 0.5 * lon_span.abs() * lat.to_radians().cos().abs().max(0.1);
    let radius_deg = half_lat.hypot(half_lon).max(0.01);
    Some(FootprintBounds {
        lat,
        lon,
        radius_deg,
    })
}

fn item_geometry(item: &StacItem) -> Option<GeoJsonGeometry> {
    if let Some(geom) = &item.geometry {
        Some(geom.clone())
    } else {
        item.bbox.and_then(polygon_geometry_from_bbox)
    }
}

fn build_vector_layer(
    id: DrawVectorId,
    rev: u64,
    geometries: Vec<GeoJsonGeometry>,
    fill_color: [f32; 4],
    line_color: [f32; 4],
    stroke_width_px: f32,
    hatch_spacing_px: f32,
) -> Option<DrawVectorLayer> {
    if geometries.is_empty() {
        return None;
    }
    let fc = GeoJsonFeatureCollection {
        features: geometries
            .into_iter()
            .map(|g| GeoJsonFeature::new(None, g))
            .collect(),
    };
    let mesh = VectorMesh::from_collection(&fc, DEFAULT_VECTOR_MAX_CHORD_DEG);
    if mesh.is_empty() {
        return None;
    }
    Some(DrawVectorLayer {
        id,
        rev,
        opacity: 1.0,
        fill_color,
        line_color,
        stroke_width_px,
        hatch_spacing_px,
        mesh: Arc::new(mesh),
    })
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
    aoi_geometry: Option<GeoJsonGeometry>,
    aoi_display_bbox: Option<[f64; 4]>,
    aoi_label: Option<String>,
    last_result: Option<StacSearchResult>,
    result_uses_custom_auth: bool,
    result_catalog_endpoint: Option<String>,
    last_error: Option<String>,
    selected_idx: Option<usize>,
    hovered_idx: Option<usize>,
    rev: u64,
    aoi_draw: Option<DrawVectorLayer>,
    aoi_preview_draw: Option<DrawVectorLayer>,
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
            aoi_geometry: None,
            aoi_display_bbox: None,
            aoi_label: None,
            last_result: None,
            result_uses_custom_auth: false,
            result_catalog_endpoint: None,
            last_error: None,
            selected_idx: None,
            hovered_idx: None,
            rev: 1,
            aoi_draw: None,
            aoi_preview_draw: None,
            footprints_draw: None,
            highlight_draw: None,
        }
    }

    fn bump_rev(&mut self) -> u64 {
        self.rev = self.rev.wrapping_add(1);
        self.rev
    }

    pub(crate) fn is_searching(&self) -> bool {
        self.searching
    }

    pub(crate) fn last_error(&self) -> Option<&str> {
        self.last_error.as_deref()
    }

    pub(crate) fn clear_error(&mut self) {
        self.last_error = None;
    }

    pub(crate) fn result_uses_custom_auth(&self) -> bool {
        self.result_uses_custom_auth
    }

    pub(crate) fn result_catalog_endpoint(&self) -> Option<&str> {
        self.result_catalog_endpoint.as_deref()
    }

    pub(crate) fn aoi_geometry(&self) -> Option<&GeoJsonGeometry> {
        self.aoi_geometry.as_ref()
    }

    pub(crate) fn aoi_label(&self) -> Option<&str> {
        self.aoi_label.as_deref()
    }

    pub(crate) fn aoi_bbox(&self) -> Option<[f64; 4]> {
        self.aoi_display_bbox
            .or_else(|| self.aoi_geometry.as_ref().and_then(geometry_bbox))
    }

    pub(crate) fn export_aoi_geojson(&self) -> Option<String> {
        self.aoi_geometry.as_ref().map(export_geojson_geometry)
    }

    pub(crate) fn export_aoi_wkt(&self) -> Option<String> {
        self.aoi_geometry.as_ref().map(export_wkt_geometry)
    }

    pub(crate) fn set_aoi_bbox(&mut self, bbox: [f64; 4], label: String) -> bool {
        let Some(geom) = polygon_geometry_from_bbox(bbox) else {
            return false;
        };
        self.aoi_preview_draw = None;
        self.aoi_display_bbox = Some(bbox);
        self.set_aoi_geometry_inner(geom, label);
        true
    }

    pub(crate) fn set_aoi_geometry(&mut self, geometry: GeoJsonGeometry, label: String) {
        self.aoi_preview_draw = None;
        self.aoi_display_bbox = None;
        self.set_aoi_geometry_inner(geometry, label);
    }

    fn set_aoi_geometry_inner(&mut self, geometry: GeoJsonGeometry, label: String) {
        self.aoi_geometry = Some(geometry.clone());
        self.aoi_label = Some(label);
        let rev = self.bump_rev();
        self.aoi_draw = build_vector_layer(
            DrawVectorId::StacAoi,
            rev,
            vec![geometry],
            STAC_AOI_FILL,
            STAC_AOI_LINE,
            3.0,
            10.0,
        );
    }

    /// Update the live drag preview rectangle without overwriting any existing
    /// committed AOI until the drag finishes.
    pub(crate) fn set_aoi_preview_bbox(&mut self, bbox: [f64; 4]) -> bool {
        let Some(geom) = polygon_geometry_from_bbox(bbox) else {
            return false;
        };
        let rev = self.bump_rev();
        self.aoi_preview_draw = build_vector_layer(
            DrawVectorId::StacAoi,
            rev,
            vec![geom],
            STAC_AOI_FILL,
            STAC_AOI_LINE,
            3.0,
            10.0,
        );
        true
    }

    /// Clear any active AOI drag preview, restoring the committed AOI overlay.
    pub(crate) fn clear_aoi_preview(&mut self) -> bool {
        if self.aoi_preview_draw.is_none() {
            return false;
        }
        self.aoi_preview_draw = None;
        true
    }

    pub(crate) fn clear_aoi(&mut self) -> bool {
        if self.aoi_geometry.is_none() && self.aoi_preview_draw.is_none() {
            return false;
        }
        self.aoi_geometry = None;
        self.aoi_display_bbox = None;
        self.aoi_label = None;
        self.aoi_draw = None;
        self.aoi_preview_draw = None;
        self.bump_rev();
        true
    }

    pub(crate) fn items(&self) -> &[StacItem] {
        self.last_result
            .as_ref()
            .map_or(&[], |res| res.items.as_slice())
    }

    pub(crate) fn matched(&self) -> Option<u64> {
        self.last_result.as_ref().and_then(|res| res.matched)
    }

    pub(crate) fn selected_idx(&self) -> Option<usize> {
        self.selected_idx
    }

    pub(crate) fn hovered_idx(&self) -> Option<usize> {
        self.hovered_idx
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
        if self.last_result.is_none() && self.footprints_draw.is_none() && self.last_error.is_none()
        {
            return false;
        }
        self.last_result = None;
        self.result_uses_custom_auth = false;
        self.result_catalog_endpoint = None;
        self.last_error = None;
        self.selected_idx = None;
        self.hovered_idx = None;
        self.footprints_draw = None;
        self.highlight_draw = None;
        self.bump_rev();
        true
    }

    /// Launch an asynchronous STAC `/search` query, cancelling any prior
    /// in-flight search immediately.
    pub(crate) fn begin_search(
        &mut self,
        catalog: StacCatalogConfig,
        uses_custom_auth: bool,
        query: StacSearchQuery,
    ) -> Result<(), String> {
        let Some(net) = self.net.clone() else {
            return Err("network subsystem is unavailable".to_owned());
        };
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

    /// Drain completed STAC search responses, rebuilding cached globe
    /// footprint meshes when results arrive. Returns `true` if scene overlays
    /// or UI state changed.
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
        if let Some(aoi) = self.aoi_preview_draw.as_ref().or(self.aoi_draw.as_ref()) {
            out.push(aoi.clone());
        }
    }
}

#[cfg(test)]
#[path = "stac_tests.rs"]
mod tests;
