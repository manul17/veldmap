//! STAC catalog search controller (`StacSession`): asynchronous catalog
//! querying via [`veldmap_net::StacClient`], canonical AOI geometry ownership
//! (drawn box, viewport, or imported GeoJSON polygon), GeoJSON/WKT export, and
//! cached Pass-2 globe scene outline [`DrawVectorLayer`] generation.
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
    NetContext, StacCatalogConfig, StacClient, StacItem, StacSearchQuery, StacSearchResult,
};

use crate::render::{DrawVectorId, DrawVectorLayer};

/// Default visual colors (sRGB RGB + linear alpha) for STAC overlays on globe.
const STAC_FOOTPRINT_FILL: [f32; 4] = [0.22, 0.62, 0.96, 0.08];
const STAC_FOOTPRINT_LINE: [f32; 4] = [0.35, 0.78, 1.0, 0.85];
const STAC_HIGHLIGHT_FILL: [f32; 4] = [1.0, 0.72, 0.14, 0.24];
const STAC_HIGHLIGHT_LINE: [f32; 4] = [1.0, 0.86, 0.24, 0.98];
const STAC_AOI_FILL: [f32; 4] = [0.16, 0.88, 0.56, 0.14];
const STAC_AOI_LINE: [f32; 4] = [0.22, 0.96, 0.66, 0.96];

/// Construct a closed CCW WGS84 polygon from `[min_lon, min_lat, max_lon, max_lat]`.
pub(crate) fn polygon_geometry_from_bbox(bbox: [f64; 4]) -> GeoJsonGeometry {
    let min_lon = bbox[0].min(bbox[2]).clamp(-180.0, 180.0);
    let max_lon = bbox[0].max(bbox[2]).clamp(-180.0, 180.0);
    let min_lat = bbox[1].min(bbox[3]).clamp(-89.9, 89.9);
    let max_lat = bbox[1].max(bbox[3]).clamp(-89.9, 89.9);
    let ring = Ring {
        points: vec![
            GeoPoint::new(min_lat, min_lon),
            GeoPoint::new(min_lat, max_lon),
            GeoPoint::new(max_lat, max_lon),
            GeoPoint::new(max_lat, min_lon),
            GeoPoint::new(min_lat, min_lon),
        ],
    };
    GeoJsonGeometry::Polygon(Polygon {
        exterior: ring,
        holes: Vec::new(),
    })
}

/// Compute `[min_lon, min_lat, max_lon, max_lat]` covering `geometry`.
pub(crate) fn geometry_bbox(geometry: &GeoJsonGeometry) -> Option<[f64; 4]> {
    let fc = GeoJsonFeatureCollection {
        features: vec![GeoJsonFeature::new(None, geometry.clone())],
    };
    fc.bbox()
}

/// Compute a camera [`FootprintBounds`] covering a `[min_lon, min_lat, max_lon, max_lat]` bbox.
pub(crate) fn footprint_bounds_from_bbox(bbox: [f64; 4]) -> Option<FootprintBounds> {
    if !bbox.iter().all(|v| v.is_finite()) {
        return None;
    }
    let min_lon = bbox[0].min(bbox[2]);
    let max_lon = bbox[0].max(bbox[2]);
    let min_lat = bbox[1].min(bbox[3]);
    let max_lat = bbox[1].max(bbox[3]);
    let lat = 0.5 * (min_lat + max_lat);
    let lon = 0.5 * (min_lon + max_lon);
    let half_lat = 0.5 * (max_lat - min_lat).abs();
    let half_lon = 0.5 * (max_lon - min_lon).abs() * lat.to_radians().cos().abs().max(0.1);
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
        item.bbox.map(polygon_geometry_from_bbox)
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
    aoi_label: Option<String>,
    last_result: Option<StacSearchResult>,
    last_error: Option<String>,
    selected_idx: Option<usize>,
    hovered_idx: Option<usize>,
    rev: u64,
    aoi_draw: Option<DrawVectorLayer>,
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
            aoi_label: None,
            last_result: None,
            last_error: None,
            selected_idx: None,
            hovered_idx: None,
            rev: 1,
            aoi_draw: None,
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

    pub(crate) fn aoi_geometry(&self) -> Option<&GeoJsonGeometry> {
        self.aoi_geometry.as_ref()
    }

    pub(crate) fn aoi_label(&self) -> Option<&str> {
        self.aoi_label.as_deref()
    }

    pub(crate) fn aoi_bbox(&self) -> Option<[f64; 4]> {
        self.aoi_geometry.as_ref().and_then(geometry_bbox)
    }

    pub(crate) fn export_aoi_geojson(&self) -> Option<String> {
        self.aoi_geometry.as_ref().map(export_geojson_geometry)
    }

    pub(crate) fn export_aoi_wkt(&self) -> Option<String> {
        self.aoi_geometry.as_ref().map(export_wkt_geometry)
    }

    pub(crate) fn set_aoi_bbox(&mut self, bbox: [f64; 4], label: String) {
        let geom = polygon_geometry_from_bbox(bbox);
        self.set_aoi_geometry(geom, label);
    }

    pub(crate) fn set_aoi_geometry(&mut self, geometry: GeoJsonGeometry, label: String) {
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

    pub(crate) fn clear_aoi(&mut self) -> bool {
        if self.aoi_geometry.is_none() {
            return false;
        }
        self.aoi_geometry = None;
        self.aoi_label = None;
        self.aoi_draw = None;
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
        if self.last_result.is_none() && self.footprints_draw.is_none() {
            return false;
        }
        self.last_result = None;
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
                let _ = tx.send(SearchDoneMsg { seq, outcome });
                (wakeup)();
            }
        });
        Ok(())
    }

    /// Cancel any in-flight STAC search.
    pub(crate) fn cancel(&mut self) {
        if let Some(tok) = self.active_cancel.take() {
            tok.cancel();
        }
        self.searching = false;
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

    /// Append cached STAC scene footprint, highlight, and AOI overlays into
    /// `out` in painter's order.
    pub(crate) fn push_draw_layers(&self, out: &mut Vec<DrawVectorLayer>) {
        if let Some(fp) = &self.footprints_draw {
            out.push(fp.clone());
        }
        if let Some(hi) = &self.highlight_draw {
            out.push(hi.clone());
        }
        if let Some(aoi) = &self.aoi_draw {
            out.push(aoi.clone());
        }
    }
}
