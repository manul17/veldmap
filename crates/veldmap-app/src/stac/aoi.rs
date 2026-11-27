//! Canonical STAC Area of Interest (AOI) state, RFC 7946 bbox/geometry helpers,
//! GeoJSON/WKT export, and Pass-2 globe overlay mesh construction.

use std::sync::Arc;

use veldmap_core::geo::{
    DEFAULT_VECTOR_MAX_CHORD_DEG, FootprintBounds, GeoJsonFeature, GeoJsonFeatureCollection,
    GeoJsonGeometry, GeoPoint, Polygon, Ring, VectorMesh, export_geojson_geometry,
    export_wkt_geometry,
};

use crate::render::{DrawVectorId, DrawVectorLayer};

const STAC_AOI_FILL: [f32; 4] = [0.16, 0.88, 0.56, 0.14];
const STAC_AOI_LINE: [f32; 4] = [0.22, 0.96, 0.66, 0.96];

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
pub(super) fn polygon_geometry_from_bbox(bbox: [f64; 4]) -> Option<GeoJsonGeometry> {
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

/// Compute a camera [`FootprintBounds`] covering a `[west, south, east, north]`
/// RFC 7946 bbox (including `west > east` antimeridian crossings).
pub(super) fn footprint_bounds_from_bbox(bbox: [f64; 4]) -> Option<FootprintBounds> {
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

/// Tessellate `geometries` into a Pass-2 globe [`DrawVectorLayer`].
pub(super) fn build_vector_layer(
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

/// Canonical Area of Interest state owned by [`super::StacSession`].
#[derive(Default)]
pub(super) struct AoiState {
    geometry: Option<GeoJsonGeometry>,
    display_bbox: Option<[f64; 4]>,
    label: Option<String>,
    draw: Option<DrawVectorLayer>,
    preview_draw: Option<DrawVectorLayer>,
}

impl AoiState {
    pub(super) fn has_geometry(&self) -> bool {
        self.geometry.is_some()
    }

    pub(super) fn geometry(&self) -> Option<&GeoJsonGeometry> {
        self.geometry.as_ref()
    }

    pub(super) fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    /// Return the precomputed display bbox `[west, south, east, north]` in O(1)
    /// without cloning or scanning vertices on UI frames.
    pub(super) const fn bbox(&self) -> Option<[f64; 4]> {
        self.display_bbox
    }

    pub(super) fn export_geojson(&self) -> Option<String> {
        self.geometry.as_ref().map(export_geojson_geometry)
    }

    pub(super) fn export_wkt(&self) -> Option<String> {
        self.geometry.as_ref().map(export_wkt_geometry)
    }

    pub(super) fn set_bbox(&mut self, bbox: [f64; 4], label: String, rev: u64) -> bool {
        let Some(geom) = polygon_geometry_from_bbox(bbox) else {
            return false;
        };
        self.preview_draw = None;
        self.display_bbox = Some(bbox);
        self.set_geometry_inner(geom, label, rev);
        true
    }

    pub(super) fn set_geometry(&mut self, geometry: GeoJsonGeometry, label: String, rev: u64) {
        self.preview_draw = None;
        self.display_bbox = geometry.bbox();
        self.set_geometry_inner(geometry, label, rev);
    }

    fn set_geometry_inner(&mut self, geometry: GeoJsonGeometry, label: String, rev: u64) {
        self.geometry = Some(geometry.clone());
        self.label = Some(label);
        self.draw = build_vector_layer(
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
    pub(super) fn set_preview_bbox(&mut self, bbox: [f64; 4], rev: u64) -> bool {
        let Some(geom) = polygon_geometry_from_bbox(bbox) else {
            return false;
        };
        self.preview_draw = build_vector_layer(
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
    pub(super) fn clear_preview(&mut self) -> bool {
        if self.preview_draw.is_none() {
            return false;
        }
        self.preview_draw = None;
        true
    }

    pub(super) fn clear(&mut self) -> bool {
        if self.geometry.is_none() && self.preview_draw.is_none() {
            return false;
        }
        self.geometry = None;
        self.display_bbox = None;
        self.label = None;
        self.draw = None;
        self.preview_draw = None;
        true
    }

    pub(super) fn draw_layer(&self) -> Option<&DrawVectorLayer> {
        self.preview_draw.as_ref().or(self.draw.as_ref())
    }
}
