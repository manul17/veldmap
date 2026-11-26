//! Read-only GeoJSON parsing, RFC 7946 GeoJSON export (with right-hand
//! winding), OGC WKT geometry export, and WGS84 globe vector mesh tessellation
//! ([`VectorMesh`]).

mod cluster;
mod export;
mod mesh;
mod parse;
mod tessellate;
mod triangulate;

use serde_json::{Map, Value};
use thiserror::Error;

use super::footprint::{FootprintBounds, frame_coords};
use super::shapes::{GeoPoint, Polygon};

pub use export::{export_geojson_feature_collection, export_geojson_geometry, export_wkt_geometry};
pub use mesh::{DEFAULT_VECTOR_MAX_CHORD_DEG, VectorCluster, VectorMesh, VectorVertex};
pub use parse::parse_geojson;

/// Hard safety cap on input GeoJSON byte length (`64 MiB`) enforced before
/// JSON DOM allocation.
pub const MAX_GEOJSON_BYTES: usize = 64 * 1024 * 1024;

/// Errors produced when parsing GeoJSON documents.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum GeoJsonError {
    #[error("invalid JSON syntax: {0}")]
    Json(String),
    #[error("invalid GeoJSON structure: {0}")]
    Structure(String),
    #[error("GeoJSON exceeds safety limit: {0}")]
    LimitExceeded(String),
}

/// Supported GeoJSON geometries (WGS84 `[lon, lat]`).
#[derive(Clone, Debug, PartialEq)]
pub enum GeoJsonGeometry {
    Point(GeoPoint),
    MultiPoint(Vec<GeoPoint>),
    LineString(Vec<GeoPoint>),
    MultiLineString(Vec<Vec<GeoPoint>>),
    Polygon(Polygon),
    MultiPolygon(Vec<Polygon>),
}

/// A single parsed GeoJSON feature.
#[derive(Clone, Debug, PartialEq)]
pub struct GeoJsonFeature {
    pub name: Option<String>,
    properties: Map<String, Value>,
    pub geometry: GeoJsonGeometry,
}

impl GeoJsonFeature {
    /// Construct a feature with an optional name, empty properties, and `geometry`.
    pub fn new(name: Option<String>, geometry: GeoJsonGeometry) -> Self {
        Self {
            name,
            properties: Map::new(),
            geometry,
        }
    }

    /// Insert a string property on this feature (chainable).
    pub fn with_string_prop(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.properties
            .insert(key.into(), Value::String(value.into()));
        self
    }

    /// Insert a numeric (`f64`) property on this feature (chainable; non-finite
    /// values serialize as JSON `null`).
    pub fn with_f64_prop(mut self, key: impl Into<String>, value: f64) -> Self {
        self.properties.insert(key.into(), Value::from(value));
        self
    }

    /// Number of stored JSON properties on this feature.
    pub fn property_count(&self) -> usize {
        self.properties.len()
    }

    /// Read-only access to the raw JSON property map.
    pub fn properties(&self) -> &Map<String, Value> {
        &self.properties
    }

    /// Look up a string property by key.
    pub fn property_str(&self, key: &str) -> Option<&str> {
        self.properties.get(key).and_then(Value::as_str)
    }

    /// Look up a numeric property by key as `f64`.
    pub fn property_f64(&self, key: &str) -> Option<f64> {
        self.properties.get(key).and_then(Value::as_f64)
    }

    /// Render stored properties as `(key, display_value)` pairs for UI panels
    /// without requiring downstream crates to depend on `serde_json`.
    pub fn property_display_pairs(&self) -> Vec<(String, String)> {
        self.properties
            .iter()
            .map(|(k, v)| {
                let text = match v {
                    Value::String(s) => s.clone(),
                    Value::Null => "null".to_string(),
                    other => other.to_string(),
                };
                (k.clone(), text)
            })
            .collect()
    }
}

/// Parsed collection of GeoJSON features.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GeoJsonFeatureCollection {
    pub features: Vec<GeoJsonFeature>,
}

impl GeoJsonFeatureCollection {
    /// Spherical framing circle enclosing all geometries in the collection.
    pub fn bounds(&self) -> Option<FootprintBounds> {
        let mut pts = Vec::new();
        for feature in &self.features {
            collect_geometry_points(&feature.geometry, &mut pts);
        }
        frame_coords(pts.iter().map(|p| (p.lat, p.lon)))
    }

    /// Axis-aligned WGS84 bounding box `[min_lon, min_lat, max_lon, max_lat]`
    /// across all finite vertices in the collection.
    pub fn bbox(&self) -> Option<[f64; 4]> {
        let mut min_lon = f64::INFINITY;
        let mut min_lat = f64::INFINITY;
        let mut max_lon = f64::NEG_INFINITY;
        let mut max_lat = f64::NEG_INFINITY;
        for feature in &self.features {
            for_each_geometry_point(&feature.geometry, |p| {
                if !p.lat.is_finite() || !p.lon.is_finite() {
                    return;
                }
                min_lon = min_lon.min(p.lon);
                min_lat = min_lat.min(p.lat);
                max_lon = max_lon.max(p.lon);
                max_lat = max_lat.max(p.lat);
            });
        }
        (min_lon.is_finite() && min_lat.is_finite()).then_some([min_lon, min_lat, max_lon, max_lat])
    }
}

fn for_each_geometry_point(geom: &GeoJsonGeometry, mut visit: impl FnMut(GeoPoint)) {
    match geom {
        GeoJsonGeometry::Point(p) => visit(*p),
        GeoJsonGeometry::MultiPoint(pts) | GeoJsonGeometry::LineString(pts) => {
            for &p in pts {
                visit(p);
            }
        }
        GeoJsonGeometry::MultiLineString(lines) => {
            for line in lines {
                for &p in line {
                    visit(p);
                }
            }
        }
        GeoJsonGeometry::Polygon(poly) => {
            for &p in &poly.exterior.points {
                visit(p);
            }
        }
        GeoJsonGeometry::MultiPolygon(polys) => {
            for poly in polys {
                for &p in &poly.exterior.points {
                    visit(p);
                }
            }
        }
    }
}

fn collect_geometry_points(geom: &GeoJsonGeometry, out: &mut Vec<GeoPoint>) {
    for_each_geometry_point(geom, |p| out.push(p));
}

fn extract_feature_name(props: &Map<String, Value>) -> Option<String> {
    for key in ["name", "NAME", "Name", "title", "id", "label"] {
        if let Some(s) = props.get(key).and_then(Value::as_str) {
            let trimmed = s.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }
    None
}

#[cfg(test)]
#[path = "geojson_tests.rs"]
mod tests;
