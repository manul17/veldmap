//! Read-only GeoJSON parsing, RFC 7946 GeoJSON export (with right-hand
//! winding), and WGS84 globe vector mesh tessellation ([`VectorMesh`]).

mod cluster;
mod export;
mod mesh;
mod parse;
mod tessellate;
mod triangulate;

use serde_json::{Map, Value};
use thiserror::Error;

use super::footprint::{FootprintBounds, GeoPoint, Polygon, frame_coords};

pub use export::export_geojson_feature_collection;
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
}

fn collect_geometry_points(geom: &GeoJsonGeometry, out: &mut Vec<GeoPoint>) {
    match geom {
        GeoJsonGeometry::Point(p) => out.push(*p),
        GeoJsonGeometry::MultiPoint(pts) | GeoJsonGeometry::LineString(pts) => {
            out.extend_from_slice(pts);
        }
        GeoJsonGeometry::MultiLineString(lines) => {
            for line in lines {
                out.extend_from_slice(line);
            }
        }
        GeoJsonGeometry::Polygon(poly) => out.extend_from_slice(&poly.exterior.points),
        GeoJsonGeometry::MultiPolygon(polys) => {
            for poly in polys {
                out.extend_from_slice(&poly.exterior.points);
            }
        }
    }
}

#[cfg(test)]
#[path = "geojson_tests.rs"]
mod tests;
