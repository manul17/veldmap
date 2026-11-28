//! GeoJSON domain data structures, safety limits, and bounding-box helpers.

use serde_json::{Map, Value};
use thiserror::Error;

use super::super::footprint::{FootprintBounds, frame_visit};
use super::super::shapes::{GeoPoint, Polygon};

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

impl GeoJsonGeometry {
    /// Axis-aligned WGS84 bounding box `[min_lon, min_lat, max_lon, max_lat]`
    /// across all finite vertices in this geometry.
    pub fn bbox(&self) -> Option<[f64; 4]> {
        let mut acc = BboxAccumulator::new();
        acc.extend_geometry(self);
        acc.finish()
    }
}

/// A single parsed GeoJSON feature.
#[derive(Clone, Debug, PartialEq)]
pub struct GeoJsonFeature {
    pub name: Option<String>,
    pub(super) properties: Map<String, Value>,
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
        frame_visit(|emit| {
            for feature in &self.features {
                for_each_geometry_point(&feature.geometry, |p| emit(p.lat, p.lon));
            }
        })
    }

    /// Axis-aligned WGS84 bounding box `[min_lon, min_lat, max_lon, max_lat]`
    /// across all finite vertices in the collection.
    #[cfg(test)]
    pub(crate) fn bbox(&self) -> Option<[f64; 4]> {
        let mut acc = BboxAccumulator::new();
        for feature in &self.features {
            acc.extend_geometry(&feature.geometry);
        }
        acc.finish()
    }
}

struct BboxAccumulator {
    min_lon: f64,
    min_lat: f64,
    max_lon: f64,
    max_lat: f64,
}

impl BboxAccumulator {
    const fn new() -> Self {
        Self {
            min_lon: f64::INFINITY,
            min_lat: f64::INFINITY,
            max_lon: f64::NEG_INFINITY,
            max_lat: f64::NEG_INFINITY,
        }
    }

    fn extend_geometry(&mut self, geom: &GeoJsonGeometry) {
        for_each_geometry_point(geom, |p| {
            if !p.is_finite() {
                return;
            }
            self.min_lon = self.min_lon.min(p.lon);
            self.min_lat = self.min_lat.min(p.lat);
            self.max_lon = self.max_lon.max(p.lon);
            self.max_lat = self.max_lat.max(p.lat);
        });
    }

    fn finish(self) -> Option<[f64; 4]> {
        (self.min_lon.is_finite() && self.min_lat.is_finite()).then_some([
            self.min_lon,
            self.min_lat,
            self.max_lon,
            self.max_lat,
        ])
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

pub(super) fn extract_feature_name(props: &Map<String, Value>) -> Option<String> {
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
