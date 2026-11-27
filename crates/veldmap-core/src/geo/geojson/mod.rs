//! Read-only GeoJSON parsing, RFC 7946 GeoJSON export (with right-hand
//! winding), OGC WKT geometry export, and WGS84 globe vector mesh tessellation
//! ([`VectorMesh`]).

mod cluster;
mod export;
mod mesh;
mod model;
mod parse;
mod tessellate;
mod triangulate;

pub use export::{
    export_geojson_feature_collection, export_geojson_geometry, export_wkt_geometry,
    geojson_geometry_to_value,
};
pub use mesh::{DEFAULT_VECTOR_MAX_CHORD_DEG, VectorCluster, VectorMesh, VectorVertex};
pub use model::{
    GeoJsonError, GeoJsonFeature, GeoJsonFeatureCollection, GeoJsonGeometry, MAX_GEOJSON_BYTES,
};
pub use parse::{parse_geojson, parse_geojson_geometry_value};

#[cfg(test)]
#[path = "geojson_tests.rs"]
mod tests;
