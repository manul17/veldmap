//! WGS84 geodesy, map projections, CRS/WKT catalog, Karney geodesics,
//! georeference bindings, MGRS indexing, nadir globe camera, frustum culling,
//! chord sagitta helpers, vector shapes, and footprint framing.

mod binding;
mod camera;
mod coords;
mod crs;
mod cull;
mod footprint;
mod geodesic;
mod geodesy;
mod geojson;
mod mgrs;
pub(crate) mod sag;
mod shapes;

pub use binding::{Frame, Tie, TieGrid};
pub use camera::{Camera, MIN_HEIGHT_M, ZOOM_PER_CLICK};
pub use coords::{LocationKind, ParseLocationError, ParsedLocation, parse_location};
pub use crs::{
    CrsError, CrsPreset, CrsSpec, CrsTextKind, CustomTmParams, Datum, DatumPreset, DatumWarning,
    HelmertConvention, KRASOVSKY, LinearUnit, PULKOVO_1942, Proj4Projection, Projection, System,
    US_SURVEY_FOOT_M, WEB_MERCATOR_LIMIT_DEG, WGS84_DATUM, builtin_crs_presets,
    ellipsoid_from_epsg, known_datum_from_code, search_crs_presets, web_mercator_lat_deg,
};
pub use cull::{Ball, CullFrustum};
pub use footprint::FootprintBounds;
pub use geodesic::{GeodesicPolygonMeasurement, polyline_length_m, ring_perimeter_and_area};
pub(crate) use geodesy::{EQUATORIAL_CIRCUMFERENCE_M, unwind};
pub use geodesy::{Ellipsoid, Geodetic, SEMI_MAJOR_M, WGS84};
pub use geojson::{
    DEFAULT_VECTOR_MAX_CHORD_DEG, GeoJsonError, GeoJsonFeature, GeoJsonFeatureCollection,
    GeoJsonGeometry, MAX_GEOJSON_BYTES, VectorCluster, VectorMesh, VectorVertex,
    export_geojson_feature_collection, export_geojson_geometry, export_wkt_geometry, parse_geojson,
};
pub use shapes::{GeoPoint, Polygon, Ring};

/// Local WGS84 surface distance approximation in metres for unit test error assertions.
#[cfg(test)]
pub(super) fn ground_m(first: (f64, f64), second: (f64, f64)) -> f64 {
    let lat = ((first.0 + second.0) / 2.0).to_radians();
    let per_lat = 111_132.92 - 559.82 * (2.0 * lat).cos();
    let per_lon = 111_412.84 * lat.cos() - 93.5 * (3.0 * lat).cos();
    ((first.0 - second.0) * per_lat).hypot((first.1 - second.1) * per_lon)
}
