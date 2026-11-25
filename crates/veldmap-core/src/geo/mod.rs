//! WGS84 geodesy, map projections, CRS/WKT catalog, Karney geodesics,
//! georeference bindings, MGRS indexing, nadir globe camera, frustum culling,
//! chord sagitta helpers, and footprint hit-testing.

mod binding;
mod camera;
mod coords;
mod crs;
mod cull;
mod datum;
mod footprint;
mod geodesic;
mod geodesy;
mod geojson;
mod mgrs;
mod proj4;
mod projection;
pub(crate) mod sag;

pub use binding::{Frame, Tie, TieGrid};
pub use camera::{Camera, MIN_HEIGHT_M, ZOOM_PER_CLICK};
pub use coords::{LocationKind, ParseLocationError, ParsedLocation, parse_location};
pub use crs::{
    CrsPreset, CrsSpec, CrsTextKind, CustomTmParams, DatumPreset, builtin_crs_presets,
    search_crs_presets,
};
pub use cull::{Ball, CullFrustum};
pub use datum::{
    Datum, DatumWarning, HelmertConvention, KRASOVSKY, LinearUnit, PULKOVO_1942, US_SURVEY_FOOT_M,
    WGS84_DATUM, ellipsoid_from_epsg, known_datum_from_code,
};
pub use footprint::{FootprintBounds, GeoPoint, Polygon, Ring, covers_polygon, covers_polygons};
pub use geodesic::{GeodesicPolygonMeasurement, polyline_length_m, ring_perimeter_and_area};
pub(crate) use geodesy::{EQUATORIAL_CIRCUMFERENCE_M, unwind};
pub use geodesy::{Ellipsoid, Geodetic, SEMI_MAJOR_M, WGS84};
pub use geojson::{
    DEFAULT_VECTOR_MAX_CHORD_DEG, GeoJsonError, GeoJsonFeature, GeoJsonFeatureCollection,
    GeoJsonGeometry, MAX_GEOJSON_BYTES, VectorCluster, VectorMesh, VectorVertex,
    export_geojson_feature_collection, parse_geojson,
};
pub use proj4::{CrsError, Proj4Projection};
pub use projection::{Projection, System, WEB_MERCATOR_LIMIT_DEG, web_mercator_lat_deg};

/// Local WGS84 surface distance approximation in metres for unit test error assertions.
#[cfg(test)]
pub(super) fn ground_m(first: (f64, f64), second: (f64, f64)) -> f64 {
    let lat = ((first.0 + second.0) / 2.0).to_radians();
    let per_lat = 111_132.92 - 559.82 * (2.0 * lat).cos();
    let per_lon = 111_412.84 * lat.cos() - 93.5 * (3.0 * lat).cos();
    ((first.0 - second.0) * per_lat).hypot((first.1 - second.1) * per_lon)
}
