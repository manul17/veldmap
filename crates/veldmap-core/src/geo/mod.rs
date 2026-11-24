//! WGS84 geodesy, map projections, CRS/WKT catalog, Karney geodesics,
//! georeference bindings, MGRS indexing, nadir globe camera, frustum culling,
//! chord sagitta helpers, and footprint hit-testing.

mod binding;
pub mod camera;
mod coords;
pub mod crs;
mod cull;
pub mod footprint;
pub mod geodesic;
pub mod geodesy;
pub mod geojson;
pub mod mgrs;
pub mod projection;
pub(crate) mod sag;

pub use binding::{BindingError, Frame, Tie, TieGrid};
pub use camera::Camera;
pub use coords::{LocationKind, ParseLocationError, ParsedLocation, parse_location};
pub use crs::{
    CrsError, CrsPreset, CrsSpec, CustomTmParams, DatumPreset, DatumWarning, LinearUnit,
    Proj4Projection, builtin_crs_presets, ellipsoid_from_epsg, known_datum_from_code,
    search_crs_presets,
};
pub use cull::{Ball, CullFrustum};
pub use footprint::{FootprintBounds, GeoPoint, Polygon, Ring};
pub use geodesic::{
    GeodesicInverse, GeodesicPolygonMeasurement, geodesic_direct, geodesic_distance_m,
    geodesic_inverse, polygon_with_holes_measurement, polyline_length_m, ring_perimeter_and_area,
    ring_signed_area_m2, sample_geodesic_arc,
};
pub use geodesy::{Ellipsoid, Geodetic, WGS84};
pub use geojson::{
    DEFAULT_VECTOR_MAX_CHORD_DEG, GeoJsonError, GeoJsonFeature, GeoJsonFeatureCollection,
    GeoJsonGeometry, VectorCluster, VectorMesh, VectorVertex, export_geojson_feature_collection,
    parse_geojson,
};
pub use mgrs::{MgrsError, MgrsGranuleFrame};
pub use projection::{
    Datum, HelmertConvention, INTERNATIONAL_FOOT_M, KRASOVSKY, PULKOVO_1942, Projection, System,
    US_SURVEY_FOOT_M, WGS84_DATUM,
};
