//! WGS84 geodesy, map projections, CRS/WKT catalog, Karney geodesics,
//! georeference bindings, MGRS indexing, nadir globe camera, frustum culling,
//! chord sagitta helpers, and footprint hit-testing.

mod binding;
pub mod camera;
mod coords;
mod crs;
mod cull;
mod datum;
pub mod footprint;
mod geodesic;
pub mod geodesy;
mod geojson;
pub mod mgrs;
mod proj4;
pub mod projection;
pub(crate) mod sag;

pub use binding::{BindingError, Frame, Tie, TieGrid};
pub use camera::Camera;
pub use coords::{LocationKind, ParseLocationError, ParsedLocation, parse_location};
pub use crs::{
    CrsPreset, CrsSpec, CrsTextKind, CustomTmParams, DatumPreset, builtin_crs_presets,
    search_crs_presets,
};
pub use cull::{Ball, CullFrustum};
pub use datum::{
    AIRY_1830, Datum, DatumWarning, HelmertConvention, INTERNATIONAL_FOOT_M, KRASOVSKY, LinearUnit,
    PULKOVO_1942, US_SURVEY_FOOT_M, WGS84_DATUM, ellipsoid_from_epsg, known_datum_from_code,
};
pub use footprint::{FootprintBounds, GeoPoint, Polygon, Ring};
pub use geodesic::{
    GeodesicInverse, GeodesicPolygonMeasurement, geodesic_direct, geodesic_distance_m,
    geodesic_inverse, polygon_with_holes_measurement, polyline_length_m, ring_perimeter_and_area,
    ring_signed_area_m2, sample_geodesic_arc,
};
pub use geodesy::{Ellipsoid, Geodetic, WGS84};
pub use geojson::{
    DEFAULT_VECTOR_MAX_CHORD_DEG, GeoJsonError, GeoJsonFeature, GeoJsonFeatureCollection,
    GeoJsonGeometry, MAX_GEOJSON_BYTES, VectorCluster, VectorMesh, VectorVertex,
    export_geojson_feature_collection, parse_geojson,
};
pub use mgrs::{MgrsError, MgrsGranuleFrame};
pub use proj4::{CrsError, Proj4Projection};
pub use projection::{Projection, System};
