//! Read-only GeoJSON parsing, RFC 7946 GeoJSON export (with right-hand
//! winding), and WGS84 globe vector mesh tessellation ([`VectorMesh`]).

mod tessellate;

use serde_json::{Map, Value, json};
use thiserror::Error;

use super::footprint::{FootprintBounds, GeoPoint, Polygon, Ring, dedup_geo_ring, frame_coords};
use super::geodesic::ring_signed_area_m2;

pub use tessellate::{DEFAULT_VECTOR_MAX_CHORD_DEG, VectorCluster, VectorMesh, VectorVertex};

/// Hard safety cap on input GeoJSON byte length (`64 MiB`) enforced before
/// JSON DOM allocation.
pub const MAX_GEOJSON_BYTES: usize = 64 * 1024 * 1024;

/// Hard safety cap on features per GeoJSON document to prevent OOM on
/// malformed or pathological inputs.
pub const MAX_GEOJSON_FEATURES: usize = 100_000;

/// Hard safety cap on parsed vertices per GeoJSON document.
pub const MAX_GEOJSON_VERTICES: usize = 2_000_000;

/// Maximum nesting depth for `GeometryCollection` containers.
const MAX_GEOMETRY_COLLECTION_DEPTH: usize = 8;

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
    pub properties: Map<String, Value>,
    pub geometry: GeoJsonGeometry,
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

/// Parse a GeoJSON string (`FeatureCollection`, `Feature`, or bare `Geometry`
/// including `GeometryCollection`) into a [`GeoJsonFeatureCollection`].
pub fn parse_geojson(input: &str) -> Result<GeoJsonFeatureCollection, GeoJsonError> {
    if input.len() > MAX_GEOJSON_BYTES {
        return Err(GeoJsonError::LimitExceeded(format!(
            "GeoJSON input exceeds {} MiB limit",
            MAX_GEOJSON_BYTES / (1024 * 1024)
        )));
    }
    let trimmed = input.trim_start_matches('\u{FEFF}');
    let root: Value =
        serde_json::from_str(trimmed).map_err(|e| GeoJsonError::Json(e.to_string()))?;
    let mut features = Vec::new();
    let mut vertex_count = 0usize;
    parse_root_value(&root, &mut features, &mut vertex_count)?;
    Ok(GeoJsonFeatureCollection { features })
}

fn parse_root_value(
    val: &Value,
    out: &mut Vec<GeoJsonFeature>,
    vertex_count: &mut usize,
) -> Result<(), GeoJsonError> {
    let (obj, kind) = object_and_type(val, "GeoJSON root")?;
    match kind {
        "FeatureCollection" => {
            let list = obj
                .get("features")
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    GeoJsonError::Structure("FeatureCollection missing 'features' array".into())
                })?;
            for item in list {
                parse_feature_object(item, out, vertex_count)?;
            }
            Ok(())
        }
        "Feature" => parse_feature_object(val, out, vertex_count),
        _ => parse_geometry_value(val, None, 0, out, vertex_count),
    }
}

fn parse_feature_object(
    val: &Value,
    out: &mut Vec<GeoJsonFeature>,
    vertex_count: &mut usize,
) -> Result<(), GeoJsonError> {
    let (obj, kind) = object_and_type(val, "Feature")?;
    if kind != "Feature" {
        return Err(GeoJsonError::Structure(format!(
            "expected 'Feature' inside FeatureCollection, found '{kind}'"
        )));
    }
    let props = obj
        .get("properties")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let Some(geom) = obj.get("geometry") else {
        return Ok(());
    };
    if geom.is_null() {
        return Ok(());
    }
    parse_geometry_value(geom, Some(&props), 0, out, vertex_count)
}

fn parse_geometry_value(
    val: &Value,
    inherited_props: Option<&Map<String, Value>>,
    depth: usize,
    out: &mut Vec<GeoJsonFeature>,
    vertex_count: &mut usize,
) -> Result<(), GeoJsonError> {
    let (obj, kind) = object_and_type(val, "Geometry")?;
    if kind == "GeometryCollection" {
        if depth >= MAX_GEOMETRY_COLLECTION_DEPTH {
            return Err(GeoJsonError::LimitExceeded(format!(
                "GeometryCollection nesting exceeds {MAX_GEOMETRY_COLLECTION_DEPTH}"
            )));
        }
        let geoms = obj
            .get("geometries")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                GeoJsonError::Structure("GeometryCollection missing 'geometries' array".into())
            })?;
        for g in geoms {
            parse_geometry_value(g, inherited_props, depth + 1, out, vertex_count)?;
        }
        return Ok(());
    }
    if out.len() >= MAX_GEOJSON_FEATURES {
        return Err(GeoJsonError::LimitExceeded(format!(
            "feature count exceeds {MAX_GEOJSON_FEATURES}"
        )));
    }
    let geometry = parse_geometry(kind, obj, vertex_count)?;
    let properties = inherited_props.cloned().unwrap_or_default();
    let name = extract_feature_name(&properties);
    out.push(GeoJsonFeature {
        name,
        properties,
        geometry,
    });
    Ok(())
}

fn object_and_type<'a>(
    val: &'a Value,
    context: &str,
) -> Result<(&'a Map<String, Value>, &'a str), GeoJsonError> {
    let obj = val
        .as_object()
        .ok_or_else(|| GeoJsonError::Structure(format!("{context} must be a JSON object")))?;
    let kind = obj
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| GeoJsonError::Structure(format!("{context} missing string 'type' field")))?;
    Ok((obj, kind))
}

fn parse_geometry(
    kind: &str,
    obj: &Map<String, Value>,
    vertex_count: &mut usize,
) -> Result<GeoJsonGeometry, GeoJsonError> {
    let coords = obj.get("coordinates").ok_or_else(|| {
        GeoJsonError::Structure(format!("geometry '{kind}' missing 'coordinates'"))
    })?;
    match kind {
        "Point" => Ok(GeoJsonGeometry::Point(parse_pos(coords, vertex_count)?)),
        "MultiPoint" => Ok(GeoJsonGeometry::MultiPoint(parse_pos_list(
            coords,
            vertex_count,
        )?)),
        "LineString" => Ok(GeoJsonGeometry::LineString(parse_pos_list(
            coords,
            vertex_count,
        )?)),
        "MultiLineString" => {
            let arr = coords
                .as_array()
                .ok_or_else(|| GeoJsonError::Structure("MultiLineString expects array".into()))?;
            let mut lines = Vec::with_capacity(arr.len());
            for item in arr {
                lines.push(parse_pos_list(item, vertex_count)?);
            }
            Ok(GeoJsonGeometry::MultiLineString(lines))
        }
        "Polygon" => Ok(GeoJsonGeometry::Polygon(parse_polygon(
            coords,
            vertex_count,
        )?)),
        "MultiPolygon" => {
            let arr = coords
                .as_array()
                .ok_or_else(|| GeoJsonError::Structure("MultiPolygon expects array".into()))?;
            let mut polys = Vec::with_capacity(arr.len());
            for item in arr {
                polys.push(parse_polygon(item, vertex_count)?);
            }
            Ok(GeoJsonGeometry::MultiPolygon(polys))
        }
        other => Err(GeoJsonError::Structure(format!(
            "unsupported GeoJSON geometry type '{other}'"
        ))),
    }
}

fn parse_pos(val: &Value, vertex_count: &mut usize) -> Result<GeoPoint, GeoJsonError> {
    *vertex_count = vertex_count.saturating_add(1);
    if *vertex_count > MAX_GEOJSON_VERTICES {
        return Err(GeoJsonError::LimitExceeded(format!(
            "vertex count exceeds {MAX_GEOJSON_VERTICES}"
        )));
    }
    let arr = val
        .as_array()
        .ok_or_else(|| GeoJsonError::Structure("position must be [lon, lat] array".into()))?;
    if arr.len() < 2 {
        return Err(GeoJsonError::Structure(
            "position must have at least 2 numbers [lon, lat]".into(),
        ));
    }
    let lon = arr[0]
        .as_f64()
        .ok_or_else(|| GeoJsonError::Structure("longitude must be finite number".into()))?;
    let lat = arr[1]
        .as_f64()
        .ok_or_else(|| GeoJsonError::Structure("latitude must be finite number".into()))?;
    if !(lat.is_finite()
        && lon.is_finite()
        && (-90.0..=90.0).contains(&lat)
        && (-360.0..=360.0).contains(&lon))
    {
        return Err(GeoJsonError::Structure(format!(
            "invalid WGS84 position [{lon}, {lat}]"
        )));
    }
    Ok(GeoPoint::new(lat, lon))
}

fn parse_pos_list(val: &Value, vertex_count: &mut usize) -> Result<Vec<GeoPoint>, GeoJsonError> {
    let arr = val
        .as_array()
        .ok_or_else(|| GeoJsonError::Structure("expected array of positions".into()))?;
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        out.push(parse_pos(item, vertex_count)?);
    }
    Ok(out)
}

fn parse_polygon(val: &Value, vertex_count: &mut usize) -> Result<Polygon, GeoJsonError> {
    let arr = val
        .as_array()
        .ok_or_else(|| GeoJsonError::Structure("Polygon expects array of linear rings".into()))?;
    let Some(ext_val) = arr.first() else {
        return Err(GeoJsonError::Structure(
            "Polygon must have at least an exterior ring".into(),
        ));
    };
    let ext_pts = parse_pos_list(ext_val, vertex_count)?;
    if dedup_geo_ring(&ext_pts).len() < 3 {
        return Err(GeoJsonError::Structure(
            "Polygon exterior ring must contain at least 3 distinct positions".into(),
        ));
    }
    let exterior = Ring { points: ext_pts };
    let mut holes = Vec::with_capacity(arr.len().saturating_sub(1));
    for hole_val in &arr[1..] {
        let hole_pts = parse_pos_list(hole_val, vertex_count)?;
        if dedup_geo_ring(&hole_pts).len() < 3 {
            return Err(GeoJsonError::Structure(
                "Polygon hole ring must contain at least 3 distinct positions".into(),
            ));
        }
        holes.push(Ring { points: hole_pts });
    }
    Ok(Polygon::with_holes(exterior, holes))
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

/// Export a [`GeoJsonFeatureCollection`] as pretty-printed RFC 7946 GeoJSON
/// (`[lon, lat]` WGS84 coordinates, closed linear rings, exterior rings
/// counter-clockwise, interior hole rings clockwise).
pub fn export_geojson_feature_collection(collection: &GeoJsonFeatureCollection) -> String {
    let features: Vec<Value> = collection
        .features
        .iter()
        .map(|f| {
            let mut props = f.properties.clone();
            if let Some(name) = &f.name
                && extract_feature_name(&props).as_deref() != Some(name.as_str())
            {
                props.insert("name".to_string(), Value::String(name.clone()));
            }
            json!({
                "type": "Feature",
                "properties": Value::Object(props),
                "geometry": geometry_to_rfc7946_value(&f.geometry),
            })
        })
        .collect();
    let doc = json!({
        "type": "FeatureCollection",
        "features": features,
    });
    serde_json::to_string_pretty(&doc).unwrap_or_else(|_| "{}".to_string()) + "\n"
}

fn point_to_json(p: GeoPoint) -> Option<Value> {
    (p.lat.is_finite() && p.lon.is_finite()).then(|| json!([p.lon, p.lat]))
}

fn points_to_json(pts: &[GeoPoint]) -> Vec<Value> {
    pts.iter().copied().filter_map(point_to_json).collect()
}

fn geometry_to_rfc7946_value(geom: &GeoJsonGeometry) -> Value {
    match geom {
        GeoJsonGeometry::Point(p) => {
            let coords = point_to_json(*p).unwrap_or_else(|| json!([0.0, 0.0]));
            json!({
                "type": "Point",
                "coordinates": coords,
            })
        }
        GeoJsonGeometry::MultiPoint(pts) => json!({
            "type": "MultiPoint",
            "coordinates": points_to_json(pts),
        }),
        GeoJsonGeometry::LineString(pts) => json!({
            "type": "LineString",
            "coordinates": points_to_json(pts),
        }),
        GeoJsonGeometry::MultiLineString(lines) => json!({
            "type": "MultiLineString",
            "coordinates": lines.iter().map(|line| points_to_json(line)).collect::<Vec<_>>(),
        }),
        GeoJsonGeometry::Polygon(poly) => json!({
            "type": "Polygon",
            "coordinates": polygon_to_rfc7946_rings(poly),
        }),
        GeoJsonGeometry::MultiPolygon(polys) => json!({
            "type": "MultiPolygon",
            "coordinates": polys.iter().map(polygon_to_rfc7946_rings).collect::<Vec<_>>(),
        }),
    }
}

fn polygon_to_rfc7946_rings(poly: &Polygon) -> Vec<Value> {
    let mut rings = Vec::with_capacity(1 + poly.holes.len());
    rings.push(ring_to_rfc7946(&poly.exterior.points, true));
    for hole in &poly.holes {
        rings.push(ring_to_rfc7946(&hole.points, false));
    }
    rings
}

/// Normalize a ring to RFC 7946 right-hand winding (`want_ccw = true` for
/// exterior rings, `want_ccw = false` for interior holes) and close the ring.
fn ring_to_rfc7946(points: &[GeoPoint], want_ccw: bool) -> Value {
    let mut cleaned = dedup_geo_ring(points);
    if cleaned.len() >= 3 {
        let signed_area = ring_signed_area_m2(&cleaned);
        let is_ccw = signed_area >= 0.0;
        if is_ccw != want_ccw {
            cleaned.reverse();
        }
    }
    if let Some(&first) = cleaned.first() {
        cleaned.push(first);
    }
    Value::Array(points_to_json(&cleaned))
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
#[path = "../geojson_tests.rs"]
mod tests;
