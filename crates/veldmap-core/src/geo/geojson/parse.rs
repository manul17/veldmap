//! Read-only GeoJSON document parser supporting `FeatureCollection`,
//! `Feature`, and bare `Geometry` / `GeometryCollection` inputs with strict
//! memory and nesting safety caps.

use serde_json::{Map, Value};

use super::super::shapes::{GeoPoint, Polygon, Ring, dedup_geo_ring};
use super::{
    GeoJsonError, GeoJsonFeature, GeoJsonFeatureCollection, GeoJsonGeometry, MAX_GEOJSON_BYTES,
    extract_feature_name,
};

/// Hard safety cap on features per GeoJSON document to prevent OOM on
/// malformed or pathological inputs.
const MAX_GEOJSON_FEATURES: usize = 100_000;

/// Hard safety cap on parsed vertices per GeoJSON document.
const MAX_GEOJSON_VERTICES: usize = 2_000_000;

/// Maximum nesting depth for `GeometryCollection` containers.
const MAX_GEOMETRY_COLLECTION_DEPTH: usize = 8;

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
