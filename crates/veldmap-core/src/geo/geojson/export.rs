//! RFC 7946 GeoJSON serializer enforcing `[lon, lat]` WGS84 coordinates,
//! closed linear rings, counter-clockwise exterior rings, and clockwise holes.

use serde_json::{Value, json};

use super::super::geodesic::ring_signed_area_m2;
use super::super::shapes::{GeoPoint, Polygon, dedup_geo_ring};
use super::{GeoJsonFeatureCollection, GeoJsonGeometry, extract_feature_name};

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
