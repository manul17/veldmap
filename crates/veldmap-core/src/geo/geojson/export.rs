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

/// Export a standalone RFC 7946 GeoJSON geometry object (suitable for STAC
/// `intersects` queries or AOI clipboard export).
pub fn export_geojson_geometry(geom: &GeoJsonGeometry) -> String {
    let val = geometry_to_rfc7946_value(geom);
    serde_json::to_string_pretty(&val).unwrap_or_else(|_| "{}".to_string())
}

/// Export a [`GeoJsonGeometry`] as OGC Well-Known Text (WKT) in WGS84
/// `lon lat` coordinate order.
pub fn export_wkt_geometry(geom: &GeoJsonGeometry) -> String {
    match geom {
        GeoJsonGeometry::Point(p) => {
            if is_finite_point(*p) {
                format!("POINT ({})", format_wkt_point(*p))
            } else {
                "POINT EMPTY".to_string()
            }
        }
        GeoJsonGeometry::MultiPoint(pts) => {
            let inner: Vec<String> = pts
                .iter()
                .copied()
                .filter(|&p| is_finite_point(p))
                .map(|p| format!("({})", format_wkt_point(p)))
                .collect();
            if inner.is_empty() {
                "MULTIPOINT EMPTY".to_string()
            } else {
                format!("MULTIPOINT ({})", inner.join(", "))
            }
        }
        GeoJsonGeometry::LineString(pts) => {
            let coords = format_wkt_coords(pts);
            if coords.is_empty() {
                "LINESTRING EMPTY".to_string()
            } else {
                format!("LINESTRING ({coords})")
            }
        }
        GeoJsonGeometry::MultiLineString(lines) => {
            let inner: Vec<String> = lines
                .iter()
                .map(|line| format_wkt_coords(line))
                .filter(|coords| !coords.is_empty())
                .map(|coords| format!("({coords})"))
                .collect();
            if inner.is_empty() {
                "MULTILINESTRING EMPTY".to_string()
            } else {
                format!("MULTILINESTRING ({})", inner.join(", "))
            }
        }
        GeoJsonGeometry::Polygon(poly) => match format_wkt_polygon(poly) {
            Some(rings) => format!("POLYGON {rings}"),
            None => "POLYGON EMPTY".to_string(),
        },
        GeoJsonGeometry::MultiPolygon(polys) => {
            let inner: Vec<String> = polys.iter().filter_map(format_wkt_polygon).collect();
            if inner.is_empty() {
                "MULTIPOLYGON EMPTY".to_string()
            } else {
                format!("MULTIPOLYGON ({})", inner.join(", "))
            }
        }
    }
}

fn is_finite_point(p: GeoPoint) -> bool {
    p.lat.is_finite() && p.lon.is_finite()
}

fn normalize_wkt_coord(v: f64) -> f64 {
    if v.abs() < 0.5e-8 { 0.0 } else { v }
}

fn format_wkt_point(p: GeoPoint) -> String {
    format!(
        "{:.8} {:.8}",
        normalize_wkt_coord(p.lon),
        normalize_wkt_coord(p.lat)
    )
}

fn format_wkt_coords(pts: &[GeoPoint]) -> String {
    pts.iter()
        .copied()
        .filter(|&p| is_finite_point(p))
        .map(format_wkt_point)
        .collect::<Vec<_>>()
        .join(", ")
}

/// Normalize a ring to RFC 7946 right-hand winding (`want_ccw = true` for
/// exterior rings, `want_ccw = false` for interior holes), filtering non-finite
/// vertices and closing the ring.
fn normalize_ring(points: &[GeoPoint], want_ccw: bool) -> Vec<GeoPoint> {
    let finite: Vec<GeoPoint> = points
        .iter()
        .copied()
        .filter(|&p| is_finite_point(p))
        .collect();
    let mut cleaned = dedup_geo_ring(&finite);
    if cleaned.len() >= 3 {
        let is_ccw = ring_signed_area_m2(&cleaned) >= 0.0;
        if is_ccw != want_ccw {
            cleaned.reverse();
        }
    }
    if let Some(&first) = cleaned.first() {
        cleaned.push(first);
    }
    cleaned
}

fn format_wkt_ring(points: &[GeoPoint], want_ccw: bool) -> Option<String> {
    let cleaned = normalize_ring(points, want_ccw);
    if cleaned.is_empty() {
        return None;
    }
    Some(format!("({})", format_wkt_coords(&cleaned)))
}

fn format_wkt_polygon(poly: &Polygon) -> Option<String> {
    let exterior = format_wkt_ring(&poly.exterior.points, true)?;
    let mut rings = Vec::with_capacity(1 + poly.holes.len());
    rings.push(exterior);
    for hole in &poly.holes {
        if let Some(ring) = format_wkt_ring(&hole.points, false) {
            rings.push(ring);
        }
    }
    Some(format!("({})", rings.join(", ")))
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
    let cleaned = normalize_ring(points, want_ccw);
    Value::Array(points_to_json(&cleaned))
}
