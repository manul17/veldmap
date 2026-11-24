//! Read-only GeoJSON parsing, RFC 7946 GeoJSON export (with right-hand
//! winding), and WGS84 globe vector mesh tessellation (pole-safe azimuthal
//! equidistant projection + hemispheric splitting + `earcut` + recursive
//! longest-edge interior triangle subdivision onto the WGS84 ellipsoid with
//! outward geodetic normals and per-cluster `f64` RTC coordinates).

use std::collections::HashMap;

use earcut::Earcut;
use glam::DVec3;
use serde_json::{Map, Value, json};
use thiserror::Error;

use super::footprint::{FootprintBounds, GeoPoint, Polygon, Ring};
use super::geodesic::{ring_signed_area_m2, sample_geodesic_arc};
use super::geodesy::{self, WGS84};

/// Hard safety cap on features per GeoJSON document to prevent OOM on
/// malformed or pathological inputs.
pub const MAX_GEOJSON_FEATURES: usize = 100_000;

/// Hard safety cap on parsed vertices per GeoJSON document.
pub const MAX_GEOJSON_VERTICES: usize = 2_000_000;

/// Default maximum chord length (degrees of arc) for polygon interior
/// subdivision and polyline densification on the WGS84 ellipsoid.
pub const DEFAULT_VECTOR_MAX_CHORD_DEG: f64 = 2.0;

/// Maximum triangles emitted for a single vector mesh to guard GPU memory.
const MAX_MESH_TRIANGLES: usize = 500_000;

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
        if pts.is_empty() {
            return None;
        }
        let coords = pts.iter().map(|p| (p.lat, p.lon));
        let first = pts[0];
        let (lat, lon) = geodesy::spherical_centroid(coords).unwrap_or((first.lat, first.lon));
        let radius_deg = pts
            .iter()
            .map(|p| geodesy::separation((lat, lon), (p.lat, p.lon)))
            .fold(0.0, f64::max);
        Some(FootprintBounds {
            lat,
            lon: geodesy::unwind(0.0, lon),
            radius_deg,
        })
    }
}

/// Parse a GeoJSON string (`FeatureCollection`, `Feature`, or bare `Geometry`
/// including `GeometryCollection`) into a [`GeoJsonFeatureCollection`].
pub fn parse_geojson(input: &str) -> Result<GeoJsonFeatureCollection, GeoJsonError> {
    let root: Value = serde_json::from_str(input).map_err(|e| GeoJsonError::Json(e.to_string()))?;
    let mut features = Vec::new();
    let mut vertex_count = 0usize;
    parse_geojson_value(&root, None, &mut features, &mut vertex_count)?;
    Ok(GeoJsonFeatureCollection { features })
}

fn parse_geojson_value(
    val: &Value,
    inherited_props: Option<&Map<String, Value>>,
    out: &mut Vec<GeoJsonFeature>,
    vertex_count: &mut usize,
) -> Result<(), GeoJsonError> {
    let obj = val
        .as_object()
        .ok_or_else(|| GeoJsonError::Structure("GeoJSON root must be a JSON object".to_string()))?;
    let kind = obj
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| GeoJsonError::Structure("missing string 'type' field".to_string()))?;

    match kind {
        "FeatureCollection" => {
            let list = obj
                .get("features")
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    GeoJsonError::Structure("FeatureCollection missing 'features' array".into())
                })?;
            for item in list {
                parse_geojson_value(item, None, out, vertex_count)?;
            }
            Ok(())
        }
        "Feature" => {
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
            parse_geojson_value(geom, Some(&props), out, vertex_count)
        }
        "GeometryCollection" => {
            let geoms = obj
                .get("geometries")
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    GeoJsonError::Structure("GeometryCollection missing 'geometries' array".into())
                })?;
            for g in geoms {
                parse_geojson_value(g, inherited_props, out, vertex_count)?;
            }
            Ok(())
        }
        _ => {
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
    }
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
    if !(lat.is_finite() && lon.is_finite() && (-90.0..=90.0).contains(&lat)) {
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
    let exterior = Ring {
        points: parse_pos_list(ext_val, vertex_count)?,
    };
    let mut holes = Vec::with_capacity(arr.len().saturating_sub(1));
    for hole_val in &arr[1..] {
        holes.push(Ring {
            points: parse_pos_list(hole_val, vertex_count)?,
        });
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
                && !props.contains_key("name")
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

fn geometry_to_rfc7946_value(geom: &GeoJsonGeometry) -> Value {
    match geom {
        GeoJsonGeometry::Point(p) => json!({
            "type": "Point",
            "coordinates": [p.lon, p.lat],
        }),
        GeoJsonGeometry::MultiPoint(pts) => json!({
            "type": "MultiPoint",
            "coordinates": pts.iter().map(|p| json!([p.lon, p.lat])).collect::<Vec<_>>(),
        }),
        GeoJsonGeometry::LineString(pts) => json!({
            "type": "LineString",
            "coordinates": pts.iter().map(|p| json!([p.lon, p.lat])).collect::<Vec<_>>(),
        }),
        GeoJsonGeometry::MultiLineString(lines) => json!({
            "type": "MultiLineString",
            "coordinates": lines.iter()
                .map(|line| line.iter().map(|p| json!([p.lon, p.lat])).collect::<Vec<_>>())
                .collect::<Vec<_>>(),
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
    let mut cleaned = strip_closing_and_dups(points);
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
    Value::Array(cleaned.iter().map(|p| json!([p.lon, p.lat])).collect())
}

// ---------------------------------------------------------------------------
// WGS84 Globe Vector Mesh Tessellation (Subdivided Earcut + RTC Outward Normals)
// ---------------------------------------------------------------------------

/// A single RTC globe vertex for draped vector fills or lines.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VectorVertex {
    /// Position relative to [`VectorMesh::origin_ecef_m`] in metres.
    pub rel_pos: [f32; 3],
    /// Outward WGS84 unit surface normal for horizon clipping (`dot(n, -world) <= 0`).
    pub normal: [f32; 3],
}

/// CPU-tessellated WGS84 vector mesh ready for depth-less Pass 2 rendering.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorMesh {
    /// Cluster RTC origin in `f64` ECEF metres.
    pub origin_ecef_m: DVec3,
    /// Filled polygon triangle vertices (subdivided on WGS84, outward CCW).
    pub fill_vertices: Vec<VectorVertex>,
    /// Triangle index buffer (`3 * tri_count`) referencing `fill_vertices`.
    pub fill_indices: Vec<u32>,
    /// Line-list stroke vertices (polygon boundaries + LineStrings, densified).
    pub line_vertices: Vec<VectorVertex>,
    /// Point marker vertices (each point expanded into a small geodesic diamond
    /// or rendered as crosshairs/markers; stored as WGS84 ECEF RTC points).
    pub point_vertices: Vec<VectorVertex>,
    /// Spherical framing bounds of the mesh.
    pub bounds: Option<FootprintBounds>,
}

impl VectorMesh {
    /// Build a subdivided WGS84 globe mesh from a [`GeoJsonFeatureCollection`].
    pub fn from_collection(collection: &GeoJsonFeatureCollection, max_chord_deg: f64) -> Self {
        let max_chord_deg = max_chord_deg.clamp(0.1, 10.0);
        let bounds = collection.bounds();
        let origin_ecef_m = bounds
            .map(|b| WGS84.to_ecef(geodesy::Geodetic::surface(b.lat, b.lon)))
            .unwrap_or(DVec3::ZERO);

        let mut fill_vertices = Vec::new();
        let mut fill_indices = Vec::new();
        let mut line_vertices = Vec::new();
        let mut point_vertices = Vec::new();

        for feature in &collection.features {
            tessellate_geometry(
                &feature.geometry,
                origin_ecef_m,
                max_chord_deg,
                &mut fill_vertices,
                &mut fill_indices,
                &mut line_vertices,
                &mut point_vertices,
            );
        }

        Self {
            origin_ecef_m,
            fill_vertices,
            fill_indices,
            line_vertices,
            point_vertices,
            bounds,
        }
    }

    /// Return whether the mesh contains no renderable primitives.
    pub fn is_empty(&self) -> bool {
        self.fill_indices.is_empty()
            && self.line_vertices.is_empty()
            && self.point_vertices.is_empty()
    }
}

fn tessellate_geometry(
    geom: &GeoJsonGeometry,
    origin: DVec3,
    max_chord_deg: f64,
    fill_vertices: &mut Vec<VectorVertex>,
    fill_indices: &mut Vec<u32>,
    line_vertices: &mut Vec<VectorVertex>,
    point_vertices: &mut Vec<VectorVertex>,
) {
    match geom {
        GeoJsonGeometry::Point(p) => point_vertices.push(make_vertex(*p, origin)),
        GeoJsonGeometry::MultiPoint(pts) => {
            for &p in pts {
                point_vertices.push(make_vertex(p, origin));
            }
        }
        GeoJsonGeometry::LineString(pts) => {
            append_polyline(pts, false, origin, max_chord_deg, line_vertices);
        }
        GeoJsonGeometry::MultiLineString(lines) => {
            for line in lines {
                append_polyline(line, false, origin, max_chord_deg, line_vertices);
            }
        }
        GeoJsonGeometry::Polygon(poly) => {
            tessellate_polygon(
                poly,
                origin,
                max_chord_deg,
                fill_vertices,
                fill_indices,
                line_vertices,
            );
        }
        GeoJsonGeometry::MultiPolygon(polys) => {
            for poly in polys {
                tessellate_polygon(
                    poly,
                    origin,
                    max_chord_deg,
                    fill_vertices,
                    fill_indices,
                    line_vertices,
                );
            }
        }
    }
}

fn make_vertex(p: GeoPoint, origin: DVec3) -> VectorVertex {
    let (ecef, normal) = WGS84.surface_ecef_and_normal(p.lat, p.lon);
    let rel = ecef - origin;
    VectorVertex {
        rel_pos: [rel.x as f32, rel.y as f32, rel.z as f32],
        normal: [normal.x as f32, normal.y as f32, normal.z as f32],
    }
}

fn unit_to_vertex(u: DVec3, origin: DVec3) -> VectorVertex {
    let u = u.normalize_or_zero();
    let lat = u.z.clamp(-1.0, 1.0).asin().to_degrees();
    let lon = u.y.atan2(u.x).to_degrees();
    make_vertex(GeoPoint::new(lat, lon), origin)
}

fn append_polyline(
    pts: &[GeoPoint],
    closed: bool,
    origin: DVec3,
    max_chord_deg: f64,
    line_vertices: &mut Vec<VectorVertex>,
) {
    if pts.len() < 2 {
        return;
    }
    let count = if closed { pts.len() } else { pts.len() - 1 };
    for i in 0..count {
        let a = pts[i];
        let b = pts[(i + 1) % pts.len()];
        let arc = sample_geodesic_arc(a, b, max_chord_deg);
        for seg in arc.windows(2) {
            line_vertices.push(make_vertex(seg[0], origin));
            line_vertices.push(make_vertex(seg[1], origin));
        }
    }
}

fn tessellate_polygon(
    poly: &Polygon,
    origin: DVec3,
    max_chord_deg: f64,
    fill_vertices: &mut Vec<VectorVertex>,
    fill_indices: &mut Vec<u32>,
    line_vertices: &mut Vec<VectorVertex>,
) {
    let ext_clean = strip_closing_and_dups(&poly.exterior.points);
    if ext_clean.len() < 3 {
        return;
    }

    // Stroke exterior + hole boundaries along true WGS84 geodesics.
    append_polyline(&ext_clean, true, origin, max_chord_deg, line_vertices);
    let mut holes_clean = Vec::with_capacity(poly.holes.len());
    for hole in &poly.holes {
        let h = strip_closing_and_dups(&hole.points);
        if h.len() >= 3 {
            append_polyline(&h, true, origin, max_chord_deg, line_vertices);
            holes_clean.push(h);
        }
    }

    // Convert rings to densified unit-sphere polygons so spherical edges stay
    // faithful under azimuthal equidistant projection and hemisphere clipping.
    let ext_units = densify_ring_units(&ext_clean, max_chord_deg);
    if ext_units.len() < 3 {
        return;
    }
    let hole_units: Vec<Vec<DVec3>> = holes_clean
        .iter()
        .map(|h| densify_ring_units(h, max_chord_deg))
        .filter(|h| h.len() >= 3)
        .collect();

    // Compute spherical centroid of the exterior ring.
    let sum: DVec3 = ext_units.iter().copied().sum();
    let center = if sum.length_squared() > 1e-12 {
        sum.normalize()
    } else {
        ext_units[0]
    };

    let max_angle_deg = ext_units
        .iter()
        .map(|u| u.dot(center).clamp(-1.0, 1.0).acos().to_degrees())
        .fold(0.0_f64, f64::max);

    // Pole-safe Azimuthal Equidistant (AEQD) works cleanly up to ~150 deg
    // angular radius without gnomonic >= 90 deg blowup. For near-global
    // polygons (>= 145 deg), split across the complementary hemispheres
    // `u . center >= 0` and `u . center <= 0` first so every piece stays <= 90 deg!
    if max_angle_deg < 145.0 {
        triangulate_in_aeqd(
            center,
            &ext_units,
            &hole_units,
            origin,
            max_chord_deg,
            fill_vertices,
            fill_indices,
        );
    } else {
        for hemi_normal in [center, -center] {
            let clipped_ext = clip_unit_ring_to_hemisphere(&ext_units, hemi_normal);
            if clipped_ext.len() < 3 {
                continue;
            }
            let clipped_holes: Vec<Vec<DVec3>> = hole_units
                .iter()
                .map(|h| clip_unit_ring_to_hemisphere(h, hemi_normal))
                .filter(|h| h.len() >= 3)
                .collect();
            triangulate_in_aeqd(
                hemi_normal,
                &clipped_ext,
                &clipped_holes,
                origin,
                max_chord_deg,
                fill_vertices,
                fill_indices,
            );
        }
    }
}

/// Pole-safe orthonormal tangent basis `(ex, ey)` at unit vector `center`
/// (`ex x ey = center`, right-handed outward).
fn tangent_basis(center: DVec3) -> (DVec3, DVec3) {
    let ref_axis = if center.z.abs() < 0.9 {
        DVec3::Z
    } else {
        DVec3::X
    };
    let ex = ref_axis.cross(center).normalize();
    let ey = center.cross(ex).normalize();
    (ex, ey)
}

/// Project unit vector `u` into pole-safe Azimuthal Equidistant `(x, y)`
/// centered at `center`.
fn aeqd_project(u: DVec3, center: DVec3, ex: DVec3, ey: DVec3) -> [f64; 2] {
    let cos_t = u.dot(center).clamp(-1.0, 1.0);
    let theta = cos_t.acos();
    if theta < 1e-12 {
        return [0.0, 0.0];
    }
    let tang = u - center * cos_t;
    let tang_len = tang.length();
    if tang_len < 1e-14 {
        return [theta, 0.0];
    }
    let scale = theta / tang_len;
    [tang.dot(ex) * scale, tang.dot(ey) * scale]
}

fn triangulate_in_aeqd(
    center: DVec3,
    exterior: &[DVec3],
    holes: &[Vec<DVec3>],
    origin: DVec3,
    max_chord_deg: f64,
    fill_vertices: &mut Vec<VectorVertex>,
    fill_indices: &mut Vec<u32>,
) {
    if fill_indices.len() / 3 >= MAX_MESH_TRIANGLES {
        return;
    }
    let (ex, ey) = tangent_basis(center);

    let mut coords_2d: Vec<[f64; 2]> =
        Vec::with_capacity(exterior.len() + holes.iter().map(Vec::len).sum::<usize>());
    let mut ring_units: Vec<DVec3> = Vec::with_capacity(coords_2d.capacity());
    let mut hole_indices: Vec<usize> = Vec::with_capacity(holes.len());

    for &u in exterior {
        coords_2d.push(aeqd_project(u, center, ex, ey));
        ring_units.push(u);
    }
    for hole in holes {
        hole_indices.push(coords_2d.len());
        for &u in hole {
            coords_2d.push(aeqd_project(u, center, ex, ey));
            ring_units.push(u);
        }
    }

    let mut tri_idx: Vec<usize> = Vec::new();
    let mut ec = Earcut::new();
    ec.earcut(coords_2d.iter().copied(), &hole_indices, &mut tri_idx);
    if tri_idx.len() < 3 {
        return;
    }

    // Build local vertex pool for recursive longest-edge interior subdivision
    // (caching edge midpoints so shared interior edges never crack!).
    let mut sub_units: Vec<DVec3> = ring_units;
    let mut midpoint_cache: HashMap<(u32, u32), u32> = HashMap::new();
    let mut work: Vec<(u32, u32, u32, u8)> = Vec::with_capacity(tri_idx.len() / 3);
    for tri in tri_idx.chunks_exact(3) {
        work.push((tri[0] as u32, tri[1] as u32, tri[2] as u32, 0));
    }

    let cos_limit = max_chord_deg.to_radians().cos();
    let mut final_tris: Vec<(u32, u32, u32)> = Vec::with_capacity(work.len());
    let max_depth = 8u8;
    let remaining_budget = MAX_MESH_TRIANGLES.saturating_sub(fill_indices.len() / 3);

    while let Some((i0, i1, i2, depth)) = work.pop() {
        let u0 = sub_units[i0 as usize];
        let u1 = sub_units[i1 as usize];
        let u2 = sub_units[i2 as usize];

        let d01 = u0.dot(u1).clamp(-1.0, 1.0);
        let d12 = u1.dot(u2).clamp(-1.0, 1.0);
        let d20 = u2.dot(u0).clamp(-1.0, 1.0);

        // Smaller cosine = larger spherical angle.
        let min_cos = d01.min(d12).min(d20);
        if min_cos < cos_limit
            && depth < max_depth
            && final_tris.len() + work.len() < remaining_budget
        {
            let mut get_mid = |a: u32, b: u32| -> u32 {
                let key = if a < b { (a, b) } else { (b, a) };
                *midpoint_cache.entry(key).or_insert_with(|| {
                    let mid = (sub_units[a as usize] + sub_units[b as usize]).normalize_or_zero();
                    let idx = sub_units.len() as u32;
                    sub_units.push(if mid.length_squared() > 1e-12 {
                        mid
                    } else {
                        sub_units[a as usize]
                    });
                    idx
                })
            };

            if d01 <= d12 && d01 <= d20 {
                let m = get_mid(i0, i1);
                work.push((i0, m, i2, depth + 1));
                work.push((m, i1, i2, depth + 1));
            } else if d12 <= d01 && d12 <= d20 {
                let m = get_mid(i1, i2);
                work.push((i0, i1, m, depth + 1));
                work.push((i0, m, i2, depth + 1));
            } else {
                let m = get_mid(i2, i0);
                work.push((i0, i1, m, depth + 1));
                work.push((m, i1, i2, depth + 1));
            }
        } else {
            final_tris.push((i0, i1, i2));
        }
    }

    // Re-project every used vertex onto WGS84 with outward geodetic normals and
    // enforce outward CCW winding using the GPU-encoded `f32` RTC positions so
    // `f32` quantization can never flip a sliver triangle against backface culling.
    let base_vertex = fill_vertices.len() as u32;
    let mut rtc_pts = Vec::with_capacity(sub_units.len());
    let mut normals = Vec::with_capacity(sub_units.len());
    for &u in &sub_units {
        let v = unit_to_vertex(u, origin);
        rtc_pts.push(DVec3::new(
            f64::from(v.rel_pos[0]),
            f64::from(v.rel_pos[1]),
            f64::from(v.rel_pos[2]),
        ));
        normals.push(DVec3::new(
            f64::from(v.normal[0]),
            f64::from(v.normal[1]),
            f64::from(v.normal[2]),
        ));
        fill_vertices.push(v);
    }

    for (i0, i1, i2) in final_tris {
        let p0 = rtc_pts[i0 as usize];
        let p1 = rtc_pts[i1 as usize];
        let p2 = rtc_pts[i2 as usize];
        let avg_n = normals[i0 as usize] + normals[i1 as usize] + normals[i2 as usize];
        let face_n = (p1 - p0).cross(p2 - p0);
        if face_n.length_squared() <= 1e-6 {
            continue;
        }
        if face_n.dot(avg_n) >= 0.0 {
            fill_indices.push(base_vertex + i0);
            fill_indices.push(base_vertex + i1);
            fill_indices.push(base_vertex + i2);
        } else {
            fill_indices.push(base_vertex + i0);
            fill_indices.push(base_vertex + i2);
            fill_indices.push(base_vertex + i1);
        }
    }
}

/// Spherical Sutherland-Hodgman clip of a closed unit-sphere ring against the
/// hemisphere half-space `u . normal >= 0`.
fn clip_unit_ring_to_hemisphere(ring: &[DVec3], normal: DVec3) -> Vec<DVec3> {
    if ring.len() < 3 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(ring.len() + 2);
    let mut prev = *ring.last().unwrap_or(&DVec3::X);
    let mut prev_d = prev.dot(normal);
    for &curr in ring {
        let curr_d = curr.dot(normal);
        let prev_in = prev_d >= 0.0;
        let curr_in = curr_d >= 0.0;
        if prev_in != curr_in {
            let denom = prev_d - curr_d;
            if denom.abs() > 1e-14 {
                let t = (prev_d / denom).clamp(0.0, 1.0);
                let inter = (prev * (1.0 - t) + curr * t).normalize_or_zero();
                if inter.length_squared() > 1e-12 {
                    out.push(inter);
                }
            }
        }
        if curr_in {
            out.push(curr);
        }
        prev = curr;
        prev_d = curr_d;
    }
    out
}

fn densify_ring_units(ring: &[GeoPoint], max_chord_deg: f64) -> Vec<DVec3> {
    let mut out = Vec::with_capacity(ring.len());
    for i in 0..ring.len() {
        let a = ring[i];
        let b = ring[(i + 1) % ring.len()];
        let ua = geodesy::unit(a.lat, a.lon);
        let ub = geodesy::unit(b.lat, b.lon);
        let ang_deg = ua.dot(ub).clamp(-1.0, 1.0).acos().to_degrees();
        let steps = ((ang_deg / max_chord_deg).ceil() as usize).clamp(1, 180);
        for s in 0..steps {
            let t = s as f64 / steps as f64;
            let p = if s == 0 {
                ua
            } else {
                let (lat, lon) = geodesy::between((a.lat, a.lon), (b.lat, b.lon), t);
                geodesy::unit(lat, lon)
            };
            if out
                .last()
                .is_none_or(|prev: &DVec3| prev.distance_squared(p) > 1e-18)
            {
                out.push(p);
            }
        }
    }
    if out.len() > 1
        && let (Some(first), Some(last)) = (out.first(), out.last())
        && first.distance_squared(*last) <= 1e-18
    {
        out.pop();
    }
    out
}

fn strip_closing_and_dups(points: &[GeoPoint]) -> Vec<GeoPoint> {
    let mut out: Vec<GeoPoint> = Vec::with_capacity(points.len());
    for &p in points {
        if out
            .last()
            .is_none_or(|prev| (prev.lat - p.lat).abs() > 1e-12 || (prev.lon - p.lon).abs() > 1e-12)
        {
            out.push(p);
        }
    }
    if out.len() > 1
        && let (Some(first), Some(last)) = (out.first(), out.last())
        && (first.lat - last.lat).abs() <= 1e-12
        && (first.lon - last.lon).abs() <= 1e-12
    {
        out.pop();
    }
    out
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
#[path = "geojson_tests.rs"]
mod tests;
