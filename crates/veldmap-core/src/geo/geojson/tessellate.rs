//! WGS84 globe vector mesh tessellation using exterior-pole stereographic
//! projection, `earcut` triangulation, conforming longest-edge interior
//! subdivision, and per-cluster `f64` RTC encoding with outward geodetic normals.

use std::collections::HashMap;

use earcut::Earcut;
use glam::DVec3;

use super::cluster::build_rtc_clusters;
use super::{GeoJsonFeatureCollection, GeoJsonGeometry};
use crate::geo::footprint::{FootprintBounds, GeoPoint, Polygon, dedup_geo_ring};
use crate::geo::geodesic::sample_geodesic_arc;
use crate::geo::geodesy::{self, WGS84};

/// Default maximum chord length (degrees of arc) for polygon interior
/// subdivision and polyline densification on the WGS84 ellipsoid.
///
/// At `0.5 deg`, maximum chord sagitta on WGS84 is `~60 m` (down from `~971 m`
/// at `2.0 deg`) while keeping regional and continental meshes compact.
pub const DEFAULT_VECTOR_MAX_CHORD_DEG: f64 = 0.5;

/// Maximum triangles emitted across all clusters of a vector mesh.
const MAX_MESH_TRIANGLES: usize = 500_000;

/// Maximum line-list vertices emitted across all clusters of a vector mesh
/// (`1_000_000 * 24 B = 24 MiB`, well within `wgpu`'s `256 MiB` limit).
const MAX_LINE_VERTICES: usize = 1_000_000;

/// Maximum point marker instances emitted across all clusters of a vector mesh
/// (`500_000 * 24 B = 12 MiB`).
const MAX_POINT_VERTICES: usize = 500_000;

/// A single RTC globe vertex for draped vector fills, lines, or point centers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VectorVertex {
    /// Position relative to [`VectorCluster::origin_ecef_m`] in metres.
    pub rel_pos: [f32; 3],
    /// Outward WGS84 unit surface normal for horizon clipping (`dot(n, -world) <= 0`).
    pub normal: [f32; 3],
}

/// One spatially localized RTC cluster within a [`VectorMesh`].
#[derive(Clone, Debug, PartialEq)]
pub struct VectorCluster {
    /// Cluster RTC origin in `f64` ECEF metres.
    pub origin_ecef_m: DVec3,
    /// Filled polygon triangle vertices (subdivided on WGS84, outward CCW).
    pub fill_vertices: Vec<VectorVertex>,
    /// Triangle index buffer (`3 * tri_count`) referencing `fill_vertices`.
    pub fill_indices: Vec<u32>,
    /// Line-list stroke vertices (polygon boundaries + LineStrings, densified).
    pub line_vertices: Vec<VectorVertex>,
    /// Point marker center vertices (instanced on GPU in `vector.wgsl`).
    pub point_vertices: Vec<VectorVertex>,
}

impl VectorCluster {
    /// Return whether this cluster contains no renderable primitives.
    pub fn is_empty(&self) -> bool {
        self.fill_indices.is_empty()
            && self.line_vertices.is_empty()
            && self.point_vertices.is_empty()
    }
}

/// CPU-tessellated WGS84 vector mesh partitioned into per-cluster `f64` RTC
/// buckets ready for depth-less Pass 2 rendering.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorMesh {
    /// Spatial RTC clusters (each carrying its own `f64` ECEF origin).
    pub clusters: Vec<VectorCluster>,
    /// Spherical framing bounds of the mesh.
    pub bounds: Option<FootprintBounds>,
}

impl VectorMesh {
    /// Build a subdivided WGS84 globe mesh from a [`GeoJsonFeatureCollection`].
    pub fn from_collection(collection: &GeoJsonFeatureCollection, max_chord_deg: f64) -> Self {
        let max_chord_deg = if max_chord_deg.is_finite() {
            max_chord_deg.clamp(0.1, 10.0)
        } else {
            DEFAULT_VECTOR_MAX_CHORD_DEG
        };
        let bounds = collection.bounds();
        let mut raw = RawPrimitives::default();
        let mut earcut = Earcut::new();

        for feature in &collection.features {
            tessellate_geometry(&feature.geometry, max_chord_deg, &mut earcut, &mut raw);
        }

        let clusters = build_rtc_clusters(raw);
        Self { clusters, bounds }
    }

    /// Return whether the mesh contains no renderable primitives.
    pub fn is_empty(&self) -> bool {
        self.clusters.iter().all(VectorCluster::is_empty)
    }

    /// Total filled polygon triangle count across all clusters.
    pub fn fill_triangle_count(&self) -> usize {
        self.clusters.iter().map(|c| c.fill_indices.len() / 3).sum()
    }

    /// Total line-list segment count across all clusters.
    pub fn line_segment_count(&self) -> usize {
        self.clusters
            .iter()
            .map(|c| c.line_vertices.len() / 2)
            .sum()
    }

    /// Total point marker count across all clusters.
    pub fn point_count(&self) -> usize {
        self.clusters.iter().map(|c| c.point_vertices.len()).sum()
    }
}

/// Intermediate `f64` ECEF primitive accumulator prior to per-cluster RTC
/// encoding.
#[derive(Default)]
pub(super) struct RawPrimitives {
    /// Triangles stored as `(p0, n0, p1, n1, p2, n2)` in `f64` ECEF metres.
    pub(super) fill_tris: Vec<[(DVec3, DVec3); 3]>,
    /// Line segments stored as `(p0, n0, p1, n1)` in `f64` ECEF metres.
    pub(super) line_segs: Vec<[(DVec3, DVec3); 2]>,
    /// Point centers stored as `(p, n)` in `f64` ECEF metres.
    pub(super) points: Vec<(DVec3, DVec3)>,
}

impl RawPrimitives {
    fn remaining_tri_budget(&self) -> usize {
        MAX_MESH_TRIANGLES.saturating_sub(self.fill_tris.len())
    }

    fn push_line_segment(&mut self, a: (DVec3, DVec3), b: (DVec3, DVec3)) -> bool {
        if self.line_segs.len().saturating_mul(2) + 2 > MAX_LINE_VERTICES {
            return false;
        }
        self.line_segs.push([a, b]);
        true
    }

    fn push_point(&mut self, p: (DVec3, DVec3)) -> bool {
        if self.points.len() >= MAX_POINT_VERTICES {
            return false;
        }
        self.points.push(p);
        true
    }
}

fn tessellate_geometry(
    geom: &GeoJsonGeometry,
    max_chord_deg: f64,
    earcut: &mut Earcut<f64>,
    raw: &mut RawPrimitives,
) {
    match geom {
        GeoJsonGeometry::Point(p) => {
            if p.lat.is_finite() && p.lon.is_finite() {
                raw.push_point(WGS84.surface_ecef_and_normal(p.lat, p.lon));
            }
        }
        GeoJsonGeometry::MultiPoint(pts) => {
            for &p in pts {
                if p.lat.is_finite()
                    && p.lon.is_finite()
                    && !raw.push_point(WGS84.surface_ecef_and_normal(p.lat, p.lon))
                {
                    break;
                }
            }
        }
        GeoJsonGeometry::LineString(pts) => {
            append_polyline(pts, max_chord_deg, raw);
        }
        GeoJsonGeometry::MultiLineString(lines) => {
            for line in lines {
                append_polyline(line, max_chord_deg, raw);
            }
        }
        GeoJsonGeometry::Polygon(poly) => {
            tessellate_polygon(poly, max_chord_deg, earcut, raw);
        }
        GeoJsonGeometry::MultiPolygon(polys) => {
            for poly in polys {
                tessellate_polygon(poly, max_chord_deg, earcut, raw);
            }
        }
    }
}

fn append_polyline(pts: &[GeoPoint], max_chord_deg: f64, raw: &mut RawPrimitives) {
    if pts.len() < 2 {
        return;
    }
    for seg_endpoints in pts.windows(2) {
        let arc = sample_geodesic_arc(seg_endpoints[0], seg_endpoints[1], max_chord_deg);
        for seg in arc.windows(2) {
            let p0 = WGS84.surface_ecef_and_normal(seg[0].lat, seg[0].lon);
            let p1 = WGS84.surface_ecef_and_normal(seg[1].lat, seg[1].lon);
            if !raw.push_line_segment(p0, p1) {
                return;
            }
        }
    }
}

fn tessellate_polygon(
    poly: &Polygon,
    max_chord_deg: f64,
    earcut: &mut Earcut<f64>,
    raw: &mut RawPrimitives,
) {
    let ext_clean = dedup_geo_ring(&poly.exterior.points);
    if ext_clean.len() < 3 {
        return;
    }

    // Densify exterior and hole rings along true WGS84 Karney geodesics so
    // stroke boundaries and filled polygon edges coincide exactly.
    let ext_geo = densify_ring_geodesic(&ext_clean, max_chord_deg);
    if ext_geo.len() < 3 {
        return;
    }
    emit_closed_ring_lines(&ext_geo, raw);

    let mut holes_units: Vec<Vec<DVec3>> = Vec::with_capacity(poly.holes.len());
    for hole in &poly.holes {
        let h_clean = dedup_geo_ring(&hole.points);
        if h_clean.len() >= 3 {
            let h_geo = densify_ring_geodesic(&h_clean, max_chord_deg);
            if h_geo.len() >= 3 {
                emit_closed_ring_lines(&h_geo, raw);
                holes_units.push(h_geo.iter().map(|p| geodesy::unit(p.lat, p.lon)).collect());
            }
        }
    }

    let ext_units: Vec<DVec3> = ext_geo
        .iter()
        .map(|p| geodesy::unit(p.lat, p.lon))
        .collect();
    triangulate_spherical_polygon(&ext_units, &holes_units, max_chord_deg, earcut, raw);
}

fn emit_closed_ring_lines(ring: &[GeoPoint], raw: &mut RawPrimitives) {
    if ring.len() < 2 {
        return;
    }
    for i in 0..ring.len() {
        let a = ring[i];
        let b = ring[(i + 1) % ring.len()];
        let p0 = WGS84.surface_ecef_and_normal(a.lat, a.lon);
        let p1 = WGS84.surface_ecef_and_normal(b.lat, b.lon);
        if !raw.push_line_segment(p0, p1) {
            return;
        }
    }
}

fn densify_ring_geodesic(ring: &[GeoPoint], max_chord_deg: f64) -> Vec<GeoPoint> {
    let mut out = Vec::with_capacity(ring.len());
    for i in 0..ring.len() {
        let a = ring[i];
        let b = ring[(i + 1) % ring.len()];
        let arc = sample_geodesic_arc(a, b, max_chord_deg);
        let emit_len = arc.len().saturating_sub(1);
        for &p in arc.iter().take(emit_len) {
            if out.last().is_none_or(|prev: &GeoPoint| {
                (prev.lat - p.lat).abs() > 1e-12 || (prev.lon - p.lon).abs() > 1e-12
            }) {
                out.push(p);
            }
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

/// Conformal stereographic projection onto the plane tangent at `center = -p_out`
/// (with `p_out` strictly exterior to the spherical polygon).
fn stereo_project(u: DVec3, center: DVec3, ex: DVec3, ey: DVec3) -> [f64; 2] {
    let denom = (1.0 + u.dot(center)).max(1e-12);
    let scale = 2.0 / denom;
    [u.dot(ex) * scale, u.dot(ey) * scale]
}

/// Exact inverse of [`stereo_project`] mapping 2D stereographic coordinates
/// back onto the unit sphere.
fn stereo_unproject(p: [f64; 2], center: DVec3, ex: DVec3, ey: DVec3) -> DVec3 {
    let r2 = p[0] * p[0] + p[1] * p[1];
    let inv = 1.0 / (4.0 + r2);
    (ex * (4.0 * p[0] * inv) + ey * (4.0 * p[1] * inv) + center * ((4.0 - r2) * inv))
        .normalize_or_zero()
}

fn shoelace_area_2d(pts: &[[f64; 2]]) -> f64 {
    if pts.len() < 3 {
        return 0.0;
    }
    let mut sum = 0.0;
    for i in 0..pts.len() {
        let a = pts[i];
        let b = pts[(i + 1) % pts.len()];
        sum += a[0] * b[1] - b[0] * a[1];
    }
    0.5 * sum
}

/// Choose a stereographic map center `center = -p_out` whose projection pole
/// `p_out` is strictly exterior to the polygon and maximally separated from
/// every ring vertex.
///
/// For compact polygons (`max_angle_deg < 45 deg` around centroid), the smaller
/// side is always treated as interior (tolerating legacy CW local GeoJSONs).
/// For wide, continental, hemispheric, and `> hemisphere` polygons
/// (`max_angle_deg >= 45 deg`), strict RFC 7946 CCW winding selects the
/// exterior pole (`shoelace_area > 0`).
fn choose_stereographic_center(exterior: &[DVec3], holes: &[Vec<DVec3>]) -> DVec3 {
    let sum: DVec3 = exterior.iter().copied().sum();
    let centroid = if sum.length_squared() > 1e-12 {
        sum.normalize()
    } else {
        exterior[0]
    };

    let max_angle_deg = exterior
        .iter()
        .map(|u| u.dot(centroid).clamp(-1.0, 1.0).acos().to_degrees())
        .fold(0.0_f64, f64::max);

    if max_angle_deg < 45.0 {
        return centroid;
    }

    // Build candidate projection poles `p_out` (so `center = -p_out`).
    let mut candidates = Vec::with_capacity(22);
    candidates.push(-centroid);
    candidates.push(centroid);
    for axis in [
        DVec3::X,
        -DVec3::X,
        DVec3::Y,
        -DVec3::Y,
        DVec3::Z,
        -DVec3::Z,
    ] {
        candidates.push(axis);
    }
    let inv_s3 = 1.0 / 3.0_f64.sqrt();
    for sx in [-1.0, 1.0] {
        for sy in [-1.0, 1.0] {
            for sz in [-1.0, 1.0] {
                candidates.push(DVec3::new(sx * inv_s3, sy * inv_s3, sz * inv_s3));
            }
        }
    }
    let step = (exterior.len() / 4).max(1);
    for i in (0..exterior.len()).step_by(step).take(4) {
        candidates.push(-exterior[i]);
    }
    // Also add outward normal offsets near the first edge in case a near-full-Earth
    // polygon leaves only a narrow exterior strip.
    if exterior.len() >= 2 {
        let edge_mid = (exterior[0] + exterior[1]).normalize_or_zero();
        let edge_n = exterior[0].cross(exterior[1]).normalize_or_zero();
        if edge_mid.length_squared() > 0.5 && edge_n.length_squared() > 0.5 {
            for eps in [0.05, 0.2] {
                candidates.push((edge_mid - edge_n * eps).normalize());
                candidates.push((edge_mid + edge_n * eps).normalize());
            }
        }
    }

    let mut best_center = centroid;
    let mut best_clearance = f64::NEG_INFINITY;
    let mut scratch = Vec::with_capacity(exterior.len());

    for p_out in candidates {
        // Clearance = minimum angular separation from `p_out` to any boundary
        // vertex (smaller dot product = larger angular separation).
        let max_dot = exterior
            .iter()
            .chain(holes.iter().flat_map(|h| h.iter()))
            .map(|u| u.dot(p_out))
            .fold(f64::NEG_INFINITY, f64::max);
        // Reject poles sitting right on a vertex (`< ~0.5 deg`).
        if max_dot > 0.999_96 {
            continue;
        }
        let center = -p_out;
        let (ex, ey) = tangent_basis(center);
        scratch.clear();
        for &u in exterior {
            scratch.push(stereo_project(u, center, ex, ey));
        }
        let area = shoelace_area_2d(&scratch);
        // For CCW RFC 7946 exterior rings, `area > 0` iff `p_out` is exterior.
        if area > 0.0 {
            let clearance = -max_dot;
            if clearance > best_clearance {
                best_clearance = clearance;
                best_center = center;
            }
        }
    }

    best_center
}

fn triangulate_spherical_polygon(
    exterior: &[DVec3],
    holes: &[Vec<DVec3>],
    max_chord_deg: f64,
    earcut: &mut Earcut<f64>,
    raw: &mut RawPrimitives,
) {
    let total_remaining = raw.remaining_tri_budget();
    if total_remaining == 0 || exterior.len() < 3 {
        return;
    }

    let center = choose_stereographic_center(exterior, holes);
    let (ex, ey) = tangent_basis(center);

    let total_verts = exterior.len() + holes.iter().map(Vec::len).sum::<usize>();
    let mut coords_2d: Vec<[f64; 2]> = Vec::with_capacity(total_verts);
    let mut ring_units: Vec<DVec3> = Vec::with_capacity(total_verts);
    let mut hole_indices: Vec<usize> = Vec::with_capacity(holes.len());

    for &u in exterior {
        coords_2d.push(stereo_project(u, center, ex, ey));
        ring_units.push(u);
    }
    for hole in holes {
        hole_indices.push(coords_2d.len());
        for &u in hole {
            coords_2d.push(stereo_project(u, center, ex, ey));
            ring_units.push(u);
        }
    }

    let mut tri_idx: Vec<usize> = Vec::new();
    earcut.earcut(coords_2d.iter().copied(), &hole_indices, &mut tri_idx);
    if tri_idx.len() < 3 {
        return;
    }

    // Reserve 15% of remaining triangle budget for the conforming midpoint
    // closure sweep so asymmetric depth/budget stops never leave T-junctions.
    let refine_budget = ((total_remaining as f64) * 0.85).floor().max(1.0) as usize;
    let mut sub_units: Vec<DVec3> = ring_units;
    let mut sub_2d: Vec<[f64; 2]> = coords_2d;
    let mut midpoint_cache: HashMap<(u32, u32), u32> = HashMap::new();
    let mut work: Vec<(u32, u32, u32, u8)> = Vec::with_capacity(tri_idx.len() / 3);
    for tri in tri_idx.chunks_exact(3) {
        work.push((tri[0] as u32, tri[1] as u32, tri[2] as u32, 0));
    }

    let max_chord_rad = max_chord_deg.to_radians();
    let mut final_tris: Vec<(u32, u32, u32)> = Vec::with_capacity(work.len());
    // Longest-edge bisection halves triangle area per depth step (~sqrt(2) per
    // edge), so depth 14 reduces a 120 deg continental edge by ~128x (< 1 deg).
    let max_depth = 14u8;

    while let Some((i0, i1, i2, depth)) = work.pop() {
        // Measure each edge's true arc length via its 2D stereographic midpoint
        // so >180 deg chords inside >hemisphere caps always rank longest
        // instead of aliasing to their shorter exterior complement.
        let arc_score = |a: u32, b: u32| -> f64 {
            let pa = sub_2d[a as usize];
            let pb = sub_2d[b as usize];
            let m = stereo_unproject(
                [0.5 * (pa[0] + pb[0]), 0.5 * (pa[1] + pb[1])],
                center,
                ex,
                ey,
            );
            let ua = sub_units[a as usize];
            let ub = sub_units[b as usize];
            ua.dot(m).clamp(-1.0, 1.0).acos() + m.dot(ub).clamp(-1.0, 1.0).acos()
        };
        let s01 = arc_score(i0, i1);
        let s12 = arc_score(i1, i2);
        let s20 = arc_score(i2, i0);

        if s01.max(s12).max(s20) > max_chord_rad
            && depth < max_depth
            && final_tris.len() + work.len() < refine_budget
        {
            let mut get_mid = |a: u32, b: u32| -> u32 {
                let key = edge_key(a, b);
                *midpoint_cache.entry(key).or_insert_with(|| {
                    let pa = sub_2d[a as usize];
                    let pb = sub_2d[b as usize];
                    let mid_2d = [0.5 * (pa[0] + pb[0]), 0.5 * (pa[1] + pb[1])];
                    let mid_u = stereo_unproject(mid_2d, center, ex, ey);
                    let idx = sub_units.len() as u32;
                    sub_2d.push(mid_2d);
                    sub_units.push(if mid_u.length_squared() > 1e-12 {
                        mid_u
                    } else {
                        sub_units[a as usize]
                    });
                    idx
                })
            };

            if s01 >= s12 && s01 >= s20 {
                let m = get_mid(i0, i1);
                work.push((i0, m, i2, depth + 1));
                work.push((m, i1, i2, depth + 1));
            } else if s12 >= s01 && s12 >= s20 {
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

    // Read-only conforming midpoint closure sweep: eliminates all T-junction
    // cracks along edges split on one side before depth or budget cutoff.
    close_t_junctions(
        &mut final_tris,
        &sub_units,
        &midpoint_cache,
        total_remaining,
    );

    // Re-project every used vertex onto WGS84 with outward geodetic normals.
    let ecef_and_n: Vec<(DVec3, DVec3)> = sub_units.iter().map(|&u| unit_to_surface(u)).collect();

    for (i0, i1, i2) in final_tris {
        if raw.fill_tris.len() >= MAX_MESH_TRIANGLES {
            break;
        }
        let (p0, n0) = ecef_and_n[i0 as usize];
        let (p1, n1) = ecef_and_n[i1 as usize];
        let (p2, n2) = ecef_and_n[i2 as usize];
        let face_n = (p1 - p0).cross(p2 - p0);
        // Preserve drone-scale survey triangles down to ~1 mm^2 while dropping
        // collapsed zero-area slivers.
        if face_n.length_squared() <= 1e-12 {
            continue;
        }
        if face_n.dot(n0 + n1 + n2) >= 0.0 {
            raw.fill_tris.push([(p0, n0), (p1, n1), (p2, n2)]);
        } else {
            raw.fill_tris.push([(p0, n0), (p2, n2), (p1, n1)]);
        }
    }
}

#[inline]
fn edge_key(a: u32, b: u32) -> (u32, u32) {
    if a < b { (a, b) } else { (b, a) }
}

fn close_t_junctions(
    tris: &mut Vec<(u32, u32, u32)>,
    sub_units: &[DVec3],
    midpoint_cache: &HashMap<(u32, u32), u32>,
    max_tris: usize,
) {
    if midpoint_cache.is_empty() {
        return;
    }
    for _ in 0..16 {
        let mut next = Vec::with_capacity(tris.len());
        let mut split_any = false;
        for &(i0, i1, i2) in tris.iter() {
            if next.len() + 4 > max_tris {
                next.push((i0, i1, i2));
                continue;
            }
            let m01 = midpoint_cache.get(&edge_key(i0, i1)).copied();
            let m12 = midpoint_cache.get(&edge_key(i1, i2)).copied();
            let m20 = midpoint_cache.get(&edge_key(i2, i0)).copied();
            match (m01, m12, m20) {
                (None, None, None) => next.push((i0, i1, i2)),
                (Some(a), Some(b), Some(c)) => {
                    split_any = true;
                    next.push((i0, a, c));
                    next.push((a, i1, b));
                    next.push((c, b, i2));
                    next.push((a, b, c));
                }
                (Some(m), None, None) => {
                    split_any = true;
                    next.push((i0, m, i2));
                    next.push((m, i1, i2));
                }
                (None, Some(m), None) => {
                    split_any = true;
                    next.push((i0, i1, m));
                    next.push((i0, m, i2));
                }
                (None, None, Some(m)) => {
                    split_any = true;
                    next.push((i0, i1, m));
                    next.push((m, i1, i2));
                }
                (Some(a), Some(b), None) => {
                    split_any = true;
                    emit_two_edge_split(i1, i2, i0, a, b, sub_units, &mut next);
                }
                (None, Some(b), Some(c)) => {
                    split_any = true;
                    emit_two_edge_split(i2, i0, i1, b, c, sub_units, &mut next);
                }
                (Some(a), None, Some(c)) => {
                    split_any = true;
                    emit_two_edge_split(i0, i1, i2, c, a, sub_units, &mut next);
                }
            }
        }
        *tris = next;
        if !split_any {
            break;
        }
    }
}

/// Split triangle `(apex, v_next, v_prev)` where `m_in` bisects `(v_prev, apex)`
/// and `m_out` bisects `(apex, v_next)` into 3 triangles using the shorter
/// interior diagonal of the remaining quad `(m_in, m_out, v_next, v_prev)`.
fn emit_two_edge_split(
    apex: u32,
    v_next: u32,
    v_prev: u32,
    m_in: u32,
    m_out: u32,
    sub_units: &[DVec3],
    out: &mut Vec<(u32, u32, u32)>,
) {
    out.push((apex, m_out, m_in));
    let d_in_next = sub_units[m_in as usize].dot(sub_units[v_next as usize]);
    let d_out_prev = sub_units[m_out as usize].dot(sub_units[v_prev as usize]);
    // Larger dot product = shorter spherical diagonal.
    if d_in_next >= d_out_prev {
        out.push((m_in, m_out, v_next));
        out.push((m_in, v_next, v_prev));
    } else {
        out.push((m_in, m_out, v_prev));
        out.push((m_out, v_next, v_prev));
    }
}

fn unit_to_surface(u: DVec3) -> (DVec3, DVec3) {
    let (lat, lon) = geodesy::angles(u.normalize_or_zero());
    WGS84.surface_ecef_and_normal(lat, lon)
}
