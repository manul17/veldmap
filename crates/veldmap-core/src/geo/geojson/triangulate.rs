//! Spherical polygon triangulation via exterior-pole stereographic projection,
//! `earcut` triangulation, longest-edge subdivision, conforming T-junction
//! closure, and WGS84 surface projection with outward geodetic normals.

use std::collections::HashMap;

use earcut::Earcut;
use glam::DVec3;

use super::mesh::RawPrimitives;
use crate::geo::geodesy::{self, WGS84};

/// Orthonormal stereographic tangent frame centered at `center = -p_out`.
#[derive(Clone, Copy)]
struct StereoFrame {
    center: DVec3,
    ex: DVec3,
    ey: DVec3,
}

impl StereoFrame {
    /// Build a pole-safe orthonormal tangent basis `(ex, ey)` at unit vector
    /// `center` (`ex x ey = center`, right-handed outward).
    fn new(center: DVec3) -> Self {
        let ref_axis = if center.z.abs() < 0.9 {
            DVec3::Z
        } else {
            DVec3::X
        };
        let ex = ref_axis.cross(center).normalize();
        let ey = center.cross(ex).normalize();
        Self { center, ex, ey }
    }

    /// Conformal stereographic projection onto the plane tangent at `center`.
    fn project(self, u: DVec3) -> [f64; 2] {
        let denom = (1.0 + u.dot(self.center)).max(1e-12);
        let scale = 2.0 / denom;
        [u.dot(self.ex) * scale, u.dot(self.ey) * scale]
    }

    /// Exact inverse of [`Self::project`] mapping 2D stereographic coordinates
    /// back onto the unit sphere.
    fn unproject(self, p: [f64; 2]) -> DVec3 {
        let r2 = p[0] * p[0] + p[1] * p[1];
        let inv = 1.0 / (4.0 + r2);
        (self.ex * (4.0 * p[0] * inv)
            + self.ey * (4.0 * p[1] * inv)
            + self.center * ((4.0 - r2) * inv))
            .normalize_or_zero()
    }
}

struct ProjectedRings {
    coords_2d: Vec<[f64; 2]>,
    units: Vec<DVec3>,
    hole_indices: Vec<usize>,
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
        let frame = StereoFrame::new(-p_out);
        scratch.clear();
        for &u in exterior {
            scratch.push(frame.project(u));
        }
        let area = shoelace_area_2d(&scratch);
        // For CCW RFC 7946 exterior rings, `area > 0` iff `p_out` is exterior.
        if area > 0.0 {
            let clearance = -max_dot;
            if clearance > best_clearance {
                best_clearance = clearance;
                best_center = frame.center;
            }
        }
    }

    best_center
}

fn project_rings_to_stereo(
    exterior: &[DVec3],
    holes: &[Vec<DVec3>],
    frame: StereoFrame,
) -> ProjectedRings {
    let total_verts = exterior.len() + holes.iter().map(Vec::len).sum::<usize>();
    let mut coords_2d = Vec::with_capacity(total_verts);
    let mut units = Vec::with_capacity(total_verts);
    let mut hole_indices = Vec::with_capacity(holes.len());

    for &u in exterior {
        coords_2d.push(frame.project(u));
        units.push(u);
    }
    for hole in holes {
        hole_indices.push(coords_2d.len());
        for &u in hole {
            coords_2d.push(frame.project(u));
            units.push(u);
        }
    }

    ProjectedRings {
        coords_2d,
        units,
        hole_indices,
    }
}

struct RefinedTriangles {
    triangles: Vec<(u32, u32, u32)>,
    midpoints: HashMap<(u32, u32), u32>,
}

fn refine_longest_edges(
    tri_idx: &[usize],
    sub_units: &mut Vec<DVec3>,
    sub_2d: &mut Vec<[f64; 2]>,
    frame: StereoFrame,
    max_chord_deg: f64,
    refine_budget: usize,
) -> RefinedTriangles {
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
            let m = frame.unproject([0.5 * (pa[0] + pb[0]), 0.5 * (pa[1] + pb[1])]);
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
                    let mid_u = frame.unproject(mid_2d);
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

    RefinedTriangles {
        triangles: final_tris,
        midpoints: midpoint_cache,
    }
}

fn emit_wgs84_triangles(
    final_tris: &[(u32, u32, u32)],
    sub_units: &[DVec3],
    raw: &mut RawPrimitives,
) {
    // Re-project every used vertex onto WGS84 with outward geodetic normals.
    let ecef_and_n: Vec<(DVec3, DVec3)> = sub_units.iter().map(|&u| unit_to_surface(u)).collect();

    for &(i0, i1, i2) in final_tris {
        let (p0, n0) = ecef_and_n[i0 as usize];
        let (p1, n1) = ecef_and_n[i1 as usize];
        let (p2, n2) = ecef_and_n[i2 as usize];
        let face_n = (p1 - p0).cross(p2 - p0);
        // Preserve drone-scale survey triangles down to ~1 mm^2 while dropping
        // collapsed zero-area slivers.
        if face_n.length_squared() <= 1e-12 {
            continue;
        }
        let tri = if face_n.dot(n0 + n1 + n2) >= 0.0 {
            [(p0, n0), (p1, n1), (p2, n2)]
        } else {
            [(p0, n0), (p2, n2), (p1, n1)]
        };
        if !raw.push_fill_triangle(tri) {
            break;
        }
    }
}

/// Triangulate a densified spherical polygon (with optional interior holes),
/// subdivide long interior edges, close T-junctions, and emit outward-facing
/// WGS84 surface triangles into `raw`.
pub(super) fn triangulate_spherical_polygon(
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

    let frame = StereoFrame::new(choose_stereographic_center(exterior, holes));
    let mut projected = project_rings_to_stereo(exterior, holes, frame);

    let mut tri_idx: Vec<usize> = Vec::new();
    earcut.earcut(
        projected.coords_2d.iter().copied(),
        &projected.hole_indices,
        &mut tri_idx,
    );
    if tri_idx.len() < 3 {
        return;
    }

    // Reserve 15% of remaining triangle budget for the conforming midpoint
    // closure sweep so asymmetric depth/budget stops never leave T-junctions.
    let refine_budget = ((total_remaining as f64) * 0.85).floor().max(1.0) as usize;
    let mut refined = refine_longest_edges(
        &tri_idx,
        &mut projected.units,
        &mut projected.coords_2d,
        frame,
        max_chord_deg,
        refine_budget,
    );

    // Read-only conforming midpoint closure sweep: eliminates all T-junction
    // cracks along edges split on one side before depth or budget cutoff.
    close_t_junctions(
        &mut refined.triangles,
        &projected.units,
        &refined.midpoints,
        total_remaining,
    );

    emit_wgs84_triangles(&refined.triangles, &projected.units, raw);
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
