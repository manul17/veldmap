//! Spatial RTC clustering and bit-exact indexed vertex encoding for
//! [`super::VectorMesh`].

use std::collections::{HashMap, HashSet};

use glam::{DVec3, Vec3};

use super::mesh::{RawPrimitives, VectorCluster, VectorVertex};

/// Maximum Euclidean radius (metres) for greedy spatial clustering of local
/// features (`<= 16 km` guarantees `< 1 mm` `f32` RTC quantization error).
const LOCAL_CLUSTER_RADIUS_M: f64 = 16_384.0;

/// Spatial cell size (metres) used to partition giant features (`> 16 km`
/// radius) into regional RTC clusters.
const REGIONAL_CELL_SIZE_M: f64 = 262_144.0;

/// Maximum number of RTC clusters per vector mesh before coarsening cell size.
const MAX_CLUSTERS: usize = 256;

#[inline]
fn cell_of(p: DVec3, inv_cell: f64) -> (i32, i32, i32) {
    (
        (p.x * inv_cell).floor() as i32,
        (p.y * inv_cell).floor() as i32,
        (p.z * inv_cell).floor() as i32,
    )
}

/// Choose the finest spatial cell size whose occupied cell count stays within
/// `MAX_CLUSTERS` (up to 4 coarsening steps), without copying primitives.
fn choose_cell_size(raw: &RawPrimitives) -> f64 {
    let mut cell_size = LOCAL_CLUSTER_RADIUS_M * 2.0;
    let mut seen: HashSet<(i32, i32, i32)> = HashSet::new();

    for attempt in 0..3 {
        seen.clear();
        let inv_cell = 1.0 / cell_size;
        let mut overflowed = false;

        for tri in &raw.fill_tris {
            let c = (tri[0].0 + tri[1].0 + tri[2].0) / 3.0;
            seen.insert(cell_of(c, inv_cell));
            if seen.len() > MAX_CLUSTERS {
                overflowed = true;
                break;
            }
        }
        if !overflowed {
            for seg in &raw.line_segs {
                let c = (seg[0].0 + seg[1].0) * 0.5;
                seen.insert(cell_of(c, inv_cell));
                if seen.len() > MAX_CLUSTERS {
                    overflowed = true;
                    break;
                }
            }
        }
        if !overflowed {
            for pt in &raw.points {
                seen.insert(cell_of(pt.0, inv_cell));
                if seen.len() > MAX_CLUSTERS {
                    overflowed = true;
                    break;
                }
            }
        }

        if !overflowed {
            break;
        }
        cell_size = if attempt == 0 {
            REGIONAL_CELL_SIZE_M
        } else {
            cell_size * 4.0
        };
    }

    cell_size
}

/// Partition `f64` ECEF primitives into spatial RTC clusters so every vertex's
/// `f32` relative coordinate stays accurate down to `MIN_HEIGHT_M = 2.0 m`
/// without blowing up draw-call count on continental polygons.
pub(super) fn build_rtc_clusters(raw: RawPrimitives) -> Vec<VectorCluster> {
    if raw.fill_tris.is_empty() && raw.line_segs.is_empty() && raw.points.is_empty() {
        return Vec::new();
    }

    let inv_cell = 1.0 / choose_cell_size(&raw);
    let mut buckets: HashMap<(i32, i32, i32), RawPrimitives> = HashMap::new();

    for tri in raw.fill_tris {
        let c = (tri[0].0 + tri[1].0 + tri[2].0) / 3.0;
        buckets
            .entry(cell_of(c, inv_cell))
            .or_default()
            .fill_tris
            .push(tri);
    }
    for seg in raw.line_segs {
        let c = (seg[0].0 + seg[1].0) * 0.5;
        buckets
            .entry(cell_of(c, inv_cell))
            .or_default()
            .line_segs
            .push(seg);
    }
    for pt in raw.points {
        buckets
            .entry(cell_of(pt.0, inv_cell))
            .or_default()
            .points
            .push(pt);
    }

    // Deterministic cluster ordering by cell key.
    let mut sorted_keys: Vec<(i32, i32, i32)> = buckets.keys().copied().collect();
    sorted_keys.sort_unstable();

    let mut clusters = Vec::with_capacity(sorted_keys.len());
    for key in sorted_keys {
        if let Some(bucket) = buckets.remove(&key) {
            let cluster = encode_cluster(bucket);
            if !cluster.is_empty() {
                clusters.push(cluster);
            }
        }
    }
    clusters
}

#[inline]
fn vertex_bits(v: VectorVertex) -> [u32; 6] {
    [
        v.rel_pos[0].to_bits(),
        v.rel_pos[1].to_bits(),
        v.rel_pos[2].to_bits(),
        v.normal[0].to_bits(),
        v.normal[1].to_bits(),
        v.normal[2].to_bits(),
    ]
}

fn encode_cluster(bucket: RawPrimitives) -> VectorCluster {
    let mut sum = DVec3::ZERO;
    let mut count = 0usize;
    for tri in &bucket.fill_tris {
        for (p, _) in tri {
            sum += *p;
            count += 1;
        }
    }
    for seg in &bucket.line_segs {
        for (p, _) in seg {
            sum += *p;
            count += 1;
        }
    }
    for (p, _) in &bucket.points {
        sum += *p;
        count += 1;
    }
    let origin_ecef_m = if count > 0 {
        sum / count as f64
    } else {
        DVec3::ZERO
    };

    let encode = |(p, n): (DVec3, DVec3)| -> VectorVertex {
        let rel = p - origin_ecef_m;
        VectorVertex {
            rel_pos: [rel.x as f32, rel.y as f32, rel.z as f32],
            normal: [n.x as f32, n.y as f32, n.z as f32],
        }
    };

    let mut fill_vertices = Vec::with_capacity(bucket.fill_tris.len());
    let mut fill_indices = Vec::with_capacity(bucket.fill_tris.len() * 3);
    let mut vertex_index: HashMap<[u32; 6], u32> = HashMap::with_capacity(bucket.fill_tris.len());

    let mut intern_fill = |v: VectorVertex| -> u32 {
        *vertex_index.entry(vertex_bits(v)).or_insert_with(|| {
            let idx = fill_vertices.len() as u32;
            fill_vertices.push(v);
            idx
        })
    };

    for tri in bucket.fill_tris {
        let v0 = encode(tri[0]);
        let v1 = encode(tri[1]);
        let v2 = encode(tri[2]);
        // Re-verify outward CCW winding in GPU-encoded `f32` RTC space so `f32`
        // quantization never flips a sliver triangle against backface culling.
        let p0 = Vec3::from_array(v0.rel_pos).as_dvec3();
        let p1 = Vec3::from_array(v1.rel_pos).as_dvec3();
        let p2 = Vec3::from_array(v2.rel_pos).as_dvec3();
        let avg_n = tri[0].1 + tri[1].1 + tri[2].1;
        let face_n = (p1 - p0).cross(p2 - p0);
        if face_n.length_squared() <= 1e-12 {
            continue;
        }
        let i0 = intern_fill(v0);
        let i1 = intern_fill(v1);
        let i2 = intern_fill(v2);
        if face_n.dot(avg_n) >= 0.0 {
            fill_indices.extend_from_slice(&[i0, i1, i2]);
        } else {
            fill_indices.extend_from_slice(&[i0, i2, i1]);
        }
    }

    let mut line_vertices = Vec::with_capacity(bucket.line_segs.len() * 2);
    for seg in bucket.line_segs {
        line_vertices.push(encode(seg[0]));
        line_vertices.push(encode(seg[1]));
    }

    let mut point_vertices = Vec::with_capacity(bucket.points.len());
    for pt in bucket.points {
        point_vertices.push(encode(pt));
    }

    VectorCluster {
        origin_ecef_m,
        fill_vertices,
        fill_indices,
        line_vertices,
        point_vertices,
    }
}
