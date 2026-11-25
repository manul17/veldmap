//! Leaf vector mesh data types and intermediate `f64` ECEF primitive
//! accumulator shared by clustering, spherical triangulation, and top-level
//! GeoJSON tessellation.

use glam::DVec3;

use crate::geo::footprint::FootprintBounds;

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
    pub(super) fn remaining_tri_budget(&self) -> usize {
        MAX_MESH_TRIANGLES.saturating_sub(self.fill_tris.len())
    }

    pub(super) fn push_fill_triangle(&mut self, tri: [(DVec3, DVec3); 3]) -> bool {
        if self.fill_tris.len() >= MAX_MESH_TRIANGLES {
            return false;
        }
        self.fill_tris.push(tri);
        true
    }

    pub(super) fn push_line_segment(&mut self, a: (DVec3, DVec3), b: (DVec3, DVec3)) -> bool {
        if self.line_segs.len().saturating_mul(2) + 2 > MAX_LINE_VERTICES {
            return false;
        }
        self.line_segs.push([a, b]);
        true
    }

    pub(super) fn push_point(&mut self, p: (DVec3, DVec3)) -> bool {
        if self.points.len() >= MAX_POINT_VERTICES {
            return false;
        }
        self.points.push(p);
        true
    }
}
