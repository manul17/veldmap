//! Untextured WGS84 ellipsoid fallback mesh and depth-less draped 15-degree
//! graticule at `height_m = 0.0`.
//!
//! Uses a single watertight `96 x 192` WGS84 grid with `(pos_hi, pos_lo)`
//! double-single ECEF vertex splitting. Note that double-single RTE assumes
//! non-reassociating IEEE-754 float arithmetic and this coarse static mesh is
//! retired in Slice 2A in favor of per-tile RTC quadtree geometry.

use std::ops::Range;

use bytemuck::{Pod, Zeroable};
use veldmap_core::geo::geodesy::{self, Geodetic, World};

/// Vertex layout shared by the WGS84 ellipsoid body and draped graticule.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GlobeVertex {
    /// High half of the ECEF vertex position.
    pub pos_hi: World,
    /// Low half of the ECEF vertex position.
    pub pos_lo: World,
    /// Outward unit WGS84 ellipsoidal normal.
    pub normal: World,
}

impl GlobeVertex {
    pub const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &wgpu::vertex_attr_array![
            0 => Float32x3,
            1 => Float32x3,
            2 => Float32x3,
        ],
    };

    /// Construct a surface vertex at `(lat_deg, lon_deg)` on `height_m = 0.0`
    /// with ECEF coordinates split into `(high, low)` `f32` halves.
    pub fn on_surface(lat_deg: f64, lon_deg: f64) -> Self {
        let (pos_hi, pos_lo) = geodesy::parts(geodesy::world(Geodetic::surface(lat_deg, lon_deg)));
        Self {
            pos_hi,
            pos_lo,
            normal: geodesy::normal(lat_deg, lon_deg),
        }
    }
}

/// Complete untextured WGS84 ellipsoid plus draped graticule line segments on
/// `height_m = 0.0`.
#[derive(Clone, Debug)]
pub struct GlobeMesh {
    pub vertices: Vec<GlobeVertex>,
    pub indices: Vec<u32>,
    /// Triangle-list index range for the watertight WGS84 ellipsoid body.
    pub body: Range<u32>,
    /// Line-list index range for the draped 15-degree graticule.
    pub graticule: Range<u32>,
}

/// Global ellipsoid grid resolution (`96 x 192` quads = exact `1.875 deg` cells,
/// placing vertices cleanly on every 15-degree graticule intersection).
const LAT_CELLS: u32 = 96;
const LON_CELLS: u32 = 192;

/// Graticule spacing in degrees (every 15 degrees).
const GRID_STEP_DEG: i32 = 15;
/// Segments along each graticule meridian/parallel for smooth curvature.
const GRID_SEGMENTS: u32 = 672;

impl GlobeMesh {
    /// Build the watertight WGS84 ellipsoid and draped 15-degree graticule
    /// directly on `height_m = 0.0`.
    pub fn build() -> Self {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        let body = build_ellipsoid(&mut vertices, &mut indices);
        let grid_start = indices.len() as u32;
        build_graticule(&mut vertices, &mut indices);
        let graticule = grid_start..indices.len() as u32;

        Self {
            vertices,
            indices,
            body,
            graticule,
        }
    }
}

fn build_ellipsoid(vertices: &mut Vec<GlobeVertex>, indices: &mut Vec<u32>) -> Range<u32> {
    let base = vertices.len() as u32;
    let index_start = indices.len() as u32;
    let stride = LON_CELLS + 1;

    for row in 0..=LAT_CELLS {
        let t_lat = f64::from(row) / f64::from(LAT_CELLS);
        let lat = 90.0 - 180.0 * t_lat;
        for col in 0..=LON_CELLS {
            let t_lon = f64::from(col) / f64::from(LON_CELLS);
            let lon = -180.0 + 360.0 * t_lon;
            vertices.push(GlobeVertex::on_surface(lat, lon));
        }
    }

    for row in 0..LAT_CELLS {
        for col in 0..LON_CELLS {
            let top = base + row * stride + col;
            let bottom = top + stride;
            indices.extend_from_slice(&[top, bottom, top + 1]);
            indices.extend_from_slice(&[top + 1, bottom, bottom + 1]);
        }
    }

    index_start..indices.len() as u32
}

fn build_graticule(vertices: &mut Vec<GlobeVertex>, indices: &mut Vec<u32>) {
    let mut push_polyline = |points: Vec<GlobeVertex>| {
        let base = vertices.len() as u32;
        let count = points.len() as u32;
        vertices.extend(points);
        for i in 0..count.saturating_sub(1) {
            indices.extend_from_slice(&[base + i, base + i + 1]);
        }
    };

    let at = GlobeVertex::on_surface;

    for step in 0..(360 / GRID_STEP_DEG) {
        let lon = -180.0 + f64::from(step * GRID_STEP_DEG);
        push_polyline(
            (0..=GRID_SEGMENTS)
                .map(|i| at(-90.0 + 180.0 * f64::from(i) / f64::from(GRID_SEGMENTS), lon))
                .collect(),
        );
    }

    for step in (-90 + GRID_STEP_DEG..90).step_by(GRID_STEP_DEG as usize) {
        let lat = f64::from(step);
        push_polyline(
            (0..=GRID_SEGMENTS)
                .map(|i| {
                    at(
                        lat,
                        -180.0 + 360.0 * f64::from(i) / f64::from(GRID_SEGMENTS),
                    )
                })
                .collect(),
        );
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::*;

    #[test]
    fn high_low_split_preserves_sub_millimetre_precision_at_2m_altitude() {
        let eye = geodesy::world(Geodetic {
            lat_deg: 47.2357,
            lon_deg: 39.7015,
            height_m: 2.0,
        });
        let (neg_eye_hi, neg_eye_lo) = geodesy::parts(-eye);

        let p0 = Geodetic::surface(47.2357, 39.7015);
        let step_m = 0.01;
        let p1 = Geodetic::surface(47.2357 + step_m / 111_320.0, 39.7015);

        let v0 = GlobeVertex::on_surface(p0.lat_deg, p0.lon_deg);
        let v1 = GlobeVertex::on_surface(p1.lat_deg, p1.lon_deg);

        let rel = |v: GlobeVertex| {
            (Vec3::from(v.pos_hi) + Vec3::from(neg_eye_hi))
                + (Vec3::from(v.pos_lo) + Vec3::from(neg_eye_lo))
        };
        let d_gpu = f64::from((rel(v1) - rel(v0)).length()) * geodesy::SEMI_MAJOR_M;
        let d_exact = (geodesy::world(p1) - geodesy::world(p0)).length() * geodesy::SEMI_MAJOR_M;
        assert!(
            (d_gpu - d_exact).abs() < 1e-4,
            "1 cm step error at 2 m altitude: gpu={d_gpu} m vs exact={d_exact} m"
        );
    }

    #[test]
    fn globe_mesh_is_watertight_and_graticule_sits_on_ellipsoid() {
        let mesh = GlobeMesh::build();
        assert!(!mesh.body.is_empty());
        assert_eq!((mesh.body.end - mesh.body.start) % 3, 0);
        assert!(!mesh.graticule.is_empty());
        assert_eq!((mesh.graticule.end - mesh.graticule.start) % 2, 0);

        // Every index must point to a valid vertex.
        let vcount = mesh.vertices.len() as u32;
        assert!(mesh.indices.iter().all(|&i| i < vcount));

        // Antimeridian seam columns (-180 deg and +180 deg) must coincide on every row.
        let stride = (LON_CELLS + 1) as usize;
        for row in 0..=LAT_CELLS as usize {
            let west = mesh.vertices[row * stride];
            let east = mesh.vertices[row * stride + LON_CELLS as usize];
            let pw = Vec3::from(west.pos_hi).as_dvec3() + Vec3::from(west.pos_lo).as_dvec3();
            let pe = Vec3::from(east.pos_hi).as_dvec3() + Vec3::from(east.pos_lo).as_dvec3();
            let gap_m = (pw - pe).length() * geodesy::SEMI_MAJOR_M;
            assert!(
                gap_m < 1e-6,
                "antimeridian seam gap on row {row}: {gap_m} m"
            );
        }

        // Sample both meridian and parallel graticule vertices and verify zero surface lift.
        let grid_len = (mesh.graticule.end - mesh.graticule.start) as usize;
        for frac in [0, 1, 7, 15, 23, 30, 34] {
            let step = (grid_len * frac) / 35;
            let idx = mesh.indices[mesh.graticule.start as usize + step] as usize;
            let v = mesh.vertices[idx];
            let actual = Vec3::from(v.pos_hi).as_dvec3() + Vec3::from(v.pos_lo).as_dvec3();
            let (lat, lon) = geodesy::surface_at(actual);
            let expected = geodesy::world(Geodetic::surface(lat, lon));
            let err_m = (actual - expected).length() * geodesy::SEMI_MAJOR_M;
            assert!(
                err_m < 1e-3,
                "graticule vertex drifted {err_m} m from ellipsoid"
            );
        }
    }
}
