//! Surface outlines on the WGS84 ellipsoid: plain lines, screen-space ribbons,
//! and hatched convex loading areas.
//!
//! All geometry sits directly on the ellipsoid (`height_m = 0.0`) because
//! draped primitives render depth-less with fragment ellipsoidal-normal
//! horizon discard.

use glam::DVec3;

use super::footprint::GeoPoint;
use super::geodesy::{self, Geodetic, World};
use super::sag;

/// Visual style of a surface outline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutlineStyle {
    Plain,
    Selected,
    Pending,
}

/// Geographic ring together with its display style.
#[derive(Clone, Debug, PartialEq)]
pub struct Outline {
    pub points: Vec<GeoPoint>,
    pub style: OutlineStyle,
}

/// High/low split WGS84 surface vertex with outward ellipsoidal normal.
///
/// GPU vertex-buffer packing (`#[repr(C)]` / `bytemuck`) stays in
/// `veldmap-app` so `veldmap-core` remains decoupled from GPU layouts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OutlineVertex {
    pub position: World,
    pub position_low: World,
    pub normal: World,
}

/// Ribbon vertex carrying neighbor offsets for screen-space widening.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RibbonVertex {
    pub position: World,
    pub position_low: World,
    pub normal: World,
    pub prev: World,
    pub next: World,
    pub side: f32,
}

fn surface_vertex(lat_deg: f64, lon_deg: f64) -> OutlineVertex {
    let (position, position_low) =
        geodesy::parts(geodesy::world(Geodetic::surface(lat_deg, lon_deg)));
    OutlineVertex {
        position,
        position_low,
        normal: geodesy::normal(lat_deg, lon_deg),
    }
}

/// Maximum angular step along a densified outline edge (shared with
/// [`super::footprint`] so hit-testing matches drawn outlines).
fn max_edge_deg() -> f64 {
    sag::outline_max_edge_deg()
}

/// Minimum cosine of reach from ring centroid for a fill fan (`cos 85 deg`).
const FILL_REACH: f64 = 0.087;

/// Tessellated GPU-ready buffers for plain outlines, selected ribbons, and
/// pending-scene hatching.
#[derive(Clone, Debug, Default)]
pub struct Outlines {
    pub vertices: Vec<OutlineVertex>,
    pub indices: Vec<u32>,
    pub ribbon: Vec<RibbonVertex>,
    pub ribbon_indices: Vec<u32>,
    pub hatch: Vec<OutlineVertex>,
    pub hatch_indices: Vec<u32>,
}

impl Outlines {
    /// Build tessellated outline geometry for all provided outlines.
    pub fn build(outlines: &[Outline]) -> Self {
        let mut built = Self::default();

        for outline in outlines {
            let loops = ring(outline);
            let drawn: Vec<&Vec<OutlineVertex>> =
                loops.iter().filter(|run| run.len() >= 3).collect();
            let whole = drawn.len() == 1;
            for ring_verts in drawn {
                match outline.style {
                    OutlineStyle::Selected => built.ribbon(ring_verts),
                    OutlineStyle::Plain => built.line(ring_verts),
                    OutlineStyle::Pending => {
                        built.line(ring_verts);
                        if whole {
                            built.fill(ring_verts);
                        }
                    }
                }
            }
        }

        built
    }

    fn line(&mut self, ring: &[OutlineVertex]) {
        let base = self.vertices.len() as u32;
        self.vertices.extend_from_slice(ring);

        let count = ring.len() as u32;
        for i in 0..count {
            self.indices
                .extend_from_slice(&[base + i, base + (i + 1) % count]);
        }
    }

    fn ribbon(&mut self, ring: &[OutlineVertex]) {
        let base = self.ribbon.len() as u32;
        let count = ring.len();
        for (index, vertex) in ring.iter().enumerate() {
            let here = (vertex.position, vertex.position_low);
            let at = |node: &OutlineVertex| (node.position, node.position_low);
            let prev = geodesy::offset(at(&ring[(index + count - 1) % count]), here);
            let next = geodesy::offset(at(&ring[(index + 1) % count]), here);
            for side in [-1.0_f32, 1.0_f32] {
                self.ribbon.push(RibbonVertex {
                    position: vertex.position,
                    position_low: vertex.position_low,
                    normal: vertex.normal,
                    prev,
                    next,
                    side,
                });
            }
        }

        for i in 0..count as u32 {
            let here = base + i * 2;
            let ahead = base + ((i + 1) % count as u32) * 2;
            self.ribbon_indices.extend_from_slice(&[
                here,
                here + 1,
                ahead,
                ahead,
                here + 1,
                ahead + 1,
            ]);
        }
    }

    fn fill(&mut self, ring: &[OutlineVertex]) {
        let direction = |vertex: &OutlineVertex| glam::Vec3::from(vertex.normal).as_dvec3();
        let Some(centre) = ring.iter().map(direction).sum::<DVec3>().try_normalize() else {
            return;
        };
        let reach = ring
            .iter()
            .map(direction)
            .map(|point| centre.dot(point))
            .fold(1.0, f64::min);
        if reach < FILL_REACH {
            return;
        }
        let steps = (reach.clamp(-1.0, 1.0).acos().to_degrees() / max_edge_deg())
            .ceil()
            .max(1.0) as u32;

        let base = self.hatch.len() as u32;
        self.hatch.push(surface_dir(centre));
        for step in 1..steps {
            let part = f64::from(step) / f64::from(steps);
            for point in ring.iter().map(direction) {
                self.hatch
                    .push(surface_dir(centre.lerp(point, part).normalize()));
            }
        }
        // Reuse exact f64-split outer ring vertices directly so the hatch
        // boundary aligns cleanly with the outline stroke (no f32 normal drift).
        self.hatch.extend_from_slice(ring);

        let count = ring.len() as u32;
        let ring_at = |level: u32| base + 1 + level * count;
        for index in 0..count {
            let (here, ahead) = (ring_at(0) + index, ring_at(0) + (index + 1) % count);
            self.hatch_indices.extend_from_slice(&[base, here, ahead]);
        }
        for level in 0..steps - 1 {
            let (inner, outer) = (ring_at(level), ring_at(level + 1));
            for index in 0..count {
                let ahead = (index + 1) % count;
                self.hatch_indices.extend_from_slice(&[
                    inner + index,
                    outer + index,
                    inner + ahead,
                    inner + ahead,
                    outer + index,
                    outer + ahead,
                ]);
            }
        }
    }
}

fn surface_dir(direction: DVec3) -> OutlineVertex {
    let (lat_deg, lon_deg) = geodesy::angles(direction);
    surface_vertex(lat_deg, lon_deg)
}

fn ring(outline: &Outline) -> Vec<Vec<OutlineVertex>> {
    let raw: Vec<(f64, f64)> = outline.points.iter().map(|p| (p.lat, p.lon)).collect();
    let places = geodesy::dedup_ring(&raw);
    if places.len() < 3 {
        return Vec::new();
    }

    let cuts = geodesy::band_seam_cuts(&places);
    let start = cuts.first().map_or(0, |cut| cut + 1);
    let (mut loops, mut current) = (Vec::new(), Vec::new());
    for step in 0..places.len() {
        let index = (start + step) % places.len();
        if cuts.contains(&index) {
            if !current.is_empty() {
                loops.push(std::mem::take(&mut current));
            }
            continue;
        }
        edge(
            &mut current,
            places[index],
            places[(index + 1) % places.len()],
        );
    }
    if !current.is_empty() {
        loops.push(current);
    }
    loops
}

fn edge(ring: &mut Vec<OutlineVertex>, from: (f64, f64), to: (f64, f64)) {
    if from == to {
        return;
    }
    geodesy::trace_edge(from, to, max_edge_deg(), |(lat_deg, lon_deg)| {
        ring.push(surface_vertex(lat_deg, lon_deg));
    });
}

#[cfg(test)]
#[path = "outlines_tests.rs"]
mod tests;
