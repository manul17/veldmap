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
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OutlineVertex {
    pub position: World,
    pub position_low: World,
    pub normal: World,
}

/// Ribbon vertex carrying neighbor offsets for screen-space widening.
#[repr(C)]
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

/// Maximum angular step along a densified outline edge (half the sagitta limit
/// chord so radial fill fan diagonals stay tight to the ellipsoid).
fn max_edge_deg() -> f64 {
    sag::max_chord_deg(sag::DEFAULT_SAG_TOLERANCE_M) * 0.5
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
        for step in 1..=steps {
            let part = f64::from(step) / f64::from(steps);
            for point in ring.iter().map(direction) {
                self.hatch
                    .push(surface_dir(centre.lerp(point, part).normalize()));
            }
        }

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
    let points = match outline.points.as_slice() {
        [first, .., last] if first.lat == last.lat && first.lon == last.lon => {
            &outline.points[..outline.points.len() - 1]
        }
        all => all,
    };
    if points.len() < 3 {
        return Vec::new();
    }

    let at = |index: usize| {
        let point = &points[index % points.len()];
        (point.lat, point.lon)
    };
    let places: Vec<(f64, f64)> = (0..points.len()).map(at).collect();
    let sweeps: Vec<f64> = (0..points.len())
        .map(|index| geodesy::sweep(at(index), at(index + 1)))
        .collect();
    let cuts = seam_cuts(&places, &sweeps);

    let start = cuts.first().map_or(0, |cut| cut + 1);
    let (mut loops, mut current) = (Vec::new(), Vec::new());
    for step in 0..points.len() {
        let index = (start + step) % points.len();
        if cuts.contains(&index) {
            if !current.is_empty() {
                loops.push(std::mem::take(&mut current));
            }
            continue;
        }
        edge(&mut current, at(index), at(index + 1));
    }
    if !current.is_empty() {
        loops.push(current);
    }
    loops
}

fn seam_cuts(places: &[(f64, f64)], sweeps: &[f64]) -> Vec<usize> {
    let cuts: Vec<usize> = (0..sweeps.len())
        .filter(|index| sweeps[*index] == 0.0)
        .collect();
    if cuts.is_empty() {
        return Vec::new();
    }

    let mut carried = false;
    for pair in 0..cuts.len() {
        let (from, to) = (cuts[pair], cuts[(pair + 1) % cuts.len()]);
        let run: Vec<usize> = (1..sweeps.len())
            .map(|step| (from + step) % sweeps.len())
            .take_while(|index| *index != to)
            .collect();
        let Some((&first, &last)) = run.first().zip(run.last()) else {
            continue;
        };
        let (opens, closes) = (places[first], places[(last + 1) % places.len()]);
        if opens.0 != closes.0 || (opens.1 - closes.1).rem_euclid(360.0) != 0.0 {
            return Vec::new();
        }
        let span: f64 = run.iter().map(|index| sweeps[*index]).sum();
        if span.abs() >= geodesy::FULL_CIRCLE_DEG {
            carried = true;
        } else if span == 0.0 {
            continue;
        } else {
            return Vec::new();
        }
    }
    if carried { cuts } else { Vec::new() }
}

fn edge(ring: &mut Vec<OutlineVertex>, from: (f64, f64), to: (f64, f64)) {
    let edge_deg = max_edge_deg();
    let most = (geodesy::TURN_DEG / edge_deg).ceil();
    let raw = (geodesy::edge_span(from, to) / edge_deg).ceil();
    let steps = if raw.is_finite() {
        raw.clamp(1.0, most) as u32
    } else {
        1
    };
    for step in 0..steps {
        let (lat_deg, lon_deg) = geodesy::edge_point(from, to, f64::from(step) / f64::from(steps));
        ring.push(surface_vertex(lat_deg, lon_deg));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn outline(points: &[(f64, f64)]) -> Outline {
        Outline {
            points: points
                .iter()
                .map(|&(lat, lon)| GeoPoint { lat, lon })
                .collect(),
            style: OutlineStyle::Plain,
        }
    }

    fn styled(points: &[(f64, f64)], style: OutlineStyle) -> Outline {
        Outline {
            style,
            ..outline(points)
        }
    }

    fn angles_of(vertex: &OutlineVertex) -> (f64, f64) {
        geodesy::angles(glam::Vec3::from(vertex.normal).as_dvec3())
    }

    #[test]
    fn an_edge_wider_than_the_earth_is_not_drawn_forever() {
        let mut endless = Vec::new();
        edge(&mut endless, (0.0, 0.0), (1.0e9, 0.0));
        let most = (geodesy::TURN_DEG / max_edge_deg()).ceil() as usize;
        assert_eq!(endless.len(), most);

        let mut nowhere = Vec::new();
        edge(&mut nowhere, (0.0, 0.0), (f64::NAN, 0.0));
        assert_eq!(nowhere.len(), 1);
    }

    #[test]
    fn a_ribbon_carries_true_offsets_to_its_neighbours() {
        let built = Outlines::build(&[styled(
            &[(0.0, 0.0), (0.0, 1.0), (1.0, 0.5)],
            OutlineStyle::Selected,
        )]);
        let count = built.ribbon.len() / 2;
        assert!(count >= 3);

        let apart = |a: World, b: World| {
            f64::from((glam::Vec3::from(a) - glam::Vec3::from(b)).length()) * geodesy::SEMI_MAJOR_M
        };
        let at = |node: usize| {
            let v = built.ribbon[node * 2];
            (v.position, v.position_low)
        };
        for index in 0..count {
            let ahead = (index + 1) % count;
            let step = geodesy::offset(at(ahead), at(index));
            let span = f64::from(glam::Vec3::from(step).length()) * geodesy::SEMI_MAJOR_M;
            assert!(span > 1.0);
            assert!(apart(built.ribbon[index * 2].next, step) < 1e-3);
            let back = geodesy::offset(at(index), at(ahead));
            assert!(apart(built.ribbon[ahead * 2].prev, back) < 1e-3);
            assert_eq!(
                built.ribbon[index * 2].next,
                built.ribbon[index * 2 + 1].next
            );
        }
    }

    #[test]
    fn each_style_goes_to_its_own_draw() {
        let square = [(10.0, 10.0), (10.0, 12.0), (12.0, 12.0), (12.0, 10.0)];

        let plain = Outlines::build(&[outline(&square)]);
        assert!(!plain.indices.is_empty());
        assert!(plain.ribbon_indices.is_empty() && plain.hatch_indices.is_empty());

        let selected = Outlines::build(&[styled(&square, OutlineStyle::Selected)]);
        assert!(!selected.ribbon_indices.is_empty());
        assert!(selected.indices.is_empty() && selected.hatch_indices.is_empty());

        let pending = Outlines::build(&[styled(&square, OutlineStyle::Pending)]);
        assert_eq!(pending.indices, plain.indices);
        assert!(!pending.hatch_indices.is_empty());
        assert!(pending.ribbon_indices.is_empty());
    }

    #[test]
    fn the_filled_area_follows_the_surface() {
        let wide = [(0.0, 0.0), (0.0, 20.0), (20.0, 20.0), (20.0, 0.0)];
        let built = Outlines::build(&[styled(&wide, OutlineStyle::Pending)]);

        let between = |left: u32, right: u32| {
            let ends = [left, right].map(|index| {
                let (lat, lon) = angles_of(&built.hatch[index as usize]);
                geodesy::unit(lat, lon)
            });
            ends[0].dot(ends[1]).clamp(-1.0, 1.0).acos().to_degrees()
        };
        let mut widest = 0.0_f64;
        for triangle in built.hatch_indices.chunks(3) {
            widest = widest
                .max(between(triangle[0], triangle[1]))
                .max(between(triangle[1], triangle[2]))
                .max(between(triangle[2], triangle[0]));
        }

        let sag_m = sag::sag_m(widest);
        assert!(
            sag_m < sag::DEFAULT_SAG_TOLERANCE_M + 10.0,
            "widest triangle edge {widest} deg sags {sag_m} m"
        );
    }

    #[test]
    fn outlines_sit_directly_on_the_wgs84_ellipsoid() {
        let square = [(45.0, 39.0), (45.0, 40.0), (46.0, 40.0), (46.0, 39.0)];
        let built = Outlines::build(&[outline(&square)]);
        for v in &built.vertices {
            let (lat, lon) = angles_of(v);
            let expected = geodesy::world(Geodetic::surface(lat, lon));
            let actual = glam::Vec3::from(v.position).as_dvec3()
                + glam::Vec3::from(v.position_low).as_dvec3();
            let err_m = (actual - expected).length() * geodesy::SEMI_MAJOR_M;
            assert!(
                err_m < 1.0,
                "outline vertex drifted {err_m} m from ellipsoid"
            );
        }
    }

    #[test]
    fn a_ring_around_the_earth_gets_no_fill() {
        let band = styled(
            &[
                (-70.0125, -180.0125),
                (70.0125, -180.0125),
                (70.0125, 179.9875),
                (-70.0125, 179.9875),
            ],
            OutlineStyle::Pending,
        );
        let built = Outlines::build(&[band]);
        assert!(!built.indices.is_empty());
        assert!(built.hatch_indices.is_empty());
    }

    #[test]
    fn a_polar_cap_is_filled_all_the_same() {
        let cap: Vec<(f64, f64)> = (0..12)
            .map(|step| (85.0, f64::from(step) * 30.0 - 180.0))
            .collect();
        let built = Outlines::build(&[styled(&cap, OutlineStyle::Pending)]);
        assert!(!built.hatch_indices.is_empty());
    }

    #[test]
    fn a_degenerate_outline_draws_nothing() {
        let built =
            Outlines::build(&[styled(&[(10.0, 10.0), (10.0, 12.0)], OutlineStyle::Pending)]);
        assert!(built.indices.is_empty());
        assert!(built.hatch_indices.is_empty());
    }

    #[test]
    fn a_whole_earth_band_keeps_its_parallels() {
        let band = outline(&[
            (-70.0125, -180.0125),
            (70.0125, -180.0125),
            (70.0125, 179.9875),
            (-70.0125, 179.9875),
        ]);
        let loops = ring(&band);
        assert_eq!(loops.len(), 2);

        let all = loops.concat();
        let lons: Vec<f64> = all.iter().map(|v| angles_of(v).1).collect();
        let east = lons.iter().filter(|lon| **lon > 60.0).count();
        let west = lons.iter().filter(|lon| **lon < -60.0).count();
        assert!(east > 30 && west > 30);

        let stray = all
            .iter()
            .map(|v| angles_of(v).0)
            .filter(|lat| (-69.0..=69.0).contains(lat))
            .count();
        assert_eq!(stray, 0);

        for run in &loops {
            let lats: Vec<f64> = run.iter().map(|v| angles_of(v).0).collect();
            let spread = lats.iter().fold(f64::MIN, |a, b| a.max(*b))
                - lats.iter().fold(f64::MAX, |a, b| a.min(*b));
            assert!(spread < 0.1);
            let angles: Vec<(f64, f64)> = run.iter().map(angles_of).collect();
            assert!(geodesy::encircles(&angles));
        }
    }

    #[test]
    fn a_band_is_cut_the_same_however_densely_it_is_written() {
        let dense = outline(&[
            (-70.0125, -180.0125),
            (70.0125, -180.0125),
            (70.0125, -90.0),
            (70.0125, 0.0),
            (70.0125, 90.0),
            (70.0125, 179.9875),
            (-70.0125, 179.9875),
            (-70.0125, 90.0),
            (-70.0125, 0.0),
            (-70.0125, -90.0),
        ]);
        let loops = ring(&dense);
        assert_eq!(loops.len(), 2);
        let stray = loops
            .concat()
            .iter()
            .map(|v| angles_of(v).0)
            .filter(|lat| (-69.0..=69.0).contains(lat))
            .count();
        assert_eq!(stray, 0);
    }

    #[test]
    fn a_band_with_a_gap_keeps_its_edges() {
        let mut points = vec![(-70.0, 10.0), (70.0, 10.0)];
        for lon in [90.0, 170.0, -110.0, -30.0, 0.0] {
            points.push((70.0, lon));
        }
        points.push((-70.0, 0.0));
        for lon in [-30.0, -110.0, 170.0, 90.0] {
            points.push((-70.0, lon));
        }
        let gapped = outline(&points);

        let loops = ring(&gapped);
        assert_eq!(loops.len(), 1);
        let angles: Vec<(f64, f64)> = loops.concat().iter().map(angles_of).collect();
        assert!(geodesy::encircles(&angles));
    }

    #[test]
    fn a_band_is_cut_the_same_wherever_its_list_begins() {
        let corners = [
            (-70.0125, -180.0125),
            (70.0125, -180.0125),
            (70.0125, -90.0),
            (70.0125, 0.0),
            (70.0125, 90.0),
            (70.0125, 179.9875),
            (-70.0125, 179.9875),
            (-70.0125, 90.0),
            (-70.0125, 0.0),
            (-70.0125, -90.0),
        ];
        for turn in 0..corners.len() {
            let mut points = corners.to_vec();
            points.rotate_left(turn);
            let loops = ring(&outline(&points));
            assert_eq!(loops.len(), 2);
        }
    }

    #[test]
    fn a_run_that_does_not_close_is_no_seam() {
        let dented = outline(&[(70.0, 0.0), (70.0, 120.0), (70.0, -120.0), (60.0, -120.0)]);
        let loops = ring(&dented);
        assert_eq!(loops.len(), 1);

        let standing = loops
            .concat()
            .iter()
            .map(angles_of)
            .filter(|(lat, lon)| (60.5..69.5).contains(lat) && (lon + 120.0).abs() < 1e-6)
            .count();
        assert!(standing > 20);
    }

    #[test]
    fn a_repeated_corner_does_not_split_the_band() {
        let doubled = outline(&[
            (-70.0125, -180.0125),
            (70.0125, -180.0125),
            (70.0125, -180.0125),
            (70.0125, 179.9875),
            (-70.0125, 179.9875),
        ]);
        let loops = ring(&doubled);
        assert_eq!(loops.len(), 2);
        let stray = loops
            .concat()
            .iter()
            .map(|v| angles_of(v).0)
            .filter(|lat| (-69.0..=69.0).contains(lat))
            .count();
        assert_eq!(stray, 0);
    }

    #[test]
    fn a_gapped_band_is_whole_and_still_gets_no_fill() {
        let mut points = vec![(-70.0, 10.0), (70.0, 10.0)];
        for lon in [90.0, 170.0, -110.0, -30.0, 0.0] {
            points.push((70.0, lon));
        }
        points.push((-70.0, 0.0));
        for lon in [-30.0, -110.0, 170.0, 90.0] {
            points.push((-70.0, lon));
        }
        let gapped = styled(&points, OutlineStyle::Pending);
        assert_eq!(ring(&gapped).len(), 1);

        let built = Outlines::build(&[gapped]);
        assert!(!built.indices.is_empty());
        assert!(built.hatch_indices.is_empty());
    }

    #[test]
    fn a_cap_written_to_the_pole_keeps_its_fill() {
        let polar = styled(
            &[
                (60.0, -180.0125),
                (90.0, -180.0125),
                (90.0, 179.9875),
                (60.0, 179.9875),
            ],
            OutlineStyle::Pending,
        );
        let built = Outlines::build(&[polar]);
        assert!(!built.hatch_indices.is_empty());
    }

    #[test]
    fn a_loop_shrunk_to_a_point_draws_nothing() {
        let polar = outline(&[
            (60.0, -180.0125),
            (90.0, -180.0125),
            (90.0, 179.9875),
            (60.0, 179.9875),
        ]);
        let loops = ring(&polar);
        assert_eq!(loops.len(), 2);
        assert!(loops.iter().any(|run| run.len() < 3));

        let built = Outlines::build(&[polar]);
        let lats: Vec<f64> = built
            .vertices
            .iter()
            .map(angles_of)
            .map(|(lat, _)| lat)
            .collect();
        assert!(!lats.is_empty());
        assert!(lats.iter().all(|lat| *lat < 89.0));
    }

    #[test]
    fn a_box_across_the_seam_takes_the_short_way() {
        let box_ = outline(&[(65.0, 170.0), (65.0, -170.0), (60.0, -170.0), (60.0, 170.0)]);
        let loops = ring(&box_);
        assert_eq!(loops.len(), 1);
        let ring_verts = loops.concat();
        let inside = |lon: f64| !(-160.0..=160.0).contains(&lon);
        assert!(ring_verts.iter().all(|v| inside(angles_of(v).1)));

        let long_way = (340.0 / max_edge_deg()).ceil() as usize;
        assert!(ring_verts.len() < long_way / 3);
    }

    #[test]
    fn a_swath_edge_still_follows_the_arc() {
        let swath = outline(&[(73.0, 10.0), (73.4, 30.0), (72.0, 31.0), (71.6, 11.0)]);
        let loops = ring(&swath);
        assert_eq!(loops.len(), 1);
        let north = loops
            .concat()
            .iter()
            .map(angles_of)
            .filter(|(_, lon)| (19.0..21.0).contains(lon))
            .map(|(lat, _)| lat)
            .fold(f64::MIN, f64::max);
        assert!(north > 73.3);
    }
}
