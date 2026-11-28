//! Canonical STAC Area of Interest (AOI) state, RFC 7946 bbox/geometry helpers,
//! GeoJSON/WKT export, and Pass-2 globe overlay mesh construction.

use std::sync::Arc;

use veldmap_core::geo::{
    DEFAULT_VECTOR_MAX_CHORD_DEG, FootprintBounds, GeoJsonFeature, GeoJsonFeatureCollection,
    GeoJsonGeometry, GeoPoint, Polygon, Ring, VectorMesh, export_geojson_geometry,
    export_wkt_geometry,
};

use crate::render::{DrawVectorId, DrawVectorLayer};

const STAC_AOI_FILL: [f32; 4] = [0.16, 0.88, 0.56, 0.14];
const STAC_AOI_LINE: [f32; 4] = [0.22, 0.96, 0.66, 0.96];

/// Maximum chord length (degrees) used for continental/global AOI and footprint
/// meshes so Earth-scale bboxes triangulate in ~1–2 ms on the UI thread instead
/// of hitting the 500k-triangle cap at 0.5 deg.
const MAX_AOI_CHORD_DEG: f64 = 6.0;

/// Compute adaptive chord length from the largest *individual* geometry's
/// angular span (rather than the whole collection's bounding sphere) so
/// globally scattered STAC footprints stay crisp at `0.5°` while single
/// continental/Earth-scale AOI polygons coarsen smoothly up to `6.0°`.
fn adaptive_aoi_chord_deg(geometries: &[GeoJsonGeometry]) -> f64 {
    let max_span_deg = geometries
        .iter()
        .filter_map(GeoJsonGeometry::bounds)
        .map(|b| (b.radius_deg * 2.0).max(0.0))
        .fold(0.0_f64, f64::max);
    if max_span_deg <= 0.0 {
        DEFAULT_VECTOR_MAX_CHORD_DEG
    } else {
        (max_span_deg / 20.0).clamp(DEFAULT_VECTOR_MAX_CHORD_DEG, MAX_AOI_CHORD_DEG)
    }
}

/// Build a CCW polar-cap circle along constant latitude `lat_deg` enclosing the
/// North Pole (`enclose_north = true`, ascending longitude) or South Pole
/// (`enclose_north = false`, descending longitude) without any interior
/// meridian stroke seam or polar hole.
fn polar_cap_circle_polygon(lat_deg: f64, enclose_north: bool, step_deg: f64) -> Polygon {
    let arc = 360.0 * lat_deg.to_radians().cos().abs();
    let steps = ((arc / step_deg.max(0.5)).ceil() as usize).clamp(24, 96);
    let mut points = Vec::with_capacity(steps + 1);
    for i in 0..=steps {
        let frac = i as f64 / steps as f64;
        let lon = if enclose_north {
            -180.0 + 360.0 * frac
        } else {
            180.0 - 360.0 * frac
        };
        points.push(GeoPoint::new(lat_deg, lon));
    }
    Polygon {
        exterior: Ring { points },
        holes: Vec::new(),
    }
}

/// Build a CCW rectangle whose South and North boundaries are sampled along
/// constant-latitude parallels using both true spherical arc length
/// (`lon_span * cos(lat)`) and a `<= 15°` longitude floor so zoomed-in polar
/// spans never collapse to zero-area 1D geodesics across the pole.
fn rect_polygon_parallels(west: f64, south: f64, east: f64, north: f64, step_deg: f64) -> Polygon {
    let lon_span = (east - west).abs();
    let step = step_deg.max(0.5);
    let parallel_steps = |lat_deg: f64| -> usize {
        if lat_deg.abs() >= 89.8 {
            // Avoid subdividing microscopic chords on near-pole boundaries (±89.8°..±90°).
            1
        } else {
            let min_steps = ((lon_span / 15.0).ceil() as usize).clamp(1, 48);
            let arc = lon_span * lat_deg.to_radians().cos().abs();
            ((arc / step).ceil() as usize).clamp(min_steps, 48)
        }
    };
    let south_steps = parallel_steps(south);
    let north_steps = parallel_steps(north);
    let mut points = Vec::with_capacity(south_steps + north_steps + 3);
    // South parallel: west -> east
    for i in 0..=south_steps {
        let lon = west + (east - west) * (i as f64 / south_steps as f64);
        points.push(GeoPoint::new(south, lon));
    }
    // North parallel: east -> west (meridians are already exact geodesics)
    for i in 0..=north_steps {
        let lon = east + (west - east) * (i as f64 / north_steps as f64);
        points.push(GeoPoint::new(north, lon));
    }
    points.push(GeoPoint::new(south, west));
    Polygon {
        exterior: Ring { points },
        holes: Vec::new(),
    }
}

/// Construct a render-ready constant-latitude parallel geometry for `bbox`.
///
/// - Polar caps (`[-180, south, 180, ~90]` or `[-180, ~-90, 180, north]`)
///   render as a single closed CCW parallel circle enclosing the pole (zero
///   interior meridian stroke seam, tessellated with strict CCW winding so
///   `> hemisphere` caps never invert onto the opposite pole).
/// - Full-circle non-polar bands (`> 350°`, including full Earth) split cleanly
///   into two non-self-touching halves.
/// - All other rectangles (including wide `179°..350°` spans) render as a
///   single parallel-sampled polygon with no interior seam.
pub(super) fn render_geometry_from_bbox(bbox: [f64; 4]) -> Option<GeoJsonGeometry> {
    if !bbox.iter().all(|v| v.is_finite()) {
        return None;
    }
    let west = bbox[0].clamp(-180.0, 180.0);
    let east = bbox[2].clamp(-180.0, 180.0);
    let south = bbox[1].min(bbox[3]).clamp(-89.9999, 89.9999);
    let north = bbox[1].max(bbox[3]).clamp(-89.9999, 89.9999);
    if (north - south) <= 1e-9 {
        return None;
    }

    let is_full_lon = west <= -179.999 && east >= 179.999;
    if is_full_lon && north >= 89.85 && south > -89.85 {
        let eff_span = (90.0 - south).max(360.0 * south.to_radians().cos().abs());
        let step_deg = (eff_span / 24.0).clamp(DEFAULT_VECTOR_MAX_CHORD_DEG, MAX_AOI_CHORD_DEG);
        return Some(GeoJsonGeometry::Polygon(polar_cap_circle_polygon(
            south, true, step_deg,
        )));
    }
    if is_full_lon && south <= -89.85 && north < 89.85 {
        let eff_span = (north + 90.0).max(360.0 * north.to_radians().cos().abs());
        let step_deg = (eff_span / 24.0).clamp(DEFAULT_VECTOR_MAX_CHORD_DEG, MAX_AOI_CHORD_DEG);
        return Some(GeoJsonGeometry::Polygon(polar_cap_circle_polygon(
            north, false, step_deg,
        )));
    }

    let mut intervals: Vec<(f64, f64)> = Vec::with_capacity(3);
    let push_split = |a: f64, b: f64, out: &mut Vec<(f64, f64)>| {
        let span = b - a;
        if span <= 1e-9 {
            return;
        }
        // Only split full-circle (`> 350°`) spans where West and East approach
        // the same meridian at ±180°, avoiding interior stroke lines on wide
        // regional/continental boxes.
        if span > 350.0 {
            let mid = 0.5 * (a + b);
            out.push((a, mid));
            out.push((mid, b));
        } else {
            out.push((a, b));
        }
    };

    if west <= east {
        push_split(west, east, &mut intervals);
    } else {
        push_split(west, 180.0, &mut intervals);
        push_split(-180.0, east, &mut intervals);
    }

    let total_lon_span: f64 = intervals.iter().map(|(a, b)| b - a).sum();
    let max_cos = south
        .to_radians()
        .cos()
        .abs()
        .max(north.to_radians().cos().abs());
    let eff_span = (north - south).max(total_lon_span * max_cos);
    let step_deg = (eff_span / 24.0).clamp(DEFAULT_VECTOR_MAX_CHORD_DEG, MAX_AOI_CHORD_DEG);
    let polys: Vec<Polygon> = intervals
        .into_iter()
        .map(|(a, b)| rect_polygon_parallels(a, south, b, north, step_deg))
        .collect();

    match polys.len() {
        0 => None,
        1 => polys.into_iter().next().map(GeoJsonGeometry::Polygon),
        _ => Some(GeoJsonGeometry::MultiPolygon(polys)),
    }
}

/// Construct an OGC-valid closed CCW RFC 7946 WGS84 polygon (or `MultiPolygon`
/// split strictly at `±180°` per RFC 7946 §3.1.9 when crossing the antimeridian
/// `west > east`) from `[west, south, east, north]`.
///
/// Parallels are sampled along constant latitude (`<= 15°` steps) so wide
/// exported GeoJSON/WKT polygons match the rendered parallel bounds without
/// OGC-invalid shared interior meridian edges.
pub(super) fn polygon_geometry_from_bbox(bbox: [f64; 4]) -> Option<GeoJsonGeometry> {
    if !bbox.iter().all(|v| v.is_finite()) {
        return None;
    }
    let west = bbox[0].clamp(-180.0, 180.0);
    let east = bbox[2].clamp(-180.0, 180.0);
    let south = bbox[1].min(bbox[3]).clamp(-89.9999, 89.9999);
    let north = bbox[1].max(bbox[3]).clamp(-89.9999, 89.9999);
    if (north - south) <= 1e-9 {
        return None;
    }

    const EXPORT_PARALLEL_STEP_DEG: f64 = 10.0;
    let mut polys: Vec<Polygon> = Vec::with_capacity(2);
    let mut push_rect = |a: f64, b: f64| {
        if (b - a) > 1e-9 {
            polys.push(rect_polygon_parallels(
                a,
                south,
                b,
                north,
                EXPORT_PARALLEL_STEP_DEG,
            ));
        }
    };

    if west <= east {
        push_rect(west, east);
    } else {
        push_rect(west, 180.0);
        push_rect(-180.0, east);
    }

    match polys.len() {
        0 => None,
        1 => polys.into_iter().next().map(GeoJsonGeometry::Polygon),
        _ => Some(GeoJsonGeometry::MultiPolygon(polys)),
    }
}

/// Compute a camera [`FootprintBounds`] covering a `[west, south, east, north]`
/// RFC 7946 bbox (including `west > east` antimeridian crossings).
pub(super) fn footprint_bounds_from_bbox(bbox: [f64; 4]) -> Option<FootprintBounds> {
    if !bbox.iter().all(|v| v.is_finite()) {
        return None;
    }
    let west = bbox[0].clamp(-180.0, 180.0);
    let east = bbox[2].clamp(-180.0, 180.0);
    let min_lat = bbox[1].min(bbox[3]).clamp(-90.0, 90.0);
    let max_lat = bbox[1].max(bbox[3]).clamp(-90.0, 90.0);
    let lat = 0.5 * (min_lat + max_lat);
    let (lon, lon_span) = if west <= east {
        (0.5 * (west + east), east - west)
    } else {
        let span = (east - west) + 360.0;
        let raw_mid = west + 0.5 * span;
        let wrapped = if raw_mid > 180.0 {
            raw_mid - 360.0
        } else {
            raw_mid
        };
        (wrapped, span)
    };
    let half_lat = 0.5 * (max_lat - min_lat).abs();
    let half_lon = 0.5 * lon_span.abs() * lat.to_radians().cos().abs().max(0.1);
    let radius_deg = half_lat.hypot(half_lon).max(0.01);
    Some(FootprintBounds {
        lat,
        lon,
        radius_deg,
    })
}

/// Tessellate `geometries` into a Pass-2 globe [`DrawVectorLayer`].
pub(super) fn build_vector_layer(
    id: DrawVectorId,
    rev: u64,
    geometries: Vec<GeoJsonGeometry>,
    fill_color: [f32; 4],
    line_color: [f32; 4],
    stroke_width_px: f32,
    hatch_spacing_px: f32,
) -> Option<DrawVectorLayer> {
    build_vector_layer_inner(
        id,
        rev,
        geometries,
        fill_color,
        line_color,
        stroke_width_px,
        hatch_spacing_px,
        false,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "Internal helper forwarding DrawVectorLayer style fields"
)]
fn build_vector_layer_inner(
    id: DrawVectorId,
    rev: u64,
    geometries: Vec<GeoJsonGeometry>,
    fill_color: [f32; 4],
    line_color: [f32; 4],
    stroke_width_px: f32,
    hatch_spacing_px: f32,
    strict_ccw: bool,
) -> Option<DrawVectorLayer> {
    if geometries.is_empty() {
        return None;
    }
    let chord_deg = adaptive_aoi_chord_deg(&geometries);
    let fc = GeoJsonFeatureCollection {
        features: geometries
            .into_iter()
            .map(|g| GeoJsonFeature::new(None, g))
            .collect(),
    };
    let mesh = if strict_ccw {
        VectorMesh::from_collection_strict_ccw(&fc, chord_deg)
    } else {
        VectorMesh::from_collection(&fc, chord_deg)
    };
    if mesh.is_empty() {
        return None;
    }
    Some(DrawVectorLayer {
        id,
        rev,
        opacity: 1.0,
        fill_color,
        line_color,
        stroke_width_px,
        hatch_spacing_px,
        mesh: Arc::new(mesh),
    })
}

/// Canonical Area of Interest state owned by [`super::StacSession`].
#[derive(Default)]
pub(super) struct AoiState {
    geometry: Option<GeoJsonGeometry>,
    /// Native `[west, south, east, north]` bbox when the AOI was set via a
    /// bounding box (sent directly as STAC `"bbox"` instead of `"intersects"`).
    query_bbox: Option<[f64; 4]>,
    display_bbox: Option<[f64; 4]>,
    label: Option<String>,
    draw: Option<DrawVectorLayer>,
    preview_draw: Option<DrawVectorLayer>,
}

impl AoiState {
    pub(super) fn has_geometry(&self) -> bool {
        self.geometry.is_some()
    }

    pub(super) fn geometry(&self) -> Option<&GeoJsonGeometry> {
        self.geometry.as_ref()
    }

    pub(super) const fn query_bbox(&self) -> Option<[f64; 4]> {
        self.query_bbox
    }

    pub(super) fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    /// Return the precomputed display bbox `[west, south, east, north]` in O(1)
    /// without cloning or scanning vertices on UI frames.
    pub(super) const fn bbox(&self) -> Option<[f64; 4]> {
        self.display_bbox
    }

    pub(super) fn export_geojson(&self) -> Option<String> {
        self.geometry.as_ref().map(export_geojson_geometry)
    }

    pub(super) fn export_wkt(&self) -> Option<String> {
        self.geometry.as_ref().map(export_wkt_geometry)
    }

    pub(super) fn set_bbox(&mut self, bbox: [f64; 4], label: String, rev: u64) -> bool {
        let Some(geom) = polygon_geometry_from_bbox(bbox) else {
            return false;
        };
        let sanitized_bbox = [
            bbox[0].clamp(-180.0, 180.0),
            bbox[1].min(bbox[3]).clamp(-90.0, 90.0),
            bbox[2].clamp(-180.0, 180.0),
            bbox[1].max(bbox[3]).clamp(-90.0, 90.0),
        ];
        let render_geom = render_geometry_from_bbox(sanitized_bbox).unwrap_or_else(|| geom.clone());
        self.preview_draw = None;
        self.query_bbox = Some(sanitized_bbox);
        self.display_bbox = Some(sanitized_bbox);
        self.geometry = Some(geom);
        self.label = Some(label);
        self.draw = build_vector_layer_inner(
            DrawVectorId::StacAoi,
            rev,
            vec![render_geom],
            STAC_AOI_FILL,
            STAC_AOI_LINE,
            3.0,
            10.0,
            true,
        );
        true
    }

    pub(super) fn set_geometry(&mut self, geometry: GeoJsonGeometry, label: String, rev: u64) {
        self.preview_draw = None;
        self.query_bbox = None;
        self.display_bbox = geometry.bbox();
        self.set_geometry_inner(geometry, label, rev);
    }

    fn set_geometry_inner(&mut self, geometry: GeoJsonGeometry, label: String, rev: u64) {
        self.geometry = Some(geometry.clone());
        self.label = Some(label);
        self.draw = build_vector_layer(
            DrawVectorId::StacAoi,
            rev,
            vec![geometry],
            STAC_AOI_FILL,
            STAC_AOI_LINE,
            3.0,
            10.0,
        );
    }

    /// Update the live drag preview rectangle without overwriting any existing
    /// committed AOI until the drag finishes.
    pub(super) fn set_preview_bbox(&mut self, bbox: [f64; 4], rev: u64) -> bool {
        let Some(render_geom) = render_geometry_from_bbox(bbox) else {
            return false;
        };
        self.preview_draw = build_vector_layer_inner(
            DrawVectorId::StacAoi,
            rev,
            vec![render_geom],
            STAC_AOI_FILL,
            STAC_AOI_LINE,
            3.0,
            10.0,
            true,
        );
        true
    }

    /// Clear any active AOI drag preview, restoring the committed AOI overlay.
    pub(super) fn clear_preview(&mut self) -> bool {
        if self.preview_draw.is_none() {
            return false;
        }
        self.preview_draw = None;
        true
    }

    pub(super) fn clear(&mut self) -> bool {
        if self.geometry.is_none() && self.preview_draw.is_none() {
            return false;
        }
        self.geometry = None;
        self.query_bbox = None;
        self.display_bbox = None;
        self.label = None;
        self.draw = None;
        self.preview_draw = None;
        true
    }

    pub(super) fn draw_layer(&self) -> Option<&DrawVectorLayer> {
        self.preview_draw.as_ref().or(self.draw.as_ref())
    }
}
