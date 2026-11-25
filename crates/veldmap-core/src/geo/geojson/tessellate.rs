//! Top-level WGS84 globe vector mesh tessellation combining geodesic line/ring
//! densification, spherical polygon triangulation, and per-cluster `f64` RTC
//! encoding.

use earcut::Earcut;
use glam::DVec3;

use super::cluster::build_rtc_clusters;
use super::mesh::{DEFAULT_VECTOR_MAX_CHORD_DEG, RawPrimitives, VectorMesh};
use super::triangulate::triangulate_spherical_polygon;
use super::{GeoJsonFeatureCollection, GeoJsonGeometry};
use crate::geo::geodesic::sample_geodesic_arc;
use crate::geo::geodesy::{self, WGS84};
use crate::geo::shapes::{
    GeoPoint, Polygon, dedup_geo_ring, push_dedup_geo_point, trim_closing_geo_point,
};

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
            push_dedup_geo_point(&mut out, p);
        }
    }
    trim_closing_geo_point(&mut out);
    out
}
