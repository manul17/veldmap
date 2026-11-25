//! Interactive geodesic measurement domain state (`MeasureSession`): placed
//! vertices, cached WGS84 Karney metrics, RFC 7946 GeoJSON export, and Pass-2
//! draped [`DrawVectorLayer`] generation.
//!
//! Kept outside `ui/` so UI modules remain strictly presentation-only and
//! `veldmap-app` needs zero direct `serde_json` dependency.

use std::sync::Arc;

use veldmap_core::geo::{
    DEFAULT_VECTOR_MAX_CHORD_DEG, GeoJsonFeature, GeoJsonFeatureCollection, GeoJsonGeometry,
    GeoPoint, GeodesicPolygonMeasurement, Polygon, Ring, VectorMesh,
    export_geojson_feature_collection, polyline_length_m, ring_perimeter_and_area,
};

use crate::render::{DrawVectorId, DrawVectorLayer};

/// Active geodesic measurement mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum MeasureMode {
    #[default]
    Off,
    Distance,
    Area,
}

/// Cached exact WGS84 Karney metrics updated only when measurement vertices or
/// mode change (never recomputed on 60–120 Hz UI frames).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct MeasureMetrics {
    pub(crate) length_m: f64,
    pub(crate) polygon: GeodesicPolygonMeasurement,
}

impl Default for MeasureMetrics {
    fn default() -> Self {
        Self {
            length_m: 0.0,
            polygon: GeodesicPolygonMeasurement {
                perimeter_m: 0.0,
                area_m2: 0.0,
            },
        }
    }
}

/// Interactive geodesic measurement session owning vertices, cached Karney
/// metrics, and the derived Pass-2 [`DrawVectorLayer`].
pub(crate) struct MeasureSession {
    mode: MeasureMode,
    last_active_mode: MeasureMode,
    points: Vec<GeoPoint>,
    rev: u64,
    metrics: MeasureMetrics,
    cached_draw: Option<DrawVectorLayer>,
}

impl Default for MeasureSession {
    fn default() -> Self {
        Self {
            mode: MeasureMode::Off,
            last_active_mode: MeasureMode::Distance,
            points: Vec::new(),
            rev: 1,
            metrics: MeasureMetrics::default(),
            cached_draw: None,
        }
    }
}

impl MeasureSession {
    pub(crate) const fn is_active(&self) -> bool {
        !matches!(self.mode, MeasureMode::Off)
    }

    pub(crate) const fn mode(&self) -> MeasureMode {
        self.mode
    }

    pub(crate) fn points(&self) -> &[GeoPoint] {
        &self.points
    }

    pub(crate) const fn metrics(&self) -> MeasureMetrics {
        self.metrics
    }

    pub(crate) fn toggle(&mut self) {
        if self.is_active() {
            self.last_active_mode = self.mode;
            self.mode = MeasureMode::Off;
        } else {
            self.mode = match self.last_active_mode {
                MeasureMode::Area => MeasureMode::Area,
                _ => MeasureMode::Distance,
            };
        }
        self.rebuild_mesh();
    }

    pub(crate) fn set_mode(&mut self, mode: MeasureMode) {
        if self.mode == mode {
            return;
        }
        if mode != MeasureMode::Off {
            self.last_active_mode = mode;
        }
        self.mode = mode;
        self.rebuild_mesh();
    }

    pub(crate) fn add_point(&mut self, pt: GeoPoint) {
        if !(pt.lat.is_finite() && pt.lon.is_finite()) {
            return;
        }
        self.points.push(pt);
        self.rebuild_mesh();
    }

    pub(crate) fn undo(&mut self) {
        if self.points.pop().is_some() {
            self.rebuild_mesh();
        }
    }

    pub(crate) fn clear(&mut self) {
        self.points.clear();
        self.rebuild_mesh();
    }

    pub(crate) fn draw_layer(&self) -> Option<DrawVectorLayer> {
        self.cached_draw.clone()
    }

    /// Build an RFC 7946 [`GeoJsonFeatureCollection`] representing the active
    /// measurement feature along with exact Karney metric properties.
    fn to_geojson_collection(&self) -> Option<GeoJsonFeatureCollection> {
        if self.points.is_empty() {
            return None;
        }
        let feature = match self.mode {
            MeasureMode::Area if self.points.len() >= 3 => GeoJsonFeature::new(
                Some("VeldMap Measurement".to_owned()),
                GeoJsonGeometry::Polygon(Polygon::new(Ring {
                    points: self.points.clone(),
                })),
            )
            .with_string_prop("measurement_type", "area")
            .with_f64_prop("perimeter_m", self.metrics.polygon.perimeter_m)
            .with_f64_prop("area_m2", self.metrics.polygon.area_m2),
            _ if self.points.len() >= 2 => GeoJsonFeature::new(
                Some("VeldMap Measurement".to_owned()),
                GeoJsonGeometry::LineString(self.points.clone()),
            )
            .with_string_prop("measurement_type", "distance")
            .with_f64_prop("length_m", self.metrics.length_m),
            _ => GeoJsonFeature::new(
                Some("VeldMap Measurement".to_owned()),
                GeoJsonGeometry::Point(self.points[0]),
            )
            .with_string_prop("measurement_type", "point"),
        };

        Some(GeoJsonFeatureCollection {
            features: vec![feature],
        })
    }

    /// Serialize the current measurement feature as RFC 7946 GeoJSON.
    pub(crate) fn export_geojson_string(&self) -> Option<String> {
        self.to_geojson_collection()
            .as_ref()
            .map(export_geojson_feature_collection)
    }

    fn recompute_metrics(&mut self) {
        self.metrics = MeasureMetrics {
            length_m: polyline_length_m(&self.points),
            polygon: if self.points.len() >= 3 {
                ring_perimeter_and_area(&self.points)
            } else {
                GeodesicPolygonMeasurement {
                    perimeter_m: 0.0,
                    area_m2: 0.0,
                }
            },
        };
    }

    fn rebuild_mesh(&mut self) {
        self.rev = self.rev.saturating_add(1);
        self.recompute_metrics();
        if self.mode == MeasureMode::Off || self.points.is_empty() {
            self.cached_draw = None;
            return;
        }
        let Some(mut fc) = self.to_geojson_collection() else {
            self.cached_draw = None;
            return;
        };
        // Also emit explicit point markers for every placed vertex so the very
        // first click and all polyline/polygon vertices are clearly visible.
        if self.points.len() >= 2 {
            fc.features.push(GeoJsonFeature::new(
                None,
                GeoJsonGeometry::MultiPoint(self.points.clone()),
            ));
        }
        let mesh = Arc::new(VectorMesh::from_collection(
            &fc,
            DEFAULT_VECTOR_MAX_CHORD_DEG,
        ));
        self.cached_draw = Some(DrawVectorLayer {
            id: DrawVectorId::Measurement,
            rev: self.rev,
            opacity: 1.0,
            fill_color: [1.0, 0.72, 0.14, 0.25],
            line_color: [1.0, 0.85, 0.20, 0.98],
            mesh,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn measure_session_caches_karney_metrics_and_exports_rfc7946_geojson() {
        let mut s = MeasureSession::default();
        assert!(!s.is_active());
        s.toggle();
        assert_eq!(s.mode(), MeasureMode::Distance);

        s.add_point(GeoPoint::new(0.0, 0.0));
        s.add_point(GeoPoint::new(0.0, 1.0));
        assert!(s.metrics().length_m > 110_000.0 && s.metrics().length_m < 112_000.0);
        assert_eq!(
            s.draw_layer().expect("draw layer").id,
            DrawVectorId::Measurement
        );
        let dist_json = s.export_geojson_string().expect("distance geojson");
        assert!(dist_json.contains(r#""measurement_type": "distance""#));
        assert!(dist_json.contains(r#""length_m""#));

        s.set_mode(MeasureMode::Area);
        s.add_point(GeoPoint::new(1.0, 1.0));
        assert!(s.metrics().polygon.area_m2 > 6.0e9);
        let area_json = s.export_geojson_string().expect("area geojson");
        assert!(area_json.contains(r#""measurement_type": "area""#));
        assert!(area_json.contains(r#""area_m2""#));
        assert!(area_json.contains(r#""perimeter_m""#));

        // Toggling off and back on restores the last active mode (Area).
        s.toggle();
        assert!(!s.is_active());
        assert!(s.draw_layer().is_none());
        s.toggle();
        assert_eq!(s.mode(), MeasureMode::Area);
        assert!(s.draw_layer().is_some());

        s.undo();
        assert_eq!(s.points().len(), 2);
        s.clear();
        assert!(s.draw_layer().is_none());
    }
}
