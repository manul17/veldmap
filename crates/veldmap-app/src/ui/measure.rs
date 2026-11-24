//! Interactive WGS84 Karney geodesic Distance & Area measurement tool with
//! live Pass-2 vector preview and RFC 7946 GeoJSON export.

use std::sync::Arc;

use eframe::egui;
use serde_json::{Map, Value, json};
use veldmap_core::geo::{
    DEFAULT_VECTOR_MAX_CHORD_DEG, GeoJsonFeature, GeoJsonFeatureCollection, GeoJsonGeometry,
    GeoPoint, Polygon, Ring, VectorMesh, export_geojson_feature_collection, polyline_length_m,
    ring_perimeter_and_area,
};

use crate::render::DrawVectorLayer;

use super::hud;

/// Reserved vector-layer ID for the transient interactive measurement overlay.
const MEASURE_VECTOR_ID: u64 = u64::MAX;

/// Active geodesic measurement mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub(crate) enum MeasureMode {
    #[default]
    Off,
    Distance,
    Area,
}

/// Action emitted by the Measure tool window.
#[derive(Clone, Debug)]
pub(crate) enum MeasureAction {
    ExportGeoJson,
}

/// Interactive measurement vertices, cached Pass-2 vector mesh, and UI state.
#[derive(Clone, Debug)]
pub(crate) struct MeasureState {
    mode: MeasureMode,
    last_active_mode: MeasureMode,
    points: Vec<GeoPoint>,
    rev: u64,
    cached_draw: Option<DrawVectorLayer>,
    notice: Option<String>,
}

impl Default for MeasureState {
    fn default() -> Self {
        Self {
            mode: MeasureMode::Off,
            last_active_mode: MeasureMode::Distance,
            points: Vec::new(),
            rev: 1,
            cached_draw: None,
            notice: None,
        }
    }
}

impl MeasureState {
    pub(crate) fn is_active(&self) -> bool {
        self.mode != MeasureMode::Off
    }

    pub(crate) fn toggle_open(&mut self) {
        if self.mode == MeasureMode::Off {
            self.mode = match self.last_active_mode {
                MeasureMode::Area => MeasureMode::Area,
                _ => MeasureMode::Distance,
            };
        } else {
            self.last_active_mode = self.mode;
            self.mode = MeasureMode::Off;
        }
        self.rebuild_mesh();
    }

    pub(crate) fn set_mode(&mut self, mode: MeasureMode) {
        if self.mode != mode {
            if mode != MeasureMode::Off {
                self.last_active_mode = mode;
            }
            self.mode = mode;
            self.rebuild_mesh();
        }
    }

    pub(crate) fn add_point(&mut self, pt: GeoPoint) {
        self.points.push(pt);
        self.notice = None;
        self.rebuild_mesh();
    }

    pub(crate) fn undo(&mut self) {
        if self.points.pop().is_some() {
            self.notice = None;
            self.rebuild_mesh();
        }
    }

    pub(crate) fn clear(&mut self) {
        if !self.points.is_empty() {
            self.points.clear();
            self.notice = None;
            self.rebuild_mesh();
        }
    }

    pub(crate) fn set_notice(&mut self, msg: String) {
        self.notice = Some(msg);
    }

    /// Return the current Pass-2 [`DrawVectorLayer`] for rendering on the globe.
    pub(crate) fn draw_vector_layer(&self) -> Option<DrawVectorLayer> {
        self.cached_draw.clone()
    }

    /// Build an RFC 7946 [`GeoJsonFeatureCollection`] representing the active
    /// measurement feature along with exact Karney metric properties.
    pub(crate) fn to_geojson_collection(&self) -> Option<GeoJsonFeatureCollection> {
        if self.points.is_empty() {
            return None;
        }
        let mut props = Map::new();
        let geometry = match self.mode {
            MeasureMode::Area if self.points.len() >= 3 => {
                let m = ring_perimeter_and_area(&self.points);
                props.insert("measurement_type".into(), Value::String("area".into()));
                props.insert("perimeter_m".into(), json!(m.perimeter_m));
                props.insert("area_m2".into(), json!(m.area_m2));
                GeoJsonGeometry::Polygon(Polygon::new(Ring {
                    points: self.points.clone(),
                }))
            }
            _ if self.points.len() >= 2 => {
                let len = polyline_length_m(&self.points);
                props.insert("measurement_type".into(), Value::String("distance".into()));
                props.insert("length_m".into(), json!(len));
                GeoJsonGeometry::LineString(self.points.clone())
            }
            _ => {
                props.insert("measurement_type".into(), Value::String("point".into()));
                GeoJsonGeometry::Point(self.points[0])
            }
        };

        Some(GeoJsonFeatureCollection {
            features: vec![GeoJsonFeature {
                name: Some("VeldMap Measurement".to_owned()),
                properties: props,
                geometry,
            }],
        })
    }

    /// Serialize the current measurement feature as RFC 7946 GeoJSON.
    pub(crate) fn export_geojson_string(&self) -> Option<String> {
        self.to_geojson_collection()
            .as_ref()
            .map(export_geojson_feature_collection)
    }

    fn rebuild_mesh(&mut self) {
        self.rev = self.rev.saturating_add(1);
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
            fc.features.push(GeoJsonFeature {
                name: None,
                properties: Map::new(),
                geometry: GeoJsonGeometry::MultiPoint(self.points.clone()),
            });
        }
        let mesh = Arc::new(VectorMesh::from_collection(
            &fc,
            DEFAULT_VECTOR_MAX_CHORD_DEG,
        ));
        self.cached_draw = Some(DrawVectorLayer {
            id: MEASURE_VECTOR_ID,
            rev: self.rev,
            opacity: 1.0,
            fill_color: [1.0, 0.72, 0.14, 0.25],
            line_color: [1.0, 0.85, 0.20, 0.98],
            mesh,
        });
    }
}

/// Format area cleanly in m², ha, or km².
pub(crate) fn format_area_m2(area_m2: f64) -> String {
    if !area_m2.is_finite() || area_m2 <= 0.0 {
        "0 m²".to_owned()
    } else if area_m2 >= 1_000_000.0 {
        format!("{:.3} km²", area_m2 / 1_000_000.0)
    } else if area_m2 >= 10_000.0 {
        format!("{:.3} ha", area_m2 / 10_000.0)
    } else {
        format!("{area_m2:.1} m²")
    }
}

/// Render the floating Geodesic Measure window when active.
pub(crate) fn show_measure_window(
    ctx: &egui::Context,
    state: &mut MeasureState,
) -> Option<MeasureAction> {
    if !state.is_active() {
        return None;
    }
    let mut action = None;
    let mut open = true;
    egui::Window::new("Geodesic Measure (WGS84 Karney)")
        .open(&mut open)
        .resizable(false)
        .default_width(300.0)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui
                    .selectable_label(state.mode == MeasureMode::Distance, "Distance")
                    .clicked()
                {
                    state.set_mode(MeasureMode::Distance);
                }
                if ui
                    .selectable_label(state.mode == MeasureMode::Area, "Area")
                    .clicked()
                {
                    state.set_mode(MeasureMode::Area);
                }
            });
            ui.separator();
            ui.weak("Click on the globe to place vertices.");
            ui.monospace(format!("Vertices: {}", state.points.len()));

            match state.mode {
                MeasureMode::Distance => {
                    let dist = polyline_length_m(&state.points);
                    ui.strong(format!("Length: {}", hud::format_distance_m(dist)));
                }
                MeasureMode::Area => {
                    if state.points.len() >= 3 {
                        let m = ring_perimeter_and_area(&state.points);
                        ui.strong(format!("Area: {}", format_area_m2(m.area_m2)));
                        ui.monospace(format!(
                            "Perimeter: {}",
                            hud::format_distance_m(m.perimeter_m)
                        ));
                    } else {
                        let dist = polyline_length_m(&state.points);
                        ui.monospace(format!("Length: {}", hud::format_distance_m(dist)));
                        ui.weak("Add at least 3 points for area.");
                    }
                }
                MeasureMode::Off => {}
            }

            ui.add_space(6.0);
            ui.horizontal(|ui| {
                if ui
                    .add_enabled(!state.points.is_empty(), egui::Button::new("Undo"))
                    .clicked()
                {
                    state.undo();
                }
                if ui
                    .add_enabled(!state.points.is_empty(), egui::Button::new("Clear"))
                    .clicked()
                {
                    state.clear();
                }
                if ui
                    .add_enabled(
                        !state.points.is_empty(),
                        egui::Button::new("Export GeoJSON…"),
                    )
                    .clicked()
                {
                    action = Some(MeasureAction::ExportGeoJson);
                }
            });

            if let Some(note) = &state.notice {
                ui.separator();
                ui.weak(note);
            }
        });

    if !open {
        state.set_mode(MeasureMode::Off);
    }
    action
}
