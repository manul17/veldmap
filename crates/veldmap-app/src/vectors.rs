//! Read-only GeoJSON vector overlay subsystem ([`VectorManager`]), off-UI-thread
//! file loader, and dedicated background I/O worker decoupled from raster
//! `OverlayManager` and `crate::storage::session`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use veldmap_core::geo::{
    DEFAULT_VECTOR_MAX_CHORD_DEG, FootprintBounds, MAX_GEOJSON_BYTES, VectorMesh, parse_geojson,
};
use veldmap_core::sync::{CancelToken, Wakeup};
use veldmap_storage::read_regular_file_utf8_limited;

use crate::render::{DrawVectorId, DrawVectorLayer};
use crate::worker::SingleThreadWorker;
use crate::{clamp_opacity, display_name_from_path};

/// Unique identifier for a loaded read-only GeoJSON vector layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct VectorLayerId(u64);

/// Read-only summary of a vector layer rendered in the Layer List UI.
#[derive(Clone, Debug)]
pub(crate) struct VectorLayerSummary {
    pub(crate) id: VectorLayerId,
    pub(crate) name: String,
    pub(crate) path: PathBuf,
    pub(crate) visible: bool,
    pub(crate) opacity: f32,
    pub(crate) feature_count: usize,
    pub(crate) triangle_count: usize,
    pub(crate) line_segment_count: usize,
    pub(crate) point_count: usize,
    pub(crate) footprint: Option<FootprintBounds>,
    pub(crate) color_rgb: [u8; 3],
}

/// Borrowed persistence view of a loaded vector overlay layer for
/// `session.json` serialization without coupling `vectors` to
/// `crate::storage::session`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct VectorPersistEntry<'a> {
    pub(crate) path: &'a Path,
    pub(crate) visible: bool,
    pub(crate) opacity: f32,
}

/// Asynchronous result of a background local GeoJSON open operation.
#[derive(Clone, Debug)]
pub(crate) enum VectorIoEvent {
    Opened {
        id: VectorLayerId,
        path: PathBuf,
        bounds: Option<FootprintBounds>,
        fly_to: bool,
    },
    Failed {
        path: PathBuf,
        error: String,
    },
}

/// Check whether `path` has a GeoJSON extension (`.geojson` or `.json`).
pub(crate) fn is_geojson_path(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("geojson") || e.eq_ignore_ascii_case("json"))
}

/// Pre-parsed and CPU-tessellated GeoJSON payload produced off the UI thread.
struct LoadedVectorData {
    path: PathBuf,
    feature_count: usize,
    mesh: Arc<VectorMesh>,
}

/// Read, validate, parse, and tessellate a local GeoJSON file off the UI thread.
fn load_vector_file(path: &Path) -> Result<LoadedVectorData, String> {
    let text = read_regular_file_utf8_limited(path, MAX_GEOJSON_BYTES as u64)
        .map_err(|e| format!("failed to read '{}': {e}", path.display()))?;
    let collection = parse_geojson(&text)
        .map_err(|e| format!("failed to parse GeoJSON '{}': {e}", path.display()))?;
    let feature_count = collection.features.len();
    let mesh = Arc::new(VectorMesh::from_collection(
        &collection,
        DEFAULT_VECTOR_MAX_CHORD_DEG,
    ));
    if mesh.is_empty() {
        return Err(format!(
            "'{}' contains no renderable GeoJSON geometries",
            path.display()
        ));
    }
    Ok(LoadedVectorData {
        path: path.to_path_buf(),
        feature_count,
        mesh,
    })
}

struct VectorIoCmd {
    path: PathBuf,
    fly_to: bool,
    visible: bool,
    opacity: f32,
}

struct VectorIoResult {
    path: PathBuf,
    fly_to: bool,
    visible: bool,
    opacity: f32,
    outcome: Result<LoadedVectorData, String>,
}

/// Dedicated single-thread background worker loading and tessellating local
/// GeoJSON files strictly off the UI thread, independent of raster
/// `OverlayManager` / `OverlayIoWorker`.
struct VectorIoWorker {
    worker: SingleThreadWorker<VectorIoCmd, VectorIoResult>,
}

impl VectorIoWorker {
    fn spawn(wakeup: Wakeup, shutdown: CancelToken) -> Self {
        let worker = SingleThreadWorker::<VectorIoCmd, VectorIoResult>::spawn(
            "veldmap-vector-io",
            move |cmd_rx, res_tx| {
                while let Ok(cmd) = cmd_rx.recv() {
                    if shutdown.is_cancelled() {
                        break;
                    }
                    let outcome = load_vector_file(&cmd.path);
                    if shutdown.is_cancelled() {
                        break;
                    }
                    let _ = res_tx.send(VectorIoResult {
                        path: cmd.path,
                        fly_to: cmd.fly_to,
                        visible: cmd.visible,
                        opacity: cmd.opacity,
                        outcome,
                    });
                    (wakeup)();
                }
            },
        );
        Self { worker }
    }

    fn begin_open(
        &self,
        path: &Path,
        fly_to: bool,
        visible: bool,
        opacity: f32,
    ) -> Result<(), String> {
        let cmd = VectorIoCmd {
            path: path.to_path_buf(),
            fly_to,
            visible,
            opacity: clamp_opacity(opacity),
        };
        if self.worker.send(cmd) {
            Ok(())
        } else {
            Err("vector I/O worker is unavailable".to_owned())
        }
    }

    fn try_recv(&self) -> Option<VectorIoResult> {
        self.worker.try_recv()
    }

    fn close_channel(&mut self) {
        self.worker.close_channel();
    }

    fn join(&mut self) {
        self.worker.join();
    }
}

struct VectorLayer {
    id: VectorLayerId,
    name: String,
    path: PathBuf,
    visible: bool,
    opacity: f32,
    fill_color: [f32; 4],
    line_color: [f32; 4],
    color_rgb: [u8; 3],
    feature_count: usize,
    mesh: Arc<VectorMesh>,
}

const VECTOR_PALETTE: [([f32; 4], [f32; 4], [u8; 3]); 6] = [
    (
        [0.12, 0.56, 0.92, 0.28],
        [0.28, 0.78, 1.0, 0.95],
        [72, 198, 255],
    ),
    (
        [0.96, 0.58, 0.12, 0.28],
        [1.0, 0.74, 0.25, 0.95],
        [255, 189, 64],
    ),
    (
        [0.16, 0.78, 0.45, 0.28],
        [0.30, 0.94, 0.58, 0.95],
        [76, 240, 148],
    ),
    (
        [0.88, 0.26, 0.54, 0.28],
        [1.0, 0.45, 0.70, 0.95],
        [255, 115, 178],
    ),
    (
        [0.62, 0.40, 0.95, 0.28],
        [0.78, 0.60, 1.0, 0.95],
        [199, 153, 255],
    ),
    (
        [0.92, 0.86, 0.18, 0.28],
        [1.0, 0.95, 0.40, 0.95],
        [255, 242, 102],
    ),
];

/// Ordered stack of read-only GeoJSON vector overlays.
struct VectorLayerStack {
    next_id: u64,
    layers: Vec<VectorLayer>,
}

impl Default for VectorLayerStack {
    fn default() -> Self {
        Self {
            next_id: 1,
            layers: Vec::new(),
        }
    }
}

impl VectorLayerStack {
    fn push_loaded(
        &mut self,
        data: LoadedVectorData,
        visible: bool,
        opacity: f32,
    ) -> (VectorLayerId, Option<FootprintBounds>) {
        let bounds = data.mesh.bounds;
        let id = VectorLayerId(self.next_id);
        self.next_id = self.next_id.saturating_add(1);
        let palette_idx = id.0.saturating_sub(1) as usize % VECTOR_PALETTE.len();
        let (fill_color, line_color, color_rgb) = VECTOR_PALETTE[palette_idx];
        let name = display_name_from_path(&data.path);

        self.layers.push(VectorLayer {
            id,
            name,
            path: data.path,
            visible,
            opacity: clamp_opacity(opacity),
            fill_color,
            line_color,
            color_rgb,
            feature_count: data.feature_count,
            mesh: data.mesh,
        });
        (id, bounds)
    }

    fn set_visible(&mut self, id: VectorLayerId, visible: bool) -> bool {
        let Some(layer) = self.layers.iter_mut().find(|l| l.id == id) else {
            return false;
        };
        if layer.visible == visible {
            return false;
        }
        layer.visible = visible;
        true
    }

    fn set_opacity(&mut self, id: VectorLayerId, opacity: f32) -> bool {
        let clamped = clamp_opacity(opacity);
        let Some(layer) = self.layers.iter_mut().find(|l| l.id == id) else {
            return false;
        };
        if (layer.opacity - clamped).abs() <= f32::EPSILON {
            return false;
        }
        layer.opacity = clamped;
        true
    }

    fn remove(&mut self, id: VectorLayerId) -> bool {
        let Some(idx) = self.layers.iter().position(|l| l.id == id) else {
            return false;
        };
        self.layers.remove(idx);
        true
    }

    fn len(&self) -> usize {
        self.layers.len()
    }

    fn summaries(&self) -> Vec<VectorLayerSummary> {
        self.layers
            .iter()
            .map(|l| VectorLayerSummary {
                id: l.id,
                name: l.name.clone(),
                path: l.path.clone(),
                visible: l.visible,
                opacity: l.opacity,
                feature_count: l.feature_count,
                triangle_count: l.mesh.fill_triangle_count(),
                line_segment_count: l.mesh.line_segment_count(),
                point_count: l.mesh.point_count(),
                footprint: l.mesh.bounds,
                color_rgb: l.color_rgb,
            })
            .collect()
    }

    fn draw_layers(&self) -> Vec<DrawVectorLayer> {
        self.layers
            .iter()
            .filter(|l| l.visible && l.opacity > 0.001)
            .map(|l| DrawVectorLayer {
                id: DrawVectorId::Layer(l.id.0),
                rev: 1,
                opacity: l.opacity,
                fill_color: l.fill_color,
                line_color: l.line_color,
                mesh: Arc::clone(&l.mesh),
            })
            .collect()
    }

    fn saved_entries_iter(&self) -> impl Iterator<Item = VectorPersistEntry<'_>> {
        self.layers.iter().map(|l| VectorPersistEntry {
            path: &l.path,
            visible: l.visible,
            opacity: l.opacity,
        })
    }
}

/// Facade coordinating read-only GeoJSON vector overlays and their dedicated
/// off-UI-thread I/O worker.
pub(crate) struct VectorManager {
    stack: VectorLayerStack,
    io: VectorIoWorker,
}

impl VectorManager {
    pub(crate) fn new(wakeup: Wakeup, shutdown: CancelToken) -> Self {
        Self {
            stack: VectorLayerStack::default(),
            io: VectorIoWorker::spawn(wakeup, shutdown),
        }
    }

    /// Queue a non-blocking local GeoJSON file open onto the background worker.
    pub(crate) fn begin_open(
        &self,
        path: &Path,
        fly_to: bool,
        visible: bool,
        opacity: f32,
    ) -> Result<(), String> {
        self.io.begin_open(path, fly_to, visible, opacity)
    }

    /// Drain completed background GeoJSON open events into the vector stack.
    pub(crate) fn poll_events(&mut self) -> Vec<VectorIoEvent> {
        let mut events = Vec::new();
        while let Some(res) = self.io.try_recv() {
            match res.outcome {
                Ok(data) => {
                    let path = data.path.clone();
                    let (id, bounds) = self.stack.push_loaded(data, res.visible, res.opacity);
                    events.push(VectorIoEvent::Opened {
                        id,
                        path,
                        bounds,
                        fly_to: res.fly_to,
                    });
                }
                Err(error) => {
                    events.push(VectorIoEvent::Failed {
                        path: res.path,
                        error,
                    });
                }
            }
        }
        events
    }

    pub(crate) fn len(&self) -> usize {
        self.stack.len()
    }

    pub(crate) fn summaries(&self) -> Vec<VectorLayerSummary> {
        self.stack.summaries()
    }

    pub(crate) fn saved_entries_iter(&self) -> impl Iterator<Item = VectorPersistEntry<'_>> {
        self.stack.saved_entries_iter()
    }

    pub(crate) fn set_visible(&mut self, id: VectorLayerId, visible: bool) -> bool {
        self.stack.set_visible(id, visible)
    }

    pub(crate) fn set_opacity(&mut self, id: VectorLayerId, opacity: f32) -> bool {
        self.stack.set_opacity(id, opacity)
    }

    pub(crate) fn remove(&mut self, id: VectorLayerId) -> bool {
        self.stack.remove(id)
    }

    pub(crate) fn draw_layers(&self) -> Vec<DrawVectorLayer> {
        self.stack.draw_layers()
    }

    /// Phase 1 shutdown: close the background worker command channel.
    pub(crate) fn begin_shutdown(&mut self) {
        self.io.close_channel();
    }

    /// Phase 3 shutdown: deterministically join the background worker thread.
    pub(crate) fn finish_shutdown(&mut self) {
        self.io.join();
    }
}

impl Drop for VectorManager {
    fn drop(&mut self) {
        self.begin_shutdown();
        self.finish_shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use veldmap_core::geo::{GeoJsonFeature, GeoJsonFeatureCollection, GeoJsonGeometry, GeoPoint};

    fn sample_loaded(name: &str) -> LoadedVectorData {
        let fc = GeoJsonFeatureCollection {
            features: vec![GeoJsonFeature::new(
                Some(name.to_owned()),
                GeoJsonGeometry::Point(GeoPoint::new(47.2, 39.7)),
            )],
        };
        LoadedVectorData {
            path: PathBuf::from(format!("/tmp/{name}.geojson")),
            feature_count: 1,
            mesh: Arc::new(VectorMesh::from_collection(
                &fc,
                DEFAULT_VECTOR_MAX_CHORD_DEG,
            )),
        }
    }

    #[test]
    fn is_geojson_path_matches_case_insensitively() {
        assert!(is_geojson_path(Path::new("parcels.geojson")));
        assert!(is_geojson_path(Path::new("PARCELS.GEOJSON")));
        assert!(is_geojson_path(Path::new("features.json")));
        assert!(!is_geojson_path(Path::new("ortho.tif")));
        assert!(!is_geojson_path(Path::new("no_extension")));
    }

    #[test]
    fn vector_layer_stack_handles_ordering_visibility_opacity_palette_and_remove() {
        let mut stack = VectorLayerStack::default();
        assert_eq!(stack.len(), 0);

        // Push enough layers to verify painter's order AND palette wrap-around.
        let mut ids = Vec::new();
        for idx in 0..=VECTOR_PALETTE.len() {
            let (id, bounds) = stack.push_loaded(sample_loaded(&format!("layer_{idx}")), true, 1.0);
            assert!(bounds.is_some());
            ids.push(id);
        }
        assert_eq!(stack.len(), VECTOR_PALETTE.len() + 1);

        let summaries = stack.summaries();
        assert_eq!(summaries[0].name, "layer_0.geojson");
        assert_eq!(summaries[0].color_rgb, VECTOR_PALETTE[0].2);
        assert_eq!(summaries[1].color_rgb, VECTOR_PALETTE[1].2);
        // Entry at `VECTOR_PALETTE.len()` wraps back to palette index 0.
        assert_eq!(
            summaries[VECTOR_PALETTE.len()].color_rgb,
            VECTOR_PALETTE[0].2
        );

        // Visibility transitions are idempotent and filter `draw_layers`.
        let first_id = ids[0];
        assert!(!stack.set_visible(first_id, true));
        assert!(stack.set_visible(first_id, false));
        assert!(!stack.set_visible(first_id, false));
        assert_eq!(stack.draw_layers().len(), VECTOR_PALETTE.len());

        // Opacity clamping & NaN guard; opacity <= 0.001 is skipped from draw_layers.
        let second_id = ids[1];
        assert!(stack.set_opacity(second_id, -2.0));
        assert!(!stack.set_opacity(second_id, 0.0));
        assert_eq!(stack.draw_layers().len(), VECTOR_PALETTE.len() - 1);
        assert!(stack.set_opacity(second_id, f32::NAN));
        let persisted: Vec<_> = stack.saved_entries_iter().collect();
        assert_eq!(persisted[1].opacity, 1.0);

        // Remove returns true once, false for missing IDs, preserving order of remaining layers.
        assert!(stack.remove(first_id));
        assert!(!stack.remove(first_id));
        assert_eq!(stack.len(), VECTOR_PALETTE.len());
        assert_eq!(stack.summaries()[0].id, second_id);
    }
}
