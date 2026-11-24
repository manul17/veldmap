//! Read-only GeoJSON vector overlay layer stack (`VectorLayerStack`),
//! off-UI-thread GeoJSON file loader (`load_vector_file`), and dedicated
//! [`VectorIoWorker`] keeping vector I/O decoupled from raster `OverlayManager`.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::{self, JoinHandle};

use veldmap_core::geo::{
    DEFAULT_VECTOR_MAX_CHORD_DEG, FootprintBounds, MAX_GEOJSON_BYTES, VectorMesh, parse_geojson,
};
use veldmap_core::sync::{CancelToken, Wakeup};
use veldmap_storage::read_regular_file_utf8_limited;

use crate::render::{DrawVectorId, DrawVectorLayer};
use crate::session::SavedVectorEntry;

/// Unique identifier for a loaded read-only GeoJSON vector layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct VectorLayerId(pub(crate) u64);

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

/// Pre-parsed and CPU-tessellated GeoJSON payload produced off the UI thread.
pub(crate) struct LoadedVectorData {
    pub(crate) path: PathBuf,
    pub(crate) feature_count: usize,
    pub(crate) mesh: Arc<VectorMesh>,
}

/// Check whether `path` has a GeoJSON extension (`.geojson` or `.json`).
pub(crate) fn is_geojson_path(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("geojson") || e.eq_ignore_ascii_case("json"))
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
    cancel: CancelToken,
}

pub(crate) struct VectorIoResult {
    pub(crate) path: PathBuf,
    pub(crate) fly_to: bool,
    pub(crate) visible: bool,
    pub(crate) opacity: f32,
    pub(crate) outcome: Result<LoadedVectorData, String>,
}

/// Dedicated single-thread background worker loading and tessellating local
/// GeoJSON files strictly off the UI thread, independent of raster
/// `OverlayManager` / `OverlayIoWorker`.
pub(crate) struct VectorIoWorker {
    tx: Option<Sender<VectorIoCmd>>,
    rx: Receiver<VectorIoResult>,
    shutdown: CancelToken,
    handle: Option<JoinHandle<()>>,
}

impl VectorIoWorker {
    pub(crate) fn spawn(wakeup: Wakeup, shutdown: CancelToken) -> Self {
        let (cmd_tx, cmd_rx) = mpsc::channel::<VectorIoCmd>();
        let (res_tx, res_rx) = mpsc::channel::<VectorIoResult>();
        let worker_shutdown = shutdown.clone();
        let handle = match thread::Builder::new()
            .name("veldmap-vector-io".to_owned())
            .spawn(move || {
                while let Ok(cmd) = cmd_rx.recv() {
                    if worker_shutdown.is_cancelled() {
                        break;
                    }
                    if cmd.cancel.is_cancelled() {
                        continue;
                    }
                    let outcome = load_vector_file(&cmd.path);
                    if cmd.cancel.is_cancelled() || worker_shutdown.is_cancelled() {
                        continue;
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
            }) {
            Ok(h) => Some(h),
            Err(err) => {
                tracing::error!(error = %err, "Failed to spawn vector I/O worker");
                None
            }
        };

        Self {
            tx: handle.is_some().then_some(cmd_tx),
            rx: res_rx,
            shutdown,
            handle,
        }
    }

    pub(crate) fn begin_open(
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
            opacity: opacity.clamp(0.0, 1.0),
            cancel: self.shutdown.child_token(),
        };
        if self.tx.as_ref().is_some_and(|tx| tx.send(cmd).is_ok()) {
            Ok(())
        } else {
            Err("vector I/O worker is unavailable".to_owned())
        }
    }

    pub(crate) fn try_recv(&self) -> Option<VectorIoResult> {
        self.rx.try_recv().ok()
    }

    pub(crate) fn close_channel(&mut self) {
        self.tx = None;
    }

    pub(crate) fn join(&mut self) {
        self.tx = None;
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for VectorIoWorker {
    fn drop(&mut self) {
        self.join();
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
pub(crate) struct VectorLayerStack {
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
    /// Insert a pre-parsed [`LoadedVectorData`] produced by [`VectorIoWorker`]
    /// into the stack.
    pub(crate) fn push_loaded(
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
        let name = data
            .path
            .file_name()
            .and_then(|n| n.to_str())
            .map_or_else(|| data.path.display().to_string(), str::to_owned);

        self.layers.push(VectorLayer {
            id,
            name,
            path: data.path,
            visible,
            opacity: opacity.clamp(0.0, 1.0),
            fill_color,
            line_color,
            color_rgb,
            feature_count: data.feature_count,
            mesh: data.mesh,
        });
        (id, bounds)
    }

    pub(crate) fn set_visible(&mut self, id: VectorLayerId, visible: bool) -> bool {
        let Some(layer) = self.layers.iter_mut().find(|l| l.id == id) else {
            return false;
        };
        if layer.visible == visible {
            return false;
        }
        layer.visible = visible;
        true
    }

    pub(crate) fn set_opacity(&mut self, id: VectorLayerId, opacity: f32) -> bool {
        let clamped = opacity.clamp(0.0, 1.0);
        let Some(layer) = self.layers.iter_mut().find(|l| l.id == id) else {
            return false;
        };
        if (layer.opacity - clamped).abs() <= f32::EPSILON {
            return false;
        }
        layer.opacity = clamped;
        true
    }

    pub(crate) fn remove(&mut self, id: VectorLayerId) -> bool {
        let Some(idx) = self.layers.iter().position(|l| l.id == id) else {
            return false;
        };
        self.layers.remove(idx);
        true
    }

    pub(crate) fn len(&self) -> usize {
        self.layers.len()
    }

    pub(crate) fn summaries(&self) -> Vec<VectorLayerSummary> {
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

    pub(crate) fn draw_layers(&self) -> Vec<DrawVectorLayer> {
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

    pub(crate) fn saved_entries(&self) -> Vec<SavedVectorEntry> {
        self.layers
            .iter()
            .map(|l| SavedVectorEntry {
                path: l.path.clone(),
                visible: l.visible,
                opacity: l.opacity,
            })
            .collect()
    }
}
