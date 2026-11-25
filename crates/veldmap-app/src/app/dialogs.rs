//! Non-blocking native OS file picker and save dialog wrapper (`rfd` with
//! `xdg-portal` + `wayland`, zero `gtk3` build dependencies; dynamically loads
//! `libdbus-1` at runtime with `zenity` fallback on Linux), preserving
//! text-path and drag-and-drop fallbacks on minimal Linux window managers.
//!
//! Uses `rfd::AsyncFileDialog` constructed on the UI thread with
//! `.set_parent(frame)` (starting a non-blocking sheet on macOS and a lazy
//! future on Windows/Linux) and drives the resulting `Send + 'static` future
//! off-thread via `pollster::block_on`, so modal pickers never freeze `eframe`
//! on any OS.

use std::future::Future;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;

use veldmap_core::sync::Wakeup;

/// Completed result from a background native file dialog.
#[derive(Clone, Debug)]
pub(super) enum FileDialogEvent {
    OpenFiles(Vec<PathBuf>),
    GeoJsonSaved {
        path: PathBuf,
        result: Result<(), String>,
    },
}

struct BusyResetGuard {
    busy: Arc<AtomicBool>,
    wakeup: Wakeup,
}

impl Drop for BusyResetGuard {
    fn drop(&mut self) {
        self.busy.store(false, Ordering::Release);
        (self.wakeup)();
    }
}

/// Manages off-UI-thread native `rfd` open/save dialogs so modal OS pickers
/// never freeze the `eframe` render loop.
pub(super) struct DialogHost {
    tx: Sender<FileDialogEvent>,
    rx: Receiver<FileDialogEvent>,
    busy: Arc<AtomicBool>,
    wakeup: Wakeup,
}

impl DialogHost {
    pub(super) fn new(wakeup: Wakeup) -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            tx,
            rx,
            busy: Arc::new(AtomicBool::new(false)),
            wakeup,
        }
    }

    pub(super) fn is_busy(&self) -> bool {
        self.busy.load(Ordering::Acquire)
    }

    fn spawn_dialog_future<F>(&self, thread_name: &str, fut: F) -> bool
    where
        F: Future<Output = Option<FileDialogEvent>> + Send + 'static,
    {
        let tx = self.tx.clone();
        let busy = Arc::clone(&self.busy);
        let wakeup = Arc::clone(&self.wakeup);
        let spawned = thread::Builder::new()
            .name(thread_name.to_owned())
            .spawn(move || {
                let _guard = BusyResetGuard { busy, wakeup };
                if let Some(ev) = pollster::block_on(fut) {
                    let _ = tx.send(ev);
                }
            });
        if spawned.is_err() {
            self.busy.store(false, Ordering::Release);
            return false;
        }
        true
    }

    fn try_acquire_busy(&self) -> bool {
        self.busy
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    /// Spawn a non-blocking native multi-file open dialog for GeoTIFF, PNG,
    /// JPEG, and GeoJSON files, parented to `frame`.
    pub(super) fn spawn_open_files(&self, frame: &eframe::Frame) -> bool {
        if !self.try_acquire_busy() {
            return false;
        }
        // Construct the future on the UI thread so macOS begins a non-blocking
        // sheet on AppKit's main thread; Windows/Linux return a lazy future.
        let fut = rfd::AsyncFileDialog::new()
            .set_parent(frame)
            .set_title("Open Raster or GeoJSON")
            .add_filter(
                "Geospatial Rasters & GeoJSON",
                &["tif", "tiff", "png", "jpg", "jpeg", "geojson", "json"],
            )
            .add_filter("GeoTIFF / COG", &["tif", "tiff"])
            .add_filter("Images (with world file)", &["png", "jpg", "jpeg"])
            .add_filter("GeoJSON", &["geojson", "json"])
            .pick_files();
        self.spawn_dialog_future("veldmap-file-dialog", async move {
            let handles = fut.await?;
            let paths: Vec<PathBuf> = handles.into_iter().map(PathBuf::from).collect();
            (!paths.is_empty()).then_some(FileDialogEvent::OpenFiles(paths))
        })
    }

    /// Spawn a non-blocking native save dialog for RFC 7946 GeoJSON export,
    /// capturing `geojson_payload` at click time and writing the file off the
    /// UI thread once the user selects a destination.
    pub(super) fn spawn_save_geojson(
        &self,
        default_name: &str,
        geojson_payload: String,
        frame: &eframe::Frame,
    ) -> bool {
        if !self.try_acquire_busy() {
            return false;
        }
        let fut = rfd::AsyncFileDialog::new()
            .set_parent(frame)
            .set_title("Export Measurement as RFC 7946 GeoJSON")
            .set_file_name(default_name)
            .add_filter("GeoJSON", &["geojson", "json"])
            .save_file();
        self.spawn_dialog_future("veldmap-save-dialog", async move {
            let handle = fut.await?;
            let path = PathBuf::from(handle);
            let result = veldmap_storage::write_atomic_durable(&path, geojson_payload.as_bytes())
                .map_err(|e| format!("failed to write '{}': {e}", path.display()));
            Some(FileDialogEvent::GeoJsonSaved { path, result })
        })
    }

    /// Drain any completed file dialog events without blocking.
    pub(super) fn poll(&self) -> Vec<FileDialogEvent> {
        let mut out = Vec::new();
        while let Ok(ev) = self.rx.try_recv() {
            out.push(ev);
        }
        out
    }
}
