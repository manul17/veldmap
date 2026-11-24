//! Non-blocking native OS file picker and save dialog wrapper (`rfd` with
//! pure-Rust `xdg-portal` + `wayland`, zero `gtk3`), preserving text-path and
//! drag-and-drop fallbacks on minimal Linux window managers.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;

use veldmap_core::sync::Wakeup;

/// Completed result from a background native file dialog.
#[derive(Clone, Debug)]
pub(crate) enum FileDialogEvent {
    OpenFiles(Vec<PathBuf>),
    SaveGeoJson(PathBuf),
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
pub(crate) struct DialogHost {
    tx: Sender<FileDialogEvent>,
    rx: Receiver<FileDialogEvent>,
    busy: Arc<AtomicBool>,
    wakeup: Wakeup,
}

impl DialogHost {
    pub(crate) fn new(wakeup: Wakeup) -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            tx,
            rx,
            busy: Arc::new(AtomicBool::new(false)),
            wakeup,
        }
    }

    pub(crate) fn is_busy(&self) -> bool {
        self.busy.load(Ordering::Acquire)
    }

    /// Spawn a background native multi-file open dialog for GeoTIFF, PNG,
    /// JPEG, and GeoJSON files.
    pub(crate) fn spawn_open_files(&self) -> bool {
        if self
            .busy
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return false;
        }
        let tx = self.tx.clone();
        let busy = Arc::clone(&self.busy);
        let wakeup = Arc::clone(&self.wakeup);
        let spawned = thread::Builder::new()
            .name("veldmap-file-dialog".to_owned())
            .spawn(move || {
                let _guard = BusyResetGuard { busy, wakeup };
                let picked = rfd::FileDialog::new()
                    .set_title("Open Raster or GeoJSON")
                    .add_filter(
                        "Geospatial Rasters & GeoJSON",
                        &["tif", "tiff", "png", "jpg", "jpeg", "geojson", "json"],
                    )
                    .add_filter("GeoTIFF / COG", &["tif", "tiff"])
                    .add_filter("Images (with world file)", &["png", "jpg", "jpeg"])
                    .add_filter("GeoJSON", &["geojson", "json"])
                    .pick_files();
                if let Some(paths) = picked
                    && !paths.is_empty()
                {
                    let _ = tx.send(FileDialogEvent::OpenFiles(paths));
                }
            });
        if spawned.is_err() {
            self.busy.store(false, Ordering::Release);
            return false;
        }
        true
    }

    /// Spawn a background native save dialog for RFC 7946 GeoJSON export.
    pub(crate) fn spawn_save_geojson(&self, default_name: &str) -> bool {
        if self
            .busy
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return false;
        }
        let tx = self.tx.clone();
        let busy = Arc::clone(&self.busy);
        let wakeup = Arc::clone(&self.wakeup);
        let file_name = default_name.to_owned();
        let spawned = thread::Builder::new()
            .name("veldmap-save-dialog".to_owned())
            .spawn(move || {
                let _guard = BusyResetGuard { busy, wakeup };
                let picked = rfd::FileDialog::new()
                    .set_title("Export Measurement as RFC 7946 GeoJSON")
                    .set_file_name(&file_name)
                    .add_filter("GeoJSON", &["geojson", "json"])
                    .save_file();
                if let Some(path) = picked {
                    let _ = tx.send(FileDialogEvent::SaveGeoJson(path));
                }
            });
        if spawned.is_err() {
            self.busy.store(false, Ordering::Release);
            return false;
        }
        true
    }

    /// Drain any completed file dialog events without blocking.
    pub(crate) fn poll(&self) -> Vec<FileDialogEvent> {
        let mut out = Vec::new();
        while let Ok(ev) = self.rx.try_recv() {
            out.push(ev);
        }
        out
    }
}
