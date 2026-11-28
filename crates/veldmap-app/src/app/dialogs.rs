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
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;

use veldmap_core::sync::{CancelToken, Wakeup};

use crate::layers::{ScannedTimelineFile, scan_dated_raster_folder};
use crate::storage::project::ProjectFile;

/// Completed result from a background native file dialog or background project
/// load.
#[derive(Clone, Debug)]
pub(super) enum FileDialogEvent {
    OpenFiles(Vec<PathBuf>),
    TimelineFolderScanned {
        dir: PathBuf,
        result: Result<Vec<ScannedTimelineFile>, String>,
    },
    GeoJsonSaved {
        path: PathBuf,
        result: Result<(), String>,
    },
    ProjectSaved {
        path: PathBuf,
        result: Result<(), String>,
    },
    ProjectLoaded {
        seq: u64,
        path: PathBuf,
        result: Result<ProjectFile, String>,
    },
    PngExportPathPicked(PathBuf),
}

/// Normalize a user-picked save path to end with `.veldmap.json` without
/// producing `.json.veldmap.json` or `.veldmap.veldmap.json` when the OS save
/// dialog auto-appends `.json` from the filter.
fn normalize_project_save_path(path: PathBuf) -> PathBuf {
    if crate::storage::project::is_project_path(&path) {
        return path;
    }
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
        let mut os = path.into_os_string();
        os.push(crate::storage::project::PROJECT_FILE_SUFFIX);
        return PathBuf::from(os);
    };
    let lower = name.to_ascii_lowercase();
    let base_name = if lower.ends_with(".json") {
        &name[..name.len() - ".json".len()]
    } else {
        name
    };
    let final_name = if base_name.to_ascii_lowercase().ends_with(".veldmap") {
        format!("{base_name}.json")
    } else {
        format!(
            "{base_name}{}",
            crate::storage::project::PROJECT_FILE_SUFFIX
        )
    };
    path.with_file_name(final_name)
}

/// Append `.png` to a user-picked PNG save path unless it already ends with
/// `.png`, preserving dotted stems such as `sentinel2.2026.11.27`.
fn normalize_png_save_path(path: PathBuf) -> PathBuf {
    if path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("png"))
    {
        return path;
    }
    let mut os = path.into_os_string();
    os.push(".png");
    PathBuf::from(os)
}

/// Native file dialog operation kind used to tailor error notices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DialogKind {
    Open,
    Save,
}

/// Error returned when a native file dialog cannot be spawned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DialogSpawnError {
    Busy,
    SpawnFailed,
}

impl DialogSpawnError {
    pub(super) const fn notice(self, kind: DialogKind) -> &'static str {
        match (self, kind) {
            (Self::Busy, _) => "Another file dialog is already open.",
            (Self::SpawnFailed, DialogKind::Open) => {
                "Unable to open native file dialog on this system."
            }
            (Self::SpawnFailed, DialogKind::Save) => {
                "Unable to open native save dialog on this system."
            }
        }
    }
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
    project_load_seq: Arc<AtomicU64>,
    wakeup: Wakeup,
}

impl DialogHost {
    pub(super) fn new(wakeup: Wakeup) -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            tx,
            rx,
            busy: Arc::new(AtomicBool::new(false)),
            project_load_seq: Arc::new(AtomicU64::new(0)),
            wakeup,
        }
    }

    /// Return the latest dispatched project-load sequence number so older slow
    /// background loads never overwrite a newer selection.
    pub(super) fn latest_project_load_seq(&self) -> u64 {
        self.project_load_seq.load(Ordering::Acquire)
    }

    /// Load a `.veldmap.json` project file off the UI thread (used for
    /// drag-and-drop, Recent Files, Open Path prompt, and CLI startup args
    /// without blocking on or racing with native OS dialog state).
    pub(super) fn spawn_load_project_path(&self, path: PathBuf) -> Result<u64, DialogSpawnError> {
        let seq = self.project_load_seq.fetch_add(1, Ordering::AcqRel) + 1;
        let tx = self.tx.clone();
        let wakeup = Arc::clone(&self.wakeup);
        thread::Builder::new()
            .name("veldmap-load-project".to_owned())
            .spawn(move || {
                let result = ProjectFile::load_from(&path)
                    .map_err(|e| format!("failed to load '{}': {e}", path.display()));
                let _ = tx.send(FileDialogEvent::ProjectLoaded { seq, path, result });
                (wakeup)();
            })
            .map_err(|_| DialogSpawnError::SpawnFailed)?;
        Ok(seq)
    }

    fn spawn_dialog_future<F>(&self, thread_name: &str, fut: F) -> Result<(), DialogSpawnError>
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
            return Err(DialogSpawnError::SpawnFailed);
        }
        Ok(())
    }

    fn try_acquire_busy(&self) -> bool {
        self.busy
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    /// Spawn a non-blocking native multi-file open dialog for GeoTIFF, PNG,
    /// JPEG, and GeoJSON files, parented to `frame`.
    pub(super) fn spawn_open_files(&self, frame: &eframe::Frame) -> Result<(), DialogSpawnError> {
        if !self.try_acquire_busy() {
            return Err(DialogSpawnError::Busy);
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

    /// Spawn a non-blocking native folder picker for building a local dated
    /// multi-temporal `Timeline` layer, running the directory + TIFF DateTime
    /// scan off the UI thread.
    pub(super) fn spawn_open_folder_timeline(
        &self,
        frame: &eframe::Frame,
    ) -> Result<(), DialogSpawnError> {
        if !self.try_acquire_busy() {
            return Err(DialogSpawnError::Busy);
        }
        let fut = rfd::AsyncFileDialog::new()
            .set_parent(frame)
            .set_title("Open Dated Raster Folder as Timeline")
            .pick_folder();
        self.spawn_dialog_future("veldmap-folder-timeline", async move {
            let handle = fut.await?;
            let dir = PathBuf::from(handle);
            let cancel = CancelToken::new();
            let result = scan_dated_raster_folder(&dir, &cancel);
            Some(FileDialogEvent::TimelineFolderScanned { dir, result })
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
    ) -> Result<(), DialogSpawnError> {
        if !self.try_acquire_busy() {
            return Err(DialogSpawnError::Busy);
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

    /// Spawn a non-blocking native save dialog for a `.veldmap.json` project
    /// file, writing the sanitized/relativized project off the UI thread.
    pub(super) fn spawn_save_project(
        &self,
        default_name: &str,
        project: ProjectFile,
        frame: &eframe::Frame,
    ) -> Result<(), DialogSpawnError> {
        if !self.try_acquire_busy() {
            return Err(DialogSpawnError::Busy);
        }
        let fut = rfd::AsyncFileDialog::new()
            .set_parent(frame)
            .set_title("Save VeldMap Project")
            .set_file_name(default_name)
            .add_filter("VeldMap Project (*.veldmap.json)", &["json"])
            .save_file();
        self.spawn_dialog_future("veldmap-save-project", async move {
            let handle = fut.await?;
            let path = normalize_project_save_path(PathBuf::from(handle));
            let result = project
                .save_to(&path)
                .map_err(|e| format!("failed to save '{}': {e}", path.display()));
            Some(FileDialogEvent::ProjectSaved { path, result })
        })
    }

    /// Spawn a non-blocking native open dialog for a `.veldmap.json` project
    /// file, reading and resolving relative paths off the UI thread.
    pub(super) fn spawn_open_project(&self, frame: &eframe::Frame) -> Result<(), DialogSpawnError> {
        if !self.try_acquire_busy() {
            return Err(DialogSpawnError::Busy);
        }
        let seq_counter = Arc::clone(&self.project_load_seq);
        let fut = rfd::AsyncFileDialog::new()
            .set_parent(frame)
            .set_title("Open VeldMap Project")
            .add_filter("VeldMap Project (*.veldmap.json)", &["json"])
            .pick_file();
        self.spawn_dialog_future("veldmap-open-project", async move {
            let handle = fut.await?;
            let path = PathBuf::from(handle);
            let seq = seq_counter.fetch_add(1, Ordering::AcqRel) + 1;
            let result = ProjectFile::load_from(&path)
                .map_err(|e| format!("failed to load '{}': {e}", path.display()));
            Some(FileDialogEvent::ProjectLoaded { seq, path, result })
        })
    }

    /// Spawn a non-blocking native save dialog selecting a destination `.png`
    /// path for cartographic PNG export.
    pub(super) fn spawn_pick_png_export_path(
        &self,
        default_name: &str,
        frame: &eframe::Frame,
    ) -> Result<(), DialogSpawnError> {
        if !self.try_acquire_busy() {
            return Err(DialogSpawnError::Busy);
        }
        let fut = rfd::AsyncFileDialog::new()
            .set_parent(frame)
            .set_title("Export Globe Viewport as PNG")
            .set_file_name(default_name)
            .add_filter("PNG Image", &["png"])
            .save_file();
        self.spawn_dialog_future("veldmap-export-png-dialog", async move {
            let handle = fut.await?;
            let path = normalize_png_save_path(PathBuf::from(handle));
            Some(FileDialogEvent::PngExportPathPicked(path))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_path_normalization_avoids_double_extensions_and_preserves_dotted_stems() {
        assert_eq!(
            normalize_project_save_path(PathBuf::from("study.json")),
            PathBuf::from("study.veldmap.json")
        );
        assert_eq!(
            normalize_project_save_path(PathBuf::from("study.veldmap")),
            PathBuf::from("study.veldmap.json")
        );
        assert_eq!(
            normalize_project_save_path(PathBuf::from("study.veldmap.json")),
            PathBuf::from("study.veldmap.json")
        );
        assert_eq!(
            normalize_project_save_path(PathBuf::from("dir/site.2026.11")),
            PathBuf::from("dir/site.2026.11.veldmap.json")
        );

        assert_eq!(
            normalize_png_save_path(PathBuf::from("sentinel2.2026.11.27")),
            PathBuf::from("sentinel2.2026.11.27.png")
        );
        assert_eq!(
            normalize_png_save_path(PathBuf::from("figure.PNG")),
            PathBuf::from("figure.PNG")
        );
    }
}
