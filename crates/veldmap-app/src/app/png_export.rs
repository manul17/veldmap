//! Non-blocking offscreen cartographic PNG export coordinator for
//! [`super::VeldmapApp`].
//!
//! Never calls blocking `device.poll(Wait)` on the UI thread: GPU readback uses
//! `map_async` + `Wakeup`, and cartographic overlay compositing + PNG encoding
//! (`veldmap_raster::encode_png_rgba8`) + durable atomic write execute on a
//! tracked background thread joined cleanly on completion or shutdown.

use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use veldmap_core::sync::{Wakeup, lock_unpoisoned};

use crate::export::{
    CartographicExportOptions, CartographicOverlaySpec, composite_cartographic_overlays,
};

use super::viewport_bridge::PngReadbackBridge;

struct ActiveCapture {
    path: PathBuf,
    spec: CartographicOverlaySpec,
    bridge: Arc<Mutex<PngReadbackBridge>>,
    submitted_to_gpu: bool,
    rearm_attempts: u8,
}

/// Completed off-UI-thread PNG encode/write outcome.
pub(super) struct PngExportDone {
    pub(super) path: PathBuf,
    pub(super) result: Result<(), String>,
}

/// Coordinates non-blocking GPU readback staging and tracked off-UI-thread
/// cartographic overlay compositing + PNG encoding.
pub(super) struct PngExportController {
    modal_open: bool,
    options: CartographicExportOptions,
    active_capture: Option<ActiveCapture>,
    encode_thread: Option<JoinHandle<()>>,
    encode_tx: Sender<PngExportDone>,
    encode_rx: Receiver<PngExportDone>,
    wakeup: Wakeup,
    last_notice: Option<String>,
}

impl PngExportController {
    pub(super) fn new(wakeup: Wakeup) -> Self {
        let (encode_tx, encode_rx) = mpsc::channel();
        Self {
            modal_open: false,
            options: CartographicExportOptions::default(),
            active_capture: None,
            encode_thread: None,
            encode_tx,
            encode_rx,
            wakeup,
            last_notice: None,
        }
    }

    pub(super) fn open_modal(&mut self, suggested_caption: String) {
        self.modal_open = true;
        self.last_notice = None;
        if self.options.caption_text.trim().is_empty() && !suggested_caption.is_empty() {
            self.options.caption_text = suggested_caption;
        }
    }

    pub(super) const fn options(&self) -> &CartographicExportOptions {
        &self.options
    }

    pub(super) fn modal_state_mut(&mut self) -> (&mut bool, &mut CartographicExportOptions) {
        (&mut self.modal_open, &mut self.options)
    }

    pub(super) fn last_notice(&self) -> Option<&str> {
        self.last_notice.as_deref()
    }

    /// Return `true` while either GPU readback or background CPU PNG encoding
    /// is in flight (disables concurrent export triggers and displays spinner).
    pub(super) fn is_busy(&self) -> bool {
        self.active_capture.is_some() || self.encode_thread.is_some()
    }

    /// Return `true` strictly while waiting on `wgpu` GPU readback / `map_async`
    /// completion so `schedule_next_repaint` polls `device.poll(Poll)` without
    /// spinning during CPU background encoding (which wakes via `Wakeup`).
    pub(super) fn needs_gpu_poll(&self) -> bool {
        self.active_capture.is_some()
    }

    /// Arm a clean-frame-safe GPU readback for the next paint callback.
    pub(super) fn begin_capture(&mut self, path: PathBuf, spec: CartographicOverlaySpec) {
        if self.is_busy() {
            return;
        }
        self.last_notice = Some("Capturing viewport…".to_owned());
        self.active_capture = Some(ActiveCapture {
            path,
            spec,
            bridge: PngReadbackBridge::new(Arc::clone(&self.wakeup)),
            submitted_to_gpu: false,
            rearm_attempts: 0,
        });
    }

    /// Refresh viewport-dependent cartographic metrics right before arming the
    /// paint callback so scale bar and North arrow match the exact captured
    /// framebuffer even if the window resized while the Save dialog was open.
    pub(super) fn refresh_capture_metrics(
        &mut self,
        pixels_per_point: f32,
        m_per_pt: f64,
        heading_deg: f64,
    ) {
        if let Some(capture) = self.active_capture.as_mut()
            && !capture.submitted_to_gpu
        {
            capture.spec.pixels_per_point = pixels_per_point;
            capture.spec.m_per_pt = m_per_pt;
            capture.spec.heading_deg = heading_deg;
        }
    }

    /// Abort an armed capture that has not yet been submitted to the GPU (for
    /// example when the viewport collapses/minimizes before the paint callback
    /// runs), preventing an indefinite repaint spin loop.
    pub(super) fn abort_unsubmitted_capture(&mut self, reason: &str) -> Option<String> {
        let should_abort = self
            .active_capture
            .as_ref()
            .is_some_and(|c| !c.submitted_to_gpu);
        if !should_abort {
            return None;
        }
        self.active_capture = None;
        let msg = format!("Export failed: {reason}");
        self.last_notice = Some(msg.clone());
        Some(msg)
    }

    /// Hand the shared [`PngReadbackBridge`] to `GlobePaintCallback` exactly
    /// once so `prepare()` records `copy_texture_to_buffer` and starts
    /// `map_async`.
    pub(super) fn take_bridge_for_callback(&mut self) -> Option<Arc<Mutex<PngReadbackBridge>>> {
        let capture = self.active_capture.as_mut()?;
        if capture.submitted_to_gpu {
            return None;
        }
        capture.submitted_to_gpu = true;
        Some(Arc::clone(&capture.bridge))
    }

    /// Advance non-blocking `map_async` completion (`PollType::Poll`) and drain
    /// finished background PNG writes.
    pub(super) fn poll(&mut self, device: &wgpu::Device) -> Option<PngExportDone> {
        enum PollAction {
            Waiting,
            Ready,
            Rearm,
            Dropped,
        }

        let action = if let Some(capture) = &self.active_capture
            && capture.submitted_to_gpu
        {
            let _ = device.poll(wgpu::PollType::Poll);
            let guard = lock_unpoisoned(&capture.bridge);
            if guard.capture_error.is_some() || guard.mapped.is_some() {
                PollAction::Ready
            } else if Arc::strong_count(&capture.bridge) == 1 && guard.readback.is_none() {
                // `GlobePaintCallback` was dropped without running `prepare()`
                // (e.g. an `egui` multi-pass layout discard or minimized frame).
                // Allow two transparent re-arms before failing cleanly so
                // multi-pass frames succeed while minimized windows never spin.
                if capture.rearm_attempts < 2 {
                    PollAction::Rearm
                } else {
                    PollAction::Dropped
                }
            } else {
                PollAction::Waiting
            }
        } else {
            PollAction::Waiting
        };

        match action {
            PollAction::Waiting => {}
            PollAction::Rearm => {
                if let Some(capture) = self.active_capture.as_mut() {
                    capture.submitted_to_gpu = false;
                    capture.rearm_attempts = capture.rearm_attempts.saturating_add(1);
                }
            }
            PollAction::Dropped => {
                let capture = self.active_capture.take()?;
                let err =
                    "Viewport paint callback was skipped; cannot capture screenshot.".to_owned();
                self.last_notice = Some(format!("Export failed: {err}"));
                return Some(PngExportDone {
                    path: capture.path,
                    result: Err(err),
                });
            }
            PollAction::Ready => {
                let capture = self.active_capture.take()?;
                let mut guard = lock_unpoisoned(&capture.bridge);
                if let Some(err) = guard.capture_error.take() {
                    self.last_notice = Some(format!("Export failed: {err}"));
                    return Some(PngExportDone {
                        path: capture.path,
                        result: Err(err),
                    });
                }
                match guard.mapped.take() {
                    Some(Ok(())) => {
                        if let Some(readback) = guard.readback.take() {
                            let width = readback.width();
                            let height = readback.height();
                            drop(guard);
                            match readback.read_tight_rgba8() {
                                Ok(rgba) => {
                                    self.spawn_background_encode(
                                        capture.path,
                                        width,
                                        height,
                                        rgba,
                                        capture.spec,
                                    );
                                }
                                Err(err) => {
                                    self.last_notice = Some(format!("GPU readback failed: {err}"));
                                    return Some(PngExportDone {
                                        path: capture.path,
                                        result: Err(err),
                                    });
                                }
                            }
                        }
                    }
                    Some(Err(err)) => {
                        self.last_notice = Some(format!("GPU readback failed: {err}"));
                        return Some(PngExportDone {
                            path: capture.path,
                            result: Err(err),
                        });
                    }
                    None => {}
                }
            }
        }

        if let Ok(done) = self.encode_rx.try_recv() {
            if let Some(handle) = self.encode_thread.take() {
                let _ = handle.join();
            }
            match &done.result {
                Ok(()) => {
                    self.last_notice = Some(format!("Saved PNG to {}", done.path.display()));
                }
                Err(err) => {
                    self.last_notice = Some(format!("PNG export failed: {err}"));
                }
            }
            return Some(done);
        }
        None
    }

    /// Join any in-flight background PNG encode/write thread deterministically
    /// on application exit so a large PNG save is never truncated mid-write.
    pub(super) fn join_background_encode(&mut self) {
        self.active_capture = None;
        if let Some(handle) = self.encode_thread.take() {
            let _ = handle.join();
        }
    }

    fn spawn_background_encode(
        &mut self,
        path: PathBuf,
        width: u32,
        height: u32,
        mut pixels: Vec<u8>,
        spec: CartographicOverlaySpec,
    ) {
        if let Some(prev) = self.encode_thread.take() {
            let _ = prev.join();
        }
        self.last_notice = Some("Encoding PNG…".to_owned());
        let tx = self.encode_tx.clone();
        let wakeup = Arc::clone(&self.wakeup);
        let thread_path = path.clone();
        let spawned = thread::Builder::new()
            .name("veldmap-png-export".to_owned())
            .spawn(move || {
                composite_cartographic_overlays(width, height, &mut pixels, &spec);
                let result = veldmap_raster::encode_png_rgba8(width, height, &pixels)
                    .map_err(|e| format!("PNG encoding failed: {e}"))
                    .and_then(|bytes| {
                        veldmap_storage::write_atomic_durable(&thread_path, &bytes).map_err(|e| {
                            format!("failed to write '{}': {e}", thread_path.display())
                        })
                    });
                let _ = tx.send(PngExportDone {
                    path: thread_path,
                    result,
                });
                (wakeup)();
            });
        match spawned {
            Ok(handle) => {
                self.encode_thread = Some(handle);
            }
            Err(err) => {
                let _ = self.encode_tx.send(PngExportDone {
                    path,
                    result: Err(format!("failed to spawn PNG export worker: {err}")),
                });
                (self.wakeup)();
            }
        }
    }
}
