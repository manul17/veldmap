//! Asynchronous local/remote raster open, cache-bypassing reload, and
//! background completion reconciliation for [`OverlayManager`].

use std::path::{Path, PathBuf};
use std::sync::Arc;

use veldmap_core::geo::{FootprintBounds, Frame};
use veldmap_raster::RasterSource;

use crate::clamp_opacity;

use super::io_worker::{OverlayIoCmd, OverlayIoResult, ReloadRequest, ReloadedPayload};
use super::layer::RemoteLayerSpec;
use super::manager::{OverlayManager, OverlayMutation, alloc_seq};
use super::opener::{OpenRasterError, bind_frame_with_override};
use super::stack::OverlayLayer;
use super::{LayerId, LayerInitOptions, OpenOverlayRequest, OverlayIoEvent, RemoteCogRequest};

impl OverlayManager {
    /// Queue a non-blocking remote Cloud-Optimized GeoTIFF open onto
    /// [`veldmap_net::NetContext`] + `remote_pool`, completely isolated from
    /// local file open/reload I/O and `fast_pool`.
    pub(crate) fn begin_open_remote_cog(&self, request: RemoteCogRequest) -> Result<(), String> {
        self.remote_opener.begin_open(&self.runtime, request)
    }

    /// Queue a non-blocking local raster open with explicit initial visibility,
    /// opacity, optional CRS override, and optional saved analytical style.
    pub(crate) fn begin_open_file(&self, request: OpenOverlayRequest<'_>) -> Result<(), String> {
        if self.io_worker.send(OverlayIoCmd::Open {
            path: request.path.to_path_buf(),
            fly_to: request.fly_to,
            options: LayerInitOptions {
                visible: request.visible,
                opacity: clamp_opacity(request.opacity),
                crs_override: request.crs_override,
                style: request.style,
            },
        }) {
            Ok(())
        } else {
            Err("overlay I/O worker is unavailable".to_owned())
        }
    }

    /// Queue a non-blocking reload of `id` onto the overlay I/O worker,
    /// cancelling active tile jobs for `old_src` immediately so stale tiles
    /// cannot race the reload, while preserving the layer if reopen fails.
    pub(crate) fn begin_reload_bypass_cache(
        &mut self,
        id: LayerId,
    ) -> Result<OverlayMutation, String> {
        let cancel = self.runtime.shutdown.child_token();
        let (request, prep) = {
            let Some(layer) = self.stack.get_mut(id) else {
                return Err("layer no longer exists".to_owned());
            };
            let Some(local_path) = layer.local_path().map(Path::to_path_buf) else {
                return Err("remote COG layers do not support local disk reload".to_owned());
            };
            layer.cancel_stats_job();
            let prep = layer.begin_reload(cancel.clone());
            let request = ReloadRequest {
                id,
                reload_seq: prep.reload_seq,
                crs_override_epoch: layer.geo().crs_override_epoch(),
                path: local_path,
                old_fingerprint_hex: layer.fingerprint_hex().to_owned(),
                crs_override: layer.geo().crs_override().cloned(),
                cancel,
            };
            (request, prep)
        };
        if let Some(prev) = prep.old_cancel {
            prev.cancel();
        }

        // Cancel in-flight jobs, rotate source write tokens, purge queued disk
        // writes, and clear runtime caches for `old_source` BEFORE `io_worker`
        // runs `clear_source` so no pre-reload tile can poison SQLite.
        self.purge_source(prep.old_source);

        if !self.io_worker.send(OverlayIoCmd::Reload(request)) {
            if let Some(layer) = self.stack.get_mut(id) {
                layer.abort_reload();
            }
            self.runtime.request_reschedule();
            return Err("overlay I/O worker is unavailable".to_owned());
        }

        Ok(OverlayMutation {
            changed: true,
            topology_changed: true,
            purge_sources: vec![prep.old_source],
        })
    }

    /// Drain completed background local open/reload results, remote COG open
    /// results, and sampled band-statistics completions.
    pub(crate) fn poll_io_events(&mut self) -> (Vec<OverlayIoEvent>, OverlayMutation) {
        let mut events = Vec::new();
        let mut mutation = OverlayMutation::default();

        while let Some(res) = self.io_worker.try_recv() {
            match res {
                OverlayIoResult::Opened {
                    path,
                    fly_to,
                    options,
                    outcome,
                } => self.handle_opened_result(
                    path,
                    fly_to,
                    options,
                    outcome,
                    &mut events,
                    &mut mutation,
                ),
                OverlayIoResult::Reloaded(payload) => {
                    self.handle_reloaded_result(payload, &mut events, &mut mutation);
                }
            }
        }

        for remote in self.remote_opener.drain() {
            match remote.outcome {
                Ok((range_source, source, frame)) => {
                    let id = LayerId(alloc_seq(&mut self.next_layer_id));
                    let relaid = alloc_seq(&mut self.next_relaid);
                    let layer = OverlayLayer::from_remote_source(RemoteLayerSpec {
                        id,
                        title: remote.title.clone(),
                        display_url: remote.display_url.clone(),
                        range_source,
                        source,
                        frame,
                        options: LayerInitOptions {
                            visible: true,
                            opacity: 1.0,
                            crs_override: None,
                            style: None,
                        },
                        relaid,
                    });
                    let bounds = self.insert_opened_layer(layer, &mut mutation);
                    events.push(OverlayIoEvent::RemoteOpened {
                        id,
                        title: remote.title,
                        display_url: remote.display_url,
                        bounds,
                        fly_to: remote.fly_to,
                    });
                }
                Err(error) => {
                    events.push(OverlayIoEvent::RemoteFailed {
                        title: remote.title,
                        display_url: remote.display_url,
                        error,
                    });
                }
            }
        }

        if self.band_stats.drain_into(&mut self.stack) {
            mutation.changed = true;
        }

        (events, mutation)
    }

    fn insert_opened_layer(
        &mut self,
        layer: OverlayLayer,
        mutation: &mut OverlayMutation,
    ) -> Option<FootprintBounds> {
        let id = layer.id();
        let bounds = layer.geo().footprint();
        self.stack.push(layer);
        self.request_band_stats_if_needed(id);
        mutation.changed = true;
        mutation.topology_changed = true;
        bounds
    }

    fn handle_opened_result(
        &mut self,
        path: PathBuf,
        fly_to: bool,
        options: LayerInitOptions,
        outcome: Result<(Arc<dyn RasterSource>, Frame), OpenRasterError>,
        events: &mut Vec<OverlayIoEvent>,
        mutation: &mut OverlayMutation,
    ) {
        match outcome {
            Ok((source, frame)) => {
                let id = LayerId(alloc_seq(&mut self.next_layer_id));
                let relaid = alloc_seq(&mut self.next_relaid);
                let event_path = path.clone();
                let layer = OverlayLayer::from_source(id, path, source, frame, options, relaid);
                let bounds = self.insert_opened_layer(layer, mutation);
                events.push(OverlayIoEvent::Opened {
                    id,
                    path: event_path,
                    bounds,
                    fly_to,
                });
            }
            Err(err) => {
                events.push(OverlayIoEvent::Failed {
                    path,
                    error: err.message,
                    can_assign_crs: err.can_assign_crs,
                });
            }
        }
    }

    fn handle_reloaded_result(
        &mut self,
        reloaded: ReloadedPayload,
        events: &mut Vec<OverlayIoEvent>,
        mutation: &mut OverlayMutation,
    ) {
        let ReloadedPayload {
            id,
            reload_seq,
            crs_override_epoch,
            path,
            outcome,
        } = reloaded;
        let (old_src, new_src) = {
            let Some(slot) = self.stack.get_mut(id) else {
                return;
            };
            if !slot.matches_reload_seq(reload_seq) {
                // Layer was superseded by a newer reload trigger.
                return;
            }
            match outcome {
                Ok((source, mut frame)) => {
                    // If `override_layer_crs` ran while this reload was in flight,
                    // re-bind the reloaded source against the live `crs_override`
                    // so it never silently reverts.
                    if slot.geo().crs_override_epoch() != crs_override_epoch {
                        match bind_frame_with_override(
                            source.description(),
                            slot.geo().crs_override(),
                        ) {
                            Ok(rebound) => frame = rebound,
                            Err(error) => {
                                slot.fail_reload(error.clone());
                                mutation.changed = true;
                                mutation.topology_changed = true;
                                events.push(OverlayIoEvent::ReloadFailed {
                                    path,
                                    error,
                                    can_assign_crs: true,
                                });
                                return;
                            }
                        }
                    }
                    let new_src = source.source_id();
                    let relaid = alloc_seq(&mut self.next_relaid);
                    let old_src = slot.replace_reloaded(path, source, frame, relaid);
                    (old_src, new_src)
                }
                Err(err) => {
                    slot.fail_reload(err.message.clone());
                    // Reload failed: leave the existing layer intact and
                    // resume normal rendering on the next frame.
                    mutation.changed = true;
                    mutation.topology_changed = true;
                    events.push(OverlayIoEvent::ReloadFailed {
                        path,
                        error: err.message,
                        can_assign_crs: err.can_assign_crs,
                    });
                    return;
                }
            }
        };

        // Purge runtime/GPU state for both `old_src` and `new_src`
        // (in case a sibling layer decoded tiles during reload).
        self.purge_source(old_src);
        mutation.purge_sources.push(old_src);
        if new_src != old_src {
            self.purge_source(new_src);
            mutation.purge_sources.push(new_src);
        }
        mutation.changed = true;
        mutation.topology_changed = true;
        self.request_band_stats_if_needed(id);
        events.push(OverlayIoEvent::Reloaded { id });
    }
}
