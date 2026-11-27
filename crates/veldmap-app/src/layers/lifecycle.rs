//! Asynchronous local/remote raster open, multi-asset STAC Band Math recipe
//! open, local Band Math builder construction, cache-bypassing reload, and
//! background completion reconciliation for [`OverlayManager`].

use std::path::{Path, PathBuf};
use std::sync::Arc;

use veldmap_core::geo::{FootprintBounds, Frame};
use veldmap_core::tile::{BandKey, PixelFormat};
use veldmap_raster::{RasterSource, SampleType, stream_spec_for_band};

use crate::clamp_opacity;

use super::asset::{GeoBinding, LayerOrigin, RasterAsset};
use super::band_math::BandMathStyle;
use super::io_worker::{OverlayIoCmd, OverlayIoResult, ReloadRequest, ReloadedPayload};
use super::layer::RemoteLayerSpec;
use super::manager::{OverlayManager, OverlayMutation, alloc_seq};
use super::opener::{OpenRasterError, bind_frame_with_override};
use super::remote_opener::{OpenedRemoteBandAsset, RemoteOpenResult};
use super::stack::OverlayLayer;
use super::{
    LayerId, LayerInitOptions, LocalBandMathSpec, OpenOverlayRequest, OverlayIoEvent,
    RemoteBandMathRequest, RemoteCogRequest,
};

fn build_remote_band_asset(
    opened: OpenedRemoteBandAsset,
    format: PixelFormat,
    relaid: u64,
) -> Result<RasterAsset, String> {
    let spec = stream_spec_for_band(BandKey::Index(opened.band_index), format);
    let meta = opened
        .source
        .pyramid_meta(spec)
        .map_err(|e| format!("band '{}': {e}", opened.label))?;
    let desc = opened.source.description();
    let geo = GeoBinding::new(
        opened.frame,
        desc.width,
        desc.height,
        meta.max_level(),
        None,
        relaid,
    );
    Ok(RasterAsset::new(
        opened.label,
        LayerOrigin::Remote {
            display_url: opened.display_url,
            range: opened.range_source,
        },
        opened.source,
        geo,
        meta,
    ))
}

fn suggest_rgb_max_hint(asset: &RasterAsset) -> Option<f64> {
    let band_info = asset
        .source()
        .description()
        .band_info(asset.meta().spec.band)?;
    let has_scaling = (band_info.scale - 1.0).abs() > 1e-12 || band_info.offset.abs() > 1e-12;
    if has_scaling {
        // Scaled reflectance (e.g. Sentinel-2 L2A scale=0.0001, offset=-0.1).
        return Some(0.35);
    }
    match band_info.sample_type {
        SampleType::U8 => Some(255.0),
        SampleType::U16 => Some(3500.0),
        SampleType::F32 | SampleType::F64 => Some(1.0),
        _ => None,
    }
}

impl OverlayManager {
    /// Queue a non-blocking remote Cloud-Optimized GeoTIFF open onto
    /// [`veldmap_net::NetContext`] + `remote_pool`, completely isolated from
    /// local file open/reload I/O and `fast_pool`.
    pub(crate) fn begin_open_remote_cog(&self, request: RemoteCogRequest) -> Result<(), String> {
        self.remote_opener.begin_open(&self.runtime, request)
    }

    /// Queue a non-blocking concurrent multi-asset remote STAC Band Math / RGB
    /// recipe open, spawning one `LayerKind::BandMath` layer without duplicate
    /// raw layers or redundant camera flights.
    pub(crate) fn begin_open_remote_band_math(
        &self,
        request: RemoteBandMathRequest,
    ) -> Result<(), String> {
        self.remote_opener
            .begin_open_band_math(&self.runtime, request)
    }

    /// Build and insert a `LayerKind::BandMath` layer from already-loaded local
    /// or remote `SingleRaster` layers (supporting both bands from a single
    /// multi-band GeoTIFF such as RGB+NIR and bands across distinct loaded
    /// rasters, plus an optional `R8Uint` SCL/QA mask layer).
    pub(crate) fn create_band_math_from_loaded(
        &mut self,
        spec: LocalBandMathSpec,
    ) -> Result<(LayerId, Option<FootprintBounds>, OverlayMutation), String> {
        let required = spec.formula.required_bands();
        if spec.inputs.len() != required {
            return Err(format!(
                "{} requires {required} input bands, got {}",
                spec.formula.label(),
                spec.inputs.len()
            ));
        }

        let mut band_assets = Vec::with_capacity(spec.inputs.len());
        for input in &spec.inputs {
            let layer = self
                .stack
                .get(input.layer_id)
                .ok_or_else(|| "selected source layer no longer exists".to_owned())?;
            let src_asset = layer
                .single_asset()
                .ok_or_else(|| "band-math inputs must be single-raster layers".to_owned())?;
            let stream_spec =
                stream_spec_for_band(BandKey::Index(input.band_index), PixelFormat::R32Float);
            let cloned = src_asset.clone_for_stream(input.label.clone(), stream_spec)?;
            band_assets.push(cloned);
        }

        let scl_asset = if let Some(scl_in) = &spec.scl {
            let layer = self
                .stack
                .get(scl_in.layer_id)
                .ok_or_else(|| "selected SCL mask layer no longer exists".to_owned())?;
            let src_asset = layer
                .single_asset()
                .ok_or_else(|| "SCL mask input must be a single-raster layer".to_owned())?;
            let stream_spec =
                stream_spec_for_band(BandKey::Index(scl_in.band_index), PixelFormat::R8Uint);
            Some(src_asset.clone_for_stream(scl_in.label.clone(), stream_spec)?)
        } else {
            None
        };

        let rgb_max_hint = band_assets.first().and_then(suggest_rgb_max_hint);
        let style = BandMathStyle::for_custom(spec.formula, scl_asset.is_some(), rgb_max_hint);
        let id = LayerId(alloc_seq(&mut self.next_layer_id));
        let title = if spec.title.trim().is_empty() {
            format!("{} Composite", spec.formula.short_label())
        } else {
            spec.title.trim().to_owned()
        };
        let layer = OverlayLayer::from_band_math(id, title, band_assets, scl_asset, style);
        let mut mutation = OverlayMutation::default();
        let bounds = self.insert_opened_layer(layer, &mut mutation);
        Ok((id, bounds, mutation))
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
    /// cannot race the reload, while preserving the layer (and any sibling
    /// layers sharing the same `SourceId`) if reopen fails.
    pub(crate) fn begin_reload_bypass_cache(
        &mut self,
        id: LayerId,
    ) -> Result<OverlayMutation, String> {
        let cancel = self.runtime.shutdown.child_token();
        let (mut request, prep, old_stream) = {
            let Some(layer) = self.stack.get_mut(id) else {
                return Err("layer no longer exists".to_owned());
            };
            let Some(local_path) = layer.local_path().map(Path::to_path_buf) else {
                return Err(
                    "remote COG and band-math layers do not support local disk reload".to_owned(),
                );
            };
            let old_stream = layer.meta().stream_key();
            layer.cancel_stats_job();
            let prep = layer.begin_reload(cancel.clone());
            let request = ReloadRequest {
                id,
                reload_seq: prep.reload_seq,
                crs_override_epoch: layer.geo().crs_override_epoch(),
                path: local_path,
                old_fingerprint_hex: layer.fingerprint_hex().to_owned(),
                clear_old_disk_cache: false,
                crs_override: layer.geo().crs_override().cloned(),
                cancel,
            };
            (request, prep, old_stream)
        };
        if let Some(prev) = prep.old_cancel {
            prev.cancel();
        }

        let unshared = self.stack.count_sharing_source(prep.old_source) <= 1;
        request.clear_old_disk_cache = unshared;

        // Cancel in-flight jobs, rotate source write tokens, purge queued disk
        // writes, and clear runtime caches BEFORE `io_worker` runs
        // `clear_source` so no pre-reload tile can poison SQLite, while
        // preserving sibling layers that share `old_source`.
        let mut purge_sources = Vec::new();
        if unshared {
            self.purge_source(prep.old_source);
            purge_sources.push(prep.old_source);
        } else if !self.stack.has_visible_source(prep.old_source) {
            self.purge_source(prep.old_source);
        } else {
            self.cancel_stream_if_unused(old_stream);
        }

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
            purge_sources,
        })
    }

    /// Drain completed background local open/reload results, remote COG &
    /// Band Math recipe open results, and sampled band-statistics completions.
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
            match remote {
                RemoteOpenResult::Single {
                    title,
                    display_url,
                    is_scl,
                    fly_to,
                    outcome,
                } => match outcome {
                    Ok((range_source, source, frame)) => {
                        let id = LayerId(alloc_seq(&mut self.next_layer_id));
                        let relaid = alloc_seq(&mut self.next_relaid);
                        let layer = OverlayLayer::from_remote_source(RemoteLayerSpec {
                            id,
                            title: title.clone(),
                            display_url: display_url.clone(),
                            range_source,
                            source,
                            frame,
                            options: LayerInitOptions {
                                visible: true,
                                opacity: 1.0,
                                crs_override: None,
                                style: None,
                            },
                            is_scl,
                            relaid,
                        });
                        let bounds = self.insert_opened_layer(layer, &mut mutation);
                        events.push(OverlayIoEvent::RemoteOpened {
                            id,
                            title,
                            display_url,
                            bounds,
                            fly_to,
                        });
                    }
                    Err(error) => {
                        events.push(OverlayIoEvent::RemoteFailed {
                            title,
                            display_url,
                            error,
                        });
                    }
                },
                RemoteOpenResult::BandMath {
                    title,
                    display_url,
                    preset,
                    fly_to,
                    outcome,
                } => match outcome {
                    Ok((opened_bands, opened_scl)) => {
                        let built_bands: Result<Vec<RasterAsset>, String> = opened_bands
                            .into_iter()
                            .map(|b| {
                                let relaid = alloc_seq(&mut self.next_relaid);
                                build_remote_band_asset(b, PixelFormat::R32Float, relaid)
                            })
                            .collect();
                        match built_bands {
                            Ok(band_assets) => {
                                let scl_asset = opened_scl.and_then(|s| {
                                    let relaid = alloc_seq(&mut self.next_relaid);
                                    match build_remote_band_asset(s, PixelFormat::R8Uint, relaid) {
                                        Ok(asset) => Some(asset),
                                        Err(err) => {
                                            tracing::warn!(
                                                title = %title,
                                                error = %err,
                                                "Optional SCL asset stream incompatible; continuing without cloud mask"
                                            );
                                            None
                                        }
                                    }
                                });
                                let id = LayerId(alloc_seq(&mut self.next_layer_id));
                                let style = BandMathStyle::from_preset(preset, scl_asset.is_some());
                                let layer = OverlayLayer::from_band_math(
                                    id,
                                    title.clone(),
                                    band_assets,
                                    scl_asset,
                                    style,
                                );
                                let bounds = self.insert_opened_layer(layer, &mut mutation);
                                events.push(OverlayIoEvent::RemoteOpened {
                                    id,
                                    title,
                                    display_url,
                                    bounds,
                                    fly_to,
                                });
                            }
                            Err(error) => {
                                events.push(OverlayIoEvent::RemoteFailed {
                                    title,
                                    display_url,
                                    error,
                                });
                            }
                        }
                    }
                    Err(error) => {
                        events.push(OverlayIoEvent::RemoteFailed {
                            title,
                            display_url,
                            error,
                        });
                    }
                },
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
        let bounds = layer.footprint();
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
        let (old_src, new_src, old_stream) = {
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
                    let old_stream = slot.meta().stream_key();
                    let new_src = source.source_id();
                    let relaid = alloc_seq(&mut self.next_relaid);
                    let old_src = slot.replace_reloaded(path, source, frame, relaid);
                    (old_src, new_src, old_stream)
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

        // Purge runtime/GPU state for `new_src`, and only purge `old_src` when
        // identical (`new_src == old_src`, explicit cache bypass) or when no
        // sibling layer still references `old_src`.
        if new_src == old_src {
            self.purge_source(new_src);
            mutation.purge_sources.push(new_src);
        } else {
            if self.stack.count_sharing_source(old_src) == 0 {
                self.purge_source(old_src);
                mutation.purge_sources.push(old_src);
            } else if !self.stack.has_visible_source(old_src) {
                self.purge_source(old_src);
            } else {
                self.cancel_stream_if_unused(old_stream);
            }
            self.purge_source(new_src);
            mutation.purge_sources.push(new_src);
        }
        mutation.changed = true;
        mutation.topology_changed = true;
        self.request_band_stats_if_needed(id);
        events.push(OverlayIoEvent::Reloaded { id });
    }
}
