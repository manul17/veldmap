//! Asynchronous local/remote raster open, multi-asset STAC Band Math recipe
//! open, local Band Math builder construction, cache-bypassing reload, and
//! background completion reconciliation for [`OverlayManager`].

use std::path::Path;

use veldmap_core::geo::FootprintBounds;
use veldmap_core::tile::{BandKey, PixelFormat};
use veldmap_raster::{SampleType, stream_spec_for_band};

use crate::clamp_opacity;

use super::super::asset::{GeoBinding, LayerOrigin, RasterAsset, has_affine_scaling};
use super::super::band_math::BandMathStyle;
use super::super::io_worker::{
    OpenedPayload, OverlayIoCmd, OverlayIoResult, ReloadRequest, ReloadedPayload,
};
use super::super::layer::RemoteLayerSpec;
use super::super::opener::bind_frame_with_override;
use super::super::remote_opener::{
    OpenedBandAsset, OpenedBandAssetOrigin, RemoteBandMathOpenPayload, RemoteOpenResult,
    RemoteSingleOpenPayload,
};
use super::super::stack::{LayerStack, OverlayLayer};
use super::super::{
    LayerId, LayerInitOptions, LocalBandMathInput, LocalBandMathSpec, OpenOverlayRequest,
    OverlayIoEvent, RemoteBandMathRequest, RemoteCogRequest,
};
use super::{OverlayManager, OverlayMutation, alloc_seq};

fn clone_loaded_input_asset(
    stack: &LayerStack,
    input: &LocalBandMathInput,
    format: PixelFormat,
    missing_err: &str,
    kind_err: &str,
) -> Result<RasterAsset, String> {
    let layer = stack
        .get(input.layer_id)
        .ok_or_else(|| missing_err.to_owned())?;
    let src_asset = layer.single_asset().ok_or_else(|| kind_err.to_owned())?;
    let stream_spec = stream_spec_for_band(BandKey::Index(input.band_index), format);
    src_asset.clone_for_stream(input.label.clone(), stream_spec)
}

fn build_band_asset(
    opened: OpenedBandAsset,
    format: PixelFormat,
    relaid: u64,
) -> Result<RasterAsset, String> {
    let spec = stream_spec_for_band(BandKey::Index(opened.band_index), format);
    let meta = opened
        .source
        .pyramid_meta(spec)
        .map_err(|e| format!("band '{}': {e}", opened.label))?;
    let desc = opened.source.description();
    let (origin, crs_override) = match opened.origin {
        OpenedBandAssetOrigin::Local { path, crs_override } => {
            (LayerOrigin::Local { path }, crs_override)
        }
        OpenedBandAssetOrigin::Remote { meta, range_source } => (
            LayerOrigin::Remote {
                meta,
                range: range_source,
            },
            None,
        ),
    };
    let geo = GeoBinding::new(
        opened.frame,
        desc.width,
        desc.height,
        meta.max_level(),
        crs_override,
        relaid,
    );
    Ok(RasterAsset::new(
        opened.label,
        origin,
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
    if has_affine_scaling(band_info) {
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
        self.remote_opener.begin_open(
            &self.runtime,
            self.workspace_epoch,
            &self.workspace_cancel,
            request,
        )
    }

    /// Queue a non-blocking concurrent multi-asset remote STAC Band Math / RGB
    /// recipe open, spawning one `LayerKind::BandMath` layer without duplicate
    /// raw layers or redundant camera flights.
    pub(crate) fn begin_open_remote_band_math(
        &self,
        request: RemoteBandMathRequest,
    ) -> Result<(), String> {
        self.remote_opener.begin_open_band_math(
            &self.runtime,
            self.workspace_epoch,
            &self.workspace_cancel,
            request,
        )
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
            band_assets.push(clone_loaded_input_asset(
                &self.stack,
                input,
                PixelFormat::R32Float,
                "selected source layer no longer exists",
                "band-math inputs must be single-raster layers",
            )?);
        }

        let scl_asset = match &spec.scl {
            Some(scl_in) => Some(clone_loaded_input_asset(
                &self.stack,
                scl_in,
                PixelFormat::R8Uint,
                "selected SCL mask layer no longer exists",
                "SCL mask input must be a single-raster layer",
            )?),
            None => None,
        };

        let rgb_max_hint = band_assets.first().and_then(suggest_rgb_max_hint);
        let style = BandMathStyle::for_custom(spec.formula, scl_asset.is_some(), rgb_max_hint);
        let id = LayerId(alloc_seq(&mut self.next_layer_id));
        let title = if spec.title.trim().is_empty() {
            format!("{} Composite", spec.formula.short_label())
        } else {
            spec.title.trim().to_owned()
        };
        let layer =
            OverlayLayer::from_band_math(id, title, band_assets, scl_asset, style, true, 1.0);
        let mut mutation = OverlayMutation::default();
        let bounds = self.insert_opened_layer(layer, &mut mutation);
        Ok((id, bounds, mutation))
    }

    /// Queue a non-blocking local raster open with explicit initial visibility,
    /// opacity, optional CRS override, and optional saved analytical style.
    pub(crate) fn begin_open_file(&self, request: OpenOverlayRequest<'_>) -> Result<(), String> {
        if self.io_worker.send(OverlayIoCmd::Open {
            epoch: self.workspace_epoch,
            cancel: self.workspace_cancel.child_token(),
            path: request.path.to_path_buf(),
            fly_to: request.fly_to,
            options: LayerInitOptions {
                visible: request.visible,
                opacity: clamp_opacity(request.opacity),
                crs_override: request.crs_override,
                style: request.style,
            },
            restore_id: request.restore_id,
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
        let cancel = self.workspace_cancel.child_token();
        let (mut request, prep, old_stream) = {
            let Some(layer) = self.stack.get_mut(id) else {
                return Err("layer no longer exists".to_owned());
            };
            let Some(local_path) = layer.local_path().map(Path::to_path_buf) else {
                return Err(
                    "remote COG and band-math layers do not support local disk reload".to_owned(),
                );
            };
            let Some(asset) = layer.single_asset() else {
                return Err(
                    "remote COG and band-math layers do not support local disk reload".to_owned(),
                );
            };
            let old_stream = asset.meta().stream_key();
            let crs_override_epoch = asset.geo().crs_override_epoch();
            let old_fingerprint_hex = asset.fingerprint_hex().to_owned();
            let crs_override = asset.geo().crs_override().cloned();
            layer.cancel_stats_job();
            let prep = layer.begin_reload(cancel.clone());
            let request = ReloadRequest {
                id,
                reload_seq: prep.reload_seq,
                crs_override_epoch,
                path: local_path,
                old_fingerprint_hex,
                clear_old_disk_cache: false,
                crs_override,
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
        self.release_sources_and_streams(
            &[prep.old_source],
            &[old_stream],
            Some(1),
            &mut purge_sources,
        );

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
                OverlayIoResult::Opened(payload) => {
                    if payload.epoch != self.workspace_epoch {
                        continue;
                    }
                    self.handle_opened_result(payload, &mut events, &mut mutation);
                }
                OverlayIoResult::Reloaded(payload) => {
                    self.handle_reloaded_result(payload, &mut events, &mut mutation);
                }
            }
        }

        for remote in self.remote_opener.drain() {
            match remote {
                RemoteOpenResult::Single(payload) => {
                    if payload.epoch != self.workspace_epoch {
                        continue;
                    }
                    self.handle_remote_single_result(payload, &mut events, &mut mutation);
                }
                RemoteOpenResult::BandMath(payload) => {
                    if payload.epoch != self.workspace_epoch {
                        continue;
                    }
                    self.handle_remote_band_math_result(payload, &mut events, &mut mutation);
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
        let bounds = layer.footprint();
        self.stack.push(layer);
        self.request_band_stats_if_needed(id);
        mutation.changed = true;
        mutation.topology_changed = true;
        bounds
    }

    fn handle_opened_result(
        &mut self,
        payload: OpenedPayload,
        events: &mut Vec<OverlayIoEvent>,
        mutation: &mut OverlayMutation,
    ) {
        let OpenedPayload {
            path,
            fly_to,
            options,
            restore_id,
            outcome,
            ..
        } = payload;
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
                    restore_id,
                });
            }
            Err(err) => {
                events.push(OverlayIoEvent::Failed {
                    path,
                    error: err.message,
                    can_assign_crs: err.can_assign_crs,
                    restore_id,
                });
            }
        }
    }

    fn handle_remote_single_result(
        &mut self,
        payload: RemoteSingleOpenPayload,
        events: &mut Vec<OverlayIoEvent>,
        mutation: &mut OverlayMutation,
    ) {
        let RemoteSingleOpenPayload {
            title,
            meta,
            options,
            fly_to,
            restore_id,
            outcome,
            ..
        } = payload;
        match outcome {
            Ok((range_source, source, frame)) => {
                let id = LayerId(alloc_seq(&mut self.next_layer_id));
                let relaid = alloc_seq(&mut self.next_relaid);
                let display_url = meta.display_url.clone();
                let layer = OverlayLayer::from_remote_source(RemoteLayerSpec {
                    id,
                    title: title.clone(),
                    meta,
                    range_source,
                    source,
                    frame,
                    options,
                    relaid,
                });
                let bounds = self.insert_opened_layer(layer, mutation);
                events.push(OverlayIoEvent::RemoteOpened {
                    id,
                    title,
                    display_url,
                    is_band_math: false,
                    bounds,
                    fly_to,
                    restore_id,
                });
            }
            Err(error) => {
                events.push(OverlayIoEvent::RemoteFailed {
                    title,
                    display_url: meta.display_url,
                    is_band_math: false,
                    error,
                    restore_id,
                });
            }
        }
    }

    fn handle_remote_band_math_result(
        &mut self,
        payload: RemoteBandMathOpenPayload,
        events: &mut Vec<OverlayIoEvent>,
        mutation: &mut OverlayMutation,
    ) {
        let RemoteBandMathOpenPayload {
            title,
            display_url,
            preset,
            visible,
            opacity,
            style: saved_style,
            fly_to,
            restore_id,
            outcome,
            ..
        } = payload;
        let built = outcome.and_then(|(opened_bands, opened_scl)| {
            let band_assets: Vec<RasterAsset> = opened_bands
                .into_iter()
                .map(|b| {
                    let relaid = alloc_seq(&mut self.next_relaid);
                    build_band_asset(b, PixelFormat::R32Float, relaid)
                })
                .collect::<Result<_, _>>()?;
            let scl_asset = opened_scl.and_then(|s| {
                let relaid = alloc_seq(&mut self.next_relaid);
                match build_band_asset(s, PixelFormat::R8Uint, relaid) {
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
            Ok((band_assets, scl_asset))
        });

        match built {
            Ok((band_assets, scl_asset)) => {
                let id = LayerId(alloc_seq(&mut self.next_layer_id));
                let style = match saved_style {
                    Some(spec) => BandMathStyle::from_spec(&spec, scl_asset.is_some()),
                    None => BandMathStyle::from_preset(preset, scl_asset.is_some()),
                };
                let layer = OverlayLayer::from_band_math(
                    id,
                    title.clone(),
                    band_assets,
                    scl_asset,
                    style,
                    visible,
                    opacity,
                );
                let bounds = self.insert_opened_layer(layer, mutation);
                events.push(OverlayIoEvent::RemoteOpened {
                    id,
                    title,
                    display_url,
                    is_band_math: true,
                    bounds,
                    fly_to,
                    restore_id,
                });
            }
            Err(error) => {
                events.push(OverlayIoEvent::RemoteFailed {
                    title,
                    display_url,
                    is_band_math: true,
                    error,
                    restore_id,
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
                    let (live_epoch, live_override, old_stream) = {
                        let Some(asset) = slot.single_asset() else {
                            return;
                        };
                        (
                            asset.geo().crs_override_epoch(),
                            asset.geo().crs_override().cloned(),
                            asset.meta().stream_key(),
                        )
                    };
                    if live_epoch != crs_override_epoch {
                        match bind_frame_with_override(source.description(), live_override.as_ref())
                        {
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
        if new_src != old_src {
            self.release_sources_and_streams(
                &[old_src],
                &[old_stream],
                Some(0),
                &mut mutation.purge_sources,
            );
        }
        self.purge_source(new_src);
        mutation.purge_sources.push(new_src);
        mutation.changed = true;
        mutation.topology_changed = true;
        self.request_band_stats_if_needed(id);
        events.push(OverlayIoEvent::Reloaded { id });
    }
}
