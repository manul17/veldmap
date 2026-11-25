//! Sequential pass (`ServeMode::Pass`) bulk-lane decode jobs, QOI hit draining,
//! and RAII flight guard.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use veldmap_core::sync::{CancelToken, lock_unpoisoned};
use veldmap_core::tile::{DecodedTile, PixelFormat, TileKey};
use veldmap_raster::{
    AdmittedPass, DecodeJob, DecodeJobKey, PassKey, PyramidMeta, RasterError, RasterSource,
    generate_linear_mipmaps,
};
use veldmap_storage::DiskTileKey;

use super::super::default_display_tile_bytes;
use super::super::qoi_cache::{QoiWriteBatch, load_qoi_display_tile};
use super::super::runtime::OverlayRuntime;

pub(super) struct PassJobSpec {
    pub(super) admitted: AdmittedPass,
    pub(super) wanted_shared: Arc<Mutex<HashSet<TileKey>>>,
    pub(super) source: Arc<dyn RasterSource>,
    pub(super) meta: PyramidMeta,
    pub(super) fingerprint_hex: String,
    pub(super) runtime: Arc<OverlayRuntime>,
}

/// RAII guard guaranteeing an active sequential pass in
/// [`veldmap_raster::TileScheduler`] is always closed even if evicted from
/// `DecodeQueue` or unwound by a worker panic.
struct PassFlightGuard {
    runtime: Arc<OverlayRuntime>,
    pass_key: PassKey,
    pass_id: u64,
    finished: bool,
}

impl PassFlightGuard {
    fn new(runtime: Arc<OverlayRuntime>, pass_key: PassKey, pass_id: u64) -> Self {
        Self {
            runtime,
            pass_key,
            pass_id,
            finished: false,
        }
    }

    fn finish(mut self, succeeded: bool) {
        self.finished = true;
        if self
            .runtime
            .finish_pass(self.pass_key, self.pass_id, succeeded)
        {
            self.runtime.request_reschedule();
        }
    }

    fn abort(mut self) {
        self.finished = true;
        self.cleanup_aborted();
    }

    fn cleanup_aborted(&self) {
        if self.runtime.abort_pass(self.pass_key, self.pass_id) {
            self.runtime.request_reschedule();
        }
    }
}

impl Drop for PassFlightGuard {
    fn drop(&mut self) {
        if !self.finished {
            self.cleanup_aborted();
        }
    }
}

fn deliver_pass_tile(
    rt: &OverlayRuntime,
    pass_key: PassKey,
    pass_id: u64,
    tk: TileKey,
    tile: DecodedTile,
    cancel: &CancelToken,
) -> bool {
    if matches!(rt.done_tx.send_if_uncancelled((tk, tile), cancel), Ok(true)) {
        rt.on_pass_tile_arrived(pass_key, pass_id, &tk);
        (rt.wakeup)();
        true
    } else {
        false
    }
}

enum QoiDrainOutcome {
    Cancelled,
    AllHit,
    NeedsPass(HashSet<TileKey>),
}

/// Stream any QOI disk-cache hits for `spec.admitted.tiles` immediately
/// (never buffering an entire rung in RAM or discarding partial hits on 1 miss).
fn drain_pass_qoi_hits(spec: &PassJobSpec) -> QoiDrainOutcome {
    let Some(ref disk) = spec.runtime.disk_cache else {
        return QoiDrainOutcome::NeedsPass(HashSet::new());
    };
    let Some(&first_tk) = spec.admitted.tiles.first() else {
        return QoiDrainOutcome::AllHit;
    };
    let pass_key = spec.admitted.key;
    let pass_id = spec.admitted.pass_id;
    let pass_cancel = &spec.admitted.cancel;
    let mut already_hit: HashSet<TileKey> = HashSet::new();
    let mut any_miss = false;
    let mut dk = DiskTileKey::from_tile_key(spec.fingerprint_hex.clone(), &first_tk);
    for &tk in &spec.admitted.tiles {
        if pass_cancel.is_cancelled() {
            return QoiDrainOutcome::Cancelled;
        }
        dk.level = tk.level;
        dk.x = tk.x;
        dk.y = tk.y;
        if let Some(tile) = load_qoi_display_tile(disk, &dk) {
            if deliver_pass_tile(&spec.runtime, pass_key, pass_id, tk, tile, pass_cancel) {
                already_hit.insert(tk);
            }
        } else {
            any_miss = true;
        }
    }
    if any_miss {
        QoiDrainOutcome::NeedsPass(already_hit)
    } else {
        QoiDrainOutcome::AllHit
    }
}

/// Execute the source's sequential `produce_pass`, batch-writing newly produced
/// display tiles to QOI disk cache and uploading wanted/apex tiles live.
fn run_source_pass(spec: &PassJobSpec, already_hit: &HashSet<TileKey>) -> Result<(), RasterError> {
    let pass_key = spec.admitted.key;
    let pass_id = spec.admitted.pass_id;
    let pass_cancel = &spec.admitted.cancel;
    let initial_missing: HashSet<TileKey> = spec
        .admitted
        .tiles
        .iter()
        .copied()
        .filter(|tk| !already_hit.contains(tk))
        .collect();
    let max_level = spec.meta.max_level();
    let mut qoi_batch = QoiWriteBatch::new();
    let ctx = spec.runtime.decode_context(pass_cancel);

    let result = spec
        .source
        .produce_pass(&ctx, &mut |level, tx, ty, tw, th, rgba| {
            if pass_cancel.is_cancelled() {
                return Err(RasterError::Cancelled);
            }
            let tk = spec.meta.tile_key(level, tx, ty);
            if already_hit.contains(&tk) {
                return Ok(());
            }
            if let Some(ref disk) = spec.runtime.disk_cache {
                qoi_batch.push_and_maybe_flush(
                    disk,
                    DiskTileKey::from_tile_key(spec.fingerprint_hex.clone(), &tk),
                    tw,
                    th,
                    &rgba,
                    pass_cancel,
                );
            }

            // Read live `wanted_shared` briefly (never nesting locks) so mid-pass
            // zooms upload tiles matching the current viewport.
            let should_upload = level == max_level
                || initial_missing.contains(&tk)
                || lock_unpoisoned(&spec.wanted_shared).contains(&tk);
            if should_upload {
                let mut tile = DecodedTile::try_new(tw, th, PixelFormat::Rgba8UnormSrgb, rgba)
                    .map_err(|e| RasterError::Decode(e.to_string()))?;
                generate_linear_mipmaps(&mut tile);
                let _ = deliver_pass_tile(&spec.runtime, pass_key, pass_id, tk, tile, pass_cancel);
            }
            Ok(())
        });

    if let Some(ref disk) = spec.runtime.disk_cache {
        if result.is_ok() {
            qoi_batch.flush(disk, pass_cancel);
        } else {
            qoi_batch.clear();
        }
    }

    result
}

fn run_pass_job(spec: PassJobSpec, guard: PassFlightGuard) {
    let already_hit = match drain_pass_qoi_hits(&spec) {
        QoiDrainOutcome::Cancelled => {
            guard.abort();
            return;
        }
        QoiDrainOutcome::AllHit => {
            guard.finish(true);
            return;
        }
        QoiDrainOutcome::NeedsPass(hits) => hits,
    };

    match run_source_pass(&spec, &already_hit) {
        Ok(()) => guard.finish(true),
        Err(err) if err.is_cancelled() => guard.abort(),
        Err(err) => {
            tracing::warn!(
                source = ?spec.meta.source,
                error = %err,
                "Sequential overlay pass failed"
            );
            guard.finish(false);
        }
    }
}

pub(super) fn submit_pass_job(spec: PassJobSpec) {
    let pass_key = spec.admitted.key;
    let pass_id = spec.admitted.pass_id;
    let pass_cancel = spec.admitted.cancel.clone();
    let runtime = Arc::clone(&spec.runtime);
    let queued_bytes = spec
        .source
        .pass_peak_bytes()
        .min(runtime.budget.bulk_lane_pending_bytes as u64) as usize;
    let guard = PassFlightGuard::new(Arc::clone(&runtime), pass_key, pass_id);

    let job = DecodeJob {
        key: DecodeJobKey::Pass(pass_key),
        queued_bytes: queued_bytes.max(default_display_tile_bytes()),
        cancel: pass_cancel,
        work: Box::new(move || run_pass_job(spec, guard)),
    };

    runtime.try_submit_guarded(&runtime.bulk_pool, job, true, || {
        let _ = runtime.abort_pass(pass_key, pass_id);
    });
}
