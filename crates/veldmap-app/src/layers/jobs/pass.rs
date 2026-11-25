//! Sequential pass (`ServeMode::Pass`) bulk-lane decode jobs, disk-cache hit
//! draining (`Qoi` & `Lz4Raw`), and RAII flight guard.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use veldmap_core::sync::{CancelToken, lock_unpoisoned};
use veldmap_core::tile::{DecodedTile, RASTER_TILE_SIZE, StreamKey, TileKey};
use veldmap_raster::{
    AdmittedPass, DecodeJob, DecodeJobKey, PyramidMeta, RasterError, RasterSource,
    generate_tile_mipmaps,
};
use veldmap_storage::DiskTileKey;

use crate::gpu_tiles::tile_format_bytes;

use super::super::disk_cache::{TileWriteBatch, load_cached_tile};
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
    stream_key: StreamKey,
    pass_id: u64,
    finished: bool,
}

impl PassFlightGuard {
    fn new(runtime: Arc<OverlayRuntime>, stream_key: StreamKey, pass_id: u64) -> Self {
        Self {
            runtime,
            stream_key,
            pass_id,
            finished: false,
        }
    }

    fn finish(mut self, succeeded: bool) {
        self.finished = true;
        if self
            .runtime
            .finish_pass(self.stream_key, self.pass_id, succeeded)
        {
            self.runtime.request_reschedule();
        }
    }

    fn abort(mut self) {
        self.finished = true;
        self.cleanup_aborted();
    }

    fn cleanup_aborted(&self) {
        if self.runtime.abort_pass(self.stream_key, self.pass_id) {
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
    stream_key: StreamKey,
    pass_id: u64,
    tk: TileKey,
    tile: DecodedTile,
    cancel: &CancelToken,
) -> bool {
    if matches!(rt.done_tx.send_if_uncancelled((tk, tile), cancel), Ok(true)) {
        rt.on_pass_tile_arrived(stream_key, pass_id, &tk);
        (rt.wakeup)();
        true
    } else {
        false
    }
}

enum CacheDrainOutcome {
    Cancelled,
    AllHit,
    NeedsPass(HashSet<TileKey>),
}

/// Stream any disk-cache hits (`Qoi` or `Lz4Raw`) for `spec.admitted.tiles`
/// immediately (never buffering an entire rung in RAM or discarding partial
/// hits on 1 miss).
fn drain_pass_cache_hits(spec: &PassJobSpec) -> CacheDrainOutcome {
    let Some(ref disk) = spec.runtime.disk_cache else {
        return CacheDrainOutcome::NeedsPass(HashSet::new());
    };
    let Some(&first_tk) = spec.admitted.tiles.first() else {
        return CacheDrainOutcome::AllHit;
    };
    let stream_key = spec.admitted.key;
    let pass_id = spec.admitted.pass_id;
    let pass_cancel = &spec.admitted.cancel;
    let mut already_hit: HashSet<TileKey> = HashSet::new();
    let mut any_miss = false;
    let mut dk = DiskTileKey::from_tile_key(spec.fingerprint_hex.clone(), &first_tk);
    for &tk in &spec.admitted.tiles {
        if pass_cancel.is_cancelled() {
            return CacheDrainOutcome::Cancelled;
        }
        dk.level = tk.level;
        dk.x = tk.x;
        dk.y = tk.y;
        if let Some(tile) = load_cached_tile(disk, &dk) {
            if deliver_pass_tile(&spec.runtime, stream_key, pass_id, tk, tile, pass_cancel) {
                already_hit.insert(tk);
            }
        } else {
            any_miss = true;
        }
    }
    if any_miss {
        CacheDrainOutcome::NeedsPass(already_hit)
    } else {
        CacheDrainOutcome::AllHit
    }
}

/// Execute the source's sequential `produce_pass`, batch-writing newly
/// produced tiles (`Qoi` or `Lz4Raw`) to disk cache and uploading wanted/apex
/// tiles live.
fn run_source_pass(spec: &PassJobSpec, already_hit: &HashSet<TileKey>) -> Result<(), RasterError> {
    let stream_key = spec.admitted.key;
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
    let stream_spec = spec.meta.spec;
    let mut write_batch = TileWriteBatch::new();
    let ctx = spec.runtime.decode_context(pass_cancel);

    let result =
        spec.source
            .produce_pass(stream_spec, &ctx, &mut |level, tx, ty, tw, th, bytes| {
                if pass_cancel.is_cancelled() {
                    return Err(RasterError::Cancelled);
                }
                let tk = spec.meta.tile_key(level, tx, ty);
                if already_hit.contains(&tk) {
                    return Ok(());
                }
                let mut tile = DecodedTile::try_new(tw, th, stream_spec.format, bytes)
                    .map_err(|e| RasterError::Decode(e.to_string()))?;
                if let Some(ref disk) = spec.runtime.disk_cache {
                    write_batch.push_and_maybe_flush(
                        disk,
                        DiskTileKey::from_tile_key(spec.fingerprint_hex.clone(), &tk),
                        &tile,
                        pass_cancel,
                    );
                }

                // Read live `wanted_shared` briefly (never nesting locks) so mid-pass
                // zooms upload tiles matching the current viewport.
                let should_upload = level == max_level
                    || initial_missing.contains(&tk)
                    || lock_unpoisoned(&spec.wanted_shared).contains(&tk);
                if should_upload {
                    generate_tile_mipmaps(&mut tile);
                    let _ = deliver_pass_tile(
                        &spec.runtime,
                        stream_key,
                        pass_id,
                        tk,
                        tile,
                        pass_cancel,
                    );
                }
                Ok(())
            });

    if let Some(ref disk) = spec.runtime.disk_cache {
        if result.is_ok() {
            write_batch.flush(disk, pass_cancel);
        } else {
            write_batch.clear();
        }
    }

    result
}

fn run_pass_job(spec: PassJobSpec, guard: PassFlightGuard) {
    let already_hit = match drain_pass_cache_hits(&spec) {
        CacheDrainOutcome::Cancelled => {
            guard.abort();
            return;
        }
        CacheDrainOutcome::AllHit => {
            guard.finish(true);
            return;
        }
        CacheDrainOutcome::NeedsPass(hits) => hits,
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
    let stream_key = spec.admitted.key;
    let pass_id = spec.admitted.pass_id;
    let pass_cancel = spec.admitted.cancel.clone();
    let runtime = Arc::clone(&spec.runtime);
    let queued_bytes = spec
        .source
        .pass_peak_bytes(spec.meta.spec)
        .min(runtime.budget.bulk_lane_pending_bytes as u64) as usize;
    let guard = PassFlightGuard::new(Arc::clone(&runtime), stream_key, pass_id);

    let job = DecodeJob {
        key: DecodeJobKey::Pass(stream_key),
        queued_bytes: queued_bytes.max(tile_format_bytes(RASTER_TILE_SIZE, spec.meta.spec.format)),
        cancel: pass_cancel,
        work: Box::new(move || run_pass_job(spec, guard)),
    };

    runtime.try_submit_guarded(&runtime.bulk_pool, job, true, || {
        let _ = runtime.abort_pass(stream_key, pass_id);
    });
}
