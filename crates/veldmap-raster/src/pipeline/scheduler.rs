//! Native raster pyramid tile scheduler (`TileScheduler`, `PassBusy`,
//! `TileFailureOutcome`).
//!
//! Coordinates progressive coarsest-to-target tile fetching across both
//! concurrent pointwise reads and single-active sequential passes per
//! [`PassKey`] (`source`, `band`, `format`, `gutter`):
//! - Only 1 active sequential pass per `PassKey` is allowed for
//!   `ServeMode::Pass` (so multi-band files sharing one `SourceId` schedule
//!   independently per band).
//! - `ServeMode::Pointwise` levels run concurrently across multiple tiles and
//!   batches while deduplicating identical `TileKey`s.
//! - Coarser in-flight passes (`pass_level > target_level`) are never cancelled
//!   on zoom-in because their output serves as valid ancestor carriers.
//! - Terminal failures (`Reject` or 2nd consecutive transient failure) are
//!   suppressed permanently until `reset_source`.

use std::collections::{HashMap, HashSet};

use veldmap_core::sync::CancelToken;
use veldmap_core::tile::{SourceId, TileKey};

use super::plan::{PassKey, PyramidMeta, ServeMode};

/// Error returned when starting a sequential pass while one is already active
/// for the same [`PassKey`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PassBusy {
    pub key: PassKey,
    pub active_pass_id: u64,
    pub active_level: u8,
}

/// Classification of a failed tile request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TileFailureOutcome {
    /// First transient failure: removed from in-flight and eligible for one retry.
    RetryEligible,
    /// Terminal rejection or second consecutive failure: suppressed permanently
    /// until `forgive_single_pass_failures` (for single-pass forgiveness) or
    /// `reset_source`.
    TerminalFailed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FailureState {
    RetriedOnce,
    TransientPassSuppressed,
    Permanent,
}

struct ActivePass {
    pass_id: u64,
    level: u8,
    pass_from: Option<u8>,
    cancel: CancelToken,
    tiles: Vec<TileKey>,
}

/// Coordinates concurrent pointwise tile requests, single-active sequential
/// passes per [`PassKey`], non-cancellation of coarser passes, and
/// terminal-failure suppression.
pub struct TileScheduler {
    next_pass_id: u64,
    active_passes: HashMap<PassKey, ActivePass>,
    in_flight: HashSet<TileKey>,
    failures: HashMap<TileKey, FailureState>,
}

impl Default for TileScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl TileScheduler {
    /// Construct an empty [`TileScheduler`].
    pub fn new() -> Self {
        Self {
            next_pass_id: 1,
            active_passes: HashMap::new(),
            in_flight: HashSet::new(),
            failures: HashMap::new(),
        }
    }

    /// Return whether `key` currently has an active `ServeMode::Pass`.
    pub fn has_active_pass(&self, key: PassKey) -> bool {
        self.active_passes.contains_key(&key)
    }

    /// Return whether any band/format stream of `source` currently has an
    /// active `ServeMode::Pass`.
    pub fn has_any_active_pass(&self, source: SourceId) -> bool {
        self.active_passes.keys().any(|k| k.source == source)
    }

    /// Return the level of the active `ServeMode::Pass` for `key`, if any.
    pub fn active_pass_level(&self, key: PassKey) -> Option<u8> {
        self.active_passes.get(&key).map(|p| p.level)
    }

    /// Return whether `key` is currently marked in-flight.
    pub fn is_in_flight(&self, key: &TileKey) -> bool {
        self.in_flight.contains(key)
    }

    /// Return whether `key` is permanently or transiently suppressed from
    /// re-requesting.
    pub fn is_terminal_failed(&self, key: &TileKey) -> bool {
        matches!(
            self.failures.get(key),
            Some(FailureState::Permanent | FailureState::TransientPassSuppressed)
        )
    }

    fn filter_candidates(
        &self,
        meta: &PyramidMeta,
        level: u8,
        candidates: &[TileKey],
        is_cached: &impl Fn(&TileKey) -> bool,
        is_pointwise: bool,
    ) -> Vec<TileKey> {
        let expected_scheme = meta.scheme();
        let mut seen = HashSet::new();
        let mut eligible = Vec::new();
        for &key in candidates {
            if key.source != meta.source
                || key.stream_spec() != meta.spec
                || key.scheme != expected_scheme
                || key.level != level
                || !seen.insert(key)
            {
                continue;
            }
            let suppressed = if is_pointwise {
                self.is_terminal_failed(&key)
            } else {
                matches!(self.failures.get(&key), Some(FailureState::Permanent))
            };
            if is_cached(&key) || self.in_flight.contains(&key) || suppressed {
                continue;
            }
            eligible.push(key);
        }
        eligible
    }

    /// Admit unique fitting `ServeMode::Pointwise` tiles at `level` that are
    /// not cached, not already in-flight, and not suppressed, marking admitted
    /// keys as in-flight.
    ///
    /// Returns empty if `level` does not exist, does not fit (`!plan.fits`), or
    /// is not served pointwise.
    pub fn admit_pointwise(
        &mut self,
        meta: &PyramidMeta,
        level: u8,
        candidates: &[TileKey],
        is_cached: impl Fn(&TileKey) -> bool,
    ) -> Vec<TileKey> {
        let Some(plan) = meta.level(level) else {
            return Vec::new();
        };
        if !plan.fits || plan.serve != ServeMode::Pointwise {
            return Vec::new();
        }
        let admitted = self.filter_candidates(meta, level, candidates, &is_cached, true);
        for &key in &admitted {
            self.in_flight.insert(key);
        }
        admitted
    }

    /// Read-only check returning unique fitting `ServeMode::Pass` candidates at
    /// `level` that warrant launching a sequential pass (callers then pass the
    /// result into [`Self::begin_pass`]).
    ///
    /// Returns empty if `level` does not fit, is not `ServeMode::Pass`, or a
    /// pass is already active for `meta.pass_key()`. Permits first-time
    /// pass-failed tiles to trigger a recovery pass so pass-only sources never
    /// permanently starve.
    pub fn check_pass_candidates(
        &self,
        meta: &PyramidMeta,
        level: u8,
        candidates: &[TileKey],
        is_cached: impl Fn(&TileKey) -> bool,
    ) -> Vec<TileKey> {
        let Some(plan) = meta.level(level) else {
            return Vec::new();
        };
        if !plan.fits
            || !matches!(plan.serve, ServeMode::Pass { .. })
            || self.has_active_pass(meta.pass_key())
        {
            return Vec::new();
        }
        self.filter_candidates(meta, level, candidates, &is_cached, false)
    }

    /// Begin the single active `ServeMode::Pass` for `key` at `level`
    /// (with optional shared cascade root `pass_from`).
    ///
    /// Returns `Err(PassBusy)` if another sequential pass is already active
    /// for `key`. `ServeMode::Pointwise` requests must NOT call this method
    /// so pointwise tiles execute concurrently.
    pub fn begin_pass(
        &mut self,
        key: PassKey,
        level: u8,
        pass_from: Option<u8>,
        cancel: CancelToken,
        tiles: Vec<TileKey>,
    ) -> Result<u64, PassBusy> {
        if let Some(existing) = self.active_passes.get(&key) {
            return Err(PassBusy {
                key,
                active_pass_id: existing.pass_id,
                active_level: existing.level,
            });
        }
        debug_assert!(
            tiles.iter().all(|t| PassKey::from_tile_key(t) == key),
            "begin_pass tiles must match PassKey"
        );
        let pass_id = self.next_pass_id;
        self.next_pass_id = self.next_pass_id.wrapping_add(1).max(1);
        for &tile in &tiles {
            self.in_flight.insert(tile);
        }
        self.active_passes.insert(
            key,
            ActivePass {
                pass_id,
                level,
                pass_from,
                cancel,
                tiles,
            },
        );
        Ok(pass_id)
    }

    /// Update the desired target level (and its `pass_from`, if served via
    /// `ServeMode::Pass`) for `key`.
    ///
    /// Preserves coarser in-flight passes (`pass.level >= target_level`) and
    /// active passes sharing the same cascade `pass_from` (e.g. `Pass { from: 0 }`)
    /// so zooming never discards valid ancestor or shared-cascade work. Cancels
    /// an active pass only when it is strictly finer than `target_level` and
    /// does not cover `target_pass_from`.
    pub fn update_target(
        &mut self,
        key: PassKey,
        target_level: u8,
        target_pass_from: Option<u8>,
    ) -> bool {
        let should_cancel = self.active_passes.get(&key).is_some_and(|pass| {
            pass.level < target_level
                && !(target_pass_from.is_some() && pass.pass_from == target_pass_from)
        });
        if !should_cancel {
            return false;
        }
        if let Some(pass) = self.active_passes.remove(&key) {
            for tile in pass.tiles {
                self.in_flight.remove(&tile);
            }
            pass.cancel.cancel();
            return true;
        }
        false
    }

    /// Mark an active pass `pass_id` for `key` as completed.
    ///
    /// Clears its remaining in-flight keys; when `succeeded` is `false`, marks
    /// undelivered keys as transiently failed until a healthy pass forgives
    /// first-time failures (second failure becomes permanent).
    pub fn finish_pass(&mut self, key: PassKey, pass_id: u64, succeeded: bool) {
        let matches = self
            .active_passes
            .get(&key)
            .is_some_and(|p| p.pass_id == pass_id);
        if !matches {
            return;
        }
        if let Some(pass) = self.active_passes.remove(&key) {
            for tile in pass.tiles {
                if self.in_flight.remove(&tile) && !succeeded {
                    let next = match self.failures.get(&tile) {
                        None => FailureState::TransientPassSuppressed,
                        Some(
                            FailureState::RetriedOnce
                            | FailureState::TransientPassSuppressed
                            | FailureState::Permanent,
                        ) => FailureState::Permanent,
                    };
                    self.failures.insert(tile, next);
                }
            }
            if succeeded {
                self.forgive_single_pass_failures(key);
            }
        }
    }

    /// Record successful arrival of `key`, clearing its in-flight and failure
    /// flags.
    pub fn on_tile_arrived(&mut self, key: &TileKey) {
        self.in_flight.remove(key);
        self.failures.remove(key);
    }

    /// Record failure for `key`.
    ///
    /// When `terminal` is `true` (e.g. corrupt/unreadable payload or invalid
    /// header), suppresses `key` permanently. When `terminal` is `false`
    /// (transient I/O or decode stumble), allows one retry and suppresses
    /// permanently on the second consecutive failure.
    pub fn on_tile_failed(&mut self, key: &TileKey, terminal: bool) -> TileFailureOutcome {
        self.in_flight.remove(key);
        if terminal {
            self.failures.insert(*key, FailureState::Permanent);
            return TileFailureOutcome::TerminalFailed;
        }
        match self.failures.get(key) {
            None => {
                self.failures.insert(*key, FailureState::RetriedOnce);
                TileFailureOutcome::RetryEligible
            }
            Some(
                FailureState::RetriedOnce
                | FailureState::TransientPassSuppressed
                | FailureState::Permanent,
            ) => {
                self.failures.insert(*key, FailureState::Permanent);
                TileFailureOutcome::TerminalFailed
            }
        }
    }

    /// Forgive tiles of `pass_key` that only failed once in a prior broken pass
    /// (demoting them to `RetriedOnce` so a subsequent healthy pass allows one
    /// retry while a second failure becomes permanent).
    pub fn forgive_single_pass_failures(&mut self, pass_key: PassKey) {
        for (key, state) in &mut self.failures {
            if PassKey::from_tile_key(key) == pass_key
                && *state == FailureState::TransientPassSuppressed
            {
                *state = FailureState::RetriedOnce;
            }
        }
    }

    /// Cancel every active pass for `source` across all bands/formats and drop
    /// all in-flight/failure state associated with `source`.
    pub fn reset_source(&mut self, source: SourceId) {
        self.active_passes.retain(|k, pass| {
            if k.source == source {
                pass.cancel.cancel();
                false
            } else {
                true
            }
        });
        self.in_flight.retain(|k| k.source != source);
        self.failures.retain(|k, _| k.source != source);
    }
}

#[cfg(test)]
#[path = "scheduler_tests.rs"]
mod tests;
