//! Native raster pyramid tile scheduler (`TileScheduler`, `AdmittedTile`,
//! `PassBusy`, `TileFailureOutcome`).
//!
//! Coordinates progressive coarsest-to-target tile fetching across both
//! concurrent pointwise reads and single-active sequential passes per
//! [`PassKey`] (`source`, `band`, `format`, `gutter`):
//! - Only 1 active sequential pass per `PassKey` is allowed for
//!   `ServeMode::Pass` (so multi-band files sharing one `SourceId` schedule
//!   independently per band).
//! - `ServeMode::Pointwise` levels run concurrently with versioned
//!   `(job_id, CancelToken)` flight entries so stale cancelled workers can
//!   never clobber fresh admissions or poison caches.
//! - Coarser in-flight passes (`pass_level > target_level`) are never cancelled
//!   on zoom-in because their output serves as valid ancestor carriers.
//! - Terminal failures (`Reject` or 2nd consecutive transient failure) are
//!   suppressed permanently until `reset_source`.
//! - Cancellation-producing methods return [`CancelToken`]s rather than firing
//!   cancel callbacks while the caller holds a `Mutex<TileScheduler>`.

use std::collections::{HashMap, HashSet};

use veldmap_core::sync::CancelToken;
use veldmap_core::tile::{SourceId, TileKey};

use super::plan::{PassKey, PyramidMeta, ServeMode};

/// Pointwise tile admitted by [`TileScheduler::admit_pointwise`] with its
/// versioned `job_id` and per-job [`CancelToken`].
#[derive(Clone, Debug)]
pub struct AdmittedTile {
    pub key: TileKey,
    pub job_id: u64,
    pub cancel: CancelToken,
}

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

enum FlightEntry {
    Pointwise { job_id: u64, cancel: CancelToken },
    Pass { pass_id: u64 },
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
    next_job_id: u64,
    next_pass_id: u64,
    active_passes: HashMap<PassKey, ActivePass>,
    in_flight: HashMap<TileKey, FlightEntry>,
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
            next_job_id: 1,
            next_pass_id: 1,
            active_passes: HashMap::new(),
            in_flight: HashMap::new(),
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
        self.in_flight.contains_key(key)
    }

    /// Return the number of tiles currently in-flight for `source`.
    pub fn in_flight_count_for_source(&self, source: SourceId) -> usize {
        self.in_flight.keys().filter(|k| k.source == source).count()
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
            if is_cached(&key) || self.in_flight.contains_key(&key) || suppressed {
                continue;
            }
            eligible.push(key);
        }
        eligible
    }

    /// Admit unique fitting `ServeMode::Pointwise` tiles at `level` that are
    /// not cached, not already in-flight, and not suppressed, assigning each
    /// admitted tile a unique `job_id` and child [`CancelToken`] derived from
    /// `parent_cancel`.
    ///
    /// Returns empty if `level` does not exist, does not fit (`!plan.fits`), or
    /// is not served pointwise.
    pub fn admit_pointwise(
        &mut self,
        meta: &PyramidMeta,
        level: u8,
        candidates: &[TileKey],
        is_cached: impl Fn(&TileKey) -> bool,
        parent_cancel: &CancelToken,
    ) -> Vec<AdmittedTile> {
        let Some(plan) = meta.level(level) else {
            return Vec::new();
        };
        if !plan.fits || plan.serve != ServeMode::Pointwise {
            return Vec::new();
        }
        let keys = self.filter_candidates(meta, level, candidates, &is_cached, true);
        let mut admitted = Vec::with_capacity(keys.len());
        for key in keys {
            let job_id = self.next_job_id;
            self.next_job_id = self.next_job_id.wrapping_add(1).max(1);
            let cancel = parent_cancel.child_token();
            self.in_flight.insert(
                key,
                FlightEntry::Pointwise {
                    job_id,
                    cancel: cancel.clone(),
                },
            );
            admitted.push(AdmittedTile {
                key,
                job_id,
                cancel,
            });
        }
        admitted
    }

    /// Remove and collect [`CancelToken`]s for in-flight **pointwise** jobs of
    /// `source` that are no longer in `wanted` or are already cached.
    ///
    /// Never touches `ServeMode::Pass` flight entries. Callers should invoke
    /// `token.cancel()` on every returned token after releasing any lock around
    /// `TileScheduler`.
    pub fn reconcile_pointwise(
        &mut self,
        source: SourceId,
        wanted: &HashSet<TileKey>,
        is_cached: impl Fn(&TileKey) -> bool,
    ) -> Vec<CancelToken> {
        let mut cancelled = Vec::new();
        self.in_flight.retain(|key, entry| {
            if key.source != source {
                return true;
            }
            match entry {
                FlightEntry::Pointwise { cancel, .. } => {
                    if !wanted.contains(key) || is_cached(key) {
                        cancelled.push(cancel.clone());
                        false
                    } else {
                        true
                    }
                }
                FlightEntry::Pass { .. } => true,
            }
        });
        cancelled
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
            self.in_flight.insert(tile, FlightEntry::Pass { pass_id });
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

    fn take_active_pass(&mut self, key: PassKey, pass_id: u64) -> Option<ActivePass> {
        let matches = self
            .active_passes
            .get(&key)
            .is_some_and(|p| p.pass_id == pass_id);
        if !matches {
            return None;
        }
        self.active_passes.remove(&key)
    }

    fn remove_pass_in_flight(&mut self, tile: &TileKey, pass_id: u64) -> bool {
        let matches = matches!(
            self.in_flight.get(tile),
            Some(FlightEntry::Pass { pass_id: id }) if *id == pass_id
        );
        if matches {
            self.in_flight.remove(tile);
        }
        matches
    }

    /// Update the desired target level (and its `pass_from`, if served via
    /// `ServeMode::Pass`) for `key`.
    ///
    /// Preserves coarser in-flight passes (`pass.level >= target_level`) and
    /// active passes sharing the same cascade `pass_from` (e.g. `Pass { from: 0 }`)
    /// so zooming never discards valid ancestor or shared-cascade work. Returns
    /// `Some(cancel_token)` when a strictly finer pass was removed so the
    /// caller can invoke `cancel()` outside any scheduler lock.
    pub fn update_target(
        &mut self,
        key: PassKey,
        target_level: u8,
        target_pass_from: Option<u8>,
    ) -> Option<CancelToken> {
        let should_cancel = self.active_passes.get(&key).is_some_and(|pass| {
            pass.level < target_level
                && !(target_pass_from.is_some() && pass.pass_from == target_pass_from)
        });
        if !should_cancel {
            return None;
        }
        let pass = self.active_passes.remove(&key)?;
        for tile in &pass.tiles {
            self.remove_pass_in_flight(tile, pass.pass_id);
        }
        Some(pass.cancel)
    }

    /// Mark an active pass `pass_id` for `key` as completed.
    ///
    /// Returns `true` if `(key, pass_id)` matched the active pass. Clears its
    /// remaining in-flight keys; when `succeeded` is `false`, marks undelivered
    /// keys as transiently failed until a healthy pass forgives first-time
    /// failures (second failure becomes permanent).
    pub fn finish_pass(&mut self, key: PassKey, pass_id: u64, succeeded: bool) -> bool {
        let Some(pass) = self.take_active_pass(key, pass_id) else {
            return false;
        };
        for tile in pass.tiles {
            if self.remove_pass_in_flight(&tile, pass.pass_id) && !succeeded {
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
        true
    }

    /// Record that pointwise `(key, job_id)` was cancelled or evicted before
    /// producing an error, clearing its in-flight state without penalizing its
    /// failure retry budget.
    ///
    /// Returns `true` iff `(key, job_id)` was currently active in-flight.
    pub fn cancel_in_flight(&mut self, key: &TileKey, job_id: u64) -> bool {
        let matches = matches!(
            self.in_flight.get(key),
            Some(FlightEntry::Pointwise { job_id: id, .. }) if *id == job_id
        );
        if matches {
            self.in_flight.remove(key);
        }
        matches
    }

    /// Abort an active `ServeMode::Pass` (due to cancellation, zoom target
    /// change, or queue eviction) without penalizing undelivered tiles toward
    /// permanent suppression.
    ///
    /// Returns `Some(cancel_token)` iff `(key, pass_id)` matched the active
    /// pass so the caller can invoke `cancel()` outside any scheduler lock.
    pub fn abort_pass(&mut self, key: PassKey, pass_id: u64) -> Option<CancelToken> {
        let pass = self.take_active_pass(key, pass_id)?;
        for tile in &pass.tiles {
            self.remove_pass_in_flight(tile, pass.pass_id);
        }
        Some(pass.cancel)
    }

    /// Record successful arrival of pointwise `(key, job_id)`, clearing its
    /// in-flight and failure flags. Returns `true` if `(key, job_id)` matched.
    pub fn on_pointwise_arrived(&mut self, key: &TileKey, job_id: u64) -> bool {
        let matches = matches!(
            self.in_flight.get(key),
            Some(FlightEntry::Pointwise { job_id: id, .. }) if *id == job_id
        );
        if matches {
            self.in_flight.remove(key);
            self.failures.remove(key);
        }
        matches
    }

    /// Record successful arrival of `key` emitted by active pass
    /// `(pass_key, pass_id)`, clearing its in-flight and failure flags only
    /// while that pass remains active.
    pub fn on_pass_tile_arrived(&mut self, pass_key: PassKey, pass_id: u64, key: &TileKey) -> bool {
        let active = self
            .active_passes
            .get(&pass_key)
            .is_some_and(|p| p.pass_id == pass_id);
        if !active {
            return false;
        }
        self.remove_pass_in_flight(key, pass_id);
        self.failures.remove(key);
        true
    }

    /// Record failure for pointwise `(key, job_id)`.
    ///
    /// Returns `None` if `(key, job_id)` is stale (no longer active in-flight),
    /// ensuring a cancelled worker can never poison a newly admitted job for
    /// the same key. When `terminal` is `true`, suppresses `key` permanently;
    /// otherwise allows one retry and suppresses permanently on the second
    /// consecutive failure.
    pub fn on_pointwise_failed(
        &mut self,
        key: &TileKey,
        job_id: u64,
        terminal: bool,
    ) -> Option<TileFailureOutcome> {
        if !self.cancel_in_flight(key, job_id) {
            return None;
        }
        if terminal {
            self.failures.insert(*key, FailureState::Permanent);
            return Some(TileFailureOutcome::TerminalFailed);
        }
        let outcome = match self.failures.get(key) {
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
        };
        Some(outcome)
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

    /// Drop all active passes, pointwise in-flight entries, and failure state
    /// for `source`, returning all [`CancelToken`]s so the caller can cancel
    /// them outside any scheduler lock.
    pub fn reset_source(&mut self, source: SourceId) -> Vec<CancelToken> {
        let mut tokens = Vec::new();
        self.active_passes.retain(|k, pass| {
            if k.source == source {
                tokens.push(pass.cancel.clone());
                false
            } else {
                true
            }
        });
        self.in_flight.retain(|k, entry| {
            if k.source == source {
                if let FlightEntry::Pointwise { cancel, .. } = entry {
                    tokens.push(cancel.clone());
                }
                false
            } else {
                true
            }
        });
        self.failures.retain(|k, _| k.source != source);
        tokens
    }
}

#[cfg(test)]
#[path = "scheduler_tests.rs"]
mod tests;
