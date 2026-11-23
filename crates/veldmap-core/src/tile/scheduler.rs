//! Native raster pyramid tile scheduler (`TileScheduler`, `PyramidMeta`,
//! `LevelPlan`, `ServeMode`).
//!
//! Coordinates progressive coarsest-to-target tile fetching across both
//! concurrent pointwise reads and single-active sequential passes:
//! - Only 1 active sequential pass per `SourceId` is allowed for
//!   `ServeMode::Pass`.
//! - `ServeMode::Pointwise` levels run concurrently across multiple tiles and
//!   batches while deduplicating identical `TileKey`s.
//! - Coarser in-flight passes (`pass_level > target_level`) are never cancelled
//!   on zoom-in because their output serves as valid ancestor carriers.
//! - Terminal failures (`Reject` or 2nd consecutive transient failure) are
//!   suppressed permanently until `reset_source`.

use std::collections::{HashMap, HashSet};

use crate::io::CancelToken;
use crate::tile::key::{BandKey, PixelFormat, SourceId, TileKey, TileScheme};

/// How one pyramid level is produced from its underlying source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ServeMode {
    /// Only chunks intersecting requested tiles are decoded.
    Pointwise,
    /// Sequential pass starting from level `from` producing `from..=max_level`
    /// together in one cascade pass.
    Pass { from: u8 },
}

/// Cost and execution plan for one pyramid level (`0` = finest native).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LevelPlan {
    pub serve: ServeMode,
    /// Estimated source pixels read for one tile (pointwise) or full pass.
    pub source_pixels: u64,
    /// Estimated working memory peak in bytes.
    pub estimated_peak_bytes: u64,
    /// Whether this level fits inside the configured memory budget.
    pub fits: bool,
}

/// Scheduler-facing description of one raster pyramid.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PyramidMeta {
    pub source: SourceId,
    pub band: BandKey,
    pub format: PixelFormat,
    pub gutter: u8,
    pub decode_rev: u32,
    pub width: u32,
    pub height: u32,
    pub tile_size: u32,
    /// Level plans ordered from `0` (finest native) to `max_level` (coarsest
    /// single-tile apex).
    pub levels: Vec<LevelPlan>,
}

impl PyramidMeta {
    /// Coarsest apex level index (`levels.len().saturating_sub(1)`).
    pub fn max_level(&self) -> u8 {
        self.levels.len().saturating_sub(1).min(u8::MAX as usize) as u8
    }

    /// Corresponding [`TileScheme::RasterPyramid`].
    pub fn scheme(&self) -> TileScheme {
        TileScheme::RasterPyramid {
            max_level: self.max_level(),
        }
    }

    /// Finest level index whose `LevelPlan::fits` is `true` (falling back to
    /// `max_level()` when none fits).
    pub fn finest_fitting_level(&self) -> u8 {
        self.levels
            .iter()
            .position(|row| row.fits)
            .map_or_else(|| self.max_level(), |idx| idx as u8)
    }

    /// Return the [`LevelPlan`] for `level`, if in range.
    pub fn level(&self, level: u8) -> Option<LevelPlan> {
        self.levels.get(level as usize).copied()
    }

    /// Return whether `level` is served pointwise.
    pub fn is_pointwise(&self, level: u8) -> bool {
        self.level(level)
            .is_some_and(|row| row.serve == ServeMode::Pointwise)
    }

    /// Return the pass start level if `level` is served by `ServeMode::Pass`.
    pub fn pass_from(&self, level: u8) -> Option<u8> {
        match self.level(level)?.serve {
            ServeMode::Pass { from } => Some(from),
            ServeMode::Pointwise => None,
        }
    }

    /// Progressive ladder of fitting levels from `max_level()` down to
    /// `target_level` (clamped to `[finest_fitting_level(), max_level()]`),
    /// keeping at most one coarsest representative per distinct `ServeMode::Pass`
    /// cascade so redundant full-file passes are skipped.
    pub fn ladder(&self, target_level: u8) -> Vec<u8> {
        if self.levels.is_empty() {
            return Vec::new();
        }
        let top = self.max_level();
        let min_fit = self.finest_fitting_level().min(top);
        let target = target_level.clamp(min_fit, top);
        let goal_pass = self.pass_from(target);

        let mut rungs = Vec::new();
        let mut seen_passes = HashSet::new();
        if target < top {
            for lvl_u16 in (u16::from(target) + 1..=u16::from(top)).rev() {
                let lvl = lvl_u16 as u8;
                let Some(plan) = self.level(lvl) else {
                    continue;
                };
                if !plan.fits {
                    continue;
                }
                match plan.serve {
                    ServeMode::Pointwise => rungs.push(lvl),
                    ServeMode::Pass { from } => {
                        if Some(from) != goal_pass && seen_passes.insert(from) {
                            rungs.push(lvl);
                        }
                    }
                }
            }
        }
        rungs.push(target);
        rungs
    }

    /// Construct the canonical [`TileKey`] for `(level, x, y)`.
    pub fn tile_key(&self, level: u8, x: u32, y: u32) -> TileKey {
        TileKey {
            source: self.source,
            band: self.band,
            scheme: self.scheme(),
            format: self.format,
            gutter: self.gutter,
            decode_rev: self.decode_rev,
            level,
            x,
            y,
        }
    }
}

/// Error returned when starting a sequential pass while one is already active
/// for the same `SourceId`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PassBusy {
    pub source: SourceId,
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
/// passes per source, non-cancellation of coarser passes, and terminal-failure
/// suppression.
pub struct TileScheduler {
    next_pass_id: u64,
    active_passes: HashMap<SourceId, ActivePass>,
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

    /// Return whether `source` currently has an active `ServeMode::Pass`.
    pub fn has_active_pass(&self, source: SourceId) -> bool {
        self.active_passes.contains_key(&source)
    }

    /// Return the level of the active `ServeMode::Pass` for `source`, if any.
    pub fn active_pass_level(&self, source: SourceId) -> Option<u8> {
        self.active_passes.get(&source).map(|p| p.level)
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

    /// Filter `candidates` down to unique tiles belonging to `(meta.source, level)`
    /// that are not cached, not already in-flight, and not suppressed, marking
    /// admitted pointwise keys as in-flight.
    ///
    /// For `ServeMode::Pointwise` levels, multiple concurrent batches are
    /// admitted regardless of whether other pointwise tiles are already
    /// in-flight. For `ServeMode::Pass` levels, returns empty when a pass is
    /// already active for `meta.source` and permits first-time pass-failed tiles
    /// to trigger a recovery pass so pass-only sources cannot permanently starve.
    pub fn admit_pointwise_or_check_pass(
        &mut self,
        meta: &PyramidMeta,
        level: u8,
        candidates: &[TileKey],
        is_cached: impl Fn(&TileKey) -> bool,
    ) -> Vec<TileKey> {
        let Some(plan) = meta.level(level) else {
            return Vec::new();
        };
        let is_pointwise = plan.serve == ServeMode::Pointwise;
        if !is_pointwise && self.has_active_pass(meta.source) {
            return Vec::new();
        }
        let mut seen = HashSet::new();
        let mut admitted = Vec::new();
        for &key in candidates {
            if key.source != meta.source || key.level != level || !seen.insert(key) {
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
            if is_pointwise {
                self.in_flight.insert(key);
            }
            admitted.push(key);
        }
        admitted
    }

    /// Begin the single active `ServeMode::Pass` for `source` at `level`
    /// (with optional shared cascade root `pass_from`).
    ///
    /// Returns `Err(PassBusy)` if another sequential pass is already active
    /// for `source`. `ServeMode::Pointwise` requests must NOT call this method
    /// so pointwise tiles execute concurrently.
    pub fn begin_pass(
        &mut self,
        source: SourceId,
        level: u8,
        pass_from: Option<u8>,
        cancel: CancelToken,
        tiles: Vec<TileKey>,
    ) -> Result<u64, PassBusy> {
        if let Some(existing) = self.active_passes.get(&source) {
            return Err(PassBusy {
                source,
                active_pass_id: existing.pass_id,
                active_level: existing.level,
            });
        }
        let pass_id = self.next_pass_id;
        self.next_pass_id = self.next_pass_id.wrapping_add(1).max(1);
        for &key in &tiles {
            self.in_flight.insert(key);
        }
        self.active_passes.insert(
            source,
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
    /// `ServeMode::Pass`) for `source`.
    ///
    /// Preserves coarser in-flight passes (`pass.level >= target_level`) and
    /// active passes sharing the same cascade `pass_from` (e.g. `Pass { from: 0 }`)
    /// so zooming never discards valid ancestor or shared-cascade work. Cancels
    /// an active pass only when it is strictly finer than `target_level` and
    /// does not cover `target_pass_from`.
    pub fn update_target(
        &mut self,
        source: SourceId,
        target_level: u8,
        target_pass_from: Option<u8>,
    ) -> bool {
        let should_cancel = self.active_passes.get(&source).is_some_and(|pass| {
            pass.level < target_level
                && !(target_pass_from.is_some() && pass.pass_from == target_pass_from)
        });
        if !should_cancel {
            return false;
        }
        if let Some(pass) = self.active_passes.remove(&source) {
            for key in pass.tiles {
                self.in_flight.remove(&key);
            }
            pass.cancel.cancel();
            return true;
        }
        false
    }

    /// Mark an active pass `pass_id` for `source` as completed.
    ///
    /// Clears its remaining in-flight keys; when `succeeded` is `false`, marks
    /// undelivered keys as transiently failed until a healthy pass forgives
    /// first-time failures (second failure becomes permanent).
    pub fn finish_pass(&mut self, source: SourceId, pass_id: u64, succeeded: bool) {
        let matches = self
            .active_passes
            .get(&source)
            .is_some_and(|p| p.pass_id == pass_id);
        if !matches {
            return;
        }
        if let Some(pass) = self.active_passes.remove(&source) {
            for key in pass.tiles {
                if self.in_flight.remove(&key) && !succeeded {
                    let next = match self.failures.get(&key) {
                        None => FailureState::TransientPassSuppressed,
                        Some(
                            FailureState::RetriedOnce
                            | FailureState::TransientPassSuppressed
                            | FailureState::Permanent,
                        ) => FailureState::Permanent,
                    };
                    self.failures.insert(key, next);
                }
            }
            if succeeded {
                self.forgive_single_pass_failures(source);
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

    /// Forgive tiles of `source` that only failed once in a prior broken pass
    /// (demoting them to `RetriedOnce` so a subsequent healthy pass allows one
    /// retry while a second failure becomes permanent).
    pub fn forgive_single_pass_failures(&mut self, source: SourceId) {
        for (key, state) in &mut self.failures {
            if key.source == source && *state == FailureState::TransientPassSuppressed {
                *state = FailureState::RetriedOnce;
            }
        }
    }

    /// Cancel any active pass for `source` and drop all in-flight/failure state
    /// associated with `source`.
    pub fn reset_source(&mut self, source: SourceId) {
        if let Some(pass) = self.active_passes.remove(&source) {
            pass.cancel.cancel();
        }
        self.in_flight.retain(|k| k.source != source);
        self.failures.retain(|k, _| k.source != source);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_meta(source: SourceId, pointwise_prefix: usize, total_levels: usize) -> PyramidMeta {
        let levels = (0..total_levels)
            .map(|lvl| {
                let serve = if lvl < pointwise_prefix {
                    ServeMode::Pointwise
                } else {
                    ServeMode::Pass { from: lvl as u8 }
                };
                LevelPlan {
                    serve,
                    source_pixels: 512 * 512,
                    estimated_peak_bytes: 4 * 1024 * 1024,
                    fits: true,
                }
            })
            .collect();
        PyramidMeta {
            source,
            band: BandKey::Display,
            format: PixelFormat::Rgba8UnormSrgb,
            gutter: 0,
            decode_rev: 1,
            width: 4096,
            height: 4096,
            tile_size: 512,
            levels,
        }
    }

    #[test]
    fn pointwise_runs_concurrently_while_pass_is_single_per_source() {
        let mut sched = TileScheduler::new();
        let src_a = SourceId::new(10);
        let src_b = SourceId::new(11);
        let meta_a = sample_meta(src_a, 2, 4);

        // Concurrent pointwise admissions on level 0 and level 1 succeed together.
        let k0 = meta_a.tile_key(0, 0, 0);
        let k1 = meta_a.tile_key(0, 1, 0);
        let k2 = meta_a.tile_key(1, 0, 0);
        assert_eq!(
            sched.admit_pointwise_or_check_pass(&meta_a, 0, &[k0, k1], |_| false),
            vec![k0, k1]
        );
        // Duplicate in-flight or mismatched-level tiles are ignored, while matching-level tiles admit.
        assert!(
            sched
                .admit_pointwise_or_check_pass(&meta_a, 0, &[k0, k2], |_| false)
                .is_empty()
        );
        let k0_dup = meta_a.tile_key(0, 2, 0);
        assert_eq!(
            sched.admit_pointwise_or_check_pass(&meta_a, 0, &[k0, k0_dup, k0_dup], |_| false),
            vec![k0_dup]
        );
        assert_eq!(
            sched.admit_pointwise_or_check_pass(&meta_a, 1, &[k2], |_| false),
            vec![k2]
        );

        // Only 1 active Serve::Pass per source.
        let cancel1 = CancelToken::new();
        let pass_tile = meta_a.tile_key(3, 0, 0);
        let pid1 = sched
            .begin_pass(src_a, 3, Some(3), cancel1.clone(), vec![pass_tile])
            .expect("first pass succeeds");
        assert!(
            sched
                .begin_pass(src_a, 2, Some(2), CancelToken::new(), vec![])
                .is_err()
        );
        // Another source can run its own pass concurrently.
        let pid_b = sched
            .begin_pass(src_b, 2, Some(2), CancelToken::new(), vec![])
            .expect("independent source pass succeeds");
        assert_ne!(pid1, pid_b);

        // Zooming IN (target=1 < active_pass_level=3) does NOT cancel coarser pass.
        assert!(!sched.update_target(src_a, 1, None));
        assert!(!cancel1.is_cancelled());
        assert!(sched.has_active_pass(src_a));

        // Zooming OUT (target=3, then suppose active pass was finer lvl 2 and target=3) cancels finer pass.
        sched.finish_pass(src_a, pid1, true);
        let cancel2 = CancelToken::new();
        let _pid2 = sched
            .begin_pass(
                src_a,
                2,
                Some(2),
                cancel2.clone(),
                vec![meta_a.tile_key(2, 0, 0)],
            )
            .expect("new pass after finish");
        assert!(sched.update_target(src_a, 3, Some(3)));
        assert!(cancel2.is_cancelled());
        assert!(!sched.has_active_pass(src_a));

        // Shared `Pass { from: 0 }` cascade is NOT cancelled when zooming out to another level in the same pass.
        let cancel3 = CancelToken::new();
        let pid3 = sched
            .begin_pass(
                src_a,
                2,
                Some(0),
                cancel3.clone(),
                vec![meta_a.tile_key(2, 0, 0)],
            )
            .expect("shared cascade pass");
        assert!(!sched.update_target(src_a, 3, Some(0)));
        assert!(!cancel3.is_cancelled());
        sched.finish_pass(src_a, pid3, true);
    }

    #[test]
    fn terminal_and_repeated_failures_are_suppressed() {
        let mut sched = TileScheduler::new();
        let meta = sample_meta(SourceId::new(22), 2, 3);
        let k_term = meta.tile_key(0, 0, 0);
        let k_retry = meta.tile_key(0, 1, 0);

        assert_eq!(
            sched.admit_pointwise_or_check_pass(&meta, 0, &[k_term, k_retry], |_| false),
            vec![k_term, k_retry]
        );

        // Terminal failure suppresses immediately.
        assert_eq!(
            sched.on_tile_failed(&k_term, true),
            TileFailureOutcome::TerminalFailed
        );
        assert!(sched.is_terminal_failed(&k_term));

        // First transient failure allows retry; second consecutive suppresses permanently.
        assert_eq!(
            sched.on_tile_failed(&k_retry, false),
            TileFailureOutcome::RetryEligible
        );
        assert!(!sched.is_terminal_failed(&k_retry));
        assert_eq!(
            sched.admit_pointwise_or_check_pass(&meta, 0, &[k_term, k_retry], |_| false),
            vec![k_retry]
        );
        assert_eq!(
            sched.on_tile_failed(&k_retry, false),
            TileFailureOutcome::TerminalFailed
        );
        assert!(sched.is_terminal_failed(&k_retry));
        assert!(
            sched
                .admit_pointwise_or_check_pass(&meta, 0, &[k_term, k_retry], |_| false)
                .is_empty()
        );

        // A healthy pass never forgives permanently rejected or twice-failed tiles.
        let pid_ok = sched
            .begin_pass(
                meta.source,
                2,
                Some(2),
                CancelToken::new(),
                vec![meta.tile_key(2, 0, 0)],
            )
            .expect("begin healthy pass");
        sched.finish_pass(meta.source, pid_ok, true);
        assert!(sched.is_terminal_failed(&k_term));
        assert!(sched.is_terminal_failed(&k_retry));

        // First-time pass failure is suppressed until a healthy pass forgives it once.
        let k_pass = meta.tile_key(1, 0, 0);
        let pid_fail = sched
            .begin_pass(meta.source, 1, None, CancelToken::new(), vec![k_pass])
            .expect("begin failing pass");
        sched.finish_pass(meta.source, pid_fail, false);
        assert!(sched.is_terminal_failed(&k_pass));
        let pid_heal = sched
            .begin_pass(
                meta.source,
                2,
                Some(2),
                CancelToken::new(),
                vec![meta.tile_key(2, 0, 0)],
            )
            .expect("begin healing pass");
        sched.finish_pass(meta.source, pid_heal, true);
        assert!(!sched.is_terminal_failed(&k_pass));
        // Second failure after forgiveness escalates permanently.
        assert_eq!(
            sched.on_tile_failed(&k_pass, false),
            TileFailureOutcome::TerminalFailed
        );
        assert!(sched.is_terminal_failed(&k_pass));
    }

    #[test]
    fn ladder_and_pass_only_recovery_avoid_redundant_passes_and_starvation() {
        let mut meta = sample_meta(SourceId::new(33), 2, 5);
        for lvl in 2..5 {
            meta.levels[lvl].serve = ServeMode::Pass { from: 0 };
        }
        // Pointwise target keeps apex pass once + pointwise rungs.
        assert_eq!(meta.ladder(0), vec![4, 1, 0]);
        // Pass target skips redundant coarser rungs sharing `Pass { from: 0 }`.
        assert_eq!(meta.ladder(3), vec![3]);

        // Pass-only source admits a recovery pass after a single pass failure, then stops on 2nd failure.
        let mut sched = TileScheduler::new();
        let pass_only = PyramidMeta {
            levels: (0..2)
                .map(|_| LevelPlan {
                    serve: ServeMode::Pass { from: 0 },
                    source_pixels: 512 * 512,
                    estimated_peak_bytes: 1024,
                    fits: true,
                })
                .collect(),
            ..meta
        };
        let k = pass_only.tile_key(1, 0, 0);
        let pid1 = sched
            .begin_pass(pass_only.source, 1, Some(0), CancelToken::new(), vec![k])
            .expect("pass 1");
        sched.finish_pass(pass_only.source, pid1, false);
        assert_eq!(
            sched.admit_pointwise_or_check_pass(&pass_only, 1, &[k], |_| false),
            vec![k]
        );
        let pid2 = sched
            .begin_pass(pass_only.source, 1, Some(0), CancelToken::new(), vec![k])
            .expect("pass 2");
        sched.finish_pass(pass_only.source, pid2, false);
        assert!(
            sched
                .admit_pointwise_or_check_pass(&pass_only, 1, &[k], |_| false)
                .is_empty()
        );
    }
}
