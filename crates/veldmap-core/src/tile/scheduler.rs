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

    /// Progressive ladder of levels from `max_level()` down to `target_level`
    /// (clamped to `[finest_fitting_level(), max_level()]`), skipping coarser
    /// levels that would trigger redundant full-file passes.
    pub fn ladder(&self, target_level: u8) -> Vec<u8> {
        if self.levels.is_empty() {
            return Vec::new();
        }
        let top = self.max_level();
        let min_fit = self.finest_fitting_level().min(top);
        let target = target_level.clamp(min_fit, top);
        let goal_pass = self.pass_from(target);

        let mut rungs: Vec<u8> = ((target + 1)..=top)
            .rev()
            .filter(|&lvl| {
                let from = self.pass_from(lvl);
                from.is_none() || from == goal_pass || (lvl == top && from == Some(top))
            })
            .collect();
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
    /// until `forgive_transient_failures` (for single-pass forgiveness) or
    /// `reset_source`.
    TerminalFailed,
}

struct ActivePass {
    pass_id: u64,
    level: u8,
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
    retried_once: HashSet<TileKey>,
    transient_failed: HashSet<TileKey>,
    permanent_failed: HashSet<TileKey>,
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
            retried_once: HashSet::new(),
            transient_failed: HashSet::new(),
            permanent_failed: HashSet::new(),
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
        self.permanent_failed.contains(key) || self.transient_failed.contains(key)
    }

    /// Filter `candidates` down to tiles that are not cached, not already
    /// in-flight, and not failed/suppressed, marking admitted keys as
    /// in-flight.
    ///
    /// For `ServeMode::Pointwise` levels, multiple concurrent batches are
    /// admitted regardless of whether other pointwise tiles are already
    /// in-flight. For `ServeMode::Pass` levels, returns empty when a pass is
    /// already active for `meta.source`.
    pub fn admit_pointwise_or_check_pass(
        &mut self,
        meta: &PyramidMeta,
        level: u8,
        candidates: &[TileKey],
        is_cached: impl Fn(&TileKey) -> bool,
    ) -> Vec<TileKey> {
        if !meta.is_pointwise(level) && self.has_active_pass(meta.source) {
            return Vec::new();
        }
        let mut admitted = Vec::new();
        for &key in candidates {
            if is_cached(&key) || self.in_flight.contains(&key) || self.is_terminal_failed(&key) {
                continue;
            }
            if meta.is_pointwise(key.level) {
                self.in_flight.insert(key);
            }
            admitted.push(key);
        }
        admitted
    }

    /// Begin the single active `ServeMode::Pass` for `source` at `level`.
    ///
    /// Returns `Err(PassBusy)` if another sequential pass is already active
    /// for `source`. `ServeMode::Pointwise` requests must NOT call this method
    /// so pointwise tiles execute concurrently.
    pub fn begin_pass(
        &mut self,
        source: SourceId,
        level: u8,
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
                cancel,
                tiles,
            },
        );
        Ok(pass_id)
    }

    /// Update the desired target level for `source`.
    ///
    /// Preserves coarser in-flight passes (`pass.level >= target_level`) so
    /// zooming in never discards half-decoded ancestor passes. Cancels an active
    /// pass only when it is strictly finer than the new target
    /// (`pass.level < target_level`).
    pub fn update_target(&mut self, source: SourceId, target_level: u8) -> bool {
        let should_cancel = self
            .active_passes
            .get(&source)
            .is_some_and(|pass| pass.level < target_level);
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
                    if self.retried_once.insert(key) {
                        self.transient_failed.insert(key);
                    } else {
                        self.transient_failed.remove(&key);
                        self.permanent_failed.insert(key);
                    }
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
        self.retried_once.remove(key);
        self.transient_failed.remove(key);
        self.permanent_failed.remove(key);
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
            self.retried_once.remove(key);
            self.transient_failed.remove(key);
            self.permanent_failed.insert(*key);
            return TileFailureOutcome::TerminalFailed;
        }
        if self.retried_once.insert(*key) {
            TileFailureOutcome::RetryEligible
        } else {
            self.transient_failed.remove(key);
            self.permanent_failed.insert(*key);
            TileFailureOutcome::TerminalFailed
        }
    }

    /// Forgive tiles of `source` that only failed once in a prior broken pass
    /// so a subsequent healthy pass can retry them without ever forgiving
    /// permanently rejected or twice-failed tiles.
    pub fn forgive_single_pass_failures(&mut self, source: SourceId) {
        self.transient_failed.retain(|key| key.source != source);
    }

    /// Cancel any active pass for `source` and drop all in-flight/failure state
    /// associated with `source`.
    pub fn reset_source(&mut self, source: SourceId) {
        if let Some(pass) = self.active_passes.remove(&source) {
            pass.cancel.cancel();
        }
        self.in_flight.retain(|k| k.source != source);
        self.retried_once.retain(|k| k.source != source);
        self.transient_failed.retain(|k| k.source != source);
        self.permanent_failed.retain(|k| k.source != source);
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
        // Duplicate in-flight tile is deduplicated, while another pointwise tile admits.
        assert_eq!(
            sched.admit_pointwise_or_check_pass(&meta_a, 0, &[k0, k2], |_| false),
            vec![k2]
        );

        // Only 1 active Serve::Pass per source.
        let cancel1 = CancelToken::new();
        let pass_tile = meta_a.tile_key(3, 0, 0);
        let pid1 = sched
            .begin_pass(src_a, 3, cancel1.clone(), vec![pass_tile])
            .expect("first pass succeeds");
        assert!(
            sched
                .begin_pass(src_a, 2, CancelToken::new(), vec![])
                .is_err()
        );
        // Another source can run its own pass concurrently.
        let pid_b = sched
            .begin_pass(src_b, 2, CancelToken::new(), vec![])
            .expect("independent source pass succeeds");
        assert_ne!(pid1, pid_b);

        // Zooming IN (target=1 < active_pass_level=3) does NOT cancel coarser pass.
        assert!(!sched.update_target(src_a, 1));
        assert!(!cancel1.is_cancelled());
        assert!(sched.has_active_pass(src_a));

        // Zooming OUT (target=3, then suppose active pass was finer lvl 2 and target=3) cancels finer pass.
        sched.finish_pass(src_a, pid1, true);
        let cancel2 = CancelToken::new();
        let _pid2 = sched
            .begin_pass(src_a, 2, cancel2.clone(), vec![meta_a.tile_key(2, 0, 0)])
            .expect("new pass after finish");
        assert!(sched.update_target(src_a, 3));
        assert!(cancel2.is_cancelled());
        assert!(!sched.has_active_pass(src_a));
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
            .begin_pass(meta.source, 1, CancelToken::new(), vec![k_pass])
            .expect("begin failing pass");
        sched.finish_pass(meta.source, pid_fail, false);
        assert!(sched.is_terminal_failed(&k_pass));
        let pid_heal = sched
            .begin_pass(
                meta.source,
                2,
                CancelToken::new(),
                vec![meta.tile_key(2, 0, 0)],
            )
            .expect("begin healing pass");
        sched.finish_pass(meta.source, pid_heal, true);
        assert!(!sched.is_terminal_failed(&k_pass));
    }
}
