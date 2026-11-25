//! Priority-evicting, byte-bounded, coarsest-first job queue backing
//! [`super::DecodePool`].

use veldmap_core::sync::CancelToken;
use veldmap_core::tile::{BandKey, PixelFormat, SourceId, StreamKey, TileKey};

/// Deduplication and scheduling identity for a [`DecodeJob`], cleanly
/// distinguishing interactive cursor probes, background band-statistics scans,
/// single-tile fast jobs, and `StreamKey`-scoped bulk cascade jobs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DecodeJobKey {
    Probe {
        source: SourceId,
        band: BandKey,
        format: PixelFormat,
        x: u32,
        y: u32,
    },
    BandStats {
        source: SourceId,
        band: BandKey,
        layer_tag: u64,
    },
    Tile(TileKey),
    Pass(StreamKey),
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct DecodeJobPriority(DecodeJobKey);

impl DecodeJobKey {
    const fn priority_tier(self) -> u8 {
        match self {
            Self::Probe { .. } => 0,
            Self::BandStats { .. } => 1,
            Self::Tile(_) => 2,
            Self::Pass(_) => 3,
        }
    }
}

impl Ord for DecodeJobPriority {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let tier_cmp = self.0.priority_tier().cmp(&other.0.priority_tier());
        if tier_cmp != std::cmp::Ordering::Equal {
            return tier_cmp;
        }
        match (self.0, other.0) {
            (
                DecodeJobKey::Probe {
                    source: s1,
                    band: b1,
                    format: f1,
                    x: x1,
                    y: y1,
                },
                DecodeJobKey::Probe {
                    source: s2,
                    band: b2,
                    format: f2,
                    x: x2,
                    y: y2,
                },
            ) => (s1, b1, f1, x1, y1).cmp(&(s2, b2, f2, x2, y2)),
            (
                DecodeJobKey::BandStats {
                    source: s1,
                    band: b1,
                    layer_tag: l1,
                },
                DecodeJobKey::BandStats {
                    source: s2,
                    band: b2,
                    layer_tag: l2,
                },
            ) => (s1, b1, l1).cmp(&(s2, b2, l2)),
            (DecodeJobKey::Tile(a), DecodeJobKey::Tile(b)) => {
                a.coarsest_first_key().cmp(&b.coarsest_first_key())
            }
            (DecodeJobKey::Pass(a), DecodeJobKey::Pass(b)) => a.cmp(&b),
            _ => std::cmp::Ordering::Equal,
        }
    }
}

impl PartialOrd for DecodeJobPriority {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl DecodeJobKey {
    /// Coarseness rank (smaller = higher priority). Pass, Probe, and BandStats
    /// jobs run at equal rank `0` and never evict one another or root tiles.
    pub(super) const fn coarseness_rank(self) -> u8 {
        match self {
            Self::Tile(key) => key.scheme.coarseness_rank(key.level),
            Self::Pass(_) | Self::Probe { .. } | Self::BandStats { .. } => 0,
        }
    }

    /// Total deterministic priority key (smaller = dispatched first).
    pub(super) const fn priority_key(self) -> DecodeJobPriority {
        DecodeJobPriority(self)
    }
}

/// Single prioritized, byte-accounted CPU decode job submitted to
/// [`super::DecodePoolHandle`].
pub struct DecodeJob {
    pub key: DecodeJobKey,
    /// Accounted payload/working bytes charged against `max_pending_bytes`.
    pub queued_bytes: usize,
    pub cancel: CancelToken,
    /// Decode closure executed on a worker thread inside `catch_unwind`.
    pub work: Box<dyn FnOnce() + Send + 'static>,
}

/// Reason a [`DecodeJob`] was not admitted by [`super::DecodePoolHandle::try_submit`],
/// returning ownership of the rejected job so callers can clean up flight state
/// before dropping it.
pub enum SubmitError {
    Closed(DecodeJob),
    Cancelled(DecodeJob),
    Duplicate(DecodeJob),
    Full(DecodeJob),
}

impl std::fmt::Debug for SubmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Closed(j) => f.debug_tuple("Closed").field(&j.key).finish(),
            Self::Cancelled(j) => f.debug_tuple("Cancelled").field(&j.key).finish(),
            Self::Duplicate(j) => f.debug_tuple("Duplicate").field(&j.key).finish(),
            Self::Full(j) => f.debug_tuple("Full").field(&j.key).finish(),
        }
    }
}

pub(super) struct DecodeQueue {
    pub(super) jobs: Vec<DecodeJob>,
    pub(super) queued_bytes: usize,
    max_pending_jobs: usize,
    max_pending_bytes: usize,
    pub(super) closed: bool,
}

impl DecodeQueue {
    pub(super) fn new(max_pending_jobs: usize, max_pending_bytes: usize) -> Self {
        Self {
            jobs: Vec::new(),
            queued_bytes: 0,
            max_pending_jobs: max_pending_jobs.max(1),
            max_pending_bytes,
            closed: false,
        }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.jobs.is_empty()
    }

    fn fits_incoming(
        &self,
        remaining_jobs: usize,
        remaining_bytes: usize,
        incoming_bytes: usize,
    ) -> bool {
        remaining_jobs == 0
            || (remaining_jobs < self.max_pending_jobs
                && remaining_bytes.saturating_add(incoming_bytes) <= self.max_pending_bytes)
    }

    fn push_job(&mut self, job: DecodeJob) {
        self.queued_bytes = self.queued_bytes.saturating_add(job.queued_bytes);
        self.jobs.push(job);
    }

    fn swap_remove_job(&mut self, idx: usize) -> DecodeJob {
        let job = self.jobs.swap_remove(idx);
        self.queued_bytes = self.queued_bytes.saturating_sub(job.queued_bytes);
        job
    }

    pub(super) fn drain_all(&mut self) -> Vec<DecodeJob> {
        self.queued_bytes = 0;
        std::mem::take(&mut self.jobs)
    }

    fn prune_cancelled(&mut self, to_drop: &mut Vec<DecodeJob>) {
        let mut i = 0;
        while i < self.jobs.len() {
            if self.jobs[i].cancel.is_cancelled() {
                let removed = self.swap_remove_job(i);
                to_drop.push(removed);
            } else {
                i += 1;
            }
        }
    }

    /// Plan eviction victims transactionally before mutating the queue so
    /// rejecting an incoming job never destroys valid finer queued jobs
    /// (always permitting a single job into an empty queue).
    fn plan_evictions(&self, key: DecodeJobKey, incoming_bytes: usize) -> Option<Vec<usize>> {
        if self.fits_incoming(self.jobs.len(), self.queued_bytes, incoming_bytes) {
            return Some(Vec::new());
        }

        let incoming_coarse = key.coarseness_rank();
        let mut candidates: Vec<(usize, DecodeJobPriority, usize)> = self
            .jobs
            .iter()
            .enumerate()
            .filter_map(|(idx, j)| {
                let victim_coarse = j.key.coarseness_rank();
                (incoming_coarse < victim_coarse).then_some((
                    idx,
                    j.key.priority_key(),
                    j.queued_bytes,
                ))
            })
            .collect();
        // Finest-first victim order (highest priority tuple first).
        candidates.sort_by_key(|c| std::cmp::Reverse(c.1));

        let mut remaining_jobs = self.jobs.len();
        let mut remaining_bytes = self.queued_bytes;
        let mut victims = Vec::new();
        for (idx, _, bytes) in candidates {
            if self.fits_incoming(remaining_jobs, remaining_bytes, incoming_bytes) {
                break;
            }
            remaining_jobs -= 1;
            remaining_bytes = remaining_bytes.saturating_sub(bytes);
            victims.push(idx);
        }

        self.fits_incoming(remaining_jobs, remaining_bytes, incoming_bytes)
            .then_some(victims)
    }

    pub(super) fn admit_job(
        &mut self,
        job: DecodeJob,
        to_drop: &mut Vec<DecodeJob>,
    ) -> Result<(), SubmitError> {
        if self.closed {
            return Err(SubmitError::Closed(job));
        }
        self.prune_cancelled(to_drop);
        if self.jobs.iter().any(|j| j.key == job.key) {
            return Err(SubmitError::Duplicate(job));
        }
        let Some(mut victims) = self.plan_evictions(job.key, job.queued_bytes) else {
            return Err(SubmitError::Full(job));
        };
        // Remove in descending index order so `swap_remove` never invalidates
        // earlier planned indices.
        victims.sort_unstable_by_key(|&idx| std::cmp::Reverse(idx));
        for idx in victims {
            to_drop.push(self.swap_remove_job(idx));
        }
        self.push_job(job);
        Ok(())
    }

    /// Pop the highest-priority (coarsest) uncancelled job, or drain all
    /// cancelled entries if none remain runnable.
    pub(super) fn pop_coarsest_or_drain_cancelled(
        &mut self,
    ) -> (Option<DecodeJob>, Vec<DecodeJob>) {
        let best_idx = self
            .jobs
            .iter()
            .enumerate()
            .filter(|(_, j)| !j.cancel.is_cancelled())
            .min_by_key(|(_, j)| j.key.priority_key())
            .map(|(i, _)| i);
        match best_idx {
            Some(i) => (Some(self.swap_remove_job(i)), Vec::new()),
            None => (None, self.drain_all()),
        }
    }
}
