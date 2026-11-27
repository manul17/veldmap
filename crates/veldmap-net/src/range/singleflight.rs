//! Panic- and cancel-safe multi-block singleflight with refcounted waiter
//! cancellation.
//!
//! Each in-flight HTTP Range GET runs inside a detached Tokio task tied to a
//! per-flight [`CancelToken`]. Waiters hold a [`WaiterGuard`] that decrements
//! the waiter count on drop and cancels the HTTP Range GET **only** when the
//! last waiter departs—so cancelling Follower A never aborts a fetch still
//! awaited by Follower B. A [`LeaderCleanupGuard`] ensures panics or aborted
//! leader tasks always wake followers and remove stale table entries.

use std::collections::HashMap;
use std::io;
use std::sync::{Arc, Mutex};

use tokio::sync::Notify;
use veldmap_core::io::IoError;
use veldmap_core::sync::{CancelToken, lock_unpoisoned};

use super::transport::RangeTransportError;

/// Cloneable error shared across all singleflight waiters of one range fetch.
#[derive(Clone, Debug)]
pub(super) enum SharedRangeError {
    Transport(RangeTransportError),
    Aborted,
}

impl From<RangeTransportError> for SharedRangeError {
    fn from(err: RangeTransportError) -> Self {
        Self::Transport(err)
    }
}

impl SharedRangeError {
    pub(super) fn into_io_error(self) -> IoError {
        match self {
            Self::Transport(RangeTransportError::Cancelled) => IoError::Cancelled,
            Self::Transport(RangeTransportError::PreconditionFailed(msg)) => {
                IoError::Io(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("remote object changed (HTTP 412 Precondition Failed): {msg}"),
                ))
            }
            Self::Transport(RangeTransportError::RangeNotSupported(msg)) => {
                IoError::Io(io::Error::new(
                    io::ErrorKind::Unsupported,
                    format!("HTTP Range not supported: {msg}"),
                ))
            }
            Self::Transport(RangeTransportError::HttpStatus { code, detail }) => IoError::Io(
                io::Error::other(format!("HTTP {code} reading remote range: {detail}")),
            ),
            Self::Transport(RangeTransportError::InvalidResponse(msg)) => {
                IoError::Io(io::Error::new(io::ErrorKind::InvalidData, msg))
            }
            Self::Transport(RangeTransportError::Network(msg)) => {
                IoError::Io(io::Error::other(msg))
            }
            Self::Aborted => IoError::Io(io::Error::other("remote range fetch task aborted")),
        }
    }
}

pub(super) type FetchedBlocks = Arc<HashMap<u64, Arc<[u8]>>>;

struct FlightState {
    waiters: usize,
    finished: bool,
    outcome: Option<Result<FetchedBlocks, SharedRangeError>>,
}

/// Shared state for a single in-flight multi-block HTTP Range fetch.
pub(super) struct BlockFlight {
    range_cancel: CancelToken,
    state: Mutex<FlightState>,
    done: Notify,
}

impl BlockFlight {
    fn new(parent_cancel: &CancelToken) -> Self {
        Self {
            range_cancel: parent_cancel.child_token(),
            state: Mutex::new(FlightState {
                waiters: 0,
                finished: false,
                outcome: None,
            }),
            done: Notify::new(),
        }
    }

    pub(super) fn cancel_token(&self) -> &CancelToken {
        &self.range_cancel
    }

    fn add_waiter(self: &Arc<Self>) -> WaiterGuard {
        {
            let mut st = lock_unpoisoned(&self.state);
            st.waiters = st.waiters.saturating_add(1);
        }
        WaiterGuard {
            flight: Arc::clone(self),
        }
    }

    /// Attempt to attach an additional waiter to an active, uncancelled flight.
    /// Returns `None` if the flight has already finished or its last waiter
    /// dropped and cancelled `range_cancel`, preventing callers from joining a
    /// dying flight and observing a spurious `Cancelled` error.
    pub(super) fn try_add_waiter(self: &Arc<Self>) -> Option<WaiterGuard> {
        {
            let mut st = lock_unpoisoned(&self.state);
            if st.finished || st.waiters == 0 || self.range_cancel.is_cancelled() {
                return None;
            }
            st.waiters = st.waiters.saturating_add(1);
        }
        Some(WaiterGuard {
            flight: Arc::clone(self),
        })
    }

    /// Await completion of this flight, returning `Err(IoError::Cancelled)`
    /// immediately if `caller_cancel` triggers (while leaving the underlying
    /// HTTP fetch alive if other waiters remain).
    pub(super) async fn wait(
        self: &Arc<Self>,
        _guard: WaiterGuard,
        caller_cancel: &CancelToken,
    ) -> Result<FetchedBlocks, IoError> {
        loop {
            if caller_cancel.is_cancelled() {
                return Err(IoError::Cancelled);
            }
            let notified = self.done.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            {
                let st = lock_unpoisoned(&self.state);
                if let Some(outcome) = &st.outcome {
                    return outcome.clone().map_err(SharedRangeError::into_io_error);
                }
            }
            tokio::select! {
                biased;
                _ = caller_cancel.cancelled() => return Err(IoError::Cancelled),
                _ = notified => {}
            }
        }
    }
}

/// RAII waiter reference; cancels the shared HTTP fetch only when the last
/// waiter drops before the fetch finishes.
pub(super) struct WaiterGuard {
    flight: Arc<BlockFlight>,
}

impl Drop for WaiterGuard {
    fn drop(&mut self) {
        let should_cancel = {
            let mut st = lock_unpoisoned(&self.flight.state);
            st.waiters = st.waiters.saturating_sub(1);
            !st.finished && st.waiters == 0
        };
        if should_cancel {
            self.flight.range_cancel.cancel();
        }
    }
}

/// RAII cleanup guard owned by the spawned leader fetch task so normal finish,
/// cancellation, or panics always remove entries and wake every waiter.
pub(super) struct LeaderCleanupGuard {
    table: Arc<SingleflightTable>,
    blocks: Vec<u64>,
    flight: Arc<BlockFlight>,
    completed: bool,
}

impl LeaderCleanupGuard {
    pub(super) fn finish(mut self, outcome: Result<FetchedBlocks, SharedRangeError>) {
        self.completed = true;
        self.complete_with(outcome);
    }

    fn complete_with(&self, outcome: Result<FetchedBlocks, SharedRangeError>) {
        {
            let mut map = lock_unpoisoned(&self.table.in_flight);
            for &blk in &self.blocks {
                if map
                    .get(&blk)
                    .is_some_and(|existing| Arc::ptr_eq(existing, &self.flight))
                {
                    map.remove(&blk);
                }
            }
        }
        {
            let mut st = lock_unpoisoned(&self.flight.state);
            st.finished = true;
            if st.outcome.is_none() {
                st.outcome = Some(outcome);
            }
        }
        self.flight.done.notify_waiters();
    }
}

impl Drop for LeaderCleanupGuard {
    fn drop(&mut self) {
        if !self.completed {
            self.complete_with(Err(SharedRangeError::Aborted));
        }
    }
}

/// Per-source singleflight map deduplicating concurrent block fetches across
/// aligned 64 KiB block indices.
pub(super) struct SingleflightTable {
    in_flight: Mutex<HashMap<u64, Arc<BlockFlight>>>,
}

pub(super) struct NewFlightSpan {
    pub(super) blocks: Vec<u64>,
    pub(super) flight: Arc<BlockFlight>,
    pub(super) waiter: WaiterGuard,
    pub(super) leader_guard: LeaderCleanupGuard,
}

pub(super) struct ClaimPlan {
    pub(super) joined: Vec<(Arc<BlockFlight>, WaiterGuard)>,
    pub(super) started: Vec<NewFlightSpan>,
}

impl SingleflightTable {
    pub(super) fn new() -> Self {
        Self {
            in_flight: Mutex::new(HashMap::new()),
        }
    }

    /// Partition `missing_blocks` (sorted, deduplicated block indices) into
    /// existing flights to join and contiguous new spans to launch, atomically
    /// registering new flights and attaching `WaiterGuard`s before unlocking.
    pub(super) fn claim_spans(
        self: &Arc<Self>,
        missing_blocks: &[u64],
        max_gap_blocks: u64,
        max_span_blocks: u64,
        parent_cancel: &CancelToken,
    ) -> ClaimPlan {
        let mut map = lock_unpoisoned(&self.in_flight);
        let mut joined: Vec<(Arc<BlockFlight>, WaiterGuard)> = Vec::new();
        let mut unclaimed: Vec<u64> = Vec::new();

        for &blk in missing_blocks {
            if let Some(existing) = map.get(&blk) {
                if joined.iter().any(|(f, _)| Arc::ptr_eq(f, existing)) {
                    continue;
                }
                if let Some(guard) = existing.try_add_waiter() {
                    joined.push((Arc::clone(existing), guard));
                    continue;
                }
            }
            unclaimed.push(blk);
        }

        // Coalesce contiguous/nearby unclaimed blocks into bounded HTTP spans.
        let mut started = Vec::new();
        let mut i = 0usize;
        while i < unclaimed.len() {
            let span_start = unclaimed[i];
            let mut span_end = span_start;
            i += 1;
            while i < unclaimed.len() {
                let next = unclaimed[i];
                let gap = next.saturating_sub(span_end + 1);
                let new_width = next.saturating_sub(span_start) + 1;
                if gap <= max_gap_blocks && new_width <= max_span_blocks {
                    // Only merge across gaps if no gap block is already in flight.
                    let gap_clear = ((span_end + 1)..next).all(|b| !map.contains_key(&b));
                    if gap_clear {
                        span_end = next;
                        i += 1;
                        continue;
                    }
                }
                break;
            }

            let blocks: Vec<u64> = (span_start..=span_end).collect();
            let flight = Arc::new(BlockFlight::new(parent_cancel));
            let waiter = flight.add_waiter();
            for &blk in &blocks {
                map.insert(blk, Arc::clone(&flight));
            }
            let leader_guard = LeaderCleanupGuard {
                table: Arc::clone(self),
                blocks: blocks.clone(),
                flight: Arc::clone(&flight),
                completed: false,
            };
            started.push(NewFlightSpan {
                blocks,
                flight,
                waiter,
                leader_guard,
            });
        }

        ClaimPlan { joined, started }
    }
}
