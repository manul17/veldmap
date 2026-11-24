//! Pure host-RAM-scaled per-cache memory budgets (`MemoryBudget`) and
//! thread-safe active decode memory reservation (`DecodeBudgetTracker`).
//!
//! Avoids lock-reentrant cross-cache shrinking by assigning independent fixed
//! startup caps to each cache and decoding lane.

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};

use thiserror::Error;

use crate::sync::{CancelToken, lock_unpoisoned};
use crate::tile::PixelFormat;

const MIB: u64 = 1024 * 1024;
const GIB: u64 = 1024 * MIB;

/// Fallback assumed host physical RAM (`8 GiB`) when OS RAM inspection is
/// unavailable.
pub const DEFAULT_FALLBACK_HOST_RAM_BYTES: u64 = 8 * GIB;

/// Fixed startup host-RAM-scaled memory caps across VeldMap's CPU and GPU
/// caches and decode lanes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryBudget {
    /// Configured physical host RAM in bytes (clamped to `[1 GiB, 256 GiB]`).
    pub host_ram_bytes: u64,
    /// Concurrent active decode working-memory ceiling in bytes.
    pub decode_working_bytes: u64,
    /// Bounded LRU decoded-chunk cache byte cap for pointwise raster reads.
    pub decoded_chunk_cache_bytes: usize,
    /// Per-tile source region ceiling in bytes (half of the chunk cache so
    /// pointwise reads stay bounded across all pixel formats).
    pub region_cap_bytes: u64,
    /// Fast/pointwise decode lane pending-queue byte cap.
    pub fast_lane_pending_bytes: usize,
    /// Bulk/pass/full-image decode lane pending-queue byte cap.
    pub bulk_lane_pending_bytes: usize,
    /// Baseline GPU tile VRAM cache cap (scaled above 256 MiB on larger hosts).
    pub gpu_tile_cache_bytes: usize,
    /// CPU-side queued decoded-tile byte cap awaiting GPU upload.
    pub queued_upload_bytes: usize,
}

impl MemoryBudget {
    /// Compute deterministic per-cache caps scaled from `host_ram_bytes`.
    pub fn for_host_ram(host_ram_bytes: u64) -> Self {
        let ram = host_ram_bytes.clamp(GIB, 256 * GIB);
        let decode_working_bytes = (ram / 8).clamp(256 * MIB, 2 * GIB);
        let decoded_chunk_cache_bytes = (decode_working_bytes / 4).clamp(64 * MIB, 512 * MIB);
        let region_cap_bytes = decoded_chunk_cache_bytes / 2;
        let lane_pending = (ram / 128).clamp(32 * MIB, 128 * MIB) as usize;
        let gpu_tile_cache_bytes = (ram / 16).clamp(256 * MIB, GIB) as usize;
        let queued_upload_bytes = (ram / 128).clamp(64 * MIB, 128 * MIB) as usize;

        Self {
            host_ram_bytes: ram,
            decode_working_bytes,
            decoded_chunk_cache_bytes: decoded_chunk_cache_bytes as usize,
            region_cap_bytes,
            fast_lane_pending_bytes: lane_pending,
            bulk_lane_pending_bytes: lane_pending,
            gpu_tile_cache_bytes,
            queued_upload_bytes,
        }
    }

    /// Derive the per-tile source region pixel ceiling for a concrete
    /// [`PixelFormat`] from [`Self::region_cap_bytes`].
    pub const fn region_cap_pixels_for(&self, format: PixelFormat) -> u64 {
        self.region_cap_bytes / (format.bytes_per_pixel() as u64)
    }
}

impl Default for MemoryBudget {
    /// Deterministic default budget assuming [`DEFAULT_FALLBACK_HOST_RAM_BYTES`].
    fn default() -> Self {
        Self::for_host_ram(DEFAULT_FALLBACK_HOST_RAM_BYTES)
    }
}

/// Errors returned when admitting or reserving decode working memory.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum BudgetError {
    #[error("decode working set {requested} B exceeds memory budget {cap} B ({detail})")]
    ExceedsCap {
        requested: u64,
        cap: u64,
        detail: String,
    },
    #[error("memory budget reservation cancelled")]
    Cancelled,
}

#[derive(Default)]
struct TrackerState {
    active_bytes: u64,
    peak_bytes: u64,
    next_ticket: u64,
    waiters: VecDeque<u64>,
}

struct WaiterGuard<'a> {
    inner: &'a TrackerInner,
    ticket: Option<u64>,
}

impl Drop for WaiterGuard<'_> {
    fn drop(&mut self) {
        let Some(ticket) = self.ticket.take() else {
            return;
        };
        let mut state = lock_unpoisoned(&self.inner.state);
        if let Some(pos) = state.waiters.iter().position(|&t| t == ticket) {
            state.waiters.remove(pos);
            if !state.waiters.is_empty() {
                self.inner.cvar.notify_all();
            }
        }
    }
}

struct TrackerInner {
    cap_bytes: u64,
    state: Mutex<TrackerState>,
    cvar: Condvar,
}

/// Thread-safe active decode working-memory tracker enforcing
/// `active_bytes <= cap_bytes` across concurrent decode workers.
#[derive(Clone)]
pub struct DecodeBudgetTracker {
    inner: Arc<TrackerInner>,
}

impl DecodeBudgetTracker {
    /// Create a new tracker capped at `cap_bytes`.
    pub fn new(cap_bytes: u64) -> Self {
        Self {
            inner: Arc::new(TrackerInner {
                cap_bytes,
                state: Mutex::new(TrackerState::default()),
                cvar: Condvar::new(),
            }),
        }
    }

    /// Configured cap in bytes.
    pub fn cap_bytes(&self) -> u64 {
        self.inner.cap_bytes
    }

    /// Currently reserved active working bytes.
    pub fn active_bytes(&self) -> u64 {
        lock_unpoisoned(&self.inner.state).active_bytes
    }

    /// High-water mark of `active_bytes` observed over the tracker's lifetime.
    pub fn peak_bytes(&self) -> u64 {
        lock_unpoisoned(&self.inner.state).peak_bytes
    }

    fn check_cap(&self, bytes: u64) -> Result<(), BudgetError> {
        if bytes > self.inner.cap_bytes {
            return Err(BudgetError::ExceedsCap {
                requested: bytes,
                cap: self.inner.cap_bytes,
                detail: format!("requested {bytes} B > cap {} B", self.inner.cap_bytes),
            });
        }
        Ok(())
    }

    /// Attempt to reserve `bytes` immediately without blocking.
    ///
    /// Returns `Err(BudgetError::ExceedsCap)` when `bytes > cap_bytes()`,
    /// `Ok(None)` when sufficient budget is occupied or queued ahead,
    /// or `Ok(Some(lease))` on success.
    pub fn try_reserve(&self, bytes: u64) -> Result<Option<BudgetLease>, BudgetError> {
        self.check_cap(bytes)?;
        let mut state = lock_unpoisoned(&self.inner.state);
        if !state.waiters.is_empty()
            || bytes > self.inner.cap_bytes.saturating_sub(state.active_bytes)
        {
            return Ok(None);
        }
        state.active_bytes += bytes;
        state.peak_bytes = state.peak_bytes.max(state.active_bytes);
        Ok(Some(BudgetLease {
            inner: Arc::clone(&self.inner),
            bytes,
        }))
    }

    /// Block in FIFO waiter order until `bytes` can be reserved within
    /// `cap_bytes()` or `cancel` triggers. Returns `Err(BudgetError::ExceedsCap)`
    /// immediately if `bytes > cap_bytes()`.
    pub fn reserve_blocking(
        &self,
        bytes: u64,
        cancel: &CancelToken,
    ) -> Result<BudgetLease, BudgetError> {
        self.check_cap(bytes)?;
        if cancel.is_cancelled() {
            return Err(BudgetError::Cancelled);
        }
        if let Some(lease) = self.try_reserve(bytes)? {
            return Ok(lease);
        }

        let wake_inner = Arc::clone(&self.inner);
        let _cancel_guard = cancel.on_cancel_guarded(move || {
            let _lock = lock_unpoisoned(&wake_inner.state);
            wake_inner.cvar.notify_all();
        });

        // Declared BEFORE `state` so on cancel/panic `state` drops first and
        // `WaiterGuard::drop` never re-locks `self.inner.state` while held.
        let mut waiter_guard = WaiterGuard {
            inner: &self.inner,
            ticket: None,
        };

        let mut state = lock_unpoisoned(&self.inner.state);
        if cancel.is_cancelled() {
            return Err(BudgetError::Cancelled);
        }
        let ticket = state.next_ticket;
        state.next_ticket = state.next_ticket.wrapping_add(1);
        state.waiters.push_back(ticket);
        waiter_guard.ticket = Some(ticket);

        loop {
            if cancel.is_cancelled() {
                return Err(BudgetError::Cancelled);
            }
            if state.waiters.front() == Some(&ticket)
                && bytes <= self.inner.cap_bytes.saturating_sub(state.active_bytes)
            {
                state.waiters.pop_front();
                waiter_guard.ticket = None;
                state.active_bytes += bytes;
                state.peak_bytes = state.peak_bytes.max(state.active_bytes);
                if !state.waiters.is_empty() {
                    self.inner.cvar.notify_all();
                }
                return Ok(BudgetLease {
                    inner: Arc::clone(&self.inner),
                    bytes,
                });
            }
            state = crate::sync::wait_unpoisoned(&self.inner.cvar, state);
        }
    }
}

/// RAII lease releasing reserved decode working memory back to
/// [`DecodeBudgetTracker`] on drop.
pub struct BudgetLease {
    inner: Arc<TrackerInner>,
    bytes: u64,
}

impl BudgetLease {
    /// Number of bytes held by this lease.
    pub const fn bytes(&self) -> u64 {
        self.bytes
    }
}

impl Drop for BudgetLease {
    fn drop(&mut self) {
        let mut state = lock_unpoisoned(&self.inner.state);
        state.active_bytes = state.active_bytes.saturating_sub(self.bytes);
        self.inner.cvar.notify_all();
    }
}

#[cfg(test)]
#[path = "budget_tests.rs"]
mod tests;
