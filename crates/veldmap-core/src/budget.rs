//! Startup host-RAM-scaled fixed per-cache memory budgets (`MemoryBudget`),
//! named working-memory peak estimation (`DecodePeak`), and thread-safe active
//! decode memory reservation (`DecodeBudgetTracker`).
//!
//! Avoids lock-reentrant cross-cache shrinking by assigning independent fixed
//! startup caps to each cache and decoding lane.

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};

use thiserror::Error;

use crate::io::{CancelToken, lock_unpoisoned};

const MIB: u64 = 1024 * 1024;
const GIB: u64 = 1024 * MIB;

/// Fallback assumed host physical RAM (`8 GiB`) when OS RAM inspection is
/// unavailable.
pub const DEFAULT_FALLBACK_HOST_RAM_BYTES: u64 = 8 * GIB;

/// Fixed startup host-RAM-scaled memory caps across VeldMap's CPU and GPU
/// caches and decode lanes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryBudget {
    /// Detected or configured physical host RAM in bytes.
    pub host_ram_bytes: u64,
    /// Concurrent active decode working-memory ceiling in bytes.
    pub decode_working_bytes: u64,
    /// Bounded LRU decoded-chunk cache byte cap for pointwise raster reads.
    pub decoded_chunk_cache_bytes: usize,
    /// Per-tile source region ceiling in pixels (derived as half of the chunk
    /// cache at 4 bytes/pixel so pointwise reads stay bounded).
    pub region_cap_pixels: u64,
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
        let region_cap_pixels = decoded_chunk_cache_bytes / 8;
        let lane_pending = (ram / 128).clamp(32 * MIB, 128 * MIB) as usize;
        let gpu_tile_cache_bytes = (ram / 16).clamp(256 * MIB, GIB) as usize;
        let queued_upload_bytes = (ram / 128).clamp(64 * MIB, 128 * MIB) as usize;

        Self {
            host_ram_bytes: ram,
            decode_working_bytes,
            decoded_chunk_cache_bytes: decoded_chunk_cache_bytes as usize,
            region_cap_pixels,
            fast_lane_pending_bytes: lane_pending,
            bulk_lane_pending_bytes: lane_pending,
            gpu_tile_cache_bytes,
            queued_upload_bytes,
        }
    }

    /// Detect physical host RAM and derive startup caps, falling back cleanly
    /// to [`DEFAULT_FALLBACK_HOST_RAM_BYTES`] (`8 GiB`) if unsupported.
    pub fn detect() -> Self {
        Self::for_host_ram(detect_host_ram_bytes().unwrap_or(DEFAULT_FALLBACK_HOST_RAM_BYTES))
    }
}

impl Default for MemoryBudget {
    fn default() -> Self {
        Self::detect()
    }
}

fn detect_host_ram_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let contents = std::fs::read_to_string("/proc/meminfo").ok()?;
        parse_proc_meminfo_total_bytes(&contents)
    }
    #[cfg(target_os = "macos")]
    {
        let out = std::process::Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()?;
        if !out.status.success() {
            return None;
        }
        let text = std::str::from_utf8(&out.stdout).ok()?.trim();
        text.parse::<u64>().ok()
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        None
    }
}

#[cfg(any(target_os = "linux", test))]
fn parse_proc_meminfo_total_bytes(meminfo: &str) -> Option<u64> {
    for line in meminfo.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            let kib_str = rest.split_whitespace().next()?;
            let kib = kib_str.parse::<u64>().ok()?;
            return kib.checked_mul(1024);
        }
    }
    None
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

/// Named working-memory peak breakdown for one decode operation or pyramid
/// level row.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DecodePeak {
    terms: Vec<(&'static str, u64)>,
}

impl DecodePeak {
    /// Construct an empty peak accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a named cost term in bytes.
    pub fn with(mut self, what: &'static str, bytes: u64) -> Self {
        self.terms.push((what, bytes));
        self
    }

    /// Return the named terms slice.
    pub fn terms(&self) -> &[(&'static str, u64)] {
        &self.terms
    }

    /// Sum of all named terms in bytes (saturating on overflow).
    pub fn total(&self) -> u64 {
        self.terms
            .iter()
            .map(|(_, bytes)| *bytes)
            .fold(0u64, u64::saturating_add)
    }

    /// Return whether `total() <= limit_bytes`.
    pub fn fits(&self, limit_bytes: u64) -> bool {
        self.total() <= limit_bytes
    }

    /// Validate that `total() <= limit_bytes`, returning a structured
    /// [`BudgetError::ExceedsCap`] naming every component when refused.
    pub fn admit(&self, limit_bytes: u64) -> Result<(), BudgetError> {
        let requested = self.total();
        if requested <= limit_bytes {
            Ok(())
        } else {
            Err(BudgetError::ExceedsCap {
                requested,
                cap: limit_bytes,
                detail: self.note(),
            })
        }
    }

    /// Human-readable summary of terms (e.g. `"band 128 MiB, cascade 48 MiB"`).
    pub fn note(&self) -> String {
        self.terms
            .iter()
            .map(|(what, bytes)| format!("{what} {} MiB", bytes / MIB))
            .collect::<Vec<_>>()
            .join(", ")
    }
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

    /// Attempt to reserve `bytes` immediately without blocking.
    ///
    /// Returns `Err(BudgetError::ExceedsCap)` when `bytes > cap_bytes()`,
    /// `Ok(None)` when sufficient budget is occupied or queued ahead,
    /// or `Ok(Some(lease))` on success.
    pub fn try_reserve(&self, bytes: u64) -> Result<Option<BudgetLease>, BudgetError> {
        if bytes > self.inner.cap_bytes {
            return Err(BudgetError::ExceedsCap {
                requested: bytes,
                cap: self.inner.cap_bytes,
                detail: format!("requested {bytes} B > cap {} B", self.inner.cap_bytes),
            });
        }
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
        if bytes > self.inner.cap_bytes {
            return Err(BudgetError::ExceedsCap {
                requested: bytes,
                cap: self.inner.cap_bytes,
                detail: format!("requested {bytes} B > cap {} B", self.inner.cap_bytes),
            });
        }
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
            state = match self.inner.cvar.wait(state) {
                Ok(guard) => guard,
                Err(poisoned) => poisoned.into_inner(),
            };
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
mod tests {
    use super::*;

    #[test]
    fn host_ram_budget_scales_within_bounds_and_parses_meminfo() {
        let small = MemoryBudget::for_host_ram(512 * MIB);
        assert_eq!(small.host_ram_bytes, GIB);
        assert_eq!(small.decode_working_bytes, 256 * MIB);
        assert_eq!(small.decoded_chunk_cache_bytes, 64 * MIB as usize);
        assert_eq!(small.gpu_tile_cache_bytes, 256 * MIB as usize);

        let mid = MemoryBudget::for_host_ram(16 * GIB);
        assert_eq!(mid.decode_working_bytes, 2 * GIB);
        assert_eq!(mid.decoded_chunk_cache_bytes, 512 * MIB as usize);
        assert_eq!(mid.gpu_tile_cache_bytes, GIB as usize);

        let sample = "MemTotal:       32845120 kB\nMemFree:        123456 kB\n";
        assert_eq!(
            parse_proc_meminfo_total_bytes(sample),
            Some(32_845_120 * 1024)
        );
    }

    #[test]
    fn decode_budget_tracker_enforces_cap_and_cancels_waiters() {
        let tracker = DecodeBudgetTracker::new(100);
        assert!(matches!(
            tracker.try_reserve(101),
            Err(BudgetError::ExceedsCap {
                requested: 101,
                cap: 100,
                ..
            })
        ));

        let lease1 = tracker.try_reserve(60).unwrap().expect("first lease");
        assert_eq!(tracker.active_bytes(), 60);
        assert!(tracker.try_reserve(50).unwrap().is_none());

        let cancel = CancelToken::new();
        let tracker_clone = tracker.clone();
        let cancel_clone = cancel.clone();
        let handle = std::thread::spawn(move || tracker_clone.reserve_blocking(50, &cancel_clone));

        // Give waiter a moment to block, then drop lease1 so it acquires 50.
        std::thread::sleep(std::time::Duration::from_millis(20));
        drop(lease1);
        let lease2 = handle.join().unwrap().expect("waiter acquired after drop");
        assert_eq!(tracker.active_bytes(), 50);
        assert_eq!(tracker.peak_bytes(), 60);

        // Cancelled waiter returns promptly.
        let cancel2 = CancelToken::new();
        let tracker_clone2 = tracker.clone();
        let cancel_clone2 = cancel2.clone();
        let handle2 =
            std::thread::spawn(move || tracker_clone2.reserve_blocking(60, &cancel_clone2));
        std::thread::sleep(std::time::Duration::from_millis(20));
        cancel2.cancel();
        assert_eq!(handle2.join().unwrap().err(), Some(BudgetError::Cancelled));

        drop(lease2);
        assert_eq!(tracker.active_bytes(), 0);

        // Large cap (`u64::MAX`) must never overflow or wrap in try_reserve/reserve_blocking.
        let huge = DecodeBudgetTracker::new(u64::MAX);
        let h1 = huge.try_reserve(100).unwrap().expect("reserve 100");
        assert!(huge.try_reserve(u64::MAX - 50).unwrap().is_none());
        drop(h1);

        // FIFO waiter queue prevents small concurrent reservations from starving a large pass.
        let fifo = DecodeBudgetTracker::new(100);
        let held = fifo.try_reserve(60).unwrap().expect("hold 60");
        let (tx_queued, rx_queued) = std::sync::mpsc::channel();
        let fifo_large = fifo.clone();
        let large_handle = std::thread::spawn(move || {
            let _ = tx_queued.send(());
            fifo_large.reserve_blocking(80, &CancelToken::new())
        });
        rx_queued.recv().unwrap();
        // Wait until the large waiter is enqueued, then verify small try_reserve yields to it.
        while lock_unpoisoned(&fifo.inner.state).waiters.is_empty() {
            std::thread::yield_now();
        }
        assert!(fifo.try_reserve(20).unwrap().is_none());
        drop(held);
        let large_lease = large_handle.join().unwrap().expect("large lease succeeds");
        assert_eq!(large_lease.bytes(), 80);
        drop(large_lease);

        // DecodePeak::admit / note cover structured errors.
        let peak = DecodePeak::new()
            .with("band", 4 * MIB)
            .with("cascade", 2 * MIB);
        assert!(peak.admit(6 * MIB).is_ok());
        assert!(matches!(
            peak.admit(5 * MIB),
            Err(BudgetError::ExceedsCap { detail, .. }) if detail.contains("band 4 MiB")
        ));
    }
}
