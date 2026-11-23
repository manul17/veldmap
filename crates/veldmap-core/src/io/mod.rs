//! Pure-`std` I/O and worker coordination primitives (`CancelToken`, `Wakeup`,
//! and [`lock_unpoisoned`]).
//!
//! Keeps `veldmap-core` permanently free of async runtime (`tokio`) or HTTP
//! (`reqwest`) dependencies while allowing synchronous decoders, local storage
//! workers, and async HTTP tasks (`tokio::select!`) to share immediate
//! cancellation and UI-repaint wakeup callbacks without coupling offline
//! subsystems to `veldmap-net`.

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::task::{Context, Poll, Waker};

pub mod byte_source;

pub use byte_source::{
    BufferedByteReader, ByteReadSpan, ByteSource, CountingByteSource, DEFAULT_READER_BUFFER_BYTES,
    IoError, MemoryByteSource, ReadStats, SourceFingerprint,
};

type CancelCallback = Box<dyn FnOnce() + Send + 'static>;

#[derive(Default)]
struct WaiterTable {
    next_id: u64,
    wakers: Vec<(u64, Waker)>,
    callbacks: Vec<(u64, CancelCallback)>,
    children: Vec<Weak<CancelState>>,
}

struct CancelState {
    cancelled: AtomicBool,
    waiters: Mutex<WaiterTable>,
}

/// UI repaint / event-loop wakeup callback invoked by background workers
/// (bundled basemap, CPU decode pool, disk-cache maintenance, HTTP fetcher)
/// when new tiles or status updates are ready.
pub type Wakeup = Arc<dyn Fn() + Send + Sync>;

/// Acquire `mutex`, recovering the inner guard if a previous holder panicked.
#[inline]
pub fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Zero-dependency, race-free cancellation token usable from both synchronous
/// worker threads and asynchronous `Future`s (`tokio::select!`).
#[derive(Clone)]
pub struct CancelToken {
    state: Arc<CancelState>,
}

impl std::fmt::Debug for CancelToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CancelToken")
            .field("cancelled", &self.is_cancelled())
            .finish()
    }
}

impl Default for CancelToken {
    fn default() -> Self {
        Self::new()
    }
}

impl CancelToken {
    /// Create a fresh uncancelled token.
    pub fn new() -> Self {
        Self {
            state: Arc::new(CancelState {
                cancelled: AtomicBool::new(false),
                waiters: Mutex::new(WaiterTable::default()),
            }),
        }
    }

    /// Return whether cancellation has been requested (lock-free fast path).
    #[inline]
    pub fn is_cancelled(&self) -> bool {
        self.state.cancelled.load(Ordering::Acquire)
    }

    /// Trigger cancellation, waking every awaiting [`CancelledFuture`],
    /// cancelling all live child tokens, and invoking every registered
    /// [`Self::on_cancel`] callback outside the internal lock. Idempotent if
    /// called more than once.
    pub fn cancel(&self) {
        let (wakers, callbacks, children) = {
            let mut table = lock_unpoisoned(&self.state.waiters);
            if self.state.cancelled.swap(true, Ordering::AcqRel) {
                return;
            }
            (
                std::mem::take(&mut table.wakers),
                std::mem::take(&mut table.callbacks),
                std::mem::take(&mut table.children),
            )
        };

        for (_, waker) in wakers {
            waker.wake();
        }
        for weak in children {
            if let Some(state) = weak.upgrade() {
                CancelToken { state }.cancel();
            }
        }
        for (_, cb) in callbacks {
            cb();
        }
    }

    /// Register a synchronous callback to run once when this token is
    /// cancelled. If the token is already cancelled, `f` runs immediately on
    /// the caller's thread.
    pub fn on_cancel(&self, f: impl FnOnce() + Send + 'static) {
        self.on_cancel_guarded(f).detach();
    }

    /// Register a synchronous callback to run once when this token is
    /// cancelled, returning a [`CancelGuard`] that unregisters the callback on
    /// drop (preventing long-lived shutdown tokens from retaining closures
    /// after a subsystem shuts down early).
    #[must_use = "dropping CancelGuard unregisters the callback; call .detach() or store the guard"]
    pub fn on_cancel_guarded(&self, f: impl FnOnce() + Send + 'static) -> CancelGuard {
        let already_cancelled = {
            let mut table = lock_unpoisoned(&self.state.waiters);
            if self.state.cancelled.load(Ordering::Acquire) {
                true
            } else {
                let id = table.next_id;
                table.next_id = table.next_id.wrapping_add(1);
                table.callbacks.push((id, Box::new(f)));
                return CancelGuard {
                    state: Arc::downgrade(&self.state),
                    id: Some(id),
                };
            }
        };
        if already_cancelled {
            f();
        }
        CancelGuard {
            state: Weak::new(),
            id: None,
        }
    }

    /// Create a child token that cancels automatically when `self` cancels.
    ///
    /// Cancelling the child does not affect the parent. Dead child `Weak`
    /// handles are pruned automatically so long-lived shutdown tokens spawning
    /// per-tile children never accumulate unbounded memory.
    pub fn child_token(&self) -> Self {
        let child = Self::new();
        let mut table = lock_unpoisoned(&self.state.waiters);
        if self.state.cancelled.load(Ordering::Acquire) {
            drop(table);
            child.cancel();
            return child;
        }
        table.children.retain(|weak| weak.strong_count() > 0);
        table.children.push(Arc::downgrade(&child.state));
        child
    }

    /// Return a `Future` that completes as soon as this token is cancelled.
    ///
    /// Dropping the returned future before cancellation unregisters its waker
    /// so long-lived shutdown tokens never accumulate stale wakers.
    pub fn cancelled(&self) -> CancelledFuture {
        CancelledFuture {
            state: Arc::clone(&self.state),
            waiter_id: None,
        }
    }
}

/// RAII guard returned by [`CancelToken::on_cancel_guarded`]; dropping it
/// unregisters the callback without holding the token's internal lock while
/// dropping the closure.
#[must_use = "dropping CancelGuard unregisters the callback; call .detach() or store the guard"]
pub struct CancelGuard {
    state: Weak<CancelState>,
    id: Option<u64>,
}

impl CancelGuard {
    /// Leave the callback registered until the token cancels (fire-and-forget).
    pub fn detach(mut self) {
        self.id = None;
    }
}

impl Drop for CancelGuard {
    fn drop(&mut self) {
        let Some(id) = self.id.take() else {
            return;
        };
        let Some(state) = self.state.upgrade() else {
            return;
        };
        // Move the removed callback out of the lock before dropping it so a
        // callback holding nested locks or guards can never deadlock.
        let removed = {
            let mut table = lock_unpoisoned(&state.waiters);
            table
                .callbacks
                .iter()
                .position(|(cid, _)| *cid == id)
                .map(|pos| table.callbacks.swap_remove(pos))
        };
        drop(removed);
    }
}

/// Future returned by [`CancelToken::cancelled`].
pub struct CancelledFuture {
    state: Arc<CancelState>,
    waiter_id: Option<u64>,
}

impl Future for CancelledFuture {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if self.state.cancelled.load(Ordering::Acquire) {
            self.waiter_id = None;
            return Poll::Ready(());
        }

        let mut table = lock_unpoisoned(&self.state.waiters);
        // Re-check under the lock to prevent a lost wakeup racing with `cancel()`.
        if self.state.cancelled.load(Ordering::Acquire) {
            drop(table);
            self.waiter_id = None;
            return Poll::Ready(());
        }

        if let Some(id) = self.waiter_id {
            if let Some((_, existing)) = table.wakers.iter_mut().find(|(wid, _)| *wid == id) {
                existing.clone_from(cx.waker());
            } else {
                table.wakers.push((id, cx.waker().clone()));
            }
        } else {
            let id = table.next_id;
            table.next_id = table.next_id.wrapping_add(1);
            table.wakers.push((id, cx.waker().clone()));
            drop(table);
            self.waiter_id = Some(id);
            return Poll::Pending;
        }

        Poll::Pending
    }
}

impl Drop for CancelledFuture {
    fn drop(&mut self) {
        if let Some(id) = self.waiter_id.take() {
            let mut table = lock_unpoisoned(&self.state.waiters);
            if let Some(pos) = table.wakers.iter().position(|(wid, _)| *wid == id) {
                table.wakers.swap_remove(pos);
            }
        }
    }
}

#[cfg(test)]
#[path = "io_tests.rs"]
mod tests;
