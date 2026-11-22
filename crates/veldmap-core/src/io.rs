//! Pure-`std` I/O coordination primitives (`CancelToken`).
//!
//! Keeps `veldmap-core` permanently free of async runtime (`tokio`) or HTTP
//! (`reqwest`) dependencies while allowing synchronous decoders and async HTTP
//! tasks (`tokio::select!`) to observe immediate cancellation without polling.

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::task::{Context, Poll, Waker};

type CancelCallback = Box<dyn FnOnce() + Send + 'static>;

#[derive(Default)]
struct WaiterTable {
    next_id: u64,
    wakers: Vec<(u64, Waker)>,
    callbacks: Vec<CancelCallback>,
    children: Vec<Weak<CancelState>>,
}

struct CancelState {
    cancelled: AtomicBool,
    waiters: Mutex<WaiterTable>,
}

fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
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
        for cb in callbacks {
            cb();
        }
    }

    /// Register a synchronous callback to run once when this token is
    /// cancelled. If the token is already cancelled, `f` runs immediately on
    /// the caller's thread.
    pub fn on_cancel(&self, f: impl FnOnce() + Send + 'static) {
        let already_cancelled = {
            let mut table = lock_unpoisoned(&self.state.waiters);
            if self.state.cancelled.load(Ordering::Acquire) {
                true
            } else {
                table.callbacks.push(Box::new(f));
                return;
            }
        };
        if already_cancelled {
            f();
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

/// Future returned by [`CancelToken::cancelled`].
pub struct CancelledFuture {
    state: Arc<CancelState>,
    waiter_id: Option<u64>,
}

impl Future for CancelledFuture {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if self.state.cancelled.load(Ordering::Acquire) {
            return Poll::Ready(());
        }

        let mut table = lock_unpoisoned(&self.state.waiters);
        // Re-check under the lock to prevent a lost wakeup racing with `cancel()`.
        if self.state.cancelled.load(Ordering::Acquire) {
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
mod tests {
    use std::sync::atomic::AtomicUsize;
    use std::task::{Wake, Waker};
    use std::thread;

    use super::*;

    struct CountWaker(AtomicUsize);

    impl Wake for CountWaker {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::AcqRel);
        }

        fn wake_by_ref(self: &Arc<Self>) {
            self.0.fetch_add(1, Ordering::AcqRel);
        }
    }

    #[test]
    fn sync_and_async_cancellation_wakes_waiters_and_cleans_up_dropped_futures() {
        let token = CancelToken::new();
        assert!(!token.is_cancelled());

        let w1 = Arc::new(CountWaker(AtomicUsize::new(0)));
        let w2 = Arc::new(CountWaker(AtomicUsize::new(0)));
        let waker1 = Waker::from(Arc::clone(&w1));
        let waker2 = Waker::from(Arc::clone(&w2));
        let mut cx1 = Context::from_waker(&waker1);
        let mut cx2 = Context::from_waker(&waker2);

        let mut fut1 = token.cancelled();
        let mut fut2 = token.cancelled();
        assert!(Pin::new(&mut fut1).poll(&mut cx1).is_pending());
        assert!(Pin::new(&mut fut2).poll(&mut cx2).is_pending());

        // Drop `fut2` before cancellation -> its waker must be unregistered.
        drop(fut2);
        assert_eq!(lock_unpoisoned(&token.state.waiters).wakers.len(), 1);

        let cb_hits = Arc::new(AtomicUsize::new(0));
        let cb_hits_clone = Arc::clone(&cb_hits);
        token.on_cancel(move || {
            cb_hits_clone.fetch_add(1, Ordering::AcqRel);
        });

        let child = token.child_token();
        assert!(!child.is_cancelled());

        token.cancel();
        // Idempotent second cancel does not double-wake or double-fire callbacks.
        token.cancel();

        assert!(token.is_cancelled());
        assert!(child.is_cancelled());
        assert_eq!(w1.0.load(Ordering::Acquire), 1);
        assert_eq!(w2.0.load(Ordering::Acquire), 0);
        assert_eq!(cb_hits.load(Ordering::Acquire), 1);
        assert!(Pin::new(&mut fut1).poll(&mut cx1).is_ready());

        // Already-cancelled fast path invokes `on_cancel` and resolves `cancelled()` immediately.
        let immediate_hits = Arc::new(AtomicUsize::new(0));
        let immediate_clone = Arc::clone(&immediate_hits);
        token.on_cancel(move || {
            immediate_clone.fetch_add(1, Ordering::AcqRel);
        });
        assert_eq!(immediate_hits.load(Ordering::Acquire), 1);

        let mut fut3 = token.cancelled();
        assert!(Pin::new(&mut fut3).poll(&mut cx1).is_ready());

        // Child of already-cancelled parent starts cancelled.
        assert!(token.child_token().is_cancelled());
    }

    #[test]
    fn dropped_child_tokens_are_pruned_without_leaking_parent_memory() {
        let parent = CancelToken::new();
        for _ in 0..1024 {
            let child = parent.child_token();
            drop(child);
        }
        let live_child = parent.child_token();
        {
            let table = lock_unpoisoned(&parent.state.waiters);
            assert_eq!(
                table.children.len(),
                1,
                "dropped children must be pruned on child_token()"
            );
            assert!(table.callbacks.is_empty());
        }
        parent.cancel();
        assert!(live_child.is_cancelled());
    }

    #[test]
    fn cancel_race_stress_never_loses_wakeup() {
        for _ in 0..256 {
            let token = CancelToken::new();
            let w = Arc::new(CountWaker(AtomicUsize::new(0)));
            let waker = Waker::from(Arc::clone(&w));
            let mut fut = token.cancelled();

            let canceller = token.clone();
            let handle = thread::spawn(move || {
                canceller.cancel();
            });

            let mut cx = Context::from_waker(&waker);
            let first = Pin::new(&mut fut).poll(&mut cx);
            handle.join().expect("join canceller");

            if first.is_pending() {
                assert_eq!(
                    w.0.load(Ordering::Acquire),
                    1,
                    "pending future missed cancel wakeup"
                );
            }
            assert!(Pin::new(&mut fut).poll(&mut cx).is_ready());
        }
    }
}
