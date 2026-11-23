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

#[test]
fn child_cancel_does_not_cancel_parent_and_guard_unregisters_cleanly() {
    let parent = CancelToken::new();
    let child = parent.child_token();

    let guard_hits = Arc::new(AtomicUsize::new(0));
    let guard_hits_clone = Arc::clone(&guard_hits);
    let guard = parent.on_cancel_guarded(move || {
        guard_hits_clone.fetch_add(1, Ordering::AcqRel);
    });
    drop(guard);

    child.cancel();
    assert!(child.is_cancelled());
    assert!(!parent.is_cancelled());

    // Re-polling CancelledFuture with a new waker replaces the stored waker.
    let w1 = Arc::new(CountWaker(AtomicUsize::new(0)));
    let w2 = Arc::new(CountWaker(AtomicUsize::new(0)));
    let waker1 = Waker::from(Arc::clone(&w1));
    let waker2 = Waker::from(Arc::clone(&w2));
    let mut cx1 = Context::from_waker(&waker1);
    let mut cx2 = Context::from_waker(&waker2);
    let mut fut = parent.cancelled();
    assert!(Pin::new(&mut fut).poll(&mut cx1).is_pending());
    assert!(Pin::new(&mut fut).poll(&mut cx2).is_pending());

    parent.cancel();
    assert_eq!(w1.0.load(Ordering::Acquire), 0);
    assert_eq!(w2.0.load(Ordering::Acquire), 1);
    assert_eq!(
        guard_hits.load(Ordering::Acquire),
        0,
        "dropped CancelGuard must unregister callback before cancel"
    );
    assert!(Pin::new(&mut fut).poll(&mut cx2).is_ready());
    assert!(fut.waiter_id.is_none());
}
