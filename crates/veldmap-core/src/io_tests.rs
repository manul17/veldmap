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

#[test]
fn buffered_reader_caches_window_and_counting_source_tracks_spans() {
    use std::io::{Read, Seek, SeekFrom};

    struct ChunkySource(Vec<u8>);
    impl ByteSource for ChunkySource {
        fn len(&self) -> u64 {
            self.0.len() as u64
        }
        fn read_at(
            &self,
            offset: u64,
            buf: &mut [u8],
            cancel: &CancelToken,
        ) -> Result<usize, IoError> {
            if cancel.is_cancelled() {
                return Err(IoError::Cancelled);
            }
            if offset >= self.0.len() as u64 || buf.is_empty() {
                return Ok(0);
            }
            // Simulate short reads (at most 5 bytes per call).
            let start = offset as usize;
            let n = (self.0.len() - start).min(buf.len()).min(5);
            buf[..n].copy_from_slice(&self.0[start..start + n]);
            Ok(n)
        }
    }

    let payload: Vec<u8> = (0..128u8).collect();
    let base: Arc<dyn ByteSource> = Arc::new(ChunkySource(payload.clone()));
    let counting = Arc::new(CountingByteSource::new(base));
    let cancel = CancelToken::new();

    // `read_exact_at` loops across short reads cleanly.
    let mut exact = [0u8; 17];
    counting
        .read_exact_at(10, &mut exact, &cancel)
        .expect("read_exact_at succeeds across short reads");
    assert_eq!(&exact, &payload[10..27]);
    assert!(counting.read_calls() >= 4);

    // `BufferedByteReader` satisfies small sequential & backward reads from its window.
    counting.reset();
    let mut reader = BufferedByteReader::with_capacity(
        Arc::clone(&counting) as Arc<dyn ByteSource>,
        cancel.clone(),
        64,
    );
    let mut two = [0u8; 2];
    reader.read_exact(&mut two).expect("read first two");
    assert_eq!(two, [0, 1]);
    let calls_after_first = counting.read_calls();
    reader
        .seek(SeekFrom::Start(0))
        .expect("seek back inside window");
    let mut four = [0u8; 4];
    reader.read_exact(&mut four).expect("read inside window");
    assert_eq!(four, [0, 1, 2, 3]);
    assert_eq!(counting.read_calls(), calls_after_first);

    // Fingerprint formatting produces 64 safe lowercase hex chars and non-zero SourceId.
    let fp = SourceFingerprint::from_digest([0xAB; 32]);
    let hex = fp.to_hex();
    assert_eq!(hex.len(), 64);
    assert!(
        hex.bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
    );
    assert_ne!(fp.to_source_id(), crate::tile::SourceId::BUNDLED_BASEMAP);
    assert!(format!("{fp:?}").contains("abababababab.."));

    // Cancelled reads fail promptly (including `std::io::Read::read_exact`,
    // which must NOT spin on `ErrorKind::Interrupted`).
    let reader_cancel = CancelToken::new();
    let mut cancel_reader = BufferedByteReader::with_capacity(
        Arc::new(ChunkySource(payload.clone())),
        reader_cancel.clone(),
        16,
    );
    let mut head = [0u8; 4];
    cancel_reader.read_exact(&mut head).expect("read head");
    assert_eq!(head, [0, 1, 2, 3]);
    reader_cancel.cancel();
    let mut after_cancel = [0u8; 8];
    let err = cancel_reader
        .read_exact(&mut after_cancel)
        .expect_err("cancelled read_exact must abort without spinning");
    assert_ne!(err.kind(), std::io::ErrorKind::Interrupted);
    assert!(
        err.get_ref()
            .and_then(|e| e.downcast_ref::<IoError>())
            .is_some_and(IoError::is_cancelled)
    );

    // Out-of-window backward seek, SeekFrom::Current/End, and large-read bypass.
    let mut nav_reader = BufferedByteReader::with_capacity(
        Arc::clone(&counting) as Arc<dyn ByteSource>,
        CancelToken::new(),
        16,
    );
    let mut large = vec![0u8; 32];
    nav_reader
        .read_exact(&mut large)
        .expect("large bypass read");
    assert_eq!(large, &payload[0..32]);
    assert_eq!(
        nav_reader
            .seek(SeekFrom::Current(-16))
            .expect("seek current"),
        16
    );
    let mut mid = [0u8; 4];
    nav_reader
        .read_exact(&mut mid)
        .expect("read after seek back");
    assert_eq!(mid, [16, 17, 18, 19]);
    assert_eq!(nav_reader.seek(SeekFrom::End(-4)).expect("seek end"), 124);
    nav_reader.read_exact(&mut mid).expect("read tail");
    assert_eq!(mid, [124, 125, 126, 127]);

    // Faulty ByteSource returning `n > buf.len()` fails cleanly instead of panicking.
    struct OverlongSource;
    impl ByteSource for OverlongSource {
        fn len(&self) -> u64 {
            64
        }
        fn read_at(&self, _: u64, buf: &mut [u8], _: &CancelToken) -> Result<usize, IoError> {
            Ok(buf.len() + 1)
        }
    }
    let overlong_tok = CancelToken::new();
    let mut small = [0u8; 4];
    assert!(matches!(
        OverlongSource.read_exact_at(0, &mut small, &overlong_tok),
        Err(IoError::InvalidInput(_))
    ));
    let mut bad_reader =
        BufferedByteReader::with_capacity(Arc::new(OverlongSource), overlong_tok, 8);
    assert!(bad_reader.read_exact(&mut small).is_err());

    cancel.cancel();
    assert!(
        counting
            .read_at(0, &mut two, &cancel)
            .is_err_and(|e| e.is_cancelled())
    );
}
