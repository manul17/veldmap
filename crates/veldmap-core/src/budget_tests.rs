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
    let handle2 = std::thread::spawn(move || tracker_clone2.reserve_blocking(60, &cancel_clone2));
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
