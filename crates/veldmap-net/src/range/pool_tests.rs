use super::*;

#[test]
fn lru_pool_evicts_least_recently_used_and_respects_cap() {
    let pool = RangeBlockPool::new(2 * RANGE_BLOCK_SIZE);
    let k0 = BlockPoolKey::Isolated {
        instance_id: 1,
        block_idx: 0,
    };
    let k1 = BlockPoolKey::Isolated {
        instance_id: 1,
        block_idx: 1,
    };
    let k2 = BlockPoolKey::Isolated {
        instance_id: 1,
        block_idx: 2,
    };
    pool.insert(k0.clone(), Arc::from(vec![0u8; RANGE_BLOCK_SIZE]));
    pool.insert(k1.clone(), Arc::from(vec![1u8; RANGE_BLOCK_SIZE]));
    // Promote k0 so k1 becomes LRU.
    assert!(pool.get(&k0).is_some());
    pool.insert(k2.clone(), Arc::from(vec![2u8; RANGE_BLOCK_SIZE]));
    assert!(pool.get(&k0).is_some());
    assert!(pool.get(&k1).is_none());
    assert!(pool.get(&k2).is_some());
    assert_eq!(pool.stats(), (2, 2 * RANGE_BLOCK_SIZE));
}

#[test]
fn prefetch_budget_enforces_cap_wakes_waiters_and_honors_cancel() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("rt");
    rt.block_on(async {
        let budget = Arc::new(PrefetchBudget::new(2 * RANGE_BLOCK_SIZE));
        let cancel = CancelToken::new();

        // Exceeding cap returns InvalidInput immediately.
        let over = budget
            .reserve(3 * RANGE_BLOCK_SIZE, &cancel)
            .await
            .err()
            .expect("over cap");
        assert!(matches!(over, IoError::InvalidInput(_)));

        // Acquire full budget, verify cancelled waiter aborts, then release wakes waiter.
        let lease_a = budget
            .reserve(2 * RANGE_BLOCK_SIZE, &cancel)
            .await
            .expect("lease_a");

        let cancel_waiter = CancelToken::new();
        let b_clone = Arc::clone(&budget);
        let cw_clone = cancel_waiter.clone();
        let cancelled_task =
            tokio::spawn(async move { b_clone.reserve(RANGE_BLOCK_SIZE, &cw_clone).await });
        tokio::task::yield_now().await;
        cancel_waiter.cancel();
        let res = cancelled_task.await.expect("join");
        assert!(matches!(res, Err(IoError::Cancelled)));

        let b_clone2 = Arc::clone(&budget);
        let ok_cancel = CancelToken::new();
        let ok_task =
            tokio::spawn(async move { b_clone2.reserve(RANGE_BLOCK_SIZE, &ok_cancel).await });
        tokio::task::yield_now().await;
        drop(lease_a);
        let lease_b = ok_task.await.expect("join").expect("woken lease");
        drop(lease_b);
    });
}

#[test]
fn prefetch_budget_enforces_fifo_order_and_wakes_on_head_drop() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("rt");
    rt.block_on(async {
        let budget = Arc::new(PrefetchBudget::new(4 * RANGE_BLOCK_SIZE));
        let cancel = CancelToken::new();

        // Hold 2 blocks so 2 blocks remain free.
        let initial = budget
            .reserve(2 * RANGE_BLOCK_SIZE, &cancel)
            .await
            .expect("initial");

        // Large waiter (3 blocks) cannot fit yet and enters the FIFO queue.
        let b_large = Arc::clone(&budget);
        let c_large = CancelToken::new();
        let large_task = tokio::spawn(async move {
            b_large
                .reserve(3 * RANGE_BLOCK_SIZE, &c_large)
                .await
                .expect("large")
        });
        tokio::task::yield_now().await;

        // Smaller waiter (1 block) would fit in the remaining 2 blocks, but
        // must wait behind the queued large waiter (FIFO non-starvation).
        let b_small = Arc::clone(&budget);
        let c_small = CancelToken::new();
        let small_task = tokio::spawn(async move {
            b_small
                .reserve(RANGE_BLOCK_SIZE, &c_small)
                .await
                .expect("small")
        });
        tokio::task::yield_now().await;
        assert!(
            !small_task.is_finished(),
            "small waiter must not jump ahead of queued large waiter"
        );

        // Releasing `initial` wakes `large_task`, which acquires 3 blocks
        // and immediately wakes `small_task` for the remaining 1 block.
        drop(initial);
        let large_lease = large_task.await.expect("large join");
        let small_lease = small_task.await.expect("small join");
        drop(small_lease);
        drop(large_lease);

        // Dropping/aborting a queued head waiter must wake the next waiter.
        let hold = budget
            .reserve(2 * RANGE_BLOCK_SIZE, &cancel)
            .await
            .expect("hold");
        let b_drop = Arc::clone(&budget);
        let c_drop = CancelToken::new();
        let head_task =
            tokio::spawn(async move { b_drop.reserve(3 * RANGE_BLOCK_SIZE, &c_drop).await });
        tokio::task::yield_now().await;

        let b_tail = Arc::clone(&budget);
        let c_tail = CancelToken::new();
        let tail_task =
            tokio::spawn(async move { b_tail.reserve(RANGE_BLOCK_SIZE, &c_tail).await });
        tokio::task::yield_now().await;
        assert!(!tail_task.is_finished());

        head_task.abort();
        let _ = head_task.await;
        let tail_lease = tail_task.await.expect("tail join").expect("tail woken");
        drop(tail_lease);
        drop(hold);
    });
}
