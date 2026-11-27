use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::Duration;

use super::*;

fn test_key(idx: u32) -> ChunkKey {
    ChunkKey::from_spec(
        SourceId::new(9),
        TileStreamSpec::new(
            veldmap_core::tile::BandKey::Display,
            veldmap_core::tile::PixelFormat::Rgba8UnormSrgb,
            0,
            1,
        ),
        0,
        idx,
    )
}

#[test]
fn chunk_key_ignores_tile_gutter_so_relief_and_scientific_streams_share_chunks() {
    let sci = TileStreamSpec::new(
        veldmap_core::tile::BandKey::Index(0),
        veldmap_core::tile::PixelFormat::R32Float,
        0,
        1,
    );
    let relief = TileStreamSpec::new(
        veldmap_core::tile::BandKey::Index(0),
        veldmap_core::tile::PixelFormat::R32Float,
        1,
        1,
    );
    assert_eq!(
        ChunkKey::from_spec(SourceId::new(7), sci, 0, 3),
        ChunkKey::from_spec(SourceId::new(7), relief, 0, 3)
    );
}

#[test]
fn singleflight_deduplicates_concurrent_callers_and_recovers_from_leader_panic() {
    let cache = Arc::new(SharedChunkCache::new(1024 * 1024));
    let decode_calls = Arc::new(AtomicUsize::new(0));
    let cancel = CancelToken::new();

    let handles: Vec<_> = (0..8)
        .map(|_| {
            let cache = Arc::clone(&cache);
            let calls = Arc::clone(&decode_calls);
            let cancel = cancel.clone();
            thread::spawn(move || {
                cache.get_or_decode(test_key(0), &cancel, || {
                    calls.fetch_add(1, Ordering::AcqRel);
                    thread::sleep(Duration::from_millis(25));
                    Ok(DecodedChunk {
                        data: vec![42u8; 64],
                        width: 4,
                        height: 4,
                        format: veldmap_core::tile::PixelFormat::Rgba8UnormSrgb,
                    })
                })
            })
        })
        .collect();

    for h in handles {
        let chunk = h.join().unwrap().expect("shared chunk");
        assert_eq!(chunk.data.len(), 64);
    }
    assert_eq!(decode_calls.load(Ordering::Acquire), 1);

    // Leader panic must unblock all followers with Err rather than deadlocking,
    // and clean up `in_flight` so a subsequent retry can succeed.
    let panicking_cache = Arc::clone(&cache);
    let cancel_p = CancelToken::new();
    let leader = {
        let cache = Arc::clone(&panicking_cache);
        let c = cancel_p.clone();
        thread::spawn(move || {
            cache.get_or_decode(test_key(1), &c, || -> Result<DecodedChunk, RasterError> {
                thread::sleep(Duration::from_millis(20));
                panic!("simulated decoder panic");
            })
        })
    };
    let follower = {
        let cache = Arc::clone(&panicking_cache);
        let c = cancel_p.clone();
        thread::spawn(move || {
            thread::sleep(Duration::from_millis(5));
            cache.get_or_decode(test_key(1), &c, || {
                unreachable!("follower should not decode")
            })
        })
    };
    assert!(leader.join().unwrap().is_err());
    assert!(follower.join().unwrap().is_err());

    // Subsequent call on key(1) retries cleanly.
    let recovered = cache
        .get_or_decode(test_key(1), &CancelToken::new(), || {
            Ok(DecodedChunk {
                data: vec![7u8; 16],
                width: 2,
                height: 2,
                format: veldmap_core::tile::PixelFormat::Rgba8UnormSrgb,
            })
        })
        .expect("retry after panic succeeds");
    assert_eq!(recovered.data[0], 7);
}

#[test]
fn follower_cancellation_returns_promptly_without_stalling() {
    let cache = Arc::new(SharedChunkCache::new(1024 * 1024));
    let leader_started = Arc::new(AtomicUsize::new(0));
    let leader_done = CancelToken::new();

    let leader = {
        let cache = Arc::clone(&cache);
        let started = Arc::clone(&leader_started);
        let done = leader_done.clone();
        thread::spawn(move || {
            cache.get_or_decode(test_key(2), &CancelToken::new(), || {
                started.store(1, Ordering::Release);
                while !done.is_cancelled() {
                    thread::sleep(Duration::from_millis(5));
                }
                Ok(DecodedChunk {
                    data: vec![9u8; 16],
                    width: 2,
                    height: 2,
                    format: veldmap_core::tile::PixelFormat::Rgba8UnormSrgb,
                })
            })
        })
    };

    while leader_started.load(Ordering::Acquire) == 0 {
        thread::sleep(Duration::from_millis(2));
    }

    let follower_cancel = CancelToken::new();
    let fc_clone = follower_cancel.clone();
    let follower = {
        let cache = Arc::clone(&cache);
        thread::spawn(move || {
            cache.get_or_decode(test_key(2), &fc_clone, || {
                unreachable!("follower must join leader")
            })
        })
    };
    thread::sleep(Duration::from_millis(15));
    follower_cancel.cancel();
    let follower_res = follower.join().unwrap();
    assert!(matches!(follower_res, Err(RasterError::Cancelled)));

    leader_done.cancel();
    assert!(leader.join().unwrap().is_ok());

    // When the leader's own token cancels, an uncancelled follower must retry
    // and become the new leader instead of inheriting a spurious cancellation.
    let leader2_started = Arc::new(AtomicUsize::new(0));
    let leader2_cancel = CancelToken::new();
    let leader2 = {
        let cache = Arc::clone(&cache);
        let started = Arc::clone(&leader2_started);
        let c = leader2_cancel.clone();
        thread::spawn(move || {
            cache.get_or_decode(test_key(3), &c, || {
                started.store(1, Ordering::Release);
                while !c.is_cancelled() {
                    thread::sleep(Duration::from_millis(2));
                }
                Err(RasterError::Cancelled)
            })
        })
    };
    while leader2_started.load(Ordering::Acquire) == 0 {
        thread::sleep(Duration::from_millis(2));
    }
    let follower2_calls = Arc::new(AtomicUsize::new(0));
    let follower2 = {
        let cache = Arc::clone(&cache);
        let calls = Arc::clone(&follower2_calls);
        thread::spawn(move || {
            cache.get_or_decode(test_key(3), &CancelToken::new(), || {
                calls.fetch_add(1, Ordering::AcqRel);
                Ok(DecodedChunk {
                    data: vec![55u8; 16],
                    width: 2,
                    height: 2,
                    format: veldmap_core::tile::PixelFormat::Rgba8UnormSrgb,
                })
            })
        })
    };
    thread::sleep(Duration::from_millis(10));
    leader2_cancel.cancel();
    assert!(matches!(
        leader2.join().unwrap(),
        Err(RasterError::Cancelled)
    ));
    let takeover = follower2
        .join()
        .unwrap()
        .expect("uncancelled follower takes over and succeeds");
    assert_eq!(takeover.data[0], 55);
    assert_eq!(follower2_calls.load(Ordering::Acquire), 1);

    // An uncancelled leader returning an error (even `Cancelled` from an
    // internal source token) must propagate once to followers without livelock.
    let leader3_started = Arc::new(AtomicUsize::new(0));
    let leader3_release = CancelToken::new();
    let leader3 = {
        let cache = Arc::clone(&cache);
        let started = Arc::clone(&leader3_started);
        let release = leader3_release.clone();
        thread::spawn(move || {
            cache.get_or_decode(test_key(4), &CancelToken::new(), || {
                started.store(1, Ordering::Release);
                while !release.is_cancelled() {
                    thread::sleep(Duration::from_millis(2));
                }
                Err(RasterError::Cancelled)
            })
        })
    };
    while leader3_started.load(Ordering::Acquire) == 0 {
        thread::sleep(Duration::from_millis(2));
    }
    let follower3 = {
        let cache = Arc::clone(&cache);
        thread::spawn(move || {
            cache.get_or_decode(test_key(4), &CancelToken::new(), || {
                unreachable!("follower must not retry uncancelled leader error")
            })
        })
    };
    thread::sleep(Duration::from_millis(10));
    leader3_release.cancel();
    assert!(matches!(
        leader3.join().unwrap(),
        Err(RasterError::Cancelled)
    ));
    assert!(matches!(
        follower3.join().unwrap(),
        Err(RasterError::Cancelled)
    ));
}
