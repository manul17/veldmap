use std::io::Cursor;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::time::Duration;

use image::{ImageBuffer, ImageFormat, Rgba};
use veldmap_core::tile::{PlanetaryScheme, QuadTileId, SourceId};

use crate::decode_map_tile_with_mipmaps;

use super::*;

fn make_png(color: [u8; 4]) -> Vec<u8> {
    let img: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::from_pixel(4, 4, Rgba(color));
    let mut out = Vec::new();
    img.write_to(&mut Cursor::new(&mut out), ImageFormat::Png)
        .expect("encode png");
    out
}

fn tile_key(level: u8, x: u32, y: u32) -> TileKey {
    TileKey::display_quad(
        SourceId::from_stable_key("decode-test"),
        QuadTileId {
            scheme: PlanetaryScheme::WebMercator,
            level,
            x,
            y,
        },
    )
}

fn default_queue() -> Arc<(Mutex<DecodeQueue>, Condvar)> {
    Arc::new((
        Mutex::new(DecodeQueue::new(
            DEFAULT_MAX_PENDING_DECODE_JOBS,
            DEFAULT_MAX_PENDING_DECODE_BYTES,
        )),
        Condvar::new(),
    ))
}

#[test]
fn queue_cap_evicts_finer_for_coarser_and_enforces_byte_budget() {
    let shared = default_queue();
    let handle = DecodePoolHandle {
        shared: Arc::clone(&shared),
    };
    let tok = CancelToken::new();

    // Fill queue with DEFAULT_MAX_PENDING_DECODE_JOBS at level 5.
    for x in 0..DEFAULT_MAX_PENDING_DECODE_JOBS as u32 {
        assert!(handle.submit(DecodeJob {
            key: tile_key(5, x, 0),
            queued_bytes: 0,
            cancel: tok.clone(),
            work: Box::new(|| {}),
        }));
    }
    // Duplicate key is rejected.
    assert!(!handle.submit(DecodeJob {
        key: tile_key(5, 0, 0),
        queued_bytes: 0,
        cancel: tok.clone(),
        work: Box::new(|| {}),
    }));
    // Finer incoming job (level 6) must be rejected without evicting level 5.
    assert!(!handle.submit(DecodeJob {
        key: tile_key(6, 0, 0),
        queued_bytes: 0,
        cancel: tok.clone(),
        work: Box::new(|| {}),
    }));
    assert_eq!(
        lock_unpoisoned(&shared.0).jobs.len(),
        DEFAULT_MAX_PENDING_DECODE_JOBS
    );
    assert!(
        lock_unpoisoned(&shared.0)
            .jobs
            .iter()
            .all(|j| j.key.level == 5)
    );

    // Coarser incoming job (level 0) evicts one level 5 job and is admitted.
    assert!(handle.submit(DecodeJob {
        key: tile_key(0, 0, 0),
        queued_bytes: 0,
        cancel: tok.clone(),
        work: Box::new(|| {}),
    }));
    {
        let q = lock_unpoisoned(&shared.0);
        assert_eq!(q.jobs.len(), DEFAULT_MAX_PENDING_DECODE_JOBS);
        assert!(q.jobs.iter().any(|j| j.key == tile_key(0, 0, 0)));
    }

    // Equal-coarseness incoming job (level 0 with smaller source) must NOT evict peer level-0 jobs.
    let peer_shared = Arc::new((Mutex::new(DecodeQueue::new(2, 1024)), Condvar::new()));
    let peer_handle = DecodePoolHandle {
        shared: Arc::clone(&peer_shared),
    };
    assert!(peer_handle.submit(DecodeJob {
        key: tile_key(0, 0, 0),
        queued_bytes: 0,
        cancel: tok.clone(),
        work: Box::new(|| {}),
    }));
    assert!(peer_handle.submit(DecodeJob {
        key: tile_key(0, 1, 0),
        queued_bytes: 0,
        cancel: tok.clone(),
        work: Box::new(|| {}),
    }));
    let peer_key = TileKey::display_quad(
        SourceId::new(1),
        QuadTileId {
            scheme: PlanetaryScheme::WebMercator,
            level: 0,
            x: 0,
            y: 0,
        },
    );
    assert!(!peer_handle.submit(DecodeJob {
        key: peer_key,
        queued_bytes: 0,
        cancel: tok.clone(),
        work: Box::new(|| {}),
    }));

    // Encoded byte cap (`DEFAULT_MAX_PENDING_DECODE_BYTES`) prevents unbounded
    // payload accumulation even with fewer than `DEFAULT_MAX_PENDING_DECODE_JOBS` jobs.
    let byte_shared = default_queue();
    let byte_handle = DecodePoolHandle {
        shared: Arc::clone(&byte_shared),
    };
    let chunk = 16 * 1024 * 1024; // 16 MiB each -> 4 jobs fill 64 MiB
    for x in 0..4u32 {
        assert!(byte_handle.submit(DecodeJob {
            key: tile_key(4, x, 0),
            queued_bytes: chunk,
            cancel: tok.clone(),
            work: Box::new(|| {}),
        }));
    }
    assert_eq!(
        lock_unpoisoned(&byte_shared.0).queued_bytes,
        DEFAULT_MAX_PENDING_DECODE_BYTES
    );
    // Finer 5th job exceeding 64 MiB is rejected.
    assert!(!byte_handle.submit(DecodeJob {
        key: tile_key(5, 0, 0),
        queued_bytes: 1024,
        cancel: tok.clone(),
        work: Box::new(|| {}),
    }));
    // Coarser job evicts one 16 MiB finer job to fit within 64 MiB.
    assert!(byte_handle.submit(DecodeJob {
        key: tile_key(1, 0, 0),
        queued_bytes: chunk,
        cancel: tok.clone(),
        work: Box::new(|| {}),
    }));
    assert!(lock_unpoisoned(&byte_shared.0).queued_bytes <= DEFAULT_MAX_PENDING_DECODE_BYTES);

    // Transactional rejection: when evicting ALL strictly-finer jobs still
    // cannot reclaim enough bytes against coarser jobs, the incoming job is
    // rejected AND existing finer jobs remain untouched.
    let tx_shared = default_queue();
    let tx_handle = DecodePoolHandle {
        shared: Arc::clone(&tx_shared),
    };
    let coarse_chunk = 13 * 1024 * 1024;
    for i in 0..4u32 {
        assert!(tx_handle.submit(DecodeJob {
            key: tile_key(0, i, 0),
            queued_bytes: coarse_chunk,
            cancel: tok.clone(),
            work: Box::new(|| {}),
        }));
    }
    let fine_key = tile_key(5, 0, 0);
    assert!(tx_handle.submit(DecodeJob {
        key: fine_key,
        queued_bytes: 4 * 1024 * 1024,
        cancel: tok.clone(),
        work: Box::new(|| {}),
    }));
    // 4 * 13 MiB + 4 MiB = 56 MiB. Incoming 16 MiB z=2 needs 52 + 16 = 68 MiB > 64 MiB
    // even if `fine_key` were evicted -> must reject without dropping `fine_key`.
    assert!(!tx_handle.submit(DecodeJob {
        key: tile_key(2, 0, 0),
        queued_bytes: 16 * 1024 * 1024,
        cancel: tok,
        work: Box::new(|| {}),
    }));
    let tx_q = lock_unpoisoned(&tx_shared.0);
    assert_eq!(tx_q.jobs.len(), 5);
    assert!(
        tx_q.jobs.iter().any(|j| j.key == fine_key),
        "failed byte-cap admission must not evict finer queued jobs"
    );
}

#[test]
fn decodes_valid_tiles_isolates_panics_and_drops_cancelled_outside_lock() {
    let (tx, rx) = mpsc::channel();
    let shutdown = CancelToken::new();
    let pool = DecodePool::spawn(
        DecodePoolConfig {
            worker_count: 1,
            ..DecodePoolConfig::default()
        },
        shutdown.clone(),
    );
    let handle = pool.handle();

    // Exercise real post-lock drop of a previously-queued cancelled job on an
    // unstarted queue: `prune_cancelled` moves the cancelled job into `to_drop`
    // while holding `DecodeQueue`'s mutex, then drops it AFTER releasing the
    // mutex. If Drop ran while holding the lock, re-entering `submit` would
    // deadlock immediately.
    let reentry_shared = default_queue();
    let reentry_handle = DecodePoolHandle {
        shared: Arc::clone(&reentry_shared),
    };
    let queued_cancel = CancelToken::new();
    let reenter_tok = CancelToken::new();
    let reenter_hits = Arc::new(AtomicUsize::new(0));
    struct ReenterOnDrop(DecodePoolHandle, CancelToken, Arc<AtomicUsize>);
    impl Drop for ReenterOnDrop {
        fn drop(&mut self) {
            self.2.fetch_add(1, Ordering::AcqRel);
            let _ = self.0.submit(DecodeJob {
                key: tile_key(9, 0, 0),
                queued_bytes: 0,
                cancel: self.1.clone(),
                work: Box::new(|| {}),
            });
        }
    }
    let guard_probe = ReenterOnDrop(
        reentry_handle.clone(),
        reenter_tok.clone(),
        Arc::clone(&reenter_hits),
    );
    assert!(reentry_handle.submit(DecodeJob {
        key: tile_key(8, 0, 0),
        queued_bytes: 0,
        cancel: queued_cancel.clone(),
        work: Box::new(move || {
            drop(guard_probe);
        }),
    }));
    // Cancel while queued, then trigger `prune_cancelled` via another `submit`.
    queued_cancel.cancel();
    assert!(reentry_handle.submit(DecodeJob {
        key: tile_key(7, 0, 0),
        queued_bytes: 0,
        cancel: reenter_tok,
        work: Box::new(|| {}),
    }));
    assert_eq!(reenter_hits.load(Ordering::Acquire), 1);

    // Panicking job is isolated without killing the worker thread.
    assert!(handle.submit(DecodeJob {
        key: tile_key(0, 0, 0),
        queued_bytes: 16,
        cancel: shutdown.child_token(),
        work: Box::new(|| panic!("simulated codec panic")),
    }));

    // Valid PNG decodes with full mip chain after the isolated panic.
    let png = make_png([10, 200, 30, 255]);
    let tx_ok = tx.clone();
    let key_ok = tile_key(1, 0, 0);
    assert!(handle.submit(DecodeJob {
        key: key_ok,
        queued_bytes: png.len(),
        cancel: shutdown.child_token(),
        work: Box::new(move || {
            let tile = decode_map_tile_with_mipmaps(&png).expect("valid png");
            let _ = tx_ok.send((key_ok, tile));
        }),
    }));

    // Corrupt bytes report Err without crashing the worker.
    let err_hits = Arc::new(AtomicUsize::new(0));
    let err_clone = Arc::clone(&err_hits);
    assert!(handle.submit(DecodeJob {
        key: tile_key(2, 0, 0),
        queued_bytes: 17,
        cancel: shutdown.child_token(),
        work: Box::new(move || {
            assert!(decode_map_tile_with_mipmaps(b"not a valid image").is_err());
            err_clone.fetch_add(1, Ordering::AcqRel);
        }),
    }));

    let (k, decoded) = rx
        .recv_timeout(Duration::from_secs(2))
        .expect("receive decoded tile");
    assert_eq!(k, key_ok);
    assert_eq!(decoded.mip_level_count(), 3); // 4x4 -> 2x2 -> 1x1

    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    while err_hits.load(Ordering::Acquire) == 0 && std::time::Instant::now() < deadline {
        thread::sleep(Duration::from_millis(5));
    }
    assert_eq!(err_hits.load(Ordering::Acquire), 1);

    pool.shutdown();
}
