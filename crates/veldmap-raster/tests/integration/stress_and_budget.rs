use std::io::Cursor;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, mpsc};
use std::thread;
use std::time::{Duration, Instant};

use ::tiff::encoder::TiffEncoder;
use ::tiff::tags::Tag;
use veldmap_core::budget::{DecodeBudgetTracker, MemoryBudget};
use veldmap_core::io::{CancelToken, CountingByteSource, MemoryByteSource, SourceFingerprint};
use veldmap_core::tile::{BandKey, PixelFormat, PlanetaryScheme, QuadTileId, SourceId, TileKey};
use veldmap_raster::pyramid::RASTER_TILE_SIZE;
use veldmap_raster::{
    DecodeContext, DecodeJob, DecodeLane, SharedChunkCache, TiffSource, TwoLaneDecodePool,
    TwoLaneDecodePoolConfig,
};

fn build_multi_tile_tiff(tiles_x: u32, tiles_y: u32, chunk_side: u32) -> Vec<u8> {
    let width = tiles_x * chunk_side;
    let height = tiles_y * chunk_side;
    let mut file = Cursor::new(Vec::new());
    {
        let mut enc = TiffEncoder::new(&mut file).expect("enc");
        let mut dir = enc.image_directory().expect("dir");
        dir.write_tag(Tag::ImageWidth, width).expect("tag");
        dir.write_tag(Tag::ImageLength, height).expect("tag");
        dir.write_tag(Tag::BitsPerSample, &[8u16, 8, 8][..])
            .expect("tag");
        dir.write_tag(Tag::Compression, 1u16).expect("tag");
        dir.write_tag(Tag::PhotometricInterpretation, 2u16)
            .expect("tag");
        dir.write_tag(Tag::SamplesPerPixel, 3u16).expect("tag");
        dir.write_tag(Tag::PlanarConfiguration, 1u16).expect("tag");
        dir.write_tag(Tag::TileWidth, chunk_side).expect("tag");
        dir.write_tag(Tag::TileLength, chunk_side).expect("tag");

        let chunk_bytes = (chunk_side * chunk_side * 3) as usize;
        let mut offsets = Vec::new();
        let mut counts = Vec::new();
        for ty in 0..tiles_y {
            for tx in 0..tiles_x {
                let pattern = vec![((tx * 37 + ty * 17) & 0xFF) as u8; chunk_bytes];
                let at = dir.write_data(&pattern[..]).expect("write");
                offsets.push(at as u32);
                counts.push(pattern.len() as u32);
            }
        }
        dir.write_tag(Tag::TileOffsets, &offsets[..]).expect("tag");
        dir.write_tag(Tag::TileByteCounts, &counts[..])
            .expect("tag");
        dir.finish().expect("finish");
    }
    file.into_inner()
}

fn dummy_key(x: u32) -> TileKey {
    TileKey::display_quad(
        SourceId::new(77),
        QuadTileId {
            scheme: PlanetaryScheme::WebMercator,
            level: 4,
            x,
            y: 0,
        },
    )
}

#[test]
fn concurrent_decode_respects_memory_cap_and_deduplicates_shared_chunks() {
    // 2x2 grid of 512x512 chunks (= 1024x1024 raster).
    let raw = build_multi_tile_tiff(2, 2, RASTER_TILE_SIZE);
    let counting = Arc::new(CountingByteSource::new(Arc::new(
        MemoryByteSource::from_vec(raw),
    )));
    let budget = MemoryBudget::detect();
    let cancel = CancelToken::new();
    let src = Arc::new(
        TiffSource::open(
            Arc::clone(&counting) as Arc<dyn veldmap_core::io::ByteSource>,
            SourceFingerprint::from_digest([5u8; 32]),
            &budget,
            &cancel,
        )
        .expect("open tiff"),
    );

    // Cap working memory strictly to 2 chunks at a time while 8 threads contend.
    let single_chunk_est = u64::from(RASTER_TILE_SIZE) * u64::from(RASTER_TILE_SIZE) * 7;
    let cap = single_chunk_est * 2;
    let tracker = Arc::new(DecodeBudgetTracker::new(cap));
    let chunk_cache = Arc::new(SharedChunkCache::new(budget.decoded_chunk_cache_bytes));
    let start_barrier = Arc::new(Barrier::new(8));

    counting.reset();
    let handles: Vec<_> = (0..8u32)
        .map(|i| {
            let src = Arc::clone(&src);
            let cache = Arc::clone(&chunk_cache);
            let tracker = Arc::clone(&tracker);
            let cancel = cancel.clone();
            let barrier = Arc::clone(&start_barrier);
            let (tx, ty) = ((i % 2), ((i / 2) % 2));
            thread::spawn(move || {
                barrier.wait();
                let ctx = DecodeContext {
                    chunk_cache: &cache,
                    budget_tracker: &tracker,
                    budget: &budget,
                    cancel: &cancel,
                };
                src.decode_pointwise_tile(0, tx, ty, &ctx)
            })
        })
        .collect();

    for (i, h) in handles.into_iter().enumerate() {
        let tile = h.join().expect("join worker").expect("concurrent decode");
        assert_eq!(
            (tile.width(), tile.height()),
            (RASTER_TILE_SIZE, RASTER_TILE_SIZE)
        );
        let (tx, ty) = (i as u32 % 2, (i as u32 / 2) % 2);
        let expected = ((tx * 37 + ty * 17) & 0xFF) as u8;
        assert_eq!(&tile.data()[..4], &[expected, expected, expected, 255]);
    }

    assert_eq!(tracker.active_bytes(), 0);
    assert!(
        tracker.peak_bytes() <= cap,
        "peak {} exceeded cap {cap}",
        tracker.peak_bytes()
    );
    let (hits, misses, dedup_waits) = chunk_cache.stats();
    assert_eq!(
        misses, 4,
        "each of 4 unique chunks must decode exactly once"
    );
    assert_eq!(
        hits + dedup_waits,
        4,
        "remaining 4 callers must share or hit cache"
    );
    // 4 singleflight leaders each read 1 chunk (512*512*3 B) + small header/IFD windows,
    // well below an un-deduplicated 8-worker read (~6.29 MB).
    let stats = counting.snapshot();
    assert!(
        stats.bytes_read <= 4 * (512 * 512 * 3) + 4 * 3 * 64 * 1024 + 64 * 1024,
        "unexpectedly high read bytes under contention: {}",
        stats.bytes_read
    );
}

#[test]
fn two_lane_decode_pool_prevents_bulk_jobs_from_blocking_fast_tiles() {
    let raw = build_multi_tile_tiff(2, 2, RASTER_TILE_SIZE);
    let budget = MemoryBudget::detect();
    let cancel = CancelToken::new();
    let src = Arc::new(
        TiffSource::open(
            Arc::new(MemoryByteSource::from_vec(raw)),
            SourceFingerprint::from_digest([7u8; 32]),
            &budget,
            &cancel,
        )
        .expect("open"),
    );
    let chunk_cache = Arc::new(SharedChunkCache::new(budget.decoded_chunk_cache_bytes));
    let tracker = Arc::new(DecodeBudgetTracker::new(budget.decode_working_bytes));

    let shutdown = CancelToken::new();
    let pool = TwoLaneDecodePool::spawn(
        TwoLaneDecodePoolConfig {
            fast_workers: 2,
            bulk_workers: 2,
            ..TwoLaneDecodePoolConfig::default()
        },
        shutdown.clone(),
    );
    let handle = pool.handle();

    let bulk_started = Arc::new(AtomicUsize::new(0));
    let release_bulk = CancelToken::new();

    // Saturate all 2 bulk workers with real `TiffSource::produce_pass` calls paused mid-emit.
    for x in 0..2u32 {
        let src = Arc::clone(&src);
        let cache = Arc::clone(&chunk_cache);
        let tracker = Arc::clone(&tracker);
        let started = Arc::clone(&bulk_started);
        let rel = release_bulk.clone();
        let pass_cancel = shutdown.child_token();
        assert!(handle.submit(
            DecodeLane::Bulk,
            DecodeJob {
                key: dummy_key(x),
                queued_bytes: 1024,
                cancel: pass_cancel.clone(),
                work: Box::new(move || {
                    let ctx = DecodeContext {
                        chunk_cache: &cache,
                        budget_tracker: &tracker,
                        budget: &budget,
                        cancel: &pass_cancel,
                    };
                    let mut signaled = false;
                    let _ = src.produce_pass(&ctx, &mut |_, _, _, _, _, _| {
                        if !signaled {
                            signaled = true;
                            started.fetch_add(1, Ordering::AcqRel);
                            while !rel.is_cancelled() {
                                thread::sleep(Duration::from_millis(2));
                            }
                        }
                        Ok(())
                    });
                }),
            }
        ));
    }

    let wait_start = Instant::now();
    while bulk_started.load(Ordering::Acquire) < 2 && wait_start.elapsed() < Duration::from_secs(2)
    {
        thread::sleep(Duration::from_millis(2));
    }
    assert_eq!(bulk_started.load(Ordering::Acquire), 2);

    // Submit real fast-lane pointwise decodes while both bulk workers remain blocked.
    let (tx, rx) = mpsc::channel();
    for x in 0..4u32 {
        let src = Arc::clone(&src);
        let cache = Arc::clone(&chunk_cache);
        let tracker = Arc::clone(&tracker);
        let fast_cancel = shutdown.child_token();
        let tx = tx.clone();
        assert!(handle.submit(
            DecodeLane::Fast,
            DecodeJob {
                key: TileKey {
                    band: BandKey::Index(0),
                    format: PixelFormat::Rgba8UnormSrgb,
                    ..dummy_key(x)
                },
                queued_bytes: 256,
                cancel: fast_cancel.clone(),
                work: Box::new(move || {
                    let ctx = DecodeContext {
                        chunk_cache: &cache,
                        budget_tracker: &tracker,
                        budget: &budget,
                        cancel: &fast_cancel,
                    };
                    let tile = src
                        .decode_pointwise_tile(0, x % 2, x / 2, &ctx)
                        .expect("fast decode");
                    let _ = tx.send((tile.width(), tile.height()));
                }),
            }
        ));
    }

    let deadline = Instant::now() + Duration::from_secs(2);
    for _ in 0..4 {
        let rem = deadline.saturating_duration_since(Instant::now());
        let dims = rx.recv_timeout(rem).expect("fast lane must not HoL stall");
        assert_eq!(dims, (RASTER_TILE_SIZE, RASTER_TILE_SIZE));
    }

    release_bulk.cancel();
    pool.shutdown();
}

#[test]
fn scheduler_and_tiff_source_coordinate_pointwise_and_pass_execution() {
    use veldmap_core::tile::{ServeMode, TileScheduler};

    let raw = build_multi_tile_tiff(2, 2, RASTER_TILE_SIZE);
    let budget = MemoryBudget::detect();
    let cancel = CancelToken::new();
    let src = Arc::new(
        TiffSource::open(
            Arc::new(MemoryByteSource::from_vec(raw)),
            SourceFingerprint::from_digest([9u8; 32]),
            &budget,
            &cancel,
        )
        .expect("open"),
    );

    let meta = src.pyramid_meta();
    let mut sched = TileScheduler::new();
    let chunk_cache = SharedChunkCache::new(budget.decoded_chunk_cache_bytes);
    let tracker = DecodeBudgetTracker::new(budget.decode_working_bytes);
    let ctx = DecodeContext {
        chunk_cache: &chunk_cache,
        budget_tracker: &tracker,
        budget: &budget,
        cancel: &cancel,
    };

    // Walk `meta.ladder(0)` and drive real `decode_pointwise_tile` / `produce_pass` per ServeMode.
    for lvl in meta.ladder(0) {
        let plan = meta.level(lvl).expect("level in range");
        let k0 = meta.tile_key(lvl, 0, 0);
        let admitted = sched.admit_pointwise_or_check_pass(&meta, lvl, &[k0], |_| false);
        assert_eq!(admitted, vec![k0]);

        match plan.serve {
            ServeMode::Pointwise => {
                let tile = src
                    .decode_pointwise_tile(k0.level, k0.x, k0.y, &ctx)
                    .expect("pointwise decode");
                assert_eq!(
                    (tile.width(), tile.height()),
                    (RASTER_TILE_SIZE, RASTER_TILE_SIZE)
                );
                sched.on_tile_arrived(&k0);
            }
            ServeMode::Pass { from } => {
                let pass_key = meta.pass_key();
                let pass_cancel = CancelToken::new();
                let pass_id = sched
                    .begin_pass(pass_key, lvl, Some(from), pass_cancel, vec![k0])
                    .expect("begin pass");
                src.produce_pass(&ctx, &mut |plvl, px, py, _, _, _| {
                    sched.on_tile_arrived(&meta.tile_key(plvl, px, py));
                    Ok(())
                })
                .expect("produce_pass");
                sched.finish_pass(pass_key, pass_id, true);
            }
        }
        assert!(!sched.is_in_flight(&k0));
    }
}

#[test]
fn concurrent_pass_and_pointwise_under_tight_budget_never_deadlock() {
    let raw = build_multi_tile_tiff(2, 2, RASTER_TILE_SIZE);
    let budget = MemoryBudget::detect();
    let cancel = CancelToken::new();
    let src = Arc::new(
        TiffSource::open(
            Arc::new(MemoryByteSource::from_vec(raw)),
            SourceFingerprint::from_digest([11u8; 32]),
            &budget,
            &cancel,
        )
        .expect("open"),
    );

    // Budget is large enough for the pass alone, NOT pass + chunk simultaneously,
    // exercising sequential non-deadlocking handoff between pass and pointwise.
    let pass_total = src.grid().pass_peak((1024, 1024)).total();
    let tracker = Arc::new(DecodeBudgetTracker::new(pass_total));
    let chunk_cache = Arc::new(SharedChunkCache::new(budget.decoded_chunk_cache_bytes));
    let barrier = Arc::new(Barrier::new(2));

    let pass_thread = {
        let src = Arc::clone(&src);
        let tracker = Arc::clone(&tracker);
        let cache = Arc::clone(&chunk_cache);
        let cancel = cancel.clone();
        let barrier = Arc::clone(&barrier);
        thread::spawn(move || {
            barrier.wait();
            let ctx = DecodeContext {
                chunk_cache: &cache,
                budget_tracker: &tracker,
                budget: &budget,
                cancel: &cancel,
            };
            let mut count = 0usize;
            src.produce_pass(&ctx, &mut |_lvl, _tx, _ty, _w, _h, _rgba| {
                count += 1;
                Ok(())
            })
            .expect("pass completes");
            count
        })
    };

    let direct_thread = {
        let src = Arc::clone(&src);
        let tracker = Arc::clone(&tracker);
        let cache = Arc::clone(&chunk_cache);
        let cancel = cancel.clone();
        let barrier = Arc::clone(&barrier);
        thread::spawn(move || {
            barrier.wait();
            let ctx = DecodeContext {
                chunk_cache: &cache,
                budget_tracker: &tracker,
                budget: &budget,
                cancel: &cancel,
            };
            src.decode_pointwise_tile(0, 1, 1, &ctx)
                .expect("pointwise completes")
        })
    };

    assert_eq!(pass_thread.join().expect("join pass"), 5);
    let tile = direct_thread.join().expect("join pointwise");
    assert_eq!(
        (tile.width(), tile.height()),
        (RASTER_TILE_SIZE, RASTER_TILE_SIZE)
    );
    assert!(tracker.peak_bytes() <= pass_total);
    assert_eq!(tracker.active_bytes(), 0);
}
