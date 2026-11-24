use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, mpsc};
use std::thread;
use std::time::{Duration, Instant};

use veldmap_core::budget::{DecodeBudgetTracker, MemoryBudget};
use veldmap_core::io::ByteSource;
use veldmap_core::io::testing::{CountingByteSource, MemoryByteSource};
use veldmap_core::sync::CancelToken;
use veldmap_core::tile::{
    BandKey, PixelFormat, PlanetaryScheme, QuadTileId, RASTER_TILE_SIZE, SourceFingerprint,
    SourceId, StreamKey, TileKey,
};
use veldmap_raster::{
    DecodeContext, DecodeJob, DecodeJobKey, DecodeLane, DecodePoolConfig, ServeMode,
    SharedChunkCache, TiffSource, TileScheduler, TwoLaneDecodePool, TwoLaneDecodePoolConfig,
};

use crate::common::build_multi_tile_tiff;

fn fast_tile_key(x: u32) -> TileKey {
    TileKey {
        band: BandKey::Index(0),
        format: PixelFormat::Rgba8UnormSrgb,
        ..TileKey::display_quad(
            SourceId::new(77),
            QuadTileId {
                scheme: PlanetaryScheme::WebMercator,
                level: 4,
                x,
                y: 0,
            },
        )
    }
}

#[test]
fn concurrent_decode_respects_memory_cap_and_deduplicates_shared_chunks() {
    // 2x2 grid of 512x512 chunks (= 1024x1024 raster).
    let raw = build_multi_tile_tiff(2, 2, RASTER_TILE_SIZE);
    let counting = Arc::new(CountingByteSource::new(Arc::new(
        MemoryByteSource::from_vec(raw),
    )));
    let budget = MemoryBudget::default();
    let cancel = CancelToken::new();
    let src = Arc::new(
        TiffSource::open(
            Arc::clone(&counting) as Arc<dyn ByteSource>,
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
    let budget = MemoryBudget::default();
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
            fast: DecodePoolConfig {
                worker_count: 2,
                ..DecodePoolConfig::default()
            },
            bulk: DecodePoolConfig {
                worker_count: 2,
                ..DecodePoolConfig::default()
            },
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
        let pass_key = StreamKey {
            source: SourceId::new(100 + u64::from(x)),
            band: BandKey::Display,
            format: PixelFormat::Rgba8UnormSrgb,
            gutter: 0,
        };
        assert!(handle.submit(
            DecodeLane::Bulk,
            DecodeJob {
                key: DecodeJobKey::Pass(pass_key),
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
                key: DecodeJobKey::Tile(fast_tile_key(x)),
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
    let raw = build_multi_tile_tiff(2, 2, RASTER_TILE_SIZE);
    let budget = MemoryBudget::default();
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

        match plan.serve {
            ServeMode::Pointwise => {
                let admitted = sched.admit_pointwise(&meta, lvl, &[k0], |_| false);
                assert_eq!(admitted, vec![k0]);
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
                let pending = sched.check_pass_candidates(&meta, lvl, &[k0], |_| false);
                assert_eq!(pending, vec![k0]);
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
    let budget = MemoryBudget::default();
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
    let pass_total = src.pass_peak_bytes();
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

#[test]
fn untiled_stripped_pass_on_bulk_lane_never_blocks_tiled_cog_pointwise_on_fast_lane() {
    use ::tiff::encoder::TiffEncoder;
    use ::tiff::tags::Tag;
    use std::io::Cursor;
    use veldmap_raster::ServeMode;

    use crate::common::{build_tiled_cog, fake_fingerprint, rgb_pattern};

    // 1. Build a real untiled stripped GeoTIFF (1024 x 1024, 16 rows/strip)
    //    whose coarse level 1 uses `ServeMode::Pass`.
    let mut strip_buf = Cursor::new(Vec::new());
    {
        let mut enc = TiffEncoder::new(&mut strip_buf).expect("strip encoder");
        let mut dir = enc.image_directory().expect("strip dir");
        let (w, h, rps) = (1024u32, 1024u32, 16u32);
        let rgb = rgb_pattern(w, h);
        dir.write_tag(Tag::ImageWidth, w).expect("tag");
        dir.write_tag(Tag::ImageLength, h).expect("tag");
        dir.write_tag(Tag::BitsPerSample, &[8u16, 8, 8][..])
            .expect("tag");
        dir.write_tag(Tag::Compression, 1u16).expect("tag");
        dir.write_tag(Tag::PhotometricInterpretation, 2u16)
            .expect("tag");
        dir.write_tag(Tag::SamplesPerPixel, 3u16).expect("tag");
        dir.write_tag(Tag::PlanarConfiguration, 1u16).expect("tag");
        dir.write_tag(Tag::RowsPerStrip, rps).expect("tag");
        let mut offsets = Vec::new();
        let mut counts = Vec::new();
        for strip in 0..(h / rps) {
            let start = (strip * rps * w * 3) as usize;
            let end = start + (rps * w * 3) as usize;
            let at = dir.write_data(&rgb[start..end]).expect("write strip");
            offsets.push(at as u32);
            counts.push((end - start) as u32);
        }
        dir.write_tag(Tag::StripOffsets, &offsets[..]).expect("tag");
        dir.write_tag(Tag::StripByteCounts, &counts[..])
            .expect("tag");
        dir.finish().expect("finish");
    }
    let strip_bytes = strip_buf.into_inner();

    // 2. Build a real tiled COG (1024 x 1024 with overview) using `ServeMode::Pointwise`.
    let (cog_bytes, _, _) = build_tiled_cog(1024, 1024, 1);

    // Cap pointwise region working set to 2 MiB so the untiled stripped 1024x1024
    // TIFF's coarse level 1 (which spans 4 MiB of base strips) selects
    // `ServeMode::Pass` on the bulk lane, while the tiled COG's 512x512 tiles
    // (`1 MiB`) stay `ServeMode::Pointwise` on the fast lane.
    let budget = MemoryBudget {
        region_cap_bytes: 2 * 1024 * 1024,
        ..MemoryBudget::default()
    };
    let shutdown = CancelToken::new();
    let strip_src = Arc::new(
        TiffSource::open(
            Arc::new(MemoryByteSource::new(Arc::from(strip_bytes.as_slice()))),
            fake_fingerprint(&strip_bytes),
            &budget,
            &shutdown,
        )
        .expect("open stripped tiff"),
    );
    let cog_src = Arc::new(
        TiffSource::open(
            Arc::new(MemoryByteSource::new(Arc::from(cog_bytes.as_slice()))),
            fake_fingerprint(&cog_bytes),
            &budget,
            &shutdown,
        )
        .expect("open tiled cog"),
    );

    let strip_meta = strip_src.pyramid_meta();
    let cog_meta = cog_src.pyramid_meta();
    assert!(!strip_src.is_tiled());
    assert!(cog_src.is_tiled());
    assert!(matches!(
        strip_meta.level(1).expect("strip lvl 1").serve,
        ServeMode::Pass { .. }
    ));
    assert_eq!(
        cog_meta.level(0).expect("cog lvl 0").serve,
        ServeMode::Pointwise
    );

    let chunk_cache = Arc::new(SharedChunkCache::new(budget.decoded_chunk_cache_bytes));
    let tracker = Arc::new(DecodeBudgetTracker::new(budget.decode_working_bytes));
    let pool = TwoLaneDecodePool::spawn(
        TwoLaneDecodePoolConfig {
            fast: DecodePoolConfig {
                worker_count: 1,
                ..DecodePoolConfig::default()
            },
            bulk: DecodePoolConfig {
                worker_count: 1,
                ..DecodePoolConfig::default()
            },
        },
        shutdown.clone(),
    );
    let handle = pool.handle();

    let bulk_paused = Arc::new(AtomicUsize::new(0));
    let release_bulk = CancelToken::new();
    let mut scheduler = TileScheduler::default();

    let pass_key = strip_meta.pass_key();
    let strip_want = vec![strip_meta.tile_key(1, 0, 0)];
    let pass_cancel = shutdown.child_token();
    let pass_id = scheduler
        .begin_pass(pass_key, 1, Some(0), pass_cancel.clone(), strip_want)
        .expect("begin stripped pass");

    {
        let src = Arc::clone(&strip_src);
        let cache = Arc::clone(&chunk_cache);
        let tr = Arc::clone(&tracker);
        let paused = Arc::clone(&bulk_paused);
        let rel = release_bulk.clone();
        assert!(handle.bulk().submit(DecodeJob {
            key: DecodeJobKey::Pass(pass_key),
            queued_bytes: 1024,
            cancel: pass_cancel.clone(),
            work: Box::new(move || {
                let ctx = DecodeContext {
                    chunk_cache: &cache,
                    budget_tracker: &tr,
                    budget: &budget,
                    cancel: &pass_cancel,
                };
                let mut signaled = false;
                let _ = src.produce_pass(&ctx, &mut |_, _, _, _, _, _| {
                    if !signaled {
                        signaled = true;
                        paused.store(1, Ordering::Release);
                        while !rel.is_cancelled() {
                            thread::sleep(Duration::from_millis(2));
                        }
                    }
                    Ok(())
                });
            }),
        }));
    }

    let wait = Instant::now();
    while bulk_paused.load(Ordering::Acquire) == 0 && wait.elapsed() < Duration::from_secs(2) {
        thread::sleep(Duration::from_millis(2));
    }
    assert_eq!(bulk_paused.load(Ordering::Acquire), 1);

    // While the untiled stripped TIFF pass occupies the bulk lane, the tiled
    // COG's pointwise tile on the fast lane must complete immediately.
    let cog_tile_key = cog_meta.tile_key(0, 1, 1);
    let admitted = scheduler.admit_pointwise(&cog_meta, 0, &[cog_tile_key], |_| false);
    assert_eq!(admitted, vec![cog_tile_key]);

    let (tx, rx) = mpsc::channel();
    {
        let src = Arc::clone(&cog_src);
        let cache = Arc::clone(&chunk_cache);
        let tr = Arc::clone(&tracker);
        let fast_cancel = shutdown.child_token();
        assert!(handle.fast().submit(DecodeJob {
            key: DecodeJobKey::Tile(cog_tile_key),
            queued_bytes: 256,
            cancel: fast_cancel.clone(),
            work: Box::new(move || {
                let ctx = DecodeContext {
                    chunk_cache: &cache,
                    budget_tracker: &tr,
                    budget: &budget,
                    cancel: &fast_cancel,
                };
                let tile = src
                    .decode_pointwise_tile(0, 1, 1, &ctx)
                    .expect("cog fast tile");
                let _ = tx.send((tile.width(), tile.height()));
            }),
        }));
    }

    let dims = rx
        .recv_timeout(Duration::from_secs(2))
        .expect("tiled COG fast tile must not HoL stall behind stripped bulk pass");
    assert_eq!(dims, (RASTER_TILE_SIZE, RASTER_TILE_SIZE));
    scheduler.on_tile_arrived(&cog_tile_key);

    release_bulk.cancel();
    pool.shutdown();
    scheduler.finish_pass(pass_key, pass_id, true);
    assert!(tracker.peak_bytes() <= budget.decode_working_bytes);
    assert_eq!(tracker.active_bytes(), 0);
}
