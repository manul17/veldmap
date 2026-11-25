use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use veldmap_core::budget::MemoryBudget;
use veldmap_core::geo::Camera;
use veldmap_core::sync::CancelToken;
use veldmap_raster::test_support::write_tiled_geotiff_cog;
use veldmap_raster::{TwoLaneDecodePool, TwoLaneDecodePoolConfig};
use veldmap_storage::TileDiskCache;
use veldmap_storage::test_support::TestTempDir;

use crate::gpu_tiles::{GpuTileCache, PendingUploads};
use crate::layers::{
    CursorReadout, CursorReadoutValue, OverlayFrameCtx, OverlayIoEvent, OverlayManager,
};

fn wait_for_io_event(mgr: &mut OverlayManager) -> OverlayIoEvent {
    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(5) {
        let (events, _mut) = mgr.poll_io_events();
        if let Some(ev) = events.into_iter().next() {
            return ev;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    panic!("timed out waiting for overlay I/O event");
}

fn wait_for_cursor_readout(mgr: &OverlayManager, lat_deg: f64, lon_deg: f64) -> CursorReadout {
    let _ = mgr.query_cursor_readout(lat_deg, lon_deg);
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        if let Some(readout) = mgr.query_cursor_readout(lat_deg, lon_deg)
            && readout.value != CursorReadoutValue::Pending
        {
            return readout;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    panic!("timed out waiting for cursor readout");
}

#[test]
fn multi_level_pointwise_cog_populates_all_ladder_rungs_in_wanted_and_supports_async_open_reload_and_cursor()
 {
    let dir = TestTempDir::new("cog-lifecycle");
    let tif_path = dir.path().join("scene_2048.tif");
    write_tiled_geotiff_cog(&tif_path, 2048, 2048, [42, 128, 210]).expect("write geotiff");

    let disk_cache = Arc::new(
        TileDiskCache::open(dir.path().join("tiles.sqlite"), 32 * 1024 * 1024).expect("disk cache"),
    );
    let budget = MemoryBudget::default();
    let shutdown = CancelToken::new();
    let pool = TwoLaneDecodePool::spawn(
        TwoLaneDecodePoolConfig::from_budget(&budget),
        shutdown.clone(),
    );
    let mut pending = PendingUploads::with_max_queued_bytes(budget.queued_upload_bytes);
    let done_tx = pending.sender().expect("upload sender");
    let wakeups = Arc::new(AtomicUsize::new(0));
    let wakeup_count = Arc::clone(&wakeups);
    let wakeup: veldmap_core::sync::Wakeup = Arc::new(move || {
        wakeup_count.fetch_add(1, Ordering::Relaxed);
    });

    let mut mgr = OverlayManager::new(
        &budget,
        Some(Arc::clone(&disk_cache)),
        pool.fast_handle(),
        pool.bulk_handle(),
        done_tx,
        wakeup,
        shutdown.clone(),
    );

    // 1. Non-blocking open completes off the UI thread.
    mgr.begin_open_file(&tif_path, true).expect("queue open");
    let (layer_id, bounds) = match wait_for_io_event(&mut mgr) {
        OverlayIoEvent::Opened {
            id, bounds, fly_to, ..
        } => {
            assert!(fly_to);
            (id, bounds.expect("valid footprint"))
        }
        other => panic!("expected Opened event, got {other:?}"),
    };
    assert_eq!(mgr.layer_summaries().len(), 1);

    // 2. Zoomed into target_level=0 on a max_level=2 COG, `layer.wanted` MUST
    //    contain intermediate rung 1 (not just 2 and 0) so `PendingUploads`
    //    never prunes intermediate progressive refinement tiles.
    let mut gpu_cache = GpuTileCache::new(budget.gpu_tile_cache_bytes);
    let cam = Camera::at(bounds.lat, bounds.lon, 3_000.0);
    let mut carriers = std::collections::HashSet::new();
    let total_weight = mgr.total_appetite_weight_bytes().max(1);
    let _ = mgr.build_and_schedule(&mut OverlayFrameCtx {
        camera: &cam,
        width_px: 1024,
        height_px: 1024,
        gpu_cache: &mut gpu_cache,
        pending: &pending,
        total_appetite_weight: total_weight,
        carrier_keys: &mut carriers,
    });
    let layer = &mgr.stack.layers()[0];
    assert_eq!(layer.meta().max_level(), 2);
    let has_lvl2 = layer.wanted().iter().any(|k| k.level == 2);
    let has_lvl1 = layer.wanted().iter().any(|k| k.level == 1);
    let has_lvl0 = layer.wanted().iter().any(|k| k.level == 0);
    assert!(
        has_lvl2 && has_lvl1 && has_lvl0,
        "wanted must cover all ladder rungs [2, 1, 0], got levels: {:?}",
        layer
            .wanted()
            .iter()
            .map(|k| k.level)
            .collect::<std::collections::HashSet<_>>()
    );

    // Wait for coarsest rung 2 to decode and land in `PendingUploads` + QOI disk cache.
    let deadline = Instant::now() + Duration::from_secs(5);
    while pending.queued_bytes() == 0 && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(
        pending.queued_bytes() > 0,
        "coarsest rung tile should enqueue"
    );
    assert!(
        disk_cache.stats().expect("stats").tile_count > 0,
        "decoded tile should populate QOI disk cache"
    );

    // 3. Zoom-invariant CPU cursor readout resolves exact RGB [42, 128, 210, 255].
    let readout = wait_for_cursor_readout(&mgr, bounds.lat, bounds.lon);
    assert_eq!(readout.value, CursorReadoutValue::Rgba([42, 128, 210, 255]));
    assert!(mgr.topmost_visible_source_pixel_size_m().is_some());

    // 4. Reload bypass cache after modifying the TIFF on disk: clears old QOI
    //    cache off-UI-thread, switches to new RGB [200, 50, 10], and preserves visibility.
    write_tiled_geotiff_cog(&tif_path, 2048, 2048, [200, 50, 10]).expect("rewrite geotiff");
    let reload_mut = mgr
        .begin_reload_bypass_cache(layer_id)
        .expect("queue reload");
    assert!(reload_mut.changed);
    assert!(mgr.layer_summaries()[0].reloading);
    assert!(
        mgr.topmost_visible_source_pixel_size_m().is_none(),
        "reloading layer must not leak old source pixel size into HUD"
    );

    match wait_for_io_event(&mut mgr) {
        OverlayIoEvent::Reloaded { id } => assert_eq!(id, layer_id),
        other => panic!("expected Reloaded event, got {other:?}"),
    }
    assert!(!mgr.layer_summaries()[0].reloading);
    assert!(mgr.topmost_visible_source_pixel_size_m().is_some());
    assert_eq!(
        disk_cache.stats().expect("stats after reload").tile_count,
        0,
        "reload must purge old QOI cache off the UI thread"
    );

    // Cursor readout after reload sees the updated pixel value [200, 50, 10, 255].
    let readout2 = wait_for_cursor_readout(&mgr, bounds.lat, bounds.lon);
    assert_eq!(readout2.value, CursorReadoutValue::Rgba([200, 50, 10, 255]));

    // 5. Failed reload (after deleting file) leaves the existing layer intact.
    std::fs::remove_file(&tif_path).expect("remove tif");
    let _ = mgr
        .begin_reload_bypass_cache(layer_id)
        .expect("queue failing reload");
    match wait_for_io_event(&mut mgr) {
        OverlayIoEvent::Failed { .. } => {}
        other => panic!("expected Failed event, got {other:?}"),
    }
    assert_eq!(mgr.layer_summaries().len(), 1);
    assert!(!mgr.layer_summaries()[0].reloading);

    // 6. Clean 3-phase shutdown joins background threads without hanging.
    mgr.begin_shutdown();
    shutdown.cancel();
    pending.close();
    pool.shutdown();
    mgr.finish_shutdown();
}
