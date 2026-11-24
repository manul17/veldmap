use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use tiff::encoder::TiffEncoder;
use tiff::tags::Tag;
use veldmap_core::budget::MemoryBudget;
use veldmap_core::geo::camera::Camera;
use veldmap_core::sync::CancelToken;
use veldmap_core::tile::{RASTER_TILE_SIZE, level_count};
use veldmap_raster::{TwoLaneDecodePool, TwoLaneDecodePoolConfig};
use veldmap_storage::TileDiskCache;

use crate::gpu_tiles::{GpuTileCache, PendingUploads};
use crate::layers::{CursorReadoutValue, OverlayIoEvent, OverlayManager};
use crate::test_support::TempTestRoot;

/// Write a real multi-level tiled GeoTIFF COG (`width x height`, 512x512 tiles,
/// `EPSG:32637`) to `path` filled with `rgb`.
fn write_tiled_geotiff_cog(path: &Path, width: u32, height: u32, rgb: [u8; 3]) {
    let tile = RASTER_TILE_SIZE;
    let mut file = std::fs::File::create(path).expect("create geotiff");
    let mut enc = TiffEncoder::new(&mut file).expect("tiff encoder");

    let write_level = |enc: &mut TiffEncoder<&mut std::fs::File>, w: u32, h: u32, is_base: bool| {
        let mut dir = enc.image_directory().expect("dir");
        dir.write_tag(Tag::ImageWidth, w).expect("tag");
        dir.write_tag(Tag::ImageLength, h).expect("tag");
        dir.write_tag(Tag::BitsPerSample, &[8u16, 8, 8][..])
            .expect("tag");
        dir.write_tag(Tag::Compression, 1u16).expect("tag");
        dir.write_tag(Tag::PhotometricInterpretation, 2u16)
            .expect("tag");
        dir.write_tag(Tag::SamplesPerPixel, 3u16).expect("tag");
        dir.write_tag(Tag::PlanarConfiguration, 1u16).expect("tag");
        dir.write_tag(Tag::TileWidth, tile).expect("tag");
        dir.write_tag(Tag::TileLength, tile).expect("tag");
        if !is_base {
            dir.write_tag(Tag::NewSubfileType, 1u32).expect("tag");
        }
        let cols = w.div_ceil(tile);
        let rows = h.div_ceil(tile);
        let mut tile_bytes = Vec::with_capacity((tile * tile * 3) as usize);
        for _ in 0..(tile * tile) {
            tile_bytes.extend_from_slice(&rgb);
        }
        let mut offsets = Vec::with_capacity((cols * rows) as usize);
        let mut counts = Vec::with_capacity((cols * rows) as usize);
        for _ in 0..(cols * rows) {
            let at = dir.write_data(&tile_bytes[..]).expect("write tile");
            offsets.push(at as u32);
            counts.push(tile_bytes.len() as u32);
        }
        dir.write_tag(Tag::TileOffsets, &offsets[..]).expect("tag");
        dir.write_tag(Tag::TileByteCounts, &counts[..])
            .expect("tag");
        if is_base {
            let geokeys: [u16; 16] = [
                1, 1, 0, 3, // header
                1024, 0, 1, 1, // Projected
                1025, 0, 1, 1, // PixelIsArea
                3072, 0, 1, 32637, // EPSG:32637
            ];
            dir.write_tag(Tag::GeoKeyDirectoryTag, &geokeys[..])
                .expect("tag");
            dir.write_tag(Tag::ModelPixelScaleTag, &[10.0_f64, 10.0, 0.0][..])
                .expect("tag");
            dir.write_tag(
                Tag::ModelTiepointTag,
                &[0.0_f64, 0.0, 0.0, 500_000.0, 5_250_000.0, 0.0][..],
            )
            .expect("tag");
        }
        dir.finish().expect("finish dir");
    };

    let levels = level_count(width, height);
    let mut cur_w = width;
    let mut cur_h = height;
    for idx in 0..levels {
        write_level(&mut enc, cur_w, cur_h, idx == 0);
        cur_w = cur_w.div_ceil(2);
        cur_h = cur_h.div_ceil(2);
    }
}

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

#[test]
fn multi_level_pointwise_cog_populates_all_ladder_rungs_in_wanted_and_supports_async_open_reload_and_cursor()
 {
    let dir = TempTestRoot::new("cog-lifecycle");
    let tif_path = dir.path().join("scene_2048.tif");
    write_tiled_geotiff_cog(&tif_path, 2048, 2048, [42, 128, 210]);

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
        pool.handle().fast().clone(),
        pool.handle().bulk().clone(),
        done_tx,
        wakeup,
        shutdown.clone(),
    );

    // 1. Non-blocking open completes off the UI thread.
    mgr.begin_open_file(&tif_path, true).expect("queue open");
    let (layer_id, bounds) = match wait_for_io_event(&mut mgr) {
        OverlayIoEvent::Opened { id, bounds, fly_to } => {
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
    let total_weight = mgr.stack().total_appetite_weight_bytes().max(1);
    let _ = mgr.build_and_schedule(
        &cam,
        1024,
        1024,
        &mut gpu_cache,
        &pending,
        total_weight,
        &mut carriers,
    );
    let layer = &mgr.stack().layers()[0];
    assert_eq!(layer.meta.max_level(), 2);
    let has_lvl2 = layer.wanted.iter().any(|k| k.level == 2);
    let has_lvl1 = layer.wanted.iter().any(|k| k.level == 1);
    let has_lvl0 = layer.wanted.iter().any(|k| k.level == 0);
    assert!(
        has_lvl2 && has_lvl1 && has_lvl0,
        "wanted must cover all ladder rungs [2, 1, 0], got levels: {:?}",
        layer
            .wanted
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
    let _ = mgr.query_cursor_readout(bounds.lat, bounds.lon);
    let cursor_deadline = Instant::now() + Duration::from_secs(5);
    let mut resolved = None;
    while Instant::now() < cursor_deadline {
        if let Some(readout) = mgr.query_cursor_readout(bounds.lat, bounds.lon)
            && readout.value != CursorReadoutValue::Pending
        {
            resolved = Some(readout);
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    let readout = resolved.expect("cursor readout resolved");
    assert_eq!(readout.value, CursorReadoutValue::Rgba([42, 128, 210, 255]));

    // 4. Reload bypass cache after modifying the TIFF on disk: clears old QOI
    //    cache off-UI-thread, switches to new RGB [200, 50, 10], and preserves visibility.
    write_tiled_geotiff_cog(&tif_path, 2048, 2048, [200, 50, 10]);
    let reload_mut = mgr
        .begin_reload_bypass_cache(layer_id)
        .expect("queue reload");
    assert!(reload_mut.changed);
    assert!(mgr.layer_summaries()[0].reloading);

    match wait_for_io_event(&mut mgr) {
        OverlayIoEvent::Reloaded { id } => assert_eq!(id, layer_id),
        other => panic!("expected Reloaded event, got {other:?}"),
    }
    assert!(!mgr.layer_summaries()[0].reloading);
    assert_eq!(
        disk_cache.stats().expect("stats after reload").tile_count,
        0,
        "reload must purge old QOI cache off the UI thread"
    );

    // Cursor readout after reload sees the updated pixel value [200, 50, 10, 255].
    let _ = mgr.query_cursor_readout(bounds.lat, bounds.lon);
    let cursor_deadline2 = Instant::now() + Duration::from_secs(5);
    let mut resolved2 = None;
    while Instant::now() < cursor_deadline2 {
        if let Some(readout) = mgr.query_cursor_readout(bounds.lat, bounds.lon)
            && readout.value != CursorReadoutValue::Pending
        {
            resolved2 = Some(readout);
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    assert_eq!(
        resolved2.expect("post-reload cursor").value,
        CursorReadoutValue::Rgba([200, 50, 10, 255])
    );

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
