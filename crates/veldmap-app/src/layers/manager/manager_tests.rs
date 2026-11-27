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
    CursorReadout, CursorReadoutValue, OpenOverlayRequest, OverlayFrameCtx, OverlayIoEvent,
    OverlayManager, OverlayManagerDeps, RemoteCogRequest,
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

    let mut mgr = OverlayManager::new(OverlayManagerDeps {
        budget,
        disk_cache: Some(Arc::clone(&disk_cache)),
        net: None,
        fast_pool: pool.fast_handle(),
        bulk_pool: pool.bulk_handle(),
        done_tx,
        wakeup,
        shutdown: shutdown.clone(),
    });

    // 1. Non-blocking open completes off the UI thread.
    mgr.begin_open_file(OpenOverlayRequest {
        path: &tif_path,
        fly_to: true,
        visible: true,
        opacity: 1.0,
        crs_override: None,
        style: None,
    })
    .expect("queue open");
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
    assert_eq!(layer.single_asset().expect("single").meta().max_level(), 2);
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

    // Wait for coarsest rung 2 to decode and land in `PendingUploads`, then
    // wait for the offloaded `TileWriteWorker` batch to flush into SQLite.
    let deadline = Instant::now() + Duration::from_secs(5);
    while pending.queued_bytes() == 0 && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(
        pending.queued_bytes() > 0,
        "coarsest rung tile should enqueue"
    );
    while disk_cache.stats().expect("stats").tile_count == 0 && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(
        disk_cache.stats().expect("stats").tile_count > 0,
        "offloaded tile writer should populate QOI disk cache"
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
        OverlayIoEvent::ReloadFailed { .. } => {}
        other => panic!("expected ReloadFailed event, got {other:?}"),
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

#[test]
fn remote_cog_streams_via_http_range_prefetch_and_excludes_local_session() {
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpListener;
    use std::sync::atomic::AtomicBool;
    use veldmap_net::{NetConfig, NetContext};

    let dir = TestTempDir::new("remote-cog-lifecycle");
    let tif_path = dir.path().join("remote_scene.tif");
    write_tiled_geotiff_cog(&tif_path, 512, 512, [90, 160, 230]).expect("write geotiff");
    let payload = Arc::new(std::fs::read(&tif_path).expect("read geotiff"));

    let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock range server");
    let addr = listener.local_addr().expect("local addr");
    listener.set_nonblocking(true).expect("nonblocking");
    let stop = Arc::new(AtomicBool::new(false));
    let stop_srv = Arc::clone(&stop);
    let data_srv = Arc::clone(&payload);

    let srv_handle = std::thread::spawn(move || {
        while !stop_srv.load(Ordering::Relaxed) {
            match listener.accept() {
                Ok((mut stream, _)) => {
                    let _ = stream.set_nonblocking(false);
                    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
                    let mut reader = BufReader::new(&stream);
                    let mut line = String::new();
                    let _ = reader.read_line(&mut line);
                    let mut range_hdr: Option<String> = None;
                    loop {
                        line.clear();
                        if reader.read_line(&mut line).unwrap_or(0) == 0
                            || line == "\r\n"
                            || line == "\n"
                        {
                            break;
                        }
                        if let Some((k, v)) = line.split_once(':')
                            && k.trim().eq_ignore_ascii_case("range")
                        {
                            range_hdr = Some(v.trim().to_owned());
                        }
                    }
                    let total = data_srv.len();
                    let (start, end) = range_hdr
                        .as_deref()
                        .and_then(|r| r.strip_prefix("bytes="))
                        .and_then(|s| s.split_once('-'))
                        .and_then(|(a, b)| {
                            let s: usize = a.parse().ok()?;
                            let e: usize = b.parse().ok()?;
                            Some((s, e.min(total.saturating_sub(1))))
                        })
                        .unwrap_or((0, total.saturating_sub(1)));
                    let slice = &data_srv[start..=end];
                    let header = format!(
                        "HTTP/1.1 206 Partial Content\r\n\
                         Content-Range: bytes {start}-{end}/{total}\r\n\
                         Content-Length: {}\r\n\
                         ETag: \"cog-v1\"\r\n\
                         Connection: close\r\n\r\n",
                        slice.len()
                    );
                    let _ = stream.write_all(header.as_bytes());
                    let _ = stream.write_all(slice);
                }
                Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(5));
                }
                Err(_) => break,
            }
        }
    });

    let disk_cache = Arc::new(
        TileDiskCache::open(dir.path().join("tiles.sqlite"), 32 * 1024 * 1024).expect("disk cache"),
    );
    let budget = MemoryBudget::default();
    let net = Arc::new(NetContext::new(NetConfig::from_budget(&budget)).expect("net ctx"));
    let shutdown = CancelToken::new();
    let pool = TwoLaneDecodePool::spawn(
        TwoLaneDecodePoolConfig::from_budget(&budget),
        shutdown.clone(),
    );
    let mut pending = PendingUploads::with_max_queued_bytes(budget.queued_upload_bytes);
    let done_tx = pending.sender().expect("upload sender");
    let wakeup: veldmap_core::sync::Wakeup = Arc::new(|| {});

    let mut mgr = OverlayManager::new(OverlayManagerDeps {
        budget,
        disk_cache: Some(Arc::clone(&disk_cache)),
        net: Some(Arc::clone(&net)),
        fast_pool: pool.fast_handle(),
        bulk_pool: pool.bulk_handle(),
        done_tx,
        wakeup,
        shutdown: shutdown.clone(),
    });

    let url = format!("http://{addr}/scene.tif?token=secret-query");
    mgr.begin_open_remote_cog(RemoteCogRequest {
        url,
        title: "Test Remote COG".to_owned(),
        extra_headers: Vec::new(),
        external_bands: Vec::new(),
        is_scl: false,
        fly_to: true,
    })
    .expect("queue remote open");

    let (layer_id, bounds) = match wait_for_io_event(&mut mgr) {
        OverlayIoEvent::RemoteOpened {
            id,
            title,
            display_url,
            bounds,
            fly_to,
        } => {
            assert_eq!(title, "Test Remote COG");
            assert!(!display_url.contains("secret-query"));
            assert!(fly_to);
            (id, bounds.expect("valid remote footprint"))
        }
        other => panic!("expected RemoteOpened, got {other:?}"),
    };

    // Remote layers must NEVER leak into local session persistence and must
    // reject local disk reload cleanly.
    assert_eq!(mgr.saved_entries_iter().count(), 0);
    assert!(mgr.layer_summaries()[0].is_remote);
    assert!(mgr.begin_reload_bypass_cache(layer_id).is_err());

    // Cache-only remote cursor probe resolves `Unavailable` before level-0 tiles stream.
    let pre_readout = wait_for_cursor_readout(&mgr, bounds.lat, bounds.lon);
    assert_eq!(pre_readout.value, CursorReadoutValue::Unavailable);

    let mut gpu_cache = GpuTileCache::new(budget.gpu_tile_cache_bytes);
    let cam = Camera::at(bounds.lat, bounds.lon, 5_000.0);
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

    let deadline = Instant::now() + Duration::from_secs(5);
    while pending.queued_bytes() == 0 && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(
        pending.queued_bytes() > 0,
        "remote COG tile should prefetch and decode into PendingUploads"
    );
    // Simulate upload completion report: must invalidate the sticky `Unavailable`
    // cursor entry so the next hover probe reads [90, 160, 230, 255] from chunk cache.
    let stream_key = mgr.stack.layers()[0]
        .single_asset()
        .expect("single")
        .meta()
        .stream_key();
    let upload_report = crate::gpu_tiles::UploadBatchReport {
        uploaded: 1,
        max_tile_bytes_by_stream: vec![(stream_key, 512 * 512 * 4)],
        ..Default::default()
    };
    mgr.record_upload_report(&upload_report);
    let post_readout = wait_for_cursor_readout(&mgr, bounds.lat, bounds.lon);
    assert_eq!(
        post_readout.value,
        CursorReadoutValue::Rgba([90, 160, 230, 255])
    );

    while disk_cache.stats().expect("stats").tile_count == 0 && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(
        disk_cache.stats().expect("stats").tile_count > 0,
        "remote COG decoded tile should persist into QOI disk cache"
    );

    mgr.begin_shutdown();
    shutdown.cancel();
    net.signal_shutdown();
    pending.close();
    pool.shutdown();
    mgr.finish_shutdown();
    net.shutdown();
    stop.store(true, Ordering::Relaxed);
    let _ = srv_handle.join();
}

#[test]
fn tile_write_worker_honors_source_purge_and_backoff_wakes_idle_ui() {
    use veldmap_core::tile::{
        DecodedTile, PixelFormat, SourceId, TileKey, TileScheme, TileStreamSpec,
    };
    use veldmap_storage::DiskTileKey;

    let dir = TestTempDir::new("tile-write-purge");
    let paths = veldmap_storage::StoragePaths::with_root(dir.path());
    let disk_cache = Arc::new(
        TileDiskCache::open(paths.tile_cache_db_path(), 32 * 1024 * 1024).expect("disk cache"),
    );
    let budget = MemoryBudget::default();
    let shutdown = CancelToken::new();
    let wake_count = Arc::new(AtomicUsize::new(0));
    let wake_clone = Arc::clone(&wake_count);
    let wakeup: veldmap_core::sync::Wakeup = Arc::new(move || {
        wake_clone.fetch_add(1, Ordering::Relaxed);
    });
    let mut pending = PendingUploads::with_max_queued_bytes(budget.queued_upload_bytes);
    let done_tx = pending.sender().expect("upload sender");
    let pool = TwoLaneDecodePool::spawn(
        TwoLaneDecodePoolConfig::from_budget(&budget),
        shutdown.clone(),
    );
    let mut mgr = OverlayManager::new(OverlayManagerDeps {
        budget,
        disk_cache: Some(Arc::clone(&disk_cache)),
        net: None,
        fast_pool: pool.fast_handle(),
        bulk_pool: pool.bulk_handle(),
        done_tx,
        wakeup,
        shutdown: shutdown.clone(),
    });

    // `note_submit_backoff` must wake an idle UI once when entering backoff.
    let before = wake_count.load(Ordering::Relaxed);
    mgr.runtime.note_submit_backoff();
    assert_eq!(wake_count.load(Ordering::Relaxed), before + 1);
    mgr.runtime.note_submit_backoff();
    assert_eq!(
        wake_count.load(Ordering::Relaxed),
        before + 1,
        "repeated backoff within active window must not spam wakeups"
    );

    // Enqueue a tile under a source write token, then immediately purge the
    // source before the 50 ms coalescing window elapses: SQLite must stay empty.
    let src = SourceId::new(999);
    let (write_gen, source_cancel) = mgr.runtime.write_token_for_source(src);
    let tk = TileKey::from_stream(
        src,
        TileStreamSpec::DISPLAY,
        TileScheme::RasterPyramid { max_level: 0 },
        0,
        0,
        0,
    );
    let disk_key = DiskTileKey::from_tile_key("deadbeef".to_owned(), &tk);
    let tile = DecodedTile::try_new(4, 4, PixelFormat::Rgba8UnormSrgb, vec![255u8; 4 * 4 * 4])
        .expect("tile");
    mgr.runtime
        .tile_writer
        .as_ref()
        .expect("writer")
        .enqueue_mip0_tile(src, disk_key, tile, write_gen, source_cancel);
    mgr.runtime.purge_source_runtime(src);

    std::thread::sleep(Duration::from_millis(80));
    assert_eq!(
        disk_cache.stats().expect("stats").tile_count,
        0,
        "purged pre-reload tile must not flush into SQLite"
    );

    mgr.begin_shutdown();
    shutdown.cancel();
    pending.close();
    pool.shutdown();
    mgr.finish_shutdown();
}

#[test]
fn band_math_multi_asset_refcounting_scl_r8uint_and_zoom_invariant_cursor_readout() {
    use veldmap_core::tile::{BandKey, PixelFormat};
    use veldmap_raster::test_support::{write_rgbnir_u16_geotiff, write_scl_u8_geotiff};

    use crate::colormap::CategoricalPaletteId;
    use crate::layers::layer::OverlayLayer;
    use crate::layers::{
        BandMathFormula, LayerId, LocalBandMathInput, LocalBandMathSpec, OverlayEdit,
    };

    let dir = TestTempDir::new("band-math-cursor-refcount");

    // 1. Write a 4-band U16 RGB+NIR GeoTIFF (64x64, EPSG:32637, 10 m/px):
    //    GDAL_METADATA sets SCALE=0.0001 across all bands.
    //    Red (band 0) = 0 DN (valid 0.0 reflectance!), Green = 1000, Blue = 500,
    //    NIR (band 3) = 5000 DN (0.5 reflectance) -> exact NDVI = (0.5 - 0.0) / (0.5 + 0.0) = 1.0!
    let rgbnir_path = dir.path().join("rgbnir.tif");
    write_rgbnir_u16_geotiff(&rgbnir_path, 64, 64, [0, 1000, 500, 5000], 0.0001)
        .expect("write rgbnir");

    // 2. Write a co-registered U8 SCL GeoTIFF (64x64, class 9 = Cloud High Prob on left half,
    //    class 4 = Vegetation on right half).
    let scl_path = dir.path().join("scl.tif");
    write_scl_u8_geotiff(&scl_path, 64, 64, 9, 4).expect("write scl");

    let budget = MemoryBudget::default();
    let shutdown = CancelToken::new();
    let pool = TwoLaneDecodePool::spawn(
        TwoLaneDecodePoolConfig::from_budget(&budget),
        shutdown.clone(),
    );
    let mut pending = PendingUploads::with_max_queued_bytes(budget.queued_upload_bytes);
    let done_tx = pending.sender().expect("upload sender");
    let wakeup: veldmap_core::sync::Wakeup = Arc::new(|| {});
    let mut mgr = OverlayManager::new(OverlayManagerDeps {
        budget,
        disk_cache: None,
        net: None,
        fast_pool: pool.fast_handle(),
        bulk_pool: pool.bulk_handle(),
        done_tx,
        wakeup,
        shutdown: shutdown.clone(),
    });

    // Open both local rasters.
    mgr.begin_open_file(OpenOverlayRequest {
        path: &rgbnir_path,
        fly_to: false,
        visible: true,
        opacity: 1.0,
        crs_override: None,
        style: None,
    })
    .expect("open rgbnir");
    let (rgbnir_id, _bounds) = match wait_for_io_event(&mut mgr) {
        OverlayIoEvent::Opened { id, bounds, .. } => (id, bounds.expect("bounds")),
        other => panic!("expected Opened for rgbnir, got {other:?}"),
    };

    mgr.begin_open_file(OpenOverlayRequest {
        path: &scl_path,
        fly_to: false,
        visible: true,
        opacity: 1.0,
        crs_override: None,
        style: None,
    })
    .expect("open scl");
    let (scl_id, _) = match wait_for_io_event(&mut mgr) {
        OverlayIoEvent::Opened { id, bounds, .. } => (id, bounds.expect("bounds")),
        other => panic!("expected Opened for scl, got {other:?}"),
    };

    // Build a multi-asset BandMath NDVI + SCL layer from the loaded layers.
    let (bm_id, _, _) = mgr
        .create_band_math_from_loaded(LocalBandMathSpec {
            title: "Local NDVI + SCL".to_owned(),
            formula: BandMathFormula::NormalizedDiff,
            inputs: vec![
                LocalBandMathInput {
                    layer_id: rgbnir_id,
                    band_index: 3, // NIR = 0.5
                    label: "NIR".to_owned(),
                },
                LocalBandMathInput {
                    layer_id: rgbnir_id,
                    band_index: 0, // Red = 0.0 (valid zero!)
                    label: "Red".to_owned(),
                },
            ],
            scl: Some(LocalBandMathInput {
                layer_id: scl_id,
                band_index: 0,
                label: "SCL".to_owned(),
            }),
        })
        .expect("create local band math");

    // Verify zoom-invariant CPU BandMath cursor readout on both cloud-masked (left)
    // and clear vegetation (right) halves.
    let frame = mgr
        .stack
        .get(bm_id)
        .expect("bm layer")
        .primary_asset()
        .geo()
        .frame()
        .clone();
    let (lat_left, lon_left) = frame.geodetic(0.25, 0.5);
    let readout_left = wait_for_cursor_readout(&mgr, lat_left, lon_left);
    match readout_left.value {
        CursorReadoutValue::BandMath(ref bm) => {
            let val = bm.computed.expect("NDVI must be valid even when Red=0.0");
            assert!((val - 1.0).abs() < 1e-9, "expected NDVI=1.0, got {val}");
            assert_eq!(bm.channels.len(), 2);
            assert!((bm.channels[0].1.value().expect("NIR") - 0.5).abs() < 1e-9);
            assert!(bm.channels[1].1.value().expect("Red").abs() < 1e-9);
            assert!(!bm.has_unavailable);
            assert_eq!(bm.scl_class, Some(9));
            assert_eq!(bm.scl_class_name, Some("Cloud High Prob."));
            assert!(bm.scl_masked, "SCL class 9 must be masked by default");
        }
        other => panic!("expected BandMath readout on left, got {other:?}"),
    }

    let (lat_right, lon_right) = frame.geodetic(0.75, 0.5);
    let readout_right = wait_for_cursor_readout(&mgr, lat_right, lon_right);
    match readout_right.value {
        CursorReadoutValue::BandMath(ref bm) => {
            assert!((bm.computed.expect("NDVI") - 1.0).abs() < 1e-9);
            assert_eq!(bm.scl_class, Some(4));
            assert_eq!(bm.scl_class_name, Some("Vegetation"));
            assert!(!bm.scl_masked, "SCL class 4 must not be masked");
        }
        other => panic!("expected BandMath readout on right, got {other:?}"),
    }

    // BandMath layers never expose a per-layer CRS override even if an input
    // source layer carries one.
    assert!(mgr.layer_crs_override(bm_id).is_none());

    // Disabling SCL mask triggers a topology change so VRAM reservation,
    // pinned root streams, and orphaned SCL jobs stay synchronized.
    let mut_scl_off = mgr.edit_layer(bm_id, OverlayEdit::SetBandMathSclEnabled(false));
    assert!(mut_scl_off.changed && mut_scl_off.topology_changed);

    // Verify multi-source refcounting: removing rgbnir_id while bm_id still references its SourceId
    // MUST NOT purge its SourceId; removing bm_id afterward MUST purge it.
    let rgbnir_src = mgr
        .stack
        .get(rgbnir_id)
        .expect("rgbnir")
        .single_asset()
        .expect("single")
        .source_id();
    let mut_rm_orig = mgr.edit_layer(rgbnir_id, OverlayEdit::Remove);
    assert!(
        !mut_rm_orig.purge_sources.contains(&rgbnir_src),
        "shared SourceId must not be purged while BandMath layer still uses it"
    );
    let mut_rm_bm = mgr.edit_layer(bm_id, OverlayEdit::Remove);
    assert!(
        mut_rm_bm.purge_sources.contains(&rgbnir_src),
        "SourceId must be purged once the final referencing layer is removed"
    );

    // Verify categorical cursor probe cache honors Sentinel2Scl vs DistinctClasses
    // palette switches on the exact same pixel (regression for ProbeKey::Single).
    let scl_stream = veldmap_raster::stream_spec_for_band(BandKey::Index(0), PixelFormat::R8Uint);
    mgr.set_layer_stream(scl_id, scl_stream)
        .expect("switch scl to categorical");
    let _ = mgr.edit_layer(
        scl_id,
        OverlayEdit::SetCategoricalPalette(CategoricalPaletteId::DistinctClasses),
    );
    let r_distinct = wait_for_cursor_readout(&mgr, lat_left, lon_left);
    assert_eq!(
        r_distinct.value,
        CursorReadoutValue::Categorical {
            class_id: 9,
            class_name: None,
        }
    );
    let _ = mgr.edit_layer(
        scl_id,
        OverlayEdit::SetCategoricalPalette(CategoricalPaletteId::Sentinel2Scl),
    );
    let r_scl = wait_for_cursor_readout(&mgr, lat_left, lon_left);
    assert_eq!(
        r_scl.value,
        CursorReadoutValue::Categorical {
            class_id: 9,
            class_name: Some("Cloud High Prob."),
        }
    );

    // Verify SCL asset with `is_scl: true` opens directly on PixelFormat::R8Uint + Sentinel2Scl
    // (instead of defaulting U8 BlackIsZero to Display RGBA8).
    let scl_src = mgr
        .stack
        .get(scl_id)
        .expect("scl")
        .single_asset()
        .expect("asset");
    let scl_layer = OverlayLayer::from_scl_source_for_test(
        LayerId(999),
        scl_path.clone(),
        Arc::clone(scl_src.source()),
        (**scl_src.geo().frame()).clone(),
    );
    assert_eq!(
        scl_layer.single_asset().expect("asset").meta().spec.format,
        PixelFormat::R8Uint,
        "SCL asset must explicitly open on R8Uint"
    );
    assert_eq!(
        scl_layer.single_asset().expect("asset").meta().spec.band,
        BandKey::Index(0)
    );
    assert_eq!(
        scl_layer.style().categorical_palette(),
        CategoricalPaletteId::Sentinel2Scl
    );

    mgr.begin_shutdown();
    shutdown.cancel();
    pending.close();
    pool.shutdown();
    mgr.finish_shutdown();
}
